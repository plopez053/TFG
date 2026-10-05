"""
Generación: de los fragmentos recuperados a la respuesta que ve el usuario.

  1. contexto: un bloque por proposición, repartiendo el límite de caracteres
  2. prompt de respuesta, con las reglas que pide cada pregunta
  3. limpieza de la respuesta del LLM
  4. fuentes: enlace a la página exacta del PDF bajo cada bloque de la respuesta
  5. respuesta final: la salida del LLM limpia y con sus fuentes (componer_respuesta)
"""
import os
import re
import threading
import urllib.parse
from collections import defaultdict, OrderedDict
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from comun.rutas import BASE_DIR, DATA_PATH
from vectorial.recuperacion import _pide_reciente, _regex_literal, clave_fecha, MESES_ES, strip_accents


# =============================================================================
# 1. Contexto para el LLM
# =============================================================================

# Los fragmentos se agrupan en un bloque por proposición ("[PLENO: fecha] título" + texto +
# resultado) y, si no caben, se reparte el límite de caracteres entre los bloques.

# Límite de caracteres del contexto según quién responde. Con Groq o Gemini como principal se usa el
# de la nube; el local (num_ctx=8192) solo si el principal es Ollama. Antes se usaba el local siempre
# que Ollama estuviera activo, y lo está siempre (calcula los embeddings), así que Groq y Gemini
# recibían 12.000 caracteres aunque admiten mucho más. 20.000 y no más porque Groq (gpt-oss-120b)
# limita los tokens por minuto.
CONTEXT_CHAR_LIMIT_LOCAL = 12000
CONTEXT_CHAR_LIMIT_GROQ = 20000
# Tamaño mínimo de cada bloque del contexto cuando hay que repartirlo (ver _format_context)
CAP_BLOQUE_MIN = 600
# Caracteres que se garantizan de cada fragmento destacado al recortar un bloque (ver _render_bloque)
VENTANA_DESTACADO = 600
DEBUG_CONTEXT_PATH = os.path.join(BASE_DIR, "debug_context.txt")


_DEBUG_CONTEXT_LOCK = threading.Lock()


# métodos de RAGPipeline (vectorial/pipeline.py) para construir el contexto
class Contexto:

    # límite de caracteres del contexto según el LLM que responde
    def _context_char_limit(self) -> int:
        nube = not str(getattr(self._llm_provider, "primary_name", "ollama")).lower().startswith("ollama")
        return CONTEXT_CHAR_LIMIT_GROQ if nube else CONTEXT_CHAR_LIMIT_LOCAL

    # fragmentos agrupados por (fecha, tema, acta), en el orden en que aparecen (= prioridad)
    @staticmethod
    def _agrupar_bloques(docs: List[Document]) -> "OrderedDict":
        groups = OrderedDict()
        for d in docs:
            date = d.metadata.get("date", "Fecha desconocida")
            topic = d.metadata.get("topic", "General")
            source = os.path.basename(d.metadata.get("source", "Acta"))
            key = (date, topic, source)

            content = d.page_content.strip()
            content = content.replace("Udalbatzako Idazkaritza Nagusia", "")
            content = content.replace("Udalbatzarreko Idazkaritza Nagusia", "")
            content = content.replace("Secretaría General del Pleno", "")

            # quita las etiquetas ASUNTO:/ORADOR:; en las actas modernas el chunk
            # va en una sola línea y no se puede quitar la primera línea sin más
            lines = content.split('\n')
            if len(lines) >= 3:
                if lines[0].startswith('ASUNTO:'):
                    lines = lines[1:]
                if lines and lines[0].startswith('ORADOR:'):
                    lines = lines[1:]
                content = '\n'.join(lines).strip()
            else:
                content = re.sub(r'^ASUNTO:\s*', '', content)
                content = re.sub(r'\bORADOR:\s*[^(]*\([^)]*\)\s*', ' ', content)
                content = re.sub(r'\s{2,}', ' ', content).strip()

            if key not in groups:
                groups[key] = {"contents": [], "vote_result": None, "lead": False, "destacados": []}
            groups[key]["contents"].append(content)
            if d.metadata.get("_lead"):
                groups[key]["lead"] = True
                groups[key]["destacados"].append((re.sub(r'\s+', ' ', content), d.metadata.get("_kw")))
            if not groups[key]["vote_result"] and d.metadata.get("vote_result"):
                groups[key]["vote_result"] = d.metadata["vote_result"]
        return groups

    # trozo de un fragmento destacado que se garantiza en el bloque: alrededor de la palabra
    # encontrada si viene del canal literal; si no, su comienzo
    @staticmethod
    def _ventana(texto: str, kw: Optional[str]) -> str:
        if kw:
            m = re.search(_regex_literal(kw), texto)
            if m:
                ini = max(0, m.start() - 200)
                return texto[ini:ini + VENTANA_DESTACADO]
        return texto[:VENTANA_DESTACADO]

    # ventanas de los fragmentos destacados (mejores del reranker y canal literal) que no
    # han entrado en el texto ya recortado del bloque
    @classmethod
    def _destacados_fuera(cls, group_data: dict, visible: str) -> List[str]:
        fuera = []
        for texto, kw in group_data.get("destacados", []):
            if kw:
                m = re.search(_regex_literal(kw), texto)
                dentro = m is None or re.search(_regex_literal(kw), visible) is not None
            else:
                dentro = texto[:200] in visible
            ventana = cls._ventana(texto, kw)
            if not dentro and ventana not in fuera:
                fuera.append(ventana)
        return fuera

    # un bloque "[PLENO: fecha] tema". Sin cap: la cabecera de la propuesta + el primer tramo del
    # debate, saltando el texto dispositivo repetido hasta la primera intervención (SR./SRA.).
    # Con cap: solo los primeros cap caracteres (para que quepan más bloques en el contexto).
    # En los dos casos, si un fragmento destacado se queda fuera del recorte (p. ej. la única
    # frase que nombra a Tubacex, al final de un debate largo), se pone delante una ventana de él.
    @classmethod
    def _render_bloque(cls, key: tuple, group_data: dict, cap: Optional[int] = None) -> str:
        date, topic, _source = key
        raw = "\n---\n".join(c for c in group_data["contents"] if c.strip())
        raw_flat = re.sub(r'\s+', ' ', raw)
        if cap is not None:
            raw = raw_flat[:cap].rstrip() + ("..." if len(raw_flat) > cap else "")
            fuera = cls._destacados_fuera(group_data, raw)
            if fuera:
                extra = " [...] ".join(fuera)[:cap]
                resto = max(0, cap - len(extra) - 7)
                raw = "..." + extra + ("... " + raw_flat[:resto].rstrip() + "..." if resto >= 100 else "...")
        else:
            head = raw_flat[:1600]
            m_int = re.search(r'SR[A]?\.\s+[A-ZÁÉÍÓÚÑ]{2,}', raw_flat)
            if m_int and m_int.start() > 1600:
                raw = head + " [...debate...] " + raw_flat[m_int.start():m_int.start() + 2200]
            elif len(raw) > 3800:
                raw = raw[:3800] + "..."
            fuera = cls._destacados_fuera(group_data, re.sub(r'\s+', ' ', raw))
            if fuera:
                raw = "..." + " [...] ".join(fuera) + "... [...] " + raw
        block = f"[PLENO: {date}] {topic[:120]}\n{raw}\n"
        if group_data["vote_result"]:
            block += f"RESULTADO: {group_data['vote_result']}\n"
        return block + "\n"

    # un bloque por cada (fecha, tema). Con max_chars, el contexto se reparte entre los bloques en
    # vez de cortarse por el final: antes, al pasarse del límite, se perdían los bloques más
    # recientes (van ordenados por fecha) y la respuesta se basaba en los de 2007-2012. Se recorta
    # cada bloque a un tamaño común (los cortos no gastan lo que no usan); si ni así caben, se
    # descartan los de menor prioridad.
    def _format_context(self, docs: List[Document], max_chars: Optional[int] = None) -> str:
        groups = self._agrupar_bloques(docs)
        bloques = list(groups.items())

        # los destacados (mejores del reranker y del canal literal) primero; el resto, cronológico
        def mostrar(sel):
            return ([kg for kg in sel if kg[1]["lead"]]
                    + sorted((kg for kg in sel if not kg[1]["lead"]), key=lambda kg: clave_fecha(kg[0][0])))

        if max_chars is None or sum(len(self._render_bloque(k, g)) for k, g in bloques) <= max_chars:
            return "".join(self._render_bloque(k, g) for k, g in mostrar(bloques)).strip()

        def tam(cap: int, n: int) -> int:
            return sum(len(self._render_bloque(k, g, cap)) for k, g in bloques[:n])

        n = len(bloques)
        while n > 1 and tam(CAP_BLOQUE_MIN, n) > max_chars:
            n -= 1                       # descarta el de menor prioridad
        lo, hi = CAP_BLOQUE_MIN, 3800
        while lo < hi:                   # el mayor cap común que cabe
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if tam(mid, n) <= max_chars else (lo, mid - 1)
        return "".join(self._render_bloque(k, g, lo) for k, g in mostrar(bloques[:n])).strip()

    # guarda el último contexto enviado al LLM, para depurar
    def _dump_debug_context(self, formatted_context: str) -> None:
        with _DEBUG_CONTEXT_LOCK:
            with open(DEBUG_CONTEXT_PATH, "w", encoding="utf-8") as f:
                f.write(formatted_context)

    # contexto formateado y truncado + fechas en orden cronológico
    def _build_context(self, docs: List[Document], question: str) -> dict:
        max_ctx = self._context_char_limit()
        formatted_context = self._format_context(docs, max_chars=max_ctx)
        if len(formatted_context) > max_ctx:
            formatted_context = formatted_context[:max_ctx] + "\n\n[...CONTEXTO TRUNCADO POR TAMAÑO...]"
        self._dump_debug_context(formatted_context)

        unique_dates = sorted(
            {d.metadata["date"] for d in docs if d.metadata.get("date")},
            key=clave_fecha,
        )
        return {
            "context": formatted_context,
            "is_multi_session": len(unique_dates) > 1,
            "unique_dates": unique_dates,
            "docs": docs,
            "question": question,
        }


# =============================================================================
# 2. Prompts de respuesta
# =============================================================================

# prompts de respuesta, compartidos por la CLI y Chainlit (build_answer_prompt)
_PROMPT_MULTI_SESSION = (
    "Eres el Cronista Oficial de Bilbao. RESPONDE SIEMPRE EN ESPAÑOL, nunca en inglés.\n\n"
    "Se te dan fragmentos de varios plenos del Ayuntamiento de Bilbao, cada uno marcado con "
    "[PLENO: fecha] título. Fechas en este contexto: {dates_found}\n\n"
    "Para CADA pleno relevante para la pregunta, escribe un bloque con este formato EXACTO:\n"
    "**[fecha DD-MM-YYYY] — [grupo proponente]**\n"
    "- Título: [el título de la propuesta, tal como aparece]\n"
    "- Propuesta: [qué se pide exactamente: puntos, cifras, medidas]\n"
    "- Resumen: [de qué se discutió, con tus propias palabras; omite esta línea si el acta no dice nada]\n"
    "- Votos: [resultado con cifras si aparecen, ej. \"Aprobada. A favor: 29, en contra: 0\"; "
    "si no hay resultado, escribe [Sin resultado en acta]]\n\n"
    "EJEMPLO — de este fragmento de contexto:\n"
    "[PLENO: 26-10-2010] 5. PROPUESTA de aprobación de una subvención nominativa\n"
    "GRUPO MUNICIPAL X ... SR. GARCIA ... solicita destinar 50.000 euros a...\n"
    "RESULTADO: Aprobada. Votos a favor: 29, en contra: 0.\n"
    "el bloque correcto es:\n"
    "**[26-10-2010] — GRUPO MUNICIPAL X**\n"
    "- Título: Aprobación de una subvención nominativa\n"
    "- Propuesta: Destinar 50.000 euros a...\n"
    "- Resumen: El Sr. García defendió la propuesta explicando...\n"
    "- Votos: Aprobada. A favor: 29, en contra: 0.\n\n"
    "REGLAS:\n"
    "- Usa SOLO información explícita en las actas. NUNCA inventes fechas, cifras, nombres o resultados.\n"
    "- Coherencia: si los votos en contra son 0 o no aparecen, el resultado no puede ser \"rechazada\". "
    "No mezcles el resultado de una enmienda con el de la votación principal.\n"
    "- Una enmienda o propuesta citada dentro de un debate NO es un acuerdo aprobado: solo está "
    "aprobado lo que diga la línea RESULTADO de ese bloque, y referido a lo que esa línea nombra.\n"
    "- Cuando el contexto trae VARIAS proposiciones seguidas y parecidas (misma sesión, "
    "estructura similar), presta especial atención a NO mezclar el título, la propuesta, "
    "el resumen o las cifras de votos de UNA proposición con los de OTRA — cada bloque "
    "debe venir ÍNTEGRAMENTE del mismo fragmento del contexto, nunca combinado.\n"
    "- Ordena los bloques de más antiguo a más reciente.\n"
    "- Un bloque por cada [fecha]-[proposición] distinta. NUNCA repitas dos veces "
    "el mismo bloque (misma fecha y mismo contenido) aunque el texto de las actas "
    "esté repetido en el contexto.\n"
    "- El [grupo proponente] de CADA bloque es el que aparece junto a [PLENO: fecha] "
    "EN ESE FRAGMENTO CONCRETO — nunca copies el grupo de otro bloque anterior. Si "
    "ese fragmento no menciona ningún grupo (p.ej. es una resolución de la Alcaldía "
    "o una dación de cuenta administrativa), escribe \"Ayuntamiento de Bilbao\" en "
    "vez de un grupo, o directamente omite ese guion.\n"
    "- Si un pleno del contexto NO tiene relación real con la pregunta, OMÍTELO "
    "POR COMPLETO — nunca escribas un bloque diciendo que no hay información o que "
    "no se encuentra en el contexto; simplemente no lo incluyas.\n"
    "- El contexto es una MUESTRA recuperada por búsqueda, NUNCA todas las actas del "
    "Ayuntamiento. Si la pregunta pide un TOTAL, un MÁXIMO o \"quién/qué grupo/concejal "
    "más\" sobre el conjunto completo (p.ej. \"cuántas en total\", \"qué grupo ha "
    "presentado más\", \"el concejal que más...\"), dejar CLARO en la conclusión que la "
    "cifra es sobre los fragmentos recuperados en esta búsqueda, NUNCA presentarla como "
    "si fuera el recuento completo o definitivo de todas las actas — este sistema no "
    "puede hacer un recuento exhaustivo sobre el corpus completo, solo puede narrar "
    "sobre lo que ha recuperado.\n"
    "- Si varios fragmentos del contexto mencionan la MISMA entidad, empresa o persona "
    "(p.ej. varias menciones sueltas de un mismo nombre en plenos distintos), cuenta y "
    "menciona TODAS las fechas/fragmentos distintos que aparecen en el contexto antes de "
    "concluir cuántas veces aparece — no te quedes solo con el primero que encuentres ni "
    "digas \"solo una vez\" si el contexto trae más de una fecha con esa mención.\n"
    "- Si la pregunta da por hecho algo concreto (una moción, un lugar, una calle, una fecha, un "
    "resultado, la postura de un grupo) y ningún fragmento lo recoge EXPLÍCITAMENTE, empieza "
    "diciendo que no consta en los fragmentos recuperados. NUNCA presentes otro fragmento "
    "parecido (otra calle, otro año, un debate general sobre el tema, otra votación) como si "
    "fuera lo que se pregunta; si lo citas, di claramente que es otra cosa.\n"
    "- Termina con un párrafo de CONCLUSIÓN que sintetice la evolución del tema.\n\n"
    "ACTAS:\n{context}\n\n"
    "PREGUNTA: {question}\n"
    "RESPUESTA:"
)


_PROMPT_SINGLE_SESSION = (
    "Eres el Cronista Oficial de Bilbao. RESPONDE SIEMPRE EN ESPAÑOL.\n\n"
    "Basándote en el ACTA de abajo, responde a: {question}\n\n"
    "Usa este formato:\n"
    "En la sesión del Pleno de Bilbao del [fecha EXACTA en formato DD-MM-YYYY, "
    "tal cual aparece en el ACTA — NUNCA la escribas en palabras "
    "(\"26 de enero de 2023\"), siempre como cifras con guiones]...\n"
    "- Título: [título de la propuesta, tal como aparece]\n"
    "- Propuesta: [qué se pide exactamente]\n"
    "- Resumen: [de qué se discutió, con tus propias palabras]\n"
    "- Votos: [resultado con cifras si aparecen; si no hay, escribe [Sin resultado en acta]]\n\n"
    "Si el acta trata VARIAS propuestas distintas, repite el bloque completo "
    "(Título/Propuesta/Resumen/Votos) una vez por propuesta — NUNCA repitas dos "
    "veces el mismo bloque para la misma propuesta.\n\n"
    "REGLAS:\n"
    "- Usa SOLO información explícita del acta. NUNCA inventes datos.\n"
    "- Si la pregunta da por hecho algo concreto (una moción, un lugar, una fecha, un resultado) "
    "y el acta no lo recoge EXPLÍCITAMENTE, empieza diciendo que no consta en el acta; no "
    "presentes otra propuesta parecida como si fuera la que se pregunta.\n"
    "- Una enmienda o propuesta citada dentro de un debate NO es un acuerdo aprobado: solo está "
    "aprobado lo que diga la línea RESULTADO de ese bloque, y referido a lo que esa línea nombra.\n"
    "- Cuando el acta trae VARIAS propuestas seguidas y parecidas (mismo formato, "
    "mismo tipo de acuerdo), presta especial atención a NO mezclar el título, la "
    "propuesta, el resumen o las cifras de votos de UNA con los de OTRA — cada "
    "bloque debe venir ÍNTEGRAMENTE del mismo punto del orden del día, nunca "
    "combinado con otro.\n\n"
    "EJEMPLO DEL ERROR MÁS FRECUENTE A EVITAR — con este fragmento de contexto:\n"
    "[PLENO: 26-01-2023] 15. PROPUESTA de aprobación de subvenciones a personas "
    "mayores por 860.817€...\nRESULTADO: Votos emitidos: 29 | a favor: 29\n"
    "[PLENO: 26-01-2023] 14. PROPUESTA de aprobación de subvenciones al Comedor "
    "de San Antonio por 619.128€...\nRESULTADO: Votos emitidos: 29 | a favor: 19, "
    "en contra: 7, abstenciones: 3\n\n"
    "INCORRECTO (un solo bloque que mezcla las dos propuestas y usa los votos de "
    "la segunda para describir la primera):\n"
    "- Título: Aprobación de subvenciones nominativas\n"
    "- Propuesta: A favor de personas mayores y del Comedor de San Antonio\n"
    "- Votos: Votos emitidos: 29 | a favor: 19, en contra: 7, abstenciones: 3\n\n"
    "CORRECTO (dos bloques, cada uno con SUS PROPIOS votos, sin mezclarlos):\n"
    "- Título: Aprobación de subvenciones a personas mayores\n"
    "- Votos: Votos emitidos: 29 | a favor: 29\n"
    "(bloque aparte)\n"
    "- Título: Aprobación de subvenciones al Comedor de San Antonio\n"
    "- Votos: Votos emitidos: 29 | a favor: 19, en contra: 7, abstenciones: 3\n\n"
    "ACTA:\n{context}\n\n"
    "PREGUNTA: {question}\n"
    "RESPUESTA:"
)


# preguntas de recuento o ranking: el RAG vectorial solo ve una muestra de fragmentos
_RECUENTO_RE = re.compile(
    r"^\W*(cu[aá]nt[oa]s|qu[eé] porcentaje|en qu[eé] a[ñn]o se .* m[aá]s|qu[eé] grupo .* m[aá]s)"
    r"|\bporcentaje\b|\branking\b", re.IGNORECASE)


AVISO_RECUENTO = ("*Nota: para recuentos y rankings exactos usa el perfil GraphRAG (SPARQL); "
                  "aquí las cifras se calculan solo sobre los fragmentos recuperados.*\n\n")


def aviso_perfil(question: str) -> str:
    return AVISO_RECUENTO if _RECUENTO_RE.search(question.strip()) else ""


# "ahora", "actualmente": el corpus llega a marzo de 2026, así que lo "actual"
# son los plenos de 2025-2026
_PIDE_ACTUAL_RE = re.compile(r"\bahora\b|\bactualmente\b|\ben la actualidad\b|\bhoy en d[ií]a\b|\bhoy\b",
                             re.IGNORECASE)


ANIO_ACTUAL_MIN = 2025


# Reglas extra del prompt según lo que pide la pregunta sobre el tiempo. Con
# "la última vez" el modelo recibía fragmentos de 2025-2026 pero concluía con
# uno de 2016 (ordenaba de antiguo a nuevo y descartaba las menciones de pasada);
# con "ahora" describía la situación actual sin ningún fragmento reciente.
def _reglas_temporales(question: str, fechas: List[str]) -> str:
    reglas = []
    if fechas and _pide_reciente(question):
        mas_nueva = max(fechas, key=clave_fecha)
        reglas.append(
            f"- La pregunta pide lo MÁS RECIENTE. El fragmento más nuevo del contexto es del {mas_nueva}. "
            "Revisa los fragmentos empezando por los más nuevos: el más reciente que trate el asunto, "
            "aunque sea de pasada, es la respuesta. En la conclusión da esa fecha como la última de "
            "los fragmentos recuperados.")
    if _PIDE_ACTUAL_RE.search(question):
        recientes = [f for f in fechas if f[-4:].isdigit() and int(f[-4:]) >= ANIO_ACTUAL_MIN]
        if not recientes:
            reglas.append(
                f"- La pregunta pregunta por la situación ACTUAL, pero el contexto no trae ningún pleno de "
                f"{ANIO_ACTUAL_MIN} o posterior. NO describas lo que se debate ahora: di claramente que los "
                "fragmentos recuperados no incluyen plenos recientes.")
    return "\n".join(reglas)


# mensajes del prompt de respuesta a partir del contexto de retrieve_context()
def construir_prompt(ctx: dict) -> List[Any]:
    if ctx["is_multi_session"]:
        # unique_dates ya viene en orden cronológico (sorted() sobre DD-MM-YYYY no vale)
        text = _PROMPT_MULTI_SESSION.format(
            dates_found=', '.join(ctx["unique_dates"]),
            context=ctx["context"],
            question=ctx["question"],
        )
        reglas = _reglas_temporales(ctx["question"], ctx["unique_dates"])
        if reglas:
            text = text.replace("\n\nACTAS:\n", "\n" + reglas + "\n\nACTAS:\n", 1)
    else:
        text = _PROMPT_SINGLE_SESSION.format(context=ctx["context"], question=ctx["question"])
    return [HumanMessage(content=text)]


# =============================================================================
# 3. Limpieza de la respuesta del LLM
# =============================================================================

# elimina párrafos repetidos (los modelos pequeños a veces copian el mismo
# bloque varias veces). Se aplica antes de insertar las fuentes.
def dedup_answer_blocks(text: str) -> str:
    paragraphs = re.split(r'\n\s*\n', text)
    seen: set = set()
    out = []
    for p in paragraphs:
        norm = re.sub(r'\s+', ' ', p).strip().lower()
        if norm and norm in seen:
            continue
        if norm:
            seen.add(norm)
        out.append(p)
    return '\n\n'.join(out)


_SIN_INFO_RE = re.compile(
    r'no se (encuentra|proporciona|dispone|menciona)|no hay informaci[oó]n'
    r'|sin informaci[oó]n relevante|no tiene relaci[oó]n',
    re.IGNORECASE,
)


_TITULO_CAMPO_RE = re.compile(r'-\s*\*{0,2}\s*T[ií]tulo\s*\*{0,2}\s*:\s*(.+)', re.IGNORECASE)


_PROPUESTA_CAMPO_RE = re.compile(r'-\s*\*{0,2}\s*Propuesta\s*\*{0,2}\s*:\s*(.+)', re.IGNORECASE)


_HEADER_LINE_RE = re.compile(r'^\s*\*{0,2}\s*\[.*?\]\s*[—-]\s*(.+?)\*{0,2}\s*$')


# elimina los bloques en los que el LLM dice que no hay información, aunque el
# prompt pide omitirlos: cabecera sola con "no hay información", o Título y
# Propuesta vacíos a la vez (con uno solo el bloque puede tener datos reales)
def strip_empty_blocks(text: str) -> str:
    paragraphs = re.split(r'\n\s*\n', text)
    out = []
    for p in paragraphs:
        lines = [l for l in p.splitlines() if l.strip()]
        if len(lines) == 1:
            m_header = _HEADER_LINE_RE.match(lines[0])
            if m_header and _SIN_INFO_RE.search(m_header.group(1)):
                continue
        m_titulo = _TITULO_CAMPO_RE.search(p)
        m_propuesta = _PROPUESTA_CAMPO_RE.search(p)
        if (m_titulo and m_propuesta
                and _SIN_INFO_RE.search(m_titulo.group(1))
                and _SIN_INFO_RE.search(m_propuesta.group(1))):
            continue
        out.append(p)
    return '\n\n'.join(out)


# =============================================================================
# 4. Fuentes: enlaces a la página del PDF
# =============================================================================

# Los enlaces no los escribe el LLM: se calculan con los metadatos (acta y página) de los
# fragmentos recuperados y se colocan bajo el bloque de la respuesta al que corresponden.
#
#   build_sources_data: una fuente por debate recuperado
#   insertar_fuentes: empareja cada bloque de la respuesta con su fuente (por título o por
#                     fecha), corrige la línea de Votos e inserta el enlace

# --- una fuente por debate recuperado ---

# ruta guardada en la BD vectorial (Windows o Unix) -> ruta local equivalente
def resolve_pdf_path(source: str) -> str:
    normalized = source.replace("\\", "/")
    parts = normalized.split("/")
    for i, part in enumerate(parts):
        if re.match(r"^\d{4}$", part) and i + 1 < len(parts):
            year, filename = part, parts[i + 1]
            local_path = os.path.join(DATA_PATH, year, filename)
            if os.path.exists(local_path):
                return local_path
    return source


# vocabulario procedimental común a casi todos los debates: no cuenta como
# palabra del asunto al comparar una fuente con la respuesta
_STOP_PROCEDIMENTAL = {
    "enmienda", "enmiendas", "modificacion", "adicion", "votos", "favor",
    "contra", "abstenciones", "emitidos", "decae", "decaen", "acepta",
    "aceptada", "rechaza", "rechazada", "queda", "aprobada", "proposicion",
    "asunto", "orador", "sesion", "punto", "secretario", "alcalde", "señor",
    "senor", "senora", "señora", "votacion", "vota", "presentada", "formulada",
    "tenor", "literal", "siguiente", "udalbatzak", "udalbatzarreko",
    "idazkaritza", "nagusia", "secretaria",
    "para", "sobre", "como", "este", "esta", "esto", "unas", "unos", "mas",
    "sino", "donde", "cuando", "entre", "desde", "hasta", "tambien", "todo",
    "toda", "todos", "todas", "cada", "otro", "otra", "otros", "otras",
    "puede", "deben", "debe", "ante", "bien", "muy",
    "bildu", "elkarrekin", "podemos", "ezker", "anitza", "equo", "berdeak",
    "partido", "popular", "socialista", "socialistas", "vascos",
}

_STOP_PALABRAS_CLAVE = {
    "grupo", "municipal", "politico", "proposicion", "proposamena", "presenta",
    "cuya", "parte", "dispositiva", "plantea", "adopcion", "acuerdo", "plenario",
    "propuesta", "pleno", "ayuntamiento", "bilbao", "equipo", "gobierno",
    "resultado", "argumentos", "fuente", "instar", "insta",
    "votacion", "sesion",
}


# palabras significativas de un texto (>=4 letras, sin acentos ni ruido estructural)
def _palabras_clave(texto: str) -> set:
    texto = strip_accents(texto.lower())
    return {w for w in re.findall(r"[a-z]{4,}", texto) if w not in _STOP_PALABRAS_CLAVE}


# metadatos de las fuentes: una entrada por debate citado
def build_sources_data(retrieved_docs, answer_text=None):
    por_grupo = OrderedDict()
    for doc in retrieved_docs:
        pdf_path = resolve_pdf_path(doc.metadata.get("source", ""))
        if not pdf_path or not os.path.exists(pdf_path):
            continue
        date = doc.metadata.get("date", "Fecha desconocida")
        topic = doc.metadata.get("topic", "")
        # la portada y el índice no son un debate citable
        if topic in ("", "General", "General / Introducción"):
            continue
        key = (date, topic, pdf_path)
        entry = por_grupo.setdefault(key, {"pdf_path": pdf_path, "date": date, "docs": [], "topics": []})
        entry["docs"].append(doc)
        if topic and topic not in entry["topics"]:
            entry["topics"].append(topic)

    sources_data = []
    for (date, _, pdf_path), info in por_grupo.items():
        docs_d = info["docs"]
        paginas = [d.metadata.get("page") for d in docs_d if d.metadata.get("page")]
        p_min = min(paginas) if paginas else None
        p_max = max(paginas) if paginas else None
        rango = (f"Pág. {p_min}" if p_min == p_max else f"Págs. {p_min}-{p_max}") if p_min else None

        year = os.path.basename(os.path.dirname(pdf_path))
        fname = os.path.basename(pdf_path)
        anchor = f"#page={p_min}" if p_min else ""
        url = f"/acta/{year}/{urllib.parse.quote(fname)}{anchor}"

        pdf_name = f"Ver PDF - Acta {date}"
        if rango:
            pdf_name += f" ({rango})"

        topic = info["topics"][0] if info["topics"] else "Tema general"
        vote_result = next(
            (d.metadata.get("vote_result") for d in docs_d if d.metadata.get("vote_result")), None
        )
        contenido = " ".join(d.page_content for d in docs_d)
        content_kw = _palabras_clave(contenido) - _STOP_PROCEDIMENTAL

        sources_data.append({
            "date": date, "topic": topic, "url": url,
            "pdf_name": pdf_name, "vote_result": vote_result, "content_kw": content_kw,
        })

    # en un pleno concreto se recuperan muchos debates del día: solo se citan
    # los que comparten al menos 4 palabras de asunto con la respuesta
    fechas_distintas = {s["date"] for s in sources_data}
    if answer_text and len(fechas_distintas) == 1:
        ans_kw = _palabras_clave(answer_text) - _STOP_PROCEDIMENTAL
        relevantes = [s for s in sources_data if len(s["content_kw"] & ans_kw) >= 4]
        if relevantes:
            sources_data = relevantes

    return sources_data


# --- colocar cada fuente bajo su bloque de la respuesta ---

_TITULO_LINE_RE = re.compile(
    r'^\s*-?\s*\*{0,2}\s*T[ií]tulo\s*\*{0,2}\s*:\s*\*{0,2}\s*(.+?)\*{0,2}\s*$',
    re.MULTILINE | re.IGNORECASE,
)
_ITEM_NUM_RE = re.compile(r'^\s*\*{0,2}\s*(\d+)\s*[.):]')
_VOTOS_LINEA_RE = re.compile(r'(-\s*\*{0,2}\s*Votos\s*\*{0,2}\s*:\s*)(.+)', re.IGNORECASE)
# admite el marcado del LLM delante ("### Conclusión", "**Conclusión**")
_CONCLUSION_RE = re.compile(r'\n[ \t]*[#*_ \t]*(?:CONCLUSI[ÓO]N|En conclusi|En resumen)', re.I)


# posiciones de una fecha en el texto, como DD-MM-YYYY o en prosa ("26 de enero de 2023")
def find_date_mentions(date_ddmmyyyy: str, text: str) -> List[int]:
    try:
        day, month, year = date_ddmmyyyy.split('-')
        day_i, month_i = int(day), int(month)
    except (ValueError, AttributeError):
        return [m.start() for m in re.finditer(re.escape(date_ddmmyyyy), text)]
    positions = [m.start() for m in re.finditer(re.escape(date_ddmmyyyy), text)]
    mes_nombre = next((k for k, v in MESES_ES.items() if v == month_i), None)
    if mes_nombre:
        patron_prosa = rf'\b{day_i}\s+de\s+{mes_nombre}\s+de\s+{year}\b'
        positions += [m.start() for m in re.finditer(patron_prosa, text, re.IGNORECASE)]
    return sorted(positions)


# tramos (inicio, fin, título) de la respuesta, uno por línea "Título:"
def find_item_anchors(text: str, end_pos: int) -> List[Tuple[int, int, str]]:
    matches = list(_TITULO_LINE_RE.finditer(text, 0, end_pos))
    anchors = []
    for i, m in enumerate(matches):
        fin = matches[i + 1].start() if i + 1 < len(matches) else end_pos
        anchors.append((m.start(), fin, m.group(1).strip()))
    return anchors


# número de punto del orden del día al inicio de un título ("18. PROPOSICIÓN...")
def item_number(text: str) -> Optional[str]:
    m = _ITEM_NUM_RE.match(text.strip())
    return m.group(1) if m else None


# fuente que corresponde a un título escrito por el LLM: primero por número de
# punto del orden del día; si no, por palabras compartidas (al menos un tercio)
def match_source_by_title(titulo: str, candidatos: List[dict]) -> Optional[dict]:
    if not candidatos:
        return None
    num = item_number(titulo)
    if num:
        exactos = [s for s in candidatos if item_number(s.get("topic", "")) == num]
        if len(exactos) == 1:
            return exactos[0]
    tit_words = _palabras_clave(titulo)
    if not tit_words:
        return None
    mejor = max(candidatos, key=lambda s: len(tit_words & _palabras_clave(s["topic"])))
    overlap = len(tit_words & _palabras_clave(mejor["topic"]))
    if overlap >= 1 and overlap / len(tit_words) >= 0.34:
        return mejor
    return None


# sustituye la línea "- Votos:" escrita por el LLM por el vote_result de los
# metadatos: con muchas propuestas parecidas el modelo mezcla las cifras
def replace_votos_line(block_text: str, vote_result: str) -> str:
    new_text, n = _VOTOS_LINEA_RE.subn(lambda m: m.group(1) + vote_result, block_text, count=1)
    return new_text if n else block_text


# sesión única con varias propuestas: todas comparten fecha, así que cada
# bloque se empareja por su "Título:". Sin coincidencia fiable no se asigna.
def _asignar_por_titulo(item_anchors, sources_data) -> list:
    asignaciones, usadas = [], set()
    for start, fin, titulo in item_anchors:
        candidatos = [s for s in sources_data if id(s) not in usadas]
        if not candidatos:
            break
        mejor = match_source_by_title(titulo, candidatos)
        if mejor is None:
            continue
        asignaciones.append((start, fin, mejor))
        usadas.add(id(mejor))
    return asignaciones


def _es_cabecera(texto: str, pos: int) -> bool:
    inicio = texto.rfind("\n", 0, pos) + 1
    return re.fullmatch(r"[\s*#_\[(]*", texto[inicio:pos]) is not None


# varios plenos: cada mención de una fecha abre un bloque; entre las fuentes de
# esa fecha se elige la que más palabras comparte con la cabecera del bloque
def _asignar_por_fecha(answer_text: str, concl_pos: int, sources_data) -> list:
    bloques = []
    for date in {s["date"] for s in sources_data}:
        for pos in find_date_mentions(date, answer_text[:concl_pos]):
            bloques.append([pos, date])
    # Solo abren bloque las fechas de cabecera (al principio de una línea, tras
    # "**[" o similar). Una fecha en prosa dentro de un párrafo partía la frase
    # y el enlace a la fuente quedaba insertado en medio.
    cabeceras = [b for b in bloques if _es_cabecera(answer_text, b[0])]
    if cabeceras:
        bloques = cabeceras
    bloques.sort()

    src_por_fecha = defaultdict(list)
    for s in sources_data:
        src_por_fecha[s["date"]].append(s)

    asignaciones, usadas = [], set()
    for idx, (start, date) in enumerate(bloques):
        candidatos = [s for s in src_por_fecha[date] if id(s) not in usadas]
        if not candidatos:
            continue
        fin = bloques[idx + 1][0] if idx + 1 < len(bloques) else concl_pos
        cab_words = _palabras_clave(answer_text[start:start + 120])
        mejor = max(candidatos, key=lambda s: len(cab_words & _palabras_clave(s["topic"])))
        if len(cab_words & _palabras_clave(mejor["topic"])) == 0:
            mejor = candidatos[0]
        asignaciones.append((start, fin, mejor))
        usadas.add(id(mejor))
    return asignaciones


# corrige la línea de Votos de cada bloque e inserta su enlace al final. Se
# recorre de derecha a izquierda para que editar un bloque no desplace las
# posiciones de los bloques que quedan por editar.
def _insertar_en_bloques(answer_text: str, asignaciones: list) -> str:
    for start, fin, s in sorted(asignaciones, key=lambda x: x[0], reverse=True):
        content = answer_text[start:fin]
        vote_gt = s.get("vote_result")
        if vote_gt:
            content = replace_votos_line(content, vote_gt)
        f = len(content)
        while f > 0 and content[f - 1] in "*#\n\r \t[":
            f -= 1
        linea = f"\n\n📄 *Fuente:* [{s['pdf_name']}]({s['url']})\n"
        if vote_gt:
            linea += f"*Resultado:* {vote_gt}\n"
        content = content[:f] + linea + content[f:]
        answer_text = answer_text[:start] + content + answer_text[fin:]
    return answer_text


# lista de fuentes al final: las que no se pudieron colocar bajo ningún bloque
def _lista_fuentes(titulo: str, fuentes: list) -> str:
    texto = f"\n\n**{titulo}:**\n"
    for s in fuentes:
        linea = f"* [{s['pdf_name']}]({s['url']})"
        if s.get("vote_result"):
            linea += f" — *{s['vote_result']}*"
        texto += linea + "\n"
    return texto


# añade a la respuesta del RAG vectorial los enlaces a las fuentes
def insertar_fuentes(answer_text: str, retrieved_docs: list, is_multi_session: bool) -> str:
    sources_data = build_sources_data(retrieved_docs, answer_text)

    if sources_data:
        concl = _CONCLUSION_RE.search(answer_text)
        concl_pos = concl.start() if concl else len(answer_text)
        item_anchors = find_item_anchors(answer_text, concl_pos)
        if not is_multi_session and len(item_anchors) >= 2:
            asignaciones = _asignar_por_titulo(item_anchors, sources_data)
        else:
            asignaciones = _asignar_por_fecha(answer_text, concl_pos, sources_data)
        answer_text = _insertar_en_bloques(answer_text, asignaciones)

        usadas = {id(s) for _, _, s in asignaciones}
        no_ubicadas = [s for s in sources_data if id(s) not in usadas]
        if no_ubicadas and not is_multi_session:
            for s in no_ubicadas:
                answer_text += f"\n\n📄 *Fuente:* [{s['pdf_name']}]({s['url']})\n"
                if s.get("vote_result"):
                    answer_text += f"*Resultado:* {s['vote_result']}\n"
        elif no_ubicadas:
            answer_text += _lista_fuentes("Otras fuentes", no_ubicadas)

    # si no se ha colocado ninguna, se listan todas al final
    if "📄" not in answer_text and "Otras fuentes" not in answer_text:
        fallback = sources_data or build_sources_data(retrieved_docs)
        if fallback:
            answer_text += _lista_fuentes("Fuentes", fallback)
    return answer_text


# =============================================================================
# 5. Respuesta final
# =============================================================================

# salida del LLM sin repeticiones ni bloques vacíos
def limpiar_respuesta(texto_llm: str) -> str:
    # el LLM a veces repite "PREGUNTA:/RESPUESTA:" del prompt al final
    texto = re.sub(r'\n+PREGUNTA\s*:.*', '', texto_llm, flags=re.DOTALL | re.IGNORECASE)
    return strip_empty_blocks(dedup_answer_blocks(texto))


# Respuesta que ve el usuario, en la interfaz y en la consola: la salida del LLM limpia, con el
# enlace a su acta bajo cada bloque y, si la pregunta es de recuento, el aviso de perfil. ctx es
# el contexto que devuelve RAGPipeline.retrieve_context.
def componer_respuesta(texto_llm: str, ctx: dict) -> str:
    # se limpia antes de insertar las fuentes, que dependen de las posiciones del texto
    texto = limpiar_respuesta(texto_llm)
    if ctx["docs"]:
        texto = insertar_fuentes(texto, ctx["docs"], ctx["is_multi_session"])
    return aviso_perfil(ctx["question"]) + texto
