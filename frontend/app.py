import sys
import os
import asyncio
import re
import glob
import urllib.parse
from collections import OrderedDict, defaultdict

# Fix Python 3.14 + sniffio incompatibility: current_task() returns None in some
# ASGI contexts even though a loop is running, causing anyio.NoEventLoopError.
import sniffio as _sniffio
from sniffio import AsyncLibraryNotFoundError as _AsyncLibraryNotFoundError
_orig_detect = _sniffio.current_async_library
def _patched_detect():
    try:
        return _orig_detect()
    except _AsyncLibraryNotFoundError:
        try:
            asyncio.get_running_loop()
            return "asyncio"
        except RuntimeError:
            raise _AsyncLibraryNotFoundError("unknown async library, or not in async context")
_sniffio.current_async_library = _patched_detect

# Agregar la raíz del proyecto al PYTHONPATH para poder importar backend.rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chainlit as cl
from chainlit.server import app as _fastapi_app
from fastapi import HTTPException
from fastapi.responses import FileResponse
from langchain_core.prompts import ChatPromptTemplate
from backend.rag import get_rag, DATA_PATH
from graphrag.graphrag.graph_rag_sparql import graph_answer as _graph_answer, _load_graph as _load_rdf_graph

# ---------------------------------------------------------------------------
# Ruta propia para servir los PDF de las actas directamente desde actas/.
# Permite enlazarlos con un hipervínculo normal (abre en pestaña del navegador y
# salta a la página con #page=N), evitando el panel lateral de Chainlit que se
# abría solo. Se sanea el nombre para impedir path traversal (../).
# ---------------------------------------------------------------------------
@_fastapi_app.get("/acta/{year}/{filename}")
async def servir_acta(year: str, filename: str):
    filename = os.path.basename(filename)
    if not re.match(r"^\d{4}$", year):
        raise HTTPException(status_code=400, detail="Año no válido")
    path = os.path.join(DATA_PATH, year, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Acta no encontrada")
    return FileResponse(path, media_type="application/pdf")


# Chainlit registra un catch-all que sirve la SPA para CUALQUIER ruta. Como se
# registró antes que la nuestra, interceptaba /acta/... y devolvía la app en vez
# del PDF. Movemos nuestra ruta al principio para que tenga prioridad.
_ruta_acta = _fastapi_app.router.routes.pop()
_fastapi_app.router.routes.insert(0, _ruta_acta)


# ---------------------------------------------------------------------------
# Pre-carga de ambos motores al importar el módulo
# ---------------------------------------------------------------------------
print("[*] Pre-cargando el motor RAG vectorial...")
_rag = get_rag()
print("[+] Motor RAG vectorial listo.")

print("[*] Pre-cargando el grafo RDF (GraphRAG)...")
_load_rdf_graph()
print("[+] Grafo RDF listo.")


def resolve_pdf_path(source: str) -> str:
    """Convierte rutas Windows de la BD al path local equivalente."""
    normalized = source.replace("\\", "/")
    parts = normalized.split("/")
    # Buscar el año (carpeta numérica de 4 dígitos) y el nombre del archivo
    for i, part in enumerate(parts):
        if re.match(r"^\d{4}$", part) and i + 1 < len(parts):
            year, filename = part, parts[i + 1]
            local_path = os.path.join(DATA_PATH, year, filename)
            if os.path.exists(local_path):
                return local_path
    # Si ya es una ruta local válida, devolverla tal cual
    return source


def _palabras_clave(texto: str) -> set:
    """Palabras significativas de un texto (≥4 letras, sin acentos ni palabras
    estructurales). Sirve para emparejar el bloque de la respuesta con su fuente
    comparando el nombre del grupo político, no solo la fecha."""
    texto = texto.lower()
    for a, b in [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ñ", "n")]:
        texto = texto.replace(a, b)
    stop = {
        "grupo", "municipal", "politico", "proposicion", "proposamena", "presenta",
        "cuya", "parte", "dispositiva", "plantea", "adopcion", "acuerdo", "plenario",
        "propuesta", "pleno", "ayuntamiento", "bilbao", "equipo", "gobierno",
        "resultado", "argumentos", "fuente", "instar", "insta",
    }
    return {w for w in re.findall(r"[a-z]{4,}", texto) if w not in stop}


# Vocabulario procedimental común a casi todos los debates/votaciones: se excluye
# al medir relevancia para que solo cuenten las palabras de ASUNTO (vivienda, IBI...).
_STOP_PROCEDIMENTAL = {
    # Trámite y votación (común a todos los debates)
    "enmienda", "enmiendas", "modificacion", "adicion", "votos", "favor",
    "contra", "abstenciones", "emitidos", "decae", "decaen", "acepta",
    "aceptada", "rechaza", "rechazada", "queda", "aprobada", "proposicion",
    "asunto", "orador", "sesion", "punto", "secretario", "alcalde", "señor",
    "senor", "senora", "señora", "votacion", "vota", "presentada", "formulada",
    "tenor", "literal", "siguiente", "udalbatzak", "udalbatzarreko",
    "idazkaritza", "nagusia", "secretaria",
    # Relleno (no aportan al asunto)
    "para", "sobre", "como", "este", "esta", "esto", "unas", "unos", "mas",
    "sino", "donde", "cuando", "entre", "desde", "hasta", "tambien", "todo",
    "toda", "todos", "todas", "cada", "otro", "otra", "otros", "otras",
    "puede", "deben", "debe", "ante", "bien", "muy",
    # Nombres de grupos/partidos (aparecen en todas sus proposiciones, sea cual sea el tema)
    "bildu", "elkarrekin", "podemos", "ezker", "anitza", "equo", "berdeak",
    "partido", "popular", "socialista", "socialistas", "vascos",
}


def build_sources_data(retrieved_docs, answer_text=None):
    """Construye los datos de las fuentes (una por acta/fecha citada).

    Usa el nº de página del metadato `page` (lo añade el indexador), así que es
    instantáneo. Para cada acta calcula el RANGO de páginas del debate (de la
    primera a la última página de sus fragmentos) para que el usuario localice
    rápido la propuesta en el PDF. Devuelve dicts serializables (sin objetos de
    Chainlit) para construir los elementos en el hilo async.
    """
    # Agrupar por (fecha, topic): una entrada por DEBATE, no por acta. Un mismo
    # pleno puede tener muchos debates (12 en el acta de 30-09-2021); agrupar por
    # acta fusionaba todos en una sola fuente con un rango de páginas absurdo
    # (1-284) y el voto del primer debate, no el preguntado. Por debate, el rango
    # y el vote_result salen ajustados al tema correcto.
    por_grupo = OrderedDict()
    for doc in retrieved_docs:
        pdf_path = resolve_pdf_path(doc.metadata.get("source", ""))
        if not pdf_path or not os.path.exists(pdf_path):
            continue
        date = doc.metadata.get("date", "Fecha desconocida")
        topic = doc.metadata.get("topic", "")
        # La sección de portada/índice del acta no es un debate citable y abarca
        # decenas de páginas: la excluimos como fuente.
        if topic in ("", "General", "General / Introducción"):
            continue
        key = (date, topic, pdf_path)
        entry = por_grupo.setdefault(key, {"pdf_path": pdf_path, "date": date, "docs": [], "topics": []})
        entry["docs"].append(doc)
        if topic and topic not in entry["topics"]:
            entry["topics"].append(topic)

    sources_data = []
    for (date, _topic_key, pdf_path), info in por_grupo.items():
        docs_d = info["docs"]
        paginas = [d.metadata.get("page") for d in docs_d if d.metadata.get("page")]
        p_min = min(paginas) if paginas else None
        p_max = max(paginas) if paginas else None
        if p_min is not None:
            rango = f"Pág. {p_min}" if p_min == p_max else f"Págs. {p_min}-{p_max}"
        else:
            rango = None

        year = os.path.basename(os.path.dirname(pdf_path))
        fname = os.path.basename(pdf_path)
        anchor = f"#page={p_min}" if p_min else ""
        url = f"/acta/{year}/{urllib.parse.quote(fname)}{anchor}"

        pdf_name = f"Ver PDF - Acta {date}"
        if rango:
            pdf_name += f" ({rango})"

        topic = info["topics"][0] if info["topics"] else "Tema general"
        short_topic = topic[:80] + "..." if len(topic) > 80 else topic
        vote_result = next(
            (d.metadata.get("vote_result") for d in docs_d if d.metadata.get("vote_result")), None
        )
        # Palabras de ASUNTO del debate (de su contenido), para medir relevancia
        # frente a la respuesta. Sin esto, una pregunta sobre una fecha concreta
        # con muchos debates listaría TODOS como fuente (también los no tratados).
        contenido = " ".join(d.page_content for d in docs_d)
        content_kw = _palabras_clave(contenido) - _STOP_PROCEDIMENTAL

        sources_data.append({
            "date": date,
            "topic": topic,
            "pdf_path": pdf_path,
            "url": url,
            "page": p_min or 1,
            "short_topic": short_topic,
            "pdf_name": pdf_name,
            "vote_result": vote_result,
            "content_kw": content_kw,
        })

    # Filtro de relevancia SOLO cuando todos los debates son de la misma fecha
    # (pregunta de un pleno concreto): ahí el buscador trae muchos debates del día
    # y hay que quedarse con los que la respuesta trata (≥4 palabras de asunto
    # compartidas, ya sin trámite, relleno ni nombres de grupo). En multisesión cada
    # fecha es un debate distinto y el emparejamiento por fecha ya lo resuelve, así
    # que no se filtra para no descartar plenos legítimamente citados.
    fechas_distintas = {s["date"] for s in sources_data}
    if answer_text and len(fechas_distintas) == 1:
        ans_kw = _palabras_clave(answer_text) - _STOP_PROCEDIMENTAL
        relevantes = [s for s in sources_data if len(s["content_kw"] & ans_kw) >= 4]
        if relevantes:  # nunca dejar la respuesta sin ninguna fuente
            sources_data = relevantes

    return sources_data


# ---------------------------------------------------------------------------
# Perfiles de chat: RAG Vectorial vs GraphRAG
# ---------------------------------------------------------------------------
@cl.set_chat_profiles
async def set_chat_profiles():
    return [
        cl.ChatProfile(
            name="RAG Vectorial",
            markdown_description=(
                "Busca en el texto de las actas.\n\n"
                "Mejor para preguntas abiertas: qué se debatió, propuestas y argumentos."
            ),
        ),
        cl.ChatProfile(
            name="GraphRAG (SPARQL)",
            markdown_description=(
                "Consulta el grafo de proposiciones.\n\n"
                "Mejor para números: cuántas proposiciones, rankings por grupo, tema o año."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers del modo GraphRAG
# ---------------------------------------------------------------------------
def _find_pdf_by_date(fecha: str) -> str:
    """Dado 'DD-MM-YYYY' devuelve la ruta al PDF del acta (si existe)."""
    parts = fecha.split("-")
    if len(parts) != 3:
        return ""
    year = parts[-1]
    patron = os.path.join(DATA_PATH, year, f"{fecha}_*.pdf")
    matches = glob.glob(patron)
    return matches[0] if matches else ""


def _fuentes_graphrag(rows: list) -> str:
    """Extrae fechas únicas de las filas SPARQL y genera links a los PDFs.

    Busca en cada fila las claves candidatas a fecha (DD-MM-YYYY).
    Para preguntas de agregación (solo ?anio, ?n) no hay fechas → devuelve "".
    """
    DATE_KEYS = ("fecha", "fechaProp", "fechaPleno", "date")
    TITLE_KEYS = ("titulo", "tituloProp", "tituloTopic", "label")

    vistas: set = set()
    links: list[str] = []

    for row in rows[:50]:
        fecha = next((row[k] for k in DATE_KEYS if k in row and re.match(r"\d{2}-\d{2}-\d{4}", row[k])), None)
        if not fecha or fecha in vistas:
            continue
        vistas.add(fecha)

        pdf = _find_pdf_by_date(fecha)
        if not pdf:
            continue

        year_dir = os.path.basename(os.path.dirname(pdf))
        fname = os.path.basename(pdf)
        url = f"/acta/{year_dir}/{urllib.parse.quote(fname)}"

        titulo = next((row[k] for k in TITLE_KEYS if k in row and row[k]), "")
        label = f"Ver PDF — Acta {fecha}"
        if titulo:
            short = titulo[:70] + "..." if len(titulo) > 70 else titulo
            label += f" | {short}"

        links.append(f"- [{label}]({url})")

    if not links:
        return ""
    return "\n\n---\n**Actas del grafo consultadas:**\n" + "\n".join(links)


# ---------------------------------------------------------------------------
# Handler del modo GraphRAG
# ---------------------------------------------------------------------------
async def handle_graphrag(question: str):
    async with cl.Step(name="Generando consulta SPARQL") as step:
        try:
            result = await asyncio.to_thread(_graph_answer, question, False)
            sparql_txt = result["sparql"]
            rows = result["rows"]
            n = len(rows)
            step.output = f"**{n} fila{'s' if n != 1 else ''} devuelta{'s' if n != 1 else ''}**"
        except Exception as exc:
            step.output = f"Error al ejecutar SPARQL: {exc}"
            await cl.Message(
                content=f"No se pudo generar una consulta válida para esta pregunta.\n\n*Error: {exc}*"
            ).send()
            return

    # Panel lateral con la consulta SPARQL exacta
    sparql_element = cl.Text(
        name="Consulta SPARQL generada",
        content=f"```sparql\n{sparql_txt}\n```\n*{n} filas devueltas*",
        display="side",
    )

    # Fuentes: PDFs enlazables extraídos de las fechas en las filas SPARQL
    answer = result.get("answer", "(sin respuesta)")
    fuentes = await asyncio.to_thread(_fuentes_graphrag, rows)
    answer += fuentes

    await cl.Message(content=answer, elements=[sparql_element]).send()


# ---------------------------------------------------------------------------
# Preguntas de ejemplo que aparecen al abrir el chat (Starters)
# ---------------------------------------------------------------------------
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Tasa turística",
            message="¿Qué debates sobre la tasa turística ha habido en el Pleno de Bilbao a lo largo de los años?",
        ),
        cl.Starter(
            label="Vivienda social",
            message="¿Qué propuestas sobre vivienda social se han debatido en el Pleno de Bilbao?",
        ),
        cl.Starter(
            label="Medio ambiente",
            message="¿Qué acuerdos sobre medio ambiente y sostenibilidad se han tomado en los plenos?",
        ),
        cl.Starter(
            label="Presupuesto 2024",
            message="¿Qué se debatió sobre el presupuesto municipal de Bilbao en 2024?",
        ),
    ]


# ---------------------------------------------------------------------------
# Inicio de sesión: guarda el motor según el perfil elegido
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start():
    profile = cl.user_session.get("chat_profile")
    cl.user_session.set("rag", _rag)
    cl.user_session.set("mode", "graphrag" if profile == "GraphRAG (SPARQL)" else "vectorial")


# ---------------------------------------------------------------------------
# Respuesta a cada mensaje del usuario
# ---------------------------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    if not question:
        return

    if cl.user_session.get("mode") == "graphrag":
        await handle_graphrag(question)
        return

    rag = cl.user_session.get("rag")

    # Fase 1: Recuperación (embeddings + expansión de topics) en hilo separado
    async with cl.Step(name="Buscando en las actas") as step:
        ctx = await asyncio.to_thread(rag.retrieve_context, question)
        step.output = "Búsqueda completada."

    formatted_context = ctx["context"]
    is_multi_session = ctx["is_multi_session"]
    unique_dates = ctx["unique_dates"]
    retrieved_docs = ctx["docs"]

    if not formatted_context.strip():
        await cl.Message(
            content="Lo siento, no he encontrado información relevante en las actas para esta pregunta. Prueba a reformularla."
        ).send()
        return

    # Construir el prompt según el tipo de pregunta
    if is_multi_session:
        dates_found = ', '.join(sorted(unique_dates))
        sys_prompt = f"""Eres el Cronista Oficial de Bilbao, experto en historia municipal. RESPONDE SIEMPRE EN ESPAÑOL.

INSTRUCCION: Se te proporcionan fragmentos de MULTIPLES plenos del Ayuntamiento de Bilbao.
Las fechas de los plenos en este contexto son: {dates_found}
Responde a la pregunta haciendo un RESUMEN CRONOLOGICO de los debates y propuestas encontrados.

REGLAS CRUCIALES:
- IDIOMA: responde ÚNICAMENTE en español castellano. Está PROHIBIDO usar inglés, ni una sola frase.
- USA SOLO la informacion que esta explicitamente en las actas proporcionadas abajo.
- NUNCA inventes fechas, cifras, nombres, resultados o detalles que no esten en el texto.
- Si no sabes el resultado de una votacion, escribe: [Sin resultado en acta]
- SIEMPRE escribe las fechas en formato DD-MM-YYYY exacto tal como aparecen en el contexto (ej: 26-10-2010), nunca solo el año.
- Para cada pleno relevante desarrolla un parrafo con este formato:
  **[fecha DD-MM-YYYY] — [grupo proponente]**
  - Propuesta: explica con DETALLE que pedia exactamente (los puntos concretos, cifras y medidas).
  - Argumentos: si el acta recoge la justificacion o los argumentos del debate, resumelos CON
    TUS PROPIAS PALABRAS (no hace falta citar textualmente; parafrasear o interpretar fielmente
    lo que dice el texto esta bien). PERO si el acta NO dice nada sobre el porque de la propuesta,
    OMITE esta linea por completo: NO te inventes una justificacion generica que no este respaldada
    por el texto (prohibido rellenar con frases como "para satisfacer la demanda de los ciudadanos"
    si esa idea no aparece en el acta).
  - Resultado: indica el resultado e INCLUYE LAS CIFRAS DE LA VOTACION si aparecen en el texto
    (ej: "Aprobada. Votos a favor: 29, en contra: 0"). Si no hay cifras, escribe solo el resultado textual.
- Ordena de mas antiguo a mas reciente.
- Termina con un parrafo de CONCLUSION que sintetice la evolucion del tema a lo largo de los anos:
  como han cambiado las propuestas, que grupos han sido mas activos y que tendencia se observa.
  Esta conclusion es la UNICA parte donde puedes hacer una sintesis propia; el resto debe ser
  estrictamente fiel al texto.

ACTAS:
{{context}}

PREGUNTA: {{question}}
RESUMEN CRONOLOGICO DETALLADO EN ESPAÑOL:"""
    else:
        sys_prompt = """Eres el Cronista Oficial de Bilbao. Tu misión es relatar lo ocurrido en el Pleno. RESPONDE SIEMPRE EN ESPAÑOL.

INSTRUCCIÓN: Basándote en el ACTA de abajo, responde a: {question}

REGLAS:
- IDIOMA: responde ÚNICAMENTE en español castellano. Prohibido usar inglés.
- Empieza directamente con: "En la sesión del Pleno de Bilbao..."
- Detalla los puntos de la propuesta (qué se pide exactamente).
- Indica el resultado final de la votación si consta.

ACTA:
{context}

PREGUNTA: {question}
CRÓNICA EN ESPAÑOL:"""

    prompt_value = ChatPromptTemplate.from_template(sys_prompt).format_messages(
        context=formatted_context, question=question
    )

    # Fase 2: Generación + fuentes en UN SOLO mensaje. Las fuentes son hipervínculos
    # normales a la ruta /acta/... (abren el PDF en una pestaña del navegador en la
    # página correcta), en lugar de elementos cl.Pdf "side" que se abrían solos.
    async with cl.Step(name="Redactando la crónica"):
        respuesta = await rag.llm.ainvoke(prompt_value)
        answer_text = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
        # Eliminar eco del prompt (PREGUNTA:/RESPUESTA: que el LLM a veces repite al final)
        answer_text = re.sub(r'\n+PREGUNTA\s*:.*', '', answer_text, flags=re.DOTALL | re.IGNORECASE)

    # Insertar enlace de fuente debajo del bloque de cada pleno en la respuesta.
    if retrieved_docs:
        async with cl.Step(name="Localizando fuentes en los PDFs"):
            sources_data = await asyncio.to_thread(build_sources_data, retrieved_docs, answer_text)

        if sources_data:
            concl = re.search(r'\n\s*(?:CONCLUSI[ÓO]N|En conclusi|En resumen)', answer_text, re.I)
            concl_pos = concl.start() if concl else len(answer_text)

            fechas = {s["date"] for s in sources_data}
            bloques = []
            for date in fechas:
                for m in re.finditer(re.escape(date), answer_text[:concl_pos]):
                    bloques.append([m.start(), date])
            bloques.sort()

            src_por_fecha = defaultdict(list)
            for s in sources_data:
                src_por_fecha[s["date"]].append(s)

            asignaciones = []
            usadas = []
            for idx, (start, date) in enumerate(bloques):
                candidatos = [s for s in src_por_fecha[date] if id(s) not in usadas]
                if not candidatos:
                    continue
                fin = bloques[idx + 1][0] if idx + 1 < len(bloques) else concl_pos
                cabecera = answer_text[start:start + 120]
                cab_words = _palabras_clave(cabecera)
                mejor = max(candidatos, key=lambda s: len(cab_words & _palabras_clave(s["topic"])))
                if len(cab_words & _palabras_clave(mejor["topic"])) == 0:
                    mejor = candidatos[0]
                asignaciones.append((fin, mejor))
                usadas.append(id(mejor))

            for fin, s in sorted(asignaciones, key=lambda x: x[0], reverse=True):
                f = fin
                while f > 0 and answer_text[f - 1] in "*#\n\r \t[":
                    f -= 1
                linea = f"\n\n📄 *Fuente:* [{s['pdf_name']}]({s['url']})\n"
                if s.get("vote_result"):
                    linea += f"*Resultado:* {s['vote_result']}\n"
                answer_text = answer_text[:f] + linea + answer_text[f:]

            no_ubicadas = [s for s in sources_data if id(s) not in usadas]
            if no_ubicadas:
                if not is_multi_session:
                    # Sesión única: la fuente va al final limpiamente
                    for s in no_ubicadas:
                        linea_extra = f"\n\n📄 *Fuente:* [{s['pdf_name']}]({s['url']})\n"
                        if s.get("vote_result"):
                            linea_extra += f"*Resultado:* {s['vote_result']}\n"
                        answer_text += linea_extra
                else:
                    answer_text += "\n\n**Otras fuentes:**\n"
                    for s in no_ubicadas:
                        linea_extra = f"* [{s['pdf_name']}]({s['url']})"
                        if s.get("vote_result"):
                            linea_extra += f" — *{s['vote_result']}*"
                        answer_text += linea_extra + "\n"

        # Red de seguridad: garantiza que SIEMPRE aparezcan fuentes si hay docs.
        # Cubre cualquier camino en que la inserción anterior no añadiera ninguna
        # (p.ej. el filtro de relevancia dejó sources_data vacío o el emparejado falló).
        if "📄" not in answer_text and "Otras fuentes" not in answer_text:
            fallback = sources_data or build_sources_data(retrieved_docs)
            if fallback:
                answer_text += "\n\n**Fuentes:**\n"
                for s in fallback:
                    linea = f"* [{s['pdf_name']}]({s['url']})"
                    if s.get("vote_result"):
                        linea += f" — *{s['vote_result']}*"
                    answer_text += linea + "\n"

    await cl.Message(content=answer_text).send()
