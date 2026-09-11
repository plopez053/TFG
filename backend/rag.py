import os
import argparse
import re
import glob
import sys
import time
import threading
import unicodedata
import urllib.parse
from collections import OrderedDict
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Añadido de forma defensiva (funciona tanto si rag.py se ejecuta directo como
# CLI, como si se importa desde otro módulo que ya lo tenga en el path) para
# poder importar la normalización de grupos, compartida con el pipeline de GraphRAG.
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from graphrag.graphrag.grupos import (  # noqa: E402
    normaliza_grupo as _normaliza_grupo_partido,
    extrae_grupo as _extrae_grupo_partido,
)
from backend.providers import (  # noqa: E402
    EMBEDDING_MODEL, LLM_MODEL_LOCAL, LLM_MODEL_GROQ, LLM_MODEL_GRAPHRAG,
    COHERE_RERANK_MODEL, COHERE_API_KEY, GROQ_API_KEY,
    ping_ollama as _ping_ollama, LLMProvider, rerank as _cohere_rerank,
)

# Autodetecta la ubicación de las actas: dentro del proyecto (portátil) o fuera (equipo potente).
_ACTAS_DENTRO = os.path.join(BASE_DIR, "actas")
_ACTAS_FUERA = os.path.join(os.path.dirname(BASE_DIR), "actas")
DATA_PATH = _ACTAS_DENTRO if os.path.isdir(_ACTAS_DENTRO) else _ACTAS_FUERA
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Umbral de distancia para descartar resultados semánticos irrelevantes.
SIMILARITY_DISTANCE_MAX = 1.4

# Límite de caracteres del contexto que se pasa al LLM. Se usa el valor LOCAL
# siempre que Ollama pueda responder (num_ctx=8192), porque invoke_llm cae al
# fallback local ante cualquier fallo de Groq.
CONTEXT_CHAR_LIMIT_LOCAL = 12000
CONTEXT_CHAR_LIMIT_GROQ = 24000

_DEBUG_CONTEXT_PATH = os.path.join(BASE_DIR, "debug_context.txt")
_DEBUG_CONTEXT_LOCK = threading.Lock()
_RAG_SINGLETON_LOCK = threading.Lock()


# True si Ollama está activo en localhost:11434
def _ping_ollama(timeout: float = 3.0) -> bool:
    try:
        import httpx
        return httpx.get("http://localhost:11434/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False
# Prompt de MultiQuery: genera variantes de la pregunta para ampliar el recall.
MULTIQUERY_PROMPT = "Genera 3 variantes de: '{question}' centradas en el sujeto principal. Una por línea:"
# Relevancia mínima del reranker para considerar que hay algo que responder.
RELEVANCE_FLOOR = 0.01

# ---------------------------------------------------------------------------
# Prompts canónicos — punto único de definición, compartidos por CLI y frontend
# (build_answer_prompt los usa para las dos vías). Usar .format(dates_found=,
# context=, question=).
# ---------------------------------------------------------------------------
_PROMPT_MULTI_SESSION = (
    "Eres el Cronista Oficial de Bilbao, experto en historia municipal. RESPONDE SIEMPRE EN ESPAÑOL.\n\n"
    "INSTRUCCION: Se te proporcionan fragmentos de MULTIPLES plenos del Ayuntamiento de Bilbao.\n"
    "Las fechas de los plenos en este contexto son: {dates_found}\n"
    "Responde a la pregunta haciendo un RESUMEN CRONOLOGICO de los debates y propuestas encontrados.\n\n"
    "REGLAS CRUCIALES:\n"
    "- IDIOMA: responde ÚNICAMENTE en español castellano. Está PROHIBIDO usar inglés, ni una sola frase.\n"
    "- USA SOLO la informacion que esta explicitamente en las actas proporcionadas abajo.\n"
    "- NUNCA inventes fechas, cifras, nombres, resultados o detalles que no esten en el texto.\n"
    "- Si no sabes el resultado de una votacion, escribe: [Sin resultado en acta]\n"
    "- SIEMPRE escribe las fechas en formato DD-MM-YYYY exacto tal como aparecen en el contexto "
    "(ej: 26-10-2010), nunca solo el año.\n"
    "- Para cada pleno relevante desarrolla un parrafo con este formato:\n"
    "  **[fecha DD-MM-YYYY] — [grupo proponente]**\n"
    "  - Propuesta: explica con DETALLE que pedia exactamente (los puntos concretos, cifras y medidas).\n"
    "  - Argumentos: si el acta recoge la justificacion o los argumentos del debate, resumelos CON\n"
    "    TUS PROPIAS PALABRAS. PERO si el acta NO dice nada sobre el porque, OMITE esta linea por\n"
    "    completo: NO te inventes una justificacion generica no respaldada por el texto.\n"
    "  - Resultado: indica el resultado e INCLUYE LAS CIFRAS DE LA VOTACION si aparecen en el texto\n"
    "    (ej: \"Aprobada. Votos a favor: 29, en contra: 0\"). Si no hay cifras, escribe solo el resultado.\n"
    "  - COHERENCIA VOTOS: si los votos en contra son 0 o no aparecen, el resultado NO puede ser\n"
    "    \"rechazada\". No mezcles el resultado de una enmienda con los votos de la votación principal.\n"
    "- Ordena de mas antiguo a mas reciente.\n"
    "- Termina con un parrafo de CONCLUSION que sintetice la evolucion del tema a lo largo de los anos.\n\n"
    "ACTAS:\n{context}\n\n"
    "PREGUNTA: {question}\n"
    "RESUMEN CRONOLOGICO DETALLADO EN ESPAÑOL:"
)

_PROMPT_SINGLE_SESSION = (
    "Eres el Cronista Oficial de Bilbao. Tu misión es relatar lo ocurrido en el Pleno. RESPONDE SIEMPRE EN ESPAÑOL.\n\n"
    "INSTRUCCIÓN: Basándote en el ACTA de abajo, responde a: {question}\n\n"
    "REGLAS:\n"
    "- IDIOMA: responde ÚNICAMENTE en español castellano. Prohibido usar inglés.\n"
    "- Empieza directamente con: \"En la sesión del Pleno de Bilbao...\"\n"
    "- Detalla los puntos de la propuesta (qué se pide exactamente).\n"
    "- Indica el resultado final de la votación si consta.\n\n"
    "ACTA:\n{context}\n\n"
    "PREGUNTA: {question}\n"
    "CRÓNICA EN ESPAÑOL:"
)

# funcion para quitar acentos y diacríticios
def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()


# Palabras omnipresentes en cualquier título/pregunta sobre el Pleno de Bilbao
# que no aportan señal temática. Único punto de esta lista: la usan tanto
# _expand_context_by_topic() (aspiradora) como _extraer_keywords_pregunta()
# (búsqueda literal complementaria a la semántica, ver _initial_search).
_STOPWORDS_TEMATICAS = {
    "bilbao", "municipal", "municipales", "partido", "popular", "grupos",
    "acuerdo", "acuerdos", "proposicion", "proposizioa", "propuesta", "propuestas",
    "mocion", "mozioa", "debate", "debates", "sesion", "reunion",
    "resultado", "votacion", "propone", "propuso", "propuesto", "presenta", "plantea",
    "siguiente", "adoptado", "tomado", "tomados", "relacionado", "relacionados",
    "dispositiva", "literal", "adopcion", "decidio", "habido", "habida",
    "cuales", "exacto",
    "febrero", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
}


# palabras >=6 caracteres de la pregunta, sin acentos ni términos omnipresentes en actas
def _extraer_keywords_pregunta(question: str) -> List[str]:
    q_clean = re.sub(r'[^a-z0-9\s]', '', strip_accents(question.lower()))
    return [w for w in re.findall(r'\w{6,}', q_clean) if w not in _STOPWORDS_TEMATICAS and not w.isdigit()]


# palabras >=6 caracteres CONSERVANDO tildes, para la búsqueda literal por substring en ChromaDB
def _extraer_keywords_literales(question: str) -> List[str]:
    q_raw = re.sub(r'[^\w\s]', '', question.lower(), flags=re.UNICODE)
    out = []
    for w in re.findall(r'\w{6,}', q_raw, flags=re.UNICODE):
        if w.isdigit():
            continue
        if strip_accents(w) in _STOPWORDS_TEMATICAS:
            continue
        out.append(w)
    return out


_TEMA_ROLLUP_CACHE: Optional[Dict[str, Any]] = None
_TEMA_ROLLUP_LOCK = threading.Lock()


# tabla cacheada 'tema nivel 1 <- subtemas' de la taxonomía SKOS (ver memoria/decisiones_tecnicas.md 1.2)
def _load_tema_rollup() -> Dict[str, Any]:
    global _TEMA_ROLLUP_CACHE
    if _TEMA_ROLLUP_CACHE is not None:
        return _TEMA_ROLLUP_CACHE
    # Lock: retrieve_context corre en threads separados por petición
    # (asyncio.to_thread desde el frontend); sin esto dos preguntas
    # concurrentes en frío parsean el TTL dos veces a la vez.
    with _TEMA_ROLLUP_LOCK:
        if _TEMA_ROLLUP_CACHE is not None:  # otro hilo ya lo cargó mientras esperábamos
            return _TEMA_ROLLUP_CACHE
        result = _build_tema_rollup()
        if result is not None:
            # Un fallo NO se cachea: así la siguiente pregunta lo reintenta
            # en vez de dejar el canal temático apagado en todo el proceso.
            _TEMA_ROLLUP_CACHE = result
            return result
        return {"detect": [], "filter_values": {}}


# construye el roll-up de temas desde la taxonomía SKOS; None si falla (no se cachea)
def _build_tema_rollup() -> Optional[Dict[str, Any]]:
    try:
        from rdflib import Graph, RDF
        from rdflib.namespace import SKOS
        from graphrag.graphrag.build_rdf import ONTOLOGY, THEMES, BO
        g = Graph()
        g.parse(ONTOLOGY, format="turtle")
        g.parse(THEMES, format="turtle")

        def es_labels(concept) -> List[str]:
            # Etiquetas en cualquier idioma: canon_theme_map() en build_rdf.py
            # (la función gemela para el grafo) también acepta prefLabel/altLabel
            # en euskera, y los dos sistemas deben clasificar igual.
            labs = [g.value(concept, SKOS.prefLabel)]
            labs += list(g.objects(concept, SKOS.altLabel))
            return [str(l) for l in labs if l]

        toplevel = {}  # uri -> prefLabel
        for c, _, _ in g.triples((None, RDF.type, BO.Tema)):
            if not list(g.triples((c, SKOS.broader, None))):
                lbl = g.value(c, SKOS.prefLabel)
                if lbl:
                    toplevel[c] = str(lbl)

        detect: List[Tuple[str, str]] = []
        filter_values: Dict[str, set] = {pref: {pref} for pref in toplevel.values()}
        for c, pref in toplevel.items():
            for lbl in es_labels(c):
                detect.append((strip_accents(lbl.lower()), pref))

        for c, _, _ in g.triples((None, RDF.type, BO.Tema)):
            parent = g.value(c, SKOS.broader)
            if parent is None or parent not in toplevel:
                continue
            parent_pref = toplevel[parent]
            sub_pref = g.value(c, SKOS.prefLabel)
            if sub_pref:
                filter_values[parent_pref].add(str(sub_pref))
            for lbl in es_labels(c):
                detect.append((strip_accents(lbl.lower()), parent_pref))

        # slug de nivel 1 por prefLabel (br:t_medioambiente -> "medioambiente"),
        # para filtrar en Chroma por la bandera tf_<slug> que escribe
        # enrich_vector_metadata.py (cubre tema_principal Y secundarios).
        parent_slug = {}
        for uri, pref in toplevel.items():
            local = str(uri).rsplit("/", 1)[-1].split("#")[-1]
            parent_slug[pref] = local[2:] if local.startswith("t_") else local

        # Más largas primero: para que "medio ambiente" no quede tapado por
        # una coincidencia parcial de un término más corto y genérico.
        detect.sort(key=lambda x: len(x[0]), reverse=True)
        return {
            "detect": detect,
            "filter_values": {p: sorted(v) for p, v in filter_values.items()},
            "parent_slug": parent_slug,
        }
    except Exception as e:
        print(f"[!] No se pudo cargar la taxonomía de temas: {type(e).__name__}: {e}", flush=True)
        return None


# clave de orden cronológico (año, mes, día) a partir de una fecha 'DD-MM-YYYY'
def _date_str_sort_key(date_str: str) -> Tuple[int, int, int]:
    try:
        parts = (date_str or '01-01-1900').split('-')
        if len(parts) == 3:
            return (int(parts[2]), int(parts[1]), int(parts[0]))
    except (ValueError, AttributeError):
        pass
    return (1900, 1, 1)


# clave de orden cronológico a partir del metadato 'date' (DD-MM-YYYY) de un Document
def _date_sort_key(doc: Document) -> Tuple[int, int, int]:
    return _date_str_sort_key(doc.metadata.get('date', ''))


# ---------------------------------------------------------------------------
# Utilidades de fuentes/presentación movidas desde frontend/app.py al backend
# para que la CLI y Chainlit compartan exactamente la misma lógica.
# ---------------------------------------------------------------------------

# convierte rutas de la BD vectorial (Windows o Unix) al path local equivalente
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


# Vocabulario procedimental común a casi todos los debates: se excluye al medir
# relevancia temática para que solo cuenten las palabras de ASUNTO del debate.
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


# palabras significativas de un texto (≥4 letras, sin acentos ni ruido estructural)
def _palabras_clave(texto: str) -> set:
    texto = strip_accents(texto.lower())
    stop = {
        "grupo", "municipal", "politico", "proposicion", "proposamena", "presenta",
        "cuya", "parte", "dispositiva", "plantea", "adopcion", "acuerdo", "plenario",
        "propuesta", "pleno", "ayuntamiento", "bilbao", "equipo", "gobierno",
        "resultado", "argumentos", "fuente", "instar", "insta",
        "votacion", "sesion",
    }
    return {w for w in re.findall(r"[a-z]{4,}", texto) if w not in stop}


# construye los metadatos de fuentes (una entrada por debate citado)
def build_sources_data(retrieved_docs, answer_text=None):
    por_grupo = OrderedDict()
    for doc in retrieved_docs:
        pdf_path = resolve_pdf_path(doc.metadata.get("source", ""))
        if not pdf_path or not os.path.exists(pdf_path):
            continue
        date = doc.metadata.get("date", "Fecha desconocida")
        topic = doc.metadata.get("topic", "")
        # Excluir portada/índice: no es un debate citable.
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
        rango = (f"Pág. {p_min}" if p_min == p_max else f"Págs. {p_min}-{p_max}") if p_min else None

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
        contenido = " ".join(d.page_content for d in docs_d)
        content_kw = _palabras_clave(contenido) - _STOP_PROCEDIMENTAL

        sources_data.append({
            "date": date, "topic": topic, "pdf_path": pdf_path,
            "url": url, "page": p_min or 1, "short_topic": short_topic,
            "pdf_name": pdf_name, "vote_result": vote_result, "content_kw": content_kw,
        })

    # Filtro de relevancia SOLO en sesión única (pregunta de un pleno concreto):
    # el buscador trae muchos debates del día y hay que quedarse con los que la
    # respuesta trata (≥4 palabras de asunto compartidas, ya sin ruido procedimental).
    fechas_distintas = {s["date"] for s in sources_data}
    if answer_text and len(fechas_distintas) == 1:
        ans_kw = _palabras_clave(answer_text) - _STOP_PROCEDIMENTAL
        relevantes = [s for s in sources_data if len(s["content_kw"] & ans_kw) >= 4]
        if relevantes:
            sources_data = relevantes

    return sources_data


class RAGPipeline:
    # inicializa el motor RAG con el modelo de embeddings configurado
    def __init__(self):
        self._ollama_ok = _ping_ollama()
        if not self._ollama_ok:
            print("[!] ADVERTENCIA: Ollama no responde en localhost:11434 "
                  "— embeddings y LLM local no disponibles", flush=True)

        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = None
        self._llm_provider = LLMProvider(prefer="groq")
        self._known_dates: Optional[List[str]] = None  # caché de os.walk
        self.last_retrieved_docs: List[Document] = []

    # invoca el LLM (Groq -> Ollama con fallback automático, ver backend/providers.py)
    def invoke_llm(self, prompt):
        return self._llm_provider.invoke(prompt)

    async def ainvoke_llm(self, prompt):
        return await self._llm_provider.ainvoke(prompt)

    # LLM principal (para compatibilidad con código externo)
    @property
    def llm(self):
        return self._llm_provider.primary

    # límite de caracteres de contexto seguro dado qué proveedor puede
    def _context_char_limit(self) -> int:
        return CONTEXT_CHAR_LIMIT_LOCAL if self._ollama_ok else CONTEXT_CHAR_LIMIT_GROQ

    # --- Métodos de Utilidad ---

    # extrae la fecha de un nombre de archivo (ej: '27-02-2025_...pdf')
    def _extract_date_from_filename(self, path: str) -> str:
        filename = os.path.basename(path)
        match = re.search(r'(\d{2}-\d{2}-\d{4})', filename)
        return match.group(1) if match else "Fecha desconocida"

    # escanea las primeras páginas del acta para mapear nombres de concejales a partidos
    def _get_party_mapping(self, pages: List[Any]) -> Dict[str, str]:
        party_mapping = {}
        header_text = "\n".join([p.page_content for p in pages[:10]])
        # "Gobierno", no "Goberno": una errata aquí haría que
        # grupos.normaliza_grupo() no reconociera este valor por defecto como
        # "GOBIERNO" y se quedara sin normalizar, invisible para cualquier
        # filtro que busque "EQUIPO DE GOBIERNO".
        current_party = "Gobierno Local/Otros"

        for line in header_text.split('\n'):
            line = line.strip()
            if not line: continue

            re_esp = re.search(r"En representación del grupo municipal\s+([A-Z\s-]+)", line, re.IGNORECASE)
            re_eus = re.search(r"([A-Z\s-]+)\s+udal talde politikoaren izenean", line, re.IGNORECASE)

            if re_esp:
                current_party = re_esp.group(1).strip().strip(':')
                continue
            if re_eus:
                current_party = re_eus.group(1).strip().strip(':')
                continue

            re_member = re.search(r"^\d+\.-?\s*(?:DON|DOÑA|SR\.|SRA\.)?\s*([A-ZÁÉÍÓÚÑ]{4,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)", line, re.IGNORECASE)
            if re_member:
                name = re_member.group(1).strip()
                paren = re.search(r'\(([^)]+)\)', line)
                raw_party = paren.group(1).strip() if paren else current_party
                normalized = _normaliza_grupo_partido(raw_party)
                # "Desconocido" solo si el texto crudo también lo era; si no,
                # conservamos el texto original en vez de perder la información
                # (mejor un partido sin canonizar que ninguno).
                party_mapping[name] = normalized if normalized != "Desconocido" else raw_party

        return party_mapping

    # --- Fase de Ingesta (ETL) ---

    # procesa un único PDF y devuelve sus chunks con metadatos (page + vote_result)
    def _process_single_pdf(self, path: str) -> List[Document]:
        speaker_regex = re.compile(r'(?:(?:EL|LA)\s+)?(?:SR\.|SRA\.)\s+([A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)\s*[:.]', re.IGNORECASE)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        # Ventanas amplias (0,500) entre etiquetas: en las actas BILINGÜES (euskera+
        # castellano) entre "Votos emitidos" y "Votos afirmativos" se intercala el
        # bloque en euskera ("Baiezko botoak: N jaun/andre: [nombres]...") + la lista
        # de concejales, que supera con creces los 80 caracteres del patrón antiguo.
        # Con 80 se perdían TODOS los votos de muchos plenos (p.ej. 27-10-2022: 0 de 26).
        vote_re = re.compile(
            r'Votos\s+emitidos[:\s]+(\d+)'
            r'.{0,500}?Votos\s+afirmativos[:\s]+(\d+)'
            r'(?:.{0,500}?Votos\s+negativos[:\s]+(\d+))?'
            r'(?:.{0,500}?Abstenciones?[:\s]+(\d+))?',
            re.IGNORECASE | re.DOTALL
        )
        # [^.] en vez de . para parar en el PUNTO que cierra la frase del resultado
        # (evita tragarse narración posterior como "- Siendo las 14:05 horas, el
        # señor Alcalde anuncia el receso..."). Margen amplio (0,400) para no cortar
        # a media palabra frases largas con varios grupos ("...el Grupo ELKARREKIN...").
        result_re = re.compile(
            r'(?:se\s+(?:acepta|aprueba|rechaza|desestima|deniega)\b[^.]{0,30}?'
            r'(?:enmienda|proposici[óo]n|propuesta|moci[óo]n|mozio|proposamen)[^.]{0,400}|'
            r'queda\s+(?:aprobad[ao]|rechazad[ao]|desestimad[ao])[^.]{0,400}|'
            r'resulta\s+(?:aprobad[ao]|rechazad[ao])[^.]{0,400})',
            re.IGNORECASE | re.DOTALL
        )
        # Resultados SIN cifras (acuerdos unánimes / por asentimiento). Muy frecuentes,
        # sobre todo en actas antiguas: "El Pleno Municipal, por unanimidad de miembros
        # presentes, acuerda...", "Aprobar por unanimidad...". Anclado en marcadores
        # formales del resultado para no confundirlo con discurso del debate.
        unanim_re = re.compile(
            r'(?:(?:el\s+pleno(?:\s+municipal)?|excmo\.?\s+ayuntamiento\s+pleno|aprobar)'
            r'[^.]{0,60}por\s+unanimidad[^.]{0,200}'
            r'|se\s+(?:aprueba|acuerda|aprueban|desestiman?|rechazan?)[^.]{0,60}'
            r'por\s+(?:unanimidad|asentimiento)[^.]{0,200})',
            re.IGNORECASE
        )
        chunks = []

        date = self._extract_date_from_filename(path)
        pdf_loader = PyPDFLoader(path)
        pages = pdf_loader.load()
        party_map = self._get_party_mapping(pages)

        # Mapa de offsets para rastrear nº de página por posición en el texto
        page_offsets = []  # (char_offset, page_number 1-indexed)
        full_text_parts = []
        cursor = 0
        for i, page in enumerate(pages):
            page_offsets.append((cursor, i + 1))
            full_text_parts.append(page.page_content)
            cursor += len(page.page_content) + 1
        full_text = "\n".join(full_text_parts)

        def page_for_offset(offset: int) -> int:
            pg = 1
            for off, num in page_offsets:
                if off <= offset:
                    pg = num
                else:
                    break
            return pg

        split_regex = re.compile(
            r'(?=\n(?:[\s]*)(?:\d+)\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA)'
            r'|\n\s*-\d+-\s*\n\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA|Proposición|Propuesta|Moción|Mozio|Dictamen|Irizpen|Asunto|Gaia|Proposamen))',
            re.IGNORECASE
        )
        segments = split_regex.split(full_text)

        # Tipos de punto del orden del día en el formato ANTIGUO (2002-2009), donde el
        # marcador es "- N - \n\n TIPO ...". Sirve para distinguir un punto real de un
        # simple número de PÁGINA (también escrito "- N -"), que NO debe partir el acta.
        _OLD_TYPES = (r'(?:Proposici[oó]n|Proposizioa|Propuesta|Preguntas?|'
                      r'Se\s+da\s+cuenta|Dar\s+cuenta|Dictamen|Moci[oó]n|'
                      r'Comparecencia|Interpelaci[oó]n|Declaraci[oó]n|Aprobar)')
        split_old_agenda = re.compile(r'(?=\n[\s]*-\s*\d+\s*-\s*\n[\s]*' + _OLD_TYPES + r')', re.IGNORECASE)
        old_topic_re = re.compile(r'^\s*-\s*(\d+)\s*-\s*\n[\s]*(' + _OLD_TYPES + r'.{0,140})', re.IGNORECASE | re.DOTALL)

        is_old_format = len(segments) <= 2
        if is_old_format:
            # 1º intentar partir por los puntos REALES del orden del día (ordinarias
            # antiguas) → topics limpios y un voto por punto. Si hay varios, usarlo.
            agenda_segs = [s for s in split_old_agenda.split(full_text) if s.strip()]
            if len(agenda_segs) > 2:
                segments = agenda_segs
            else:
                # Extraordinarias (p.ej. presupuestos): sin marcadores de orden del día;
                # se mantiene el troceo por página (comportamiento previo, sin regresión).
                split_regex_old = re.compile(r'(?=\n-\s*\d+\s*-\s*\n)')
                segments = split_regex_old.split(full_text)

        def _limpiar(txt):
            return re.split(
                r'\s*-{3,}\s*|\s+-\s+|\s*https?://|\s+Egiaztatzeko|\s+Verificaci|\s+Siendo\s+las\b',
                txt.strip()
            )[0].strip()

        chunk_index_in_doc = 0
        seg_search_start = 0
        for segment in segments:
            if not segment.strip():
                continue
            current_speaker, current_party = "Desconocido", "Desconocido"

            # Posición del segmento en el texto completo (para calcular página)
            segment_offset = full_text.find(segment, seg_search_start)
            if segment_offset != -1:
                seg_search_start = segment_offset + max(1, len(segment) - 200)

            # Extraer vote_result a nivel de segmento limpio (sin solapamiento de chunks)
            seg_flat = re.sub(r'\s+', ' ', segment)
            resultado_text = None
            rm = result_re.search(seg_flat)
            if rm:
                resultado_text = re.split(
                    r'\s*-{3,}\s*|\s+-\s+|\s*https?://|\s+Egiaztatzeko|\s+Verificaci|\s+Siendo\s+las\b',
                    rm.group(0).strip()
                )[0].strip()
            resultado_num = None
            votes = list(vote_re.finditer(seg_flat))
            if votes:
                vm = votes[-1]
                emitidos, favor, contra, absten = vm.group(1), vm.group(2), vm.group(3), vm.group(4)
                # Basta con "a favor": las votaciones unánimes no traen "en contra"
                # y antes se descartaban (vote_result quedaba None y se perdía el voto).
                if favor:
                    partes = [f"a favor: {favor}"]
                    if contra:
                        partes.append(f"en contra: {contra}")
                    if absten:
                        partes.append(f"abstenciones: {absten}")
                    cab = f"Votos emitidos: {emitidos} | " if emitidos else ""
                    resultado_num = cab + ", ".join(partes)
            # Prioridad: cifras > texto unánime/asentimiento. Se evita guardar texto de
            # resultado "a secas" sin cifras (podría ser discurso); solo se acepta sin
            # cifras si es claramente un acuerdo unánime o por asentimiento.

            if resultado_num:
                vote_result = f"{resultado_text} ({resultado_num})" if resultado_text else resultado_num
            elif resultado_text and re.search(r'unanimidad|asentimiento', resultado_text, re.I):
                vote_result = resultado_text
            else:
                um = unanim_re.search(seg_flat)
                vote_result = _limpiar(um.group(0)) if um else None

            if is_old_format:
                # 1º: encabezado REAL del punto ("- N - \n TIPO ...") → "N. TIPO ...".
                tm_old = old_topic_re.search(segment.lstrip('\n'))
                if tm_old:
                    texto = re.sub(r'\s+', ' ', tm_old.group(2)).strip()
                    current_topic = f"{tm_old.group(1)}. {texto[:115]}"
                else:
                    # 2º: fallback (extraordinarias troceadas por página): 1ª línea.
                    topic_raw = re.search(r'^-\s*\d+\s*-\s*\n\s*(.{0,200})', segment.lstrip('\n'), re.DOTALL)
                    if topic_raw:
                        first_line = topic_raw.group(1).strip().split('\n')[0].strip()
                        first_line = re.sub(r' {2,}', ' ', first_line)
                        current_topic = first_line[:120] if first_line else "General / Introducción"
                    else:
                        current_topic = "General / Introducción"
            else:
                topic_match_std = re.search(
                    r'^\s*(\d+\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA).{0,400})',
                    segment, re.IGNORECASE | re.DOTALL
                )
                topic_match_hist = re.search(
                    r'^\s*-\s*(\d+)\s*-\s*\n\s*((?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA|Proposición|Propuesta|Moción|Mozio|Dictamen|Irizpen|Asunto|Gaia|Proposamen).{0,400})',
                    segment, re.IGNORECASE | re.DOTALL
                )
                if topic_match_std:
                    current_topic = topic_match_std.group(1).strip().replace('\n', ' ')
                elif topic_match_hist:
                    clean_text = topic_match_hist.group(2).strip().replace('\n', ' ')
                    current_topic = f"{topic_match_hist.group(1)}. {clean_text}"
                else:
                    current_topic = "General / Introducción"

            chunk_search_start = 0
            for chunk_text in text_splitter.split_text(segment):
                match = speaker_regex.search(chunk_text)
                if match:
                    current_speaker = match.group(1).strip()
                    if len(current_speaker) < 50:
                        for kn, kp in party_map.items():
                            if kn in current_speaker or current_speaker in kn:
                                current_party = kp
                                break

                # Página del chunk: buscar su posición en el segmento
                if segment_offset != -1:
                    chunk_pos = segment.find(chunk_text[:60], chunk_search_start)
                    if chunk_pos != -1:
                        page_num = page_for_offset(segment_offset + chunk_pos)
                        chunk_search_start = chunk_pos
                    else:
                        page_num = page_for_offset(segment_offset)
                else:
                    page_num = 1

                metadata = {
                    "source": path, "date": date, "speaker": current_speaker,
                    "party": current_party, "topic": current_topic,
                    "chunk_index": chunk_index_in_doc, "page": page_num,
                }
                if vote_result:
                    metadata["vote_result"] = vote_result

                chunks.append(Document(
                    page_content=f"ASUNTO: {current_topic}\nORADOR: {current_speaker} ({current_party})\n\n{chunk_text}",
                    metadata=metadata
                ))
                chunk_index_in_doc += 1
        return chunks

    # carga los PDFs y los divide en fragmentos con metadatos de forma recursiva
    def load_and_split_documents(self) -> List[Document]:
        pdf_files = glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True)

        if not pdf_files:
            print(f"[!] No se encontraron archivos PDF en {DATA_PATH}")
            return []

        print(f"[*] Encontrados {len(pdf_files)} archivos PDF. Procesando...")

        all_chunks = []
        for path in tqdm(pdf_files, desc="Procesando Actas", unit="pdf"):
            try:
                all_chunks.extend(self._process_single_pdf(path))
            except Exception as e:
                print(f"[!] Error cargando {path}: {e}")

        return all_chunks

    # crea o carga la base de datos vectorial ChromaDB
    def create_vector_store(self):
        if os.path.exists(CHROMA_PATH):
            print(f"[*] Cargando base de datos vectorial desde {CHROMA_PATH}...")
            self.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        else:
            chunks = self.load_and_split_documents()
            if not chunks: return
            print("[*] Generando embeddings...")
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=CHROMA_PATH)
            print(f"[+] Base de datos guardada en {CHROMA_PATH}")

    # --- Fase de Recuperación (Retrieval) ---

    # analiza la pregunta para aplicar filtros de fecha inteligentes
    def _get_temporal_filter(self, question: str) -> Optional[Dict[str, Any]]:
        # Recopilar fechas conocidas del sistema de archivos (una sola vez por instancia)
        if self._known_dates is None:
            dates = []
            for root, dirs, files in os.walk(DATA_PATH):
                for file in files:
                    if file.endswith(".pdf"):
                        d = self._extract_date_from_filename(file)
                        if d: dates.append(d)
            self._known_dates = list(set(dates))
        known_dates = self._known_dates

        # Prioridad 1: fecha exacta DD-MM-YYYY en la pregunta
        exact_match = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', question)
        if exact_match:
            exact_date = exact_match.group(0)
            if exact_date in known_dates:
                print(f"[*] Filtro fecha exacta: {exact_date}")
                return {"date": {"$eq": exact_date}}, [exact_date]
            # Fallback a mes-año si la fecha exacta no existe
            target_pattern = f"{exact_match.group(2)}-{exact_match.group(3)}"
            valid_dates = [d for d in known_dates if target_pattern in d]
            if valid_dates:
                print(f"[*] Filtro Temporal (mes-año): {target_pattern} ({len(valid_dates)} actas)")
                return {"date": {"$in": valid_dates}}, valid_dates
            return None, []

        # Prioridad 2: año + nombre de mes en español
        months = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        year_match = re.search(r'(20\d{2})', question)
        month_found = next((m for m in months if m in question.lower()), None)

        if not year_match: return None, []

        target_year = year_match.group(1)
        target_month = f"{str(months.index(month_found)+1).zfill(2)}" if month_found else None
        target_pattern = f"{target_month}-{target_year}" if target_month else target_year

        valid_dates = [d for d in known_dates if target_pattern in d]
        if valid_dates:
            print(f"[*] Filtro Temporal: {target_pattern} ({len(valid_dates)} actas detectadas)")
            return {"date": {"$in": valid_dates}}, valid_dates
        return None, []

    # detecta si la pregunta menciona un grupo político concreto
    def _detect_party_in_question(self, question: str) -> Optional[str]:
        q = question.lower()
        # Más específico primero para evitar falsos positivos
        if re.search(r'eh\s*bildu|euskal\s+herria\s+bildu|herri\s+batasuna|\bhb\b', q):
            return "EH BILDU"
        if re.search(r'elkarrekin|bilbao\s+en\s+com[uú]n|\bpodemos\b', q):
            return "ELKARREKIN BILBAO"
        if re.search(r'\bgoazen\b', q):
            return "GOAZEN BILBAO"
        if re.search(r'pse[\s\-]ee|\bsocialistas?\s+vascos?\b|\bpartido\s+socialista\b|\bpse\b', q):
            return "PSE-EE"
        if re.search(r'eaj[\s\-]pnv|\bpnv\b|\bnacionalistas?\s+vascos?\b', q):
            return "EAJ-PNV"
        if re.search(r'\bpartido\s+popular\b|\bgrupo\s+(?:municipal\s+)?pp\b|\bel\s+pp\b|\bdel\s+pp\b|\bpopulares\b', q):
            return "PP"
        if re.search(r'udalberri', q):
            return "UDALBERRI"
        if re.search(r'\bciudadanos\b', q):
            return "CIUDADANOS"
        if re.search(r'ezker\s+batua|izquierda\s+unida', q):
            return "EZKER BATUA-IU"
        if re.search(r'\baralar\b', q):
            return "ARALAR"
        if re.search(r'\bvox\b', q):
            return "VOX"
        if re.search(r'equipo\s+de\s+gobierno|gobierno\s+municipal', q):
            return "EQUIPO DE GOBIERNO"
        if re.search(r'grupo\s+mixto', q):
            return "GRUPO MIXTO"
        return None

    # técnica de la 'Aspiradora': Expande los fragmentos semánticos a debates completos
    def _expand_context_by_topic(self, initial_docs: List[Document], question: str) -> List[Document]:
        target_topics, seen = [], set()

        # Normalizamos acentos y caracteres especiales para asegurar coincidencia robusta
        def normalize(txt: str) -> str:
            txt = strip_accents(txt.lower())
            txt = re.sub(r'[^a-z0-9\s]', '', txt)
            return txt

        q_clean = normalize(question)
        # Lista de stopwords y extracción de keywords compartidas con la búsqueda
        # literal complementaria de _initial_search (ver _extraer_keywords_pregunta).
        q_keywords = _extraer_keywords_pregunta(question)

        # --- PASADA 1: Puntuar temas de la búsqueda semántica ---
        for doc in initial_docs:
            topic = doc.metadata.get("topic")
            source = doc.metadata.get("source")
            if topic and topic not in ["General", "General / Introducción"]:
                source_bn = os.path.basename(source)
                if (source_bn, topic) not in seen:
                    norm_topic = normalize(topic)
                    norm_content = normalize(doc.page_content)
                    # Stem de 6 chars: más específico que 5, menos falsos positivos.
                    # Puntuación: 20 pts título (señal fuerte) / 2 pts contenido (señal débil)
                    score = sum(
                        20 if kw[:6] in norm_topic else 2 if kw[:6] in norm_content else 0
                        for kw in q_keywords
                    )
                    seen.add((source_bn, topic))
                    target_topics.append({"topic": topic, "source": source, "score": score, "basename": source_bn})



        # PENALIZACIÓN DE TEMAS PRESUPUESTARIOS: Los debates de presupuestos son muy largos
        # y mencionan de pasada cualquier tema (medio ambiente, pobreza, movilidad…).
        # Si la pregunta no menciona "presupuesto" ni "modificación", penalizamos esos temas
        # para que no contaminen búsquedas temáticas específicas.
        q_mentions_budget = bool(re.search(r'presupuest|modificaci[oó]n\s+presup|ordenanza', q_clean, re.I))
        if not q_mentions_budget:
            for t in target_topics:
                t_norm = normalize(t['topic'])
                if re.search(r'presupuest|modificac|ordenanza', t_norm):
                    t['score'] *= 0.2

        # FILTRO DE TÍTULO: si algún tema tiene una keyword en su título (score
        # >= 20), se descartan los que solo puntúan por menciones de pasada en
        # el contenido. Si ninguno llega a 20, se mantienen todos.
        title_matched = [t for t in target_topics if t['score'] >= 20]
        if title_matched:
            target_topics = title_matched

        # DOBLE UMBRAL para eliminar ruido:
        # 1) Umbral absoluto: evita que temas con score bajo pasen cuando todos los scores son bajos.
        #    Con stem 6 chars + peso 20/2: score=4 → al menos 2 contenidos relevantes.
        #    score=20 → una keyword clave en el título (señal muy fuerte).
        # 2) Umbral relativo: descarta temas con menos del 40% del mejor score.
        MIN_ABSOLUTE_SCORE = 4
        target_topics = sorted(
            [t for t in target_topics if t['score'] >= MIN_ABSOLUTE_SCORE],
            key=lambda x: x['score'], reverse=True
        )

        if target_topics:
            max_s = target_topics[0]['score']
            if max_s > 0:
                # Umbral relativo: descartar temas por debajo del 40% del score máximo
                global_threshold = max_s * 0.4
                target_topics = [t for t in target_topics if t['score'] >= global_threshold]

            # Estrategia de diversidad temporal: preferimos 1 tema por año distinto
            # para que preguntas "a lo largo de los años" cubran múltiples sesiones
            # Solo los temas que ya superaron el umbral global pueden ser "representantes de año"
            seen_years = {}
            diverse_topics = []
            for t in target_topics:
                # Extraer año del nombre del fichero fuente (ej: "28-09-2017_Ordinaria...")
                year = ""
                src_bn = t.get("basename", "")
                year_match = re.search(r'(\d{4})', src_bn)
                if year_match:
                    year = year_match.group(1)
                
                if year not in seen_years:
                    seen_years[year] = t['score']
                    diverse_topics.append(t)
                elif t['score'] >= seen_years[year] * 0.9:
                    # Segundo tema del mismo año solo si es casi tan bueno
                    diverse_topics.append(t)
            
            # Para preguntas multi-año, subimos el límite a 15 temas
            # El filtro de 34.000 chars en retrieve_context impide que el contexto se desborde
            target_topics = diverse_topics[:15]

        docs, final_seen = [], set()
        if not target_topics: return initial_docs[:15], []

        print(f"[*] Temas priorizados: {[t['topic'][:50] for t in target_topics]}")
        for t_info in target_topics:
            t_name, src = t_info['topic'], t_info['source']
            t_prefix = f"ASUNTO: {t_name[:40]}"
            try:
                res = self.vector_store.get(where={"source": src})
                pairs = sorted(zip(res['metadatas'], res['documents']), key=lambda x: x[0].get('chunk_index', 0))
                
                # ESTRATEGIA DE BYPASS DE ÍNDICE: Buscamos el tema pero saltamos los primeros fragmentos
                # El índice suele estar en los primeros 100 fragmentos. El debate real mucho después.
                topic_indices = [i for i, (m, d) in enumerate(pairs) 
                                 if (m.get('topic') == t_name or d.lstrip().startswith(t_prefix))]
                
                if topic_indices:
                    # El tema suele aparecer DOS veces en el acta:
                    #   1) en el ORDEN DEL DÍA al principio -> una racha de fragmentos cortos (solo títulos)
                    #   2) en el DEBATE real mucho después -> una racha de fragmentos largos (cuerpo + votación)
                    # Agrupamos los índices en rachas contiguas y nos quedamos con la que
                    # tiene MÁS texto: ese es el debate real, no la entrada del índice.
                    runs = []
                    current_run = [topic_indices[0]]
                    for idx in topic_indices[1:]:
                        if idx == current_run[-1] + 1:
                            current_run.append(idx)
                        else:
                            runs.append(current_run)
                            current_run = [idx]
                    runs.append(current_run)

                    best_run = max(runs, key=lambda run: sum(len(pairs[i][1]) for i in run))
                    selected = [pairs[i] for i in best_run]

                    # Filtro de seguridad: descartar fragmentos de otros temas
                    _num = t_name.split('.')[0].strip()
                    t_topic_number = (_num + ".") if _num.isdigit() else t_name[:80]
                    seen_content = set()
                    for m, d in selected:
                        topic_val = m.get('topic', '')
                        if topic_val and not topic_val.startswith(t_topic_number):
                            continue
                            
                        # Usamos los ÚLTIMOS 150 caracteres para la deduplicación
                        # porque los primeros siempre son "ASUNTO: [título largo]" y colisionan
                        content_key = d[-150:].strip() if len(d) > 150 else d.strip()
                        if content_key not in seen_content:
                            seen_content.add(content_key)
                            key = (src, m.get('chunk_index', d[:50]))
                            if key not in final_seen:
                                final_seen.add(key)
                                docs.append(Document(page_content=d, metadata=m))
            except Exception as e:
                print(f"[!] Error en expansión: {e}")
        
        return docs, target_topics



    # reordena los candidatos por relevancia con Cohere (ver backend/providers.py)
    def _rerank_with_cohere(self, docs: List[Document], question: str, top_n: int = 30) -> List[Document]:
        return _cohere_rerank(docs, question, top_n)

    # formatea los documentos agrupando por (fecha, tema) para compactar el contexto
    def _format_context(self, docs: List[Document]) -> str:
        # Agrupar por (fecha, topic) para unir el inicio y el final del debate
        groups = OrderedDict()
        for d in docs:
            date = d.metadata.get("date", "Fecha desconocida")
            topic = d.metadata.get("topic", "General")
            source = os.path.basename(d.metadata.get("source", "Acta"))
            key = (date, topic, source)
            
            content = d.page_content.strip()
            content = content.replace("Udalbatzako Idazkaritza Nagusia", "")
            content = content.replace("Udalbatzarreko Idazkaritza Nagusia", "")
            content = content.replace("Secretar\u00eda General del Pleno", "")
            
            # Quitar las etiquetas ASUNTO:/ORADOR: SIN borrar el cuerpo de la propuesta.
            # OJO: en las actas modernas el chunk entero va en UNA sola línea, así que
            # el antiguo split('\n') + quitar la primera línea borraba toda la propuesta.
            lines = content.split('\n')
            if len(lines) >= 3:
                # Formato antiguo: cabeceras ASUNTO:/ORADOR: en líneas propias.
                if lines[0].startswith('ASUNTO:'):
                    lines = lines[1:]
                if lines and lines[0].startswith('ORADOR:'):
                    lines = lines[1:]
                content = '\n'.join(lines).strip()
            else:
                # Formato moderno (todo en una línea): quitamos solo las ETIQUETAS
                # con regex y conservamos el cuerpo de la propuesta.
                content = re.sub(r'^ASUNTO:\s*', '', content)
                content = re.sub(r'\bORADOR:\s*[^(]*\([^)]*\)\s*', ' ', content)
                content = re.sub(r'\s{2,}', ' ', content).strip()
            
            if key not in groups:
                groups[key] = {"contents": [], "vote_result": None}
            groups[key]["contents"].append(content)
            if not groups[key]["vote_result"] and d.metadata.get("vote_result"):
                groups[key]["vote_result"] = d.metadata["vote_result"]

        final_text = ""
        for (date, topic, source), group_data in groups.items():
            contents = group_data["contents"]
            resultado = group_data["vote_result"]  # extraído del segmento limpio, sin solapamiento

            raw = "\n---\n".join(c for c in contents if c.strip())
            raw_flat = re.sub(r'\s+', ' ', raw)
            
            # Construir el cuerpo: cabecera (texto dispositivo de la propuesta) + primer
            # tramo del DEBATE real. Las actas repiten muchas veces el texto dispositivo
            # antes de las intervenciones, así que saltamos a la primera intervención de
            # un concejal (SR./SRA. NOMBRE) para que los argumentos entren en el contexto
            # sin disparar el nº de tokens. Si no hay intervención localizable, recortamos normal.
            head = raw_flat[:1600]
            m_int = re.search(r'SR[A]?\.\s+[A-ZÁÉÍÓÚÑ]{2,}', raw_flat)
            if m_int and m_int.start() > 1600:
                raw = head + " [...debate...] " + raw_flat[m_int.start():m_int.start() + 2200]
            elif len(raw) > 3800:
                raw = raw[:3800] + "..."

            short_topic = topic[:120]
            block = f"[PLENO: {date}] {short_topic}\n{raw}\n"
            if resultado:
                block += f"RESULTADO: {resultado}\n"
            block += "\n"
            final_text += block
            
        return final_text.strip()

    # construye los mensajes del prompt de respuesta; punto único compartido por CLI y frontend
    def build_answer_prompt(self, ctx: dict) -> List[Any]:
        from langchain_core.messages import HumanMessage
        formatted_context = ctx["context"]
        question = ctx["question"]
        is_multi_session = ctx["is_multi_session"]
        unique_dates = ctx["unique_dates"]

        if is_multi_session:
            # unique_dates ya viene ordenado cronológicamente desde retrieve_context()
            # (NO usar sorted() sobre strings DD-MM-YYYY: ordena por día, no por fecha real).
            dates_found = ', '.join(unique_dates)
            text = _PROMPT_MULTI_SESSION.format(
                dates_found=dates_found,
                context=formatted_context,
                question=question,
            )
        else:
            text = _PROMPT_SINGLE_SESSION.format(
                context=formatted_context,
                question=question,
            )
        return [HumanMessage(content=text)]


    # --- Pipeline de recuperación compartido (usado por retrieve_context y query) ---

    # multiQuery: genera variantes de búsqueda para ampliar el recall (más la original)
    def _query_variations(self, question: str) -> List[str]:
        try:
            vars_txt = self.invoke_llm(MULTIQUERY_PROMPT.format(question=question)).content
            variations = [v.strip() for v in vars_txt.split('\n') if v.strip()] + [question]
        except Exception as e:
            print(f"[!] MultiQuery falló, usando pregunta original: {e}", flush=True)
            variations = [question]
        return variations[:6]

    # búsqueda semántica inicial sobre todas las variantes (con o sin filtro de fecha)
    def _initial_search(self, variations: List[str], valid_dates: list,
                        exact_date: bool, k: int) -> List[Document]:
        docs: List[Document] = []
        for v in variations:
            targets = valid_dates if valid_dates else [None]
            for d_val in targets:
                flt = {"date": {"$eq": d_val}} if d_val else None
                try:
                    if exact_date:
                        # Fecha única: tomar todo el pleno sin filtrar por score
                        docs.extend(self.vector_store.similarity_search(v, k=k, filter=flt))
                    else:
                        res = self.vector_store.similarity_search_with_score(v, k=k, filter=flt)
                        docs.extend([d for d, s in res if s < SIMILARITY_DISTANCE_MAX])
                except Exception as e:
                    print(f"[!] ChromaDB búsqueda fallida: {type(e).__name__}: {e}", flush=True)
        return docs

    # búsqueda LITERAL complementaria a la semántica
    def _keyword_search(self, question: str, valid_dates: list, k_per_kw: int = 15) -> List[Document]:
        keywords = _extraer_keywords_literales(question)
        if not keywords:
            return []
        # Las 3 más largas: más distintivas, menos probabilidad de aparecer por
        # casualidad dentro de un fragmento no relacionado con la pregunta.
        keywords = sorted(set(keywords), key=len, reverse=True)[:3]

        # Filtro de fecha en UNA sola llamada por keyword (con $in si hay varias
        # fechas válidas), en vez de una llamada por cada (keyword, fecha) — con
        # un filtro de año completo (~12 actas) el bucle por fecha tardaba ~7s;
        # con $in tarda igual que con una fecha sola.
        if len(valid_dates) == 1:
            where = {"date": {"$eq": valid_dates[0]}}
        elif valid_dates:
            where = {"date": {"$in": valid_dates}}
        else:
            where = None

        docs: List[Document] = []
        for kw in keywords:
            # Variante SIN tilde como respaldo: cubre al usuario que escribe sin
            # acentos y a actas antiguas cuyo texto perdió la tilde por OCR. No
            # sustituye a la variante con tilde, que cubre la mayoría del corpus.
            variantes = {kw}
            sin_tilde = strip_accents(kw)
            if sin_tilde != kw:
                variantes.add(sin_tilde)
            # Variante SINGULAR: la pregunta suele ir en plural ("bibliotecas")
            # y el corpus en singular ("biblioteca"); `$contains` no salva ese
            # salto. Buscar la raíz sin la marca de plural lo cubre en los dos
            # sentidos. Cohere descarta luego el ruido de una raíz demasiado laxa.
            for base in (kw, sin_tilde):
                if base.endswith("es") and len(base) >= 7:
                    variantes.add(strip_accents(base[:-2]))
                elif base.endswith("s") and len(base) >= 6:
                    variantes.add(strip_accents(base[:-1]))
            # Variante CAPITALIZADA: `$contains` distingue mayúsculas y las
            # keywords van en minúscula, pero los nombres propios ("Tubacex",
            # "Zorrotzaurre") van en mayúscula en las actas. Sin esto una empresa
            # con una sola mención se pierde.
            for v in list(variantes):
                variantes.add(v.capitalize())
                variantes.add(v.title())
            for variante in variantes:
                try:
                    res = self.vector_store.get(
                        where=where, where_document={"$contains": variante},
                        limit=k_per_kw, include=["metadatas", "documents"],
                    )
                    for content, meta in zip(res["documents"], res["metadatas"]):
                        m = dict(meta)
                        m["_kw"] = kw  # marca de canal literal (ver _rerank_with_cohere)
                        docs.append(Document(page_content=content, metadata=m))
                except Exception as e:
                    print(f"[!] Búsqueda literal fallida para '{variante}': {type(e).__name__}: {e}", flush=True)
        return docs

    # busca proposiciones clasificadas con el tema de la pregunta aunque el texto no repita la palabra (ver memoria 1.1)
    def _thematic_search(self, question: str, valid_dates: list,
                         max_proposals: int = 40, pool_limit: int = 20000) -> List[Document]:
        rollup = _load_tema_rollup()
        if not rollup["detect"]:
            return []
        q_norm = strip_accents(question.lower())
        matched_label = None
        for label_norm, label_display in rollup["detect"]:
            if re.search(rf'\b{re.escape(label_norm)}\b', q_norm):
                matched_label = label_display
                break
        if not matched_label:
            return []

        # Filtro por la bandera tf_<slug>, que cubre tanto el tema_principal
        # como los temas secundarios rodados a nivel 1 (una proposición con
        # tema_principal="movilidad" y un secundario "calidad del aire" aparece
        # al preguntar por medio ambiente). Fallback al filtro por
        # tema_principal + subtemas si el chunk aún no tiene banderas.
        slug = rollup.get("parent_slug", {}).get(matched_label)
        filter_values = rollup["filter_values"].get(matched_label, [matched_label])
        tema_clause = (
            {"$or": [{f"tf_{slug}": True}, {"tema_principal": {"$in": filter_values}}]}
            if slug else {"tema_principal": {"$in": filter_values}}
        )
        clauses = [tema_clause]
        if len(valid_dates) == 1:
            clauses.append({"date": {"$eq": valid_dates[0]}})
        elif valid_dates:
            clauses.append({"date": {"$in": valid_dates}})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}

        # Fetch SOLO metadata (sin documents): un tema grande puede tener
        # >10.000 chunks y traer el texto de todos para descartar casi todos
        # después es lento. pool_limit alto (20.000): con un límite bajo, los
        # temas grandes (movilidad, presupuestos...) se truncaban antes del
        # muestreo y se perdían silenciosamente los años más recientes.
        try:
            res = self.vector_store.get(where=where, limit=pool_limit, include=["metadatas"])
        except Exception as e:
            print(f"[!] Búsqueda temática fallida: {type(e).__name__}: {e}", flush=True)
            return []

        # Se agrupa por prop_id y se muestrea repartido cronológicamente para
        # cubrir MUCHAS proposiciones distintas sobre profundizar en pocas.
        by_prop: Dict[str, List[Tuple[str, int]]] = {}
        prop_date: Dict[str, str] = {}
        for cid, meta in zip(res["ids"], res["metadatas"]):
            pid = meta.get("prop_id")
            if not pid:
                continue
            by_prop.setdefault(pid, []).append((cid, meta.get("chunk_index", 0)))
            prop_date.setdefault(pid, meta.get("date", ""))

        pids = sorted(by_prop, key=lambda p: _date_str_sort_key(prop_date.get(p, "")))
        if len(pids) > max_proposals:
            # Paso normalizado por (n-1)/(k-1): así el primer y el último
            # elemento siempre entran (con n/k la proposición más reciente
            # quedaba siempre fuera).
            step = (len(pids) - 1) / (max_proposals - 1)
            pids = [pids[round(i * step)] for i in range(max_proposals)]

        print(f"[*] Tema detectado en la pregunta: '{matched_label}' "
              f"-> {len(by_prop)} proposiciones distintas (muestra de {len(pids)})")
        if not pids:
            return []

        # Segundo fetch CON texto: de cada proposición se coge el chunk del
        # cuerpo MÁS LARGO (descartando el primero y los 2 últimos, que suelen
        # ser el orden del día y el recuento de votos, que Cohere puntúa ~0).
        ids_cuerpo = []
        for p in pids:
            orden = [cid for cid, _ in sorted(by_prop[p], key=lambda ci: ci[1])]
            ids_cuerpo += (orden[1:-2] or orden)[:8]
        res2 = self.vector_store.get(ids=ids_cuerpo, include=["metadatas", "documents"])
        mejor: Dict[str, Tuple[str, dict, int]] = {}
        for c, m in zip(res2["documents"], res2["metadatas"]):
            pid = m.get("prop_id")
            if not pid:
                continue
            cuerpo = re.sub(r"votos\s+emitidos.*", "", c, flags=re.I | re.S)
            n = len(cuerpo.strip())
            if pid not in mejor or n > mejor[pid][2]:
                mejor[pid] = (c, m, n)
        return [Document(page_content=c, metadata=m) for c, m, _ in mejor.values()]

    # si la pregunta menciona un grupo, conserva solo los chunks de ese grupo (con margen si quedan pocos)
    def _apply_party_filter(self, docs: List[Document], question: str,
                            min_docs: int = 3) -> List[Document]:
        target_party = self._detect_party_in_question(question)
        if not target_party:
            return docs
        # metadata['party'] ya está normalizado: se compara canónico contra
        # canónico, no como substring ("PP" no está literal en "Grupo Municipal
        # Partido Popular").
        def es_proponente(d: Document) -> bool:
            # grupo_proponente se calcula en la indexación con más contexto;
            # solo se recalcula al vuelo (sobre metadata['topic']) si falta.
            grupo_meta = d.metadata.get("grupo_proponente")
            if grupo_meta:
                return grupo_meta == target_party
            return _extrae_grupo_partido(d.metadata.get("topic", "")) == target_party

        party_filtered = [
            d for d in docs
            if target_party.lower() in d.metadata.get("party", "").lower()
            or es_proponente(d)
        ]
        if len(party_filtered) >= min_docs:
            print(f"[*] Filtro de grupo '{target_party}': {len(party_filtered)} docs")
            return party_filtered
        return docs

    # elimina duplicados por (source, chunk_index) conservando el orden de entrada
    def _dedup_docs(self, docs: List[Document], limit: Optional[int] = None) -> List[Document]:
        seen, out = set(), []
        for d in docs:
            key = (d.metadata.get('source', ''), d.metadata.get('chunk_index', d.page_content[:50]))
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out[:limit] if limit else out

    # elimina docs cuyo topic no esté entre los temas priorizados (si el filtro no vacía todo)
    def _filter_by_target_topics(self, docs: List[Document], target_topics: list) -> List[Document]:
        if not target_topics:
            return docs
        valid_prefixes = []
        for t in target_topics:
            tp = t['topic']
            num = tp.split('.')[0].strip()
            valid_prefixes.append((num + ".") if num.isdigit() else tp[:80])
        filtered = [d for d in docs if any(d.metadata.get('topic', '').startswith(p) for p in valid_prefixes)]
        return filtered if filtered else docs

    # multiQuery -> búsqueda híbrida (semántica+literal+temática) -> filtro de grupo -> rerank Cohere
    def _retrieve_and_rank(self, question: str, k: int) -> Tuple[List[Document], bool]:
        variations = self._query_variations(question)
        _, valid_dates = self._get_temporal_filter(question)
        exact_date = len(valid_dates) == 1
        docs = self._initial_search(variations, valid_dates, exact_date, k)
        # Búsqueda híbrida: semántica + literal + temática. Cada canal aporta
        # máx ~22 docs (Cohere solo rerankea los primeros ~64 por orden de
        # entrada). Orden: literal -> temático -> semántico.
        docs = self._dedup_docs(
            self._keyword_search(question, valid_dates)[:22]
            + self._thematic_search(question, valid_dates)[:22]
            + docs
        )
        docs = self._apply_party_filter(docs, question)
        if COHERE_API_KEY:
            docs = self._rerank_with_cohere(docs, question, top_n=50)
        return docs, exact_date

    # conjunto final de docs: sin expandir si es un pleno concreto, aspiradora + rescate si no (ver memoria 1.5)
    def _select_final_docs(self, candidates: List[Document], question: str,
                           exact_date: bool) -> List[Document]:
        if exact_date:
            docs = self._dedup_docs(candidates, limit=40)
        else:
            docs, target_topics = self._expand_context_by_topic(candidates, question)
            docs = self._filter_by_target_topics(docs, target_topics)
            # RESCATE: la aspiradora descarta chunks de "General / Introducción"
            # o de "dación de cuenta" aunque Cohere los puntúe alto (ahí vive
            # parte del contenido de preguntas generales). Los 8 mejores de
            # Cohere sobreviven siempre y van los PRIMEROS del contexto.
            ranked = [d for d in candidates if "_rerank_score" in d.metadata]
            # Los chunks del canal LITERAL (mencionan textualmente un término de
            # la pregunta) tampoco los puede tirar el filtro de tema aunque
            # Cohere no los vea.
            kw_hits = [d for d in candidates if d.metadata.get("_kw")][:10]
            lead = self._dedup_docs(ranked[:8] + kw_hits)
            lead_keys = {self._doc_key(d) for d in lead}
            tail = sorted(
                (d for d in self._dedup_docs(docs) if self._doc_key(d) not in lead_keys),
                key=_date_sort_key,
            )
            # Re-aplicar filtro de partido sobre el corpus completo. Umbral
            # min_docs=1: los docs ya son del tema correcto, puede ser agresivo.
            return self._apply_party_filter(lead + tail, question, min_docs=1)
        return sorted(docs, key=_date_sort_key)

    @staticmethod
    def _doc_key(d: Document) -> tuple:
        return (d.metadata.get("source", ""), d.metadata.get("chunk_index", d.page_content[:50]))

    # vuelca el contexto enviado al LLM para inspección técnica
    def _dump_debug_context(self, formatted_context: str) -> None:
        with _DEBUG_CONTEXT_LOCK:
            with open(_DEBUG_CONTEXT_PATH, "w", encoding="utf-8") as f:
                f.write(formatted_context)

    # --- Punto de Entrada Principal ---

    # fase de recuperación: devuelve el contexto y el prompt listos para el LLM
    def retrieve_context(self, question: str) -> dict:
        if not self.vector_store: self.create_vector_store()

        all_initial_docs, exact_date = self._retrieve_and_rank(question, k=80)

        # Umbral de relevancia: si la pregunta no menciona una fecha y ni el
        # mejor fragmento llega al mínimo, no hay nada que responder (evita
        # inventar fuentes con preguntas ajenas a las actas). Solo se aplica si
        # Cohere asignó scores.
        if COHERE_API_KEY and not exact_date and all_initial_docs:
            rerank_scores = [d.metadata["_rerank_score"] for d in all_initial_docs if "_rerank_score" in d.metadata]
            max_score = max(rerank_scores) if rerank_scores else None
            if max_score is not None and max_score < RELEVANCE_FLOOR:
                print(f"[*] Relevancia máxima {max_score:.4f} < {RELEVANCE_FLOOR}: sin resultados.")
                return {
                    "context": "", "is_multi_session": False,
                    "unique_dates": [], "docs": [], "question": question,
                }

        docs = self._select_final_docs(all_initial_docs, question, exact_date)
        self.last_retrieved_docs = docs

        formatted_context = self._format_context(docs)
        max_ctx = self._context_char_limit()
        if len(formatted_context) > max_ctx:
            formatted_context = formatted_context[:max_ctx] + "\n\n[...CONTEXTO TRUNCADO POR TAMAÑO...]"

        self._dump_debug_context(formatted_context)

        unique_dates = sorted(
            set(d.metadata.get("date", "") for d in docs if d.metadata.get("date")),
            key=_date_str_sort_key,
        )
        is_multi_session = len(unique_dates) > 1

        return {
            "context": formatted_context,
            "is_multi_session": is_multi_session,
            "unique_dates": unique_dates,
            "docs": docs,
            "question": question,
        }

    # flujo principal de la CLI: Recuperación, Expansión y Generación con fuentes
    def query(self, question: str) -> str:
        if not self.vector_store: self.create_vector_store()

        # 1-2. Recuperación compartida (MultiQuery, búsqueda, filtro de grupo, rerank)
        all_initial_docs, exact_date = self._retrieve_and_rank(question, k=50)

        # 3. Expansión/selección y ordenación cronológica (más antiguo primero)
        docs = self._select_final_docs(all_initial_docs, question, exact_date)
        self.last_retrieved_docs = docs

        formatted_context = self._format_context(docs)

        max_ctx = self._context_char_limit()
        if len(formatted_context) > max_ctx:
            formatted_context = formatted_context[:max_ctx] + "\n\n[...CONTEXTO TRUNCADO POR TAMAÑO...]"

        # Detectar si la pregunta es general (varios años/sesiones) o específica (un pleno concreto)
        unique_dates = sorted(
            set(d.metadata.get("date", "") for d in docs if d.metadata.get("date")),
            key=_date_str_sort_key,
        )
        is_multi_session = len(unique_dates) > 1

        self._dump_debug_context(formatted_context)

        # Guarda contra alucinaciones: si el contexto está vacío no se invoca el LLM
        if not formatted_context.strip():
            return ("Lo siento, no he encontrado fragmentos relevantes en las actas para responder a tu pregunta. "
                    "Puede que el tema no esté cubierto en los documentos indexados, o que el modelo de embeddings "
                    "no haya podido conectarse. Prueba a reformular la pregunta.")

        # Prompt canónico compartido con Chainlit (build_answer_prompt).
        prompt_msgs = self.build_answer_prompt({
            "context": formatted_context,
            "question": question,
            "is_multi_session": is_multi_session,
            "unique_dates": unique_dates,
        })
        print(f"[*] Generando crónica detallada...")

        # Reintento automático: en máquinas con poca RAM ollama puede tardar en recargar
        # el modelo LLM después de las llamadas de embedding, rechazando la conexión
        # durante ese breve intervalo. Reintentos con backoff corto solucionan el problema.
        response = None
        for attempt in range(3):
            try:
                resp = self.invoke_llm(prompt_msgs)
                response = resp.content if hasattr(resp, "content") else str(resp)
                break
            except Exception as e:
                if attempt < 2 and any(msg in str(e) for msg in ("Connection refused", "RemoteProtocolError", "Server disconnected", "ConnectError")):
                    print(f"[!] LLM cargando modelo, reintentando ({attempt+2}/3)...", flush=True)
                    time.sleep(10)
                else:
                    raise
        
        # 5. Añadir fuentes enriquecidas (deduplicadas por acta+tema)
        sources = "\n\n" + "="*60 + "\nFUENTES UTILIZADAS:\n"
        seen_src_keys: set = set()
        src_idx = 1
        for d in docs:
            src = os.path.basename(d.metadata.get("source", "Acta"))
            top = d.metadata.get("topic", "")
            key = (src, top[:60])
            if key in seen_src_keys:
                continue
            seen_src_keys.add(key)
            spk = d.metadata.get("speaker", "")
            snippet = d.page_content.replace('\n', ' ')[:400] + "..."
            sources += f" [{src_idx}] {src} | {spk} | {top[:60]}\n     -> \"{snippet}\"\n\n"
            src_idx += 1
        
        return response + sources

# Singleton compartido para que app.py pueda importarlo sin re-inicializar la BD
_rag_instance: Optional[RAGPipeline] = None

# devuelve la instancia global del RAG (la crea la primera vez que se llama)
def get_rag() -> RAGPipeline:
    global _rag_instance
    with _RAG_SINGLETON_LOCK:
        if _rag_instance is None:
            _rag_instance = RAGPipeline()
            _rag_instance.vector_store = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=_rag_instance.embeddings
            )
            # Precalentar el LLM para que el primer query no espere la carga del modelo
            try:
                _rag_instance.invoke_llm("ok")
            except Exception:
                pass
    return _rag_instance

def main():
    parser = argparse.ArgumentParser(description="RAG System for Bilbao Actas")
    parser.add_argument("--query", type=str, help="Pregunta directa")
    args = parser.parse_args()

    rag = RAGPipeline()
    if not os.path.exists(CHROMA_PATH):
        rag.create_vector_store()
    else:
        rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)

    # Precalentar el modelo LLM antes de cualquier query: fuerza la carga de
    # pesos ahora, con reintentos, para que la primera llamada real (MultiQuery)
    # no espere la carga ni falle por RAM dejando el pool HTTP de httpx en mal estado.
    print("[*] Cargando modelo LLM...")
    for attempt in range(3):
        try:
            rag.llm.invoke("ok")
            print("[+] Modelo listo.")
            break
        except Exception as e:
            if attempt < 2:
                print(f"[!] Modelo cargando, reintentando ({attempt+2}/3)...")
                time.sleep(10)

    if args.query:
        print("\nRESPUESTA:")
        res = rag.query(args.query)
        try:
            print(res)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(res.encode('utf-8'))
    else:
        print(f"\n--- RAG Bilbao Ready [Model: {LLM_MODEL_GROQ if GROQ_API_KEY else LLM_MODEL_LOCAL}] ---")
        while True:
            try:
                q = input("\nPregunta: ")
                if not q.strip(): continue
                if q.lower() in ['exit', 'quit', 'salir']: break
                
                res = rag.query(q)
                try:
                    print(res)
                except UnicodeEncodeError:
                    sys.stdout.buffer.write(res.encode('utf-8'))
                    
            except (KeyboardInterrupt, EOFError): break
            except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    main()
