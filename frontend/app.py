import sys
import os
import asyncio
import re
import pypdf

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
from backend.rag import get_rag, DATA_PATH

# ---------------------------------------------------------------------------
# Pre-carga del RAG al importar el módulo (evita problemas de threading con ChromaDB)
# ---------------------------------------------------------------------------
print("[*] Pre-cargando el motor RAG...")
_rag = get_rag()
print("[+] Motor RAG listo.")

# Caché para evitar volver a leer/parsear los mismos PDFs en una sesión
_pdf_cache = {}

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

def find_page_in_pdf(pdf_path: str, chunk_content: str) -> int:
    """
    Encuentra la página (1-indexed) de un fragmento de texto dentro del PDF.
    Usa caché y búsqueda lazy para máxima velocidad.
    """
    # El fragmento del RAG suele empezar por "ASUNTO: ...\nORADOR: ...\n\n"
    parts = chunk_content.split("\n\n", 1)
    core_text = parts[1].strip() if len(parts) > 1 else chunk_content.strip()
    
    def clean_txt(t):
        return re.sub(r'[^a-z0-9]', '', t.lower()).strip()
        
    core_clean = clean_txt(core_text[:100]) # primeros 100 caracteres significativos
    if not core_clean:
        return 1
        
    if pdf_path not in _pdf_cache:
        try:
            reader = pypdf.PdfReader(pdf_path)
            _pdf_cache[pdf_path] = {
                "reader": reader,
                "pages_text": [None] * len(reader.pages)
            }
        except Exception as e:
            print(f"Error cargando PDF {pdf_path}: {e}")
            return 1
            
    cache_entry = _pdf_cache[pdf_path]
    reader = cache_entry["reader"]
    pages_text = cache_entry["pages_text"]
    
    # 1. Comprobar páginas ya leídas/cargadas en caché
    for idx, txt in enumerate(pages_text):
        if txt is not None and core_clean in txt:
            return idx + 1
            
    # 2. Cargar páginas perezosamente (lazy) hasta encontrar la coincidencia
    for idx in range(len(pages_text)):
        if pages_text[idx] is None:
            try:
                page_raw = reader.pages[idx].extract_text() or ""
                pages_text[idx] = clean_txt(page_raw)
                if core_clean in pages_text[idx]:
                    return idx + 1
            except Exception as e:
                pages_text[idx] = ""
                
    return 1


def build_sources_data(retrieved_docs):
    """Construye los datos de las fuentes (una por acta/fecha citada).

    Escanea cada PDF para localizar la página del fragmento. Es la parte lenta,
    por eso se llama desde un hilo. Devuelve una lista de dicts serializables
    (sin objetos de Chainlit) para construir los elementos en el hilo async.
    """
    sources_data = []
    seen_dates = set()
    for doc in retrieved_docs:
        pdf_path = resolve_pdf_path(doc.metadata.get("source", ""))
        if not pdf_path or not os.path.exists(pdf_path):
            continue

        date = doc.metadata.get("date", "Fecha desconocida")
        if date in seen_dates:
            continue
        seen_dates.add(date)

        page_num = find_page_in_pdf(pdf_path, doc.page_content)
        topic = doc.metadata.get("topic", "Tema general")
        short_topic = topic[:80] + "..." if len(topic) > 80 else topic

        sources_data.append({
            "pdf_path": pdf_path,
            "page": page_num,
            "content": doc.page_content,
            "speaker": doc.metadata.get("speaker", "Desconocido"),
            "party": doc.metadata.get("party", "Desconocido"),
            "short_topic": short_topic,
            "pdf_name": f"Ver PDF - Acta {date} (Pág. {page_num})",
            "text_name": f"Comprobar Fuentes - Acta {date} (Pág. {page_num})",
        })
    return sources_data


# ---------------------------------------------------------------------------
# Preguntas de ejemplo que aparecen al abrir el chat (Starters)
# ---------------------------------------------------------------------------
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Tasa turística",
            message="¿Qué debates sobre la tasa turística ha habido en el Pleno de Bilbao a lo largo de los años?",
            icon="https://em-content.zobj.net/source/twitter/376/hotel_1f3e8.png",
        ),
        cl.Starter(
            label="Vivienda social",
            message="¿Qué propuestas sobre vivienda social se han debatido en el Pleno de Bilbao?",
            icon="https://em-content.zobj.net/source/twitter/376/house_1f3e0.png",
        ),
        cl.Starter(
            label="Medio ambiente",
            message="¿Qué acuerdos sobre medio ambiente y sostenibilidad se han tomado en los plenos?",
            icon="https://em-content.zobj.net/source/twitter/376/evergreen-tree_1f332.png",
        ),
        cl.Starter(
            label="Presupuesto 2024",
            message="¿Qué se debatió sobre el presupuesto municipal de Bilbao en 2024?",
            icon="https://em-content.zobj.net/source/twitter/376/euro-banknote_1f4b6.png",
        ),
    ]


# ---------------------------------------------------------------------------
# Inicio de sesión: guarda el RAG ya cargado en la sesión del usuario
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("rag", _rag)


# ---------------------------------------------------------------------------
# Respuesta a cada mensaje del usuario
# ---------------------------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message):
    rag = cl.user_session.get("rag")
    question = message.content.strip()

    if not question:
        return

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
        sys_prompt = f"""Eres el Cronista Oficial de Bilbao, experto en historia municipal.

INSTRUCCION: Se te proporcionan fragmentos de MULTIPLES plenos del Ayuntamiento de Bilbao.
Las fechas de los plenos en este contexto son: {dates_found}
Responde a la pregunta haciendo un RESUMEN CRONOLOGICO de los debates y propuestas encontrados.

REGLAS CRUCIALES:
- USA SOLO la informacion que esta explicitamente en las actas proporcionadas abajo.
- NUNCA inventes fechas, cifras, nombres, resultados o detalles que no esten en el texto.
- Si no sabes el resultado de una votacion, escribe: [Sin resultado en acta]
- Para cada pleno relevante desarrolla un parrafo con este formato:
  **[fecha] — [grupo proponente]**
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
RESUMEN CRONOLOGICO DETALLADO:"""
    else:
        sys_prompt = """Eres el Cronista Oficial de Bilbao. Tu misión es relatar lo ocurrido en el Pleno.

INSTRUCCIÓN: Basándote en el ACTA de abajo, responde a: {question}

REGLAS:
- Empieza directamente con: "En la sesión del Pleno de Bilbao..."
- Detalla los puntos de la propuesta (qué se pide exactamente).
- Indica el resultado final de la votación si consta.

ACTA:
{context}

PREGUNTA: {question}
CRÓNICA:"""

    from langchain_core.prompts import ChatPromptTemplate
    prompt_value = ChatPromptTemplate.from_template(sys_prompt).format_messages(
        context=formatted_context, question=question
    )

    # Fase 2: Generación con streaming — los tokens aparecen en pantalla en tiempo real
    answer_msg = cl.Message(content="")
    async for chunk in rag.llm.astream(prompt_value):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        await answer_msg.stream_token(token)

    # Construir fuentes: el escaneo de los PDFs para localizar la página es lento
    # (cientos de páginas en las actas modernas), así que lo hacemos en un hilo
    # para no bloquear la interfaz mientras se localizan las fuentes.
    elements = []
    sources_markdown = []

    if retrieved_docs:
        async with cl.Step(name="Localizando fuentes en los PDFs"):
            sources_data = await asyncio.to_thread(build_sources_data, retrieved_docs)

        if sources_data:
            sources_markdown.append("\n\n**Fuentes consultadas (haz clic para abrirlas):**")
            for s in sources_data:
                elements.append(cl.Pdf(name=s["pdf_name"], path=s["pdf_path"], page=s["page"], display="page"))
                elements.append(cl.Text(name=s["text_name"], content=s["content"], display="page"))
                sources_markdown.append(
                    f"* {s['pdf_name']} | {s['text_name']}\n"
                    f"  *(Orador: {s['speaker']} ({s['party']}) | Asunto: {s['short_topic']})*"
                )
            answer_msg.content += "\n" + "\n".join(sources_markdown)

    answer_msg.elements = elements
    await answer_msg.send()
