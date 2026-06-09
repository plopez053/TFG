import sys
import os
import asyncio
import re
import pypdf

# Agregar la raíz del proyecto al PYTHONPATH para poder importar backend.rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chainlit as cl
from backend.rag import get_rag

# ---------------------------------------------------------------------------
# Pre-carga del RAG al importar el módulo (evita problemas de threading con ChromaDB)
# ---------------------------------------------------------------------------
print("[*] Pre-cargando el motor RAG...")
_rag = get_rag()
print("[+] Motor RAG listo.")

# Caché para evitar volver a leer/parsear los mismos PDFs en una sesión
_pdf_cache = {}

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

    # Ejecutamos el RAG en un hilo separado para no bloquear el bucle async
    async with cl.Step(name="Buscando en las actas") as step:
        full_response = await asyncio.to_thread(rag.query, question)
        retrieved_docs = getattr(rag, "last_retrieved_docs", [])
        step.output = "Busqueda completada."

    # Separamos la respuesta principal de las fuentes
    separator = "=" * 60
    if separator in full_response:
        parts = full_response.split(separator, 1)
        answer_text = parts[0].strip()
    else:
        answer_text = full_response.strip()

    # Construimos los elementos de fuentes (PDFs y Textos)
    elements = []
    sources_markdown = []

    if retrieved_docs:
        sources_markdown.append("\n\n**Fuentes consultadas (haz clic para abrirlas):**")
        
        seen_sources = set()
        count = 1
        
        for doc in retrieved_docs:
            pdf_path = doc.metadata.get("source")
            if not pdf_path or not os.path.exists(pdf_path):
                continue
                
            file_name = os.path.basename(pdf_path)
            date = doc.metadata.get("date", "Fecha desconocida")
            
            # Encontrar el número de página dinámicamente usando pypdf
            page_num = find_page_in_pdf(pdf_path, doc.page_content)
            
            # Identificador único para evitar duplicar la misma página de la misma acta
            source_key = (pdf_path, page_num)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            
            speaker = doc.metadata.get("speaker", "Desconocido")
            party = doc.metadata.get("party", "Desconocido")
            topic = doc.metadata.get("topic", "Tema general")
            
            # Acortar temas para mejor presentación
            short_topic = topic[:80] + "..." if len(topic) > 80 else topic
            
            # Nombres únicos para los botones de Chainlit
            pdf_element_name = f"Ver PDF - Acta {date} (Pág. {page_num})"
            text_element_name = f"Comprobar Fuentes - Acta {date} (Pág. {page_num})"
            
            # Elemento PDF (abre la página exacta en el lateral)
            elements.append(
                cl.Pdf(
                    name=pdf_element_name,
                    path=pdf_path,
                    page=page_num,
                    display="page"
                )
            )
            
            # Elemento de texto (muestra el fragmento exacto en el lateral)
            elements.append(
                cl.Text(
                    name=text_element_name,
                    content=doc.page_content,
                    display="page"
                )
            )
            
            # Generar fila de la lista de fuentes
            sources_markdown.append(
                f"* {pdf_element_name} | {text_element_name}\n"
                f"  *(Orador: {speaker} ({party}) | Asunto: {short_topic})*"
            )
            
            count += 1
            if count > 6:  # Mostrar como máximo 6 fuentes para no sobrecargar
                break
                
        if len(sources_markdown) > 1:
            answer_text += "\n" + "\n".join(sources_markdown)

    await cl.Message(
        content=answer_text,
        elements=elements,
    ).send()
