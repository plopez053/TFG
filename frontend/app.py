import sys
import os
import asyncio
import re
import glob
import urllib.parse
from collections import defaultdict

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
from backend.rag import (
    get_rag, DATA_PATH, strip_accents,
    resolve_pdf_path, _palabras_clave, _STOP_PROCEDIMENTAL, build_sources_data,
)
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


# --- Helpers del modo GraphRAG ---

# ruta del PDF del acta de una fecha 'DD-MM-YYYY', o "" si no se encuentra
def _find_pdf_by_date(fecha: str) -> str:
    parts = fecha.split("-")
    if len(parts) != 3:
        return ""
    year = parts[-1]
    patron = os.path.join(DATA_PATH, year, f"{fecha}_*.pdf")
    matches = glob.glob(patron)
    return matches[0] if matches else ""


# extrae fechas únicas de las filas SPARQL y genera links a los PDFs
def _fuentes_graphrag(rows: list) -> str:
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


# genera la SPARQL, la ejecuta y devuelve la respuesta narrada + fuentes (modo GraphRAG)
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
# Diferenciadas por perfil: RAG Vectorial vs GraphRAG (SPARQL)
# ---------------------------------------------------------------------------
@cl.set_starters
async def set_starters(chat_profile: str):
    if chat_profile == "GraphRAG (SPARQL)":
        return [
            cl.Starter(
                label="Ranking por grupo político",
                message="¿Qué grupo político ha presentado más proposiciones en el Pleno de Bilbao?",
            ),
            cl.Starter(
                label="Propuestas aprobadas por año",
                message="¿Cuántas proposiciones se aprobaron cada año entre 2015 y 2024?",
            ),
            cl.Starter(
                label="Temas más debatidos",
                message="¿Cuáles son los 10 temas sobre los que más proposiciones se han presentado en el Pleno?",
            ),
            cl.Starter(
                label="Actividad de un grupo",
                message="¿Cuántas proposiciones presentó EH Bildu en 2023 y cuántas se aprobaron?",
            ),
        ]
    # RAG Vectorial: preguntas abiertas que aprovechan la búsqueda semántica y
    # el resumen cronológico de debates con argumentos y contexto textual.
    return [
        cl.Starter(
            label="Turismo e impacto en la ciudad",
            message="¿Qué debates ha habido en el Pleno de Bilbao sobre el turismo, su regulación y su impacto en la ciudad?",
        ),
        cl.Starter(
            label="Argumentos sobre vivienda",
            message="¿Qué argumentos han dado los distintos grupos políticos en los debates sobre vivienda social?",
        ),
        cl.Starter(
            label="Movilidad sostenible",
            message="¿Qué propuestas y acuerdos sobre movilidad sostenible y transporte público se han debatido en los plenos?",
        ),
        cl.Starter(
            label="Debate presupuesto 2024",
            message="¿Qué se debatió y argumentó en torno al presupuesto municipal de Bilbao en 2024?",
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
    retrieved_docs = ctx["docs"]

    if not formatted_context.strip():
        await cl.Message(
            content="Lo siento, no he encontrado información relevante en las actas para esta pregunta. Prueba a reformularla."
        ).send()
        return

    # Prompt canónico compartido con la CLI (backend/rag.py → build_answer_prompt).
    # La lógica multi-sesión / sesión única y las reglas del cronista
    # se gestionan allí: aquí solo delegamos y obtenemos los mensajes listos.
    prompt_value = rag.build_answer_prompt(ctx)

    # Fase 2: Generación + fuentes en UN SOLO mensaje. Las fuentes son hipervínculos
    # normales a la ruta /acta/... (abren el PDF en una pestaña del navegador en la
    # página correcta), en lugar de elementos cl.Pdf "side" que se abrían solos.
    async with cl.Step(name="Redactando la crónica"):
        respuesta = await rag.ainvoke_llm(prompt_value)
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
