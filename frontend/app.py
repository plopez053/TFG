"""
Interfaz Chainlit del chatbot de las actas del Pleno de Bilbao.

  1. Arranque: parche de sniffio, ruta /acta para servir los PDF y precarga de los motores
  2. Configuración del chat: perfiles, preguntas de ejemplo e inicio de sesión
  3. Respuesta a cada mensaje, según el perfil: RAG vectorial o GraphRAG
"""
import sys
import os
import asyncio
import re

# Python 3.14 + sniffio: current_task() devuelve None en algunos contextos ASGI
# aunque haya un bucle en marcha, y anyio lanza NoEventLoopError
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

# raíz del proyecto en el path para importar comun/, vectorial/ y grafo/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chainlit as cl  # noqa: E402
from chainlit.server import app as _fastapi_app  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from comun.rutas import DATA_PATH  # noqa: E402
from vectorial.generacion import componer_respuesta  # noqa: E402
from vectorial.pipeline import get_rag  # noqa: E402
from grafo.consulta.recursos import _load_graph as _load_rdf_graph  # noqa: E402
from grafo.consulta.respuesta import graph_answer as _graph_answer  # noqa: E402

PERFIL_VECTORIAL = "RAG Vectorial"
PERFIL_GRAPHRAG = "GraphRAG (SPARQL)"


# =============================================================================
# 1. Arranque
# =============================================================================

# Sirve los PDF de las actas para enlazarlos con un hipervínculo normal (se
# abren en otra pestaña y saltan a la página con #page=N). El nombre se sanea
# para impedir path traversal (../).
@_fastapi_app.get("/acta/{year}/{filename}")
async def servir_acta(year: str, filename: str):
    filename = os.path.basename(filename)
    if not re.match(r"^\d{4}$", year):
        raise HTTPException(status_code=400, detail="Año no válido")
    path = os.path.join(DATA_PATH, year, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Acta no encontrada")
    return FileResponse(path, media_type="application/pdf")


# Chainlit registra antes una ruta comodín que sirve la aplicación para
# cualquier URL; se pone /acta la primera para que tenga prioridad.
_ruta_acta = _fastapi_app.router.routes.pop()
_fastapi_app.router.routes.insert(0, _ruta_acta)

print("[*] Pre-cargando el motor RAG vectorial...")
_rag = get_rag()
print("[+] Motor RAG vectorial listo.")

print("[*] Pre-cargando el grafo RDF (GraphRAG)...")
_load_rdf_graph()
print("[+] Grafo RDF listo.")


# =============================================================================
# 2. Configuración del chat
# =============================================================================

@cl.set_chat_profiles
async def set_chat_profiles():
    return [
        cl.ChatProfile(
            name=PERFIL_VECTORIAL,
            markdown_description=(
                "Busca en el texto de las actas.\n\n"
                "Mejor para preguntas abiertas: qué se debatió, propuestas y argumentos."
            ),
        ),
        cl.ChatProfile(
            name=PERFIL_GRAPHRAG,
            markdown_description=(
                "Consulta el grafo de proposiciones.\n\n"
                "Mejor para números: cuántas proposiciones, rankings por grupo, tema o año."
            ),
        ),
    ]


# preguntas de ejemplo al abrir el chat, distintas para cada perfil
@cl.set_starters
async def set_starters(chat_profile: str):
    if chat_profile == PERFIL_GRAPHRAG:
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


@cl.on_chat_start
async def on_chat_start():
    profile = cl.user_session.get("chat_profile")
    cl.user_session.set("rag", _rag)
    modo = "graphrag" if profile == PERFIL_GRAPHRAG else "vectorial"
    cl.user_session.set("mode", modo)


# =============================================================================
# 3. Respuesta a cada mensaje
# =============================================================================

@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    if not question:
        return
    if cl.user_session.get("mode") == "graphrag":
        await responder_graphrag(question)
    else:
        await responder_vectorial(question, cl.user_session.get("rag"))


# RAG vectorial: recuperación -> LLM -> fuentes bajo cada bloque
async def responder_vectorial(question: str, rag):
    async with cl.Step(name="Buscando en las actas") as step:
        ctx = await asyncio.to_thread(rag.retrieve_context, question)
        step.output = "Búsqueda completada."

    if not ctx["context"].strip():
        await cl.Message(
            content="Lo siento, no he encontrado información relevante en las actas para esta pregunta. Prueba a reformularla."
        ).send()
        return

    async with cl.Step(name="Redactando la crónica"):
        respuesta = await rag.ainvoke_llm(rag.build_answer_prompt(ctx))
        answer_text = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

    async with cl.Step(name="Localizando fuentes en los PDFs"):
        answer = await asyncio.to_thread(componer_respuesta, answer_text, ctx)

    await cl.Message(content=answer).send()


# GraphRAG: el LLM genera una consulta SPARQL, se ejecuta sobre el grafo y se narra
async def responder_graphrag(question: str):
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

    # la consulta SPARQL exacta, en el panel lateral
    sparql_element = cl.Text(
        name="Consulta SPARQL generada",
        content=f"```sparql\n{sparql_txt}\n```\n*{n} filas devueltas*",
        display="side",
    )
    await cl.Message(content=result.get("answer", "(sin respuesta)"), elements=[sparql_element]).send()

