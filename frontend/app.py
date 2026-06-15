import sys
import os
import asyncio
import re

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
from backend.rag import get_rag, DATA_PATH

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
# Pre-carga del RAG al importar el módulo (evita problemas de threading con ChromaDB)
# ---------------------------------------------------------------------------
print("[*] Pre-cargando el motor RAG...")
_rag = get_rag()
print("[+] Motor RAG listo.")


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


def build_sources_data(retrieved_docs):
    """Construye los datos de las fuentes (una por acta/fecha citada).

    Usa el nº de página del metadato `page` (lo añade el indexador), así que es
    instantáneo. Para cada acta calcula el RANGO de páginas del debate (de la
    primera a la última página de sus fragmentos) para que el usuario localice
    rápido la propuesta en el PDF. Devuelve dicts serializables (sin objetos de
    Chainlit) para construir los elementos en el hilo async.
    """
    from collections import OrderedDict

    # Agrupar por (fecha, tema) = una PROPUESTA, conservando el orden de aparición.
    # Así cada propuesta del acta tiene su propia fuente con su rango de páginas,
    # aunque varias propuestas sean del mismo día.
    por_grupo = OrderedDict()
    for doc in retrieved_docs:
        pdf_path = resolve_pdf_path(doc.metadata.get("source", ""))
        if not pdf_path or not os.path.exists(pdf_path):
            continue
        date = doc.metadata.get("date", "Fecha desconocida")
        topic = doc.metadata.get("topic", "Tema general")
        key = (date, topic)
        por_grupo.setdefault(key, {"pdf_path": pdf_path, "docs": []})["docs"].append(doc)

    import urllib.parse

    sources_data = []
    for (date, topic), info in por_grupo.items():
        docs_d = info["docs"]
        first = docs_d[0]
        paginas = [d.metadata.get("page") for d in docs_d if d.metadata.get("page")]
        if not paginas:
            paginas = [1]
        p_min, p_max = min(paginas), max(paginas)
        rango = f"Pág. {p_min}" if p_min == p_max else f"Págs. {p_min}-{p_max}"

        # URL de la ruta propia /acta/{año}/{fichero}#page=N (abre el PDF en una
        # pestaña del navegador, saltando a la primera página del debate).
        pdf_path = info["pdf_path"]
        year = os.path.basename(os.path.dirname(pdf_path))
        fname = os.path.basename(pdf_path)
        url = f"/acta/{year}/{urllib.parse.quote(fname)}#page={p_min}"

        short_topic = topic[:80] + "..." if len(topic) > 80 else topic
        sources_data.append({
            "date": date,
            "topic": topic,
            "pdf_path": pdf_path,
            "url": url,
            "page": p_min,
            "speaker": first.metadata.get("speaker", "Desconocido"),
            "party": first.metadata.get("party", "Desconocido"),
            "short_topic": short_topic,
            "pdf_name": f"Ver PDF - Acta {date} ({rango})",
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

    # Fase 2: Generación + fuentes en UN SOLO mensaje. Las fuentes son hipervínculos
    # normales a la ruta /acta/... (abren el PDF en una pestaña del navegador en la
    # página correcta), en lugar de elementos cl.Pdf "side" que se abrían solos.
    async with cl.Step(name="Redactando la crónica"):
        respuesta = await rag.llm.ainvoke(prompt_value)
        answer_text = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

    # Localizar las fuentes y colocar el enlace de cada acta DEBAJO de su bloque de fecha.
    if retrieved_docs:
        async with cl.Step(name="Localizando fuentes en los PDFs"):
            sources_data = await asyncio.to_thread(build_sources_data, retrieved_docs)

        if sources_data:
            from collections import defaultdict

            # La conclusión final acota el último bloque (no metemos fuentes dentro de ella).
            concl = re.search(r'\n\s*(?:CONCLUSI[ÓO]N|En conclusi|En resumen)', answer_text, re.I)
            concl_pos = concl.start() if concl else len(answer_text)

            # Inicio de cada bloque de propuesta = cada aparición de una fecha conocida.
            fechas = {s["date"] for s in sources_data}
            bloques = []  # (pos_inicio, fecha)
            for date in fechas:
                for m in re.finditer(re.escape(date), answer_text[:concl_pos]):
                    bloques.append([m.start(), date])
            bloques.sort()

            # Fuentes agrupadas por fecha.
            src_por_fecha = defaultdict(list)
            for s in sources_data:
                src_por_fecha[s["date"]].append(s)

            # Asignar a cada bloque la fuente de su misma fecha cuyo GRUPO coincide más
            # (comparando el nombre del grupo en la cabecera con el tema de la fuente).
            # Si ninguna coincide, se usa el orden de aparición como respaldo.
            asignaciones = []  # (pos_fin_bloque, source)
            usadas = []
            for idx, (start, date) in enumerate(bloques):
                candidatos = [s for s in src_por_fecha[date] if id(s) not in usadas]
                if not candidatos:
                    continue
                fin = bloques[idx + 1][0] if idx + 1 < len(bloques) else concl_pos
                cabecera = answer_text[start:start + 120]  # "fecha — Grupo Municipal ..."
                cab_words = _palabras_clave(cabecera)
                mejor = max(candidatos, key=lambda s: len(cab_words & _palabras_clave(s["topic"])))
                if len(cab_words & _palabras_clave(mejor["topic"])) == 0:
                    mejor = candidatos[0]  # sin coincidencia de grupo → respaldo por orden
                asignaciones.append((fin, mejor))
                usadas.append(id(mejor))

            # Insertar de atrás hacia delante, retrocediendo sobre el formato del
            # encabezado siguiente (**, #, saltos) para NO romper la negrita del título.
            for fin, s in sorted(asignaciones, key=lambda x: x[0], reverse=True):
                f = fin
                while f > 0 and answer_text[f - 1] in "*#\n\r \t":
                    f -= 1
                linea = f"\n\n📄 *Fuente:* [{s['pdf_name']}]({s['url']})\n"
                answer_text = answer_text[:f] + linea + answer_text[f:]

            # Fuentes que no se pudieron ubicar en el texto: al final, agrupadas.
            no_ubicadas = [s for s in sources_data if id(s) not in usadas]
            if no_ubicadas:
                answer_text += "\n\n**Otras fuentes consultadas:**\n"
                for s in no_ubicadas:
                    answer_text += f"* [{s['pdf_name']}]({s['url']})\n"

    await cl.Message(content=answer_text).send()
