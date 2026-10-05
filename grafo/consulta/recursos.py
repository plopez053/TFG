"""
Recursos del GraphRAG: el grafo RDF en memoria, con la ejecución de consultas
SPARQL con límite de tiempo, y los modelos de lenguaje.
"""
import ctypes
import os
import threading

from comun.proveedores import (build_gemini as _build_gemini, GOOGLE_CLOUD_PROJECT, LLM_MODEL_GEMINI,
                               LLM_MODEL_GRAPHRAG, LLM_MODEL_GROQ, ping_ollama as _ping_ollama,
                               texto_llm as _texto_llm)
from comun.rutas import GRAFO_TTL as GRAPH_TTL


# =============================================================================
# El grafo RDF
# =============================================================================

_graph = None
_graph_lock = threading.Lock()


def _load_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                from rdflib import Graph, Namespace
                g = Graph()
                g.parse(GRAPH_TTL, format="turtle")
                bo = Namespace("http://bilbao.tfg/ontology#")
                # el .ttl ya trae sin copias en euskera, con fechas, voto por grupo
                # de las votaciones nominales, un pleno por acta y legislaturas
                # (construccion/enriquecer.py); antes se hacía aquí, en memoria
                if (None, bo.enriquecido, None) not in g:
                    print("[!] El grafo no está enriquecido: reconstrúyelo con python -m grafo.construccion.build_rdf", flush=True)
                _graph = g
    return _graph


# Esquema compacto + 3 ejemplos elegidos por parecido a la pregunta. Según el
# estudio de ablación (ESTUDIO_SPARQL_LOCAL.md) acierta más que el prompt largo
# anterior, con un tercio del tamaño.
_PREFIXES = """PREFIX bo: <http://bilbao.tfg/ontology#>
PREFIX br: <http://bilbao.tfg/resource/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"""


class ConsultaDemasiadoLenta(Exception):
    pass


# Límite por consulta. Las consultas correctas tardan como mucho ~3 s (medido
# sobre 300 del estudio), pero una mal generada (variables sin enlazar ->
# producto cartesiano) puede pasar de 10 minutos. rdflib no permite cancelar
# una consulta: se ejecuta en un hilo y, si se pasa del límite, se le inyecta
# una excepción, que rdflib (Python puro) recibe en su siguiente instrucción.
SPARQL_TIMEOUT_S = 20


def _ejecutar(g, sparql: str, timeout: float = SPARQL_TIMEOUT_S) -> list:
    resultado = {}

    def trabajo():
        try:
            resultado["rows"] = [{str(v): str(row[v]) for v in row.labels} for row in g.query(sparql)]
        except BaseException as e:
            resultado["error"] = e

    hilo = threading.Thread(target=trabajo, daemon=True)
    hilo.start()
    hilo.join(timeout)
    if hilo.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(hilo.ident), ctypes.py_object(ConsultaDemasiadoLenta))
        hilo.join(5)
        raise ConsultaDemasiadoLenta(
            f"la consulta tardó más de {timeout:.0f} s; probablemente hay variables sin "
            "enlazar entre sí (producto cartesiano)")
    if "error" in resultado:
        raise resultado["error"]
    return resultado["rows"]


# =============================================================================
# Modelos de lenguaje
# =============================================================================

# LLM del GraphRAG: Ollama (qwen3:8b) para generar SPARQL, Groq para narrar y
# Gemini para el núcleo de la pregunta, cada uno con los otros de respaldo.

_llm_cache: dict = {}
_llm_cache_lock = threading.Lock()


# LLM del proveedor indicado, con caché por proveedor
def _get_llm(provider: str):
    if provider not in _llm_cache:
        with _llm_cache_lock:
            if provider not in _llm_cache:
                if provider == "ollama":
                    from langchain_ollama import ChatOllama
                    # num_predict alto: narrar un listado largo (p.ej. 50
                    # proposiciones) no cabe en el límite por defecto.
                    # reasoning=False: el modo "pensamiento" de qwen3 no mejora
                    # esta tarea y puede gastar todo num_predict sin llegar a
                    # escribir el SPARQL (misma configuración que el estudio).
                    opts = dict(model=LLM_MODEL_GRAPHRAG, temperature=0,
                                num_predict=8192, num_ctx=8192,
                                client_kwargs={"timeout": 300})
                    if LLM_MODEL_GRAPHRAG.startswith("qwen3"):
                        opts["reasoning"] = False
                    _llm_cache[provider] = ChatOllama(**opts)
                    print(f"[+] GraphRAG LLM: Ollama ({LLM_MODEL_GRAPHRAG})", flush=True)
                elif provider == "groq":
                    from langchain_groq import ChatGroq
                    groq_key = os.environ.get("GROQ_API_KEY", "")
                    _llm_cache[provider] = ChatGroq(
                        model=LLM_MODEL_GROQ, temperature=0, api_key=groq_key, max_tokens=8192
                    )
                    print(f"[+] GraphRAG LLM: Groq ({LLM_MODEL_GROQ})", flush=True)
                elif provider == "gemini":
                    _llm_cache[provider] = _build_gemini()
                    print(f"[+] GraphRAG LLM: Gemini ({LLM_MODEL_GEMINI}, Vertex)", flush=True)
    return _llm_cache[provider]


# invoca el LLM con fallback automático: para SPARQL ollama -> gemini -> groq;
# para narrar groq -> gemini -> ollama (el modelo local narraba peor y a veces
# en bucle cuando Groq agotaba la cuota); con prefer="gemini", gemini primero
def _llm_invoke(prompt: str, prefer: str = "ollama") -> str:
    orden = {"ollama": ["ollama", "gemini", "groq"], "groq": ["groq", "gemini", "ollama"],
             "gemini": ["gemini", "groq", "ollama"]}[prefer]
    groq_key = os.environ.get("GROQ_API_KEY", "")
    errores = []

    for proveedor in orden:
        if proveedor == "gemini":
            if not GOOGLE_CLOUD_PROJECT:
                continue
            try:
                return _texto_llm(_get_llm("gemini").invoke(prompt))
            except Exception as e:
                print(f"[!] GraphRAG Gemini falló — {type(e).__name__}: {e}", flush=True)
                errores.append(str(e))
        elif proveedor == "ollama":
            if not _ping_ollama():
                print("[!] GraphRAG: Ollama no disponible en localhost:11434", flush=True)
                continue
            try:
                return _get_llm("ollama").invoke(prompt).content
            except Exception as e:
                print(f"\n[!] GraphRAG Ollama falló — {type(e).__name__}: {e}", flush=True)
                errores.append(str(e))
        else:
            if not groq_key:
                continue
            try:
                print(f"[~] GraphRAG usando Groq ({'preferido' if prefer=='groq' else 'fallback'})...", flush=True)
                texto = _get_llm("groq").invoke(prompt).content
                # gpt-oss-120b puede agotar el límite de salida razonando y devolver vacío
                if not str(texto).strip():
                    raise ValueError("respuesta vacía")
                return texto
            except Exception as e:
                print(f"[!] GraphRAG Groq también falló — {type(e).__name__}: {e}", flush=True)
                errores.append(str(e))

    raise RuntimeError(f"GraphRAG: ningún LLM disponible (Ollama y Groq fallaron): {errores}")
