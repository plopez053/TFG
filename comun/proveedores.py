"""
Proveedores externos de modelos: LLM (Groq, con Gemini y Ollama de respaldo,
en ese orden) y reordenación (Cohere, con un reordenador local de respaldo).
Carga el .env del proyecto al importarse.
"""
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# --- modelos y claves ---
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL_LOCAL = "qwen2.5:7b"        # LLM local (narración de respaldo, SPARQL de GraphRAG)
LLM_MODEL_GROQ = "openai/gpt-oss-120b"
LLM_MODEL_GRAPHRAG = "qwen3:8b"
COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Gemini en Vertex AI (créditos del proyecto de Google Cloud). En el estudio
# de SPARQL, con el prompt de producción, acertó 102 de 105 (qwen3:8b, 99).
# Solo está disponible en la región "global".
LLM_MODEL_GEMINI = "gemini-3.5-flash-lite"
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GEMINI_LOCATION = "global"

OLLAMA_URL = "http://localhost:11434"


# --- LLM ---

# Cliente de Gemini en Vertex, o None si no hay proyecto configurado
def build_gemini(max_tokens: int = 8192):
    if not GOOGLE_CLOUD_PROJECT:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=LLM_MODEL_GEMINI, temperature=0, vertexai=True,
                                  project=GOOGLE_CLOUD_PROJECT, location=GEMINI_LOCATION,
                                  max_output_tokens=max_tokens)


# Texto de la respuesta de un LLM: Gemini devuelve una lista de partes
# ([{"type": "text", "text": ...}]) en vez de un str
def texto_llm(resp) -> str:
    c = resp.content if hasattr(resp, "content") else resp
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


# True si Ollama está activo en localhost:11434
def ping_ollama(timeout: float = 3.0) -> bool:
    try:
        import httpx
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


# Gemini con la misma interfaz que los demás: .content es siempre un str
class GeminiTexto:
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, prompt, *a, **kw):
        from langchain_core.messages import AIMessage
        return AIMessage(content=texto_llm(self._llm.invoke(prompt, *a, **kw)))

    async def ainvoke(self, prompt, *a, **kw):
        from langchain_core.messages import AIMessage
        return AIMessage(content=texto_llm(await self._llm.ainvoke(prompt, *a, **kw)))


class LLMProvider:

    def __init__(self, prefer: str = "groq", *, groq_model: str = LLM_MODEL_GROQ,
                 local_model: str = LLM_MODEL_LOCAL, num_ctx: int = 8192,
                 ollama_timeout: int = 600, verbose: bool = True):
        self.prefer = prefer
        self._verbose = verbose

        groq = self._build_groq(groq_model)
        gemini = self._build_gemini()
        ollama = self._build_ollama(local_model, num_ctx, ollama_timeout)
        # Gemini va antes que Ollama: cuando Groq agota la cuota, el modelo
        # local responde peor y a veces entra en bucle
        orden = [groq, gemini, ollama] if prefer == "groq" else [ollama, groq, gemini]
        candidatos = [c for c in orden if c is not None]

        if not candidatos:
            raise RuntimeError("Ningún proveedor LLM disponible: configura GROQ_API_KEY o arranca Ollama.")

        (self.primary_name, self.primary) = candidatos[0]
        (self.fallback_name, self.fallback) = candidatos[1] if len(candidatos) > 1 else (None, None)
        (self.fallback2_name, self.fallback2) = candidatos[2] if len(candidatos) > 2 else (None, None)
        if verbose:
            fb = " -> ".join(n for n in (self.fallback_name, self.fallback2_name) if n) or "sin fallback"
            print(f"[*] LLM principal: {self.primary_name} | Fallback: {fb}", flush=True)

    def _build_gemini(self):
        try:
            llm = build_gemini()
        except Exception as e:
            print(f"[!] Gemini no inicializado: {e}", flush=True)
            return None
        if llm is None:
            return None
        if self._verbose:
            print(f"[+] LLM disponible: Gemini ({LLM_MODEL_GEMINI}, Vertex)", flush=True)
        return (f"Gemini/{LLM_MODEL_GEMINI}", GeminiTexto(llm))

    def _build_groq(self, model):
        if not GROQ_API_KEY:
            if self._verbose:
                print("[!] GROQ_API_KEY no configurada — Groq no disponible", flush=True)
            return None
        try:
            from langchain_groq import ChatGroq
            # sin reasoning_effort bajo, gpt-oss-120b puede gastar todo el límite de salida razonando y devolver la respuesta vacia
            llm = ChatGroq(model=model, temperature=0, api_key=GROQ_API_KEY,
                          max_tokens=8192, reasoning_effort="low")
            if self._verbose:
                print(f"[+] LLM disponible: Groq ({model})", flush=True)
            return (f"Groq/{model}", llm)
        except Exception as e:
            print(f"[!] Groq no inicializado: {e}", flush=True)
            return None

    def _build_ollama(self, model, num_ctx, timeout):
        if not ping_ollama():
            return None
        try:
            from langchain_ollama import ChatOllama
            # num_predict evita que qwen2.5:7b se quede en un bucle de repetición
            # (el timeout no salta porque Ollama sigue enviando tokens)
            llm = ChatOllama(model=model, temperature=0, num_ctx=num_ctx,
                             num_predict=4096, client_kwargs={"timeout": timeout})
            if self._verbose:
                print(f"[+] LLM disponible: Ollama local ({model})", flush=True)
            return (f"Ollama/{model}", llm)
        except Exception as e:
            print(f"[!] Ollama LLM no inicializado: {e}", flush=True)
            return None

    def _cadena(self):
        return [(n, l) for n, l in ((self.primary_name, self.primary), (self.fallback_name, self.fallback),
                                    (self.fallback2_name, self.fallback2)) if l is not None]

    def invoke(self, prompt):
        cadena = self._cadena()
        for i, (nombre, llm) in enumerate(cadena):
            try:
                return llm.invoke(prompt)
            except Exception as e:
                if i == len(cadena) - 1:
                    raise
                print(f"\n[!] LLM {nombre} falló — {type(e).__name__}: {e}", flush=True)
                print(f"[~] Usando LLM de respaldo ({cadena[i + 1][0]})...", flush=True)

    async def ainvoke(self, prompt):
        cadena = self._cadena()
        for i, (nombre, llm) in enumerate(cadena):
            try:
                return await llm.ainvoke(prompt)
            except Exception as e:
                if i == len(cadena) - 1:
                    raise
                print(f"\n[!] LLM {nombre} falló (async) — {type(e).__name__}: {e}", flush=True)
                print(f"[~] Usando LLM de respaldo ({cadena[i + 1][0]}, async)...", flush=True)


# --- reranker (Cohere) ---

_cohere_client = None
# La clave Trial de Cohere tiene un tope de 1000 llamadas al mes. Al agotarse,
# no tiene sentido reintentar en cada pregunta: se pasa al reranker local.
_cohere_agotado = False

# Reranker local de respaldo: similitud coseno con bge-m3 (Ollama). No es un
# cross-encoder como el de Cohere, pero ordena los candidatos por relevancia a
# la pregunta, que es lo que necesitan el rescate y la priorización de temas
RERANK_LOCAL_MODEL = "bge-m3"
_emb_local = None
_emb_cache: dict = {}


def _sin_duplicados(docs: list, max_candidates: int) -> list:
    # sin duplicados (mismo inicio de texto) y como mucho max_candidates
    vistos: set = set()
    unicos = []
    for d in docs:
        k = d.page_content[:200]
        if k not in vistos:
            vistos.add(k)
            unicos.append(d)
    return unicos[:max_candidates]


def _cerrar_ranking(reranked: list, candidatos: list) -> list:
    # los del canal literal que el reranker deja fuera se añaden al final con
    # una puntuación mínima, para que el rescate de _select_final_docs los vea
    en_top = {id(d) for d in reranked}
    for d in candidatos:
        if d.metadata.get("_kw") and id(d) not in en_top:
            d.metadata.setdefault("_rerank_score", 0.02)
            reranked.append(d)
    return reranked


def rerank_local(docs: list, query: str, top_n: int = 30, *, max_candidates: int = 80) -> list:
    global _emb_local
    import numpy as np
    candidatos = _sin_duplicados(docs, max_candidates)
    if not candidatos:
        return docs
    if _emb_local is None:
        from langchain_ollama import OllamaEmbeddings
        _emb_local = OllamaEmbeddings(model=RERANK_LOCAL_MODEL)
    textos = [d.page_content[:1500] for d in candidatos]
    nuevos = [t for t in dict.fromkeys(textos) if t not in _emb_cache]
    if nuevos:
        for t, v in zip(nuevos, _emb_local.embed_documents(nuevos)):
            _emb_cache[t] = np.asarray(v, dtype="float32")
    q = np.asarray(_emb_local.embed_query(query), dtype="float32")
    q /= (np.linalg.norm(q) or 1.0)
    puntos = []
    for t in textos:
        v = _emb_cache[t]
        puntos.append(float(v @ q / (np.linalg.norm(v) or 1.0)))
    orden = sorted(range(len(candidatos)), key=lambda i: -puntos[i])[:min(top_n, len(candidatos))]
    reranked = []
    for i in orden:
        candidatos[i].metadata["_rerank_score"] = puntos[i]
        reranked.append(candidatos[i])
    print(f"[*] Reranking local ({RERANK_LOCAL_MODEL}): {len(candidatos)} candidatos -> top {len(reranked)}")
    return _cerrar_ranking(reranked, candidatos)


def rerank(docs: list, query: str, top_n: int = 30, *, max_candidates: int = 80) -> list:
    global _cohere_client, _cohere_agotado
    if not docs:
        return docs
    if COHERE_API_KEY and not _cohere_agotado:
        try:
            import cohere
            if _cohere_client is None:
                _cohere_client = cohere.Client(COHERE_API_KEY)
            candidatos = _sin_duplicados(docs, max_candidates)
            res = _cohere_client.rerank(
                model=COHERE_RERANK_MODEL, query=query,
                documents=[d.page_content for d in candidatos],
                top_n=min(top_n, len(candidatos)),
            )
            reranked = []
            for r in res.results:
                doc = candidatos[r.index]
                doc.metadata["_rerank_score"] = float(r.relevance_score)
                reranked.append(doc)
            print(f"[*] Cohere reranking: {len(candidatos)} candidatos -> top {len(reranked)}")
            return _cerrar_ranking(reranked, candidatos)
        except Exception as e:
            msg = str(e)
            if "/ month" in msg:
                _cohere_agotado = True
                print("[!] Cohere: cuota mensual agotada; se usa el reranker local.", flush=True)
            else:
                print(f"[!] Cohere reranking fallido ({msg[:150]}), se usa el reranker local.", flush=True)
    try:
        return rerank_local(docs, query, top_n, max_candidates=max_candidates)
    except Exception as e:
        print(f"[!] Reranker local fallido, usando ChromaDB directo: {e}", flush=True)
        return docs
