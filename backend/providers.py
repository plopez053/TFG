"""Adaptadores de los servicios externos: LLM (Groq + Ollama con fallback
automático), reranker (Cohere) y healthcheck de Ollama.

Aísla los SDK, las claves y los reintentos del resto del código: el pipeline
(RAGPipeline, graph_rag_sparql) orquesta, esto habla con las APIs.
"""
import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# --- modelos y claves ---
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL_LOCAL = "qwen2.5:7b"        # LLM local (narración de respaldo, SPARQL de GraphRAG)
LLM_MODEL_GROQ = "openai/gpt-oss-120b"
LLM_MODEL_GRAPHRAG = "qwen2.5:7b"
COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

OLLAMA_URL = "http://localhost:11434"


# True si Ollama está activo en localhost:11434
def ping_ollama(timeout: float = 3.0) -> bool:
    try:
        import httpx
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


class LLMProvider:
    """LLM con proveedor principal + respaldo automático.

    prefer='groq'   -> Groq primero, Ollama de respaldo (narración).
    prefer='ollama' -> Ollama primero, Groq de respaldo (generación de SPARQL).
    Cae al respaldo ante CUALQUIER fallo del principal (rate limit, red...).
    """

    def __init__(self, prefer: str = "groq", *, groq_model: str = LLM_MODEL_GROQ,
                 local_model: str = LLM_MODEL_LOCAL, num_ctx: int = 8192,
                 ollama_timeout: int = 600, verbose: bool = True):
        self.prefer = prefer
        self._verbose = verbose
        candidatos = []

        groq = self._build_groq(groq_model)
        ollama = self._build_ollama(local_model, num_ctx, ollama_timeout)
        orden = [groq, ollama] if prefer == "groq" else [ollama, groq]
        candidatos = [c for c in orden if c is not None]

        if not candidatos:
            raise RuntimeError("Ningún proveedor LLM disponible: configura GROQ_API_KEY o arranca Ollama.")

        (self.primary_name, self.primary) = candidatos[0]
        (self.fallback_name, self.fallback) = candidatos[1] if len(candidatos) > 1 else (None, None)
        if verbose:
            fb = self.fallback_name or "sin fallback"
            print(f"[*] LLM principal: {self.primary_name} | Fallback: {fb}", flush=True)

    def _build_groq(self, model):
        if not GROQ_API_KEY:
            if self._verbose:
                print("[!] GROQ_API_KEY no configurada — Groq no disponible", flush=True)
            return None
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=model, temperature=0, api_key=GROQ_API_KEY)
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
            # timeout explícito: sin él una conexión colgada con Ollama bloquea
            # el proceso indefinidamente en lugar de fallar y reintentar.
            llm = ChatOllama(model=model, temperature=0, num_ctx=num_ctx,
                             client_kwargs={"timeout": timeout})
            if self._verbose:
                print(f"[+] LLM disponible: Ollama local ({model})", flush=True)
            return (f"Ollama/{model}", llm)
        except Exception as e:
            print(f"[!] Ollama LLM no inicializado: {e}", flush=True)
            return None

    def invoke(self, prompt):
        try:
            return self.primary.invoke(prompt)
        except Exception as e:
            if self.fallback is None:
                raise
            print(f"\n[!] LLM principal falló — {type(e).__name__}: {e}", flush=True)
            print("[~] Usando LLM de respaldo...", flush=True)
            return self.fallback.invoke(prompt)

    async def ainvoke(self, prompt):
        try:
            return await self.primary.ainvoke(prompt)
        except Exception as e:
            if self.fallback is None:
                raise
            print(f"\n[!] LLM principal falló (async) — {type(e).__name__}: {e}", flush=True)
            print("[~] Usando LLM de respaldo (async)...", flush=True)
            return await self.fallback.ainvoke(prompt)


# --- reranker (Cohere) ---
_cohere_client = None


def rerank(docs: list, query: str, top_n: int = 30, *, max_candidates: int = 80) -> list:
    """Reordena los documentos por relevancia con Cohere y devuelve el top_n.

    Guarda el score en metadata['_rerank_score']. Los docs marcados con
    metadata['_kw'] (canal literal) que Cohere deja fuera se reincorporan al
    final con un score bajo. Si no hay clave o la API falla, devuelve `docs`.
    """
    global _cohere_client
    if not COHERE_API_KEY or not docs:
        return docs
    try:
        import cohere
        if _cohere_client is None:
            _cohere_client = cohere.Client(COHERE_API_KEY)

        vistos: set = set()
        unicos = []
        for d in docs:
            k = d.page_content[:200]
            if k not in vistos:
                vistos.add(k)
                unicos.append(d)
        candidatos = unicos[:max_candidates]

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
        en_top = {id(d) for d in reranked}
        for d in candidatos:
            if d.metadata.get("_kw") and id(d) not in en_top:
                d.metadata.setdefault("_rerank_score", 0.02)
                reranked.append(d)
        print(f"[*] Cohere reranking: {len(candidatos)} candidatos -> top {len(reranked)}")
        return reranked
    except Exception as e:
        print(f"[!] Cohere reranking fallido, usando ChromaDB directo: {e}")
        return docs
