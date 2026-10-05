"""
RAG vectorial: RAGPipeline reúne las etapas (indexado.py, recuperacion.py y
generacion.py) y las expone a la interfaz (retrieve_context y build_answer_prompt;
la respuesta final la compone generacion.componer_respuesta) y a la línea de
comandos (query).

Uso desde la línea de comandos:
    python -m vectorial.pipeline [--query "pregunta"]
"""
import argparse
import os
import sys
import threading
import time
from typing import Any, List, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from comun.proveedores import (COHERE_API_KEY, EMBEDDING_MODEL, GROQ_API_KEY, LLM_MODEL_GROQ, LLM_MODEL_LOCAL,
                               LLMProvider, ping_ollama as _ping_ollama)
from comun.rutas import CHROMA_PATH
from vectorial.generacion import componer_respuesta, construir_prompt, Contexto
from vectorial.indexado import Indexado
from vectorial.recuperacion import Recuperacion, Seleccion


# relevancia mínima de Cohere para considerar que hay algo que responder
RELEVANCE_FLOOR = 0.01


_RAG_SINGLETON_LOCK = threading.Lock()


class RAGPipeline(Indexado, Recuperacion, Seleccion, Contexto):

    def __init__(self):
        self._ollama_ok = _ping_ollama()
        if not self._ollama_ok:
            print("[!] ADVERTENCIA: Ollama no responde en localhost:11434 "
                  "— embeddings y LLM local no disponibles", flush=True)

        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = None
        self._llm_provider = LLMProvider(prefer="groq")
        self._known_dates: Optional[List[str]] = None
        self._indice_props = None
        self._indice_props_cargado = False

    def invoke_llm(self, prompt):
        return self._llm_provider.invoke(prompt)

    async def ainvoke_llm(self, prompt):
        return await self._llm_provider.ainvoke(prompt)

    @property
    def llm(self):
        return self._llm_provider.primary

    # mensajes del prompt de respuesta; lo usan la CLI y Chainlit
    def build_answer_prompt(self, ctx: dict) -> List[Any]:
        return construir_prompt(ctx)

    # recuperación para Chainlit: devuelve el contexto; el frontend llama al LLM
    def retrieve_context(self, question: str) -> dict:
        if not self.vector_store:
            self.create_vector_store()

        all_initial_docs, exact_date = self._retrieve_and_rank(question, k=80)

        # si ni el mejor fragmento supera el umbral de Cohere, la pregunta no
        # trata de las actas y no se responde
        if COHERE_API_KEY and not exact_date and all_initial_docs:
            # sin las puntuaciones de relleno (_cerrar_ranking): no las ha dado el reordenador
            rerank_scores = [d.metadata["_rerank_score"] for d in all_initial_docs
                             if "_rerank_score" in d.metadata and not d.metadata.get("_rerank_relleno")]
            max_score = max(rerank_scores) if rerank_scores else None
            if max_score is not None and max_score < RELEVANCE_FLOOR:
                print(f"[*] Relevancia máxima {max_score:.4f} < {RELEVANCE_FLOOR}: sin resultados.")
                return {
                    "context": "", "is_multi_session": False,
                    "unique_dates": [], "docs": [], "question": question,
                }

        docs = self._select_final_docs(all_initial_docs, question, exact_date)
        return self._build_context(docs, question)

    # flujo completo para la CLI: el mismo camino que la interfaz
    def query(self, question: str) -> str:
        ctx = self.retrieve_context(question)
        # con el contexto vacío no se llama al LLM, para que no invente
        if not ctx["context"].strip():
            return ("Lo siento, no he encontrado fragmentos relevantes en las actas para responder a tu pregunta. "
                    "Puede que el tema no esté cubierto en los documentos indexados, o que el modelo de embeddings "
                    "no haya podido conectarse. Prueba a reformular la pregunta.")

        print("[*] Generando crónica detallada...")
        response = self._invoke_with_retry(self.build_answer_prompt(ctx))
        return componer_respuesta(response, ctx)

    # con poca RAM, Ollama rechaza conexiones mientras recarga el modelo
    def _invoke_with_retry(self, prompt_msgs) -> str:
        for attempt in range(3):
            try:
                resp = self.invoke_llm(prompt_msgs)
                return resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                transitorio = any(msg in str(e) for msg in (
                    "Connection refused", "RemoteProtocolError", "Server disconnected", "ConnectError"))
                if attempt < 2 and transitorio:
                    print(f"[!] LLM cargando modelo, reintentando ({attempt+2}/3)...", flush=True)
                    time.sleep(10)
                else:
                    raise


_rag_instance: Optional[RAGPipeline] = None


# instancia única del RAG para el frontend (se crea en la primera llamada)
def get_rag() -> RAGPipeline:
    global _rag_instance
    with _RAG_SINGLETON_LOCK:
        if _rag_instance is None:
            _rag_instance = RAGPipeline()
            _rag_instance.vector_store = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=_rag_instance.embeddings
            )
            # precarga el LLM para que la primera pregunta no espere
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

    # precarga el modelo para que la primera pregunta no espere
    print("[*] Cargando modelo LLM...")
    for attempt in range(3):
        try:
            rag.llm.invoke("ok")
            print("[+] Modelo listo.")
            break
        except Exception:
            if attempt < 2:
                print(f"[!] Modelo cargando, reintentando ({attempt+2}/3)...")
                time.sleep(10)

    def imprimir(res: str):
        try:
            print(res)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(res.encode('utf-8'))

    if args.query:
        print("\nRESPUESTA:")
        imprimir(rag.query(args.query))
        return

    print(f"\n--- RAG Bilbao Ready [Model: {LLM_MODEL_GROQ if GROQ_API_KEY else LLM_MODEL_LOCAL}] ---")
    while True:
        try:
            q = input("\nPregunta: ")
            if not q.strip():
                continue
            if q.lower() in ['exit', 'quit', 'salir']:
                break
            imprimir(rag.query(q))
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
