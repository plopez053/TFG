# -*- coding: utf-8 -*-
"""
Evaluación RAGAS, fase 2: calcula Context Relevance, Faithfulness, Answer
Relevancy y Semantic Similarity sobre el dataset de generar.py.

Se ejecuta en .venv-ragas (ragas necesita una versión de langchain distinta
de la del proyecto). Juez: gemini-2.5-pro en Vertex AI; requiere en .env
GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION y GOOGLE_APPLICATION_CREDENTIALS.
Embeddings: nomic-embed-text en Ollama, el mismo modelo que usa el RAG.

Uso:
    .venv-ragas\\Scripts\\python.exe evaluacion\\ragas\\puntuar.py
"""
import sys, os, io, json, asyncio, statistics

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import instructor
from google import genai
from langchain_ollama import OllamaEmbeddings

from ragas.llms.base import InstructorLLM
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRelevance, SemanticSimilarity

DATASET_PATH = os.path.join(HERE, "dataset.json")
GROUND_TRUTH_PATH = os.path.join(HERE, "referencias.json")
RESULTS_JSON = os.path.join(HERE, "resultados.json")
RESULTS_CSV = os.path.join(HERE, "resultados.csv")

JUDGE_MODEL = "gemini-2.5-pro"
EMBED_MODEL = "nomic-embed-text"


# adaptador: las métricas de ragas 0.4 piden BaseRagasEmbedding
class OllamaRagasEmbedding(BaseRagasEmbedding):

    def __init__(self, model: str = EMBED_MODEL):
        super().__init__()
        self._emb = OllamaEmbeddings(model=model)

    def embed_text(self, text: str, **kwargs) -> list:
        return self._emb.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list:
        return await self._emb.aembed_query(text)


def load_merged_dataset() -> list:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        generated = json.load(f)
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    items = []
    for key in sorted(generated.keys(), key=int):
        gen = generated[key]
        gt = ground_truth.get(key)
        if gt is None:
            print(f"[!] Sin ground truth para la pregunta {key} -- se omite de la evaluación")
            continue
        items.append({
            "key": key,
            "question": gen["question"],
            "retrieved_contexts": gen["retrieved_contexts"],
            "response": gen["response"],
            "reference": gt["ground_truth"],
        })
    return items


async def score_with_retry(coro_fn, label, retries=5, base_wait=15):
    for attempt in range(retries):
        try:
            result = await coro_fn()
            return float(result)
        except Exception as e:
            if attempt == retries - 1:
                print(f"    [!] {label}: fallo definitivo tras {retries} intentos: {e}")
                return None
            wait = base_wait * (attempt + 1)
            print(f"    [!] {label}: error ({type(e).__name__}: {e}); reintento en {wait}s")
            await asyncio.sleep(wait)


# se guarda tras cada pregunta para poder reanudar si se corta
def save_results(rows):
    metric_cols = ["context_relevance", "faithfulness", "answer_relevancy", "semantic_similarity"]
    summary = {}
    for col in metric_cols:
        vals = [r[col] for r in rows if r.get(col) is not None]
        if vals:
            summary[col] = {
                "media": round(statistics.mean(vals), 4),
                "desviacion_tipica": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"por_pregunta": rows, "resumen": summary}, f, ensure_ascii=False, indent=2)

    import csv
    with open(RESULTS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question"] + metric_cols)
        w.writeheader()
        w.writerows(rows)
    return summary


async def run_eval():
    items = load_merged_dataset()
    print(f"Cargadas {len(items)} muestras para evaluar.")

    # reanudación: se saltan las preguntas que ya tienen las 4 métricas
    rows_by_id = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            prev = json.load(f)
        for r in prev.get("por_pregunta", []):
            rows_by_id[r["id"]] = r
        n_done = sum(
            1 for r in rows_by_id.values()
            if all(r.get(c) is not None for c in
                   ["context_relevance", "faithfulness", "answer_relevancy", "semantic_similarity"])
        )
        if n_done:
            print(f"[i] Reanudando: {n_done} preguntas ya puntuadas en {RESULTS_JSON}, se reutilizan.")

    google_client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    # cliente async construido a mano: llm_factory(provider="google") lo crea
    # síncrono y las métricas llaman a agenerate()
    patched_client = instructor.from_genai(google_client, use_async=True)
    llm = InstructorLLM(client=patched_client, model=JUDGE_MODEL, provider="google")
    embeddings = OllamaRagasEmbedding(EMBED_MODEL)

    context_relevance = ContextRelevance(llm=llm)
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    semantic_similarity = SemanticSimilarity(embeddings=embeddings)

    metric_cols = ["context_relevance", "faithfulness", "answer_relevancy", "semantic_similarity"]
    for i, item in enumerate(items, start=1):
        existing = rows_by_id.get(item["key"])
        if existing and all(existing.get(c) is not None for c in metric_cols):
            print(f"[{i}/{len(items)}] {item['question']} -- ya puntuada, se omite")
            continue

        print(f"[{i}/{len(items)}] {item['question']}")
        row = {"id": item["key"], "question": item["question"]}

        row["context_relevance"] = await score_with_retry(
            lambda it=item: context_relevance.ascore(
                user_input=it["question"], retrieved_contexts=it["retrieved_contexts"]),
            "context_relevance")

        row["faithfulness"] = await score_with_retry(
            lambda it=item: faithfulness.ascore(
                user_input=it["question"], response=it["response"],
                retrieved_contexts=it["retrieved_contexts"]),
            "faithfulness")

        row["answer_relevancy"] = await score_with_retry(
            lambda it=item: answer_relevancy.ascore(
                user_input=it["question"], response=it["response"]),
            "answer_relevancy")

        row["semantic_similarity"] = await score_with_retry(
            lambda it=item: semantic_similarity.ascore(
                reference=it["reference"], response=it["response"]),
            "semantic_similarity")

        print(f"    -> {row}")
        rows_by_id[item["key"]] = row
        save_results([rows_by_id[it["key"]] for it in items if it["key"] in rows_by_id])

    rows = [rows_by_id[it["key"]] for it in items if it["key"] in rows_by_id]
    summary = save_results(rows)

    print("\n=== RESUMEN (media ± desviación típica) ===")
    for col, s in summary.items():
        print(f"  {col}: {s['media']:.4f} ± {s['desviacion_tipica']:.4f}  (n={s['n']})")
    print(f"\nDetalle por pregunta: {RESULTS_CSV}")
    print(f"JSON completo: {RESULTS_JSON}")


if __name__ == "__main__":
    asyncio.run(run_eval())
