"""Fase 1 de GraphRAG: extrae UNIDADES DE PROPOSICIÓN desde los PDF de las actas.

Reutiliza la segmentación ya corregida de `backend.rag._process_single_pdf` (topics
limpios + vote_result), agrupa los chunks por (acta, tema) para reconstruir cada
proposición y vuelca un JSONL con, por proposición:

    date, topic, party (grupo proponente), vote_result, page_ini, page_fin,
    source (ruta pdf), text (texto reconstruido y deduplicado, truncado).

Estas unidades son el sustrato del grafo: cada una será un nodo :Proposicion,
enlazado a su :Grupo, :Pleno y :Resultado, y a las entidades/temas que el LLM
extraiga del `text` en la Fase 2.

Uso:  python graphrag/extract_proposals.py
Salida: graphrag/proposals.jsonl
"""
import os
import re
import sys
import json
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import RAGPipeline, DATA_PATH  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proposals.jsonl")
MAX_TEXT = 6000  # chars de texto por proposición que se pasan al LLM en la Fase 2
SKIP_TOPICS = {"", "General", "General / Introducción"}


def _strip_header(chunk_text: str) -> str:
    """Quita el prefijo 'ASUNTO: ...\\nORADOR: ...\\n\\n' que añade el indexador."""
    parts = chunk_text.split("\n\n", 1)
    return parts[1] if len(parts) == 2 and parts[0].startswith("ASUNTO:") else chunk_text


def _merge_overlapping(texts):
    """Une chunks consecutivos solapados (chunk_overlap=200) sin repetir el solape."""
    merged = ""
    for t in texts:
        t = t.strip()
        if not merged:
            merged = t
            continue
        # buscar el mayor solape entre el final de `merged` y el inicio de `t`
        overlap = 0
        maxo = min(len(merged), len(t), 400)
        for k in range(maxo, 20, -1):
            if merged[-k:] == t[:k]:
                overlap = k
                break
        merged += t[overlap:]
    return merged


def extract_proposals():
    rag = RAGPipeline()
    import glob
    pdfs = sorted(glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True))
    print(f"[*] {len(pdfs)} actas en {DATA_PATH}")

    proposals = []
    for path in pdfs:
        try:
            chunks = rag._process_single_pdf(path)
        except Exception as e:
            print(f"[!] {os.path.basename(path)}: {e}")
            continue

        # agrupar por tema dentro de esta acta (cada tema = una proposición/punto)
        by_topic = defaultdict(list)
        for c in chunks:
            by_topic[c.metadata.get("topic", "")].append(c)

        for topic, cs in by_topic.items():
            if topic in SKIP_TOPICS:
                continue
            cs = sorted(cs, key=lambda c: c.metadata.get("chunk_index", 0))
            date = cs[0].metadata.get("date", "")
            pages = [c.metadata.get("page") for c in cs if c.metadata.get("page")]
            parties = [c.metadata.get("party") for c in cs
                       if c.metadata.get("party") and c.metadata.get("party") != "Desconocido"]
            party = Counter(parties).most_common(1)[0][0] if parties else "Desconocido"
            vote = next((c.metadata.get("vote_result") for c in cs if c.metadata.get("vote_result")), None)
            text = _merge_overlapping([_strip_header(c.page_content) for c in cs])[:MAX_TEXT]

            proposals.append({
                "date": date,
                "topic": topic,
                "party": party,
                "vote_result": vote,
                "page_ini": min(pages) if pages else None,
                "page_fin": max(pages) if pages else None,
                "source": os.path.relpath(path, os.path.dirname(DATA_PATH)),
                "text": text,
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in proposals:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # resumen
    con_voto = sum(1 for p in proposals if p["vote_result"])
    años = Counter(p["date"].split("-")[-1] for p in proposals if p["date"])
    print(f"[+] {len(proposals)} proposiciones → {OUT_PATH}")
    print(f"    con vote_result: {con_voto} ({100*con_voto//max(len(proposals),1)}%)")
    print(f"    por año: {dict(sorted(años.items()))}")
    return proposals


if __name__ == "__main__":
    extract_proposals()
