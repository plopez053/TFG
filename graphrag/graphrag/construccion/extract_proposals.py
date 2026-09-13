import os
import re
import sys
import json
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))            # .../graphrag/graphrag/construccion
GRAPHRAG = os.path.dirname(HERE)                               # .../graphrag/graphrag (datos + utils compartidos)
ROOT = os.path.dirname(os.path.dirname(GRAPHRAG))              # .../TFG/TFG (proyecto)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from backend.rag import RAGPipeline, DATA_PATH  # noqa: E402
from votos_parse import parse_listas_voto  # noqa: E402

OUT_PATH = os.path.join(GRAPHRAG, "proposals.jsonl")
# 30.000 chars ~ 8-9k tokens: cabe de sobra en el contexto de Gemini o Groq.
# El límite real de cuánto se le pasa al LLM lo pone build_graph.py::_ctx_chars
# según el modelo (los locales pequeños reciben menos). Este tope solo evita
# guardar textos gigantes en proposals.jsonl.
MAX_TEXT = 30000
SKIP_TOPICS = {"", "General", "General / Introducción"}


# quita el prefijo 'ASUNTO: ...\nORADOR: ...\n\n' que añade el indexador
def _strip_header(chunk_text: str) -> str:
    parts = chunk_text.split("\n\n", 1)
    return parts[1] if len(parts) == 2 and parts[0].startswith("ASUNTO:") else chunk_text


# une chunks consecutivos solapados (chunk_overlap=200) sin repetir el solape
def _merge_overlapping(texts):
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
            texto_completo = _merge_overlapping([_strip_header(c.page_content) for c in cs])
            text = texto_completo[:MAX_TEXT]
            # Capa determinista: listas nominales de voto tal cual las escribe el
            # acta ("Votos afirmativos: 14 señoras/señores: ..."). Se parsea del
            # texto COMPLETO (sin recortar a MAX_TEXT) porque la votación va al
            # final de la proposición. build_rdf.py resuelve apellido->concejal->grupo.
            votos_nominales = parse_listas_voto(texto_completo)
            # Apellidos de los concejales que intervinieron en el debate de esta
            # proposición (el indexador ya los detecta con speaker_regex). Se
            # usan en build_rdf.py para enlazar la proposición con los nodos
            # :Concejal del censo (ver concejales.jsonl).
            oradores = sorted({
                c.metadata.get("speaker") for c in cs
                if c.metadata.get("speaker") and c.metadata.get("speaker") not in ("Desconocido", "")
            })

            proposals.append({
                "date": date,
                "topic": topic,
                "party": party,
                "vote_result": vote,
                "page_ini": min(pages) if pages else None,
                "page_fin": max(pages) if pages else None,
                "source": os.path.relpath(path, os.path.dirname(DATA_PATH)),
                "text": text,
                "votos_nominales": votos_nominales,
                "oradores": oradores,
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in proposals:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # resumen
    con_voto = sum(1 for p in proposals if p["vote_result"])
    años = Counter(p["date"].split("-")[-1] for p in proposals if p["date"])
    print(f"[+] {len(proposals)} proposiciones -> {OUT_PATH}")
    print(f"    con vote_result: {con_voto} ({100*con_voto//max(len(proposals),1)}%)")
    print(f"    por año: {dict(sorted(años.items()))}")
    return proposals


if __name__ == "__main__":
    extract_proposals()
