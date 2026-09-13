# -*- coding: utf-8 -*-
import os, json, math
import example_bank as EB

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bank_embeds.json")
_embedder = None
_bank_vecs = None
_q_cache = {}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from langchain_ollama import OllamaEmbeddings
        _embedder = OllamaEmbeddings(model="bge-m3")
    return _embedder


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def _bank_embeddings():
    global _bank_vecs
    if _bank_vecs is not None:
        return _bank_vecs
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(EB.BANK) and all(cached[i]["q"] == EB.BANK[i]["q"] for i in range(len(EB.BANK))):
            _bank_vecs = [c["vec"] for c in cached]
            return _bank_vecs
    emb = _get_embedder()
    vecs = emb.embed_documents([ex["q"] for ex in EB.BANK])
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump([dict(q=EB.BANK[i]["q"], vec=vecs[i]) for i in range(len(EB.BANK))], f)
    _bank_vecs = vecs
    return vecs


def _embed_q(pregunta):
    if pregunta in _q_cache:
        return _q_cache[pregunta]
    v = _get_embedder().embed_query(pregunta)
    _q_cache[pregunta] = v
    return v


def nearest(pregunta, k=3):
    qv = _embed_q(pregunta)
    vecs = _bank_embeddings()
    scored = [(_cos(qv, vecs[i]), EB.BANK[i]) for i in range(len(EB.BANK))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]


def compare(pregunta, k=3):
    jac = EB.nearest(pregunta, k)
    emb = nearest(pregunta, k)
    return [e["q"][:40] for e in jac], [e["q"][:40] for e in emb]


if __name__ == "__main__":
    from gold import GOLD
    diffs = 0
    for g in GOLD:
        j, e = compare(g["q"], 3)
        same = (j == e)
        diffs += (not same)
        print(f"{g['id']:5} {'=' if same else '!=':2} jaccard={j}")
        if not same:
            print(f"          embed  ={e}")
    print(f"\n{diffs}/{len(GOLD)} preguntas con selección DISTINTA entre jaccard y embedding")
