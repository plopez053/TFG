# -*- coding: utf-8 -*-
import os, sys, time, re
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # TFG/TFG
_GRAPHRAG = os.path.join(_ROOT, "graphrag", "graphrag")
sys.path.insert(0, _GRAPHRAG)
# OJO: NO poner os.environ.setdefault("GROQ_API_KEY", "") aqui -- si se fija a
# "" ANTES de que backend/graph_rag_sparql carguen el .env real, python-dotenv
# ya no lo sobreescribe (load_dotenv sin override=True respeta un env var ya
# existente, aunque sea "") y la key queda vacia para todo el proceso -> Groq
# responde "Illegal header value b'Bearer '" / APIConnectionError en TODAS las
# llamadas. Causó ~40 min de fallos falsos en el estudio de modelos (2026-09-11).

import rdflib
from graph_rag_sparql import _clean_sparql
from gold import PREFIXES

_TTL = os.path.join(_GRAPHRAG, "bilbao_reasoned.ttl")
_G = None


def graph():
    global _G
    if _G is None:
        _G = rdflib.Graph()
        _G.parse(_TTL, format="turtle")
    return _G


# ---------------- LLMs ----------------
_llm_cache = {}


GEMINI_ALIAS = {"gemini": "gemini-3.5-flash-lite"}


GROQ_ALIAS = {"groq": "openai/gpt-oss-120b", "groq-20b": "openai/gpt-oss-20b",
              "groq-qwen27b": "qwen/qwen3.6-27b"}


def get_llm(model):
    if model.startswith("groq"):
        # SIN caché: un cliente httpx de larga vida en este proceso empezó a
        # fallar con "Connection error" en TODAS las llamadas tras varios
        # minutos (pool de conexiones roto, probablemente el proxy/NAT corta
        # el keep-alive) -- un cliente nuevo por llamada evita reusar el pool.
        from langchain_groq import ChatGroq
        from dotenv import load_dotenv
        env_path = os.path.join(_ROOT, ".env")
        load_dotenv(env_path, override=True)
        k = os.environ.get("GROQ_API_KEY_REAL") or os.environ.get("GROQ_API_KEY", "")
        if not k:
            raise RuntimeError(f"GROQ_API_KEY vacia (buscado en {env_path})")
        name = GROQ_ALIAS.get(model, model)
        return ChatGroq(model=name, temperature=0, api_key=k, max_retries=2)
    if model in _llm_cache:
        return _llm_cache[model]
    if model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from dotenv import load_dotenv
        env_path = os.path.join(_ROOT, ".env")
        load_dotenv(env_path, override=True)
        k = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        name = GEMINI_ALIAS.get(model, model)
        llm = ChatGoogleGenerativeAI(model=name, temperature=0, google_api_key=k)
    else:
        from langchain_ollama import ChatOllama
        opts = dict(model=model, temperature=0, num_ctx=8192, client_kwargs={"timeout": 300})
        if model.startswith("qwen3"):
            opts["reasoning"] = False          # sin "thinking"
        llm = ChatOllama(**opts)
    _llm_cache[model] = llm
    return llm


def _msg_text(msg):
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                out.append(b.get("text", "") or b.get("content", "") or "")
        return "".join(out)
    return str(c)


# ---------------- ejecución + scoring ----------------
def run_sparql(sparql):
    q = sparql if re.search(r"\bPREFIX\b", sparql) else PREFIXES + "\n" + sparql
    if not q.strip():
        return None, "empty query"
    try:
        res = graph().query(q)
        rows = []
        for r in res:                     # rdflib evalúa PEREZOSAMENTE aquí dentro
            if hasattr(r, "__len__") and not isinstance(r, str):
                rows.append([str(x) for x in r])
            else:
                rows.append([str(r)])
        return rows, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def _nums(flat):
    return [int(x) for x in flat if re.fullmatch(r"-?\d+", x)]


def score(rows, err, gold):
    if err is not None:
        return "error"
    flat = [v for row in rows for v in row]
    joined = " | ".join(flat).lower()
    ns = _nums(flat)
    is_zeroish = (len(rows) == 0) or (len(flat) > 0 and all(
        (v in ("", "0") or v.lower() == "none") for v in flat))

    if gold.get("graceful"):
        # la respuesta correcta ES 0 / vacío
        return "ok" if is_zeroish or ns == [0] else "wrong"

    if is_zeroish:
        return "empty"

    for m in gold.get("must", []):
        if m == "_ANY_":
            if not ns or max(ns) == 0:
                return "wrong"
        elif m.lower() not in joined:
            return "wrong"
    for m in gold.get("must_not", []):
        if m in flat:
            return "wrong"
    if gold.get("top") and rows:
        if gold["top"].lower() not in " | ".join(rows[0]).lower():
            return "wrong"
    if gold.get("min_rows") and len(rows) < gold["min_rows"]:
        return "wrong"
    return "ok"


def generate(arm_fn, model, pregunta):
    prompt = arm_fn(pregunta)
    llm = get_llm(model)
    t0 = time.time()
    raw = _msg_text(llm.invoke(prompt))
    dt = time.time() - t0
    return dict(sparql=_clean_sparql(raw), latency=round(dt, 1), raw=raw, prompt_chars=len(prompt))


def evaluate(arm_name, arm_fn, model, gold, reps=3):
    out = []
    for g in gold:
        for rep in range(reps):
            try:
                gen = generate(arm_fn, model, g["q"])
            except Exception as e:
                out.append(dict(arm=arm_name, model=model, id=g["id"], shape=g["shape"],
                                rep=rep, verdict="error", err=f"GEN {type(e).__name__}: {str(e)[:120]}",
                                sparql="", latency=None, prompt_chars=None))
                continue
            rows, err = run_sparql(gen["sparql"])
            v = score(rows, err, g)
            out.append(dict(arm=arm_name, model=model, id=g["id"], shape=g["shape"], rep=rep,
                            verdict=v, err=err, sparql=gen["sparql"], latency=gen["latency"],
                            prompt_chars=gen["prompt_chars"],
                            nrows=(len(rows) if rows is not None else 0)))
    return out
