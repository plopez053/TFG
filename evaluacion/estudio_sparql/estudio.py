"""
Estudio de generación de SPARQL: 35 preguntas de referencia (gold.py), varias
formas de construir el prompt (brazos.py) y 10 modelos.

    python evaluacion/estudio_sparql/estudio.py ejecutar --arms A_min,F_full --model qwen3:8b [--reps 3] [--out f.jsonl]
    python evaluacion/estudio_sparql/estudio.py analizar [resultados.jsonl]
    python evaluacion/estudio_sparql/estudio.py plantillas
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # raíz del proyecto

import argparse
import collections
import json
import re
import time

import rdflib

from comun.rutas import GRAFO_TTL
from gold import GOLD, PREFIXES
from grafo.consulta.guardas import _clean_sparql
from grafo.consulta.plantillas import answer


# =============================================================================
# Modelos, ejecución del SPARQL y puntuación
# =============================================================================

# OJO: no fijar GROQ_API_KEY a "" antes de que se cargue el .env: load_dotenv sin
# override=True respeta una variable ya existente, aunque esté vacía, y Groq falla en todas
# las llamadas (costó 40 minutos de fallos falsos en el estudio, 11-09-2026).

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # raíz del proyecto

_TTL = GRAFO_TTL
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


# =============================================================================
# Órdenes
# =============================================================================

# genera y puntúa: cada brazo con cada pregunta de referencia, varias repeticiones. Reanudable.
def ejecutar():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "study_results.jsonl"))
    ap.add_argument("--groq-key", default="")
    ap.add_argument("--sleep", type=float, default=0.0, help="pausa entre generaciones (s), p.ej. groq TPM")
    args = ap.parse_args()

    if args.model == "groq" and args.groq_key:
        os.environ["GROQ_API_KEY_REAL"] = args.groq_key

    # los brazos se importan DESPUÉS de fijar la clave
    import brazos as B
    ARM_FNS = {**B.ARMS, **B.ARMS_F, **B.ARMS_EMBED}
    POST = {**B.POST, **B.POST_EMBED}

    arms_sel = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms_sel:
        if a not in ARM_FNS:
            sys.exit(f"arm desconocido: {a} (disponibles: {list(ARM_FNS)})")

    # reanudar: qué (arm,model,id,rep) ya está hecho
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                d = json.loads(line)
                done.add((d["arm"], d["model"], d["id"], d["rep"]))
            except Exception:
                pass
    print(f"[{len(done)} generaciones ya hechas]", flush=True)

    fout = open(args.out, "a", encoding="utf-8")
    agg = {}
    for arm_name in arms_sel:
        fn = ARM_FNS[arm_name]
        for g in GOLD:
            for rep in range(args.reps):
                key = (arm_name, args.model, g["id"], rep)
                if key in done:
                    continue
                t0 = time.time()
                try:
                    gen = generate(fn, args.model, g["q"])
                    sp = gen["sparql"]
                    if arm_name in POST:
                        sp = POST[arm_name](sp, g["q"])
                    rows, err = run_sparql(sp)
                    v = score(rows, err, g)
                    gen["sparql"] = sp
                    rec = dict(arm=arm_name, model=args.model, id=g["id"], shape=g["shape"],
                               rep=rep, verdict=v, err=err, latency=gen["latency"],
                               prompt_chars=gen["prompt_chars"],
                               nrows=(len(rows) if rows is not None else 0),
                               sparql=gen["sparql"])
                except Exception as e:
                    rec = dict(arm=arm_name, model=args.model, id=g["id"], shape=g["shape"],
                               rep=rep, verdict="error", err=f"GEN {type(e).__name__}: {str(e)[:140]}",
                               latency=round(time.time() - t0, 1), prompt_chars=None, nrows=0, sparql="")
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if args.sleep:
                    time.sleep(args.sleep)
                agg.setdefault(arm_name, []).append(rec["verdict"])
                print(f"[{arm_name:>13}] {g['id']} r{rep} {rec['verdict']:5} "
                      f"({rec['latency']}s)  {rec.get('err') or ''}"[:120], flush=True)

    print("\n==== resumen de esta corrida ====")
    for arm_name, vs in agg.items():
        n = len(vs) or 1
        ok = vs.count("ok")
        print(f"{arm_name:>14}: ok {ok}/{n} ({100*ok/n:.0f}%)  "
              f"empty {vs.count('empty')}  wrong {vs.count('wrong')}  error {vs.count('error')}")


# tablas de acierto por modelo, brazo y forma de consulta
def analizar():
    IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "study_results.jsonl")

    recs = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    print(f"{len(recs)} generaciones\n")

    # --- por (modelo, arm): tasas + latencia + tamaño de prompt ---
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["model"], r["arm"])].append(r)

    def pct(vs, k): return 100 * vs.count(k) / max(1, len(vs))

    print("| Modelo | Arm | n | OK % | vacío % | mal % | error % | lat. med | prompt |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for (model, arm), rs in sorted(by.items()):
        vs = [r["verdict"] for r in rs]
        lat = [r["latency"] for r in rs if r.get("latency")]
        pc = [r["prompt_chars"] for r in rs if r.get("prompt_chars")]
        print(f"| {model} | {arm} | {len(rs)} | {pct(vs,'ok'):.0f} | {pct(vs,'empty'):.0f} | "
              f"{pct(vs,'wrong'):.0f} | {pct(vs,'error'):.0f} | "
              f"{(sum(lat)/len(lat) if lat else 0):.1f}s | {(sum(pc)//len(pc) if pc else 0)} |")

    # --- acierto por FORMA de consulta (mayoría de reps), arm E/D vs C ---
    print("\n### Acierto por forma de consulta (voto de mayoría entre reps)\n")
    shapes = sorted({r["shape"] for r in recs})
    arms = sorted({r["arm"] for r in recs})
    models = sorted({r["model"] for r in recs})
    for model in models:
        print(f"\n**{model}**\n")
        print("| forma | " + " | ".join(arms) + " |")
        print("|---|" + "---|" * len(arms))
        for sh in shapes:
            cells = []
            for arm in arms:
                # por id: mayoría
                byid = collections.defaultdict(list)
                for r in recs:
                    if r["model"] == model and r["arm"] == arm and r["shape"] == sh:
                        byid[r["id"]].append(r["verdict"])
                if not byid:
                    cells.append("-")
                    continue
                oks = sum(1 for vs in byid.values() if vs.count("ok") > len(vs) / 2)
                cells.append(f"{oks}/{len(byid)}")
            print(f"| {sh} | " + " | ".join(cells) + " |")

    # --- casos que NINGÚN arm resuelve nunca (para el capítulo de límites) ---
    print("\n### Casos nunca resueltos (0 OK en ninguna corrida)\n")
    allids = collections.defaultdict(list)
    for r in recs:
        allids[r["id"]].append(r["verdict"])
    for cid, vs in sorted(allids.items()):
        if "ok" not in vs:
            print(f"- {cid}: {collections.Counter(vs)}")


# las plantillas sin LLM (grafo/consulta/plantillas.py) con las preguntas de referencia
def plantillas():
    n_sparql = n_llm_needed = n_no_match = n_ok = n_wrong = n_empty = 0
    for g in GOLD:
        r = answer(g["q"])
        if r["sparql"] is None:
            if r["template"] is None:
                n_no_match += 1
                tag = "NO_MATCH"
            else:
                n_llm_needed += 1
                tag = f"NEEDS_LLM({r['template']})"
            print(f"{g['id']:5} {tag:28} sim={r['similarity']:.2f}  {g['q'][:55]}")
            continue
        n_sparql += 1
        rows, err = run_sparql(r["sparql"])
        v = score(rows, err, g)
        if v == "ok":
            n_ok += 1
        elif v == "wrong":
            n_wrong += 1
        else:
            n_empty += 1
        flag = "OK" if v == "ok" else v.upper()
        print(f"{g['id']:5} [{flag:5}] tmpl={r['template']:16} sim={r['similarity']:.2f}  slots={r['slots']}")

    tot = len(GOLD)
    print(f"\n== {tot} preguntas gold ==")
    print(f"plantilla generó SPARQL: {n_sparql}  (ok={n_ok} wrong={n_wrong} empty={n_empty})")
    print(f"forma reconocida pero SIN plantilla (fallback LLM): {n_llm_needed}")
    print(f"forma NO reconocida (fallback LLM): {n_no_match}")
    print(f"cobertura zero-LLM correcta: {n_ok}/{tot} ({100*n_ok/tot:.0f}%)")


ORDENES = {"ejecutar": ejecutar, "analizar": analizar, "plantillas": plantillas}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if len(sys.argv) < 2 or sys.argv[1] not in ORDENES:
        sys.exit("Uso: python evaluacion/estudio_sparql/estudio.py " + "|".join(ORDENES))
    ORDENES[sys.argv.pop(1)]()
