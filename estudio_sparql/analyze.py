# -*- coding: utf-8 -*-
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
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
