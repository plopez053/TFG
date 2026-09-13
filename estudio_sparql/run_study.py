# -*- coding: utf-8 -*-
import os, sys, json, argparse, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "graphrag", "graphrag"))

from gold import GOLD
from runner import generate, run_sparql, score

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

# import arms DESPUÉS de fijar la key (F puede no existir aún)
import arms as A
ARM_FNS = dict(A.ARMS)
POST = {}
try:
    import arms_f
    ARM_FNS.update(arms_f.ARMS_F)
    POST = dict(getattr(arms_f, "POST", {}))
except Exception as e:
    print(f"[arms_f no disponible: {e}]", flush=True)
try:
    import arms_embed
    ARM_FNS.update(arms_embed.ARMS_EMBED)
    POST.update(getattr(arms_embed, "POST_EMBED", {}))
except Exception as e:
    print(f"[arms_embed no disponible: {e}]", flush=True)

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
