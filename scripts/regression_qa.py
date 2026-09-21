# -*- coding: utf-8 -*-
import sys, os, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

RETRIES = 3


# todos los valores de todas las filas, como strings
def _flat(rows):
    out = []
    for r in rows or []:
        out += [str(v) for v in (r.values() if isinstance(r, dict) else r)]
    return out


# ---------------------------------------------------------------------------
# GraphRAG — se comprueba `rows` (resultado del SPARQL), no la narración.
# `expect`: valores que DEBEN aparecer en las filas.
# `forbid`: valores que NO deben aparecer (típicamente la cifra falsa antigua).
# `top`: si es ranking, el primer valor de la primera fila.
# ---------------------------------------------------------------------------
# Baselines recalculados 2026-09-09 tras el re-enriquecimiento con Gemini
# (3923 -> 3422 proposiciones: se limpiaron ~500 fragmentos basura de 2007-2016;
#  clasificación temática y de resultado distinta = modelo Gemini vs qwen3 +
#  texto completo 30k vs 6k). Verdad calculada con scratchpad/ground_truth.py.
#
# Baselines de RESULTADO (aprobada/rechazada/aprobada con enmienda) recalculados
# 2026-09-11 tras arreglar resultado_cruzado() en build_rdf.py (ver
# memoria/decisiones_tecnicas.md §4.5): el LLM clasificaba sistemáticamente mal
# "decae" como "aprobada con enmienda" o "rechazada"; la cifra vieja no era la
# verdad del acta, era el bug. La cifra de PRESENTADAS (total, sin filtrar por
# resultado) no cambia — solo bajan rechazadas/aprobadas al reclasificarse ~630
# proposiciones a "decae".
#
# 7 baselines recalculados 2026-09-16 (PP 2015, medio ambiente top3, vivienda
# por grupo, movilidad EH Bildu, % vivienda/tasa aprobación EH Bildu vs PP,
# concejal firmante, Otxandiano): NO eran no-determinismo del LLM como se creyó
# el 2026-09-15 — verificado con consulta SPARQL independiente Y con DOS
# generadores distintos (qwen3:8b local y Groq openai/gpt-oss-120b) dando
# exactamente el mismo resultado, distinto del baseline viejo. El grafo había
# derivado ligeramente desde que se fijaron esos baselines (pequeños ajustes de
# clasificación en rondas posteriores). Ground truth en
# scratchpad/ (sesión 2026-09-16), medio ambiente PP/EH BILDU quedan
# EMPATADOS a 60 (se quitó el "top" fijo de ese caso).
#
# 6 baselines recalculados 2026-09-16 (misma sesión, más tarde, Ronda 40) tras
# un FIX REAL DE DATOS, no una recalibración de test: el usuario pidió
# verificar que estos baselines eran fieles a los PDFs de verdad, y al abrir
# la página real de un caso (PP 2015, ítem 49, 24-09-2015) se encontró que
# resultado_cruzado() (build_rdf.py) usaba \bdecae\b (singular) y se perdía
# "decaen" (plural, "...por lo que decaen tanto la enmienda de UDALBERRI como
# la proposición del PP") en 51 casos reales del corpus — Y que
# extract_proposals.py se quedaba con el PRIMER vote_result no vacío de cada
# topic en vez del ÚLTIMO, capturando el voto de una enmienda intermedia en
# vez del resultado final decisivo en proposiciones con varias enmiendas
# competidoras (279 registros con vote_result distinto tras el fix). Ambos
# corregidos, corpus reextraído (extract_proposals.py, sin LLM, ~5-10 min) y
# grafo reconstruido (build_rdf.py). Los números de PRESENTADAS no cambian
# (nadie se crea ni se destruye una proposición); solo bajan/suben aprobadas/
# rechazadas al reclasificarse correctamente a "decae". PP 2015 aprobadas
# 13->12, rechazadas 2021 9->10, EH Bildu vivienda aprobadas 9->5, año más
# rechazadas 2018 sigue siendo el año pero 48->46, movilidad EH Bildu
# rechazadas 8->10, tasa aprobación EH Bildu/PP 646/85+978/168 -> 646/73+978/150.
# LIMITACIÓN CONOCIDA, no arreglada esta sesión: en al menos 1 caso concreto
# (el propio PP 2015 ítem 49) el texto decisivo real queda atribuido al TEMA
# SIGUIENTE (ítem 50) por un corte de página/segmento en backend/rag.py que
# no coincide con el marcador real "-49-/-50-" del acta — ninguna de las dos
# correcciones de arriba puede arreglar esto porque el texto correcto nunca
# llega a asociarse al topic correcto en el chunking. Alcance no cuantificado
# (probablemente raro: solo afecta a proposiciones cuyo párrafo de resultado
# final cae justo en el salto de página que coincide con el marcador de tema
# siguiente). Ver graphrag/graphrag/construccion/extract_proposals.py y
# build_rdf.py (_RE_DECAE_PROP) para el detalle de ambos fixes.
#
# 2 baselines recalculados 2026-09-17 (Ronda 42) tras arreglar el bug de
# "segmentación" que quedaba documentado como límite conocido justo arriba
# (85-82/3422, ~2,4%). La causa real NO era un corte de página (esa hipótesis,
# investigada a fondo en la Ronda 40, resultó incorrecta): era un hueco de
# vocabulario en `result_re` (backend/rag.py) — la palabra "decae" NUNCA
# formaba parte de ninguna rama del regex (solo "se acepta/aprueba/...",
# "queda ...", "resulta ..."), así que un desenlace real y frecuente como "El
# Pleno municipal acepta la Enmienda..., sin necesidad de someterla a
# votación, por lo que decae la proposición..." no disparaba nada. Además,
# cuando SÍ se capturaba el texto de resultado pero sin cifras de voto (caso
# típico de "decae"/"retirada" sin votación formal), se descartaba entero
# salvo que mencionara literalmente "unanimidad"/"asentimiento" — perdiendo el
# dato igualmente. Fix: rama "decaen?\b[^.]{0,400}" añadida a `result_re` +
# `_STRONG_RESULT_RE` que ya no exige unanimidad/asentimiento para conservar
# un resultado_text que venga de un marcador formal de cierre de acta (queda/
# resulta/decae). Verificado con `scratchpad/scan_segmentacion_v2.py`: 82->25
# discrepancias (-70%); 9 casos concretos verificados a mano contra el PDF
# real (25-11-2010 puntos 17/19, 28-11-2007 punto 23, 18-06-2008 punto 22,
# 27-11-2008 punto 22, 25-02-2010 puntos 15/16, 27-05-2010 punto 31), todos
# correctos tras el fix. Re-extracción completa (236 actas, sin LLM) + merge
# por id en proposals_enriched.jsonl (292 registros con vote_result distinto)
# + grafo reconstruido. Los 2 baselines que cambiaron se verificaron con
# consulta SPARQL independiente contra el grafo reconstruido, no solo con la
# propia regresión: EH Bildu vivienda aprobadas 5->3 (Aprobada 2 +
# AprobadaConEnmienda 1, total tema sin cambios en 63), tasa aprobación EH
# Bildu 71->49 y PP 148->128 (totales presentados sin cambios: 646/978) —
# ambos bajan porque proposiciones que antes se contaban como "aprobada"/
# "aprobada con enmienda" en realidad habían decaído tras aceptarse una
# enmienda sin necesidad de votación, y ahora se clasifican correctamente
# como "decae". Los 25/3422 (0,7%) restantes SÍ son la causa más profunda ya
# diagnosticada (desalineación segmento/chunk en el indexador compartido,
# verificada de nuevo en 2 casos: 28-03-2012 puntos 14/15, texto de un punto
# se filtra al chunk fusionado del punto vecino cruzando un salto de página) —
# sigue fuera de alcance por el mismo motivo que antes (tocar el indexador
# compartido por GraphRAG y el RAG vectorial). Detalle completo en memoria.
GRAPH_CASES = [
    dict(q="¿Cuántas proposiciones sobre movilidad y transporte se presentaron en 2019?",
         expect=[], forbid=["0"], answer_has="2019"),  # 18 (trataSobre) o 24 (rollup); solo exigimos que NO dé 0
    dict(q="¿Cuántas proposiciones presentó el Partido Popular en 2015 y cuántas se aprobaron?",
         expect=["38", "11"], forbid=[]),
    dict(q="¿Cuántas proposiciones sobre euskera se han rechazado en total?",
         expect=["2"], forbid=["55", "4"]),
    dict(q="¿Qué tres grupos han presentado más proposiciones sobre medio ambiente?",
         expect=["EH BILDU"]),  # PP y EH BILDU empatan a 60 (verificado 2026-09-16) -> sin "top" fijo, el orden entre empatados es válido en cualquier sentido
    dict(q="¿En qué año se presentaron más proposiciones sobre seguridad?",
         expect=["2025", "36"], forbid=[]),
    dict(q="¿Cuántas proposiciones se rechazaron en 2021?",
         expect=["10"], forbid=[]),
    dict(q="¿Qué grupo ha presentado más proposiciones sobre educación?",
         expect=["EH BILDU"], top="EH BILDU"),
    dict(q="¿Cuántas proposiciones sobre presupuestos y fiscalidad se han presentado en total, incluyendo subtemas?",
         expect=["733"], forbid=["0"]),
    dict(q="¿Cuántas proposiciones sobre vivienda ha presentado cada grupo?",
         expect=["EH BILDU", "63"], top="EH BILDU"),
    dict(q="¿Cuántas proposiciones aprobó EH Bildu sobre vivienda?",
         expect=["3"], forbid=[]),
    dict(q="¿Cuántas proposiciones hay en total en el grafo?",
         expect=["3422"], forbid=["3923"]),
    # --- patrones analíticos ---
    dict(q="¿Cuántas proposiciones sobre desahucios se han presentado?",
         expect=["14"], forbid=[]),
    dict(q="¿En qué año se rechazaron más proposiciones en el Pleno de Bilbao?",
         expect=["2018", "46"], forbid=[]),
    dict(q="¿Cuántas proposiciones sobre movilidad ha presentado EH Bildu y cuántas se han rechazado?",
         expect=["122", "10"], forbid=[]),
    dict(q="¿Qué porcentaje de las proposiciones sobre vivienda se han aprobado?",
         expect=["200"], forbid=["0"]),  # el total del tema (200) es lo estable; el % lo calcula el LLM y varía
    dict(q="¿Qué grupo tiene mejor tasa de aprobación de sus proposiciones, EH Bildu o el PP?",
         expect=["646", "49", "978", "128"], forbid=[]),
    dict(q="¿Cuántas proposiciones conjuntas entre varios grupos ha habido?",
         graceful=True),  # el grafo no lo distingue -> debe degradar sin romper
    # --- capa de personas / enmiendas ---
    dict(q="¿Qué concejal o concejala ha presentado (firmado) más proposiciones?",
         expect=["Cristina Ruiz Bujedo", "79"], top="Cristina Ruiz Bujedo"),
    dict(q="¿En cuántos debates del Pleno ha intervenido Xabier Otxandiano?",
         expect=["56"], forbid=["0"]),
    dict(q="¿Cuántas enmiendas ha presentado el Partido Popular en total?",
         expect=["24"], forbid=["936"]),
    dict(q="¿Quién ha sido alcalde de Bilbao en el periodo de las actas?",
         expect=["Juan Maria Aburto Rique"], forbid=[]),
    # --- capa determinista de voto (parseado del acta) ---
    dict(q="¿Qué concejal o concejala ha votado más veces en contra de las proposiciones?",
         expect=["Yolanda Diez Saiz"], top="Yolanda Diez Saiz"),
]


def run_graph():
    from graphrag.graphrag.graph_rag_sparql import graph_answer
    passed = 0
    for c in GRAPH_CASES:
        ok, detail, tries = False, "", 0
        for attempt in range(RETRIES):
            tries = attempt + 1
            try:
                res = graph_answer(c["q"], verbose=False)
                if c.get("graceful"):
                    # solo exige que no reviente y dé una respuesta no vacía
                    ok = bool((res.get("answer") or "").strip())
                    detail = "respuesta vacía" if not ok else ""
                    if ok:
                        break
                    continue
                vals = _flat(res["rows"])
                miss = [e for e in c.get("expect", []) if e not in vals]
                bad = [f for f in c.get("forbid", []) if f in vals]
                top_ok = True
                if c.get("top"):
                    first = res["rows"][0] if res["rows"] else {}
                    fv = list(first.values()) if isinstance(first, dict) else list(first)
                    top_ok = any(c["top"] in str(x) for x in fv)
                ans_ok = (c["answer_has"].lower() in (res.get("answer") or "").lower()) if c.get("answer_has") else True
                if not miss and not bad and top_ok and ans_ok:
                    ok = True
                    break
                detail = f"faltan={miss} prohibidos={bad} top_ok={top_ok} answer_has_ok={ans_ok} filas={res['rows'][:3]}"
            except Exception as e:
                detail = f"EXCEPCION {type(e).__name__}: {e}"
        mark = "PASA" if ok else "FALLA"
        extra = f" (reintentos: {tries})" if ok and tries > 1 else ""
        passed += ok
        print(f"  [{mark}]{extra} {c['q'][:62]}")
        if not ok:
            print(f"         -> {detail}")
    print(f"\nGraphRAG: {passed}/{len(GRAPH_CASES)}")
    return passed, len(GRAPH_CASES)


# ---------------------------------------------------------------------------
# RAG vectorial — se comprueba la RECUPERACIÓN: qué plenos entran al contexto.
# `expect_dates`: al menos uno de estos plenos debe estar entre los docs finales.
# `min_dates`: nº mínimo de plenos distintos recuperados (0 = solo "no vacío").
# `must_not_be_empty`: el contexto no puede quedar vacío (falso "no encontrado").
# ---------------------------------------------------------------------------
VECTOR_CASES = [
    dict(q="¿Qué se dijo sobre la estabilización y la OPE del empleo público municipal del Ayuntamiento de Bilbao?",
         blob_has="estabiliza", must_not_be_empty=True),
    dict(q="¿Qué iniciativas ha habido sobre las bibliotecas municipales y los equipamientos culturales de barrio?",
         blob_has="biblioteca", min_dates=3, must_not_be_empty=True),
    dict(q="¿Qué propuestas se han hecho sobre la situación de las personas sin hogar en Bilbao?",
         blob_has="sin hogar", min_dates=3, must_not_be_empty=True),
    dict(q="¿Qué se ha debatido sobre la memoria de las víctimas del terrorismo en el Pleno de Bilbao?",
         blob_has="terrorismo", expect_dates=["31-01-2019", "30-06-2022", "24-11-2015", "26-09-2019"],
         must_not_be_empty=True),
    dict(q="¿Qué se debatió sobre el tranvía en Bilbao?",
         blob_has="tranv", min_dates=3, must_not_be_empty=True),
    dict(q="¿Qué propuso el Partido Popular sobre aparcamientos?",
         blob_has="aparcamiento", min_dates=2, must_not_be_empty=True),
    dict(q="¿Qué ocurrió en el pleno del 26-01-2023?",
         expect_dates=["26-01-2023"], must_not_be_empty=True),
    dict(q="¿Qué se debatió sobre los desahucios en Bilbao?",
         blob_has="desahuci", must_not_be_empty=True),
    dict(q="¿Qué propuestas sobre vivienda social ha habido?",
         blob_has="vivienda", min_dates=5, must_not_be_empty=True),
    # --- subtema sin canónico propio + años recientes ---
    dict(q="¿Qué se ha debatido sobre la calidad del aire y la Zona de Bajas Emisiones en Bilbao?",
         blob_has="emisiones", min_dates=4, must_not_be_empty=True,
         # al menos un pleno reciente (>=2019): el fallo era que solo
         # salía contenido pre-2016
         recent_date=2019),
    dict(q="¿Qué iniciativas ha habido sobre la peatonalización de calles en Bilbao?",
         min_dates=4, must_not_be_empty=True),
    # --- mención única de un nombre propio (empresa) ---
    dict(q="¿Qué se ha dicho sobre la empresa Tubacex en el Pleno de Bilbao?",
         expect_dates=["27-05-2021"], blob_has="tubacex", must_not_be_empty=True),
    dict(q="¿Qué se ha debatido sobre Petronor en Bilbao?",
         blob_has="petronor", min_dates=2, must_not_be_empty=True),
]


def run_vector():
    from backend.rag import RAGPipeline, CHROMA_PATH
    from langchain_chroma import Chroma
    rag = RAGPipeline()
    rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)
    passed = 0
    for i, c in enumerate(VECTOR_CASES):
        ok, problems, tries = False, [], 0
        for attempt in range(RETRIES):
            tries = attempt + 1
            # La recuperación tiene un paso con LLM (MultiQuery) no determinista;
            # y la trial key de Cohere son 10 llamadas/min -> pausa entre casos.
            if i or attempt:
                time.sleep(8)
            try:
                cand, exact = rag._retrieve_and_rank(c["q"], k=80)
                docs = rag._select_final_docs(cand, c["q"], exact)
                dates = sorted({d.metadata.get("date", "") for d in docs if d.metadata.get("date")})
                blob = " ".join(d.page_content.lower() for d in docs)
                problems = []
                if c.get("must_not_be_empty") and not docs:
                    problems.append("contexto VACIO")
                if c.get("expect_dates") and not any(x in dates for x in c["expect_dates"]):
                    problems.append(f"ninguno de {c['expect_dates']} en {dates[:8]}")
                if c.get("min_dates") and len(dates) < c["min_dates"]:
                    problems.append(f"{len(dates)} plenos < {c['min_dates']}")
                if c.get("blob_has") and c["blob_has"].lower() not in blob:
                    problems.append(f"'{c['blob_has']}' no aparece en el contexto")
                if c.get("recent_date"):
                    years = [int(d[-4:]) for d in dates if len(d) >= 4 and d[-4:].isdigit()]
                    if not any(y >= c["recent_date"] for y in years):
                        problems.append(f"ningún pleno >= {c['recent_date']} (años: {sorted(years)})")
                if not problems:
                    ok = True
                    break
            except Exception as e:
                problems = [f"EXCEPCION {type(e).__name__}: {e}"]
        passed += ok
        extra = f" (reintentos: {tries})" if ok and tries > 1 else ""
        print(f"  [{'PASA' if ok else 'FALLA'}]{extra} {c['q'][:62]}")
        if not ok:
            print(f"         -> {' ; '.join(problems)}")
    print(f"\nRAG vectorial: {passed}/{len(VECTOR_CASES)}")
    return passed, len(VECTOR_CASES)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    tot_p = tot_n = 0
    if which in ("all", "graph"):
        print("=== GraphRAG ===")
        p, n = run_graph(); tot_p += p; tot_n += n
    if which in ("all", "vector"):
        print("\n=== RAG vectorial ===")
        p, n = run_vector(); tot_p += p; tot_n += n
    print(f"\n{'='*60}\nTOTAL: {tot_p}/{tot_n}   ({time.time()-t0:.0f}s)")
    sys.exit(0 if tot_p == tot_n else 1)
