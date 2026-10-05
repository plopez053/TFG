# -*- coding: utf-8 -*-
"""
Evaluación RAGAS del RAG vectorial, fase 1: genera el dataset (pregunta,
contexto recuperado y respuesta) con el mismo camino que la interfaz
(retrieve_context + build_answer_prompt). La fase 2 (puntuar.py) calcula las métricas.

Narrador: Gemini (gemini-3.5-flash-lite, Vertex) en las 50. Es el respaldo de
Groq en la app y, en la práctica, el que responde: en la novena medida narró 19
de 20 respuestas del vectorial, porque Groq (gpt-oss-120b) tiene un tope de
200.000 tokens al día y de 8.000 por minuto, y con el contexto de 20.000
caracteres algunas preguntas no caben (error 413). Con un solo narrador el
dataset es homogéneo.

  - respuesta = la narración del LLM, sin el pie de fuentes ni el aviso de perfil
    (texto que no genera el LLM); la respuesta de la interfaz se guarda aparte;
  - cada entrada de retrieved_contexts es un bloque "[PLENO: fecha] tema" entero;
  - se anota qué LLM respondió y qué reranker se usó en cada pregunta.

Reanudable: las preguntas ya guardadas se saltan; si Gemini falla tres veces
en una pregunta se deja sin generar y basta relanzar.

Uso (entorno principal):
    python evaluacion/ragas/generar.py [--solo N,M,...]
Después, en el entorno de RAGAS:
    .venv-ragas\\Scripts\\python.exe evaluacion\\ragas\\puntuar.py
"""
import sys, os, re, json, time, glob, hashlib, subprocess

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import comun.proveedores as P
import vectorial.recuperacion as R
from vectorial.generacion import componer_respuesta, limpiar_respuesta
from vectorial.pipeline import get_rag


# huella del código y de los datos del grafo con los que se genera el conjunto
CODIGO = sorted(os.path.relpath(f, ROOT).replace(os.sep, "/") for d in ("comun", "vectorial", "grafo", "frontend")
                for f in glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True)) + [
    "datos/grafo/ontology.ttl", "datos/grafo/themes_skos.ttl", "datos/grafo/bilbao_reasoned.ttl"]


def huella():
    out = {}
    for f in CODIGO:
        with open(os.path.join(ROOT, f), "rb") as fh:
            out[f] = hashlib.sha256(fh.read()).hexdigest()[:16]
    try:
        out["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        out["git_commit"] = None
    return out


# Las 50 preguntas; sus respuestas de referencia, escritas y verificadas a mano contra las actas,
# están en referencias.json.
# 1-13: las de VECTOR_CASES en evaluacion/regresion.py. El orden debe coincidir
# con las claves de referencias.json.
QUESTIONS = [
    "¿Qué se dijo sobre la estabilización y la OPE del empleo público municipal del Ayuntamiento de Bilbao?",
    "¿Qué iniciativas ha habido sobre las bibliotecas municipales y los equipamientos culturales de barrio?",
    "¿Qué propuestas se han hecho sobre la situación de las personas sin hogar en Bilbao?",
    "¿Qué se ha debatido sobre la memoria de las víctimas del terrorismo en el Pleno de Bilbao?",
    "¿Qué se debatió sobre el tranvía en Bilbao?",
    "¿Qué propuso el Partido Popular sobre aparcamientos?",
    "¿Qué ocurrió en el pleno del 26-01-2023?",
    "¿Qué se debatió sobre los desahucios en Bilbao?",
    "¿Qué propuestas sobre vivienda social ha habido?",
    "¿Qué se ha debatido sobre la calidad del aire y la Zona de Bajas Emisiones en Bilbao?",
    "¿Qué iniciativas ha habido sobre la peatonalización de calles en Bilbao?",
    "¿Qué se ha dicho sobre la empresa Tubacex en el Pleno de Bilbao?",
    "¿Qué se ha debatido sobre Petronor en Bilbao?",
    # 14-50: temas con pocas proposiciones, para poder verificar la referencia
    "¿Qué se ha debatido en el Pleno sobre el Tour de Francia en Bilbao?",
    "¿Qué propuestas se han presentado sobre la pobreza energética?",
    "¿Qué se ha debatido sobre los patinetes eléctricos y los vehículos de movilidad personal?",
    "¿Qué iniciativas ha habido sobre el ruido y la contaminación acústica en Bilbao?",
    "¿Qué se ha debatido sobre la prostitución en Bilbao?",
    "¿Qué propuestas ha habido sobre el Mercado de la Ribera?",
    "¿Qué propuso el Partido Popular sobre la okupación de viviendas?",
    "¿Qué se ha debatido sobre la emergencia climática y el cambio climático?",
    "¿Qué se ha propuesto sobre la lucha contra la droga y las adicciones?",
    "¿Qué se ha debatido sobre las corridas de toros y la tauromaquia en Bilbao?",
    "¿Qué iniciativas ha habido sobre la acogida de personas refugiadas?",
    "¿Qué se ha debatido en el Pleno sobre Palestina y Gaza?",
    "¿Qué se ha debatido sobre la monarquía y la Casa Real en el Pleno de Bilbao?",
    "¿Qué subvenciones o acuerdos ha habido relacionados con el Athletic Club?",
    "¿Qué se ha debatido sobre el servicio de bomberos de Bilbao?",
    "¿Qué propuestas ha habido sobre el arbolado urbano y la tala de árboles?",
    "¿Qué se ha debatido sobre la regulación de las terrazas de hostelería?",
    "¿Qué se ha debatido sobre el Museo Guggenheim en el Pleno?",
    "¿Qué propuestas ha habido sobre los polideportivos municipales?",
    "¿Qué propuso EH Bildu sobre los presos y la política de dispersión?",
    "¿Qué medidas se debatieron en el Pleno sobre la pandemia de COVID-19?",
    "¿Qué iniciativas ha habido en defensa de los derechos del colectivo LGTBI?",
    "¿Qué se ha debatido sobre el Teatro Arriaga?",
    "¿Qué propuestas ha habido sobre memoria histórica y símbolos franquistas?",
    "¿Qué se ha debatido sobre el Consorcio de Aguas Bilbao Bizkaia?",
    "¿Qué se ha debatido sobre los cementerios municipales?",
    "¿Qué apoyo ha dado el Ayuntamiento de Bilbao al pueblo saharaui?",
    "¿Qué propuestas se han hecho sobre el barrio de Otxarkoaga?",
    "¿Qué se ha debatido sobre Azkuna Zentroa (antigua Alhóndiga)?",
    "¿Qué propuso el Partido Popular sobre Aste Nagusia?",
    "¿Qué se ha propuesto sobre ascensores y elementos mecánicos para mejorar la accesibilidad en los barrios?",
    "¿Qué iniciativas ha habido sobre energía solar y energías renovables?",
    "¿Qué se ha debatido sobre el sector del taxi en Bilbao?",
    "¿Qué propuestas ha habido sobre el servicio de Bizkaibus en Bilbao?",
    "¿Qué se ha propuesto sobre la salud mental en Bilbao?",
    "¿Qué ocurrió en el pleno del 17-08-2012?",
    "¿Qué ocurrió en el pleno del 25-06-2019?",
]

OUT_PATH = os.path.join(HERE, "dataset.json")
META_PATH = os.path.join(HERE, "dataset_meta.json")
PAUSA = 10          # Cohere trial: 10 llamadas/min
REINTENTOS = 3

_reranker = []      # qué reranker ordenó los candidatos en la pregunta actual
_rerank_local_original = P.rerank_local


def _rerank_local_anotado(*a, **kw):
    _reranker.append("local")
    return _rerank_local_original(*a, **kw)


P.rerank_local = _rerank_local_anotado  # rerank() lo llama por su nombre global
_rerank_original = R._cohere_rerank


def _rerank_anotado(*a, **kw):
    n = len(_reranker)
    r = _rerank_original(*a, **kw)
    if len(_reranker) == n:
        _reranker.append("cohere")
    return r


R._cohere_rerank = _rerank_anotado


class Anotado:
    """Envuelve un LLM de la cadena del proveedor para anotar si fue él quien respondió."""
    def __init__(self, llm, nombre, log):
        self._llm, self._nombre, self._log = llm, nombre, log

    def invoke(self, prompt, *a, **kw):
        r = self._llm.invoke(prompt, *a, **kw)
        self._log.append(self._nombre)
        return r

    async def ainvoke(self, prompt, *a, **kw):
        r = await self._llm.ainvoke(prompt, *a, **kw)
        self._log.append(self._nombre)
        return r

    def __getattr__(self, nombre):
        return getattr(self._llm, nombre)


def bloques(contexto: str) -> list:
    return [b.strip() for b in re.split(r"\n\s*\n(?=\[PLENO: )", contexto) if b.strip()]


def main():
    solo = None
    if "--solo" in sys.argv:
        solo = {int(x) for x in sys.argv[sys.argv.index("--solo") + 1].split(",")}

    codigo = huella()
    dataset = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            dataset = json.load(f)
        with open(META_PATH, encoding="utf-8") as f:
            if json.load(f)["codigo"] != codigo:
                sys.exit("[!] El código ha cambiado desde la corrida anterior: no se mezclan resultados.")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"inicio": time.strftime("%Y-%m-%d %H:%M"), "codigo": codigo,
                   "narrador": f"Gemini/{P.LLM_MODEL_GEMINI}"}, f, indent=2)

    rag = get_rag()
    prov = rag._llm_provider
    gemini = [(n, l) for n, l in prov._cadena() if n.startswith("Gemini")]
    if not gemini:
        sys.exit("[!] Gemini no está disponible: no se genera.")
    # Gemini como único narrador; primary_name sigue siendo de la nube, así que
    # el límite de contexto es el mismo que en la app (20.000 caracteres)
    (prov.primary_name, prov.primary), = gemini
    prov.fallback = prov.fallback2 = prov.fallback_name = prov.fallback2_name = None
    print(f"[*] Narrador: {prov.primary_name}; límite de contexto {rag._context_char_limit()}")
    narradores = []
    for attr in ("primary", "fallback", "fallback2"):
        if getattr(prov, attr) is not None:
            setattr(prov, attr, Anotado(getattr(prov, attr), getattr(prov, attr + "_name"), narradores))

    primera = True
    for i, q in enumerate(QUESTIONS, start=1):
        if str(i) in dataset or (solo and i not in solo):
            continue
        for intento in range(1, REINTENTOS + 1):
            if not primera:
                time.sleep(PAUSA)
            primera = False
            print(f"[{i}/{len(QUESTIONS)}] {q}" + (f" (intento {intento})" if intento > 1 else ""))
            _reranker.clear()
            ctx = rag.retrieve_context(q)
            if _reranker and _reranker[-1] == "local" and not P._cohere_agotado and intento < REINTENTOS:
                print("    [~] Cohere no respondió (límite por minuto); se repite la pregunta")
                time.sleep(30)
                continue

            narrador = None
            if not ctx["context"].strip():
                narracion = app = ("Lo siento, no he encontrado información relevante en las actas para esta "
                                   "pregunta. Prueba a reformularla.")
            else:
                narradores.clear()
                try:
                    resp = prov.invoke(rag.build_answer_prompt(ctx))
                except Exception as e:
                    print(f"    [!] LLM: {type(e).__name__}: {str(e)[:200]}")
                    if intento < REINTENTOS:
                        time.sleep(60)
                        continue
                    print("    [!] Se deja sin generar; relanzar para reintentarla.")
                    break
                narrador = narradores[-1] if narradores else None
                narracion = limpiar_respuesta(P.texto_llm(resp))
                app = componer_respuesta(P.texto_llm(resp), ctx)
            dataset[str(i)] = {
                "question": q,
                "retrieved_contexts": bloques(ctx["context"]),
                "response": narracion,
                "response_app": app,
                "dates_retrieved": ctx["unique_dates"],
                "narrador": narrador,
                "reranker": _reranker[-1] if _reranker else None,
            }
            print(f"    -> {len(dataset[str(i)]['retrieved_contexts'])} bloques ({len(ctx['context'])} chars), "
                  f"{len(ctx['unique_dates'])} plenos, respuesta {len(narracion)} chars, "
                  f"narrador {narrador}, reranker {dataset[str(i)]['reranker']}")
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(dict(sorted(dataset.items(), key=lambda kv: int(kv[0]))), f, ensure_ascii=False, indent=2)
            break

    faltan = [i for i in range(1, len(QUESTIONS) + 1) if str(i) not in dataset]
    print(f"\nGuardado en {OUT_PATH}. Generadas {len(dataset)}/{len(QUESTIONS)}"
          + (f"; faltan {faltan}" if faltan else ""))


if __name__ == "__main__":
    main()
