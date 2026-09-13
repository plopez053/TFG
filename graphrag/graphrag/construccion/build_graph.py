import os
import re
import sys
import time
import json
import argparse
import json_repair

HERE = os.path.dirname(os.path.abspath(__file__))            # .../graphrag/graphrag/construccion
GRAPHRAG = os.path.dirname(HERE)                               # .../graphrag/graphrag (datos + utils compartidos)
ROOT = os.path.dirname(os.path.dirname(GRAPHRAG))              # .../TFG/TFG (proyecto)
sys.path.insert(0, HERE)
sys.path.insert(0, GRAPHRAG)
sys.path.insert(0, ROOT)

# La rama Gemini NO importa backend.rag (que es quien carga el .env), así que
# aquí se carga explícitamente para ver GOOGLE_API_KEY / GROQ_API_KEY.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass
from grupos import normaliza_grupo, prop_id  # noqa: E402
from jsonl_utils import load_jsonl, iter_jsonl  # noqa: E402

PROPOSALS = os.path.join(GRAPHRAG, "proposals.jsonl")
ENRICHED = os.path.join(GRAPHRAG, "proposals_enriched.jsonl")

# Vocabulario controlado de temas → permite agregaciones limpias ("cuántas de vivienda por grupo")
# Debe coincidir con los prefLabel de nivel 1 de themes_skos.ttl (build_rdf.py normaliza
# acentos al emparejar, así que un desajuste aquí no rompería nada, pero mantenerlos
# iguales evita confundir a quien audite ambos ficheros).
TEMAS = [
    "vivienda", "urbanismo", "movilidad y transporte", "medio ambiente", "euskera",
    "cultura", "deporte", "educación", "igualdad y feminismo", "servicios sociales",
    "empleo y economía", "presupuestos y fiscalidad", "seguridad", "participación ciudadana",
    "turismo", "sanidad", "memoria histórica", "derechos humanos", "otros",
]

# normaliza_grupo y prop_id se importan de grupos.py (compartido con build_rdf.py).
# extrae_grupo (extracción del proponente desde el título/texto) NO se usa aquí:
# esta fase solo necesita normalizar el metadato `party` del orador; la extracción
# del proponente real ocurre en build_rdf.py, que sí importa extrae_grupo.

# ---------------------------------------------------------------------------
# Paso --enrich : extracción LLM por proposición
# ---------------------------------------------------------------------------
# Grupos municipales estándar del Pleno de Bilbao — el LLM debe usar estas formas
# exactas al nombrar grupos en "proponente_grupo", "votos_por_grupo" y "enmienda.por".
GRUPOS_ESTANDAR = (
    "EAJ-PNV", "PSE-EE", "EH BILDU", "PP", "ELKARREKIN BILBAO", "GOAZEN BILBAO",
    "UDALBERRI", "EZKER BATUA-IU", "ARALAR", "CIUDADANOS", "VOX",
    "EQUIPO DE GOBIERNO", "GRUPO MIXTO",
)

EXTRACT_PROMPT = """Eres un analista de actas del Pleno del Ayuntamiento de Bilbao. Del texto de
UNA proposición (incluye su parte dispositiva, el debate y la votación) devuelve SOLO este JSON,
sin texto antes ni después:

{{
  "resumen": "<1-2 frases: qué pide EXACTAMENTE la parte dispositiva, en tus palabras>",
  "tema_principal": "<UNO de: {temas}>",
  "temas": ["<2-4 etiquetas temáticas concretas, en minúscula; p.ej. 'alquiler', 'zona de bajas emisiones'>"],
  "resultado": "<aprobada | rechazada | decae | retirada | aprobada con enmienda | sin resultado>",
  "proponente_persona": "<nombre y apellidos del concejal/a que FIRMA la proposición si el texto lo dice; si no, null>",
  "votos_por_grupo": {{"favor": ["<grupo>"], "contra": ["<grupo>"], "abstencion": ["<grupo>"]}},
  "enmienda": null,
  "entidades": [{{"nombre": "<entidad>", "tipo": "persona|lugar|organizacion"}}]
}}

Si hubo enmienda, "enmienda" es un objeto:
  {{"por": "<grupo>", "tipo": "modificacion|adicion|sustitucion|transaccional",
    "resumen": "<1 frase: qué cambia>",
    "resultado": "<aceptada_por_proponente | aprobada_en_votacion | rechazada_en_votacion | sin_votar>"}}
  - "aceptada_por_proponente": el grupo proponente la asume y se vota el texto YA enmendado (no hay votación separada de la enmienda).
  - "aprobada_en_votacion" / "rechazada_en_votacion": SÍ hubo votación específica de la enmienda; si el acta da las cifras de ESA votación, ponlas en "enmienda.votos": {{"favor": N, "contra": N, "abstencion": N}} (enteros; omítelo si no constan).

Reglas:
- Grupos: usa EXACTAMENTE una de estas formas → {grupos}. No inventes variantes.
- "votos_por_grupo": SOLO lo que el acta diga explícitamente ("votan a favor los grupos...",
  "en contra el Grupo Popular", "se abstiene..."). Si el acta no desglosa el voto, deja las
  tres listas vacías. Un grupo va en UNA sola lista.
- "entidades": lo que la proposición trata o cita de forma sustantiva —
  lugares de Bilbao (barrios, calles, plazas, equipamientos: "Zorrotzaurre",
  "Guggenheim", "San Mamés"), organizaciones y empresas ("Metro Bilbao",
  "Iberdrola", "Osakidetza", "Gobierno Vasco"), y personas relevantes NO
  concejales (víctimas, artistas, cargos externos). Máx 12, nombre completo.
  NO incluyas: los concejales que debaten (van aparte), grupos políticos,
  "Ayuntamiento de Bilbao", ni órganos genéricos ("Pleno", "Junta de Gobierno",
  "Comisiones", "Área de...", "Servicio de...").
- "resultado": si el acta no lo dice claramente, "sin resultado". "decae" = la proposición
  original decae al aceptarse una enmienda que la sustituye.
- Responde SOLO el JSON, sin ```.

TEXTO:
{texto}
"""


# texto plano de una respuesta LangChain (content puede ser str o lista de bloques)
def _msg_text(msg) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                out.append(b.get("text") or b.get("content") or "")
        return "".join(out)
    return str(c)


def _parse_json(txt: str):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        pass
    # Fallback: los modelos locales pequeños a veces truncan una cadena de texto
    # sin cerrarla dentro de listas largas (proposiciones con muchas entidades)
    # -> JSON casi válido, roto en un punto. json_repair reconstruye el resto en
    # vez de descartar la proposición entera por un carácter.
    try:
        data = json_repair.loads(m.group(0))
        # Devolver {} tal cual (como hace json.loads arriba): enrich() distingue
        # data=None (irrecuperable) de data={} (válido pero vacío), esa decisión
        # es de quien llama, no de esta función.
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# caracteres de proposición que se pasan al LLM según el modelo
CTX_CHARS = {"gemini": 20000, "groq": 30000}

GEMINI_ALIAS = {"gemini": "gemini-3.5-flash-lite"}


def _gemini_name(model: str) -> str:
    return GEMINI_ALIAS.get(model, model)


# devuelve el LLM para la extracción
def get_llm(model: str):
    if model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        name = _gemini_name(model)
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Falta GOOGLE_API_KEY en el entorno / .env "
                             "(consíguela en https://aistudio.google.com/apikey)")
        return ChatGoogleGenerativeAI(model=name, temperature=0, google_api_key=key)
    if model == "groq":
        from langchain_groq import ChatGroq
        from backend.providers import LLM_MODEL_GROQ, GROQ_API_KEY
        return ChatGroq(model=LLM_MODEL_GROQ, api_key=GROQ_API_KEY, temperature=0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=0)


def _ctx_chars(model: str) -> int:
    for k, v in CTX_CHARS.items():
        if model.startswith(k):
            return v
    return 5000  # Ollama local: mantener el límite conservador de siempre


def enrich(limit=None, model="groq", rpm=None):
    llm = get_llm(model)
    print(f"[*] modelo de extracción: {model}")

    # think=False desactiva el bloque de pensamiento de los modelos "razonadores"
    # (qwen3); no lo necesita esta tarea. Ollama lo ignora si no lo soporta.
    invoke_kwargs = {"think": False} if "qwen3" in model.lower() else {}
    ctx = _ctx_chars(model)
    # pausa fija entre proposiciones para no chocar con el límite de peticiones/min
    pausa = (60.0 / rpm) if (rpm and model.startswith("gemini")) else 0.0

    proposals = load_jsonl(PROPOSALS)
    done = set()
    if os.path.exists(ENRICHED):
        for r in iter_jsonl(ENRICHED):
            done.add(r["id"])
    print(f"[*] {len(proposals)} proposiciones | ya enriquecidas: {len(done)} | "
          f"contexto: {ctx} chars | pausa: {pausa:.1f}s")

    out = open(ENRICHED, "a", encoding="utf-8")
    n = 0
    fallos = 0
    for p in proposals:
        pid = prop_id(p)
        if pid in done:
            continue
        if limit and n >= limit:
            break
        if pausa:
            time.sleep(pausa)
        prompt = EXTRACT_PROMPT.format(
            temas=", ".join(TEMAS), grupos=", ".join(GRUPOS_ESTANDAR), texto=p["text"][:ctx])
        # Hasta 3 reintentos por proposición (cortes de Ollama, JSON truncado).
        # data=None = no se pudo parsear JSON -> se reintenta y, si falla, se descarta.
        # data={} = JSON válido pero vacío (fragmento sin nada que clasificar) ->
        # se escribe con los valores por defecto.
        data = None
        ultimo_error = None
        for _intento in range(4):
            try:
                msg = llm.invoke(prompt, **invoke_kwargs)
                resp = _msg_text(msg)   # normaliza: content puede venir como lista
                data = _parse_json(resp)
            except Exception as e:
                ultimo_error = str(e)[:160]
                data = None
                # 429 / rate limit: espera y reintenta sin gastar un intento
                # (no es un fallo de la proposición).
                if re.search(r"429|rate.?limit|quota|resource.?exhausted", ultimo_error, re.I):
                    espera = 30 * (_intento + 1)
                    print(f"[~] rate limit; espero {espera}s...", flush=True)
                    time.sleep(espera)
                    continue
            if data is not None:
                break
            # Pequeña espera antes de reintentar (no tras el último intento):
            # sin ella, un corte puntual de Ollama recibía los 3 intentos en
            # milisegundos, sin dar tiempo a que el servicio volviera.
            if _intento < 3:
                time.sleep(3)
        if data is None:  # ni una sola vez se pudo parsear JSON tras 3 intentos → NO escribir
            detalle = f" (último error: {ultimo_error})" if ultimo_error else " (JSON no recuperable)"
            print(f"[!] {pid} descartada tras 3 intentos{detalle}")
            # Cada proposición fallida cuesta hasta 3 llamadas, así que el
            # umbral de "para por sospecha de cuota agotada" es 7 (~21 llamadas),
            # no 20.
            fallos += 1
            if fallos >= 7:
                print("[!] 7 proposiciones seguidas sin éxito -- hasta 21 llamadas API "
                      "(¿cuota agotada?). Paro; reanuda más tarde.")
                break
            continue
        fallos = 0
        vpg = data.get("votos_por_grupo") or {}
        rec = {
            "id": pid,
            "date": p["date"],
            "topic": p["topic"],
            "grupo": normaliza_grupo(p["party"]),
            "vote_result": p["vote_result"],
            "page_ini": p["page_ini"],
            "source": p["source"],
            "tema_principal": (data.get("tema_principal") or "otros").lower().strip(),
            "temas": data.get("temas") or [],
            "resultado": (data.get("resultado") or "sin resultado").lower().strip(),
            "entidades": data.get("entidades") or [],
            # --- campos de la pasada de enriquecimiento rica ---
            "resumen": (data.get("resumen") or "").strip(),
            "proponente_persona": (data.get("proponente_persona") or None),
            "votos_por_grupo": {
                "favor": vpg.get("favor") or [],
                "contra": vpg.get("contra") or [],
                "abstencion": vpg.get("abstencion") or [],
            } if isinstance(vpg, dict) else {"favor": [], "contra": [], "abstencion": []},
            "enmienda": data.get("enmienda") if isinstance(data.get("enmienda"), dict) else None,
            "oradores": p.get("oradores") or [],
            "votos_nominales": p.get("votos_nominales"),  # capa determinista (passthrough)
            "model": model,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        n += 1
        if n % 25 == 0:
            print(f"    enriquecidas {n} (última: {p['date']} · {rec['tema_principal']})")
    out.close()
    print(f"[+] +{n} enriquecidas -> {ENRICHED}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gemini", help="gemini, groq o un modelo de Ollama")
    ap.add_argument("--rpm", type=int, default=8, help="peticiones/min máximas (solo Gemini)")
    args = ap.parse_args()
    if args.enrich:
        enrich(limit=args.limit, model=args.model, rpm=args.rpm)
    else:
        print("Usa --enrich")
