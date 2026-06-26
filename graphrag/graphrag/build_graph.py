"""Fase 2: enriquece cada proposición con un LLM (extrae tema, subtemas, resultado, entidades).

Uso:
  python graphrag/build_graph.py --enrich
  python graphrag/build_graph.py --enrich --model qwen2.5:7b   # modelo local Ollama
"""
import os
import re
import sys
import json
import argparse
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
PROPOSALS = os.path.join(HERE, "proposals.jsonl")
ENRICHED = os.path.join(HERE, "proposals_enriched.jsonl")

# Vocabulario controlado de temas → permite agregaciones limpias ("cuántas de vivienda por grupo")
TEMAS = [
    "vivienda", "urbanismo", "movilidad y transporte", "medio ambiente", "euskera",
    "cultura", "deporte", "educacion", "igualdad y feminismo", "servicios sociales",
    "empleo y economia", "presupuestos y fiscalidad", "seguridad", "participacion ciudadana",
    "turismo", "sanidad", "memoria historica", "derechos humanos", "otros",
]

# ---------------------------------------------------------------------------
# Normalización de grupos (los nombres llegan con muchas variantes)
# ---------------------------------------------------------------------------
def normaliza_grupo(p: str) -> str:
    s = re.sub(r"\s+", " ", (p or "")).upper()
    d = s.replace(" ", "")  # despaciado: tolera cortes de OCR ("BIL DU" -> "BILDU")
    if "GOAZEN" in d:
        return "GOAZEN BILBAO"
    if "ELKARREKIN" in d or "PODEMOS" in d or "EZKERANITZA" in d or "EQUO" in d:
        return "ELKARREKIN BILBAO"
    # UDALBERRI antes de BILBAO EN COMÚN para no confundir
    if "UDALBERRI" in d:
        return "UDALBERRI"
    # "Bilbao en Común" fue la coalición pre-Elkarrekin (mismos integrantes)
    if "BILBAOENCOMUN" in d or "BILBAOENCOMÚN" in d or "BILBAOENKUMUN" in d:
        return "ELKARREKIN BILBAO"
    if "BILDU" in d or "EUSKALHERRIA" in d or "EAE-ANV" in d or "EAEANV" in d:
        return "EH BILDU"
    # HB (Herri Batasuna) y EA (Eusko Alkartasuna) — predecesores de EH BILDU
    if re.search(r"\bHB\b", s) or "HERRIBATASUNA" in d or "HERRIKO\s*BATASUNA" in d.replace(" ", ""):
        return "EH BILDU"
    if "SOCIALIST" in d or "PSE" in d or "PSOE" in d:
        return "PSE-EE"
    # Sozialista Abertzaleak (nombre histórico en Basque de los socialistas)
    if "SOZIALISTAK" in d or "SOZIALIST" in d or "ABERTZALEAK" in d:
        return "PSE-EE"
    if "PNV" in d or "EAJ" in d or "NACIONALIST" in d or "JELTZALE" in d:
        return "EAJ-PNV"
    if "POPULAR" in d or "P.P" in d or re.search(r"\bP\s?P\b", s):
        return "PP"
    if "EZKERBATUA" in d or "IZQUIERDAUNIDA" in d or "BERDEAK" in d or re.search(r"\bI\s?U\b", s):
        return "EZKER BATUA-IU"
    if "ARALAR" in d:
        return "ARALAR"
    if "CIUDADANOS" in d or re.search(r"\bC\s?S\b", s):
        return "CIUDADANOS"
    if "VOX" in d:
        return "VOX"
    # Gobierno/Equipo: "Gorbernu Taldeak" (Basque), "Junta de Gobierno", "Alcaldía"
    if "GOBIERNO" in d or "GOBERNUTAL" in d or "GORBERNU" in d or "ALCALD" in d:
        return "EQUIPO DE GOBIERNO"
    if "MIXTO" in d:
        return "GRUPO MIXTO"
    if not p or p.strip() == "" or p == "Desconocido":
        return "Desconocido"
    return p.strip()[:40]


# Extrae el GRUPO PROPONENTE del título/texto del punto ("...que presenta el Grupo
# Municipal EH BILDU..."). Es mucho más fiable que el metadato `party` (que es del orador).
_GRUPO_RE = re.compile(
    r"(?:que\s+presenta[n]?\s+(?:el|los)|presentad[ao]\s+por\s+(?:el|los)|del|de\s+la)\s+grupo[s]?\s+"
    r"(?:pol[ií]tico[s]?\s+)?(?:municipal(?:es)?\s+)?(.{3,60}?)(?:\s*,|\.|cuya|que\s+su|presenta|$)",
    re.IGNORECASE,
)

# Regex para proposiciones en Basque: "X udal taldeak aurkezten duen"
_GRUPO_EUSKERA_RE = re.compile(
    r"^(.{3,60}?)\s+udal\s+tald(?:eak|e(?:ak)?)\s+aurkezten",
    re.IGNORECASE,
)

# Segunda regex: busca el nombre del partido directamente en el texto cuando
# no aparece el patrón "Grupo Municipal X". Orden de alternativas: los más
# largos/específicos primero para evitar capturas parciales.
_PARTIDO_DIRECTO_RE = re.compile(
    r"\b("
    r"EH\s+BILDU|EH-BILDU|EHBILDU|HERRI\s+BATASUNA|EUSKAL\s+HERRIA\s+BILDU"
    r"|ELKARREKIN\s+BILBAO|ELKARREKIN"
    r"|GOAZEN\s+BILBAO|GOAZEN"
    r"|EZKER\s+BATUA[- ]IU|EZKER\s+BATUA|IZQUIERDA\s+UNIDA"
    r"|PSE[- ]EE|PSE\s*EE|PARTIDO\s+SOCIALISTA|SOZIALISTA\s+ABERTZALEAK"
    r"|EAJ[- ]PNV|EAJ\s*PNV|PARTIDO\s+NACIONALISTA\s+VASCO"
    r"|PARTIDO\s+POPULAR"
    r"|UDALBERRI"
    r"|BILBAO\s+EN\s+COM[ÚU]N"
    r"|ARALAR"
    r"|CIUDADANOS"
    r"|VOX"
    r"|EQUIPO\s+DE\s+GOBIERNO|GOBIERNO\s+MUNICIPAL|GORBERNU\s+TALDEA"
    r"|GRUPO\s+MIXTO"
    r")\b",
    re.IGNORECASE,
)


def extrae_grupo(topic: str, text: str = "") -> str:
    # Intento 1: patrón estructurado "Grupo(s) Municipal(es) X"
    for src in (topic or "", (text or "")[:600]):
        m = _GRUPO_RE.search(src)
        if m:
            raw = m.group(1).strip()
            # Para conjuntas "PSE-EE, EH BILDU y PP" tomar solo el primero
            raw = re.split(r"\s*[,y]\s+(?:EH|PP|PSE|EAJ|ELK|GOA|UDA)", raw)[0]
            g = normaliza_grupo(raw)
            if g != "Desconocido":
                return g
    # Intento 2: proposiciones en Basque "PARTIDO POPULAR udal taldeak aurkezten"
    for src in (topic or "", (text or "")[:600]):
        m = _GRUPO_EUSKERA_RE.search(src)
        if m:
            g = normaliza_grupo(m.group(1))
            if g != "Desconocido":
                return g
    # Intento 3: nombre del partido directamente en el texto
    for src in (topic or "", (text or "")[:900]):
        m = _PARTIDO_DIRECTO_RE.search(src)
        if m:
            g = normaliza_grupo(m.group(1))
            if g != "Desconocido":
                return g
    return "Desconocido"


def prop_id(p: dict) -> str:
    return hashlib.md5((p["date"] + "||" + p["topic"]).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Paso --enrich : extracción LLM por proposición
# ---------------------------------------------------------------------------
EXTRACT_PROMPT = """Eres un analista de actas municipales. Del texto de UNA proposición del
Pleno de Bilbao, extrae en JSON EXACTO (sin texto adicional):

{{
  "tema_principal": "<uno de: {temas}>",
  "temas": ["<2-4 etiquetas temáticas libres y concretas>"],
  "resultado": "<aprobada | rechazada | decae | retirada | aprobada con enmienda | sin resultado>",
  "entidades": [{{"nombre": "<entidad>", "tipo": "<persona|lugar|organizacion>"}}]
}}

Reglas:
- "entidades": personas, lugares (barrios/calles/equipamientos de Bilbao) y organizaciones
  CONCRETAS mencionadas (máx 8). NO incluyas los grupos políticos ni "Ayuntamiento de Bilbao".
- Si no hay resultado claro en el texto, usa "sin resultado".
- Responde SOLO el JSON.

TEXTO:
{texto}
"""


def _parse_json(txt: str):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def get_llm(model: str):
    """Devuelve el LLM para la extracción. 'groq' = ChatGroq (cuidado con la cuota);
    cualquier otro valor = modelo de Ollama LOCAL (p.ej. 'qwen2.5:7b', 'mistral')."""
    if model == "groq":
        from langchain_groq import ChatGroq
        from backend.rag import LLM_MODEL_GROQ, GROQ_API_KEY
        return ChatGroq(model=LLM_MODEL_GROQ, api_key=GROQ_API_KEY, temperature=0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=0)


def _clean_failed():
    """Elimina de proposals_enriched.jsonl los registros FALLIDOS (sin temas ni
    entidades), que quedaron envenenados al agotarse la cuota de Groq, para que se
    reintenten. Conserva los que tienen contenido real."""
    if not os.path.exists(ENRICHED):
        return
    recs = [json.loads(l) for l in open(ENRICHED, encoding="utf-8")]
    buenos = [r for r in recs if r.get("temas") or r.get("entidades")]
    if len(buenos) != len(recs):
        with open(ENRICHED, "w", encoding="utf-8") as f:
            for r in buenos:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[*] limpieza: {len(recs) - len(buenos)} registros fallidos eliminados "
              f"(se reintentarán); {len(buenos)} válidos conservados")


def enrich(limit=None, model="groq"):
    _clean_failed()
    llm = get_llm(model)
    print(f"[*] modelo de extracción: {model}")

    proposals = [json.loads(l) for l in open(PROPOSALS, encoding="utf-8")]
    done = set()
    if os.path.exists(ENRICHED):
        for l in open(ENRICHED, encoding="utf-8"):
            done.add(json.loads(l)["id"])
    print(f"[*] {len(proposals)} proposiciones | ya enriquecidas: {len(done)}")

    out = open(ENRICHED, "a", encoding="utf-8")
    n = 0
    fallos = 0
    for p in proposals:
        pid = prop_id(p)
        if pid in done:
            continue
        if limit and n >= limit:
            break
        prompt = EXTRACT_PROMPT.format(temas=", ".join(TEMAS), texto=p["text"][:5000])
        try:
            resp = llm.invoke(prompt).content
            data = _parse_json(resp)
        except Exception as e:
            print(f"[!] {pid} fallo LLM: {str(e)[:120]}")
            fallos += 1
            if fallos >= 20:
                print("[!] 20 fallos seguidos (¿cuota agotada?). Paro; reanuda más tarde.")
                break
            continue
        if not data:  # respuesta no parseable → NO escribir, se reintentará
            fallos += 1
            continue
        fallos = 0
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
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        n += 1
        if n % 25 == 0:
            print(f"    enriquecidas {n} (última: {p['date']} · {rec['tema_principal']})")
    out.close()
    print(f"[+] +{n} enriquecidas → {ENRICHED}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="groq",
                    help="'groq' (cuidado cuota) o un modelo de Ollama LOCAL, p.ej. qwen2.5:7b")
    args = ap.parse_args()
    if args.enrich:
        enrich(limit=args.limit, model=args.model)
    else:
        print("Usa --enrich")
