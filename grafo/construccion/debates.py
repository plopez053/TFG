"""
Lo que se dice en el debate de cada punto:

    python -m grafo.construccion.debates intervenciones          # intervenciones.jsonl (sin LLM)
    python -m grafo.construccion.debates resumir [--limite 20]   # intervenciones_resumen.jsonl (LLM, reanudable)

y los nombres propios citados en cada debate, que enriquecer.py añade al grafo.
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from rdflib import Literal, Namespace

from comun.rutas import dato_grafo, GRAFO_DATOS as GRAPHRAG


# =============================================================================
# Intervenciones de cada punto
# =============================================================================

# Extrae las intervenciones de cada punto del orden del día (sin LLM).
#
# Una intervención es lo que dice un orador seguido en un punto: el texto entre
# dos marcadores "SR. APELLIDO:" / "SRA. APELLIDO:" / "SR. ALCALDE:". Sale de
# proposals.jsonl (puntos originales) y proposals_extra.jsonl (puntos recuperados),
# que guardan el texto completo de cada punto; el id del punto es el de
# proposals_enriched*.jsonl, el mismo que usa el grafo.
#
# Salida: intervenciones.jsonl, una línea por intervención:
#   id, punto, fecha, pdf, pagina, orden, orador (como en el acta), apellidos
#   (normalizados), texto.

SALIDA = dato_grafo("intervenciones.jsonl")

# "SR. GIL:", "SRA. RUIZ BUJEDO:", "SR. ALCALDE:", "ALCALDE:", "SR. PRESIDENTE:"
_MARCA = re.compile(
    r"(?m)^[ \t]*((?:SR\.|SRA\.|SRES?\.|ALCALDE|ALCALDESA|PRESIDEN\w+|SECRETAR\w+)"
    r"[A-ZÁÉÍÓÚÑÜa-z\. \-’']{0,40}?"
    r"|[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ\.’' \-]{1,30}? (?:AND|JN|ZIN)\.)\s*:\s*[“\"«]?")
# pie y cabecera de página que se cuelan en el texto
_RUIDO = re.compile(r"(?m)^.*(?:Udalbatzako Idazkaritza Nagusia|Secretaría General del Pleno|P á g i n a \d+).*$\n?")
_ROLES = ("ALCALDE", "ALCALDESA", "PRESIDEN", "SECRETAR")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def apellidos(etiqueta: str) -> str:
    """'SRA. RUIZ BUJEDO' -> 'ruizbujedo'; 'SR. ALCALDE' -> 'alcalde'."""
    e = re.sub(r"^(SRES?\.|SRA\.|SR\.)\s*", "", etiqueta.strip(), flags=re.IGNORECASE)
    e = re.sub(r"\s+(AND|JN|ZIN)\.?$", "", e, flags=re.IGNORECASE)
    e = re.sub(r"[^a-z]", "", norm(e))
    for rol in ("alcaldesa", "alcalde", "alkate"):
        if e.startswith(rol):
            return "alcalde"
    if e.startswith("presiden"):
        return "alcalde"
    if e.startswith("secretar"):
        return "secretario"
    return e


def partir(texto: str):
    """Lista de (etiqueta, texto) en el orden del acta. Lo anterior al primer orador se descarta."""
    texto = _RUIDO.sub("", texto or "")
    marcas = list(_MARCA.finditer(texto))
    out = []
    for i, m in enumerate(marcas):
        fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(texto)
        cuerpo = re.sub(r"\s+", " ", texto[m.end():fin]).strip(" “”\"«»")
        if cuerpo:
            out.append((re.sub(r"\s+", " ", m.group(1)).strip(), cuerpo))
    return out


def _clave(r):
    return (r.get("source"), r.get("topic"), str(r.get("page_ini")))


def _cargar_dato(nombre):
    ruta = os.path.join(GRAPHRAG, nombre)
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extraer_intervenciones():
    ids = {}
    for fn in ("proposals_enriched.jsonl", "proposals_enriched_extra.jsonl"):
        for r in _cargar_dato(fn):
            ids[_clave(r)] = r["id"]
    n, sin_id = 0, 0
    with open(SALIDA + ".tmp", "w", encoding="utf-8") as out:
        for fn in ("proposals.jsonl", "proposals_extra.jsonl"):
            for r in _cargar_dato(fn):
                pid = r.get("id") or ids.get(_clave(r))
                if not pid:
                    sin_id += 1
                    continue
                for orden, (etq, cuerpo) in enumerate(partir(r.get("text", ""))):
                    iid = hashlib.sha1(f"{pid}|{orden}".encode()).hexdigest()[:16]
                    out.write(json.dumps({
                        "id": iid, "punto": pid, "fecha": r.get("date"),
                        "pdf": os.path.basename((r.get("source") or "").replace("\\", "/")),
                        "pagina": r.get("page_ini"), "orden": orden, "orador": etq,
                        "apellidos": apellidos(etq), "texto": cuerpo}, ensure_ascii=False) + "\n")
                    n += 1
    os.replace(SALIDA + ".tmp", SALIDA)
    return n, sin_id


# =============================================================================
# Resumen y postura de cada intervención
# =============================================================================

# Resume la postura de cada intervención con Gemini (Vertex). Reanudable.
#
# Lee intervenciones.jsonl y añade una línea por intervención a
# intervenciones_resumen.jsonl (id, resumen, postura). Lo ya hecho se salta, así
# que se puede interrumpir y relanzar. Las intervenciones cortas (< MIN_CHARS) no
# pasan por el LLM: su resumen es el propio texto y la postura queda "neutra".

ENTRADA = dato_grafo("intervenciones.jsonl")
SALIDA_RESUMEN = dato_grafo("intervenciones_resumen.jsonl")
MIN_CHARS = 300
MAX_CHARS = 12000
HILOS = 6
POSTURAS = {"a_favor", "en_contra", "abstencion", "neutra"}

PROMPT = """Eres analista de los plenos del Ayuntamiento de Bilbao. Resume la intervención de un orador en un punto del orden del día.

PUNTO: {titulo}
ORADOR: {orador}

INTERVENCIÓN:
{texto}

Responde SOLO con un JSON de esta forma:
{{"resumen": "una o dos frases en español: qué defiende, pide, critica o informa el orador, sin inventar nada que no esté en el texto",
 "postura": "a_favor | en_contra | abstencion | neutra"}}

"postura" es la que mantiene el orador ante la propuesta del punto: a_favor si la apoya, en_contra si la rechaza o la critica, abstencion si anuncia que se abstiene, neutra si solo pregunta, informa, modera o no toma partido."""


def _cargar(ruta):
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _titulos():
    out = {}
    for fn in ("proposals_enriched.jsonl", "proposals_enriched_extra.jsonl"):
        for r in _cargar(dato_grafo(fn)):
            out[r["id"]] = " ".join((r.get("topic") or "").split())[:300]
    return out


def resumir():
    limite = None
    if "--limite" in sys.argv:
        limite = int(sys.argv[sys.argv.index("--limite") + 1])
    from grafo.construccion.build_graph import get_llm, _parse_json, _msg_text
    llm = get_llm("vertex")
    titulos = _titulos()
    hechas = {r["id"] for r in _cargar(SALIDA_RESUMEN)}
    todas = [r for r in _cargar(ENTRADA) if r["id"] not in hechas]
    cortas = [r for r in todas if len(r["texto"]) < MIN_CHARS]
    largas = [r for r in todas if len(r["texto"]) >= MIN_CHARS]
    if limite:
        largas = largas[:limite]
        cortas = []
    cerrojo = threading.Lock()
    out = open(SALIDA_RESUMEN, "a", encoding="utf-8")
    hecho = [0]
    t0 = time.time()

    def guardar(r, resumen, postura):
        with cerrojo:
            out.write(json.dumps({"id": r["id"], "resumen": resumen, "postura": postura,
                                  "modelo": "vertex"}, ensure_ascii=False) + "\n")
            out.flush()
            hecho[0] += 1
            if hecho[0] % 100 == 0:
                print(f"  {hecho[0]} hechas, {(time.time() - t0) / 60:.1f} min", flush=True)

    for r in cortas:
        guardar(r, r["texto"], "neutra")

    def trabajar(r):
        prompt = PROMPT.format(titulo=titulos.get(r["punto"], ""), orador=r["orador"],
                               texto=r["texto"][:MAX_CHARS])
        for intento in range(4):
            try:
                resp = llm.invoke(prompt)
                data = _parse_json(_msg_text(resp))
                resumen = " ".join(str(data.get("resumen", "")).split())
                postura = str(data.get("postura", "neutra")).strip().lower()
                if resumen:
                    guardar(r, resumen, postura if postura in POSTURAS else "neutra")
                    return
            except Exception as exc:  # cuota o JSON mal formado: se reintenta y, si no, se salta
                print(f"  [!] {r['id']} intento {intento + 1}: {str(exc)[:120]}", flush=True)
                time.sleep(4 * (intento + 1))

    print(f"[*] {len(largas)} intervenciones por resumir, {len(cortas)} cortas")
    with ThreadPoolExecutor(HILOS) as pool:
        list(pool.map(trabajar, largas))
    out.close()
    print(f"[+] hecho: {hecho[0]} -> {SALIDA_RESUMEN}")


# =============================================================================
# Nombres propios citados en el debate
# =============================================================================

# Nombres propios del debate de cada punto (bo:nombradoEnDebate).
#
# El grafo indexaba de qué trata cada proposición (título, subtemas) y las
# entidades que el LLM extrajo de su encabezado, pero no lo que se nombra en el
# debate. En la octava evaluación el Mercado del Ensanche (26-02-2026) y el
# Palacio de Justicia (28-09-2023) solo aparecían en el debate, y el GraphRAG,
# que responde solo con el grafo, no los encontraba aunque el RAG
# vectorial sí tenía ese texto. Para que las dos estructuras tengan la misma
# información, aquí se guardan en el grafo los nombres propios de cada debate.
#
# Sin LLM: se toman del texto de las actas (texto_actas.corpus()) las
# secuencias de palabras con mayúscula inicial ("Palacio de
# Justicia", "Zabalburu"), con la palabra común que las precede si la hay
# ("mercado del Ensanche", "plaza Zabalburu"). Se guardan normalizadas (sin
# tildes, en minúsculas) y sin repetir, una por triple.
#
# GraphRAG las usa como menciones, solo en las formas que distinguen "trata" de
# "menciona" (la última vez, listados, votaciones, "¿se ha hablado de...?"); los
# recuentos y rankings no cambian.

BO = Namespace("http://bilbao.tfg/ontology#")
BR = Namespace("http://bilbao.tfg/resource/")

_MAY = r"[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:-[A-ZÁÉÍÓÚÑÜ]?[a-záéíóúñü]+)?"
_NOMBRE = rf"{_MAY}(?:\s+(?:(?:de|del|de la|de los|de las|y|e)\s+)?{_MAY})*"
_PATRON = re.compile(rf"(?:\b([a-záéíóúñü]{{4,}})\s+(?:(?:de|del)\s+(?:la\s+|los\s+|las\s+|el\s+)?)?)?\b({_NOMBRE})")
# palabras con mayúscula que no nombran nada (inicio de frase, tratamientos, cargos)
_VACIAS = {"el", "la", "los", "las", "en", "por", "para", "con", "sin", "sobre", "desde", "hasta", "como",
           "que", "pero", "porque", "aunque", "cuando", "donde", "este", "esta", "estos", "estas", "ese", "esa",
           "eso", "esto", "nosotros", "ustedes", "usted", "yo", "no", "si", "y", "a", "al", "de", "del", "se",
           "lo", "le", "les", "su", "sus", "mi", "muchas", "gracias", "bueno", "bien", "entonces", "ademas",
           "senor", "senora", "senores", "senoras", "sr", "sra", "don", "dona", "alcalde", "alcaldesa",
           "concejal", "concejala", "presidente", "presidenta", "pleno", "grupo", "grupos", "municipal",
           "proposicion", "propuesta", "enmienda", "mocion", "punto", "acuerdo", "votacion", "orden", "dia",
           "hay", "ha", "han", "es", "son", "era", "fue", "va", "vamos", "creo", "quiero", "decir"}
_MAX_POR_PUNTO = 120
# una frase que sale en más del 2 % de los debates no distingue ninguno ("bilbao", "gobierno vasco")
_DF_MAX = 0.02
# cabeceras y pies de página de las actas
_CABECERA = re.compile(r"idazkaritza|secretaria general|udalbatza|osoko bilkur")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn")).strip()


# El sistema es en castellano: en las actas bilingües se descartan las frases en euskera. Se
# decide por frase contando palabras funcionales de cada lengua (un fragmento puede traer las dos)
_FUNC_ES = {"de", "la", "el", "que", "los", "en", "y", "las", "del", "por", "para", "con", "se", "no", "una",
            "un", "es", "lo", "al", "como", "mas", "pero", "sus", "su", "esta", "este", "son", "han", "ha"}
_FUNC_EU = {"eta", "da", "du", "dute", "dira", "ditu", "dituzte", "bat", "ere", "baina", "izan", "egin", "behar",
            "dugu", "ez", "gure", "hau", "hori", "beste", "zer", "nola", "oso", "bere", "dago", "daude", "dela",
            "duela", "edo", "baino", "bezala", "dagoen", "zuen", "zen", "ziren", "ditugu", "dugun", "dituen",
            "horren", "honen", "batzuk", "guztiak", "gehiago", "orain", "ondoren", "eskerrik", "asko", "jauna",
            "andrea", "alkate", "taldea", "taldeak", "proposamena", "udala", "udalak", "bozketa"}


# terminaciones del euskera ("taldearen", "erakundeei", "kontseiluari", "Euskadiko", "Ekintzak")
_SUFIJO_EU = re.compile(r"(ari|ei|aren|eko|ko|etan|agan|ekin|entzat|tzak|tzen|tzea|tzeko|ezko|tzat)$")
_PALABRAS_EU = {"gobernu", "adibidez", "diogu", "toki", "erakunde", "administrazio", "entitate", "udal",
                "batzorde", "osoko", "bilkura", "jaun", "andre", "lehendakari"}


def _nombre_castellano(frase_norm: str) -> bool:
    return not any(_SUFIJO_EU.search(w) or w in _PALABRAS_EU for w in frase_norm.split())


def _castellano(frase: str) -> bool:
    w = re.findall(r"[a-zñ]+", _norm(frase))
    es, eu = sum(x in _FUNC_ES for x in w), sum(x in _FUNC_EU for x in w)
    return eu == 0 or es > 2 * eu


def _frases_castellano(texto: str) -> list:
    return [f for f in re.split(r"(?<=[.!?;:])\s+|\n+", texto or "") if f.strip() and _castellano(f)]


# frases nombradas en un texto (solo en las frases en castellano): "mercado del ensanche",
# "palacio de justicia", "zabalburu"
def nombres(texto: str) -> set:
    out = set()
    for frase in _frases_castellano(texto):
        out |= _nombres_frase(frase)
    return out


def _nombres_frase(texto: str) -> set:
    out = set()
    for m in _PATRON.finditer(texto or ""):
        cabeza, nombre = m.group(1), m.group(2)
        palabras = [w for w in _norm(nombre).split() if w not in {"de", "del", "la", "los", "las", "y", "e"}]
        if not palabras or all(w in _VACIAS for w in palabras):
            continue
        # una sola palabra al empezar una frase suele ser una palabra corriente ("Además")
        if len(palabras) == 1 and len(palabras[0]) < 4:
            continue
        ini = m.start(2)
        if len(palabras) == 1 and re.search(r"(^|[.!?:;]\s*|\n\s*)$", texto[max(0, ini - 3):ini]) and not cabeza:
            continue
        frase = _norm(nombre)
        if _CABECERA.search(frase) or not _nombre_castellano(frase):
            continue
        if cabeza and cabeza not in _VACIAS and _nombre_castellano(_norm(cabeza)):
            out.add(_norm(texto[m.start(1):m.end(2)]))
        out.add(frase)
    return out


# {pid castellano: [frase, ...]} a partir del corpus de texto (texto.corpus()): sin las frases
# demasiado frecuentes y, como mucho, las _MAX_POR_PUNTO más raras de cada debate
def nombres_por_punto(corpus, copia_de=None) -> dict:
    copia_de = copia_de or {}
    por_pid = defaultdict(set)
    for _norm_t, orig, _iso, _fecha, _punto, pid, _orador, _pag, _pdf in corpus:
        if pid:
            por_pid[copia_de.get(pid, pid)].update(nombres(orig))
    df = defaultdict(int)
    for frases in por_pid.values():
        for f in frases:
            df[f] += 1
    tope = _DF_MAX * len(por_pid)
    return {pid: sorted((f for f in frases if df[f] <= tope), key=lambda f: (df[f], -len(f), f))[:_MAX_POR_PUNTO]
            for pid, frases in por_pid.items()}


def anadir(g, corpus, copia_de) -> int:
    existentes = {str(p).rsplit("prop_", 1)[-1] for p in g.subjects(BO.tituloTopic, None)}
    n = 0
    for pid, frases in nombres_por_punto(corpus, copia_de).items():
        if pid not in existentes:
            continue
        for f in frases:
            g.add((BR[f"prop_{pid}"], BO.nombradoEnDebate, Literal(f)))
            n += 1
    return n


def _intervenciones():
    n, sin_id = extraer_intervenciones()
    print(f"{n} intervenciones -> {SALIDA} ({sin_id} puntos sin id)")


PASOS = {"intervenciones": _intervenciones, "resumir": resumir}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PASOS:
        sys.exit("Uso: python -m grafo.construccion.debates " + "|".join(PASOS))
    PASOS[sys.argv.pop(1)]()
