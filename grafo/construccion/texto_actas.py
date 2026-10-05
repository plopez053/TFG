"""
El texto de las actas para la construcción del grafo:

  - texto de cada PDF, página a página, en caché
  - transcripción (OCR con Gemini) de las actas escaneadas
  - puntos del orden del día a partir del texto completo
  - fragmentos del índice vectorial, para lo que se busca en los debates

    python -m grafo.construccion.texto_actas [acta.pdf ...]   # OCR de las actas sin capa de texto
"""
import glob
import io
import json
import os
import pickle
import re
import sys
import threading
import time
import unicodedata
from typing import Iterator, List

from comun.rutas import BASE_DIR as ROOT, CACHE_GRAFO, CHROMA_PATH, DATA_PATH as ACTAS


# =============================================================================
# Ficheros JSONL de datos intermedios
# =============================================================================

# carga un JSONL completo en una lista de dicts
def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# itera un JSONL línea a línea sin cargarlo entero en memoria
def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# =============================================================================
# Texto de cada acta, página a página
# =============================================================================

# Texto de cada acta, página a página, guardado en caché (leer 236 PDF con
# PyPDFLoader tarda varios minutos). Mismo lector que el indexador del RAG
# vectorial (vectorial/indexado.py), para que las páginas coincidan.

CACHE = os.path.join(CACHE_GRAFO, "pdf_texto")


def actas():
    return sorted(glob.glob(os.path.join(ACTAS, "**", "*.pdf"), recursive=True))


def fecha(pdf: str) -> str:
    m = re.search(r"(\d{2}-\d{2}-\d{4})", os.path.basename(pdf))
    return m.group(1) if m else ""


# [texto de la página 1, texto de la página 2, ...]
def paginas(pdf: str):
    os.makedirs(CACHE, exist_ok=True)
    destino = os.path.join(CACHE, os.path.basename(pdf) + ".json")
    if os.path.exists(destino):
        with open(destino, encoding="utf-8") as f:
            return json.load(f)
    from langchain_community.document_loaders import PyPDFLoader
    pags = [p.page_content for p in PyPDFLoader(pdf).load()]
    tmp = destino + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pags, f, ensure_ascii=False)
    os.replace(tmp, destino)
    return pags


# =============================================================================
# OCR de las actas escaneadas
# =============================================================================

# Transcripción de las actas escaneadas (PDF sin capa de texto) con Gemini en
# Vertex AI, página a página. Eran 4 extraordinarias (18-02-2022, 21-03-2024,
# 15-07-2025, 30-04-2026) que no estaban ni en el grafo ni en la base vectorial
# porque PyPDFLoader no saca texto de ellas.
#
# La transcripción se guarda en la caché de texto (cache/pdf_texto), que usan el
# segmentador del grafo y scripts/indexar_faltantes.py. Reanudable por página.

PROMPT = ("Transcribe literalmente el texto de esta página de un acta del Pleno del Ayuntamiento de Bilbao. "
          "Respeta el orden de lectura (si hay dos columnas, euskera y castellano, primero la columna de la "
          "izquierda entera y luego la de la derecha), los saltos de línea y la numeración de los puntos "
          "(\"1. PROPUESTA...\"). No resumas, no comentes, no añadas nada: solo el texto de la página.")
REGISTRO = os.path.join(CACHE, "_ocr.json")


def _cliente():
    from google import genai
    from comun.proveedores import GOOGLE_CLOUD_PROJECT, GEMINI_LOCATION, LLM_MODEL_GEMINI
    return genai.Client(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location=GEMINI_LOCATION), LLM_MODEL_GEMINI


def _pagina_pdf(pdf: str, i: int) -> bytes:
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter()
    w.add_page(PdfReader(pdf).pages[i])
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def transcribir(pdf: str, cliente, modelo):
    from google.genai import types
    nombre = os.path.basename(pdf)
    parcial = os.path.join(CACHE, nombre + ".ocr_parcial.json")
    pags = json.load(open(parcial, encoding="utf-8")) if os.path.exists(parcial) else []
    from pypdf import PdfReader
    total = len(PdfReader(pdf).pages)
    for i in range(len(pags), total):
        for intento in range(4):
            try:
                r = cliente.models.generate_content(
                    model=modelo,
                    contents=[types.Part.from_bytes(data=_pagina_pdf(pdf, i), mime_type="application/pdf"), PROMPT])
                pags.append(r.text or "")
                break
            except Exception as e:
                print(f"[~] {nombre} pág. {i + 1}: {type(e).__name__}: {str(e)[:120]}; reintento", flush=True)
                time.sleep(15 * (intento + 1))
        else:
            raise SystemExit(f"[!] {nombre} pág. {i + 1}: no se pudo transcribir; relanza para seguir")
        json.dump(pags, open(parcial, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"    {nombre} pág. {i + 1}/{total}: {len(pags[-1])} caracteres", flush=True)
    destino = os.path.join(CACHE, nombre + ".json")
    json.dump(pags, open(destino, "w", encoding="utf-8"), ensure_ascii=False)
    os.remove(parcial)
    reg = json.load(open(REGISTRO, encoding="utf-8")) if os.path.exists(REGISTRO) else {}
    reg[nombre] = {"paginas": total, "modelo": modelo}
    json.dump(reg, open(REGISTRO, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# =============================================================================
# Puntos del orden del día
# =============================================================================

# Puntos del orden del día de cada acta, a partir del texto completo del PDF.
#
# El grafo se construyó con los fragmentos del indexador del RAG vectorial, que
# solo corta en "N. PROPUESTA / PROPOSICIÓN / MOCIÓN / DICTAMEN". Desde 2022 las
# actas tienen un apartado "VII. Participación de vecinos/as, asociaciones y
# entidades en el Pleno" con puntos "N. INICIATIVA VECINAL" / "N. HERRI EKIMENA",
# y algunos plenos abren con un bloque de iniciativas: ninguno se cortaba, y su
# texto quedaba pegado al punto anterior. Tampoco se cortaban las extraordinarias
# sin puntos numerados (presupuestos, debate del estado de la ciudad).
#
# Aquí se corta en cualquier encabezado de punto al principio de línea,
# castellano o euskera, y solo si el número sigue la secuencia del acta (las
# listas numeradas de una parte dispositiva, "1. Instar...", no son puntos).

# (sin "Debate" ni "Elección": "2. Debate y votación de las enmiendas del PP"
# describe el procedimiento de una sesión de presupuestos, no es un punto; y
# "Declaración" solo institucional o de urgencia: "3.- Declaración de la calidad
# del suelo" es un apartado de una ordenanza)
_TIPOS = (r"INICIATIVA\s+VECINAL|HERRI\s+EKIMENA|PROPOSICI[ÓO]N|PROPOSIZIOA|PROPUESTA|PROPOSAMENA|MOCI[ÓO]N|MOZIOA"
          r"|DICTAMEN|IRIZPENA|PREGUNTAS?|GALDER\w*|SE\s+DA\s+CUENTA|DAR\s+CUENTA|KONTU\s+EMATEN"
          r"|DECLARACI[ÓO]N\s+(?:INSTITUCIONAL|DE\s+(?:LA\s+)?UR\s?GENCIA)"
          r"|ADIERAZPEN\w*|COMPARECENCIA|INTERPELACI[ÓO]N|TOMA\s+DE|RATIFICACI[ÓO]N|APROBAR|APROBACI[ÓO]N|ASUNTO|GAIA")
# "12. PROPOSICIÓN..." o, en el formato antiguo, "-12-" y en la línea siguiente
# (con líneas en blanco por medio) "Proposición...". Un rótulo de sección
# ("PROPUESTAS-PROPOSAMENAK", "PREGUNTAS-GALDERAK") tras un número de página no
# es un punto.
_CAB = re.compile(rf"\n[ \t]*(?:(\d{{1,3}})[ \t]?\.[ \t]?-?|-[ \t]?(\d{{1,3}})[ \t]?-[ \t]*\n(?:[ \t]*\n)*)[ \t]*({_TIPOS})\b"
                  r"(?![ \t]*[-–][ \t]*[A-ZÁÉÍÓÚ])", re.I)
# Actas de 2021 y principios de 2022: "-11-" solo en una línea y, debajo, el grupo
# y "Udal Taldeak aurkezten duen proposamena" (la versión en euskera va antes que
# la castellana, sin "PROPOSIZIOA" en la cabecera).
_CAB_EU = re.compile(r"\n[ \t]*-[ \t]?(\d{1,3})[ \t]?-[ \t]*\n(?:[ \t]*\n)*"
                     r"(?=[^\n]{0,160}\n(?:[ \t]*\n)*[^\n]{0,100}aurkezten[ \t\n]+duen|[^\n]{0,160}aurkezten[ \t\n]+duen)",
                     re.I)
_TIPO_EU = "PROPOSAMENA"
# tipos que abren un punto aunque el número salte uno o dos (un punto sin
# encabezado reconocible en medio)
_FUERTES = re.compile(r"INICIATIVA|HERRI|PROPOSICI|PROPOSIZ|PROPUESTA|PROPOSAM|MOCI|MOZIO|DICTAMEN|IRIZPEN|PREGUNTA|GALDER|SE\s+DA\s+CUENTA|DAR\s+CUENTA|TOMA\s+DE", re.I)
_ABRE = re.compile(r"(Proposici[óo]n\s+(que\s+presenta|del\s+grupo|vecinal|de\s+fecha)|PROPOSICI[ÓO]N\s+del|Se\s+da\s+cuenta\s+de"
                   r"|Propuesta\s+de|PROPUESTA\s+(de|relativa)|Preguntas\s+que|INICIATIVA\s+VECINAL|Iniciativa\s+vecinal"
                   r"|TOMA\s+DE\s+CONOCIMIENTO|Moci[óo]n\s+que)", re.I)
# el primer punto de casi todas las sesiones: aprobar el acta anterior, o la urgencia
_ACTA = re.compile(r"(Aprob\w*\s+(por\s+unanimi\s?dad\s+)?(y\s+en\s+sus\s+propios\s+t[ée]\s?rminos\s+)?(de\s+)?(el|las?)\s+a\s?ctas?"
                   r"|DECLARACI[ÓO]N\s+de\s+(la\s+)?ur\s?gencia|Ratificaci[óo]n\s+de\s+la\s+urgencia)", re.I)
_EUSKERA = re.compile(r"HERRI\s+EKIMENA|PROPOSIZIOA|PROPOSAMENA|MOZIOA|IRIZPENA|GALDER|KONTU\s+EMATEN|ADIERAZPEN|GAIA", re.I)


# Cabeceras de punto sin número ("Proposición que presenta el Grupo...", precedidas
# de un guion suelto) para partir un segmento que se ha tragado a otros: en las
# actas de 2021-2022 los puntos de un apartado no siempre llevan número.
_CAB_SIN_NUMERO = re.compile(
    r"\n[ \t]*(?=(?:Proposici[óo]n que presenta (?:el|la) Grupo|PROPOSICI[ÓO]N del grupo"
    r"|Proposici[óo]n vecinal|INICIATIVA VECINAL|Iniciativa vecinal|PROPUESTA de |Propuesta de "
    r"|Se da cuenta de|Preguntas que formulan|PREGUNTAS que formulan|Moci[óo]n que presenta)[^\n]{0,200})")
_SEGMENTO_GRANDE = 80000
_MIN_SUBPUNTO = 1500


def _norm_cab(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())[:45]


# [(inicio, texto)] de un segmento grande partido en sus cabeceras sin número; vacío
# si no hay al menos dos
def _subdividir(seg: str):
    cortes = []
    ultimo = None
    for m in _CAB_SIN_NUMERO.finditer(seg):
        ini = m.end()
        clave = _norm_cab(seg[ini:ini + 80])
        # la misma cabecera repetida enseguida (orden del día / cuerpo): una sola
        if cortes and clave == ultimo and ini - cortes[-1] < 4000:
            continue
        if cortes and ini - cortes[-1] < _MIN_SUBPUNTO:
            continue
        cortes.append(ini)
        ultimo = clave
    if len(cortes) < 2:
        return []
    cortes.append(len(seg))
    return [(cortes[i], seg[cortes[i]:cortes[i + 1]]) for i in range(len(cortes) - 1)]


_BASQUE = re.compile(r"aurkezten|zuzenketa|xedapen|honela\s+dioena|udal\s+taldeak", re.I)
_TITULO_ES = re.compile(r"(?:Proposici[óo]n|PROPOSICI[ÓO]N|Iniciativa\s+vecinal|INICIATIVA\s+VECINAL)[^\n]{0,400}")


def _titulo_castellano(seg: str, n):
    # el segmento empieza en euskera (el bloque euskera va antes del castellano):
    # el título es la cabecera castellana de la proposición
    if _BASQUE.search(seg[:700]):
        m = _TITULO_ES.search(seg)
        if m:
            return f"{n}. " + re.sub(r"\s+", " ", seg[m.start():m.start() + 600]).strip()[:600]
    return None


def _titulo(texto: str) -> str:
    t = re.sub(r"\s+", " ", texto[:900]).strip()
    # hasta el primer punto y aparte razonable (máx. 600 caracteres)
    return t[:600]


# [{"numero", "titulo", "titulo_eu", "texto", "pagina_ini", "pagina_fin"}] de un acta
def puntos(pdf: str):
    pags = paginas(pdf)
    inicios, cursor = [], 0
    for p in pags:
        inicios.append(cursor)
        cursor += len(p) + 1
    texto = "\n" + "\n".join(pags)

    def pagina(off):
        import bisect
        return max(1, bisect.bisect_right(inicios, off - 1))

    cabs = []
    ultimo = 0
    candidatas = [(m.start(), int(m.group(1) or m.group(2)), m.group(3), bool(m.group(2)), m.start(3)) for m in _CAB.finditer(texto)]
    candidatas += [(m.start(), int(m.group(1)), _TIPO_EU, True, m.end()) for m in _CAB_EU.finditer(texto)]
    for inicio, n, tipo, viejo, pos in sorted(set(candidatas)):
        if n == ultimo:                                   # pareja castellano/euskera del mismo punto
            ok = True
        elif n == ultimo + 1:
            ok = not re.match(r"APROBA", tipo, re.I) or n == 1
        elif ultimo < n <= ultimo + (6 if viejo else 3):
            # en el formato antiguo ("-N-") los números saltan más: los puntos sin
            # encabezado reconocible quedan dentro del anterior
            ok = bool(_FUERTES.match(tipo))
        elif ultimo < n <= ultimo + 40:
            # salto grande (puntos intermedios sin encabezado reconocible): solo
            # si abre como un punto ("Proposición que presenta el Grupo...")
            ok = bool(_ABRE.match(texto[pos:pos + 60]))
        elif n == 1 and ultimo >= 1:                      # el orden del día al principio y luego el cuerpo
            # pero no una lista numerada de un acuerdo ("1. Aprobar inicialmente...")
            ok = bool(_ABRE.match(texto[pos:pos + 60]) or _ACTA.match(texto[pos:pos + 120]))
        else:
            ok = False
        if ok:
            cabs.append((inicio, n, tipo))
            ultimo = n
    segmentos = {}
    for i, (ini, n, tipo) in enumerate(cabs):
        fin = cabs[i + 1][0] if i + 1 < len(cabs) else len(texto)
        seg = texto[ini:fin]
        s = segmentos.setdefault(n, {"numero": n, "partes": [], "titulo": "", "titulo_eu": "",
                                     "pagina_ini": pagina(ini), "pagina_fin": pagina(fin), "_ini": ini})
        s["partes"].append(seg)
        s["pagina_fin"] = max(s["pagina_fin"], pagina(fin))
        clave = "titulo_eu" if _EUSKERA.match(tipo) else "titulo"
        # el título es el del segmento más largo (el cuerpo, no la línea del orden del día)
        if len(seg) > len(s.get("_largo_" + clave, "")):
            s["_largo_" + clave] = seg
            s[clave] = _titulo(seg)
    out = []
    for n in sorted(segmentos):
        s = segmentos[n]
        partes = s.pop("partes")
        cuerpo = "".join(partes)
        ini_abs = s.pop("_ini", 0) if len(partes) == 1 else None
        s.pop("_largo_titulo", None)
        s.pop("_largo_titulo_eu", None)
        s["titulo"] = s["titulo"] or s["titulo_eu"]
        s["texto"] = cuerpo
        es = _titulo_castellano(cuerpo, n)
        if es:
            s["titulo"] = es
        # un segmento que se ha tragado a otros (puntos sin número ni cabecera
        # reconocible): se parte en sus cabeceras; los trozos llevan el número del
        # segmento por cien más su orden (3401, 3402...)
        trozos = _subdividir(cuerpo) if len(cuerpo) > _SEGMENTO_GRANDE else []
        if trozos:
            for k, (off, txt) in enumerate(trozos, 1):
                # la página de cada trozo, si el segmento es de una sola pieza del texto
                pi = pagina(ini_abs + off) if ini_abs is not None else s["pagina_ini"]
                out.append({"numero": n * 100 + k, "texto": txt, "titulo": _titulo(txt), "titulo_eu": "",
                            "pagina_ini": pi, "pagina_fin": s["pagina_fin"]})
            # lo anterior a la primera cabecera sigue siendo del segmento
            if trozos[0][0] > _MIN_SUBPUNTO:
                s["texto"] = cuerpo[:trozos[0][0]]
                out.append(s)
            continue
        out.append(s)
    return out


# =============================================================================
# Fragmentos del índice vectorial
# =============================================================================

# Texto de las actas para la construcción del grafo: todos los fragmentos del
# índice vectorial (ChromaDB), normalizados y con sus metadatos, guardados en
# caché. Lo usan enriquecer.py (resultados que el acta resuelve de forma
# explícita) y debates.py (nombres propios citados en cada debate).

_CACHE = os.path.join(CACHE_GRAFO, "texto_actas.pkl")
_LOCK = threading.Lock()
_CORPUS = None
# cerrojo propio: corpus() llama a _coleccion() con _LOCK tomado (no es reentrante)
_LOCK_COLECCION = threading.Lock()
_COLECCION = None


# colección de fragmentos del índice vectorial
def _coleccion():
    global _COLECCION
    if _COLECCION is None:
        with _LOCK_COLECCION:
            if _COLECCION is None:
                import chromadb
                cli = chromadb.PersistentClient(path=CHROMA_PATH)
                col = cli.list_collections()[0]
                _COLECCION = cli.get_collection(col.name if hasattr(col, "name") else col)
    return _COLECCION


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _iso(fecha: str) -> str:
    p = (fecha or "").split("-")
    return f"{p[2]}-{p[1]}-{p[0]}" if len(p) == 3 else ""


# [(texto normalizado, texto original, fecha ISO, fecha, punto, prop_id, orador, página, pdf)]
def corpus():
    global _CORPUS
    if _CORPUS is None:
        with _LOCK:
            if _CORPUS is None:
                col = _coleccion()
                n = col.count()
                try:
                    with open(_CACHE, "rb") as f:
                        guardado = pickle.load(f)
                    if guardado["n"] == n:
                        _CORPUS = guardado["docs"]
                except (OSError, EOFError, KeyError, pickle.UnpicklingError):
                    pass
                if _CORPUS is None:
                    docs, off = [], 0
                    while True:
                        r = col.get(include=["documents", "metadatas"], limit=5000, offset=off)
                        if not r["ids"]:
                            break
                        for d, m in zip(r["documents"], r["metadatas"]):
                            texto = re.sub(r"\s+", " ", d or "")
                            docs.append((_norm(texto), texto, _iso(m.get("date", "")), m.get("date", ""),
                                         m.get("topic", ""), m.get("prop_id"), m.get("speaker", ""),
                                         m.get("page", 0), os.path.basename(m.get("source", "") or "")))
                        off += len(r["ids"])
                    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
                    tmp = _CACHE + ".tmp"
                    with open(tmp, "wb") as f:
                        pickle.dump({"n": n, "docs": docs}, f)
                    os.replace(tmp, _CACHE)
                    _CORPUS = docs
    return _CORPUS


# transcribe las actas sin capa de texto (todas, o solo las indicadas por su nombre de fichero)
def transcribir_escaneadas(nombres=()):
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass
    cliente, modelo = _cliente()
    nombres = set(nombres)
    for pdf in actas():
        if nombres and os.path.basename(pdf) not in nombres:
            continue
        if sum(len(p.strip()) for p in paginas(pdf)) == 0:
            print(f"[*] {os.path.basename(pdf)}: sin texto, se transcribe con {modelo}", flush=True)
            transcribir(pdf, cliente, modelo)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    transcribir_escaneadas(sys.argv[1:])
