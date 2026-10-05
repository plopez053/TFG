"""
Extracción de los datos de las actas (sin LLM), una entrada por línea en
datos/grafo/*.jsonl:

    python -m grafo.construccion.extraer concejales      # concejales.jsonl, personal_tecnico.jsonl
    python -m grafo.construccion.extraer constitutivas   # comprobación: reparto por grupo en cada legislatura
    python -m grafo.construccion.extraer proposiciones   # proposals.jsonl (fase 1)
    python -m grafo.construccion.extraer extra           # proposals_extra.jsonl y votaciones.jsonl
"""
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

from pypdf import PdfReader

from comun.rutas import DATA_PATH, dato_grafo
from grafo.construccion import texto_actas as T
from grafo.construccion.texto_actas import iter_jsonl
from grafo.construccion.votos import parse_listas_voto
from vectorial.pipeline import RAGPipeline


# =============================================================================
# Concejales y personal técnico
# =============================================================================

# Censo de concejales y personal técnico a partir de la lista de asistentes del
# encabezado de cada acta (concejales.jsonl, personal_tecnico.jsonl), con el grupo
# de cada concejal según concejales_partido.json.

OUT = dato_grafo("concejales.jsonl")
OUT_TEC = dato_grafo("personal_tecnico.jsonl")
PARTIDO_JSON = dato_grafo("concejales_partido.json")

_TRAT = r"(?:don|do[nñ]a|d\.|d[nñ]a\.|excmo\.?\s*sr\.?\s*don|sr\.?\s*don|sra\.?\s*do[nñ]a)"


def _fecha_de_ruta(path: str) -> str:
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", os.path.basename(path))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


# clave para agrupar variantes de OCR del mismo nombre (sin espacios ni acentos)
def _clave_ocr(nombre: str) -> str:
    return re.sub(r"[^a-z]", "", _norm(nombre).lower())


# El Secretario General, el Interventor y los adjuntos a la Secretaría asisten a
# TODOS los plenos pero NO son concejales. Se excluyen por nombre y por contexto.
_NO_CONCEJALES = {
    "jon zabala basterra", "jesus alboniga iturbe", "jose luis burgos ibarguren",
    "javier balparda arregui", "imanol bilbao eizagirre",
    "mikel astorkiza abasolo", "mikel astorquiza abasolo",
    "alvaro maestro corrales", "aizbea atela uriarte",
}

# Fusión explícita de variantes de grafía que las heurísticas no colapsan solas
# (euskera/castellano, apodos, partículas). Clave = _clave_ocr(variante rara) ->
# _clave_ocr(nombre canónico que se conserva). Se aplica al recolectar.
_ALIAS_FUSION = {
    "xabierochandianomartinez": "xabierotxandianomartinez",
    "davidlopateguiescudero": "davidlopategiescudero",
    "inigozubizarretaaguirrezabal": "inigozubizarretaagirrezabal",
    "aitziberibaibarriagadeechevarrieta": "aitziberibaibarriagaetxebarrieta",
    "juanfelixmadariagabrizuela": "felixmadariagabrizuela",
    "txemaoleagazalvidea": "josemariaoleagazalvidea",
    "goyozurrotobajas": "gregoriozurrotobajas",
    "mariacarmenmunozlopez": "carmenmunozlopez",
    "patxixabierfernandezmonje": "xabierfernandezmonje",
    "mirenitxasoerrotetasagastagoya": "itxasoerrotetasagastagoya",
    "inigozubizarretagirrezabal": "inigozubizarretaagirrezabal",
}


def _limpia_nombre(nom: str) -> str:
    nom = nom.strip(" .,-")
    # un tratamiento a media cadena => empieza otra persona: cortar ahí
    nom = re.split(r"\s+(?:do[nñ]a?|d\.|d[nñ]a\.)\s+[A-ZÁÉÍÓÚ]", nom, 1)[0]
    # cortar coletillas que se cuelan al final
    nom = re.split(r"\b(asistid|a\s*sistid|estando|Interventor|Secretario|para celebrar|"
                   r"bajo la|que ser|siendo las|en el sal[oó]n)\b", nom, 1, re.I)[0]
    nom = re.split(r"\.\s+[A-Z]", nom, 1)[0]                    # "... Lopez. A sistidos..."
    nom = re.sub(r"\s*-\s*", "-", nom)                          # "Diez -Andino" -> "Diez-Andino"
    nom = re.sub(r"\s+y(\s+el)?\s*$", "", nom, flags=re.I)      # "... Rique y" / "... y el"
    # unir espacios espurios de OCR dentro de una palabra: la continuación
    # empieza SIEMPRE en minúscula ("H elena"->Helena, "Etx arte"->Etxarte,
    # "Fe lix"->Felix, "Alvare z"->Alvarez, "Lorenz o"->Lorenzo).
    nom = re.sub(r"\b(\w{1,3})\s+(?!de\b|del\b|la\b|las\b|los\b|y\b)([a-zà-ÿ]{2,})\b", r"\1\2", nom)
    nom = re.sub(r"\b(\w{2,})\s+([a-zà-ÿ])\b", r"\1\2", nom)
    return nom.strip(" .,-")


# de 'doña Nekane Alonso Santamaría, don Alfonso Gil Invernón y doña ...'
def _split_nombres(bloque: str):
    bloque = re.sub(r"\s+", " ", bloque)
    trozos = re.split(rf"\s*[,;]\s*|\s+y\s+(?={_TRAT}\b)", bloque, flags=re.I)
    nombres = []
    for t in trozos:
        m = re.match(rf"^{_TRAT}\s+(.+)$", t.strip(), re.I)
        if not m:
            continue
        nom = _limpia_nombre(m.group(1))
        if not (4 <= len(nom) <= 55 and " " in nom and not re.search(r"\d", nom)):
            continue
        if _norm(nom).lower() in _NO_CONCEJALES:
            continue
        nombres.append(nom)
    return nombres


# variantes de apellido (eu/es, plegado fonético) de un nombre completo
def _apellidos_variantes(nombre_completo: str):
    palabras = _norm(nombre_completo).split()
    if len(palabras) < 2:
        return {nombre_completo.upper()}
    # apellidos = las últimas 2 palabras salvo que haya partículas (de, del, la...)
    corte = 1 if len(palabras) <= 3 else 2
    apes = palabras[corte:]
    v = {" ".join(apes).upper(), apes[0].upper()}
    if len(apes) >= 2:
        v.add(apes[-1].upper())
        v.add(" ".join(apes[:2]).upper())
    return {x for x in v if len(x) >= 3}


def extraer_concejales():
    partido_tab = {}
    if os.path.exists(PARTIDO_JSON):
        raw = json.load(open(PARTIDO_JSON, encoding="utf-8"))
        # {mandato: {NOMBRE: grupo}} -> buscamos por nombre normalizado.
        # Se ignora cualquier clave que empiece por "_" (comentarios/instrucciones).
        for mandato, gente in raw.items():
            if mandato.startswith("_") or not isinstance(gente, dict):
                continue
            for nom, grp in gente.items():
                partido_tab[_norm(nom).upper()] = grp
        print(f"[*] tabla de partidos: {len(partido_tab)} concejales curados")
    else:
        print(f"[!] no existe {os.path.basename(PARTIDO_JSON)} -> todos los grupos 'Desconocido'. "
              "Créalo (plantilla en PLAN_MEJORA_GRAFO.md, Tarea 1).")

    pdfs = sorted(glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True))
    print(f"[*] {len(pdfs)} actas")

    # persona (clave sin espacios ni acentos) -> {variantes:Counter, fechas:set, alcalde:bool}
    from collections import Counter as _C
    personas = defaultdict(lambda: {"variantes": _C(), "fechas": set(), "alcalde": False})
    # personal técnico: {clave -> {variantes, fechas, cargos:set}}
    tecnicos = defaultdict(lambda: {"variantes": _C(), "fechas": set(), "cargos": set()})

    for path in pdfs:
        fecha = _fecha_de_ruta(path)
        try:
            from pypdf import PdfReader
            txt = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages[:4])
        except Exception:
            continue
        txt = re.sub(r"\s+", " ", txt)

        # Alcalde/sa
        m_alc = re.search(rf"[Pp]residencia del .{{0,60}}?Alcalde[^:]{{0,40}}?:?\s*{_TRAT}\s+([A-ZÁÉÍÓÚÑa-záéíóúñ' ]{{6,50}}?)(?:\s+(?:En el|En la|bajo|,|para))",
                          txt)
        if not m_alc:
            m_alc = re.search(rf"bajo la presidencia del .{{0,40}}?Alcalde[^,]{{0,30}}?{_TRAT}\s+([A-Za-záéíóúñ' ]{{6,50}}?)\s+y,?\s+para", txt)
        if m_alc:
            nom = _norm(_limpia_nombre(m_alc.group(1)))
            k = _clave_ocr(nom)
            if len(k) >= 8:
                personas[k]["variantes"][nom] += 1
                personas[k]["alcalde"] = True
                if fecha:
                    personas[k]["fechas"].add(fecha)

        # Personal técnico (Secretario General, Interventor)
        for m in re.finditer(rf"(Secretari[oa] General del Pleno|Interventor[a]? General(?: Municipal)?)[,]?\s+{_TRAT}\s+([A-Za-záéíóúñ' ]{{6,45}}?)(?:\s*(?:y|,|\.|estando|$))",
                             txt, re.I):
            cargo = "Secretario General" if "secretari" in m.group(1).lower() else "Interventor General"
            nom = _norm(_limpia_nombre(m.group(2)))
            k = _clave_ocr(nom)
            if len(k) >= 8:
                tecnicos[k]["variantes"][nom] += 1
                tecnicos[k]["cargos"].add(cargo)
                if fecha:
                    tecnicos[k]["fechas"].add(fecha)

        # Tenientes de Alcalde + Concejales/as
        for m in re.finditer(r"(?:Tenientes de Alcalde|Concejal(?:es)?/?a?s?)\s*:?\s*(.{20,1500}?)(?:asistid[oa]s? por|estando presente|Existe,? en consecuencia|Secretario General del Pleno)",
                             txt, re.I):
            for nom in _split_nombres(m.group(1)):
                nom = _norm(nom)
                k = _clave_ocr(nom)
                if len(k) < 8:
                    continue
                personas[k]["variantes"][nom] += 1
                if fecha:
                    personas[k]["fechas"].add(fecha)

    # fusión explícita de variantes de grafía (euskera/castellano, apodos)
    for var, canon in _ALIAS_FUSION.items():
        if var in personas and var != canon:
            dst = personas[canon]
            src = personas.pop(var)
            dst["variantes"].update(src["variantes"])
            dst["fechas"] |= src["fechas"]
            dst["alcalde"] = dst["alcalde"] or src["alcalde"]

    # volcado
    def _key_fecha(f):
        d = f.split("-")
        return (int(d[2]), int(d[1]), int(d[0])) if len(d) == 3 else (0, 0, 0)

    _PART = ("de", "del", "la", "las", "los", "y", "ma", "mª")

    def _fold(s):
        # pliegue fonético euskera<->castellano + a-z0-9 para casar apellidos
        for a, b in (("tx", "ch"), ("tz", "z"), ("gui", "gi"), ("gue", "ge"),
                     ("qu", "k"), ("gu", "g"), ("v", "b"), ("ss", "s"), ("ii", "i"),
                     ("y", "i"), ("z", "s")):
            s = s.replace(a, b)
        return re.sub(r"[^a-z0-9]", "", s)

    def _partes(nom):
        w = [x for x in _norm(nom).lower().split() if x not in _PART]
        corte = 1 if len(w) <= 3 else 2
        return set(w[:corte] or w[:1]), _fold("".join(w[corte:] or w[-1:]))

    def _fwords(nom):
        return [_fold(x) for x in _norm(nom).lower().split() if x not in _PART]

    def _es_prefijo(a, b):
        # a y b son listas de palabras plegadas; ¿una es prefijo de la otra
        # (nombre truncado, p.ej. "jose lorenzo delgado" c "...delgado vicente")?
        s, l = (a, b) if len(a) <= len(b) else (b, a)
        return len(s) >= 2 and 0 < len(l) - len(s) <= 2 and l[:len(s)] == s

    # tabla de partidos también indexada por apellidos plegados (fallback para
    # variantes de grafía: "Goyo/Gregorio Zurro Tobajas", "Diez-Andino"...)
    part_by_ape = {}
    for nom_t, grp_t in list(partido_tab.items()):
        part_by_ape[_partes(nom_t)[1]] = grp_t

    def _grupo_de(nombre):
        g = partido_tab.get(_norm(nombre).upper())
        if g:
            return g
        return part_by_ape.get(_partes(nombre)[1], "Desconocido")

    filas = []
    for k, v in personas.items():
        fechas = sorted(v["fechas"], key=_key_fecha)
        cand = sorted(v["variantes"].items(),
                      key=lambda kv: (-kv[1], sum(1 for w in kv[0].split() if len(w) <= 2)))
        nombre = cand[0][0]
        if len(nombre.split()) < 2 or len(fechas) < 2:
            continue  # ruido: solo aparece 1 vez, o nombre incompleto
        given, apes = _partes(nombre)
        filas.append({
            "nombre": nombre,
            "apellidos": sorted(_apellidos_variantes(nombre)),
            "grupo": _grupo_de(nombre),
            "desde": fechas[0],
            "hasta": fechas[-1],
            "n_actas": len(fechas),
            "es_alcalde": v["alcalde"],
            "_given": given,
            "_apes": apes,
            "_fw": _fwords(nombre),
        })

    # Segunda pasada de fusión: mismos apellidos plegados y nombres de pila que
    # se solapan -> misma persona ("Itziar Miren Urtasun Jimeno" == "Itziar
    # Urtasun Jimeno"; "Xabier Otxandiano" == "Xabier Ochandiano"). Hermanos con
    # los mismos apellidos NO se fusionan (nombres de pila disjuntos: Ángel vs
    # Gabriel Rodrigo Izquierdo). Gana la variante con más n_actas.
    filas.sort(key=lambda x: -x["n_actas"])
    fusion = []
    for r in filas:
        for f0 in fusion:
            misma = (f0["_apes"] == r["_apes"] and (f0["_given"] & r["_given"])) or \
                    (_es_prefijo(f0["_fw"], r["_fw"]) and next(iter(f0["_given"] or [""])) == next(iter(r["_given"] or [""])))
            if misma:
                f0["n_actas"] += r["n_actas"]
                f0["desde"] = min(f0["desde"], r["desde"], key=_key_fecha)
                f0["hasta"] = max(f0["hasta"], r["hasta"], key=_key_fecha)
                f0["es_alcalde"] = f0["es_alcalde"] or r["es_alcalde"]
                f0["_given"] |= r["_given"]
                if len(r["_fw"]) > len(f0["_fw"]):
                    f0["_fw"] = r["_fw"]
                if f0["grupo"] == "Desconocido":
                    f0["grupo"] = r["grupo"]
                break
        else:
            fusion.append(r)
    for r in fusion:
        for kk in ("_given", "_apes", "_fw"):
            r.pop(kk, None)
    filas = sorted(fusion, key=lambda x: (-x["n_actas"], x["nombre"]))

    with open(OUT, "w", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sin_grupo = sum(1 for r in filas if r["grupo"] == "Desconocido")
    print(f"[+] {len(filas)} concejales -> {OUT}")
    print(f"    con grupo: {len(filas) - sin_grupo} | sin grupo: {sin_grupo}")

    # personal técnico
    tfilas = []
    for k, v in tecnicos.items():
        fechas = sorted(v["fechas"], key=_key_fecha)
        if len(fechas) < 2:
            continue
        cand = sorted(v["variantes"].items(),
                      key=lambda kv: (-kv[1], sum(1 for w in kv[0].split() if len(w) <= 2)))
        tfilas.append({
            "nombre": cand[0][0],
            "apellidos": sorted(_apellidos_variantes(cand[0][0])),
            "cargo": " / ".join(sorted(v["cargos"])),
            "desde": fechas[0], "hasta": fechas[-1], "n_actas": len(fechas),
        })
    tfilas.sort(key=lambda x: -x["n_actas"])
    with open(OUT_TEC, "w", encoding="utf-8") as f:
        for r in tfilas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] {len(tfilas)} personal técnico -> {OUT_TEC}")
    print("    revisa a mano: nombres, partidos, fechas de mandato, "
          "y los que salgan con pocas n_actas (posible ruido de OCR).")
    return filas


# =============================================================================
# Comprobación con las actas de constitución
# =============================================================================

# Reparto de concejales por grupo en cada legislatura, leído de las actas de
# sesión constitutiva (2011, 2015, 2019 y 2023). Uso puntual: imprime la tabla
# con la que se comprobó concejales_partido.json.

ACTAS = {
    "2011-2015": os.path.join(DATA_PATH, "2011", "11-06-2011_Extraordinaria_Acta.pdf"),
    "2015-2019": os.path.join(DATA_PATH, "2015", "13-06-2015_Extraordinaria_Acta.pdf"),
    "2019-2023": os.path.join(DATA_PATH, "2019", "15-06-2019_Extraordinaria_Acta.pdf"),
    "2023-2027": os.path.join(DATA_PATH, "2023", "17-06-2023_Extraordinaria_Acta.pdf"),
}

PARTIDOS = [
    (r"nacionalista vasco|eaj-?pnv|euzko alderdi", "EAJ-PNV"),
    (r"socialista de euskadi|pse-?ee|psoe", "PSE-EE"),
    (r"euskal herria bildu|eh ?bildu|bildu-eusko|alternatiba eraikitzen", "EH BILDU"),
    (r"partido popular|\(pp\)", "PP"),
    (r"elkarrekin|podemos|ezker anitza|equo", "ELKARREKIN BILBAO"),
    (r"udalberri", "UDALBERRI"),
    (r"goazen", "GOAZEN BILBAO"),
    (r"ganemos", "GANEMOS"),
    (r"ezker batua|ezker anitza-iu", "EZKER BATUA-IU"),
    (r"vox", "VOX"),
]

SKIP = re.compile(
    r"secretar[ií]a general|udalbatzarreko|idazkari|^-\s*\d+\s*-$|jaun/andreok|"
    r"sr\.?\s*secretario|acto seguido|mesa de edad|hitza ematen|zin egiten|"
    r"presidencia|constituci[oó]n|^-+$|hautetsi|^\d+\.|electoral", re.I)


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


def lines_in_order(path, npages=4):
    r = PdfReader(path)
    out = []
    for pi in range(min(npages, len(r.pages))):
        parts = []
        r.pages[pi].extract_text(
            visitor_text=lambda t, cm, tm, f, s: parts.append((round(tm[5], 1), round(tm[4], 1), t))
            if t.strip() else None)
        parts.sort(key=lambda z: (-z[0], z[1]))
        cur, buf = None, []
        for y, x, tx in parts:
            if cur is None or abs(y - cur) < 3:
                buf.append((x, tx)); cur = y if cur is None else cur
            else:
                out.append("".join(t for _, t in sorted(buf))); buf = [(x, tx)]; cur = y
        if buf:
            out.append("".join(t for _, t in sorted(buf)))
    return out


def es_nombre(ln):
    ln = ln.strip(" .,-:")
    if not (6 <= len(ln) <= 55) or " " not in ln:
        return None
    letras = re.sub(r"[^A-Za-zÁÉÍÓÚÑÜ' -]", "", ln)
    if len(letras) < len(ln) - 1:
        return None
    # mayoritariamente mayusculas
    up = sum(1 for c in ln if c.isupper())
    lo = sum(1 for c in ln if c.islower())
    if up < 4 or lo > up:
        return None
    if SKIP.search(ln):
        return None
    return norm(ln).upper()


def detecta_partido(ln):
    low = norm(ln).lower()
    if ":" not in ln and "(" not in ln:
        return None
    for rx, canon in PARTIDOS:
        if re.search(rx, low):
            return canon
    return None


def constitutivas():
    HARD_STOP = re.compile(r"sr\.?\s*secretario|mesa de edad|acto seguido|"
                           r"comprobaci[oó]n de las credenciales|juramento o promesa", re.I)
    res = {}
    for mandato, path in ACTAS.items():
        tabla = {}
        cur = None
        for ln in lines_in_order(path):
            if HARD_STOP.search(ln):
                cur = None
                continue
            p = detecta_partido(ln)
            if p:
                cur = p
                continue
            if cur is None:
                continue
            nom = es_nombre(ln)
            if nom:
                tabla.setdefault(cur, []).append(nom)
        res[mandato] = tabla
    print(json.dumps(res, ensure_ascii=False, indent=1))


# =============================================================================
# Proposiciones (fase 1)
# =============================================================================

# Fase 1: una entrada por proposición de cada acta (proposals.jsonl), con el
# mismo troceo que el índice vectorial: título, texto completo, fecha, resultado
# y listas de voto nominal.
#
#     python -m grafo.construccion.extract_proposals

OUT_PATH = dato_grafo("proposals.jsonl")
# 30.000 chars ~ 8-9k tokens: cabe de sobra en el contexto de Gemini o Groq.
# El límite real de cuánto se le pasa al LLM lo pone build_graph.py::_ctx_chars
# según el modelo (los locales pequeños reciben menos). Este tope solo evita
# guardar textos gigantes en proposals.jsonl.
MAX_TEXT = 30000
SKIP_TOPICS = {"", "General", "General / Introducción"}


# quita el prefijo 'ASUNTO: ...\nORADOR: ...\n\n' que añade el indexador
def _strip_header(chunk_text: str) -> str:
    parts = chunk_text.split("\n\n", 1)
    return parts[1] if len(parts) == 2 and parts[0].startswith("ASUNTO:") else chunk_text


# une chunks consecutivos solapados (chunk_overlap=200) sin repetir el solape
def _merge_overlapping(texts):
    merged = ""
    for t in texts:
        t = t.strip()
        if not merged:
            merged = t
            continue
        # buscar el mayor solape entre el final de `merged` y el inicio de `t`
        overlap = 0
        maxo = min(len(merged), len(t), 400)
        for k in range(maxo, 20, -1):
            if merged[-k:] == t[:k]:
                overlap = k
                break
        merged += t[overlap:]
    return merged


def extract_proposals():
    rag = RAGPipeline()
    import glob
    pdfs = sorted(glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True))
    print(f"[*] {len(pdfs)} actas en {DATA_PATH}")

    proposals = []
    for path in pdfs:
        try:
            chunks = rag._process_single_pdf(path)
        except Exception as e:
            print(f"[!] {os.path.basename(path)}: {e}")
            continue

        # agrupar por tema dentro de esta acta (cada tema = una proposición/punto)
        by_topic = defaultdict(list)
        for c in chunks:
            by_topic[c.metadata.get("topic", "")].append(c)

        for topic, cs in by_topic.items():
            if topic in SKIP_TOPICS:
                continue
            cs = sorted(cs, key=lambda c: c.metadata.get("chunk_index", 0))
            date = cs[0].metadata.get("date", "")
            pages = [c.metadata.get("page") for c in cs if c.metadata.get("page")]
            parties = [c.metadata.get("party") for c in cs
                       if c.metadata.get("party") and c.metadata.get("party") != "Desconocido"]
            party = Counter(parties).most_common(1)[0][0] if parties else "Desconocido"
            # el ÚLTIMO vote_result no vacío entre los chunks del topic (ya
            # ordenados por chunk_index = orden real en el acta), no el
            # primero: en proposiciones con varias enmiendas compitiendo, el
            # acta registra un voto por cada enmienda antes del resultado
            # final -- quedarse con el primero capturaba un voto intermedio
            # (p.ej. "rechazada la enmienda de GOAZEN BILBAO") en vez del que
            # decide de verdad la suerte de la proposición ("se acepta la
            # enmienda del Equipo de Gobierno, por lo que decaen... la
            # proposición del PP"). Verificado contra el PDF real (Ronda 40,
            # 2026-09-16, 24-09-2015 ítem 49): con "primero" se perdía el
            # "decaen" final y la proposición quedaba mal clasificada como
            # "aprobada con enmienda" por el fallback del LLM.
            votos_topic = [c.metadata.get("vote_result") for c in cs if c.metadata.get("vote_result")]
            vote = votos_topic[-1] if votos_topic else None
            texto_completo = _merge_overlapping([_strip_header(c.page_content) for c in cs])
            text = texto_completo[:MAX_TEXT]
            # Capa determinista: listas nominales de voto tal cual las escribe el
            # acta ("Votos afirmativos: 14 señoras/señores: ..."). Se parsea del
            # texto COMPLETO (sin recortar a MAX_TEXT) porque la votación va al
            # final de la proposición. build_rdf.py resuelve apellido->concejal->grupo.
            votos_nominales = parse_listas_voto(texto_completo)
            # Apellidos de los concejales que intervinieron en el debate de esta
            # proposición (el indexador ya los detecta con speaker_regex). Se
            # usan en build_rdf.py para enlazar la proposición con los nodos
            # :Concejal del censo (ver concejales.jsonl).
            oradores = sorted({
                c.metadata.get("speaker") for c in cs
                if c.metadata.get("speaker") and c.metadata.get("speaker") not in ("Desconocido", "")
            })

            proposals.append({
                "date": date,
                "topic": topic,
                "party": party,
                "vote_result": vote,
                "page_ini": min(pages) if pages else None,
                "page_fin": max(pages) if pages else None,
                "source": os.path.relpath(path, os.path.dirname(DATA_PATH)),
                "text": text,
                "votos_nominales": votos_nominales,
                "oradores": oradores,
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in proposals:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # resumen
    con_voto = sum(1 for p in proposals if p["vote_result"])
    años = Counter(p["date"].split("-")[-1] for p in proposals if p["date"])
    print(f"[+] {len(proposals)} proposiciones -> {OUT_PATH}")
    print(f"    con vote_result: {con_voto} ({100*con_voto//max(len(proposals),1)}%)")
    print(f"    por año: {dict(sorted(años.items()))}")
    return proposals


# =============================================================================
# Puntos que faltaban y votaciones de todos los puntos
# =============================================================================

# Lo que la extracción original no recogió, sacado del texto completo de cada
# acta (texto_actas.py), y las votaciones de todos los puntos.
#
#   proposals_extra.jsonl  puntos del orden del día que no estaban en el grafo
#                          (iniciativas vecinales desde 2022, extraordinarias de
#                          punto único...), en el formato de proposals.jsonl, para
#                          enriquecerlos con el LLM (build_graph.py --extra)
#   votaciones.jsonl       cada votación de cada punto ("Se somete a votación la
#                          enmienda / la proposición ... En su virtud, ..."), con
#                          su objeto, recuento, listas nominales y decisión. Antes
#                          solo se guardaba una por punto, y en 150 era la de una
#                          enmienda.
#
# Reanudable por acta (votaciones.jsonl y proposals_extra.jsonl se reescriben
# enteros al final; la caché de texto está en cache/pdf_texto).

ENRICHED = dato_grafo("proposals_enriched.jsonl")
OUT_EXTRA = dato_grafo("proposals_extra.jsonl")
OUT_VOT = dato_grafo("votaciones.jsonl")


def _num(topic: str):
    m = re.match(r"\W*(\d{1,3})\b", topic or "")
    return int(m.group(1)) if m else None


# --- votaciones ---

_SOMETE = re.compile(r"se\s+somete[n]?\s+a\s+votaci[óo]n(?:\s+nominal)?\s+(.{5,500}?)(?:,?\s+siendo\s+el\s+c[óo]mputo|,?\s+con\s+el\s+(?:siguiente\s+)?resultado|:\s*votos)", re.I)
_EMITIDOS = re.compile(r"votos\s+emitidos\s*:?\s*(\d+)", re.I)
_CIFRA = {"favor": re.compile(r"votos\s+afirmati\s?vos\s*:?\s*(\d+)", re.I),
          "contra": re.compile(r"votos\s+negativos\s*:?\s*(\d+)", re.I),
          "abst": re.compile(r"abstenciones\s*:?\s*(\d+)", re.I)}
_VIRTUD = re.compile(r"en\s+su\s+virt\s?ud,?\s+([^.]{5,400})", re.I)


def _objeto(txt: str) -> str:
    t = txt.lower()
    if re.search(r"\benmiendas?\b", t) and not re.search(r"^(la|el)\s+(proposici|propuesta|iniciativa|moci)", t):
        return "enmienda"
    if re.search(r"\bpuntos?\b", t):
        return "por_puntos"
    if re.search(r"proposici|propuesta|iniciativa|moci[óo]n|dictamen|presupuesto|proyecto|acuerdo", t):
        return "punto"
    return "otro"


# [{"orden", "objeto", "descripcion", "emitidos", "favor", "contra", "abst", "listas", "decision"}]
def votaciones(texto: str):
    plano = re.sub(r"\s+", " ", texto)
    ms = list(_SOMETE.finditer(plano))
    out = []
    for i, m in enumerate(ms):
        fin = ms[i + 1].start() if i + 1 < len(ms) else min(len(plano), m.end() + 6000)
        ventana = plano[m.end():fin]
        v = {"orden": i + 1, "descripcion": m.group(1).strip()[:400], "objeto": _objeto(m.group(1))}
        e = _EMITIDOS.search(ventana)
        v["emitidos"] = int(e.group(1)) if e else None
        for k, rx in _CIFRA.items():
            c = rx.search(ventana)
            v[k] = int(c.group(1)) if c else (0 if e else None)
        d = _VIRTUD.search(ventana)
        v["decision"] = d.group(1).strip()[:400] if d else ""
        # listas nominales (las del bloque en castellano)
        listas = parse_listas_voto(ventana[:d.end()] if d else ventana)
        v["listas"] = listas
        out.append(v)
    return out


# --- puntos que faltan ---

def _titulo_unico(texto: str) -> str:
    plano = re.sub(r"\s+", " ", texto)
    for rx in (r"punto [úu]nico del orden del d[íi]a:?\s*[“\"«]?(?:[^“\"«]{0,200}?[\.»”\"]\s*)?([A-ZÁÉÍÓÚÑ][^”\"»]{15,300})",
               r"(Debate sobre el Estado de la Ciudad[^.]{0,120})",
               r"[úu]nico punto del orden del d[íi]a[^.]{0,40}\.\s*(.{15,250}?)\s-",
               r"ORDEN DEL D[ÍI]A:?\s*(.{15,300}?)\s(?:SR\.|SRA\.|-)"):
        m = re.search(rx, plano, re.I)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip(" .:-")[:300]
    return ""


def _texto_para_llm(texto: str) -> str:
    # la votación va al final: con textos largos, el principio y el final
    if len(texto) <= MAX_TEXT:
        return texto
    return texto[:MAX_TEXT - 9000] + "\n[...]\n" + texto[-9000:]


def extraer_extra():
    sys.stdout.reconfigure(encoding="utf-8")
    from vectorial.indexado import _SPEAKER_RE, _extraer_resultado
    from comun.grupos import prop_id

    existentes = defaultdict(dict)   # pdf -> numero -> [ids]
    sin_numero = defaultdict(list)
    titulos = defaultdict(set)       # fecha -> títulos normalizados (para no duplicar con otra numeración)

    def _clave_titulo(t):
        import unicodedata
        t = unicodedata.normalize("NFD", (t or "").lower())
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        t = re.sub(r"^\W*-?\s*\d+\s*-?\s*\.?-?\s*", "", t)
        return re.sub(r"[^a-z0-9]+", "", t)[:40]
    huerfanos = defaultdict(list)    # pdf -> [(página, id, número)] de los puntos del grafo
    paginas_ini = set()              # (pdf, página donde empieza el punto): el mismo punto con otra numeración
    for r in iter_jsonl(ENRICHED):
        titulos[r.get("date")].add(_clave_titulo(r.get("topic")))
        paginas_ini.add((os.path.basename(str(r.get("source", "")).replace("\\", "/")), str(r.get("page_ini"))))
        try:
            huerfanos[os.path.basename(str(r.get("source", "")).replace("\\", "/"))].append(
                (int(r.get("page_ini")), r["id"], _num(r.get("topic"))))
        except (TypeError, ValueError):
            pass
        pdf = os.path.basename(str(r.get("source", "")).replace("\\", "/"))
        n = _num(r.get("topic"))
        if n is None:
            sin_numero[pdf].append(r["id"])
        else:
            existentes[pdf].setdefault(n, []).append(r["id"])

    extra, vots = [], []
    nuevos_por_acta = {}
    for i, pdf in enumerate(T.actas(), 1):
        nombre = os.path.basename(pdf)
        fecha = T.fecha(pdf)
        source = os.path.relpath(pdf, os.path.dirname(T.ACTAS))
        segs = T.puntos(pdf)
        # punto único solo para las actas de las que el grafo no tiene ningún
        # punto (las extraordinarias de un solo asunto); si el segmentador no
        # reconoce el formato de un acta que sí está en el grafo, no se inventa
        ya_en_grafo = bool(existentes.get(nombre) or sin_numero.get(nombre))
        if len(segs) <= 1 and ya_en_grafo:
            segs = [s for s in segs if s["numero"] in existentes.get(nombre, {})]
        if not segs and not ya_en_grafo:
            # extraordinaria de punto único (debate del estado de la ciudad,
            # presupuestos, medalla de oro...): el acta entera es el punto
            texto = "\n".join(T.paginas(pdf))
            segs = [{"numero": 0, "titulo": _titulo_unico(texto), "texto": texto, "pagina_ini": 1,
                     "pagina_fin": len(T.paginas(pdf))}]
        nuevos = 0
        # Un segmento que se ha tragado puntos que el grafo sí tiene (el
        # segmentador no reconoció sus encabezados) mezclaría sus votaciones: no
        # se usa
        numeros = sorted(s["numero"] for s in segs)
        en_grafo = set(existentes.get(nombre, {}))
        contaminados = set()
        for a, b in zip(numeros, numeros[1:] + [10 ** 6]):
            if any(a < e < b for e in en_grafo):
                contaminados.add(a)
        for s in segs:
            n = s["numero"]
            if n in contaminados:
                continue
            ids = existentes.get(nombre, {}).get(n) or ([] if n else sin_numero.get(nombre, []))
            if not ids and n:
                # punto del grafo con la numeración corrupta ("116." es la página en que
                # empieza): se enlaza si es el único punto sin número del acta dentro de
                # las páginas de este segmento
                try:
                    rango = range(int(s["pagina_ini"]), int(s["pagina_fin"]) + 1)
                except (TypeError, ValueError):
                    rango = range(0)
                # corrupta = el "número" es la página donde empieza ("116." en la página 116)
                # y solo al segmento que empieza más cerca de su página (si hay dos solapados,
                # el punto sería de los dos)
                def _mejor(pg):
                    return max((int(x["pagina_ini"]) for x in segs if int(x["pagina_ini"]) <= pg), default=-1)
                cand = [pid for pg, pid, num in huerfanos.get(nombre, [])
                        if pg in rango and num == pg and num not in numeros and _mejor(pg) == int(s["pagina_ini"])]
                if len(cand) == 1:
                    ids = cand
            vs = votaciones(s["texto"])
            if ids:
                for pid in ids:
                    vots.append({"id": pid, "pdf": nombre, "numero": n, "votaciones": vs})
                continue
            if len(s["texto"]) < 200:
                continue
            # ya está en el grafo con otra numeración (mismo título, misma fecha)
            # (por la página de inicio: el título de casi todas las proposiciones
            # empieza igual y con él se descartaban las que el grafo no tenía)
            if (nombre, str(s["pagina_ini"])) in paginas_ini:
                continue
            titulo = s["titulo"] if n else (f"0. PUNTO ÚNICO: {s['titulo']}" if s["titulo"] else "0. PUNTO ÚNICO de la sesión extraordinaria")
            oradores = sorted({m.group(1).strip() for m in _SPEAKER_RE.finditer(s["texto"]) if len(m.group(1)) < 50})
            rec = {"date": fecha, "topic": titulo, "party": "Desconocido",
                   "vote_result": _extraer_resultado(s["texto"]), "page_ini": s["pagina_ini"],
                   "page_fin": s["pagina_fin"], "source": source, "text": _texto_para_llm(s["texto"]),
                   "votos_nominales": parse_listas_voto(s["texto"]), "oradores": oradores, "origen": "segmentar"}
            rec["id"] = prop_id(rec)
            extra.append(rec)
            vots.append({"id": rec["id"], "pdf": nombre, "numero": n, "votaciones": vs})
            nuevos += 1
        nuevos_por_acta[nombre] = nuevos
        if i % 25 == 0:
            print(f"[{i}] {nombre}: {len(segs)} puntos, {nuevos} nuevos (total nuevos {len(extra)})", flush=True)

    for ruta, filas in ((OUT_EXTRA, extra), (OUT_VOT, vots)):
        tmp = ruta + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in filas:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, ruta)
    print(f"[+] {len(extra)} puntos nuevos -> {OUT_EXTRA}")
    print(f"[+] votaciones de {len(vots)} puntos ({sum(len(v['votaciones']) for v in vots)} votaciones) -> {OUT_VOT}")
    print("[*] actas con más puntos nuevos:", sorted(nuevos_por_acta.items(), key=lambda x: -x[1])[:15])


PASOS = {"concejales": extraer_concejales, "constitutivas": constitutivas,
         "proposiciones": extract_proposals, "extra": extraer_extra}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PASOS:
        sys.exit("Uso: python -m grafo.construccion.extraer " + "|".join(PASOS))
    PASOS[sys.argv.pop(1)]()
