"""
Plantillas de consulta sin LLM (el "brazo G" del estudio de generación de
SPARQL): la pregunta se compara por similitud de embeddings (bge-m3) con las
frases de ejemplo de un catálogo de formas (conteo, ranking por grupo, ratio,
evolución temporal...), se extraen con expresiones regulares sus huecos (tema,
grupo, año, resultado...) y se rellena la plantilla SPARQL de esa forma.

Se usa después de la consulta directa (consulta_directa.py) y antes de que el
LLM genere la consulta (generacion.py). Si la forma no se reconoce, o reconocida
no tiene plantilla, devuelve sparql=None y la pregunta pasa al LLM.
"""
import math
import re

# --- similitud de embeddings con las frases del catálogo ---

_embedder = None
_q_cache = {}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from langchain_ollama import OllamaEmbeddings
        _embedder = OllamaEmbeddings(model="bge-m3")
    return _embedder


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def _embed_q(pregunta):
    if pregunta in _q_cache:
        return _q_cache[pregunta]
    v = _get_embedder().embed_query(pregunta)
    _q_cache[pregunta] = v
    return v


# ---------------------------------------------------------------- vocabulario
_TEMAS = [  # (patrón regex, uri) -- orden importa: más específico primero
    (r"vivienda\s+social", "t_vivienda_social"),
    (r"vivienda\s+vac[ií]a", "t_vivienda_vacia"),
    (r"desahuci", "t_desahucios"),
    (r"alquiler", "t_alquiler"),
    (r"bicicleta", "t_bicicleta"),
    (r"vivienda", "t_vivienda"),
    (r"urbanismo", "t_urbanismo"),
    (r"movilidad|transporte", "t_movilidad"),
    (r"medio\s*ambiente|medioambient", "t_medioambiente"),
    (r"euskera|euskara", "t_euskera"),
    (r"cultura", "t_cultura"),
    (r"deporte", "t_deporte"),
    (r"educaci[oó]n", "t_educacion"),
    (r"igualdad|feminismo", "t_igualdad"),
    (r"servicios?\s+sociales", "t_serviciossociales"),
    (r"empleo|econom[ií]a", "t_empleoeconomia"),
    (r"presupuestos?|fiscalidad", "t_presupuestos"),
    (r"seguridad", "t_seguridad"),
    (r"participaci[oó]n", "t_participacion"),
    (r"turismo", "t_turismo"),
    (r"sanidad|salud", "t_sanidad"),
    (r"memoria\s+hist[oó]rica", "t_memoriahistorica"),
    (r"derechos\s+humanos", "t_derechoshumanos"),
]

_GRUPOS = [  # (patrón regex, uri, label) -- multi-palabra antes que sigla suelta
    (r"partido\s+popular|\bPP\b", "grupo_pp", "PP"),
    (r"eh[\s-]?bildu|\bbildu\b", "grupo_eh_bildu", "EH BILDU"),
    (r"pse-?ee|socialistas", "grupo_pse_ee", "PSE-EE"),
    (r"elkarrekin(\s+bilbao)?", "grupo_elkarrekin_bilbao", "ELKARREKIN BILBAO"),
    (r"goazen(\s+bilbao)?", "grupo_goazen_bilbao", "GOAZEN BILBAO"),
    (r"udalberri", "grupo_udalberri", "UDALBERRI"),
    (r"eaj-?pnv|\bpnv\b", "grupo_eaj_pnv", "EAJ-PNV"),
    (r"\b(?-i:Ciudadanos|Cs|C's)\b", "grupo_ciudadanos", "CIUDADANOS"),
    (r"\bvox\b", "grupo_vox", "VOX"),
    (r"\baralar\b", "grupo_aralar", "ARALAR"),
    (r"ezker\s+batua(-iu)?", "grupo_ezker_batua_iu", "EZKER BATUA-IU"),
    (r"equipo\s+de\s+gobierno", "grupo_equipo_de_gobierno", "EQUIPO DE GOBIERNO"),
    (r"grupo\s+mixto", "grupo_grupo_mixto", "GRUPO MIXTO"),
]

_STOP_ENT = {"bilbao", "pleno", "ayuntamiento", "se", "en", "el", "la", "algún",
             "algun", "alguno", "que", "cual", "cuál", "del", "de",
             # interrogativos: al empezar la pregunta van en mayúscula
             "qué", "cuándo", "cuando", "cómo", "como", "dónde", "donde", "quién", "quien",
             "cuántas", "cuántos", "cuantas", "cuantos", "cuáles", "cuales", "por", "ha", "han"}


def extract_tema(q):
    ql = q.lower()
    for pat, uri in _TEMAS:
        if re.search(pat, ql):
            return uri
    return None


def extract_grupo(q):
    for pat, uri, _lbl in _GRUPOS:
        if re.search(pat, q, re.I):
            return uri
    return None


# TODOS los grupos mencionados en la pregunta, en el orden en que aparecen en
# el texto (no en el orden de _GRUPOS) -- para plantillas que comparan varios
# grupos a la vez, a diferencia de extract_grupo() que solo quiere el primero
# que casa en _GRUPOS.
def extract_grupos_multi(q):
    encontrados = []
    for pat, uri, _lbl in _GRUPOS:
        m = re.search(pat, q, re.I)
        if m:
            encontrados.append((m.start(), uri))
    encontrados.sort(key=lambda x: x[0])
    return [uri for _pos, uri in encontrados]


# cuántos grupos DISTINTOS menciona la pregunta -- extract_grupo() solo
# devuelve el primero que aparece en _GRUPOS (no el primero en la frase), así
# que una pregunta que compara 2 grupos ("EH Bildu o el PP") monta una
# consulta que solo filtra por uno de los dos sin avisar. Detectado en la
# Ronda 39 (2026-09-16) con g22 del estudio: Arm G generaba una consulta
# válida y con filas, pero SOLO sobre el PP, ignorando EH Bildu por completo.
def _n_grupos_mencionados(q):
    return sum(1 for pat, _uri, _lbl in _GRUPOS if re.search(pat, q, re.I))


def extract_anio(q):
    m = re.search(r"\b(19|20)\d{2}\b", q)
    return int(m.group(0)) if m else None


def extract_resultado(q):
    ql = q.lower()
    if "sin resultado" in ql or "no llegaron a tener un resultado" in ql:
        return "sinresultado"
    if re.search(r"aprob", ql):     # aprobada/aprobó/aprobaron/aprobación
        return "aprobada"
    if re.search(r"rechaz", ql):    # rechazada/rechazó/rechazaron
        return "rechazada"
    return None


def extract_direccion(q):
    ql = q.lower()
    if "abstuv" in ql or "abstenci" in ql:
        return "abstencion"
    if "en contra" in ql:
        return "contra"
    if "a favor" in ql:
        return "favor"
    return None


def extract_limit(q):
    ql = q.lower()
    if re.search(r"\btres\b|\b3\b", ql):
        return 3
    if re.search(r"\bdos\b|\b2\b", ql):
        return 2
    if re.search(r"cada\s+grupo|todos\s+los\s+grupos", ql):
        return 20
    return 1


def extract_entidad(q):
    for w in re.findall(r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}", q):
        if w.lower() not in _STOP_ENT:
            return w
    return None


# --------------------------------------------------------------- plantillas
def _resultado_filter(res, var_p="?p", var_r="?r"):
    if res == "aprobada":
        return f"{var_p} bo:tieneResultado {var_r} . FILTER({var_r} IN (bo:Aprobada, bo:AprobadaConEnmienda))"
    if res == "rechazada":
        return f"{var_p} bo:tieneResultado bo:Rechazada"
    if res == "sinresultado":
        return f"{var_p} bo:tieneResultado bo:SinResultado"
    return None


def tpl_conteo(q, slots):
    tema, grupo, anio, res = slots.get("tema"), slots.get("grupo"), slots.get("anio"), slots.get("resultado")
    if not any([tema, grupo, anio, res]):
        return None
    where = ["?p a bo:Proposicion"]
    if tema:
        where.append(f"bo:trataTemaAmplio br:{tema}")
    if grupo:
        where.append(f"bo:presentadaPor br:{grupo}")
    if anio:
        where.append(f"bo:anio {anio}")
    body = " ; ".join(where) + " ."
    rf = _resultado_filter(res)
    if rf:
        body += f" {rf} ."
    return f"SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {{ {body} }}"


def tpl_total_simple(q, slots):
    return "SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE { ?p a bo:Proposicion . }"


def tpl_ranking_grupo(q, slots):
    tema, limit = slots.get("tema"), slots.get("limit") or 10
    where = ["?p a bo:Proposicion"]
    if tema:
        where.append(f"bo:trataTemaAmplio br:{tema}")
    where.append("bo:presentadaPor ?g")
    body = " ; ".join(where) + " . ?g rdfs:label ?ng . FILTER(?g != br:grupo_desconocido)"
    return (f"SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {{ {body} }} "
            f"GROUP BY ?ng ORDER BY DESC(?n) LIMIT {limit}")


def tpl_ranking_enmiendas(q, slots):
    return ("SELECT ?ng (COUNT(DISTINCT ?e) AS ?n) WHERE { "
            "?e a bo:Enmienda ; bo:enmiendaPor ?g . ?g rdfs:label ?ng . "
            "FILTER(?g != br:grupo_desconocido) } GROUP BY ?ng ORDER BY DESC(?n) LIMIT 20")


def tpl_conteo_enmienda_grupo(q, slots):
    grupo = slots.get("grupo")
    if not grupo:
        return None
    return f"SELECT (COUNT(DISTINCT ?e) AS ?n) WHERE {{ ?e a bo:Enmienda ; bo:enmiendaPor br:{grupo} . }}"


def tpl_temporal_top(q, slots):
    tema, res = slots.get("tema"), slots.get("resultado")
    where = ["?p a bo:Proposicion"]
    if tema:
        where.append(f"bo:trataTemaAmplio br:{tema}")
    where.append("bo:anio ?anio")
    body = " ; ".join(where) + " ."
    rf = _resultado_filter(res)
    if rf:
        body += f" {rf} ."
    return (f"SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {{ {body} }} "
            f"GROUP BY ?anio ORDER BY DESC(?n) LIMIT 1")


def tpl_temporal_evol(q, slots):
    tema = slots.get("tema")
    where = ["?p a bo:Proposicion"]
    if tema:
        where.append(f"bo:trataTemaAmplio br:{tema}")
    where.append("bo:anio ?anio")
    body = " ; ".join(where) + " ."
    return f"SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {{ {body} }} GROUP BY ?anio ORDER BY ASC(?anio)"


def tpl_ratio(q, slots):
    tema, grupo, anio, res = slots.get("tema"), slots.get("grupo"), slots.get("anio"), slots.get("resultado") or "aprobada"
    where = ["?p a bo:Proposicion"]
    if tema:
        where.append(f"bo:trataTemaAmplio br:{tema}")
    if grupo:
        where.append(f"bo:presentadaPor br:{grupo}")
    if anio:
        where.append(f"bo:anio {anio}")
    body = " ; ".join(where) + " ."
    if res == "aprobada":
        opt = "OPTIONAL { ?p bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) . BIND(?p AS ?sub) }"
    else:
        opt = "OPTIONAL { ?p bo:tieneResultado bo:Rechazada . BIND(?p AS ?sub) }"
    return (f"SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?sub) AS ?sub) "
            f"WHERE {{ {body} {opt} }}")


# comparar 2+ grupos nombrados en la misma pregunta ("¿qué grupo tiene mejor
# tasa de aprobación, EH Bildu o el PP?") -- UNA fila por grupo, mismo tema/
# año/resultado para los dos, con VALUES en vez de repetir la consulta. Antes
# esta forma se delegaba siempre al LLM (ver _n_grupos_mencionados en
# answer()); el LLM la resolvía de forma no fiable (predicado equivocado,
# o solo comparaba UN grupo aunque generara filas para los dos, o añadía un
# filtro de año no pedido) -- confirmado como el único caso que seguía
# fallando en la regresión tras la Ronda 39. Ronda 41, 2026-09-17.
def tpl_ratio_multigrupo(q, slots):
    tema, anio, res = slots.get("tema"), slots.get("anio"), slots.get("resultado") or "aprobada"
    grupos = slots.get("grupos_multi") or []
    if len(grupos) < 2:
        return None
    where = ["?p a bo:Proposicion"]
    if tema:
        where.append(f"bo:trataTemaAmplio br:{tema}")
    if anio:
        where.append(f"bo:anio {anio}")
    where.append("bo:presentadaPor ?g")
    body = " ; ".join(where) + " . ?g rdfs:label ?ng ."
    values = " ".join(f"br:{gr}" for gr in grupos)
    if res == "aprobada":
        opt = "OPTIONAL { ?p bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) . BIND(?p AS ?sub) }"
    else:
        opt = "OPTIONAL { ?p bo:tieneResultado bo:Rechazada . BIND(?p AS ?sub) }"
    return (f"SELECT ?ng (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?sub) AS ?sub) WHERE {{ "
            f"VALUES ?g {{ {values} }} {body} {opt} }} GROUP BY ?g ?ng")


def tpl_entidad(q, slots):
    ent = slots.get("entidad")
    if not ent:
        return None
    return (f'SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {{ '
            f'?p bo:menciona ?e . ?e rdfs:label ?n . FILTER(REGEX(STR(?n), "\\\\b{ent}\\\\b", "i")) }}')


def tpl_voto_grupo(q, slots):
    grupo, tema, direccion = slots.get("grupo"), slots.get("tema"), slots.get("direccion")
    if not grupo or not direccion:
        return None
    pred = {"favor": "votoAFavorDe", "contra": "votoEnContraDe", "abstencion": "seAbstuvo"}[direccion]
    where = ["?p a bo:Proposicion"]
    if tema:
        where.append(f"bo:trataTemaAmplio br:{tema}")
    where.append(f"bo:{pred} br:{grupo}")
    body = " ; ".join(where) + " ."
    return f"SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {{ {body} }}"


def tpl_coautoria(q, slots):
    return "SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE { ?p bo:presentadaPor ?a, ?b . FILTER(?a != ?b) }"


def tpl_tema_extremo(q, slots):
    orden = "ASC" if "menos" in q.lower() else "DESC"
    return (f"SELECT ?lab (COUNT(DISTINCT ?p) AS ?n) WHERE {{ "
            f"?p a bo:Proposicion ; bo:trataSobre ?t . ?t skos:prefLabel ?lab . }} "
            f"GROUP BY ?lab ORDER BY {orden}(?n) LIMIT 1")


# -------------------------------------------------------- catálogo de formas
# (id, [frases prototipo para el embedding], builder|None, slots_usados)
CATALOG = [
    ("conteo", [
        "¿Cuántas proposiciones sobre un tema se han presentado?",
        "¿Cuántas proposiciones presentó un grupo en un año?",
        "¿Cuántas proposiciones sobre un tema ha presentado un grupo?",
        "¿Cuántas proposiciones se han aprobado o rechazado sobre un tema?",
    ], tpl_conteo, ["tema", "grupo", "anio", "resultado"]),
    ("total_simple", [
        "¿Cuántas proposiciones hay en total?",
        "¿Cuántas proposiciones no llegaron a tener un resultado registrado?",
    ], tpl_total_simple, []),
    ("ranking_grupo", [
        "¿Qué grupo ha presentado más proposiciones sobre un tema?",
        "¿Qué tres grupos han presentado más proposiciones sobre un tema?",
        "¿Cuántas proposiciones ha presentado cada grupo en total?",
    ], tpl_ranking_grupo, ["tema", "limit"]),
    ("ranking_enmiendas", [
        "¿Qué grupo ha presentado más enmiendas en total?",
    ], tpl_ranking_enmiendas, []),
    ("conteo_enmienda_grupo", [
        "¿Cuántas enmiendas ha presentado un grupo en total?",
    ], tpl_conteo_enmienda_grupo, ["grupo"]),
    ("temporal_top", [
        "¿En qué año se presentaron más proposiciones sobre un tema?",
        "¿En qué año se rechazaron más proposiciones?",
    ], tpl_temporal_top, ["tema", "resultado"]),
    ("temporal_evol", [
        "¿Cómo ha evolucionado el número de proposiciones sobre un tema por año?",
    ], tpl_temporal_evol, ["tema"]),
    ("ratio", [
        "¿Cuántas proposiciones presentó un grupo en un año y cuántas se aprobaron?",
        "¿Qué porcentaje de las proposiciones sobre un tema se han aprobado?",
        "¿Qué porcentaje de las proposiciones sobre un tema se han rechazado?",
    ], tpl_ratio, ["tema", "grupo", "anio", "resultado"]),
    ("entidad", [
        "¿Se ha mencionado a una empresa o lugar en algún pleno?",
    ], tpl_entidad, ["entidad"]),
    ("voto_grupo", [
        "¿Cuántas veces votó a favor un grupo en proposiciones sobre un tema?",
        "¿Cuántas veces se abstuvo un grupo en proposiciones sobre un tema?",
        "¿Cuántas veces votó en contra un grupo en proposiciones sobre un tema?",
    ], tpl_voto_grupo, ["grupo", "tema", "direccion"]),
    ("coautoria", [
        "¿Cuántas proposiciones conjuntas entre varios grupos ha habido?",
    ], tpl_coautoria, []),
    ("tema_extremo", [
        "¿Cuál es el tema más tratado en las proposiciones?",
        "¿Cuál es el tema menos tratado en las proposiciones?",
    ], tpl_tema_extremo, []),
    # formas reconocidas pero NO cubiertas por plantilla (persona / voto nominal)
    ("persona", [
        "¿Qué concejal ha presentado o firmado más proposiciones?",
        "¿En cuántos debates ha intervenido un concejal?",
        "¿Quién ha sido alcalde de Bilbao?",
    ], None, []),
    ("voto_nominal", [
        "¿Qué concejal ha votado más veces en contra?",
        "¿Cuántas veces ha votado en contra un concejal concreto?",
        "¿Cómo votó un grupo las proposiciones sobre un tema?",
    ], None, []),
]

_proto_vecs = None
_proto_index = None  # lista paralela de (form_id, builder, slots_usados)


def _build_index():
    global _proto_vecs, _proto_index
    if _proto_vecs is not None:
        return
    texts, idx = [], []
    for form_id, phrases, builder, slots in CATALOG:
        for ph in phrases:
            texts.append(ph)
            idx.append((form_id, builder, slots))
    emb = _get_embedder()
    _proto_vecs = emb.embed_documents(texts)
    _proto_index = idx


def match_form(pregunta, threshold=0.55):
    _build_index()
    qv = _embed_q(pregunta)
    best = (-1.0, None)
    for v, (form_id, builder, slots) in zip(_proto_vecs, _proto_index):
        s = _cos(qv, v)
        if s > best[0]:
            best = (s, (form_id, builder, slots))
    sim, info = best
    if sim < threshold or info is None:
        return None, sim
    return info, sim


_RATIO_2COUNT = re.compile(r"\by\s+cu[aá]nt[oa]s\b")
_RATIO_PCT = re.compile(r"porcentaje|\btasa\b|%")
_YEAR_INTERROGATIVE = re.compile(r"qu[eé]\s+a[ñn]o")


def _disambiguate_temporal(pregunta, form_id):
    if form_id in ("temporal_top", "temporal_evol"):
        if extract_anio(pregunta) and not _YEAR_INTERROGATIVE.search(pregunta.lower()):
            return "conteo"
    return form_id


def _disambiguate_conteo_ratio(pregunta, form_id):
    two_counts = bool(_RATIO_2COUNT.search(pregunta.lower())) or bool(_RATIO_PCT.search(pregunta.lower()))
    if two_counts and form_id != "ratio":
        return "ratio"
    if form_id == "ratio" and not two_counts:
        return "conteo"
    return form_id


def answer(pregunta, threshold=0.55):
    info, sim = match_form(pregunta, threshold)
    if info is None:
        return dict(sparql=None, template=None, similarity=round(sim, 3), slots={})
    form_id, builder, slot_names = info
    form_id = _disambiguate_temporal(pregunta, form_id)
    form_id = _disambiguate_conteo_ratio(pregunta, form_id)
    if form_id == "ratio":
        builder, slot_names = tpl_ratio, ["tema", "grupo", "anio", "resultado"]
    elif form_id == "conteo":
        builder, slot_names = tpl_conteo, ["tema", "grupo", "anio", "resultado"]
    slots = {}
    if "direccion" in slot_names:
        slots["direccion"] = extract_direccion(pregunta)
    if "tema" in slot_names:
        slots["tema"] = extract_tema(pregunta)
    if "grupo" in slot_names:
        slots["grupo"] = extract_grupo(pregunta)
    if "anio" in slot_names:
        slots["anio"] = extract_anio(pregunta)
    if "resultado" in slot_names:
        slots["resultado"] = extract_resultado(pregunta)
    if "limit" in slot_names:
        slots["limit"] = extract_limit(pregunta)
    if "entidad" in slot_names:
        slots["entidad"] = extract_entidad(pregunta)
    if "grupo" in slot_names and _n_grupos_mencionados(pregunta) >= 2:
        # pregunta compara 2+ grupos a la vez (p.ej. "EH Bildu o el PP").
        # Para "ratio" (comparar tasas/conteos) SÍ hay plantilla dedicada
        # (tpl_ratio_multigrupo, una fila por grupo con VALUES). Para el
        # resto de formas (conteo/voto_grupo/...) ninguna plantilla admite
        # más de un grupo por consulta; delegar al LLM en vez de responder
        # comparando solo uno en silencio.
        if form_id == "ratio":
            slots["grupos_multi"] = extract_grupos_multi(pregunta)
            sparql = tpl_ratio_multigrupo(pregunta, slots)
            return dict(sparql=sparql, template="ratio_multigrupo", similarity=round(sim, 3), slots=slots)
        return dict(sparql=None, template=form_id, similarity=round(sim, 3), slots=slots)
    if builder is None:
        return dict(sparql=None, template=form_id, similarity=round(sim, 3), slots=slots)
    sparql = builder(pregunta, slots)
    return dict(sparql=sparql, template=form_id, similarity=round(sim, 3), slots=slots)
