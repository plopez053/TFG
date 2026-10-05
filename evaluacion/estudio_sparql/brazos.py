"""
Brazos del estudio de generación de SPARQL: cada brazo es una forma de
construir el prompt (esquema mínimo, vocabulario, prompt completo, ejemplos
elegidos por solapamiento de palabras o por embeddings) y, en los brazos F,
una corrección del SPARQL generado.
"""
import json
import math
import os
import re

from grafo.consulta.generacion import SCHEMA as SCHEMA_FULL


# =============================================================================
# Banco de ejemplos y selección por solapamiento de palabras
# =============================================================================

BANK = [
    dict(q="¿Cuántas proposiciones sobre cultura se han presentado?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_cultura . }"""),

    dict(q="¿Cuántas proposiciones presentó el PSE-EE en 2018?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_pse_ee ; bo:anio 2018 . }"""),

    dict(q="¿Cuántas proposiciones sobre sanidad se han aprobado?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_sanidad ; bo:tieneResultado ?r .
  FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) }"""),

    dict(q="¿Qué grupo ha presentado más proposiciones sobre seguridad?",
         sparql="""SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_seguridad ; bo:presentadaPor ?g .
  ?g rdfs:label ?ng . FILTER(?g != br:grupo_desconocido) }
GROUP BY ?ng ORDER BY DESC(?n) LIMIT 10"""),

    dict(q="¿Cuántas proposiciones ha presentado cada grupo en total?",
         sparql="""SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor ?g . ?g rdfs:label ?ng .
  FILTER(?g != br:grupo_desconocido) }
GROUP BY ?ng ORDER BY DESC(?n) LIMIT 20"""),

    dict(q="¿En qué año hubo más proposiciones sobre cultura?",
         sparql="""SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_cultura ; bo:anio ?anio . }
GROUP BY ?anio ORDER BY DESC(?n) LIMIT 1"""),

    dict(q="¿Cómo ha evolucionado el número de proposiciones sobre movilidad por año?",
         sparql="""SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_movilidad ; bo:anio ?anio . }
GROUP BY ?anio ORDER BY ASC(?anio)"""),

    dict(q="¿Cuántas proposiciones presentó el PP en 2016 y cuántas se aprobaron?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?ap) AS ?aprob) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_pp ; bo:anio 2016 .
  OPTIONAL { ?p bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) . BIND(?p AS ?ap) } }"""),

    dict(q="¿Qué porcentaje de las proposiciones sobre seguridad se rechazaron?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?re) AS ?rech) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_seguridad .
  OPTIONAL { ?p bo:tieneResultado bo:Rechazada . BIND(?p AS ?re) } }"""),

    dict(q="¿Qué concejal o concejala ha intervenido en más debates del Pleno?",
         sparql="""SELECT ?n (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:intervino ?con . ?con rdfs:label ?n . }
GROUP BY ?n ORDER BY DESC(?c) LIMIT 1"""),

    dict(q="¿Cuántas proposiciones ha firmado el concejal Gorka Otxandiano?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:proponePersona ?con . ?con rdfs:label ?n .
  FILTER(REGEX(STR(?n), "gorka", "i") && REGEX(STR(?n), "otxandiano", "i")) }"""),

    dict(q="¿Qué concejal o concejala ha votado a favor de más proposiciones?",
         sparql="""SELECT ?n (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:concejalVotoAFavor ?con . ?con rdfs:label ?n . }
GROUP BY ?n ORDER BY DESC(?c) LIMIT 1"""),

    dict(q="¿Cuántas veces se abstuvo el grupo EH Bildu en proposiciones sobre urbanismo?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_urbanismo ; bo:seAbstuvo br:grupo_eh_bildu . }"""),

    dict(q="¿Se ha mencionado a Petronor en algún pleno?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:menciona ?e . ?e rdfs:label ?n . FILTER(REGEX(STR(?n), "\\\\bpetronor\\\\b", "i")) }"""),

    dict(q="¿Cuántas enmiendas ha presentado el grupo EH Bildu?",
         sparql="""SELECT (COUNT(DISTINCT ?e) AS ?n) WHERE {
  ?e a bo:Enmienda ; bo:enmiendaPor br:grupo_eh_bildu . }"""),

    dict(q="¿Cuál es el tema menos tratado en las proposiciones del Pleno?",
         sparql="""SELECT ?lab (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataSobre ?t . ?t skos:prefLabel ?lab . }
GROUP BY ?lab ORDER BY ASC(?n) LIMIT 1"""),

    dict(q="¿Cuántas proposiciones sobre el alquiler se han presentado?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_alquiler . }"""),

    dict(q="¿Cuántas proposiciones hay sobre igualdad y feminismo presentadas por el PP?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_igualdad ; bo:presentadaPor br:grupo_pp . }"""),
]


def _tok(s):
    import re
    return set(re.findall(r"[a-záéíóúñ0-9]+", s.lower()))


def nearest(pregunta, k=3):
    qt = _tok(pregunta)
    scored = []
    for ex in BANK:
        et = _tok(ex["q"])
        j = len(qt & et) / max(1, len(qt | et))
        scored.append((j, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]


def fixed(k=6):
    return [BANK[0], BANK[4], BANK[6], BANK[7], BANK[10], BANK[3]][:k]


# =============================================================================
# Selección de ejemplos por embeddings (bge-m3)
# =============================================================================

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bank_embeds.json")
_embedder = None
_bank_vecs = None
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


def _bank_embeddings():
    global _bank_vecs
    if _bank_vecs is not None:
        return _bank_vecs
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(BANK) and all(cached[i]["q"] == BANK[i]["q"] for i in range(len(BANK))):
            _bank_vecs = [c["vec"] for c in cached]
            return _bank_vecs
    emb = _get_embedder()
    vecs = emb.embed_documents([ex["q"] for ex in BANK])
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump([dict(q=BANK[i]["q"], vec=vecs[i]) for i in range(len(BANK))], f)
    _bank_vecs = vecs
    return vecs


def _embed_q(pregunta):
    if pregunta in _q_cache:
        return _q_cache[pregunta]
    v = _get_embedder().embed_query(pregunta)
    _q_cache[pregunta] = v
    return v


def nearest_embed(pregunta, k=3):
    qv = _embed_q(pregunta)
    vecs = _bank_embeddings()
    scored = [(_cos(qv, vecs[i]), BANK[i]) for i in range(len(BANK))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]


def compare(pregunta, k=3):
    jac = nearest(pregunta, k)
    emb = nearest_embed(pregunta, k)
    return [e["q"][:40] for e in jac], [e["q"][:40] for e in emb]


# =============================================================================
# Brazos A a E: variantes del prompt
# =============================================================================

# --------------------------------------------------------------------------
INSTR = ("Eres experto en SPARQL. Genera UNA consulta SPARQL válida que responda "
         "la pregunta. Devuelve SOLO la consulta (con sus PREFIX), sin explicaciones ni ```.\n")

PREFIXES = """PREFIX bo: <http://bilbao.tfg/ontology#>
PREFIX br: <http://bilbao.tfg/resource/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"""

# ---- esquema MÍNIMO (zero-shot puro: solo el modelo de datos) --------------
SCHEMA_MIN = PREFIXES + """

Grafo RDF del Pleno del Ayuntamiento de Bilbao (2007-2026).

CLASES Y PROPIEDADES:
  ?p a bo:Proposicion ; bo:tituloTopic ?titulo ; bo:fecha ?fecha ; bo:anio ?anio (xsd:integer) .
  ?p bo:presentadaPor ?g          # grupo que presenta (UNO solo por proposición)
  ?p bo:enPleno ?pleno
  ?p bo:tieneResultado ?res       # individuos: bo:Aprobada bo:Rechazada bo:Decae bo:Retirada bo:AprobadaConEnmienda bo:SinResultado
  ?p bo:trataSobre ?t             # tema principal (exacto)
  ?p bo:trataTemaAmplio ?t        # tema + subtemas (roll-up del razonador) -- usar por defecto para temas
  ?p bo:menciona ?ent             # entidades (empresas, personas externas, lugares)
  ?p bo:intervino ?concejal       # concejal/a que intervino en el debate
  ?p bo:proponePersona ?concejal  # concejal/a que firma la proposición
  ?p bo:votoAFavorDe / bo:votoEnContraDe / bo:seAbstuvo ?g          # voto por GRUPO
  ?p bo:concejalVotoAFavor / bo:concejalVotoEnContra / bo:concejalVotoAbstencion ?concejal   # voto NOMINAL
  ?p bo:tieneEnmienda ?enm . ?enm a bo:Enmienda ; bo:enmiendaPor ?g .
  ?g a bo:Grupo ; rdfs:label ?nombreGrupo .
  ?t a bo:Tema ; skos:prefLabel ?labelTema .
  ?concejal a bo:Concejal ; rdfs:label ?nombreConcejal ; bo:perteneceA ?g ; bo:esAlcalde ?bool .
  ?ent a bo:Entidad ; rdfs:label ?nombreEnt .
"""

# ---- listas de vocabulario (URIs exactas) ---------------------------------
VOCAB = """
GRUPOS (URIs br:): grupo_pp(PP) grupo_eh_bildu(EH BILDU) grupo_pse_ee(PSE-EE)
  grupo_elkarrekin_bilbao(ELKARREKIN BILBAO) grupo_goazen_bilbao(GOAZEN BILBAO)
  grupo_udalberri(UDALBERRI) grupo_eaj_pnv(EAJ-PNV) grupo_ciudadanos(CIUDADANOS)
  grupo_vox(VOX) grupo_ezker_batua_iu(EZKER BATUA-IU) grupo_equipo_de_gobierno(EQUIPO DE GOBIERNO)
  grupo_grupo_mixto(GRUPO MIXTO) grupo_desconocido(excluir de rankings)

TEMAS CANÓNICOS (URIs br:t_): vivienda urbanismo movilidad medioambiente euskera
  cultura deporte educacion igualdad serviciossociales empleoeconomia presupuestos
  seguridad participacion turismo sanidad memoriahistorica derechoshumanos otros

SUBTEMAS (URIs br:t_, cuelgan de un tema canónico; con trataTemaAmplio incluyen el padre):
  alquiler desahucios vivienda_social vivienda_vacia | aparcamiento bicicleta bilbobus
  metro_bilbao transporte_publico tranvia accesibilidad | comercio empleo empresas hosteleria
  financiacion ordenanzas_fiscales presupuesto_municipal retribuciones subvenciones
  barrios espacio_publico ordenacion_urbana rehabilitacion | discapacidad exclusion_social
  personas_mayores | policia_municipal | reciclaje | violencia_genero | arte fiestas museos | transparencia

Si un concepto no está en estas listas, resuélvelo con:
  ?p bo:trataTemaAmplio ?t . ?t skos:prefLabel ?lab . FILTER(REGEX(STR(?lab), "palabra", "i"))
NUNCA inventes una URI de tema que no esté arriba.
"""

# ---- reglas críticas (versión corta) ------------------------------------
RULES_KEY = """
REGLAS:
- Nombres de persona/entidad: ?x rdfs:label ?n . FILTER(REGEX(STR(?n), "apellido", "i")). NUNCA nodo anónimo [rdfs:label "x"].
- Año: FILTER(?anio = 2023) (entero, sin comillas ni ^^xsd:*).
- "aprobadas" sin más matiz = bo:Aprobada Y bo:AprobadaConEnmienda: FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)).
- Conteo simple de un resultado: pon bo:tieneResultado DIRECTO en el WHERE, nunca en OPTIONAL.
- Ratio (total + subconjunto en la misma consulta): usa OPTIONAL { ... BIND(?p AS ?sub) } y COUNT de cada uno, NUNCA FILTER.
- Total simple: SELECT (COUNT(DISTINCT ?p) AS ?n) sin GROUP BY.
- Ranking: COUNT + GROUP BY + ORDER BY DESC + LIMIT; incluye el label (grupo/tema) en el SELECT, no solo el COUNT.
- Filtras por grupo concreto + quieres su label: ?p bo:presentadaPor ?g . ?g rdfs:label ?ng . FILTER(?g = br:grupo_pp).
- No añadas filtro de tema si la pregunta no menciona un tema.
- Evita UNION. No calcules porcentajes dentro del SPARQL.
"""


def _ex_block(examples):
    out = ["EJEMPLOS (pregunta -> SPARQL):"]
    for ex in examples:
        out.append(f"\nP: {ex['q']}\n{PREFIXES}\n{ex['sparql']}")
    return "\n".join(out)


def _wrap(schema, pregunta, examples=None):
    p = INSTR + "\n" + schema
    if examples:
        p += "\n\n" + _ex_block(examples)
    p += f"\n\nPREGUNTA: {pregunta}\n\nSPARQL:"
    return p


# ============================ ARMS =======================================
def A_min(pregunta):
    return _wrap(SCHEMA_MIN, pregunta)


def B_vocab(pregunta):
    return _wrap(SCHEMA_MIN + VOCAB, pregunta)


def C_full(pregunta):
    # prompt de producción tal cual (SCHEMA_FULL trae {{ }} de str.format)
    sch = SCHEMA_FULL.replace("{{", "{").replace("}}", "}")
    return _wrap(sch, pregunta)


def D_compact_dyn(pregunta):
    return _wrap(SCHEMA_MIN + VOCAB + RULES_KEY, pregunta, nearest(pregunta, 3))


def E_full_dyn(pregunta):
    sch = SCHEMA_FULL.replace("{{", "{").replace("}}", "}")
    # recorta los EJEMPLO fijos del schema de producción, mete dinámicos
    idx = sch.find("EJEMPLO — proposiciones por grupo")
    if idx > 0:
        sch = sch[:idx].rstrip()
    return _wrap(sch, pregunta, nearest(pregunta, 3))


# los brazos F se definen más abajo, tras la Fase 1 (capa de alias / renombrado)
ARMS = {
    "A_min": A_min,
    "B_vocab": B_vocab,
    "C_full": C_full,
    "D_compact_dyn": D_compact_dyn,
    "E_full_dyn": E_full_dyn,
}


# =============================================================================
# Brazos F: corrección de alias en el SPARQL generado
# =============================================================================

# ---- mapa de grupos (forma laxa -> URI canónica) -------------------------
_GRUPOS = {
    "pp": "grupo_pp", "partidopopular": "grupo_pp",
    "ehbildu": "grupo_eh_bildu", "bildu": "grupo_eh_bildu", "ehb": "grupo_eh_bildu",
    "pseee": "grupo_pse_ee", "pse": "grupo_pse_ee", "socialistas": "grupo_pse_ee",
    "elkarrekinbilbao": "grupo_elkarrekin_bilbao", "elkarrekin": "grupo_elkarrekin_bilbao",
    "goazenbilbao": "grupo_goazen_bilbao", "goazen": "grupo_goazen_bilbao",
    "udalberri": "grupo_udalberri", "eajpnv": "grupo_eaj_pnv", "pnv": "grupo_eaj_pnv",
    "eaj": "grupo_eaj_pnv", "ciudadanos": "grupo_ciudadanos", "cs": "grupo_ciudadanos",
    "vox": "grupo_vox", "ezkerbatuaiu": "grupo_ezker_batua_iu", "ezkerbatua": "grupo_ezker_batua_iu",
    "equipodegobierno": "grupo_equipo_de_gobierno", "gobierno": "grupo_equipo_de_gobierno",
    "grupomixto": "grupo_grupo_mixto", "mixto": "grupo_grupo_mixto",
}
_TEMAS = {"vivienda", "urbanismo", "movilidad", "medioambiente", "euskera", "cultura",
          "deporte", "educacion", "igualdad", "serviciossociales", "empleoeconomia",
          "presupuestos", "seguridad", "participacion", "turismo", "sanidad",
          "memoriahistorica", "derechoshumanos", "otros"}

_VOTO_PERSONA = ("proponePersona", "intervino", "concejalVotoAFavor",
                 "concejalVotoEnContra", "concejalVotoAbstencion")


def _canon_grupo(raw):
    k = re.sub(r"[^a-z0-9]", "", raw.lower())
    if k in _GRUPOS:
        return "br:" + _GRUPOS[k]
    # ya es canónica (grupo_xxx)
    if k.startswith("grupo"):
        return "br:" + re.sub(r"^grupo_?", "grupo_", raw.lower())
    return None


def alias_rewrite(sparql: str) -> str:
    if not sparql:
        return sparql
    s = sparql

    # 0) <br:x> -> br:x  (el modelo a veces mete la URI prefijada entre <>)
    s = re.sub(r"<(br:[A-Za-z_]\w*)>", r"\1", s)
    s = re.sub(r"<(bo:[A-Za-z_]\w*)>", r"\1", s)

    # 1) tema con prefijo bo: -> br:   (bo:t_euskera -> br:t_euskera)
    s = re.sub(r"\bbo:(t_[a-z_]+)\b", r"br:\1", s)
    # 1b) bo:<TemaCapitalizado> -> br:t_<minuscula>   (bo:Euskera -> br:t_euskera)
    def _bo_tema(m):
        w = m.group(1).lower()
        return f"br:t_{w}" if w in _TEMAS else m.group(0)
    s = re.sub(r"\bbo:([A-ZÁÉÍÓÚ][a-záéíóú]+)\b", _bo_tema, s)

    # 2) grupo con URI no canónica: br:EHBildu / br:EH_Bildu -> br:grupo_eh_bildu
    def _grp(m):
        c = _canon_grupo(m.group(1))
        return c or m.group(0)
    s = re.sub(r"\bbr:([A-Za-z][A-Za-z_]*)\b", lambda m: (
        _canon_grupo(m.group(1)) or m.group(0)) if re.sub(r"[^a-z0-9]", "", m.group(1).lower()) in _GRUPOS else m.group(0), s)

    # 3) entidad como URI:  ?p bo:menciona br:Iberdrola   /  bo:menciona br:entidad_mercadona
    #    -> ?p bo:menciona ?ent_f . ?ent_f rdfs:label ?ent_fl . FILTER(REGEX(...))
    def _ent(m):
        pre, uri = m.group(1), m.group(2)
        uri = re.sub(r"^(entidad_|ent_|lugar_|org_|organizacion_)", "", uri, flags=re.I)
        body = (f'{pre} bo:menciona ?ent_f . ?ent_f rdfs:label ?ent_fl . '
                f'FILTER({_regex_and("?ent_fl", uri)})')
        return _tail(body, pre, m.group(3))
    s = re.sub(r"(\?\w+)\s+bo:menciona\s+br:([A-Za-z_]\w*)\s*([;.])", _ent, s)

    # 4) persona como URI con predicados de persona / voto nominal (term = ; o .)
    def _per(m):
        pre, pred, uri = m.group(1), m.group(2), m.group(3)
        body = (f'{pre} bo:{pred} ?per_f . ?per_f rdfs:label ?per_fl . '
                f'FILTER({_regex_and("?per_fl", uri)})')
        return _tail(body, pre, m.group(4))
    s = re.sub(r"(\?\w+)\s+bo:(" + "|".join(_VOTO_PERSONA) + r")\s+br:([A-Za-z_]\w*)\s*([;.])", _per, s)

    # 4b) voto de GRUPO con un br:<algo> que NO es un grupo canónico -> es persona
    s = re.sub(r"(\?\w+)\s+bo:(votoAFavorDe|votoEnContraDe|seAbstuvo)\s+br:(?!grupo_)([A-Za-z_]\w*)\s*([;.])",
               _per_grp, s)

    # 5) rdfs:label "Nombre Apellido" EXACTO  ->  REGEX (el modelo pone el nombre
    #    literal, casi nunca coincide con el label completo del grafo + tildes).
    def _lbl(m):
        var, lit, term = m.group(1), m.group(2), m.group(3)
        if len(lit.split()) > 4 or len(lit) < 3:
            return m.group(0)
        body = f'{var} rdfs:label ?lbl_f . FILTER({_regex_and("?lbl_f", lit)})'
        return _tail(body, var, term)
    s = re.sub(r'(\?\w+)\s+rdfs:label\s+"([^"]{3,50})"\s*([;.)}])', _lbl, s)

    return s


def _tail(body, subj, term):
    if term == ";":
        return body + f" . {subj} "
    return body + " " + term


def _per_grp(m):
    pre, pred, uri = m.group(1), m.group(2), m.group(3)
    term = m.group(4) if (m.lastindex or 0) >= 4 else "."
    nom = {"votoAFavorDe": "concejalVotoAFavor", "votoEnContraDe": "concejalVotoEnContra",
           "seAbstuvo": "concejalVotoAbstencion"}[pred]
    body = (f'{pre} bo:{nom} ?per_f . ?per_f rdfs:label ?per_fl . '
            f'FILTER({_regex_and("?per_fl", uri)})')
    return _tail(body, pre, term)


_STOP = {"concejal", "concejala", "concejales", "sr", "sra", "don", "dona", "doña",
         "grupo", "municipal", "el", "la", "los", "las", "de", "del", "senor",
         "senora", "señor", "señora", "persona", "entidad"}


def _tokens(uri: str):
    u = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", uri)      # camelCase
    u = re.sub(r"[_\-]+", " ", u)
    toks = [w for w in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+", u.lower()) if len(w) > 1]
    keep = [w for w in toks if w not in _STOP]
    return keep or toks


def _regex_and(var: str, uri: str) -> str:
    toks = _tokens(uri) or [uri.lower()]
    return " && ".join(f'REGEX(STR({var}), "{t}", "i")' for t in toks)


def _sanitize(sparql, pregunta):
    try:
        from grafo.consulta.guardas import _sanitize_sparql
        return _sanitize_sparql(sparql, verbose=False, pregunta=pregunta)
    except Exception:
        return sparql


def post_F(sparql, pregunta):
    return _sanitize(alias_rewrite(sparql), pregunta)


# --- prompts de F: base compacta (D) y base completa-dinámica (E) ---------
def F_compact(pregunta):
    return D_compact_dyn(pregunta)


def F_full(pregunta):
    return E_full_dyn(pregunta)


ARMS_F = {
    "F_compact": F_compact,   # D_compact_dyn + post_F
    "F_full": F_full,         # E_full_dyn + post_F
}
POST = {"F_compact": post_F, "F_full": post_F}
POST_ALIAS = set(ARMS_F)     # compat


# =============================================================================
# Brazos con ejemplos elegidos por embeddings
# =============================================================================

def D_embed_dyn(pregunta):
    return _wrap(SCHEMA_MIN + VOCAB + RULES_KEY, pregunta, nearest_embed(pregunta, 3))


def F_embed(pregunta):
    return D_embed_dyn(pregunta)


ARMS_EMBED = {
    "D_embed_dyn": D_embed_dyn,
    "F_embed": F_embed,
}
POST_EMBED = {"F_embed": post_F}
