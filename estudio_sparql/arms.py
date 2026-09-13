# -*- coding: utf-8 -*-
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # TFG/TFG
sys.path.insert(0, os.path.join(_ROOT, "graphrag", "graphrag"))
from graph_rag_sparql import SCHEMA as SCHEMA_FULL          # el prompt de producción
import example_bank as EB

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
    return _wrap(SCHEMA_MIN + VOCAB + RULES_KEY, pregunta, EB.nearest(pregunta, 3))


def E_full_dyn(pregunta):
    sch = SCHEMA_FULL.replace("{{", "{").replace("}}", "}")
    # recorta los EJEMPLO fijos del schema de producción, mete dinámicos
    idx = sch.find("EJEMPLO — proposiciones por grupo")
    if idx > 0:
        sch = sch[:idx].rstrip()
    return _wrap(sch, pregunta, EB.nearest(pregunta, 3))


# F_adapted se define en arms_f.py tras la Fase 1 (capa de alias / renombrado)
ARMS = {
    "A_min": A_min,
    "B_vocab": B_vocab,
    "C_full": C_full,
    "D_compact_dyn": D_compact_dyn,
    "E_full_dyn": E_full_dyn,
}
