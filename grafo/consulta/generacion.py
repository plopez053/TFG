"""
Del texto de la pregunta a filas del grafo cuando no hay consulta directa:
primero las plantillas (plantillas.py, sin LLM) y, si no encajan, el LLM genera
la consulta, que se corrige (guardas.py), se ejecuta y se reintenta con el
error si falla.
"""
import re
import threading

from comun.proveedores import LLM_MODEL_GRAPHRAG
from grafo.consulta import plantillas as _arm_g
from grafo.consulta.guardas import (_cos, _mensaje_temas, _preparar_sparql, _temas_inexistentes,
                                    _variables_sin_enlazar, _vocabulario_inexistente)
from grafo.consulta.pregunta import _preparar_asunto, analizar, cobertura
from grafo.consulta.recursos import _ejecutar, _llm_invoke, _load_graph, _PREFIXES


# =============================================================================
# Prompt para el LLM
# =============================================================================

# Prompt para que el LLM genere SPARQL: esquema compacto del grafo y los tres
# ejemplos del banco más parecidos a la pregunta.

SCHEMA = _PREFIXES + """

Grafo RDF (razonado con OWL-RL) del Pleno del Ayuntamiento de Bilbao (2007-2026).

CLASES Y PROPIEDADES:
  ?p a bo:Proposicion ; bo:tituloTopic ?titulo ; bo:fecha ?fecha ; bo:anio ?anio (xsd:integer) .
  ?p bo:fechaISO ?d               # xsd:date "AAAA-MM-DD": para ordenar por fecha y para rangos de fechas
  ?p bo:mes ?m                    # mes (1-12) de la sesion
  ?p bo:presentadaPor ?g          # grupo que presenta (UNO solo por proposicion)
  ?p bo:presentadaPorParticular ?ent   # SOLO si NO la presenta un grupo: particular/asociacion vecinal/AMPA (?ent a bo:Persona o bo:Organizacion)
  ?x a bo:PuntoOrdenDia           # cualquier punto del acta; "?p a bo:Proposicion" deja fuera preguntas, daciones de cuenta y tramites
  ?p bo:tipoPunto "proposicion_ciudadana"   # tipo de punto: proposicion_grupo, propuesta_gobierno, proposicion_ciudadana (vecinal),
                                  # y los que NO son proposiciones: pregunta, dacion_cuenta, tramite, debate_estado_ciudad (excluirlos al contar)
  ?p bo:votacionRegistrada "enmienda"   # si el recuento/voto por grupo guardado es el de una enmienda y no el de la proposicion
  ?p bo:tieneVotacion ?v . ?v bo:objetoVotacion "enmienda"|"punto"|"por_puntos" ; bo:esDecisiva true ;
     bo:votosFavorVotacion ?f ; bo:grupoVotaAFavor|bo:grupoVotaEnContra|bo:grupoSeAbstiene ?g ; bo:decision ?txt   # cada votacion del acta
  ?p bo:enBarrio ?b . ?b a bo:Barrio ; rdfs:label "Otxarkoaga" .   ?p bo:enDistrito ?dist . ?dist rdfs:label "Distrito 1 Deusto"
  ?p bo:importe ?euros ; bo:beneficiario ?quien   # importe del titulo (subvencion, credito), no gasto total
  ?p bo:fechaPresentacion ?d   # iniciativas vecinales
  ?p bo:enPleno ?pleno            # un pleno por acta (sesion)
  ?pleno a bo:Pleno ; bo:fechaSesion ?d ; bo:tipoSesion "ordinaria"|"extraordinaria" ; bo:enLegislatura ?leg .
  ?p bo:enLegislatura ?leg . ?leg a bo:Legislatura ; rdfs:label "Legislatura 2019-2023"   # de constitucion a constitucion
  ?p bo:tieneResultado ?res       # individuos: bo:Aprobada bo:Rechazada bo:Decae bo:Retirada bo:AprobadaConEnmienda bo:SinResultado
  ?p bo:trataSobre ?t             # tema principal (exacto)
  ?p bo:trataTemaAmplio ?t        # tema + subtemas (roll-up del razonador) -- usar por defecto para temas
  ?p bo:menciona ?ent             # entidades (empresas, personas externas, lugares)
  ?p bo:intervino ?concejal       # concejal/a que intervino en el debate
  ?p bo:proponePersona ?concejal  # concejal/a que firma la proposicion
  ?p bo:votoAFavorDe / bo:votoEnContraDe / bo:seAbstuvo ?g          # voto por GRUPO
  ?p bo:concejalVotoAFavor / bo:concejalVotoEnContra / bo:concejalVotoAbstencion ?concejal   # voto NOMINAL
  ?p bo:tieneEnmienda ?enm . ?enm a bo:Enmienda ; bo:enmiendaPor ?g .
  ?g a bo:Grupo ; rdfs:label ?nombreGrupo .
  ?t a bo:Tema ; skos:prefLabel ?labelTema .
  ?concejal a bo:Concejal ; rdfs:label ?nombreConcejal ; bo:perteneceA ?g ; bo:esAlcalde ?bool .
  ?ent a bo:Entidad ; rdfs:label ?nombreEnt .

GRUPOS (URIs br:): grupo_pp(PP) grupo_eh_bildu(EH BILDU) grupo_pse_ee(PSE-EE)
  grupo_elkarrekin_bilbao(ELKARREKIN BILBAO) grupo_goazen_bilbao(GOAZEN BILBAO)
  grupo_udalberri(UDALBERRI) grupo_eaj_pnv(EAJ-PNV) grupo_ciudadanos(CIUDADANOS)
  grupo_vox(VOX) grupo_ezker_batua_iu(EZKER BATUA-IU) grupo_equipo_de_gobierno(EQUIPO DE GOBIERNO)
  grupo_grupo_mixto(GRUPO MIXTO) grupo_desconocido(excluir de rankings)

TEMAS CANONICOS (URIs br:t_): vivienda urbanismo movilidad medioambiente euskera
  cultura deporte educacion igualdad serviciossociales empleoeconomia presupuestos
  seguridad participacion turismo sanidad memoriahistorica derechoshumanos otros

SUBTEMAS (URIs br:t_, cuelgan de un tema canonico; con trataTemaAmplio incluyen el padre):
  alquiler desahucios vivienda_social vivienda_vacia | aparcamiento bicicleta bilbobus
  metro_bilbao transporte_publico tranvia accesibilidad | comercio empleo empresas hosteleria
  financiacion ordenanzas_fiscales presupuesto_municipal retribuciones subvenciones
  barrios espacio_publico ordenacion_urbana rehabilitacion | discapacidad exclusion_social
  personas_mayores | policia_municipal | reciclaje | violencia_genero | arte fiestas museos | transparencia

Si un concepto no esta en estas listas, resuelvelo con:
  ?p bo:trataTemaAmplio ?t . ?t skos:prefLabel ?lab . FILTER(REGEX(STR(?lab), "palabra", "i"))
NUNCA inventes una URI de tema que no este arriba.

REGLAS:
- Nombres de persona/entidad: ?x rdfs:label ?n . FILTER(REGEX(STR(?n), "apellido", "i")). NUNCA nodo anonimo [rdfs:label "x"].
- Ano: FILTER(?anio = 2023) (entero, sin comillas ni ^^xsd:*).
- Rango de anos ("entre 2019 y 2023", "de 2015 a 2019"), ambos incluidos:
  ?p bo:anio ?anio . FILTER(?anio >= 2019 && ?anio <= 2023). Toda variable de un FILTER debe estar en un triple.
  No anadas un filtro de ano si la pregunta no menciona ninguno.
- Fecha exacta: bo:fecha se guarda como "DD-MM-AAAA" CON GUIONES (p.ej. "24-09-2015"), NUNCA con barras.
- Lo mas reciente ("la ultima vez", "la ultima proposicion"): ?p bo:fechaISO ?d ... ORDER BY DESC(?d) LIMIT 1.
- Meses o estaciones: ?p bo:mes ?m . FILTER(?m IN (3, 4, 5, 6)) (primavera 3-6, verano 6-9, otono 9-12, invierno 12-3).
- Unanimidad: aprobada, con algun bo:votoAFavorDe y sin grupos en contra ni abstenciones:
  FILTER NOT EXISTS { ?p bo:votoEnContraDe ?x } FILTER NOT EXISTS { ?p bo:seAbstuvo ?y }.
- Oposicion: grupos distintos de br:grupo_equipo_de_gobierno y br:grupo_eaj_pnv (gobiernan en todo el periodo),
  de br:grupo_pse_ee desde 2015 (socio de gobierno) y de br:grupo_desconocido: FILTER(?g NOT IN (...)).
- El voto por grupo es el de la votacion decisiva, que a veces es la de una enmienda. bo:Decae = la
  proposicion no llego a votarse porque se aprobo una enmienda que la sustituye.
- El grafo NO guarda importes economicos ni la asistencia de los concejales: no inventes propiedades para eso.
- Pregunta sobre UNA proposicion concreta (nombra grupo+fecha+un tema/asunto especifico, tipo
  "que paso con la proposicion de X sobre Y del [fecha]"): ademas de filtrar por grupo/fecha,
  anade SIEMPRE FILTER(REGEX(STR(?titulo), "palabra_clave_del_asunto", "i")) sobre bo:tituloTopic
  -- si no, la consulta trae TODAS las proposiciones de ese grupo en esa fecha (puede haber varias)
  y no se puede saber cual es la que se pregunta.
- "aprobadas" sin mas matiz = bo:Aprobada Y bo:AprobadaConEnmienda: FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)).
- Conteo simple de un resultado: pon bo:tieneResultado DIRECTO en el WHERE, nunca en OPTIONAL.
- Ratio (total + subconjunto en la misma consulta): usa OPTIONAL { ... BIND(?p AS ?sub) } y COUNT de cada uno, NUNCA FILTER.
- Total simple: SELECT (COUNT(DISTINCT ?p) AS ?n) sin GROUP BY.
- Ranking: COUNT + GROUP BY + ORDER BY DESC + LIMIT; incluye el label (grupo/tema) en el SELECT, no solo el COUNT.
- Filtras por grupo concreto + quieres su label: ?p bo:presentadaPor ?g . ?g rdfs:label ?ng . FILTER(?g = br:grupo_pp).
- No anadas filtro de tema si la pregunta no menciona un tema.
- Evita UNION. No calcules porcentajes dentro del SPARQL.
- "que argumentos/postura/opinion ha dado el grupo X en/a favor/en contra de <tema>" es sobre las
  proposiciones que X MISMO presento sobre ese tema (bo:presentadaPor + bo:trataTemaAmplio), NO su
  historial de voto en propuestas ajenas. Usa bo:votoAFavorDe/votoEnContraDe SOLO si la pregunta
  habla explicitamente de como VOTO/VOTACION X una propuesta (de otro grupo).
"""


# banco de 25 preguntas -> SPARQL de referencia para el few-shot dinamico
# (ninguna es del gold set del estudio, para no medir sobre los propios ejemplos)
_EXAMPLE_BANK = [
    dict(q=u"\u00bfCu\u00e1ntas proposiciones sobre cultura se han presentado?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_cultura . }"""),
    dict(q=u"\u00bfCu\u00e1ntas proposiciones present\u00f3 el PSE-EE en 2018?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_pse_ee ; bo:anio 2018 . }"""),
    dict(q=u"\u00bfQu\u00e9 proposiciones ha presentado alguna asociaci\u00f3n vecinal o particular?",
         sparql="""SELECT ?p ?nombre WHERE {
  ?p a bo:Proposicion ; bo:presentadaPorParticular ?ent .
  ?ent rdfs:label ?nombre . } LIMIT 50"""),
    dict(q=u"\u00bfCu\u00e1ntas proposiciones sobre sanidad se han aprobado?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_sanidad ; bo:tieneResultado ?r .
  FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) }"""),
    dict(q=u"\u00bfQu\u00e9 grupo ha presentado m\u00e1s proposiciones sobre seguridad?",
         sparql="""SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_seguridad ; bo:presentadaPor ?g .
  ?g rdfs:label ?ng . FILTER(?g != br:grupo_desconocido) }
GROUP BY ?ng ORDER BY DESC(?n) LIMIT 10"""),
    dict(q=u"\u00bfCu\u00e1ntas proposiciones ha presentado cada grupo en total?",
         sparql="""SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor ?g . ?g rdfs:label ?ng .
  FILTER(?g != br:grupo_desconocido) }
GROUP BY ?ng ORDER BY DESC(?n) LIMIT 20"""),
    dict(q=u"\u00bfEn qu\u00e9 a\u00f1o hubo m\u00e1s proposiciones sobre cultura?",
         sparql="""SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_cultura ; bo:anio ?anio . }
GROUP BY ?anio ORDER BY DESC(?n) LIMIT 1"""),
    dict(q=u"\u00bfC\u00f3mo ha evolucionado el n\u00famero de proposiciones sobre movilidad por a\u00f1o?",
         sparql="""SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_movilidad ; bo:anio ?anio . }
GROUP BY ?anio ORDER BY ASC(?anio)"""),
    dict(q=u"\u00bfCu\u00e1ntas proposiciones present\u00f3 el PP en 2016 y cu\u00e1ntas se aprobaron?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?ap) AS ?aprob) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_pp ; bo:anio 2016 .
  OPTIONAL { ?p bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) . BIND(?p AS ?ap) } }"""),
    dict(q=u"\u00bfQu\u00e9 porcentaje de las proposiciones sobre seguridad se rechazaron?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?re) AS ?rech) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_seguridad .
  OPTIONAL { ?p bo:tieneResultado bo:Rechazada . BIND(?p AS ?re) } }"""),
    dict(q=u"\u00bfQu\u00e9 concejal o concejala ha intervenido en m\u00e1s debates del Pleno?",
         sparql="""SELECT ?n (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:intervino ?con . ?con rdfs:label ?n . }
GROUP BY ?n ORDER BY DESC(?c) LIMIT 1"""),
    dict(q=u"\u00bfCu\u00e1ntas proposiciones ha firmado el concejal Gorka Otxandiano?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:proponePersona ?con . ?con rdfs:label ?n .
  FILTER(REGEX(STR(?n), "gorka", "i") && REGEX(STR(?n), "otxandiano", "i")) }"""),
    dict(q=u"\u00bfQu\u00e9 concejal o concejala ha votado a favor de m\u00e1s proposiciones?",
         sparql="""SELECT ?n (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:concejalVotoAFavor ?con . ?con rdfs:label ?n . }
GROUP BY ?n ORDER BY DESC(?c) LIMIT 1"""),
    dict(q=u"\u00bfCu\u00e1ntas veces se abstuvo el grupo EH Bildu en proposiciones sobre urbanismo?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_urbanismo ; bo:seAbstuvo br:grupo_eh_bildu . }"""),
    dict(q=u"\u00bfSe ha mencionado a Petronor en alg\u00fan pleno?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:menciona ?e . ?e rdfs:label ?n . FILTER(REGEX(STR(?n), "\\\\bpetronor\\\\b", "i")) }"""),
    dict(q=u"\u00bfCu\u00e1ntas enmiendas ha presentado el grupo EH Bildu?",
         sparql="""SELECT (COUNT(DISTINCT ?e) AS ?n) WHERE {
  ?e a bo:Enmienda ; bo:enmiendaPor br:grupo_eh_bildu . }"""),
    dict(q=u"\u00bfCu\u00e1l es el tema menos tratado en las proposiciones del Pleno?",
         sparql="""SELECT ?lab (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataSobre ?t . ?t skos:prefLabel ?lab . }
GROUP BY ?lab ORDER BY ASC(?n) LIMIT 1"""),
    dict(q=u"\u00bfCu\u00e1ntas proposiciones sobre el alquiler se han presentado?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_alquiler . }"""),
    dict(q=u"\u00bfCu\u00e1ntas proposiciones hay sobre igualdad y feminismo presentadas por el PP?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_igualdad ; bo:presentadaPor br:grupo_pp . }"""),
    dict(q=u"\u00bfQu\u00e9 argumentos ha dado Elkarrekin Bilbao en contra de los pisos tur\u00edsticos?",
         sparql="""SELECT ?p WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_elkarrekin_bilbao ; bo:trataTemaAmplio br:t_vivienda . }
LIMIT 20"""),
    dict(q=u"\u00bfQu\u00e9 pas\u00f3 con la proposici\u00f3n del PP sobre el Bilbob\u00fas del 24 de septiembre de 2015?",
         sparql="""SELECT ?p ?titulo ?res WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_pp ; bo:fecha "24-09-2015" ;
     bo:tituloTopic ?titulo ; bo:tieneResultado ?res .
  FILTER(REGEX(STR(?titulo), "bilbobus", "i")) }"""),
    # sin este ejemplo, "¿cuántas han decaído?" copiaba un conteo simple por
    # año y olvidaba el filtro de resultado
    dict(q=u"\u00bfCu\u00e1ntas proposiciones han decaido en 2019?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:tieneResultado bo:Decae ; bo:anio 2019 . }"""),
    dict(q=u"¿Cuándo se presentó la última proposición sobre euskera?",
         sparql="""SELECT ?p ?fecha ?titulo WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_euskera ; bo:fechaISO ?d ; bo:fecha ?fecha ; bo:tituloTopic ?titulo . }
ORDER BY DESC(?d) LIMIT 1"""),
    dict(q=u"¿Cuántas proposiciones presentó el PP entre 2012 y 2014?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_pp ; bo:anio ?anio . FILTER(?anio >= 2012 && ?anio <= 2014) }"""),
    dict(q=u"¿Cuántas proposiciones se rechazaron en el verano de 2018?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:anio 2018 ; bo:mes ?m ; bo:tieneResultado bo:Rechazada . FILTER(?m IN (6, 7, 8, 9)) }"""),
]


def _tok_pregunta(s):
    return set(re.findall(u"[a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00f10-9]+", s.lower()))


# 3 ejemplos del banco mas parecidos por solape de palabras (Jaccard) -- ver ESTUDIO_SPARQL_LOCAL.md Fase 2/4
def _nearest_examples_jaccard(pregunta, k=3):
    qt = _tok_pregunta(pregunta)
    scored = []
    for ex in _EXAMPLE_BANK:
        et = _tok_pregunta(ex["q"])
        j = len(qt & et) / max(1, len(qt | et))
        scored.append((j, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]


_bank_embed_vecs = None
_bank_embed_lock = threading.Lock()


# 3 ejemplos del banco mas parecidos por similitud coseno de embeddings bge-m3 (local, sin API externa)
def _nearest_examples_embed(pregunta, k=3):
    global _bank_embed_vecs
    from langchain_ollama import OllamaEmbeddings
    embedder = OllamaEmbeddings(model="bge-m3")
    if _bank_embed_vecs is None:
        with _bank_embed_lock:
            if _bank_embed_vecs is None:
                _bank_embed_vecs = embedder.embed_documents([ex["q"] for ex in _EXAMPLE_BANK])
    qv = embedder.embed_query(pregunta)
    scored = [(_cos(qv, _bank_embed_vecs[i]), _EXAMPLE_BANK[i]) for i in range(len(_EXAMPLE_BANK))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]


# el estudio (ver Fase 4) encontro que el embedding solo mejora con qwen3:8b o
# mejor -- con qwen2.5:7b el solape de palabras iguala o mejora, mas rapido y
# sin competir por VRAM con el modelo de generacion
def _nearest_examples(pregunta, k=3):
    if "qwen3" in LLM_MODEL_GRAPHRAG.lower():
        try:
            return _nearest_examples_embed(pregunta, k)
        except Exception as e:
            print(f"[!] embedding bge-m3 no disponible, uso solape de palabras: {e}", flush=True)
    return _nearest_examples_jaccard(pregunta, k)


_SPARQL_INSTR = ("Eres experto en SPARQL. Genera UNA consulta SPARQL valida que responda "
                  "la pregunta. Devuelve SOLO la consulta (con sus PREFIX), sin explicaciones ni ```.\n")


# construye el prompt de generacion: schema compacto + 3 ejemplos dinamicos + pregunta
def _build_sparql_prompt(pregunta: str, analisis=None) -> str:
    if analisis is None:
        analisis = analizar(pregunta)
        _preparar_asunto(_load_graph(), analisis, pregunta)
    examples = _nearest_examples(pregunta, 3)
    ex_block = "\n".join(f"\nP: {ex['q']}\n{_PREFIXES}\n{ex['sparql']}" for ex in examples)
    pista = getattr(analisis, "pista", "")
    return (f"{_SPARQL_INSTR}\n{SCHEMA}\n\nEJEMPLOS (pregunta -> SPARQL):{ex_block}"
            f"{pista}\n\nPREGUNTA: {pregunta}\n\nSPARQL:")


# =============================================================================
# Plantillas, generación y reintentos
# =============================================================================

def _try_arm_g(pregunta, g, verbose, analisis=None):
    analisis = analisis or analizar(pregunta)
    if analisis.requiere_llm():
        return None
    try:
        res = _arm_g.answer(pregunta)
    except Exception as e:
        if verbose:
            print(f"[Arm G] fallo al intentar el filtro previo, cae a LLM: {e}", flush=True)
        return None
    if not res.get("sparql"):
        return None
    faltan = cobertura(analisis, res["sparql"])
    if faltan:
        if verbose:
            print(f"[Arm G] la plantilla {res['template']} no cubre {faltan}; cae a LLM", flush=True)
        return None
    try:
        rows = _ejecutar(g, res["sparql"])
    except Exception:
        return None
    if not rows:
        return None
    if verbose:
        print(f"[Arm G] resuelto sin LLM (tmpl={res['template']} sim={res['similarity']})"
              f"\n[SPARQL]\n{res['sparql']}\n")
    return res["sparql"], rows


# Prompt de corrección. Repite el esquema entero: con un "corrígela" a secas
# el modelo inventaba vocabulario (PREFIX example.org, WITH, subconsultas).
def _prompt_correccion(pregunta: str, sparql: str, error, faltan=None) -> str:
    if faltan:
        lista = "\n".join(f"- {f}" for f in faltan)
        return (f"{SCHEMA}\n\nPREGUNTA: {pregunta}\n\n"
                f"Esta consulta SPARQL se ejecutó, pero NO cubre todo lo que pide la pregunta:\n{sparql}\n\n"
                f"Le falta (o le sobra):\n{lista}\n\n"
                "Genera una consulta NUEVA que responda exactamente a la pregunta y tenga en cuenta "
                "todo lo anterior, usando SOLO el vocabulario del schema. Devuelve SOLO la consulta SPARQL.")
    if error is not None:
        return (f"{SCHEMA}\n\nPREGUNTA: {pregunta}\n\n"
                f"Esta consulta SPARQL DIO ERROR: {error}\n"
                f"Consulta con error:\n{sparql}\n\n"
                "Genera una consulta NUEVA que responda la pregunta, más simple, usando "
                "SOLO el vocabulario del schema de arriba (NO inventes PREFIX, WITH, "
                "subconsultas ni URIs). Para porcentajes/ratios devuelve solo el total y "
                "el subconjunto con OPTIONAL+BIND. Devuelve SOLO la consulta SPARQL.")
    return (f"{SCHEMA}\n\nPREGUNTA: {pregunta}\n\n"
            f"Esta consulta SPARQL se ejecutó bien pero NO devolvió NINGUNA fila (o solo un recuento de 0):\n{sparql}\n\n"
            "Revisa también que cada relación sea la que pide la pregunta (p.ej. quién PRESENTA "
            "la proposición es bo:presentadaPor ?g con el grupo directamente, sin pasos intermedios). "
            "Antes de repetir el mismo patrón, revisa: ¿la URI de tema/grupo que usaste está "
            "en la lista exacta del schema? ¿Inventaste la URI de una entidad o persona en vez "
            "de resolverla con rdfs:label + REGEX? ¿el predicado existe tal cual? Genera una "
            "consulta ALTERNATIVA (distinta de la anterior) que sí pueda encontrar datos si "
            "existen en el grafo. Si tras revisarlo la consulta anterior ya era correcta y el "
            "grafo simplemente no tiene ese dato, repite la misma. Devuelve SOLO la consulta SPARQL.")


_RESPUESTA_IRRECUPERABLE = (
    "No he podido traducir esa pregunta a una consulta válida sobre el "
    "grafo. Puede que pida un dato que el grafo no distingue (por ejemplo, "
    "proposiciones presentadas conjuntamente por varios grupos: el grafo "
    "guarda un único grupo por proposición). Prueba a reformularla de forma "
    "más concreta.")


# Genera la SPARQL con el LLM y la ejecuta. Se pide una corrección (hasta 2)
# si usa términos que no existen en el grafo, si da error, si devuelve 0 filas
# (suele ser una URI mal resuelta, no la ausencia del dato) o si no cubre lo
# que menciona la pregunta. Devuelve (sparql, filas, error, faltan): error solo
# si ninguna consulta se pudo ejecutar; faltan, lo que la mejor consulta no cubre.
def _generar_y_ejecutar(g, pregunta: str, sparql_provider: str, verbose: bool, analisis):
    sparql = _preparar_sparql(_llm_invoke(_build_sparql_prompt(pregunta, analisis), prefer=sparql_provider),
                              verbose, pregunta)
    if verbose:
        print(f"\n[SPARQL]\n{sparql}\n")

    mejor = None  # (n_faltan, sparql, filas, faltan) de la mejor consulta con filas
    cero = None   # (sparql, filas) de la primera consulta completa cuyo recuento dio 0
    for intento in range(3):
        invalidos = _vocabulario_inexistente(g, sparql)
        sueltas = _variables_sin_enlazar(sparql)
        temas_malos = _temas_inexistentes(g, sparql)
        # COUNT sobre una subconsulta con LIMIT: el resultado es el límite
        # ("10 proposiciones en los últimos plenos" era el LIMIT 10)
        limite_contado = bool(re.search(r"\bCOUNT\s*\(", sparql, re.I)
                              and re.search(r"\{\s*SELECT\b[^{}]*(?:\{[^{}]*\}[^{}]*)*\bLIMIT\s+\d+", sparql, re.I))
        if limite_contado and not invalidos:
            rows, error = [], ValueError(
                "cuenta con COUNT sobre una subconsulta con LIMIT, así que el resultado es el límite y no un "
                "recuento. Para 'los últimos plenos' filtra por fecha (bo:fechaISO) en vez de usar LIMIT")
        elif invalidos:
            rows, error = [], ValueError(f"usa términos que no existen en el grafo: {', '.join(invalidos)}")
        elif sueltas:
            rows, error = [], ValueError(
                "estas variables se usan en un FILTER pero no aparecen en ningún triple, así que el "
                f"filtro descarta todo: {', '.join('?' + v for v in sueltas)}")
        elif temas_malos:
            rows, error = [], ValueError(_mensaje_temas(temas_malos))
        else:
            try:
                rows, error = _ejecutar(g, sparql), None
            except Exception as e:
                rows, error = [], e
        faltan = cobertura(analisis, sparql) if error is None else []
        if error is None and rows:
            # Un COUNT siempre devuelve una fila: si todo es 0 se trata como una
            # consulta vacía (casi siempre es una relación mal planteada, no la
            # ausencia del dato) y se pide otra. Si la nueva tampoco encuentra
            # nada, se responde con este 0.
            if not faltan and _recuento_cero(rows):
                cero = cero or (sparql, rows)
                rows = []
            elif not faltan:
                return sparql, rows, None, []
            elif mejor is None or len(faltan) < mejor[0]:
                mejor = (len(faltan), sparql, rows, faltan)
        if intento == 2:
            break

        fix = _llm_invoke(_prompt_correccion(pregunta, sparql, error, faltan if rows else None),
                          prefer=sparql_provider)
        nueva = _preparar_sparql(fix, verbose, pregunta)
        # el modelo confirma la misma consulta que dio 0: se da por buena
        if cero and re.sub(r"\s+", " ", nueva).strip() == re.sub(r"\s+", " ", cero[0]).strip():
            break
        sparql = nueva
        if verbose:
            print(f"[SPARQL corregido #{intento + 1}]\n{sparql}\n")

    if mejor:
        return mejor[1], mejor[2], None, mejor[3]
    if cero:
        return cero[0], cero[1], None, []
    return sparql, rows, error, faltan


# una sola fila y todos sus valores numéricos a 0 (p.ej. COUNT = 0)
def _recuento_cero(rows: list) -> bool:
    if len(rows) != 1:
        return False
    nums = [v for v in rows[0].values() if re.fullmatch(r"-?\d+(\.\d+)?", str(v).strip())]
    return bool(nums) and all(float(v) == 0 for v in nums)


# colapsa 'SELECT COUNT(?x) ... GROUP BY ?x' a una fila con el total real
def _fix_degenerate_groupby(rows: list, sparql: str) -> list:
    m_groupby = re.search(r"GROUP BY\s+([\?\w\s]+?)\s*(?:ORDER BY|LIMIT|$)", sparql, re.IGNORECASE)
    if not m_groupby or not rows:
        return rows
    group_vars = re.findall(r"\?\w+", m_groupby.group(1))
    count_vars = re.findall(r"COUNT\(\s*(\?\w+)\s*\)", sparql, re.IGNORECASE)
    if len(group_vars) == 1 and count_vars and all(cv == group_vars[0] for cv in count_vars):
        m_alias = re.search(r"COUNT\(\s*\?\w+\s*\)\s+AS\s+\?(\w+)", sparql, re.IGNORECASE)
        col = m_alias.group(1) if m_alias else "total"
        return [{col: str(len(rows))}]
    return rows
