import os
import re
import sys
import threading
import unicodedata

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# modelos y healthcheck de Ollama, del módulo de proveedores (no del pipeline vectorial)
from backend.providers import LLM_MODEL_GROQ, LLM_MODEL_GRAPHRAG, ping_ollama as _ping_ollama

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_TTL = os.path.join(HERE, "bilbao_reasoned.ttl")

# ---------------------------------------------------------------------------
# Schema SPARQL compacto + banco de ejemplos few-shot + capa de alias.
# Sustituye al mega-prompt de 310 lineas tras el estudio de ablacion (ver
# graphrag/graphrag/ESTUDIO_SPARQL_LOCAL.md): un schema compacto + 3 ejemplos
# dinamicos (elegidos por parecido a la pregunta) + reescritura determinista
# de las invenciones sistematicas del LLM da mejor acierto y menos respuestas
# falsas que el schema largo, con 1/3 del tamano de prompt.
# ---------------------------------------------------------------------------
_PREFIXES = """PREFIX bo: <http://bilbao.tfg/ontology#>
PREFIX br: <http://bilbao.tfg/resource/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"""

SCHEMA = _PREFIXES + """

Grafo RDF (razonado con OWL-RL) del Pleno del Ayuntamiento de Bilbao (2007-2026).

CLASES Y PROPIEDADES:
  ?p a bo:Proposicion ; bo:tituloTopic ?titulo ; bo:fecha ?fecha ; bo:anio ?anio (xsd:integer) .
  ?p bo:presentadaPor ?g          # grupo que presenta (UNO solo por proposicion)
  ?p bo:presentadaPorParticular ?ent   # SOLO si NO la presenta un grupo: particular/asociacion vecinal/AMPA (?ent a bo:Persona o bo:Organizacion)
  ?p bo:enPleno ?pleno
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
- Fecha exacta: bo:fecha se guarda como "DD-MM-AAAA" CON GUIONES (p.ej. "24-09-2015"), NUNCA con barras.
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

# banco de 18 preguntas -> SPARQL de referencia para el few-shot dinamico
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
    # Ronda 42 (2026-09-17): ningun ejemplo anterior usaba bo:Decae ni la
    # palabra "decaido/decaido" -- verificado en vivo que "Cuantas
    # proposiciones han decaido en 2019?" generaba SPARQL SIN ningun filtro
    # de resultado (contaba TODAS las proposiciones de 2019, 111, en vez de
    # solo las decaidas), porque los 3 ejemplos mas cercanos por similitud
    # eran de conteo simple por anio, sin bo:tieneResultado. La consulta
    # ejecutaba sin error y con filas -> se aceptaba tal cual, sin reintento,
    # dando una respuesta con seguridad total pero equivocada.
    dict(q=u"\u00bfCu\u00e1ntas proposiciones han decaido en 2019?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:tieneResultado bo:Decae ; bo:anio 2019 . }"""),
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

    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb + 1e-9)

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
def _build_sparql_prompt(pregunta: str) -> str:
    examples = _nearest_examples(pregunta, 3)
    ex_block = "\n".join(f"\nP: {ex['q']}\n{_PREFIXES}\n{ex['sparql']}" for ex in examples)
    return (f"{_SPARQL_INSTR}\n{SCHEMA}\n\nEJEMPLOS (pregunta -> SPARQL):{ex_block}"
            f"\n\nPREGUNTA: {pregunta}\n\nSPARQL:")

ANSWER_PROMPT = """Eres un analista político experto en el Ayuntamiento de Bilbao.
Basándote ÚNICAMENTE en los datos del grafo que te proporciono, genera una respuesta en español que sea:
- Narrativa y clara: no solo números, explica qué significan
- Precisa: cita las cifras exactas del grafo
- Contextual: si hay datos temporales, describe la evolución; si hay varios grupos, compáralos
- Completa: menciona los casos más destacados y cualquier patrón interesante

REGLA CRÍTICA: si los datos están vacíos ("sin resultados en el grafo"), responde honestamente que
no se encontraron datos para esa consulta. NUNCA inventes cifras hipotéticas ni pongas ejemplos
ilustrativos: cualquier cifra que no aparezca en los datos es una alucinación.

REGLA CRÍTICA sobre qué son las cifras: TODOS los números de DATOS DEL GRAFO cuentan
PROPOSICIONES (iniciativas presentadas en el Pleno) — NUNCA personas, "miembros",
concejales ni votantes, aunque la fila hable de un grupo político. Si una columna se
llama "n", "total" o similar junto a un grupo/tema/año, significa "número de
proposiciones", no "número de miembros del grupo".

REGLA CRÍTICA al comparar filas ("quién tiene más", "el más activo", rankings): lee las
cifras de TODAS las filas con cuidado antes de concluir cuál es la mayor — no asumas que
la segunda fila es la primera en importancia. Si vas a nombrar un "máximo" o "mínimo",
verifica que su cifra sea realmente la más alta/baja de todas las que ves en los datos.

REGLA CRÍTICA sobre el filtro ya aplicado: la consulta SPARQL de abajo YA se ejecutó contra
el grafo — su cláusula WHERE ya filtró exactamente lo que pide la pregunta (un tema, grupo,
año, resultado...). CADA fila de DATOS DEL GRAFO ya cumple ese filtro; no dudes de si están
relacionadas con la pregunta ni pidas "más datos para confirmar la relación" — la relación
ya está garantizada por la propia consulta. Si el filtro es sobre un tema y el resultado es
un conteo, ese número ES la respuesta a "cuántas proposiciones hay de ese tema".

REGLA CRÍTICA sobre valores "None"/vacíos en una fila: si una fila tiene una columna numérica
(COUNT, total...) con un valor real mayor que 0 pero OTRAS columnas de esa misma fila salen
"None" o vacías, NO significa que no haya datos — solo significa que esas columnas concretas
no se enlazaron en el SPARQL (variable sin usar en el WHERE). El número sigue siendo válido y
es la respuesta. Solo trata una pregunta como "sin datos" si TODAS las filas están vacías o
si la lista de filas está vacía del todo — nunca por ver "None" en una columna aislada.

REGLA CRÍTICA sobre LIMIT: si la consulta SPARQL de abajo termina en "LIMIT N", las filas
que ves son SOLO las N primeras de un ranking, NO todas. NUNCA digas que suman "el total",
"la totalidad", "todas las proposiciones del tema" ni "no hay más grupos/años relevantes":
hay más filas que la consulta no ha traído. Describe solo lo que ves ("los 3 grupos que
más han presentado son...") sin afirmar nada sobre el resto.

REGLA CRÍTICA sobre aritmética: NO calcules restas de años, porcentajes ni sumas que no
estén ya en los datos, salvo que la pregunta lo pida explícitamente y los números necesarios
estén los dos en las filas. Si mencionas dos años (p.ej. 2019 y 2022), NO añadas "X años
después" — limítate a nombrar los años. Un cálculo mental mal hecho es una alucinación.

CONSULTA SPARQL YA EJECUTADA (para que entiendas qué significan las filas, no para repetirla):
{sparql}

PREGUNTA: {pregunta}

DATOS DEL GRAFO (ya filtrados según la consulta de arriba):
{filas}

RESPUESTA:"""

_graph = None
_graph_lock = threading.Lock()
_llm_cache: dict = {}
_llm_cache_lock = threading.Lock()


def _load_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                from rdflib import Graph
                g = Graph()
                g.parse(GRAPH_TTL, format="turtle")
                _graph = g
    return _graph


# devuelve el LLM del proveedor indicado, con caché por proveedor
def _get_llm(provider: str):
    if provider not in _llm_cache:
        with _llm_cache_lock:
            if provider not in _llm_cache:
                if provider == "ollama":
                    from langchain_ollama import ChatOllama
                    # num_predict alto: la narración de una consulta de LISTADO
                    # (p.ej. las 50 proposiciones de un particular/asociación,
                    # con dos tablas y categorización) necesita bastante más que
                    # el límite por defecto -- verificado un corte real a mitad
                    # de frase con el límite implícito anterior.
                    # reasoning=False + num_ctx=8192: misma config que ganó el estudio de
                    # ablación (ESTUDIO_SPARQL_LOCAL.md, estudio_sparql/runner.py) para
                    # qwen3:8b -- el modo "pensamiento" no mejora esta tarea estructurada
                    # y, sin desactivarlo, una pregunta con agregación difícil (p.ej.
                    # enmiendas por grupo) puede consumir los 8192 tokens de num_predict
                    # enteros en razonamiento oculto sin llegar a emitir el SPARQL: medido
                    # en vivo, ~32 min y 3 intentos en blanco en vez de un fallo rápido.
                    opts = dict(model=LLM_MODEL_GRAPHRAG, temperature=0,
                                num_predict=8192, num_ctx=8192,
                                client_kwargs={"timeout": 300})
                    if LLM_MODEL_GRAPHRAG.startswith("qwen3"):
                        opts["reasoning"] = False
                    _llm_cache[provider] = ChatOllama(**opts)
                    print(f"[+] GraphRAG LLM: Ollama ({LLM_MODEL_GRAPHRAG})", flush=True)
                elif provider == "groq":
                    from langchain_groq import ChatGroq
                    groq_key = os.environ.get("GROQ_API_KEY", "")
                    _llm_cache[provider] = ChatGroq(
                        model=LLM_MODEL_GROQ, temperature=0, api_key=groq_key, max_tokens=8192
                    )
                    print(f"[+] GraphRAG LLM: Groq ({LLM_MODEL_GROQ})", flush=True)
    return _llm_cache[provider]


# invoca el LLM con fallback automático (ollama para SPARQL, groq para narración)
def _llm_invoke(prompt: str, prefer: str = "ollama") -> str:
    orden = ["ollama", "groq"] if prefer == "ollama" else ["groq", "ollama"]
    groq_key = os.environ.get("GROQ_API_KEY", "")
    errores = []

    for proveedor in orden:
        if proveedor == "ollama":
            if not _ping_ollama():
                print("[!] GraphRAG: Ollama no disponible en localhost:11434", flush=True)
                continue
            try:
                return _get_llm("ollama").invoke(prompt).content
            except Exception as e:
                print(f"\n[!] GraphRAG Ollama falló — {type(e).__name__}: {e}", flush=True)
                errores.append(str(e))
        else:
            if not groq_key:
                continue
            try:
                print(f"[~] GraphRAG usando Groq ({'preferido' if prefer=='groq' else 'fallback'})...", flush=True)
                return _get_llm("groq").invoke(prompt).content
            except Exception as e:
                print(f"[!] GraphRAG Groq también falló — {type(e).__name__}: {e}", flush=True)
                errores.append(str(e))

    raise RuntimeError(f"GraphRAG: ningún LLM disponible (Ollama y Groq fallaron): {errores}")


_SPARQL_MODIFIER_RE = re.compile(
    r"(GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET|HAVING|VALUES|BINDINGS)\b", re.IGNORECASE)


def _clean_sparql(txt: str) -> str:
    txt = re.sub(r"```(?:sparql)?", "", txt).strip()
    m = re.search(r"\b(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)\b", txt, re.IGNORECASE)
    if m:
        txt = txt[m.start():]
    # El LLM copia el escape de llaves de los ejemplos del SCHEMA ({{ }} de
    # str.format) — rdflib lo tolera pero rompe los regex de _sanitize_sparql.
    txt = txt.replace("{{", "{").replace("}}", "}")
    # Quitar las líneas PREFIX que escribe el LLM: rdflib ya tiene bo:/br:/skos:/
    # rdfs:/xsd: enlazados desde el grafo, y el LLM a veces inventa el namespace
    # (PREFIX bo: <http://example.org/...> -> 0 resultados en silencio).
    txt = re.sub(r"(?im)^\s*PREFIX\s+\w*:\s*<[^>]*>\s*\.?\s*$\n?", "", txt)
    # Cortar la prosa que el LLM a veces añade DESPUÉS de la consulta ("Esta
    # consulta filtra...", "### Explicación:") pese a pedirle "solo SPARQL":
    # rompía g.query() con "Expected end of text". Se recorre contando llaves y
    # se corta cuando la profundidad vuelve a 0 (fin del patrón WHERE) + los
    # modificadores de solución que sigan. Contar llaves —y no rfind('}')—
    # porque la prosa suele CITAR trozos de la consulta, con sus '}' incluidos.
    depth, seen_open = 0, False
    for i, ch in enumerate(txt):
        if ch == "{":
            depth += 1
            seen_open = True
        elif ch == "}":
            depth -= 1
            if seen_open and depth == 0:
                rest = txt[i + 1:]
                mm = re.match(
                    r"\s*(?:(?:GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET|HAVING|VALUES)\b[^\n]*\s*)*",
                    rest, re.IGNORECASE)
                txt = txt[:i + 1] + (mm.group(0) if mm else "")
                break
    return txt.strip()


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


_RESULT_INDIVIDUALS = ("bo:Aprobada", "bo:Rechazada", "bo:Decae", "bo:Retirada",
                       "bo:AprobadaConEnmienda", "bo:SinResultado")

# rdfs:label reales de los grupos (literales planos, sin lang tag).
_GRUPO_LABELS = {
    "PP", "EH BILDU", "PSE-EE", "ELKARREKIN BILBAO", "GOAZEN BILBAO", "UDALBERRI",
    "EZKER BATUA-IU", "EAJ-PNV", "CIUDADANOS", "EQUIPO DE GOBIERNO", "GRUPO MIXTO",
}
_GRUPO_ALIAS = {
    "partido popular": "PP", "popular": "PP", "pp": "PP",
    "partido socialista": "PSE-EE", "socialistas vascos": "PSE-EE", "pse": "PSE-EE",
    "pse-ee": "PSE-EE", "psoe": "PSE-EE",
    "partido nacionalista vasco": "EAJ-PNV", "pnv": "EAJ-PNV", "eaj": "EAJ-PNV",
    "eaj-pnv": "EAJ-PNV", "jeltzale": "EAJ-PNV",
    "bildu": "EH BILDU", "eh bildu": "EH BILDU", "euskal herria bildu": "EH BILDU",
    "elkarrekin": "ELKARREKIN BILBAO", "elkarrekin bilbao": "ELKARREKIN BILBAO",
    "podemos": "ELKARREKIN BILBAO",
    "goazen": "GOAZEN BILBAO", "goazen bilbao": "GOAZEN BILBAO",
    "udalberri": "UDALBERRI", "bilbao en comun": "UDALBERRI", "bilbao en común": "UDALBERRI",
    "ezker batua": "EZKER BATUA-IU", "izquierda unida": "EZKER BATUA-IU", "iu": "EZKER BATUA-IU",
    "ciudadanos": "CIUDADANOS", "equipo de gobierno": "EQUIPO DE GOBIERNO",
    "gobierno municipal": "EQUIPO DE GOBIERNO", "grupo mixto": "GRUPO MIXTO",
}

_CANON_TEMAS = (
    "vivienda", "urbanismo", "movilidad", "medioambiente", "euskera", "cultura",
    "deporte", "educacion", "igualdad", "serviciossociales", "empleoeconomia",
    "presupuestos", "seguridad", "participacion", "turismo", "sanidad",
    "memoriahistorica", "derechoshumanos", "otros",
)

# URIs de tema REALES (nivel 1 + subtemas) de themes_skos.ttl — para detectar
# cuándo el LLM inventa un slug (br:t_movilidad_y_transporte, bo:t_seguridad...)
# que da 0 resultados en silencio. Si themes_skos.ttl cambia, este set también.
_REAL_TEMA_URIS = frozenset((
    "t_accesibilidad", "t_alquiler", "t_aparcamiento", "t_arte", "t_barrios",
    "t_bibliotecas", "t_bicicleta", "t_bilbobus", "t_calidad_aire",
    "t_peatonalizacion", "t_comercio", "t_cultura",
    "t_deporte", "t_derechoshumanos", "t_desahucios", "t_discapacidad",
    "t_educacion", "t_empleo", "t_empleoeconomia", "t_empresas", "t_energia",
    "t_espacio_publico", "t_euskera", "t_exclusion_social", "t_fiestas",
    "t_financiacion", "t_hosteleria", "t_igualdad", "t_juventud",
    "t_medioambiente", "t_memoriahistorica", "t_metro_bilbao", "t_movilidad",
    "t_museos", "t_ordenacion_urbana", "t_ordenanzas_fiscales", "t_otros",
    "t_participacion", "t_personas_mayores", "t_policia_municipal",
    "t_presupuesto_municipal", "t_presupuestos", "t_reciclaje",
    "t_rehabilitacion", "t_retribuciones", "t_sanidad", "t_seguridad",
    "t_serviciossociales", "t_subvenciones", "t_transparencia",
    "t_transporte_publico", "t_tranvia", "t_turismo", "t_urbanismo",
    "t_violencia_genero", "t_vivienda", "t_vivienda_social", "t_vivienda_vacia",
))

# label (prefLabel/altLabel, en minúsculas) -> slug del tema de NIVEL 1 al que
# pertenece (generado de themes_skos.ttl siguiendo skos:broader hasta la raíz).
# Usado por dos guardas de _sanitize_sparql: (a) colapsar un OR de etiquetas
# sueltas que en realidad pertenecen todas al mismo tema padre a una única
# bo:trataTemaAmplio directa (el roll-up materializado ya las incluye todas,
# enumerarlas a mano se queda corto si falta alguna -- verificado: "movilidad
# por año" enumerando solo 3 etiquetas dio 2024=24, el roll-up completo da 50);
# (b) detectar cuando bo:trataTemaAmplio+REGEX se usa sobre un literal que NO
# es un tema real (un nombre propio de empresa/persona confundido con un tema).
_LABEL_TOPLEVEL = {
    "accesibilidad": "t_movilidad", "alquiler": "t_vivienda",
    "alquiler social": "t_vivienda", "alquileres": "t_vivienda",
    "aparcamiento": "t_movilidad", "aparcamientos": "t_movilidad",
    "arte": "t_cultura", "ascensores": "t_movilidad", "aste nagusia": "t_cultura",
    "aurrekontuak eta fiskalitatea": "t_presupuestos", "autobuses": "t_movilidad",
    "autobús": "t_movilidad", "ayudas": "t_presupuestos",
    "ayudas económicas": "t_presupuestos", "ayudas sociales": "t_serviciossociales",
    "barrio": "t_urbanismo", "barrios": "t_urbanismo", "basuras": "t_medioambiente",
    "berdintasuna eta feminismoa": "t_igualdad", "besteak": "t_otros",
    "biblioteca": "t_cultura", "bibliotecas": "t_cultura", "bicicleta": "t_movilidad",
    "bicicletas": "t_movilidad", "bidebarrieta": "t_cultura", "bidegorri": "t_movilidad",
    "bidegorris": "t_movilidad", "bilbobus": "t_movilidad", "bonificaciones": "t_presupuestos",
    "buen gobierno": "t_participacion", "calidad del aire": "t_medioambiente",
    "calles peatonales": "t_movilidad", "carril bici": "t_movilidad",
    "circulación": "t_movilidad", "ciudad 30": "t_movilidad", "comercio": "t_empleoeconomia",
    "comercio local": "t_empleoeconomia", "contaminación atmosférica": "t_medioambiente",
    "contaminación del aire": "t_medioambiente", "cultura": "t_cultura",
    "deporte": "t_deporte", "deportes": "t_deporte", "derechos humanos": "t_derechoshumanos",
    "desahucio": "t_vivienda", "desahucios": "t_vivienda", "desempleo": "t_empleoeconomia",
    "discapacidad": "t_serviciossociales", "distritos": "t_urbanismo",
    "diversidad funcional": "t_serviciossociales", "economía": "t_empleoeconomia",
    "educación": "t_educacion", "eficiencia energética": "t_medioambiente",
    "emisiones": "t_medioambiente", "empleo": "t_empleoeconomia",
    "empleo público": "t_empleoeconomia", "empleo y economía": "t_empleoeconomia",
    "empresa": "t_empleoeconomia", "empresas": "t_empleoeconomia",
    "energética": "t_medioambiente", "energético": "t_medioambiente",
    "energía": "t_medioambiente", "energías renovables": "t_medioambiente",
    "enplegua eta ekonomia": "t_empleoeconomia", "envejecimiento": "t_serviciossociales",
    "ertzaintza": "t_seguridad", "espacio público": "t_urbanismo", "etxebizitza": "t_vivienda",
    "euskara": "t_euskera", "euskera": "t_euskera", "exclusión social": "t_serviciossociales",
    "feminismo": "t_igualdad", "fiestas": "t_cultura", "fiestas populares": "t_cultura",
    "financiación": "t_presupuestos", "fiscalidad": "t_presupuestos",
    "gaztedia": "t_serviciossociales", "giza eskubideak": "t_derechoshumanos",
    "gizarte zerbitzuak": "t_serviciossociales", "herritarren parte-hartzea": "t_participacion",
    "hezkuntza": "t_educacion", "hirigintza": "t_urbanismo", "hostelería": "t_empleoeconomia",
    "igualdad": "t_igualdad", "igualdad de género": "t_igualdad",
    "igualdad de mujeres y hombres": "t_igualdad", "igualdad y feminismo": "t_igualdad",
    "impuestos": "t_presupuestos", "industria": "t_empleoeconomia",
    "ingurumena": "t_medioambiente", "inversiones": "t_presupuestos",
    "joven": "t_serviciossociales", "juvenil": "t_serviciossociales",
    "juveniles": "t_serviciossociales", "juventud": "t_serviciossociales",
    "jóvenes": "t_serviciossociales", "kirola": "t_deporte", "kultura": "t_cultura",
    "limitación de velocidad": "t_movilidad", "limpieza": "t_medioambiente",
    "limpieza viaria": "t_medioambiente", "mayores": "t_serviciossociales",
    "medio ambiente": "t_medioambiente", "medioambiente": "t_medioambiente",
    "memoria historikoa": "t_memoriahistorica", "memoria histórica": "t_memoriahistorica",
    "mercados": "t_empleoeconomia", "metro": "t_movilidad", "metro bilbao": "t_movilidad",
    "movilidad": "t_movilidad", "movilidad y transporte": "t_movilidad",
    "mugikortasuna eta garraioa": "t_movilidad", "mujeres": "t_igualdad",
    "museo": "t_cultura", "museos": "t_cultura", "ordenación urbana": "t_urbanismo",
    "ordenanza fiscal": "t_presupuestos", "ordenanzas fiscales": "t_presupuestos",
    "osasuna": "t_sanidad", "ota": "t_movilidad", "otros": "t_otros",
    "pacificación del tráfico": "t_movilidad", "parking": "t_movilidad",
    "paro": "t_empleoeconomia", "parques y jardines": "t_urbanismo",
    "participación": "t_participacion", "participación ciudadana": "t_participacion",
    "peatonalizaciones": "t_movilidad", "peatonalización": "t_movilidad",
    "personas mayores": "t_serviciossociales", "pgou": "t_urbanismo",
    "plan general": "t_urbanismo", "planeamiento": "t_urbanismo", "plazas": "t_urbanismo",
    "pobreza": "t_serviciossociales", "pobreza energética": "t_medioambiente",
    "policía": "t_seguridad", "policía municipal": "t_seguridad",
    "presupuesto": "t_presupuestos", "presupuesto municipal": "t_presupuestos",
    "presupuestos": "t_presupuestos", "presupuestos municipales": "t_presupuestos",
    "presupuestos y fiscalidad": "t_presupuestos", "pymes": "t_empleoeconomia",
    "reciclaje": "t_medioambiente", "reducción de emisiones": "t_medioambiente",
    "regeneración urbana": "t_urbanismo", "rehabilitación": "t_urbanismo",
    "residuos": "t_medioambiente", "retribuciones": "t_presupuestos",
    "sala de estudio": "t_cultura", "salarios": "t_presupuestos",
    "salas de estudio": "t_cultura", "salud": "t_sanidad", "salud pública": "t_sanidad",
    "sanidad": "t_sanidad", "seguridad": "t_seguridad", "seguridad ciudadana": "t_seguridad",
    "segurtasuna": "t_seguridad", "servicios sociales": "t_serviciossociales",
    "sostenibilidad": "t_medioambiente", "subvenciones": "t_presupuestos",
    "subvención": "t_presupuestos", "sueldos": "t_presupuestos",
    "tasas": "t_presupuestos", "tercera edad": "t_serviciossociales",
    "transición energética": "t_medioambiente", "transparencia": "t_participacion",
    "transporte": "t_movilidad", "transporte público": "t_movilidad",
    "tranvía": "t_movilidad", "tráfico": "t_movilidad", "turismo": "t_turismo",
    "turismoa": "t_turismo", "urbanismo": "t_urbanismo",
    "violencia de género": "t_igualdad", "violencia machista": "t_igualdad",
    "vivienda": "t_vivienda", "vivienda deshabitada": "t_vivienda",
    "vivienda en alquiler": "t_vivienda", "vivienda protegida": "t_vivienda",
    "vivienda pública": "t_vivienda", "vivienda social": "t_vivienda",
    "vivienda vacía": "t_vivienda", "viviendas": "t_vivienda",
    "viviendas deshabitadas": "t_vivienda", "viviendas sociales": "t_vivienda",
    "viviendas vacías": "t_vivienda", "vpo": "t_vivienda", "zbe": "t_medioambiente",
    "zona 30": "t_movilidad", "zona de bajas emisiones": "t_medioambiente",
    "zonas de bajas emisiones": "t_medioambiente",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# mismo diccionario, pero con las claves sin acentos -- el LLM genera "tranvia"
# tan a menudo como "tranvía", y el lookup debe casar los dos.
_LABEL_TOPLEVEL_NORM = {_strip_accents(k): v for k, v in _LABEL_TOPLEVEL.items()}


# ---------------------------------------------------------------------------
# Validación semántica por embeddings (Ronda 41, 2026-09-17): generaliza la
# idea de las guardas deterministas de arriba -- en vez de un patrón regex
# nuevo por cada invención concreta que se va descubriendo una a una, esto
# cubre el "resto" (la cola larga de slugs inventados que no casan por
# substring con ninguno de los 19 temas de nivel 1 de _CANON_TEMAS, p.ej.
# "br:t_ciclovia" para bicicleta, o "br:t_okupacion" para desahucios -- sin
# solape de texto con el canónico real, el fixer por substring de abajo
# nunca los habría corregido). Sigue el enfoque de Sharma/Pal/Zouaq
# ("Reducing Hallucinations in LM-based SPARQL Query Generation Using
# Post-Generation Memory Retrieval", arXiv:2502.13369, citado en la
# bibliografía del TFG, Ronda 38): recuperación semántica contra las
# entidades REALES del grafo para corregir lo que el LLM se inventó, en vez
# de una guarda por patrón. Usa bge-m3 (mismo modelo que ya usa el proyecto
# en estudio_sparql/embed_bank.py y en la selección de few-shot dinámica de
# más arriba en este mismo archivo) contra TODAS las etiquetas reales
# (skos:prefLabel + skos:altLabel, castellano y euskera) de los 61 temas y
# subtemas de themes_skos.ttl -- no solo los 19 de nivel 1.
_THEMES_TTL = os.path.join(HERE, "themes_skos.ttl")
_tema_embed_cache = None
_tema_embed_lock = threading.Lock()


def _tema_embed_index():
    global _tema_embed_cache
    if _tema_embed_cache is not None:
        return _tema_embed_cache
    with _tema_embed_lock:
        if _tema_embed_cache is not None:
            return _tema_embed_cache
        import rdflib
        from rdflib.namespace import SKOS
        g = rdflib.Graph()
        g.parse(_THEMES_TTL, format="turtle")
        BO_ = rdflib.Namespace("http://bilbao.tfg/ontology#")
        textos, slugs = [], []
        for t in g.subjects(rdflib.RDF.type, BO_.Tema):
            slug = str(t).rsplit("/", 1)[-1]
            for pred in (SKOS.prefLabel, SKOS.altLabel):
                for lab in g.objects(t, pred):
                    textos.append(str(lab))
                    slugs.append(slug)
        from langchain_ollama import OllamaEmbeddings
        embedder = OllamaEmbeddings(model="bge-m3")
        vecs = embedder.embed_documents(textos)
        _tema_embed_cache = (textos, slugs, vecs, embedder)
        return _tema_embed_cache


# Umbral calibrado a mano contra 8 slugs inventados de prueba (Ronda 41):
# coincidencias genuinas (ciclovía->bicicleta, tráfico->movilidad, residuos->
# reciclaje, contaminación acústica->calidad_aire) puntuaron 0.71-1.00;
# coincidencias forzadas sin tema real equivalente en la taxonomía
# (arbolado, empadronamiento, wifi público, okupación) puntuaron 0.62-0.68 --
# corte limpio en 0.70. Por debajo, mejor dejar el slug sin corregir (cae al
# mecanismo ya existente de "0 filas -> reintento") que sustituir con
# confianza por un tema que no encaja de verdad.
def _fix_tema_uri_semantic(uri_slug: str, umbral: float = 0.70):
    try:
        textos, slugs, vecs, embedder = _tema_embed_index()
        qv = embedder.embed_query(uri_slug.replace("_", " ").replace("-", " "))

        def _cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            return dot / (na * nb + 1e-9)

        mejor_sim, mejor_slug = -1.0, None
        for i, v in enumerate(vecs):
            s = _cos(qv, v)
            if s > mejor_sim:
                mejor_sim, mejor_slug = s, slugs[i]
        if mejor_sim >= umbral:
            return f"br:{mejor_slug}"
    except Exception:
        pass
    return None


# mapea un slug de tema inventado por el LLM (br:t_presupuestos_fiscalidad...) al canónico real
def _fix_tema_uri(uri_slug: str) -> str:
    s = uri_slug.replace("-", "").replace("_", "")
    for canon in _CANON_TEMAS:
        if s.startswith(canon) or canon.startswith(s) or canon in s:
            return f"br:t_{canon}"
    # substring falló (p.ej. "ciclovia" no comparte texto con "bicicleta") --
    # fallback semántico antes de rendirse.
    semantico = _fix_tema_uri_semantic(uri_slug)
    return semantico or f"br:t_{uri_slug}"


# ---------------------------------------------------------------------------
# Capa de alias no destructiva ("adaptar la ontología al modelo", ver
# ESTUDIO_SPARQL_LOCAL.md Fase 1-2): reescribe las invenciones SISTEMÁTICAS
# del LLM (entidad/persona/grupo puestos como URI en vez de resolverse con
# label+REGEX) al patrón real del grafo, ANTES de sanitizar/ejecutar.
# ---------------------------------------------------------------------------
_ALIAS_GRUPOS = {
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
_ALIAS_VOTO_PERSONA = ("proponePersona", "intervino", "concejalVotoAFavor",
                       "concejalVotoEnContra", "concejalVotoAbstencion")
_ALIAS_STOP = {"concejal", "concejala", "concejales", "sr", "sra", "don", "dona", "doña",
              "grupo", "municipal", "el", "la", "los", "las", "de", "del", "senor",
              "senora", "señor", "señora", "persona", "entidad"}


def _alias_canon_grupo(raw: str):
    k = re.sub(r"[^a-z0-9]", "", raw.lower())
    if k in _ALIAS_GRUPOS:
        return "br:" + _ALIAS_GRUPOS[k]
    if k.startswith("grupo"):
        return "br:" + re.sub(r"^grupo_?", "grupo_", raw.lower())
    return None


def _alias_tokens(uri: str):
    u = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", uri)
    u = re.sub(r"[_\-]+", " ", u)
    toks = [w for w in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+", u.lower()) if len(w) > 1]
    keep = [w for w in toks if w not in _ALIAS_STOP]
    return keep or toks


def _alias_regex_and(var: str, uri: str) -> str:
    toks = _alias_tokens(uri) or [uri.lower()]
    return " && ".join(f'REGEX(STR({var}), "{t}", "i")' for t in toks)


def _alias_tail(body: str, subj: str, term: str) -> str:
    if term == ";":
        return body + f" . {subj} "
    return body + " " + term


def _alias_per_grp(m):
    pre, pred, uri = m.group(1), m.group(2), m.group(3)
    term = m.group(4) if (m.lastindex or 0) >= 4 else "."
    nom = {"votoAFavorDe": "concejalVotoAFavor", "votoEnContraDe": "concejalVotoEnContra",
           "seAbstuvo": "concejalVotoAbstencion"}[pred]
    body = (f'{pre} bo:{nom} ?per_f . ?per_f rdfs:label ?per_fl . '
            f'FILTER({_alias_regex_and("?per_fl", uri)})')
    return _alias_tail(body, pre, term)


def _alias_rewrite(sparql: str) -> str:
    if not sparql:
        return sparql
    s = sparql

    # <br:x> -> br:x (el modelo a veces mete la URI prefijada entre <>)
    s = re.sub(r"<(br:[A-Za-z_]\w*)>", r"\1", s)
    s = re.sub(r"<(bo:[A-Za-z_]\w*)>", r"\1", s)

    # tema con prefijo bo: -> br: (bo:t_euskera -> br:t_euskera)
    s = re.sub(r"\bbo:(t_[a-z_]+)\b", r"br:\1", s)

    # grupo con URI no canonica: br:EHBildu / br:EH_Bildu -> br:grupo_eh_bildu
    s = re.sub(r"\bbr:([A-Za-z][A-Za-z_]*)\b", lambda m: (
        _alias_canon_grupo(m.group(1)) or m.group(0)) if re.sub(r"[^a-z0-9]", "", m.group(1).lower()) in _ALIAS_GRUPOS else m.group(0), s)

    # entidad como URI: ?p bo:menciona br:Iberdrola -> ?p bo:menciona ?ent_f . ?ent_f rdfs:label ?ent_fl . FILTER(...)
    def _ent(m):
        pre, uri = m.group(1), m.group(2)
        uri = re.sub(r"^(entidad_|ent_|lugar_|org_|organizacion_)", "", uri, flags=re.I)
        body = (f'{pre} bo:menciona ?ent_f . ?ent_f rdfs:label ?ent_fl . '
                f'FILTER({_alias_regex_and("?ent_fl", uri)})')
        return _alias_tail(body, pre, m.group(3))
    s = re.sub(r"(\?\w+)\s+bo:menciona\s+br:([A-Za-z_]\w*)\s*([;.])", _ent, s)

    # persona como URI con predicados de persona / voto nominal
    def _per(m):
        pre, pred, uri = m.group(1), m.group(2), m.group(3)
        body = (f'{pre} bo:{pred} ?per_f . ?per_f rdfs:label ?per_fl . '
                f'FILTER({_alias_regex_and("?per_fl", uri)})')
        return _alias_tail(body, pre, m.group(4))
    s = re.sub(r"(\?\w+)\s+bo:(" + "|".join(_ALIAS_VOTO_PERSONA) + r")\s+br:([A-Za-z_]\w*)\s*([;.])", _per, s)

    # voto de GRUPO con un br:<algo> que NO es un grupo canonico -> es persona
    s = re.sub(r"(\?\w+)\s+bo:(votoAFavorDe|votoEnContraDe|seAbstuvo)\s+br:(?!grupo_)([A-Za-z_]\w*)\s*([;.])",
               _alias_per_grp, s)

    # rdfs:label "Nombre Apellido" EXACTO -> REGEX (rara vez coincide letra a letra con el grafo)
    def _lbl(m):
        var, lit, term = m.group(1), m.group(2), m.group(3)
        if len(lit.split()) > 4 or len(lit) < 3:
            return m.group(0)
        body = f'{var} rdfs:label ?lbl_f . FILTER({_alias_regex_and("?lbl_f", lit)})'
        return _alias_tail(body, var, term)
    s = re.sub(r'(\?\w+)\s+rdfs:label\s+"([^"]{3,50})"\s*([;.)}])', _lbl, s)

    return s


# reescribe anti-patrones del LLM que dan resultado equivocado sin error (lista en memoria/decisiones_tecnicas.md 3.1)
def _sanitize_sparql(sparql: str, verbose: bool = False, pregunta: str = "") -> str:
    original = sparql

    # --- (0) COUNT(DISTINCT ?x) donde ?x nunca se liga en el WHERE ---
    # El LLM a veces nombra la variable del SELECT sin que coincida con
    # ninguna del cuerpo de la consulta (p.ej. "SELECT (COUNT(DISTINCT ?c) AS
    # ?c) WHERE { ?p a bo:Proposicion ; ... }" -- ?c no aparece en ningún
    # triple). Una variable sin ligar en COUNT(DISTINCT ...) da 0 SIEMPRE, sin
    # error, independientemente de si existen filas reales. Si ?x no aparece
    # en ningún otro sitio del cuerpo, se sustituye por la variable de
    # Proposicion (la que sí se liga en "?p a bo:Proposicion"), que es
    # prácticamente siempre la entidad que se pretende contar.
    m0 = re.search(r"COUNT\s*\(\s*DISTINCT\s+\?(\w+)\s*\)", sparql, re.I)
    if m0:
        cvar = m0.group(1)
        where_m = re.search(r"WHERE\s*\{(.*)\}\s*$", sparql, re.I | re.S)
        body = where_m.group(1) if where_m else sparql
        if not re.search(rf"(?<!\w)\?{cvar}\b", body):
            subj_m = re.search(r"\?(\w+)\s+a\s+bo:Proposicion\b", sparql)
            if subj_m and subj_m.group(1) != cvar:
                sparql = re.sub(rf"(?<!\w)\?{cvar}\b", f"?{subj_m.group(1)}", sparql)

    # --- (1) OPTIONAL no-op en consulta de agregación ---
    if re.search(r"\bCOUNT\s*\(", sparql, re.I):
        # variables usadas dentro de agregados del SELECT (COUNT/SUM/AVG/MIN/MAX)
        agg_vars = set(re.findall(r"(?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?(\?\w+)", sparql, re.I))
        gb_match = re.search(r"GROUP\s+BY\s+([^\n]+)", sparql, re.I)
        gb_vars = set(re.findall(r"\?\w+", gb_match.group(1))) if gb_match else set()

        select_clause = re.search(r"SELECT\b(.*?)\bWHERE", sparql, re.I | re.S)
        select_vars = set(re.findall(r"\?\w+", select_clause.group(1))) if select_clause else set()

        def _promote(m):
            body = m.group(1)
            opt_vars = set(re.findall(r"\?\w+", body))
            bind_vars = set(re.findall(r"\bAS\s+(\?\w+)", body, re.I))
            # variables NUEVAS que aporta el OPTIONAL (por BIND o por aparecer aquí
            # por primera vez). Si alguna se usa en el SELECT/agregados/GROUP BY,
            # el OPTIONAL sí influye en el resultado → no tocar.
            introduced = bind_vars | (opt_vars - set(re.findall(r"\?\w+", sparql[:m.start()])))
            if introduced & (agg_vars | gb_vars | select_vars):
                return m.group(0)
            # No aporta nada al resultado. Si tiene un objeto concreto (individuo
            # de resultado, recurso br:*), el LLM lo puso como filtro → promover.
            if any(ind in body for ind in _RESULT_INDIVIDUALS) or re.search(r"\bbr:\w+", body):
                return body.strip().rstrip(".").strip() + " .\n  "
            return m.group(0)

        sparql = re.sub(r"OPTIONAL\s*\{([^{}]*)\}", _promote, sparql)

    # --- (2) JOIN cartesiano por rdfs:label ---
    for subj, obj in re.findall(r"\?(\w+)\s+rdfs:label\s+\?(\w+)", sparql):
        if len(re.findall(rf"\?{subj}\b", sparql)) == 1:  # ?subj solo aparece aquí
            sparql = re.sub(rf"\?{subj}\s+rdfs:label\s+\?{obj}\s*\.?", "", sparql)
            sparql = re.sub(rf"\(\s*[^()]*\?{obj}[^()]*\)\s*", "", sparql)  # agregados con ?obj
            sparql = re.sub(rf"(?<!\w)\?{obj}\b", "", sparql)               # ?obj suelto en SELECT
            gb = re.search(r"GROUP\s+BY\s+([^\n]*)", sparql, re.I)
            if gb and not re.search(r"\?\w+", gb.group(1)):
                sparql = re.sub(r"GROUP\s+BY\s*[^\n]*\n?", "", sparql, flags=re.I)

    # --- (2b) doble bo:Proposicion sin vincular (ratio mal construido) ---
    # Para un ratio (total + subconjunto), a veces el LLM declara DOS variables
    # "a bo:Proposicion" SIN VINCULAR en vez de OPTIONAL+BIND sobre la misma
    # variable: el COUNT del subconjunto deja de estar filtrado por las
    # condiciones de la variable principal (p.ej. el grupo) y cuenta de más en
    # silencio (verificado: "movilidad de EH Bildu rechazadas" daba 74 en vez
    # de 15 — contaba TODAS las rechazadas de movilidad, no solo las de EH
    # Bildu). Solo se corrige si comparten al menos una condición (tema/grupo)
    # -- si no comparten nada probablemente son dos conteos independientes de
    # verdad (p.ej. comparar dos grupos), y no se toca.
    try:
        decl_vars = []
        for m in re.finditer(r"\?(\w+)\s+a\s+bo:Proposicion\b", sparql):
            if m.group(1) not in decl_vars:
                decl_vars.append(m.group(1))
        if len(decl_vars) >= 2:
            counted = set(re.findall(r"COUNT\s*\(\s*(?:DISTINCT\s+)?\?(\w+)\s*\)", sparql, re.I))
            primary = decl_vars[0]

            def _bloque(var):
                return re.search(rf"\?{var}\s+a\s+bo:Proposicion\s*;\s*(.*?)\s*\.\s*(?=\?|\}}|$)", sparql, re.S)

            m_p = _bloque(primary)
            if primary in counted and m_p:
                primary_preds = {p.strip() for p in m_p.group(1).split(";")}
                for var in decl_vars[1:]:
                    if var not in counted:
                        continue
                    m_r = _bloque(var)
                    if not m_r:
                        continue
                    r_preds = [p.strip() for p in m_r.group(1).split(";")]
                    compartido = set(r_preds) & primary_preds
                    extra = [p for p in r_preds if p not in primary_preds]
                    if not compartido or not extra:
                        continue  # sin condicion compartida = probablemente conteos independientes de verdad
                    nuevo = f"OPTIONAL {{ ?{primary} {' ; '.join(extra)} . BIND(?{primary} AS ?{var}) }} "
                    sparql = sparql.replace(m_r.group(0), nuevo, 1)
    except Exception:
        pass  # cualquier fallo de parseo: dejar la consulta tal cual, no romper el resto de guardas

    # --- (3) roll-up temático a mano con skos:broader en vez de bo:trataTemaAmplio ---
    # El LLM ve "incluyendo subtemas" y escribe
    #   ?prop bo:trataSobre ?t . ?t skos:broader* br:t_presupuestos_fiscalidad
    # (a) skos:broader NO tiene el cierre transitivo materializado y (b) el URI
    # suele estar inventado -> 0 resultados. El razonador YA materializó el
    # roll-up en bo:trataTemaAmplio: basta con eso.
    m_bro = re.search(
        r"(?:bo:trataSobre|bo:trataTemaAmplio)\s+\?(\w+)\s*\.\s*"
        r"\?\1\s+skos:broader[*+]?\s+br:t_(\w+)\b\s*(?:;[^.}]*)?\.",
        sparql)
    if m_bro:
        tema_var, tema_slug = m_bro.groups()
        canon = _fix_tema_uri(tema_slug)
        sparql = sparql[:m_bro.start()] + f"bo:trataTemaAmplio {canon} ." + sparql[m_bro.end():]
        # limpiar referencias sueltas al ?tema que ya no existe
        sparql = re.sub(rf"\?{tema_var}\s+\w+:\w+\s+\?\w+\s*[;.]?", "", sparql)
        sparql = re.sub(rf"(?<!\w)\?{tema_var}\b", "", sparql)

    # --- (4) URI de tema inventada por el LLM (da 0 en silencio) ---
    # br:t_movilidad_y_transporte -> br:t_movilidad ; bo:t_seguridad -> br:t_seguridad
    def _fix_uri(m):
        pref, slug = m.group(1), m.group(2)
        if f"t_{slug}" in _REAL_TEMA_URIS:
            return f"br:t_{slug}"  # real: solo corrige el prefijo (bo: -> br:)
        canon = _fix_tema_uri(slug)
        return canon if canon != f"br:t_{slug}" else m.group(0)
    sparql = re.sub(r"\b(bo|br):t_(\w+)\b", _fix_uri, sparql)

    # --- (4a) bo:anio con tipo equivocado ---
    # El grafo guarda bo:anio como xsd:integer (literal plano). El LLM a veces
    # escribe `bo:anio "2015"^^xsd:int` (tipo distinto) o `bo:anio "2015"` (str)
    # como TRIPLE -> 0 resultados. Se normaliza a entero plano en cualquier
    # posición (triple o FILTER).
    sparql = re.sub(r'"(\d{4})"\s*\^\^\s*xsd:\w+', r"\1", sparql)
    sparql = re.sub(r'(bo:anio\s+)"(\d{4})"', r"\1\2", sparql)

    # --- (4a-bis) bo:fecha con tipo xsd:date equivocado ---
    # El grafo guarda bo:fecha como STRING plano "DD-MM-AAAA" (Literal sin
    # tipo, build_rdf.py: Literal(r["date"])), no como xsd:date. Si el LLM
    # escribe `bo:fecha "25-11-2010"^^xsd:date`, rdflib intenta parsear el
    # literal como fecha ISO al CONSTRUIR la consulta -- "25-11-2010" no es
    # ISO (DD-MM-AAAA, no AAAA-MM-DD) -> ValueError, la pregunta entera
    # fallaba sin llegar siquiera a ejecutarse contra el grafo (verificado en
    # vivo, Ronda 42, 2026-09-17: "¿Qué ocurrió con la proposición ... del
    # pleno del 25-11-2010?"). Se quita el tipo de cualquier literal con
    # forma DD-MM-AAAA (no solo tras bo:fecha, por si el LLM lo repite en un
    # FILTER aparte sobre la misma variable).
    sparql = re.sub(r'"(\d{2}-\d{2}-\d{4})"\s*\^\^\s*xsd:\w+', r'"\1"', sparql)

    # --- (4b) rdfs:label de grupo con nombre no canónico y/o lang tag ---
    # El LLM escribe `?g rdfs:label "Partido Popular"@es` en vez de usar
    # br:grupo_pp; los labels reales son planos y son "PP", "EH BILDU"... ->
    # 0 resultados en silencio.
    def _fix_label(m):
        nombre, canon_directo = m.group(1), m.group(1).strip().upper()
        if canon_directo in _GRUPO_LABELS:
            return f'rdfs:label "{canon_directo}"'  # bien, solo quita el @lang
        alias = _GRUPO_ALIAS.get(re.sub(r"\s+", " ", nombre.strip().lower()))
        return f'rdfs:label "{alias}"' if alias else m.group(0)
    sparql = re.sub(r'rdfs:label\s+"([^"]+)"(?:@\w+)?', _fix_label, sparql)

    # --- (5) "aprobada" en la pregunta = bo:Aprobada + bo:AprobadaConEnmienda ---
    # El LLM filtra solo por bo:Aprobada (estricta), infravalorando mucho el
    # conteo (PP 2015: 8 estrictas vs 31 con enmienda). Si la pregunta habla de
    # aprobación en general (sin pedir "sin enmienda"), se amplían las dos.
    pl = pregunta.lower()
    if re.search(r"aprob", pl) and not re.search(r"sin enmienda|con enmienda|estrict|tal cual|sin modificar", pl):
        # bo:tieneResultado bo:Aprobada puede venir seguido de "." / "}" / salto
        # de línea (fin del triple) o de ";" (sigue una lista de predicados sobre
        # el MISMO sujeto, p.ej. "...bo:tieneResultado bo:Aprobada ; bo:presentadaPor
        # ..."). Con ";" no se puede sustituir por "bo:tieneResultado ?_rApr .
        # FILTER(...)" tal cual -- dejaría un "." seguido de ";" inválido, o
        # cortaría la lista de predicados del sujeto. En su lugar: se cambia el
        # objeto a la variable y el FILTER se añade aparte, justo antes del "}"
        # que cierra el bloque ENVOLVENTE de ESE match concreto (contando
        # profundidad de llaves desde el punto de sustitución) -- insertarlo
        # antes del último "}" de toda la consulta (versión previa de este fix)
        # es incorrecto si el match está dentro de un OPTIONAL anidado: el
        # FILTER queda FUERA del OPTIONAL y lo convierte en obligatorio,
        # rompiendo el patrón de ratio/porcentaje (verificado en vivo: "%
        # aprobadas de vivienda" pasó de total=200 a total=41=aprob, el WHERE
        # entero quedó restringido a solo las aprobadas). Cada aparición se
        # trata por separado, de atrás hacia adelante para no invalidar índices.
        for mm in reversed(list(re.finditer(r"bo:tieneResultado\s+bo:Aprobada\b", sparql))):
            depth = 0
            cierre = None
            for i in range(mm.end(), len(sparql)):
                if sparql[i] == "{":
                    depth += 1
                elif sparql[i] == "}":
                    if depth == 0:
                        cierre = i
                        break
                    depth -= 1
            if cierre is None:
                continue
            filtro = " FILTER(?_rApr IN (bo:Aprobada, bo:AprobadaConEnmienda)) "
            sparql = (sparql[:mm.start()] + "bo:tieneResultado ?_rApr"
                      + sparql[mm.end():cierre] + filtro + sparql[cierre:])

    # --- (6) nombres de propiedad inexistentes que el LLM inventa por analogía ---
    # bo:interviene / bo:intervieneEn -> bo:intervino ; bo:proponente / bo:firmadaPor
    # -> bo:proponePersona ; bo:esAlcaldeDe / bo:alcalde -> patrón bo:esAlcalde true.
    sparql = re.sub(r"\bbo:intervien\w*\b", "bo:intervino", sparql)
    sparql = re.sub(r"\bbo:(?:firmadaPor|proponente|propuestaPor)\b", "bo:proponePersona", sparql)
    # fecha/año inventados: el LLM escribe `bo:presentadaEn ?f . FILTER(YEAR(?f)=2019)`
    # o `bo:fechaPresentacion`; el grafo SOLO tiene bo:anio (entero) y bo:fecha (str).
    _FECHA_FAKE = r"bo:(?:presentadaEn|presentadaEl|fechaPresentacion|fechaProposicion|fechaDePresentacion)"
    sparql = re.sub(r";\s*" + _FECHA_FAKE + r"\s+\?\w+(?=\s*[.;])", "", sparql)   # en medio de lista ";"
    sparql = re.sub(_FECHA_FAKE + r"\s+\?\w+\s*;\s*", "", sparql)                  # al principio de lista
    sparql = re.sub(_FECHA_FAKE + r"\s+\?\w+\s*(?=\})", "", sparql)               # triple suelto
    sparql = re.sub(r"YEAR\s*\(\s*\?\w+\s*\)", "?anio", sparql)
    sparql = re.sub(r"\?anio\s*=\s*(\d{4})\s*&&\s*\?anio\s*=\s*\1", r"?anio = \1", sparql)

    # --- (7) alcalde buscado como si tuviera label "Alcalde de Bilbao" o clase propia ---
    # El grafo NO tiene un nodo "Alcalde de Bilbao": el/la alcalde/sa es un
    # bo:Concejal con bo:esAlcalde true. Reescribe el patrón equivocado.
    sparql = re.sub(
        r"\?\w+\s+rdfs:label\s+\"[^\"]*[Aa]lcalde[^\"]*\"\s*[.;]",
        "?_alc bo:esAlcalde true .", sparql)
    sparql = re.sub(r"\?\w+\s+a\s+bo:Alcalde\b\s*[.;]", "?_alc bo:esAlcalde true .", sparql)

    # --- (7b) voto de PERSONA con el predicado de GRUPO ---
    # Si la pregunta es sobre un/a concejal/a ("qué concejal votó más...", "quién
    # votó en contra") y NO menciona grupo/partido, pero el SPARQL usa
    # bo:votoAFavorDe/votoEnContraDe/seAbstuvo (nivel GRUPO), el LLM se equivocó
    # de predicado: quería el NOMINAL (bo:concejalVoto*). El grafo tiene los dos
    # con nombres casi iguales y el LLM los confunde.
    if re.search(r"concejal|concejala|qu[ée]\s+persona|\bqui[ée]n\b", pregunta, re.I) \
       and not re.search(r"\bgrupo\b|\bpartido\b", pregunta, re.I):
        sparql = re.sub(r"\bbo:votoAFavorDe\b", "bo:concejalVotoAFavor", sparql)
        sparql = re.sub(r"\bbo:votoEnContraDe\b", "bo:concejalVotoEnContra", sparql)
        sparql = re.sub(r"\bbo:seAbstuvo\b", "bo:concejalVotoAbstencion", sparql)

    # --- (8) CONTAINS de subcadena con un término de UNA palabra -> REGEX con
    # límite de palabra. "zara" con CONTAINS casa "Zaragoza"/"Zarautz"/"Zarandoa";
    # con REGEX "\bzara\b" no. Solo términos de una palabra (los multipalabra
    # -"zona de bajas emisiones"- se dejan en CONTAINS, ahí no hay falso positivo).
    def _cont2regex(m):
        var, term = m.group(1), m.group(2)
        if not re.fullmatch(r"[0-9A-Za-zñáéíóúü]{3,}", term):  # una palabra, sin metacaracteres
            return m.group(0)
        return f'REGEX(STR({var}), "\\\\b{term}\\\\b", "i")'
    sparql = re.sub(
        r'CONTAINS\s*\(\s*LCASE\s*\(\s*(?:STR\s*\(\s*)?(\?\w+)\s*\)?\s*\)\s*,\s*"([^"]+)"\s*\)',
        _cont2regex, sparql, flags=re.I)

    # --- (9) REGEX de nombre/etiqueta insensible a TILDES ---
    # rdflib REGEX con flag "i" ignora mayúsculas pero NO acentos: "díez" no casa
    # "Diez". El LLM pone/quita tildes de forma arbitraria en apellidos (Díez,
    # Fernández, Muñoz, Ibarretxe...). Expandimos cada vocal/ñ del patrón a una
    # clase [aá] para que case en ambos sentidos. Solo toca literales de REGEX;
    # ampliar el match nunca da falsos positivos aquí (sigue anclado con \b).
    _ACC = {"a": "[aá]", "á": "[aá]", "e": "[eé]", "é": "[eé]", "i": "[ií]",
            "í": "[ií]", "o": "[oó]", "ó": "[oó]", "u": "[uúü]", "ú": "[uúü]",
            "ü": "[uúü]", "n": "[nñ]", "ñ": "[nñ]"}

    def _acc_insens(m):
        head, pat = m.group(1), m.group(2)
        if "[" in pat:  # ya trae clases, no tocar
            return m.group(0)
        return head + "".join(_ACC.get(c.lower(), c) if c.isalpha() else c
                              for c in pat) + '"'

    sparql = re.sub(
        r'(REGEX\s*\(\s*(?:LCASE\s*\(\s*)?(?:STR\s*\(\s*)?\?\w+\s*\)*\s*,\s*")([^"]+)"',
        _acc_insens, sparql, flags=re.I)

    # --- (10) "\b" de límite de palabra con UNA sola barra invertida dentro de un
    # literal REGEX -> SPARQL lo interpreta como ECHAR de retroceso (\x08), no como
    # backslash+b, y el regex deja de casar NADA sin dar ningún error (falso negativo
    # silencioso). Hace falta escribir DOS barras en el texto fuente de la consulta
    # para que, tras el unescape de SPARQL, llegue una sola al motor de regex.
    # Verificado en vivo: "Torre BBVA" con \b (una barra) -> 0 filas; con \\b -> 1.
    # El propio ejemplo few-shot de Petronor tenía este bug (corregido también).
    def _fix_lone_b_escape(m):
        head, pat = m.group(1), m.group(2)
        return head + re.sub(r"(?<!\\)\\b", r"\\\\b", pat) + '"'
    sparql = re.sub(
        r'(REGEX\s*\(\s*(?:LCASE\s*\(\s*)?(?:STR\s*\(\s*)?\?\w+\s*\)*\s*,\s*")([^"]+)"',
        _fix_lone_b_escape, sparql, flags=re.I)

    # --- (11) roll-up temático hecho a mano con un OR de etiquetas sueltas ---
    # "?p bo:trataTemaAmplio ?t . ?t skos:prefLabel ?lab .
    #  FILTER(STR(?lab)="movilidad" || STR(?lab)="tranvia" || ...)"
    # se queda corto en silencio si el LLM no enumera TODOS los subtemas reales
    # (verificado en vivo: "movilidad por año" con 3 etiquetas sueltas dio
    # 2024=24; el roll-up completo real da 2024=50). Si TODAS las etiquetas
    # listadas resuelven, vía la taxonomía real, al MISMO tema de nivel 1, se
    # colapsa a una única bo:trataTemaAmplio sobre ese tema padre -- el
    # razonador ya materializó el roll-up completo, así que no hace falta (ni
    # es fiable) que el LLM enumere subtemas a mano.
    def _collapse_rollup(m):
        tvar, lvar, disj = m.group(1), m.group(2), m.group(3)
        labels = re.findall(r'STR\s*\(\s*\?' + re.escape(lvar) + r'\s*\)\s*=\s*"([^"]+)"', disj, re.I)
        if len(labels) < 2:
            return m.group(0)
        tops = [_LABEL_TOPLEVEL_NORM.get(_strip_accents(lab.lower())) for lab in labels]
        if any(t is None for t in tops) or len(set(tops)) != 1:
            return m.group(0)  # etiqueta desconocida o no todas del mismo padre: no tocar
        return f"bo:trataTemaAmplio br:{tops[0]} ."
    sparql = re.sub(
        r"bo:trataTemaAmplio\s+\?(\w+)\s*\.\s*\?\1\s+skos:prefLabel\s+\?(\w+)\s*\.\s*"
        r'FILTER\s*\(\s*((?:STR\s*\(\s*\?\2\s*\)\s*=\s*"[^"]+"\s*(?:\|\|\s*)?)+)\)\s*\.?',
        _collapse_rollup, sparql, flags=re.I)

    # --- (12) bo:trataTemaAmplio con una etiqueta que NO es un tema real ---
    # (nombre propio de empresa/persona/lugar confundido con un tema de la
    # taxonomía) -> 0 resultados en silencio, porque ese "tema" no existe.
    # bo:menciona SÍ es el predicado correcto para "qué se ha dicho sobre <nombre
    # propio>" -- verificado en vivo: "qué se ha dicho sobre el BBVA" generó
    # bo:trataTemaAmplio con REGEX "bbva" 2 de 3 veces (no determinista), pese a
    # que el esquema ya trae un ejemplo few-shot equivalente (Petronor,
    # bo:menciona). Solo se reescribe si el literal no casa NINGÚN tema/subtema
    # real conocido (evita falsos positivos: "vivienda", "tráfico"... sí siguen
    # usando trataTemaAmplio).
    def _fix_tema_vs_entidad(m):
        tvar, lvar, pat = m.group(1), m.group(2), m.group(3)
        # colapsa cada clase [eé] a su primera letra -- NO la borra entera: el
        # propio guard (9) de arriba expande cada vocal de un literal legítimo
        # ("presupuestos" -> "pr[eé]s[uú]p[uú][eé]st[oó]s") ANTES de llegar aquí;
        # borrar la clase entera (versión previa de este guard) destrozaba la
        # palabra hasta dejarla irreconocible y causaba un falso "no es un tema
        # real" -- verificado en vivo: "presupuestos y fiscalidad, incluyendo
        # subtemas" pasó de 733 a 0 por este bug exacto.
        sin_clases = re.sub(r"\[([^\]]+)\]", lambda cm: cm.group(1)[0], pat)
        sin_clases = re.sub(r"\\+b", "", sin_clases)
        # el patrón puede ser una alternancia "presupuestos|fiscalidad" (varios
        # temas reales a la vez, no una sola entidad) -- si CUALQUIER término
        # de la alternancia casa un tema/subtema real, se deja tal cual.
        terminos = [_strip_accents(re.sub(r"[^a-záéíóúñ]", "", t.lower())) for t in sin_clases.split("|")]
        if any(term and any(term in lab or lab in term for lab in _LABEL_TOPLEVEL_NORM) for term in terminos):
            return m.group(0)  # sí es un tema/subtema real (o una alternancia de varios): no tocar
        return (f'bo:menciona ?{tvar} . ?{tvar} rdfs:label ?{lvar} . '
                f'FILTER(REGEX(STR(?{lvar}), "{pat}", "i"))')
    sparql = re.sub(
        r'bo:trataTemaAmplio\s+\?(\w+)\s*\.\s*\?\1\s+skos:prefLabel\s+\?(\w+)\s*\.\s*'
        r'FILTER\s*\(\s*REGEX\s*\(\s*STR\s*\(\s*\?\2\s*\)\s*,\s*"([^"]+)"\s*,\s*"i"\s*\)\s*\)\s*\.?',
        _fix_tema_vs_entidad, sparql, flags=re.I)

    # --- (13) "mandato actual/legislatura...hasta ahora" reducido a un único
    # año exacto en vez de un rango abierto ---
    # "desde que empezó el mandato actual en 2023 hasta ahora" es un RANGO
    # plurianual (desde el inicio del mandato hasta la fecha más reciente del
    # corpus), no un año exacto -- verificado en vivo: el LLM ve "2023" en la
    # pregunta y genera FILTER(?anio = 2023) (328, solo ese año) en vez de
    # FILTER(?anio >= 2023) (~1065 reales, 2023-2026). Solo se reescribe si la
    # pregunta menciona explícitamente mandato/legislatura EN CURSO (para no
    # tocar preguntas legítimas sobre un año exacto o un mandato ya cerrado).
    _MANDATO_INICIO = (2007, 2011, 2015, 2019, 2023)
    if re.search(r"\bmandato\b|\blegislatura\b", pregunta, re.I) and \
       re.search(r"actual|en curso|hasta (la fecha|ahora|hoy)", pregunta, re.I):
        def _mandato_range(m):
            var, anio = m.group(1), int(m.group(2))
            if anio in _MANDATO_INICIO:
                return f"FILTER(?{var} >= {anio})"
            return m.group(0)
        sparql = re.sub(r"FILTER\s*\(\s*\?(\w+)\s*=\s*(\d{4})\s*\)", _mandato_range, sparql)

    # --- (14) prefijo de espacio de nombres inventado (p.ej. "r3:label" en
    # vez de "rdfs:label") ---
    # Visto en vivo (Ronda 40, 2026-09-16): al pedirle al LLM que "corrija"
    # una consulta con varios "rdfs:label" repetidos (p.ej. una pregunta de
    # una sola proposición con muchos OPTIONAL: grupo, tema, entidad,
    # concejal...), en vez de arreglar el error real introdujo un prefijo que
    # nunca declaró ("r3:") copiando el patrón "xx:label" -- rdflib falla con
    # "Unknown namespace prefix" y, al pedir corrección otra vez con el mismo
    # contexto, el modelo repitió y amplió el mismo error en vez de
    # eliminarlo (verificado: 3/3 intentos fallidos, la pregunta degradó a
    # "no he podido traducir esa pregunta" en vez de responder). Cualquier
    # prefijo NO declarado en el PREFIX de cabecera se reescribe al prefijo
    # real ya usado en la MISMA consulta con ese mismo nombre local (p.ej.
    # otro "rdfs:label" presente); si no hay ninguna coincidencia así,
    # "label"/"comment"/etc. caen a rdfs: por ser el vocabulario común más
    # repetido del schema. No toca "bo:"/"br:"/"skos:"/"rdfs:"/"xsd:" (los
    # únicos realmente declarados) ni las URIs http:// del PREFIX en sí.
    _KNOWN_PREFIXES = {"bo", "br", "skos", "rdfs", "xsd", "rdf", "owl"}
    _RDFS_TERMS = {"label", "comment", "seeAlso", "subClassOf", "domain", "range"}
    _declared_used = {}
    for _pfx, _local in re.findall(r"\b([a-zA-Z][\w-]*):([a-zA-Z_][\w-]*)", sparql):
        if _pfx in _KNOWN_PREFIXES:
            _declared_used.setdefault(_local, _pfx)

    def _fix_unknown_prefix(m):
        pfx, local = m.group(1), m.group(2)
        if pfx in _KNOWN_PREFIXES:
            return m.group(0)
        real = _declared_used.get(local) or ("rdfs" if local in _RDFS_TERMS else None)
        return f"{real}:{local}" if real else m.group(0)

    sparql = re.sub(r"\b([a-zA-Z][\w-]*):([a-zA-Z_][\w-]*)", _fix_unknown_prefix, sparql)

    # --- (15) fecha literal en formato DD/MM/YYYY en vez de DD-MM-YYYY ---
    # bo:fecha se guarda SIEMPRE con guiones y cero a la izquierda ("24-09-2015",
    # ver build_rdf.py: Literal(r["date"]) tal cual viene del acta) pero el
    # SCHEMA no lo dice explícitamente y el LLM a veces escribe la fecha con
    # barras (formato habitual en español) -- comparación silenciosa contra un
    # string que nunca puede coincidir, 0 filas sin error (verificado en vivo,
    # Ronda 40: "24/09/2015" no encuentra nada, "24-09-2015" sí). Cualquier
    # literal "D/M/AAAA" o "DD/MM/AAAA" se normaliza a guiones + 2 dígitos.
    def _fix_date_slash(m):
        d, mth, y = m.group(1), m.group(2), m.group(3)
        return f'"{int(d):02d}-{int(mth):02d}-{y}"'
    sparql = re.sub(r'"(\d{1,2})/(\d{1,2})/(\d{4})"', _fix_date_slash, sparql)

    sparql = re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", sparql)).strip()
    if verbose and sparql != original:
        print(f"[SPARQL saneado]\n{sparql}\n")
    return sparql


# si la pregunta pide un %, calcula el ratio en código y lo inyecta al contexto
def _augment_ratios(rows: list, pregunta: str) -> str:
    if not re.search(r"porcentaje|proporci[oó]n|\btasa\b|ratio|\bpor ?ciento\b|%", pregunta, re.I):
        return ""
    out = []
    for r in rows[:20]:
        if not isinstance(r, dict):
            continue
        nums = [(k, int(v)) for k, v in r.items()
                if isinstance(v, str) and re.fullmatch(r"\d+", v)]
        if len(nums) < 2:
            continue
        nums.sort(key=lambda kv: kv[1])
        (kn, n), (kd, d) = nums[0], nums[-1]
        if d > 0 and n <= d and kn != kd:
            etiqueta = " ".join(str(v) for k, v in r.items()
                                if isinstance(v, str) and not re.fullmatch(r"\d+", v)) or "total"
            out.append(f"  {etiqueta}: {n}/{d} = {100 * n / d:.1f}%")
    return ("\nPORCENTAJES YA CALCULADOS (usa EXACTAMENTE estos, no recalcules):\n"
            + "\n".join(out)) if out else ""


# ---------------------------------------------------------------------------
# Filtro previo cero-LLM (Ronda 39, 2026-09-16): estudio_sparql/arm_g.py es
# un prototipo de investigación que reconoce ~13 formas de pregunta frecuentes
# (por embedding bge-m3) y monta la SPARQL con plantillas + regex, sin tocar
# ningún LLM. Contra las 22 preguntas reales de regression_qa.py resuelve
# 17/22 (77%) en ~300ms y gratis, con 0 fallos conocidos tras añadir la guarda
# de "2+ grupos mencionados" (ver arm_g._n_grupos_mencionados). Se acepta su
# respuesta SOLO si la consulta ejecuta sin error y devuelve filas no vacías;
# cualquier otro caso (patrón no reconocido, plantilla no aplicable, o el
# propio Ollama de embeddings caído) cae al camino normal con LLM sin más
# efecto que la latencia de haberlo intentado.
_ESTUDIO_SPARQL_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "estudio_sparql")


def _try_arm_g(pregunta, g, verbose):
    try:
        if _ESTUDIO_SPARQL_DIR not in sys.path:
            sys.path.append(_ESTUDIO_SPARQL_DIR)   # arm_g.py hace "import embed_bank" a pelo
        import arm_g as _arm_g
    except ImportError:
        return None
    try:
        res = _arm_g.answer(pregunta)
    except Exception as e:
        if verbose:
            print(f"[Arm G] fallo al intentar el filtro previo, cae a LLM: {e}", flush=True)
        return None
    if not res.get("sparql"):
        return None
    try:
        rows = [{str(v): str(row[v]) for v in row.labels} for row in g.query(res["sparql"])]
    except Exception:
        return None
    if not rows:
        return None
    if verbose:
        print(f"[Arm G] resuelto sin LLM (tmpl={res['template']} sim={res['similarity']})"
              f"\n[SPARQL]\n{res['sparql']}\n")
    return res["sparql"], rows


def graph_answer(pregunta: str, verbose=True, sparql_provider: str = "ollama"):
    # sparql_provider: "ollama" (qwen3:8b, por defecto -- local y gratis) o
    # "groq" (openai/gpt-oss-120b) para el paso de GENERACIÓN de SPARQL (no
    # afecta a la narración del paso 3, que ya usa Groq por separado). Permite
    # comparar en vivo cuál acierta más en los casos que qwen3:8b falla de
    # forma no determinista (ver ESTUDIO_SPARQL_LOCAL.md: Groq ~90% vs
    # qwen3:8b+F_embed 94% en el gold set -- ayuda pero no es infalible, y
    # comparte la misma cuota diaria de Groq que la narración).
    g = _load_graph()

    # 0. Filtro previo cero-LLM (ver _try_arm_g) -- si resuelve la pregunta,
    # nos saltamos los pasos 1-2 (generación + reintentos con LLM) y vamos
    # directos a la narración con estas mismas filas.
    armg = _try_arm_g(pregunta, g, verbose)
    sparql, rows = armg if armg else (None, None)

    if rows is None:
        # 1. Generar SPARQL
        sparql = _sanitize_sparql(
            _alias_rewrite(_clean_sparql(_llm_invoke(_build_sparql_prompt(pregunta), prefer=sparql_provider))), verbose, pregunta)
        if verbose:
            print(f"\n[SPARQL]\n{sparql}\n")

        # 2. Ejecutar (hasta 2 reintentos si hay error de sintaxis O si la consulta
        #    ejecuta bien pero devuelve 0 filas — antes ese caso no reintentaba y
        #    se narraba directamente como "no hay datos", aunque a menudo el dato
        #    sí existe y lo que falló fue la consulta: una URI de tema/entidad mal
        #    resuelta). El prompt de corrección REPITE el schema entero: "corrígela"
        #    a secas hacía que el modelo inventara vocabulario (PREFIX example.org,
        #    WITH, subconsultas) porque perdía el contexto de qué existe en el grafo.
        error = None
        for intento in range(3):
            try:
                rows = [{str(v): str(row[v]) for v in row.labels} for row in g.query(sparql)]
                error = None
            except Exception as e:
                rows = []
                error = e

            vacia = error is None and not rows
            if error is None and not vacia:
                break                      # filas de verdad -> narrar con esto
            if intento == 2:
                if error is not None:
                    # Ninguna consulta válida en 3 intentos: la pregunta cae fuera de
                    # lo que el modelo sabe expresar en SPARQL sobre este grafo (p.ej.
                    # "proposiciones conjuntas" — el grafo solo guarda un grupo por
                    # proposición). Degradar con gracia en vez de romper el chat.
                    if verbose:
                        print(f"[!] SPARQL irrecuperable tras 3 intentos: {error}", flush=True)
                    return {
                        "sparql": sparql,
                        "rows": [],
                        "answer": ("No he podido traducir esa pregunta a una consulta válida sobre el "
                                   "grafo. Puede que pida un dato que el grafo no distingue (por ejemplo, "
                                   "proposiciones presentadas conjuntamente por varios grupos: el grafo "
                                   "guarda un único grupo por proposición). Prueba a reformularla de forma "
                                   "más concreta."),
                    }
                break                      # 0 filas tras 2 intentos de reformular -> se acepta como respuesta

            if error is not None:
                fix = _llm_invoke(
                    f"{SCHEMA}\n\nPREGUNTA: {pregunta}\n\n"
                    f"Esta consulta SPARQL DIO ERROR: {error}\n"
                    f"Consulta con error:\n{sparql}\n\n"
                    "Genera una consulta NUEVA que responda la pregunta, más simple, usando "
                    "SOLO el vocabulario del schema de arriba (NO inventes PREFIX, WITH, "
                    "subconsultas ni URIs). Para porcentajes/ratios devuelve solo el total y "
                    "el subconjunto con OPTIONAL+BIND. Devuelve SOLO la consulta SPARQL.",
                    prefer=sparql_provider,
                )
            else:
                fix = _llm_invoke(
                    f"{SCHEMA}\n\nPREGUNTA: {pregunta}\n\n"
                    f"Esta consulta SPARQL se ejecutó bien pero NO devolvió NINGUNA fila:\n{sparql}\n\n"
                    "Antes de repetir el mismo patrón, revisa: ¿la URI de tema/grupo que usaste está "
                    "en la lista exacta del schema? ¿Inventaste la URI de una entidad o persona en vez "
                    "de resolverla con rdfs:label + REGEX? ¿el predicado existe tal cual? Genera una "
                    "consulta ALTERNATIVA (distinta de la anterior) que sí pueda encontrar datos si "
                    "existen en el grafo. Si tras revisarlo la consulta anterior ya era correcta y el "
                    "grafo simplemente no tiene ese dato, repite la misma. Devuelve SOLO la consulta SPARQL.",
                    prefer=sparql_provider,
                )
            sparql = _sanitize_sparql(_alias_rewrite(_clean_sparql(fix)), verbose, pregunta)
            if verbose:
                print(f"[SPARQL corregido #{intento + 1}]\n{sparql}\n")

    rows = _fix_degenerate_groupby(rows, sparql)
    filas = "\n".join(str(r) for r in rows[:50]) or "(sin resultados en el grafo)"
    filas += _augment_ratios(rows, pregunta)
    if verbose:
        print(f"[FILAS] {len(rows)}")

    # 3. Respuesta narrativa basada solo en los datos del grafo
    answer = _llm_invoke(ANSWER_PROMPT.format(pregunta=pregunta, filas=filas, sparql=sparql), prefer="groq")

    # 4. Citas de fuente (página real del PDF) cuando la consulta trae
    # proposiciones individuales -- mismo formato que el RAG vectorial, para
    # que el usuario pueda verificar el dato en el acta igual en los dos
    # sistemas. graph_sources() ya existía pero nunca se llamaba desde aquí.
    fuentes = graph_sources(rows)
    if fuentes:
        bloque = "\n\n" + "=" * 60 + "\nFUENTES UTILIZADAS:\n"
        for i, f in enumerate(fuentes, 1):
            pdf = os.path.basename(f.get("pdf", "Acta"))
            bloque += f" [{i}] {pdf} | pág. {f.get('pagina', '?')} | {f.get('fecha', '')} | {f.get('titulo', '')[:60]}\n"
        answer += bloque

    return {"sparql": sparql, "rows": rows, "answer": answer}


# URI de una proposición individual (br:prop_<id>) — 100% de las 3422
# proposiciones tienen bo:fecha/bo:pagina/bo:fuentePdf/bo:tituloTopic
# (verificado), así que cualquier fila que la traiga permite una cita exacta
# a la página real del PDF, igual de precisa que las del RAG vectorial.
_PROP_URI_RE = re.compile(r'^http://bilbao\.tfg/resource/prop_[0-9a-f]+$')


# extrae URIs de proposición de las filas SPARQL y devuelve sus datos de cita
# (fecha, página, ruta del PDF, título) en una sola consulta por lotes.
# Solo funciona si la consulta generada por el LLM enlaza ?p directamente —
# preguntas de LISTADO ("qué proposiciones...", "lista las...") lo hacen;
# preguntas de AGREGADO puro (COUNT/GROUP BY sin ?p en el SELECT) no tienen
# ninguna proposición individual que citar — devuelve [] en ese caso, no un
# error (ver memoria/decisiones_tecnicas.md 2.10: es una limitación estructural
# de qué puede citarse desde un agregado, no un fallo).
def graph_sources(rows: list, limit: int = 15) -> list:
    uris = []
    seen = set()
    for row in rows:
        for v in row.values():
            if v not in seen and _PROP_URI_RE.match(v):
                seen.add(v)
                uris.append(v)
                if len(uris) >= limit:
                    break
        if len(uris) >= limit:
            break
    if not uris:
        return []

    g = _load_graph()
    values = " ".join(f"<{u}>" for u in uris)
    q = _PREFIXES + f"""
    SELECT ?p ?fecha ?pagina ?pdf ?titulo WHERE {{
      VALUES ?p {{ {values} }}
      ?p bo:fecha ?fecha ; bo:pagina ?pagina ; bo:fuentePdf ?pdf ; bo:tituloTopic ?titulo .
    }}"""
    out = []
    for row in g.query(q):
        d = {str(v): str(row[v]) for v in row.labels}
        out.append(d)
    # mismo orden que aparecieron en las filas originales, no el de la VALUES
    orden = {u: i for i, u in enumerate(uris)}
    out.sort(key=lambda d: orden.get(d.get("p", ""), 999))
    return out


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "¿Cuántas proposiciones sobre vivienda ha presentado cada grupo?"
    res = graph_answer(q)
    print("\n=== RESPUESTA ===\n" + res["answer"])
