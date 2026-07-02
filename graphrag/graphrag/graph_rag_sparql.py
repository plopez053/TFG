"""Fase 4: GraphRAG puro — SPARQL sobre el grafo RDF razonado → respuesta narrativa.

Flujo: pregunta → SPARQL → filas estructuradas (con títulos, fechas, resultados) → LLM → respuesta.

El LLM genera una respuesta NARRATIVA usando SOLO los datos del grafo:
agrega cifras, describe tendencias temporales y menciona ejemplos concretos.
NO usa ChromaDB (eso es el RAG vectorial, el sistema comparado).

Uso:
  python graphrag/graphrag/graph_rag_sparql.py "¿cuántas proposiciones de vivienda por grupo?"
"""
import os
import re
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_TTL = os.path.join(HERE, "bilbao_reasoned.ttl")
GRAPHRAG_LLM_MODEL = "qwen2.5:7b"

# ---------------------------------------------------------------------------
# Schema SPARQL — URIs exactas para evitar errores de generación
# ---------------------------------------------------------------------------
SCHEMA = """GRAFO RDF (razonado con OWL-RL) del Pleno del Ayuntamiento de Bilbao.

PREFIJOS — úsalos siempre:
  PREFIX bo:   <http://bilbao.tfg/ontology#>
  PREFIX br:   <http://bilbao.tfg/resource/>
  PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

CLASES:
  ?prop  a bo:Proposicion ; bo:tituloTopic ?titulo ; bo:fecha ?fecha ; bo:anio ?anio ;
           bo:votosFavor ?vf ; bo:votosContra ?vc .
  ?grupo a bo:Grupo  ; rdfs:label ?nombreGrupo .
  ?pleno a bo:Pleno  ; rdfs:label ?fechaPleno ; bo:anio ?anio .
  ?tema  a bo:Tema   ; skos:prefLabel ?labelTema .
  ?ent   a bo:Entidad ; rdfs:label ?nombreEnt .

PROPIEDADES (nombres exactos):
  ?prop bo:presentadaPor   ?grupo   # quién presenta
  ?prop bo:enPleno         ?pleno   # en qué pleno
  ?prop bo:tieneResultado  ?res     # resultado: usar con los individuos de abajo
  ?prop bo:trataSobre      ?tema    # tema directo (canónico o subtema libre)
  ?prop bo:trataTemaAmplio ?tema    # tema + subtemas inferidos por OWL-RL (roll-up)
  ?prop bo:menciona        ?ent     # entidades mencionadas

INDIVIDUOS de resultado (usar con bo:tieneResultado, NUNCA con bo:votoTexto):
  bo:Aprobada  bo:Rechazada  bo:Decae  bo:Retirada  bo:AprobadaConEnmienda  bo:SinResultado

URIs EXACTAS de los grupos (usar cuando filtres por grupo concreto):
  br:grupo_pp                  → PP
  br:grupo_eh_bildu            → EH BILDU
  br:grupo_pse_ee              → PSE-EE
  br:grupo_elkarrekin_bilbao   → ELKARREKIN BILBAO
  br:grupo_goazen_bilbao       → GOAZEN BILBAO
  br:grupo_udalberri           → UDALBERRI
  br:grupo_eaj_pnv             → EAJ-PNV
  br:grupo_ciudadanos          → CIUDADANOS
  br:grupo_vox                 → VOX
  br:grupo_ezker_batua_iu      → EZKER BATUA-IU
  br:grupo_equipo_de_gobierno  → EQUIPO DE GOBIERNO
  br:grupo_grupo_mixto         → GRUPO MIXTO

URIs EXACTAS de los 18 TEMAS CANÓNICOS (usar cuando filtres por tema concreto):
  br:t_vivienda          → vivienda
  br:t_urbanismo         → urbanismo
  br:t_movilidad         → movilidad y transporte
  br:t_medioambiente     → medio ambiente
  br:t_euskera           → euskera
  br:t_cultura           → cultura
  br:t_deporte           → deporte
  br:t_educacion         → educacion
  br:t_igualdad          → igualdad y feminismo
  br:t_serviciossociales → servicios sociales
  br:t_empleoeconomia    → empleo y economia
  br:t_presupuestos      → presupuestos y fiscalidad
  br:t_seguridad         → seguridad
  br:t_participacion     → participacion ciudadana
  br:t_turismo           → turismo
  br:t_sanidad           → sanidad
  br:t_memoriahistorica  → memoria historica
  br:t_derechoshumanos   → derechos humanos

CUÁNDO usar trataSobre vs trataTemaAmplio:
- bo:trataSobre: recupera proposiciones que tienen ese tema como principal O secundario directo.
  Usa SIEMPRE la URI exacta del tema canónico: ?prop bo:trataSobre br:t_vivienda
  O usa FILTER si no sabes la URI: ?prop bo:trataSobre ?tema . ?tema skos:prefLabel ?lab . FILTER(CONTAINS(LCASE(STR(?lab)), "vivienda"))
- bo:trataTemaAmplio: incluye TAMBIÉN las proposiciones con subtemas inferidos (roll-up OWL-RL).
  Útil cuando quieres todos los subtemas ("alquiler social" ⊑ "vivienda"). Puede devolver más filas.
- Para preguntas de conteo/ranking por tema, PREFERIR bo:trataSobre con URI exacta: más preciso y rápido.

REGLAS obligatorias:
- NUNCA uses el patrón de nodo anónimo [skos:prefLabel "vivienda"]: usa URI exacta o variable+FILTER.
    CORRECTO:  ?prop bo:trataSobre br:t_vivienda
    CORRECTO:  ?prop bo:trataSobre ?tema . ?tema skos:prefLabel ?lab . FILTER(CONTAINS(LCASE(STR(?lab)), "vivienda"))
    INCORRECTO: ?prop bo:trataSobre [skos:prefLabel "vivienda"]
- Para filtrar por resultado: ?prop bo:tieneResultado bo:Rechazada
- Para filtrar por grupo: ?prop bo:presentadaPor br:grupo_pp
- Para el label del grupo: ?grupo rdfs:label ?nombreGrupo
- bo:anio se almacena como xsd:integer. Para filtrar por año usa FILTER(?anio = 2023) con entero sin comillas.
  CORRECTO:   FILTER(?anio = 2023)
  INCORRECTO: FILTER(?anio = "2023") ← cadena de texto, no coincide
  INCORRECTO: FILTER(?anio = "2023"^^xsd:int) ← tipo incorrecto (es xsd:integer, no xsd:int)
- Si filtras por un grupo concreto Y quieres su label, SIEMPRE enlaza ?grupo como variable primero:
  CORRECTO:   ?prop bo:presentadaPor ?grupo . ?grupo rdfs:label ?nombreGrupo .
              FILTER(?grupo = br:grupo_eh_bildu)
  INCORRECTO: ?prop bo:presentadaPor br:grupo_eh_bildu . ?grupo rdfs:label ?nombreGrupo .
              (aquí ?grupo queda desligado → error o 0 resultados)
- Para rankings usa COUNT + GROUP BY + ORDER BY DESC + LIMIT 50.
- EXCLUYE br:grupo_desconocido de los rankings a menos que se pida explícitamente.
- NUNCA uses bo:votoTexto para filtrar resultados, es un literal de texto libre.
- NUNCA uses UNION en una sola consulta (no soportado): usa dos consultas separadas si es necesario,
  o usa OPTIONAL + FILTER para combinar condiciones alternativas.
- Si la pregunta pide detalles de proposiciones concretas, incluye ?titulo, ?fecha, ?anio en el SELECT.
- Si la pregunta pide evolución temporal, agrupa por ?anio y ordena por ?anio ASC.
- NUNCA añadas un filtro bo:trataSobre a menos que la pregunta mencione explícitamente un tema concreto.
  Si la pregunta es sobre TODAS las proposiciones, omite ese triple completamente.
- PARA CALCULAR RATIOS (total + subconjunto filtrado): USA SIEMPRE OPTIONAL+BIND, NUNCA FILTER.
  Un FILTER en el WHERE elimina las filas que no lo cumplen → el COUNT total queda incorrecto.
  CORRECTO:  OPTIONAL {{ ?prop bo:tieneResultado bo:Rechazada . BIND(?prop AS ?rechazada) }}
             → SELECT ... (COUNT(?prop) AS ?total) (COUNT(?rechazada) AS ?rechazadas)
  INCORRECTO: FILTER(?resultado = bo:Rechazada)  ← destruye el total y el COUNT queda a 0
- NUNCA uses LIMIT 1 si la pregunta pide COMPARAR grupos: LIMIT 1 elimina toda la comparación.
  Usa LIMIT 20 (o más) para mostrar todos los grupos relevantes en comparaciones.

EJEMPLO — proposiciones por grupo en un tema concreto:
  SELECT ?grupo ?nombreGrupo (COUNT(?prop) AS ?n)
  WHERE {{
    ?prop a bo:Proposicion ;
          bo:trataSobre br:t_vivienda ;
          bo:presentadaPor ?grupo .
    ?grupo rdfs:label ?nombreGrupo .
    FILTER(?grupo != br:grupo_desconocido)
  }}
  GROUP BY ?grupo ?nombreGrupo ORDER BY DESC(?n) LIMIT 20

EJEMPLO — evolución temporal de todas las proposiciones aprobadas por año (sin filtro de tema):
  SELECT ?anio (COUNT(?prop) AS ?n)
  WHERE {{
    ?prop a bo:Proposicion ;
          bo:tieneResultado bo:Aprobada ;
          bo:anio ?anio .
  }}
  GROUP BY ?anio ORDER BY ASC(?anio)

EJEMPLO — evolución temporal de proposiciones sobre un tema específico por año:
  SELECT ?anio (COUNT(?prop) AS ?n)
  WHERE {{
    ?prop a bo:Proposicion ;
          bo:trataSobre br:t_euskera ;
          bo:anio ?anio .
  }}
  GROUP BY ?anio ORDER BY ASC(?anio)

EJEMPLO — tasa de rechazo por grupo (ratio: rechazadas / total):
  SELECT ?grupo ?nombreGrupo (COUNT(?prop) AS ?total)
         (COUNT(?rechazada) AS ?rechazadas)
  WHERE {{
    ?prop a bo:Proposicion ;
          bo:presentadaPor ?grupo .
    ?grupo rdfs:label ?nombreGrupo .
    FILTER(?grupo != br:grupo_desconocido)
    OPTIONAL {{ ?prop bo:tieneResultado bo:Rechazada . BIND(?prop AS ?rechazada) }}
  }}
  GROUP BY ?grupo ?nombreGrupo ORDER BY DESC(?rechazadas) LIMIT 20

EJEMPLO — doble conteo (total de un tema + cuántas aprobadas):
  SELECT (COUNT(?prop) AS ?total) (COUNT(?aprobada) AS ?aprobadas)
  WHERE {{
    ?prop a bo:Proposicion ;
          bo:trataSobre br:t_medioambiente .
    OPTIONAL {{ ?prop bo:tieneResultado bo:Aprobada . BIND(?prop AS ?aprobada) }}
  }}

EJEMPLO — total presentadas y aprobadas por un grupo en un año concreto:
  SELECT ?nombreGrupo (COUNT(?prop) AS ?total) (COUNT(?aprobada) AS ?aprobadas)
  WHERE {{
    ?prop a bo:Proposicion ;
          bo:presentadaPor ?grupo ;
          bo:anio ?anio .
    ?grupo rdfs:label ?nombreGrupo .
    FILTER(?grupo = br:grupo_eh_bildu)
    FILTER(?anio = 2023)
    OPTIONAL {{ ?prop bo:tieneResultado bo:Aprobada . BIND(?prop AS ?aprobada) }}
  }}
  GROUP BY ?nombreGrupo
"""

SPARQL_PROMPT = """Eres experto en SPARQL. Genera UNA consulta SPARQL válida que responda la pregunta.
Sigue TODAS las reglas del schema. Devuelve SOLO la consulta SPARQL (con sus PREFIX), sin explicaciones ni ```.

{schema}

PREGUNTA: {pregunta}

SPARQL:"""

ANSWER_PROMPT = """Eres un analista político experto en el Ayuntamiento de Bilbao.
Basándote ÚNICAMENTE en los datos del grafo que te proporciono, genera una respuesta en español que sea:
- Narrativa y clara: no solo números, explica qué significan
- Precisa: cita las cifras exactas del grafo
- Contextual: si hay datos temporales, describe la evolución; si hay varios grupos, compáralos
- Completa: menciona los casos más destacados y cualquier patrón interesante

REGLA CRÍTICA: si los datos están vacíos ("sin resultados en el grafo"), responde honestamente que
no se encontraron datos para esa consulta. NUNCA inventes cifras hipotéticas ni pongas ejemplos
ilustrativos: cualquier cifra que no aparezca en los datos es una alucinación.

PREGUNTA: {pregunta}

DATOS DEL GRAFO:
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


def _ping_ollama(timeout: float = 3.0) -> bool:
    """True si Ollama responde en localhost:11434."""
    try:
        import httpx
        return httpx.get("http://localhost:11434/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def _get_llm(provider: str):
    """Devuelve el LLM del proveedor indicado, con caché por proveedor."""
    if provider not in _llm_cache:
        with _llm_cache_lock:
            if provider not in _llm_cache:
                if provider == "ollama":
                    from langchain_ollama import ChatOllama
                    _llm_cache[provider] = ChatOllama(model=GRAPHRAG_LLM_MODEL, temperature=0)
                    print(f"[+] GraphRAG LLM: Ollama ({GRAPHRAG_LLM_MODEL})", flush=True)
                elif provider == "groq":
                    from langchain_groq import ChatGroq
                    groq_key = os.environ.get("GROQ_API_KEY", "")
                    _llm_cache[provider] = ChatGroq(
                        model="llama-3.3-70b-versatile", temperature=0, api_key=groq_key
                    )
                    print("[+] GraphRAG LLM: Groq (llama-3.3-70b-versatile)", flush=True)
    return _llm_cache[provider]


def _llm_invoke(prompt: str) -> str:
    """Invoca el LLM con fallback Ollama → Groq. Registra errores en terminal."""
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if _ping_ollama():
        try:
            return _get_llm("ollama").invoke(prompt).content
        except Exception as e:
            print(f"\n[!] GraphRAG Ollama falló — {type(e).__name__}: {e}", flush=True)
    else:
        print("[!] GraphRAG: Ollama no disponible en localhost:11434", flush=True)

    if groq_key:
        try:
            print("[~] GraphRAG usando Groq como fallback...", flush=True)
            return _get_llm("groq").invoke(prompt).content
        except Exception as e:
            print(f"[!] GraphRAG Groq también falló — {type(e).__name__}: {e}", flush=True)

    raise RuntimeError("GraphRAG: ningún LLM disponible (Ollama y Groq fallaron)")


def _clean_sparql(txt: str) -> str:
    txt = re.sub(r"```(?:sparql)?", "", txt).strip()
    m = re.search(r"PREFIX\b", txt)
    return txt[m.start():].strip() if m else txt


def graph_answer(pregunta: str, verbose=True):
    g = _load_graph()

    # 1. Generar SPARQL
    sparql = _clean_sparql(_llm_invoke(SPARQL_PROMPT.format(schema=SCHEMA, pregunta=pregunta)))
    if verbose:
        print(f"\n[SPARQL]\n{sparql}\n")

    # 2. Ejecutar (con un reintento si hay error de sintaxis)
    try:
        rows = [{str(v): str(row[v]) for v in row.labels} for row in g.query(sparql)]
    except Exception as e:
        fix = _llm_invoke(
            f"Esta consulta SPARQL dio error: {e}\nCorrígela. Devuelve SOLO SPARQL.\n\n{sparql}"
        )
        sparql = _clean_sparql(fix)
        if verbose:
            print(f"[SPARQL corregido]\n{sparql}\n")
        rows = [{str(v): str(row[v]) for v in row.labels} for row in g.query(sparql)]

    filas = "\n".join(str(r) for r in rows[:50]) or "(sin resultados en el grafo)"
    if verbose:
        print(f"[FILAS] {len(rows)}")

    # 3. Respuesta narrativa basada solo en los datos del grafo
    answer = _llm_invoke(ANSWER_PROMPT.format(pregunta=pregunta, filas=filas))
    return {"sparql": sparql, "rows": rows, "answer": answer}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "¿Cuántas proposiciones sobre vivienda ha presentado cada grupo?"
    res = graph_answer(q)
    print("\n=== RESPUESTA ===\n" + res["answer"])
