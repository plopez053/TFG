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

# modelos y healthcheck de Ollama, del módulo de proveedores (no del pipeline vectorial)
from backend.providers import LLM_MODEL_GROQ, LLM_MODEL_GRAPHRAG, ping_ollama as _ping_ollama

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_TTL = os.path.join(HERE, "bilbao_reasoned.ttl")

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
  ?prop bo:presentadaPor   ?grupo   # quién presenta — SIEMPRE UN SOLO grupo por
      # proposición (una "proposición conjunta EH BILDU y ELKARREKIN" se guarda con
      # un único grupo). NO se puede contar "proposiciones conjuntas" ni props con
      # varios grupos: si la pregunta lo pide, responde que el grafo no lo distingue.
  ?prop bo:enPleno         ?pleno   # en qué pleno
  ?prop bo:tieneResultado  ?res     # resultado: usar con los individuos de abajo
  ?prop bo:trataSobre      ?tema    # tema directo (canónico o subtema libre)
  ?prop bo:trataTemaAmplio ?tema    # tema + subtemas inferidos por OWL-RL (roll-up)
  ?prop bo:menciona        ?ent     # entidades mencionadas (empresas, personas, lugares)
  ?prop bo:intervino       ?concejal    # concejal/a que intervino en el debate
  ?prop bo:proponePersona  ?concejal    # concejal/a que FIRMA la proposición
  ?concejal a bo:Concejal ; rdfs:label ?nombreConcejal ; bo:perteneceA ?grupo .
  ?concejal bo:esAlcalde true .   # el/la alcalde/sa es un concejal marcado así
  ?tec a bo:PersonalTecnico ; rdfs:label ?n ; bo:cargoTecnico ?cargo .  # Secretario/Interventor (no votan)
  ?prop bo:votoAFavorDe    ?grupo   # GRUPO que votó a favor (derivado del voto nominal)
  ?prop bo:votoEnContraDe  ?grupo   # GRUPO que votó en contra
  ?prop bo:seAbstuvo       ?grupo   # GRUPO que se abstuvo
  ?prop bo:concejalVotoAFavor      ?concejal   # CONCEJAL concreto que votó a favor (voto NOMINAL)
  ?prop bo:concejalVotoEnContra    ?concejal   # CONCEJAL que votó en contra
  ?prop bo:concejalVotoAbstencion  ?concejal   # CONCEJAL que se abstuvo
  ?prop bo:votoFuente      ?f       # "acta" = voto parseado literal del acta (fiable, ~1400 props);
                                    # "llm" = inferido (fallback, ~120). Filtra por "acta" si quieres solo lo seguro.
  ?prop bo:tieneEnmienda   ?enm . ?enm bo:enmiendaPor ?grupo ; bo:resumenEnmienda ?txt .
  ?prop bo:resumen ?r ; bo:pagina ?pag ; bo:fuentePdf ?pdf    # resumen dispositivo, pág. del PDF

Para "¿qué dijo / en qué intervino el concejal X?": ?prop bo:intervino ?c .
  ?c rdfs:label ?n . FILTER(CONTAINS(LCASE(STR(?n)), "apellido")). Para "concejal más
  activo": GROUP BY ?c ?n ORDER BY DESC(COUNT(?prop)). Para "cómo votó el grupo X en
  una proposición": mira bo:votoAFavorDe / bo:votoEnContraDe / bo:seAbstuvo (solo están
  en las proposiciones cuyo acta desglosa el voto; si no aparece, no se sabe).

Para "¿qué se ha dicho / debatido sobre <NOMBRE>?" cuando <NOMBRE> es una empresa,
persona o lugar (Iberdrola, Zorrotzaurre, una persona...), NO es un tema: usa
  ?prop bo:menciona ?ent . ?ent rdfs:label ?nombreEnt .
  FILTER(REGEX(STR(?nombreEnt), "\\\\bzara\\\\b", "i"))
IMPORTANTE: para el nombre usa REGEX con \\\\b...\\\\b (límite de palabra), NO
CONTAINS: "zara" con CONTAINS también casa "Zaragoza", "Zarautz", "Zarandoa".
Con varios nombres: FILTER(REGEX(...,"\\\\bmercadona\\\\b","i") || REGEX(...,"\\\\bzara\\\\b","i")).
Devuelve ?titulo ?fecha ?nombreEnt. OJO: el grafo solo tiene las entidades que el
enriquecimiento consideró relevantes; muchas menciones de pasada NO están (eso lo
cubre mejor el RAG vectorial). Si sale 0, la respuesta honesta es que el grafo no
recoge esa entidad, no que no se haya hablado del tema.

INDIVIDUOS de resultado (usar con bo:tieneResultado, NUNCA con bo:votoTexto):
  bo:Aprobada  bo:Rechazada  bo:Decae  bo:Retirada  bo:AprobadaConEnmienda  bo:SinResultado
  IMPORTANTE: "aprobada" en lenguaje normal = bo:Aprobada Y bo:AprobadaConEnmienda
  (una proposición aprobada con enmienda SÍ salió adelante, con cambios). Cuando la
  pregunta diga "cuántas se aprobaron" / "aprobadas" sin más matiz, cuenta LAS DOS:
    - conteo simple:  ?prop ... ; bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda))
    - en un ratio (total + aprobadas juntos):
        OPTIONAL {{ ?prop bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) . BIND(?prop AS ?aprobada) }}
  Usa solo bo:Aprobada si la pregunta pide explícitamente "sin enmiendas" / "tal cual".

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

URIs EXACTAS de los 19 TEMAS CANÓNICOS (usar cuando filtres por tema concreto):
  br:t_vivienda          → vivienda
  br:t_urbanismo         → urbanismo
  br:t_movilidad         → movilidad y transporte
  br:t_medioambiente     → medio ambiente
  br:t_euskera           → euskera
  br:t_cultura           → cultura
  br:t_deporte           → deporte
  br:t_educacion         → educación
  br:t_igualdad          → igualdad y feminismo
  br:t_serviciossociales → servicios sociales
  br:t_empleoeconomia    → empleo y economía
  br:t_presupuestos      → presupuestos y fiscalidad
  br:t_seguridad         → seguridad
  br:t_participacion     → participación ciudadana
  br:t_turismo           → turismo
  br:t_sanidad           → sanidad
  br:t_memoriahistorica  → memoria histórica
  br:t_derechoshumanos   → derechos humanos
  br:t_otros             → otros (proposiciones sin tema claro en ninguna otra categoría)

URIs EXACTAS de SUBTEMAS específicos (nivel 2, ⊑ de un tema canónico de arriba):
úsalas cuando la pregunta mencione algo más concreto que el tema general —
CON bo:trataTemaAmplio devuelven también todo lo que cuelgue del subtema, y
de paso el roll-up incluye automáticamente el tema padre. Evita el fallback de
CONTAINS + skos:prefLabel salvo que el concepto no esté en esta lista: ese
fallback solo mira prefLabel (no altLabel/sinónimos) y solo el texto literal
que tú elijas, así que suele perderse variantes, plurales y sinónimos.
  -- vivienda: br:t_alquiler, br:t_desahucios, br:t_vivienda_social, br:t_vivienda_vacia
  -- movilidad y transporte: br:t_aparcamiento, br:t_bicicleta, br:t_bilbobus,
     br:t_metro_bilbao, br:t_transporte_publico, br:t_tranvia, br:t_accesibilidad
  -- empleo y economía: br:t_comercio, br:t_empleo, br:t_empresas, br:t_hosteleria
  -- presupuestos y fiscalidad: br:t_financiacion, br:t_ordenanzas_fiscales,
     br:t_presupuesto_municipal, br:t_retribuciones, br:t_subvenciones
  -- urbanismo: br:t_barrios, br:t_espacio_publico, br:t_ordenacion_urbana, br:t_rehabilitacion
  -- servicios sociales: br:t_discapacidad, br:t_exclusion_social, br:t_personas_mayores
  -- seguridad: br:t_policia_municipal
  -- medio ambiente: br:t_reciclaje
  -- igualdad y feminismo: br:t_violencia_genero
  -- cultura: br:t_arte, br:t_fiestas, br:t_museos
  -- participación ciudadana: br:t_transparencia

CUÁNDO usar trataSobre vs trataTemaAmplio — USA trataTemaAmplio POR DEFECTO:
- bo:trataTemaAmplio: el tema Y todos sus subtemas inferidos por el razonador OWL-RL
  (p.ej. "desahucios", "alquiler", "vivienda social" cuentan como "vivienda"; "subvenciones",
  "IBI", "ordenanzas fiscales" cuentan como "presupuestos y fiscalidad"). ÚSALO SIEMPRE QUE LA
  PREGUNTA MENCIONE UN TEMA DE FORMA GENERAL ("¿cuántas propuestas de vivienda?", "¿qué grupo
  presentó más sobre presupuestos?", "temas más frecuentes de EH Bildu") — es la respuesta que
  espera un usuario real, que casi nunca pedirá subtemas explícitamente aunque los quiera
  incluidos. Usa SIEMPRE la URI exacta del tema canónico: ?prop bo:trataTemaAmplio br:t_vivienda
  Si la pregunta dice literalmente "incluyendo subtemas" / "con subtemas" / "en total": ES
  bo:trataTemaAmplio con la URI del tema, sin más. NUNCA recorras skos:broader tú mismo
  (no tiene cierre transitivo materializado) ni inventes URIs tipo br:t_presupuestos_fiscalidad
  (el tema es br:t_presupuestos; la lista completa de URIs está arriba).
- bo:trataSobre: SOLO el tema principal o un tema secundario EXACTO del texto de la proposición,
  SIN incluir subtemas inferidos. Úsalo ÚNICAMENTE si la pregunta pide explícitamente excluir
  subtemas ("solo lo etiquetado directamente como X, no sus variantes") — muy raro en la práctica.
  Si no sabes la URI exacta puedes usar FILTER: ?prop bo:trataSobre ?tema . ?tema skos:prefLabel ?lab .
  FILTER(CONTAINS(LCASE(STR(?lab)), "vivienda"))

REGLAS obligatorias:
- NUNCA uses el patrón de nodo anónimo [skos:prefLabel "vivienda"]: usa URI exacta o variable+FILTER.
    CORRECTO:  ?prop bo:trataSobre br:t_vivienda
    CORRECTO:  ?prop bo:trataSobre ?tema . ?tema skos:prefLabel ?lab . FILTER(CONTAINS(LCASE(STR(?lab)), "vivienda"))
    INCORRECTO: ?prop bo:trataSobre [skos:prefLabel "vivienda"]
- Para filtrar por resultado: ?prop bo:tieneResultado bo:Rechazada
  Si SOLO quieres CONTAR las de un resultado (p.ej. "cuántas rechazadas de euskera"),
  pon el triple bo:tieneResultado DIRECTO en el WHERE, NUNCA dentro de un OPTIONAL:
  un OPTIONAL cuyas variables no se usan luego en un COUNT no filtra nada y el
  conteo saldrá con TODAS (bug real: "euskera rechazadas" devolvió 55 en vez de 3).
  CORRECTO:   SELECT (COUNT(?prop) AS ?n) WHERE {{ ?prop ...tema... ; bo:tieneResultado bo:Rechazada }}
  INCORRECTO: SELECT (COUNT(?prop) AS ?n) WHERE {{ ?prop ...tema... . OPTIONAL {{ ?prop bo:tieneResultado bo:Rechazada }} }}
  (el OPTIONAL+BIND solo es para RATIOS: total + subconjunto en la MISMA consulta, ver abajo)
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
- Para rankings (desglose por grupo/año/tema) usa COUNT + GROUP BY + ORDER BY DESC + LIMIT 50.
- Para un TOTAL simple (sin desglose por nada) NUNCA agrupes por la variable que estás
  contando: GROUP BY ?prop convierte cada proposición en su propio grupo de tamaño 1
  y el COUNT dejará de servir para nada, devolviendo cientos de filas con "1" en vez
  de un único total.
  CORRECTO:   SELECT (COUNT(?prop) AS ?total) WHERE {{ ?prop a bo:Proposicion ; ... }}
              (sin GROUP BY — una sola fila con el total)
- Para PORCENTAJES ("qué porcentaje de X se aprobó", tasas, ratios en %): NUNCA calcules
  la división/porcentaje dentro del SPARQL (no inventes bloques "WITH", subconsultas
  anidadas complejas ni operadores que no conozcas bien) — limítate a devolver el total y
  el subconjunto con OPTIONAL+BIND (ver ejemplo de ratios más abajo). El porcentaje se
  calcula después, en la respuesta narrativa, a partir de esos dos números.
  INCORRECTO: SELECT (COUNT(?prop) AS ?total) WHERE {{ ... }} GROUP BY ?prop
- EXCLUYE br:grupo_desconocido de los rankings a menos que se pida explícitamente.
- NUNCA uses bo:votoTexto para filtrar resultados, es un literal de texto libre.
- EVITA UNION: complica la consulta y suele generarse mal. Usa OPTIONAL + FILTER
  para combinar condiciones alternativas, o dos consultas separadas si es necesario.
- Si la pregunta pide detalles de proposiciones concretas, incluye ?titulo, ?fecha, ?anio en el SELECT.
- Si la pregunta pide evolución temporal, agrupa por ?anio y ordena por ?anio ASC.
- NUNCA añadas un filtro bo:trataTemaAmplio/bo:trataSobre a menos que la pregunta mencione
  explícitamente un tema concreto. Si la pregunta es sobre TODAS las proposiciones, omite ese
  triple completamente.
- PARA CALCULAR RATIOS (total + subconjunto filtrado): USA SIEMPRE OPTIONAL+BIND, NUNCA FILTER.
  Un FILTER en el WHERE elimina las filas que no lo cumplen → el COUNT total queda incorrecto.
  CORRECTO:  OPTIONAL {{ ?prop bo:tieneResultado bo:Rechazada . BIND(?prop AS ?rechazada) }}
             → SELECT ... (COUNT(?prop) AS ?total) (COUNT(?rechazada) AS ?rechazadas)
  INCORRECTO: FILTER(?resultado = bo:Rechazada)  ← destruye el total y el COUNT queda a 0
- NUNCA uses LIMIT 1 si la pregunta pide COMPARAR grupos: LIMIT 1 elimina toda la comparación.
  Usa LIMIT 20 (o más) para mostrar todos los grupos relevantes en comparaciones.
- Si agrupas POR TEMA (ranking de temas, "el tema más/menos tratado"), incluye SIEMPRE
  ?tema Y su ?labelTema en el SELECT — no solo el COUNT, o la respuesta no podrá decir DE
  QUÉ tema se trata.
  CORRECTO: SELECT ?tema ?labelTema (COUNT(?prop) AS ?n) WHERE {{ ?prop bo:trataSobre ?tema .
            ?tema skos:prefLabel ?labelTema . }} GROUP BY ?tema ?labelTema ORDER BY ASC(?n) LIMIT 5

EJEMPLO — proposiciones por grupo en un tema concreto (pregunta general → trataTemaAmplio):
  SELECT ?grupo ?nombreGrupo (COUNT(?prop) AS ?n)
  WHERE {{
    ?prop a bo:Proposicion ;
          bo:trataTemaAmplio br:t_vivienda ;
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
          bo:trataTemaAmplio br:t_euskera ;
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
          bo:trataTemaAmplio br:t_medioambiente .
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

EJEMPLO — CONCEJAL/A (una PERSONA, no un grupo) que ha firmado más proposiciones.
OJO: si la pregunta dice "concejal", "concejala", "quién" o un nombre de persona,
NO uses bo:presentadaPor (eso es el GRUPO); usa bo:proponePersona o bo:intervino,
que apuntan a nodos ?c con `?c a bo:Concejal ; rdfs:label ?nombre`:
  SELECT ?nombre (COUNT(DISTINCT ?prop) AS ?n)
  WHERE {{
    ?prop bo:proponePersona ?c .
    ?c rdfs:label ?nombre .
  }}
  GROUP BY ?c ?nombre ORDER BY DESC(?n) LIMIT 20

EJEMPLO — en cuántos debates ha INTERVENIDO un/a concejal/a concreto/a (Otxandiano,
Aburto, Goirizelaia...). El nombre SIEMPRE con FILTER(CONTAINS(LCASE(...))), nunca "=":
  SELECT (COUNT(DISTINCT ?prop) AS ?nDebates)
  WHERE {{
    ?prop bo:intervino ?c .
    ?c rdfs:label ?nombre .
    FILTER(CONTAINS(LCASE(STR(?nombre)), "otxandiano"))
  }}

EJEMPLO — quién ha sido alcalde/sa (son concejales con bo:esAlcalde true; hay 3:
Azkuna, Areso, Aburto — el actual es Aburto):
  SELECT ?nombre WHERE {{ ?c bo:esAlcalde true ; rdfs:label ?nombre . }}

EJEMPLO — cuántas ENMIENDAS ha presentado un grupo (nodos bo:Enmienda, NO proposiciones):
  SELECT (COUNT(DISTINCT ?enm) AS ?nEnmiendas)
  WHERE {{ ?enm a bo:Enmienda ; bo:enmiendaPor br:grupo_pp . }}

VOTO por grupo (bo:votoAFavorDe / bo:votoEnContraDe / bo:seAbstuvo) — existe en
~1.400 proposiciones (las que traen la lista nominal en el acta). Ej. "¿cómo votó
EH Bildu las proposiciones de vivienda?":
  SELECT (COUNT(DISTINCT ?pf) AS ?aFavor) (COUNT(DISTINCT ?pc) AS ?enContra) (COUNT(DISTINCT ?pa) AS ?abst)
  WHERE {{
    ?prop bo:trataTemaAmplio br:t_vivienda .
    OPTIONAL {{ ?prop bo:votoAFavorDe   br:grupo_eh_bildu . BIND(?prop AS ?pf) }}
    OPTIONAL {{ ?prop bo:votoEnContraDe br:grupo_eh_bildu . BIND(?prop AS ?pc) }}
    OPTIONAL {{ ?prop bo:seAbstuvo      br:grupo_eh_bildu . BIND(?prop AS ?pa) }}
  }}

VOTO NOMINAL (por concejal) — ej. "¿qué concejal ha votado más veces en contra?":
  SELECT ?n (COUNT(*) AS ?veces) WHERE {{ ?prop bo:concejalVotoEnContra ?c . ?c rdfs:label ?n }}
  GROUP BY ?c ?n ORDER BY DESC(?veces) LIMIT 20
Ej. "¿cómo votó [concejal] la proposición sobre X?": ?prop bo:trataTemaAmplio br:t_X .
  {{ ?prop bo:concejalVotoAFavor ?c }} UNION {{ ?prop bo:concejalVotoEnContra ?c }} UNION {{ ?prop bo:concejalVotoAbstencion ?c }}
  ?c rdfs:label ?n . FILTER(CONTAINS(LCASE(STR(?n)), "apellido"))
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
                    _llm_cache[provider] = ChatOllama(model=LLM_MODEL_GRAPHRAG, temperature=0,
                                                     client_kwargs={"timeout": 300})
                    print(f"[+] GraphRAG LLM: Ollama ({LLM_MODEL_GRAPHRAG})", flush=True)
                elif provider == "groq":
                    from langchain_groq import ChatGroq
                    groq_key = os.environ.get("GROQ_API_KEY", "")
                    _llm_cache[provider] = ChatGroq(
                        model=LLM_MODEL_GROQ, temperature=0, api_key=groq_key
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


# mapea un slug de tema inventado por el LLM (br:t_presupuestos_fiscalidad...) al canónico real
def _fix_tema_uri(uri_slug: str) -> str:
    s = uri_slug.replace("-", "").replace("_", "")
    for canon in _CANON_TEMAS:
        if s.startswith(canon) or canon.startswith(s) or canon in s:
            return f"br:t_{canon}"
    return f"br:t_{uri_slug}"


# reescribe anti-patrones del LLM que dan resultado equivocado sin error (lista en memoria/decisiones_tecnicas.md 3.1)
def _sanitize_sparql(sparql: str, verbose: bool = False, pregunta: str = "") -> str:
    original = sparql

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
        sparql = re.sub(
            r"bo:tieneResultado\s+bo:Aprobada\b(\s*[.}\n])",
            r"bo:tieneResultado ?_rApr . FILTER(?_rApr IN (bo:Aprobada, bo:AprobadaConEnmienda))\1",
            sparql)

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


def graph_answer(pregunta: str, verbose=True):
    g = _load_graph()

    # 1. Generar SPARQL
    sparql = _sanitize_sparql(
        _clean_sparql(_llm_invoke(SPARQL_PROMPT.format(schema=SCHEMA, pregunta=pregunta))), verbose, pregunta)
    if verbose:
        print(f"\n[SPARQL]\n{sparql}\n")

    # 2. Ejecutar (hasta 2 reintentos si hay error de sintaxis). El prompt de
    #    corrección REPITE el schema entero: "corrígela" a secas hacía que el
    #    modelo inventara vocabulario (PREFIX example.org, WITH, subconsultas)
    #    porque perdía el contexto de qué existe en el grafo.
    for intento in range(3):
        try:
            rows = [{str(v): str(row[v]) for v in row.labels} for row in g.query(sparql)]
            break
        except Exception as e:
            if intento == 2:
                # Ninguna consulta válida en 3 intentos: la pregunta cae fuera de
                # lo que el modelo sabe expresar en SPARQL sobre este grafo (p.ej.
                # "proposiciones conjuntas" — el grafo solo guarda un grupo por
                # proposición). Degradar con gracia en vez de romper el chat.
                if verbose:
                    print(f"[!] SPARQL irrecuperable tras 3 intentos: {e}", flush=True)
                return {
                    "sparql": sparql,
                    "rows": [],
                    "answer": ("No he podido traducir esa pregunta a una consulta válida sobre el "
                               "grafo. Puede que pida un dato que el grafo no distingue (por ejemplo, "
                               "proposiciones presentadas conjuntamente por varios grupos: el grafo "
                               "guarda un único grupo por proposición). Prueba a reformularla de forma "
                               "más concreta."),
                }
            fix = _llm_invoke(
                f"{SCHEMA}\n\nPREGUNTA: {pregunta}\n\n"
                f"Esta consulta SPARQL DIO ERROR: {e}\n"
                f"Consulta con error:\n{sparql}\n\n"
                "Genera una consulta NUEVA que responda la pregunta, más simple, usando "
                "SOLO el vocabulario del schema de arriba (NO inventes PREFIX, WITH, "
                "subconsultas ni URIs). Para porcentajes/ratios devuelve solo el total y "
                "el subconjunto con OPTIONAL+BIND. Devuelve SOLO la consulta SPARQL."
            )
            sparql = _sanitize_sparql(_clean_sparql(fix), verbose, pregunta)
            if verbose:
                print(f"[SPARQL corregido #{intento + 1}]\n{sparql}\n")

    rows = _fix_degenerate_groupby(rows, sparql)
    filas = "\n".join(str(r) for r in rows[:50]) or "(sin resultados en el grafo)"
    filas += _augment_ratios(rows, pregunta)
    if verbose:
        print(f"[FILAS] {len(rows)}")

    # 3. Respuesta narrativa basada solo en los datos del grafo
    answer = _llm_invoke(ANSWER_PROMPT.format(pregunta=pregunta, filas=filas, sparql=sparql), prefer="groq")
    return {"sparql": sparql, "rows": rows, "answer": answer}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "¿Cuántas proposiciones sobre vivienda ha presentado cada grupo?"
    res = graph_answer(q)
    print("\n=== RESPUESTA ===\n" + res["answer"])
