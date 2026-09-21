# Batería de verificación — Ronda 42 (2026-09-17)

Objetivo explícito del usuario: (1) arreglar, si era posible, el bug de
"segmentación" que quedaba documentado como límite conocido (82/3422
proposiciones, 2,4%, con clasificación de resultado dudosa); (2) lanzar una
batería de verificación **comparando GraphRAG y el RAG vectorial sobre las
mismas preguntas**, verificando siempre la información contra el PDF real, y
dando **mucho peso a la estructura de la respuesta**, no solo al contenido.

## Parte 1 — El bug de "segmentación": causa real distinta de lo que se creía

La Ronda 40 (2026-09-16) había diagnosticado la causa como "el párrafo
decisivo se atribuye al TEMA SIGUIENTE por un corte de página" y la había
dejado como límite conocido, fuera de alcance por el riesgo de tocar el
indexador compartido por GraphRAG y el RAG vectorial.

Investigado de nuevo con `scratchpad/scan_segmentacion_v2.py` (heurístico que
compara la clasificación actual de cada proposición contra lo que implica su
propio texto completo) y reproduciendo `_process_single_pdf` en aislado
contra 2 casos reales (`25-11-2010_Ordinaria_Acta.pdf`, puntos 17 y 19):
**la hipótesis del corte de página era incorrecta**. La causa real era mucho
más simple: `result_re` (el regex que extrae la frase de resultado en
`backend/rag.py`) **nunca reconocía la palabra "decae" por sí sola**, y
además descartaba el texto de resultado capturado si no traía cifras de voto
ni decía literalmente "unanimidad"/"asentimiento" — perdiendo desenlaces
reales y frecuentes como *"El Pleno municipal acepta la Enmienda..., sin
necesidad de someterla a votación, por lo que decae la proposición..."*.

**Fix**: rama `decaen?\b[^.]{0,400}` añadida a `result_re` + nueva
`_STRONG_RESULT_RE` que ya no exige cifras/unanimidad para conservar un
resultado_text que venga de un marcador formal de cierre de acta
(queda/resulta/decae). Verificado en aislado contra 9 casos reales del PDF
(25-11-2010, 28-11-2007, 18-06-2008, 27-11-2008, 25-02-2010, 27-05-2010) —
los 9 ahora capturan "decae" correctamente.

**Impacto medido**: 993→1179 proposiciones con "decae" detectado en bruto
(+186); el heurístico de discrepancias bajó de **82 a 25/3422 (-70%)**. Los
25 restantes SÍ son la causa más profunda ya diagnosticada (desalineación
segmento↔chunk cuando el debate cruza una página, verificada de nuevo con un
caso nuevo: 28-03-2012, puntos 14/15) — sigue sin arreglar, mismo riesgo que
antes de tocar el indexador compartido.

Pipeline completo relanzado (re-extracción sin LLM, merge en
`proposals_enriched.jsonl`, grafo reconstruido, metadatos del RAG vectorial
resincronizados). 2 baselines de `regression_qa.py`/`gold.py` recalibrados
y verificados con SPARQL independiente: EH Bildu vivienda aprobadas 5→3,
tasa aprobación EH Bildu 71→49 / PP 148→128 (totales sin cambios).

## Parte 2 — Batería a fondo: GraphRAG vs. vectorial, con énfasis en estructura

**Metodología**: para GraphRAG, `graph_answer()` + `graph_sources()`
directos (igual que en producción). Para el RAG vectorial, en vez de usar el
CLI simplificado (`rag.query()`), se construyó un arnés que **replica
literalmente el pipeline real de `frontend/app.py::on_message`** —
`dedup_answer_blocks`, `strip_empty_blocks`, anclaje de fuentes por
título/fecha, corrección de la línea de Votos, "Otras fuentes", red de
seguridad — llamando a las mismas funciones de `backend/rag.py` que usa la
aplicación de verdad, para poder inspeccionar la estructura EXACTA que vería
un usuario real, no una aproximación simplificada.

| # | Pregunta | Sistemas | Resultado |
|---|----------|----------|-----------|
| Q1 | Proposición del PP sobre el Centro Municipal de Distrito Abando-Indautxu (25-11-2010) | GraphRAG + Vectorial | Ambos coinciden: **Decae**. Verificado contra el PDF real (pág. 117: *"En su virtud, decae la proposición formulada por el Grupo Municipal P.P."*). Encontrados y arreglados 3 bugs reales por el camino (ver abajo). |
| Q2 | Proposición sobre el "buzón del ciudadano" (PSE-EE, misma fecha) | GraphRAG + Vectorial | Ambos coinciden: **Decae**, con cita de página real (100). Consistente tras los 3 fixes. |
| Q3 | ¿Se ha mencionado a Tubacex en algún pleno? | GraphRAG + Vectorial | **Los dos sistemas coinciden en el hecho de fondo por mecanismos distintos**: GraphRAG cuenta 1 mención vía `bo:menciona` (entidad extraída por LLM); el vectorial encuentra la mención de pasada (27-05-2021, dentro de una proposición ajena) y concluye correctamente que no es tema central de ninguna proposición. Verificado contra el PDF real (pág. 76). |
| Q4 | Vivienda: total y aprobadas del PP | GraphRAG | 47 presentadas, 5 aprobadas (10,6%) — consulta SPARQL estándar ya validada en otros casos, sin incidencias. |
| Q5 | ¿Qué ocurrió en el pleno del 26-01-2023? (9 propuestas) | Vectorial | Caso de estrés de estructura: sesión única con 9 propuestas distintas. Tras el fix de posiciones desfasadas (bug 3), **las 9 fuentes se insertan limpiamente, sin partir ninguna línea**. Emparejamiento título↔fuente no siempre exacto en algún caso puntual (limitación ya conocida desde la Ronda 36, no nueva). |
| Q6 | ¿Cuántas proposiciones han decaído en 2019? | GraphRAG | **Bug real encontrado (bug 4, ver abajo)**: la primera versión ignoraba el filtro de resultado por completo. |
| Q7 | ¿Qué se ha debatido sobre el tranvía en Bilbao? | Vectorial (multi-sesión) | Estructura limpia tras el fix 3, 4 sesiones citadas + "Otras fuentes", sin bloques partidos ni duplicados. |

### Bug 1 — SPARQL con `bo:fecha "..."^^xsd:date` hacía crashear la pregunta entera

El grafo guarda `bo:fecha` como string plano `"DD-MM-AAAA"` (sin tipo). Si el
LLM generador añadía `^^xsd:date`, rdflib intentaba parsear el literal como
fecha ISO al construir la consulta — `"25-11-2010"` no es ISO
(`AAAA-MM-DD`) → `ValueError`, la pregunta fallaba entera sin llegar siquiera
a ejecutarse contra el grafo. **Fix**: nueva guarda en `_sanitize_sparql()`
que quita el tipo de cualquier literal con forma DD-MM-AAAA. Verificado en
aislado y en vivo.

### Bug 2 — El fix de "decae" no llegaba al RAG vectorial: `vote_result` desactualizado en ChromaDB

Comparando la misma pregunta (Q1) en los dos sistemas, GraphRAG mostraba
"Decae" correctamente pero el vectorial mostraba solo cifras de voto en
bruto. Causa: `scripts/enrich_vector_metadata.py` (Ronda 14) parchea campos
DERIVADOS (`resultado`, `votos_favor`, `votos_contra`...) desde
`proposals_enriched.jsonl`, pero nunca sobrescribía el campo `vote_result`
ORIGINAL de cada chunk — el que `build_sources_data`/`replace_votos_line`
(Ronda 36) leen literalmente para sustituir la línea de Votos del LLM. Ese
campo se calculó en la indexación original de ChromaDB y no se actualiza
salvo un reindexado completo (horas). **Fix**: el script ahora también
sobrescribe `vote_result` con el valor ya corregido (sin pérdida de
precisión: todos los chunks de un mismo segmento comparten idénticamente el
mismo valor). Aplicado sobre 97.150 chunks, 100% cobertura, 0 misses.

### Bug 3 — Posiciones desfasadas al insertar la fuente: partía la línea de Votos por la mitad

Arreglado el bug 2, apareció uno más estructural: el enlace de fuente se
insertaba en una posición equivocada, **partiendo la línea de Votos ya
corregida por la mitad** (ej. `"...formulada por el Grupo Municipal P (Votos"
+ [FUENTE]/[RESULTADO] + "emitidos: 29 | ...)"`). Causa: el código hacía DOS
pasadas separadas sobre el texto — primero corregía todas las líneas de
Votos (cambiando su longitud), y DESPUÉS, en un bucle aparte, insertaba las
fuentes reusando posiciones calculadas ANTES de esa corrección. Como el
`vote_result` real casi nunca mide lo mismo que la línea que escribió el
LLM, la posición quedaba desfasada. Este patrón llevaba así desde la Ronda 36
sin detectarse porque ningún caso de prueba anterior tenía una diferencia de
longitud grande en el ÚLTIMO bloque del texto.

**Fix**: las dos pasadas se fusionaron en una sola, por bloque, procesando en
orden DESCENDENTE de posición (se demuestra seguro: procesando de derecha a
izquierda, ninguna edición afecta a los índices de bloques aún no
procesados, que quedan siempre a su izquierda). Aplicado a las dos ramas de
`frontend/app.py::on_message` — la de sesión única con varias propuestas Y
la de fecha (que cubre sesión única de UNA propuesta + multi-sesión, la más
común en preguntas reales). Verificado con las 5 preguntas de la batería
usando el arnés fiel a producción: 0 líneas partidas, incluido el caso más
exigente (Q5, 9 bloques de fuente todos limpios).

### Bug 4 — GraphRAG: "¿Cuántas proposiciones han decaído en 2019?" ignoraba el filtro de resultado

La consulta generada fue `SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE { ?p a
bo:Proposicion ; bo:anio 2019 . }` — sin ningún filtro de resultado, contando
las 111 proposiciones totales de 2019 en vez de solo las decaídas. Ejecutó
sin error y con filas → se aceptó tal cual, sin reintento: la clase de fallo
más peligrosa, una respuesta segura pero equivocada. Causa: ningún ejemplo
del banco de few-shot usaba `bo:Decae` ni la palabra "decaído", así que por
similitud la pregunta se emparejaba con un ejemplo de conteo simple sin
filtro. **Fix**: nuevo ejemplo few-shot con `bo:tieneResultado bo:Decae`.
**Matiz honesto**: probado tras el fix con Groq agotado (cuota diaria
consumida por esta misma batería) y cayendo a Ollama qwen3:8b, el modelo
local siguió generando la consulta sin el filtro pese al ejemplo casi
idéntico — coherente con limitaciones ya documentadas del modelo local
ignorando incluso ejemplos muy cercanos. El fix mejora la situación (ahora
hay señal donde antes no había ninguna) pero no está confirmado con Groq
(el modelo preferido en producción) por cuota agotada — pendiente de
re-verificar.

### Otros hallazgos, documentados sin arreglar

- **Precisión de página dentro de un debate largo**: la página citada para
  el resultado de una proposición (calculada sobre el último chunk con
  `vote_result`) no siempre es la página exacta del párrafo decisivo cuando
  el debate ocupa varias páginas (Q1: se cita pág. 112, el párrafo decisivo
  real está en la 117, mismo acta). No es un dato falso — la página citada sí
  pertenece a esa proposición — es una imprecisión de localización dentro de
  un rango. No arreglado, coste/beneficio bajo.
- **Emparejamiento fuente↔título imperfecto con muchas propuestas**:
  confirmado de nuevo en Q5 (9 propuestas) — limitación ya conocida y
  aceptada desde la Ronda 36 (nunca 100% de acierto), no se intentó mejorar
  más aquí.
- **El caso de "25/3422" de desalineación segmento↔chunk** sigue sin
  arreglar, documentado en la Parte 1.

## Conclusión

Comparar los dos sistemas sobre las mismas preguntas, en vez de probarlos por
separado, sacó a la luz 3 bugs reales (2 y 3) que las baterías anteriores —
centradas en un solo sistema a la vez — no habían detectado: el fix de un
bug de datos (Ronda 42, parte 1) se había aplicado correctamente al grafo
pero no había llegado al RAG vectorial por una desactualización de caché de
metadatos (bug 2), y un bug de estructura preexistente desde la Ronda 36
solo se manifestaba cuando el texto sustituido medía sensiblemente distinto
del original (bug 3) — nunca detectado porque ningún caso de prueba anterior
forzaba esa condición. El énfasis en la estructura de la respuesta, no solo
en el contenido, fue lo que permitió encontrar el bug 3.

**Regresión completa final** (tras todos los fixes de código de esta ronda,
partes 1 y 2 combinadas; el fix de `frontend/app.py` no lo ejercita esta
suite — no llama a `on_message`, ver Ronda 36): **GraphRAG 22/22, vectorial
13/13 — 35/35**, sin regresiones. Arm G: 28/35 (80%), 0 wrong.

**Nota de cuota**: la cuenta de Groq (`openai/gpt-oss-120b`) agotó su cupo
diario (200.000 TPD) hacia la mitad de esta batería — todas las pruebas
posteriores cayeron con gracia a Ollama (qwen3:8b/qwen2.5:7b), confirmando
una vez más que el mecanismo de fallback funciona, aunque impidió confirmar
el bug 4 con el modelo preferido de producción.
