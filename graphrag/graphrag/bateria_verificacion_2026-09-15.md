# Batería de verificación GraphRAG vs RAG vectorial — 2026-09-15

_Batería nueva, con preguntas no repetidas de rondas anteriores: enmiendas por grupo, intervenciones de concejales, un grupo minoritario real (VOX), argumentos cualitativos (Elkarrekin y pisos turísticos), recuento acumulado de un mandato, evolución temporal y una entidad de mención rara (BBVA). Verificación de cifras del grafo contra consultas SPARQL independientes escritas a mano, y de las respuestas vectoriales contra el texto real de los PDF (pypdf, página exacta)._

_GraphRAG: generación SPARQL con Ollama qwen3:8b local, narración con Groq openai/gpt-oss-120b (fallback Ollama). RAG vectorial: recuperación híbrida (semántica + literal + temática) + rerank Cohere, narración Groq/Ollama. **Límites de API tenidos en cuenta:** las 14 llamadas de narración de esta batería comparten la misma cuota diaria de Groq (`openai/gpt-oss-120b`, 200k tokens/día) entre los dos sistemas, así que se lanzaron con pausas de 20-25s entre preguntas (respetando también el límite de 8000 tok/min) y las dos mitades (grafo y vectorial) se ejecutaron en procesos separados, no simultáneos, para no competir por la GPU local (Ollama sirve a la vez qwen3:8b para SPARQL y nomic-embed-text para los embeddings del vectorial). Cohere (plan trial, 10 llamadas/min) solo lo usa el lado vectorial, un rerank por pregunta._

---

## 0. Bugs reales encontrados y corregidos durante esta batería

Antes de las 7 preguntas, la primera consulta (enmiendas) reveló un problema de rendimiento grave: **32 minutos y 3 intentos en blanco** en vez de una respuesta o un fallo rápido. Investigado a fondo y corregido en `graph_rag_sparql.py`, con verificación en vivo de cada fix:

1. **`reasoning=False` + `num_ctx=8192` que faltaban en el cliente Ollama de GraphRAG.** El estudio de ablación de este mismo proyecto (`ESTUDIO_SPARQL_LOCAL.md`, Fase 6) ya había demostrado que qwen3:8b con el modo "pensamiento" desactivado da mejor precisión (94-97%) y mucha más velocidad para generar SPARQL — pero esa configuración nunca se trasladó a producción. Sin ella, una pregunta con agregación más difícil (enmiendas por grupo, un GROUP BY de dos saltos) hizo que el modelo "pensara" sin parar hasta agotar el presupuesto de 8192 tokens, 3 veces seguidas, sin llegar a emitir SPARQL válido. Con el fix: la misma pregunta pasó de 1923s a **39s**, con el SPARQL correcto a la primera.
2. **Bug de Python real en el manejo de errores**: `graph_answer()` referenciaba la variable `e` de un `except Exception as e` fuera de su ámbito (Python la borra automáticamente al salir del bloque `except`), lo que convertía cualquier fallo genuino de SPARQL en un crash (`UnboundLocalError`) en vez de la respuesta de "no he podido traducir la pregunta" ya prevista. Corregido usando `error` (que sí persiste) en su lugar.
3. **Falso negativo silencioso en las búsquedas por nombre con límite de palabra (`\b`)**: el ejemplo few-shot de "Petronor" en el propio esquema enseñaba al LLM a escribir `REGEX(..., "\bpetronor\b", ...)` con una sola barra invertida. SPARQL interpreta `\b` (una barra) como el carácter de retroceso (ECHAR), no como backslash literal — el patrón de regex queda corrompido y no casa NADA, sin ningún error visible. Verificado en vivo con rdflib: con una barra, "Torre BBVA" da 0 resultados; con dos, da 1 (correcto). Corregido el ejemplo few-shot y añadida una guarda determinista nueva en `_sanitize_sparql` (dobla automáticamente cualquier `\b` suelto dentro de un literal REGEX) para que no vuelva a colar un falso negativo de este tipo aunque el LLM lo repita.

Estos tres fixes se verificaron cada uno de forma aislada (con rdflib directo) antes de darlos por buenos, y **no se relanzó `regression_qa.py` completo** por límite de tiempo de esta sesión — recomendado antes de dar el cambio por definitivo, siguiendo la práctica habitual del proyecto.

---

## 1. ¿Qué grupo político ha presentado más enmiendas a proposiciones de otros grupos, y cuántas tiene registradas en total?

### GraphRAG

```sparql
SELECT ?ng (COUNT(DISTINCT ?e) AS ?n) WHERE {
  ?e a bo:Enmienda ; bo:enmiendaPor ?g .
  ?g rdfs:label ?ng . FILTER(?g != br:grupo_desconocido) }
GROUP BY ?ng ORDER BY DESC(?n) LIMIT 10
```

**Filas (8):** EQUIPO DE GOBIERNO 1125, EAJ-PNV 125, EH BILDU 66, ELKARREKIN BILBAO 37, PSE-EE 33, PP 24, UDALBERRI 14, GOAZEN BILBAO 8.

**Respuesta:** El **EQUIPO DE GOBIERNO** lidera con **1125 enmiendas**, muy por delante de EAJ-PNV (125) y el resto (8 a 66).

**Verificación:** consulta SPARQL independiente contra el grafo → mismos 8 grupos y mismas cifras exactas, en el mismo orden. Dato estructural esperado: el Equipo de Gobierno enmienda por definición casi cualquier proposición de la oposición que no comparte, así que domina el ranking; el resto son enmiendas entre grupos de la oposición.

### RAG vectorial

**Respuesta:** con los 4 fragmentos de enmiendas que su muestra recuperó (28-01-2010, 28-11-2007, 28-03-2019, 28-09-2017), narra un **empate a 2 enmiendas entre EAJ-PNV, EB-Berdeak y PARTIDO POPULAR**, señalando aparte al Equipo de Gobierno como "no es un grupo político, se incluye solo para referencia".

**Verificación:** las 4 enmiendas concretas citadas son reales (coinciden con el patrón de las actas), pero la conclusión de "empate a 2" es **engañosa por muestreo**: no es un recuento exhaustivo, es lo que cupo en el top-k de una búsqueda semántica sobre una pregunta genérica. El propio matiz de excluir al Equipo de Gobierno de la comparación entre "grupos políticos" es razonable y distinto al criterio del grafo (que sí lo trata como `bo:Grupo` más), pero no cambia que la cifra de "2" está muy lejos de cualquier recuento real incluso entre los grupos de oposición (EAJ-PNV real: 125).

**Nota de comparación:** ejemplo limpio de la diferencia estructural entre los dos sistemas para preguntas de tipo "¿quién más...?": el grafo cuenta sobre las 3422 proposiciones completas; el vectorial narra sobre una muestra de un puñado de fragmentos y la presenta como si fuera el cuadro completo, sin señal de que es parcial.

---

## 2. ¿Qué concejal ha intervenido en más ocasiones en los debates del pleno, y a qué grupo pertenece?

### GraphRAG

```sparql
SELECT ?n ?ng (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:intervino ?con . ?con rdfs:label ?n .
  ?con bo:perteneceA ?g . ?g rdfs:label ?ng .
} GROUP BY ?n ?ng ORDER BY DESC(?c) LIMIT 1
```

**Filas (1):** Juan María Aburto Rique, EAJ-PNV, 563.

**Respuesta:** **Juan María Aburto Rique** (EAJ-PNV) es el concejal con más intervenciones registradas, **563**.

**Verificación:** ✅ CORRECTO. Consulta independiente contra el grafo → mismo nombre, mismo grupo, misma cifra exacta. Plausible y esperado: Aburto es el alcalde de Bilbao desde 2015 (hasta la fecha del corpus), preside y toma la palabra en prácticamente todas las sesiones.

### RAG vectorial

**Respuesta:** narra al **"Sr. Gil"** (Grupo Municipal Popular) con **2 intervenciones** (29-11-2012, 25-09-2014) como el concejal más activo, según su muestra de fragmentos.

**Verificación:** ⚠️ MUY INCOMPLETO. 2 frente al 563 real — la pregunta pide un máximo global sobre todo el corpus, algo que una búsqueda por similitud semántica de top-k no puede resolver por construcción. Nota adicional: el propio proyecto tiene documentado un problema de concejales homónimos apellidados "Gil" (Ronda 2026-09-13) — no se puede descartar que esta respuesta mezcle a más de un concejal distinto bajo la misma etiqueta "Sr. Gil".

**Nota de comparación:** el ejemplo más claro de toda la batería de la ventaja estructural del grafo para preguntas de tipo "el máximo/el que más": el vectorial "no sabe contar" sobre el corpus completo, solo puede narrar sobre lo que recuperó, y aquí ni siquiera acierta la identidad correcta del concejal.

---

## 3. ¿Cuántas proposiciones ha presentado el grupo VOX en el pleno de Bilbao?

_Pregunta elegida tras comprobar el grafo de antemano: VOX SÍ existe como grupo canónico en la taxonomía (`br:grupo_vox`), pero un recuento independiente confirmó 0 proposiciones presentadas — no es una pregunta trampa con un partido inventado, es un caso real de "grupo conocido, actividad nula" que además se menciona 16 veces en debates por OTROS grupos (verificado con búsqueda literal en ChromaDB), lo que la convierte en una buena prueba de si el sistema confunde "se habla de VOX" con "VOX presenta algo"._

### GraphRAG

```sparql
SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_vox . }
```

**Filas (1):** 0.

**Respuesta:** "el grupo VOX no ha presentado ninguna proposición en el pleno... no se dispone de información adicional que sugiera actividad en este sentido."

**Verificación:** ✅ CORRECTO. Coincide con el recuento independiente. Sin alucinación.

### RAG vectorial

**Respuesta:** "no aparece ninguna proposición presentada por el Grupo Municipal VOX."

**Verificación:** ✅ CORRECTO. También distingue correctamente "mencionado por otros" (16 apariciones reales de la palabra VOX en el corpus, comprobado aparte) de "presentado por VOX" (0) — no se dejó engañar por la coincidencia léxica.

**Nota de comparación:** único caso de la batería en el que ambos sistemas coinciden exactamente y sin matices — buen comportamiento anti-alucinación de los dos, en un caso genuinamente capcioso (un grupo real, en la taxonomía, pero sin actividad).

---

## 4. ¿Qué argumentos ha dado el grupo Elkarrekin Bilbao en contra de los pisos turísticos?

### GraphRAG

Tras 2 reformulaciones automáticas, la consulta final usó `bo:votoEnContraDe` (voto en contra de OTRA proposición) en vez de `bo:presentadaPor` (proposición propia) — interpretación equivocada de "en contra de los pisos turísticos" como "votó en contra de una proposición", no como "posición política contraria al fenómeno". Devolvió 4 URIs de proposiciones sin texto.

**Respuesta:** "No se encontraron datos en el grafo que describan los argumentos específicos... las filas obtenidas solo indican la existencia de cuatro proposiciones, pero no contienen información sobre el contenido."

**Verificación:** el grafo SÍ tiene lo necesario para responder bien (2 proposiciones reales de Elkarrekin sobre pisos turísticos — confirmado por búsqueda directa: 25-01-2024 y 30-01-2025 — con `bo:presentadaPor` correcto), pero el SPARQL generado nunca las alcanzó por elegir el predicado equivocado. Comportamiento honesto (no inventó nada con las 4 URIs irrelevantes que sí trajo), pero la respuesta de fondo es un **fallo real de traducción de la pregunta**, no solo una limitación estructural del grafo.

### RAG vectorial

**Respuesta:** identifica la proposición del 25-01-2024 (impuesto al turismo) y narra con detalle los argumentos reales: viviendas turísticas ilegales que deben sancionarse, tasa progresiva según tipo de alojamiento, fondo para mitigar el efecto del turismo en los barrios, ejemplo de Donostia como advertencia. Cita el resultado: decae tras enmienda del Equipo de Gobierno (28 votos: 16 a favor, 6 en contra, 6 abstenciones).

**Verificación:** ✅✅ VERIFICADO A FONDO contra el PDF real (`25-01-2024_Ordinaria_Acta.pdf`). La página 73 contiene textualmente "pisos turísticos e ilegales por todos lados, más encarecimiento de la vivienda y la expulsión de vecinos y vecinas de sus barrios" (cita real de la portavoz de Elkarrekin comparando con Donostia) — coincide con el argumento narrado. La página 77 confirma **exactas** las cifras de la votación citadas por el sistema: "Ezezko botoak: 6" (en contra: 6) y "Abstentzioak: 6" (abstenciones: 6), y la resolución literal "se aprueba la enmienda de modificación formulada por el EQUIPO DE GOBIERNO, por lo que decae la proposición presentada por el grupo municipal ELKARREKIN BILBAO" — coincide palabra por palabra con la respuesta del sistema.

**Nota de comparación:** el caso inverso al resto de la batería — aquí el vectorial da una respuesta rica, precisa y verificada al detalle, mientras que GraphRAG falla por completo la traducción semántica de la pregunta. Ilustra bien la fortaleza específica del RAG vectorial: preguntas sobre argumentos, matices y contenido narrativo de un debate concreto, donde "en contra de X" es una postura política y no un campo estructurado del grafo.

---

## 5. ¿Cuántas proposiciones se han presentado en total desde que empezó el mandato actual en 2023 hasta ahora?

_Referencia independiente: el mandato actual empezó el 17-06-2023 (acta constitutiva). Total real desde esa fecha hasta la más reciente del corpus (26-03-2026): **1065** proposiciones (o 1200 si se cuenta por año natural 2023-2026 completo, incluyendo los meses de la legislatura anterior en 2023)._

### GraphRAG

```sparql
SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:anio ?anio . FILTER(?anio = 2023) }
```

**Filas (1):** 328.

**Respuesta:** "se han presentado **328 proposiciones** en total desde el inicio del mandato actual, que comenzó en 2023, hasta la fecha actual."

**Verificación:** ❌ INCORRECTO Y ENGAÑOSO. El SPARQL solo cuenta el año natural 2023 (328, cifra en sí correcta para ESE filtro), pero la narración lo presenta como si cubriera "hasta la fecha actual" — cuando en realidad faltan 2024, 2025 y el tramo de 2026 del corpus. El total real desde el inicio del mandato es **1065** (o 1200 contando el año 2023 completo + 2024 + 2025 + 2026), más de 3 veces la cifra dada. El sistema no traduce "mandato" (un periodo plurianual desde una fecha concreta) en un filtro correcto, y encima lo redacta con una confianza que sugiere exhaustividad donde no la hay.

### RAG vectorial

**Respuesta:** narra 6 proposiciones concretas entre el 23-02-2023 y el 30-11-2023 y concluye "un total de **seis proposiciones**" en ese periodo.

**Verificación:** También muy incompleto — 6 es apenas el 1,8% incluso de las 328 reales de solo el año 2023, sin contar 2024-2026. A diferencia del grafo, aquí la limitación es transparente por construcción (es una narración de fragmentos recuperados, no pretende ser un recuento formal), aunque la palabra "total" en la conclusión puede inducir a error igualmente.

**Nota de comparación:** el hallazgo más importante de esta batería para la comparativa del TFG. Ninguno de los dos sistemas resuelve bien una pregunta de recuento acumulado sobre un periodo definido por un evento (inicio de mandato) en vez de por un valor exacto del dato (un año concreto), pero **por razones estructurales opuestas**: GraphRAG falla por una interpretación incorrecta del alcance temporal en la traducción a SPARQL (arreglable en principio, p. ej. añadiendo una regla o ejemplo few-shot para "mandato"/"legislatura" → rango de años, no año exacto); el vectorial falla porque un recuento exhaustivo está fuera de lo que una recuperación top-k puede ofrecer, sin importar cómo se reformule la pregunta — es una limitación de arquitectura, no de prompt.

---

## 6. ¿Cómo ha evolucionado el número de proposiciones sobre movilidad y transporte a lo largo de los años?

_Referencia independiente (roll-up completo `bo:trataTemaAmplio br:t_movilidad`, incluye todos los subtemas: aparcamiento, bicicleta, accesibilidad, metro, bilbobus, transporte público, peatonalización, tranvía): 2024 → 50, 2025 → 53 (cifras más altas que las que dio el sistema en vivo, ver abajo)._

### GraphRAG

```sparql
SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio ?t . ?t skos:prefLabel ?lab .
  FILTER(STR(?lab) = "movilidad" || STR(?lab) = "transporte público" || STR(?lab) = "tranvia") .
  ?p bo:anio ?anio .
} GROUP BY ?anio ORDER BY ASC(?anio)
```

**Filas (20):** de 2 (2007) a picos de 17 (2016), 24 (2024) y 22 (2025), con caída a 10 en 2026. Tabla completa y narrativa de tendencia por etapas (arranque bajo, crecimiento 2008-2010, consolidación 2012-2016, oscilación 2017-2021, reactivación 2022-2025).

**Verificación:** ⚠️ TABLA COMPLETA Y BIEN NARRADA, PERO SISTEMÁTICAMENTE INCOMPLETA. El filtro solo compara la etiqueta exacta contra 3 subtemas sueltos ("movilidad", "transporte público", "tranvia") en vez de usar el roll-up temático completo bajo `br:t_movilidad` (que en la ontología de este mismo proyecto incluye también aparcamiento, bicicleta, accesibilidad, metro y bilbobus). Verificado con una consulta independiente sobre el roll-up completo: 2024 real es **50**, no 24 — el sistema en vivo se queda en poco menos de la mitad. La forma de la tendencia (creciente hacia 2024-2025) es cualitativamente correcta, pero las cifras absolutas están sistemáticamente por debajo de la realidad en todos los años en los que existen proposiciones de los subtemas no incluidos.

### RAG vectorial

**Respuesta:** narra 4 propuestas concretas de 4 años sueltos (2010, 2011, 2014, 2016) y concluye que el número "ha sido escaso pero constante", con una evolución temática de "gestión de infraestructuras específicas" hacia "sostenibilidad y transición ecológica".

**Verificación:** ❌ NO responde realmente a una pregunta de evolución temporal — ni identifica el pico real de 2024-2025 (que si acaso está anotado por el grafo, con matices, como el año de mayor actividad), ni ofrecer cifras por año. Es una muestra anecdótica de 4 ejemplos presentada como si fuera un resumen de la tendencia completa.

**Nota de comparación:** las preguntas de "evolución a lo largo del tiempo" son exactamente el tipo de pregunta que un grafo con recuento agregable puede responder bien y una búsqueda semántica no puede en absoluto — pero este caso concreto también deja una lección importante para el propio GraphRAG: **tener el roll-up temático correcto en la ontología no basta si el LLM generador de SPARQL decide no usarlo** y en su lugar enumera subtemas sueltos a mano. La cifra resultante "parece" razonable (una tabla completa, una narrativa coherente) pero está sistemáticamente subestimada sin ningún aviso.

---

## 7. ¿Qué se ha dicho en el pleno sobre el BBVA?

_Referencia independiente: solo 1 proposición tiene a "Torre BBVA" como entidad canónica vinculada (`bo:menciona`) en el grafo — la extracción de entidades del enriquecimiento es mucho menos exhaustiva que una búsqueda literal; una búsqueda literal directa en ChromaDB encuentra la cadena "BBVA" en **34 fechas de pleno distintas** del corpus completo._

### GraphRAG

Se repitió la pregunta 3 veces tras el fix del bug de `\b` (sección 0). El primer intento, antes del fix, generó `bo:menciona` (el predicado correcto) pero con el bug del backslash → 0 resultados (falso). Tras el fix, los 2 intentos siguientes generaron consultas sintácticamente válidas pero con el predicado equivocado (`bo:trataTemaAmplio`, tratando "BBVA" como si pudiera ser un tema/categoría en vez de una entidad mencionada) → 0 resultados, por una razón distinta y también incorrecta para la intención real de la pregunta.

```sparql
-- intento típico tras el fix (predicado equivocado):
SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio ?t . ?t skos:prefLabel ?lab .
  FILTER(REGEX(STR(?lab), "\\bbbv[aá]\\b", "i")) }
```

**Respuesta (repetida, 2/3 intentos):** "no aparece ninguna proposición que haya tratado el tema amplio 'BBVA'... no hay contenido que describir."

**Verificación:** el bug técnico del backslash quedó confirmado y corregido (verificado end-to-end con rdflib: 0→1 con el predicado correcto `bo:menciona`), pero persiste un problema distinto y no determinista: en 3 intentos de la MISMA pregunta, el generador de SPARQL solo acertó el predicado correcto una vez (y esa vez tenía el bug de escritura), y las otras dos confundió "qué se ha dicho sobre BBVA" con "¿es BBVA un tema del pleno?" — algo que el propio esquema ya distingue con un ejemplo few-shot equivalente (Petronor, `bo:menciona`), pero que el modelo local de 8B no aplica de forma fiable a un nombre distinto en cada ejecución.

### RAG vectorial

**Respuesta:** identifica una proposición del 29-11-2012 (Grupo BILDU, sobre la BBK) donde se menciona de pasada que "la BBK moviliza 55 de cada 100 pesetas... el resto se reparte entre otras entidades financieras... entre ellas Santander y BBVA", y concluye que "el BBVA solo aparece mencionado una vez... de forma incidental."

**Verificación:** ✅ el hecho puntual citado es real y está bien contextualizado (mención incidental, no debate sustantivo). ⚠️ PERO la conclusión de "solo una vez" es una infraestimación real: una búsqueda literal directa confirma que la cadena "BBVA" aparece en **34 fechas de pleno distintas** del corpus (no 1). El propio bloque de FUENTES del sistema cita 12 documentos distintos, más que la "una vez" de la conclusión — el narrador no sintetizó correctamente su propia recuperación.

**Nota de comparación:** ambos sistemas, por caminos distintos, terminan subestimando la cobertura real de una entidad de mención dispersa (no concentrada en una sola proposición memorable como Tubacex en rondas anteriores, sino repartida en menciones de pasada a lo largo de muchos años). El grafo falla en la traducción de la pregunta a SPARQL (elige mal el predicado, de forma no determinista); el vectorial recupera bien pero su narrador no agrega correctamente sus propias fuentes citadas. Ninguno de los dos falla por alucinar un dato falso — los dos fallan por quedarse cortos de forma silenciosa.

---

## Resumen general

**Bugs de código reales encontrados y corregidos** (no solo matices de calidad): el modo "pensamiento" de qwen3:8b sin desactivar en la generación de SPARQL de producción (32 min → 39s tras el fix), un `UnboundLocalError` real en el manejo de errores de `graph_answer()`, y un límite de palabra `\b` mal escapado en un ejemplo few-shot que producía falsos negativos silenciosos (verificado con rdflib: confirma el problema y confirma el fix).

**GraphRAG** gana con claridad en preguntas de recuento exhaustivo/ranking sobre datos estructurados simples (enmiendas por grupo, concejal con más intervenciones, comprobar un grupo minoritario real) — en esos casos, las cifras coinciden EXACTAS con consultas SPARQL independientes. Pero esta batería también expone que el "éxito" del grafo depende por completo de que el LLM generador de SPARQL elija el predicado y el alcance temático/temporal correctos, algo que **no es determinista incluso repitiendo la misma pregunta** (BBVA: 3 intentos, 3 resultados distintos) y que puede producir una tabla completa y bien narrada que aun así está sistemáticamente subestimada sin ningún aviso (movilidad por año, mandato 2023).

**RAG vectorial** gana con claridad en la única pregunta puramente argumentativa/cualitativa de la batería (Elkarrekin y los pisos turísticos), con una respuesta verificada palabra por palabra contra el PDF real, cifras de voto exactas incluidas. Pero falla sistemáticamente en cualquier pregunta que exija un recuento o máximo sobre el corpus completo (2 intervenciones de "Sr. Gil" frente a las 563 reales de Aburto; "seis proposiciones" del mandato 2023 frente a las 1065 reales) — una limitación de arquitectura (retrieval top-k), no de prompt, que ninguna reformulación puede arreglar sin un cambio de recuperación (como ya hizo la Ronda 15 con el canal temático, pero que no cubre recuentos globales).

**Coincidencia perfecta entre ambos** solo en la pregunta con verdad más simple y menos ambigua de la batería (VOX, 0 proposiciones) — un buen recordatorio de que la comparación es más reveladora cuanto más compleja o ambigua es la pregunta.
