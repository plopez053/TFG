# Decisiones técnicas — material para la memoria del TFG

Este documento recoge el "por qué" de las decisiones de diseño no obvias del
sistema. Se ha sacado de los comentarios del código para: (a) dejar el código con
comentarios sencillos de "qué hace", y (b) tener el razonamiento agrupado y listo
para redactar la memoria.

Organizado por subsistema.

---

## 1. RAG vectorial — recuperación

### 1.1 Búsqueda híbrida: tres canales

La recuperación no es solo búsqueda semántica por embeddings. Se combinan tres
canales porque cada uno cubre un fallo del anterior:

| Canal | Busca por | Cubre |
|---|---|---|
| Semántico | cercanía de embeddings (nomic-embed-text) | afinidad de significado |
| Literal (`_keyword_search`) | coincidencia textual exacta de palabras clave / nombres propios | menciones únicas que el embedding no acerca (ej. una empresa citada una sola vez) |
| Temático (`_thematic_search`) | clasificación: chunks con `tema_principal` = tema de la pregunta | preguntas generales sobre un tema amplio ("¿qué propuestas de vivienda ha habido?") |

**Motivación del canal temático:** una pregunta vaga sobre un tema ("qué se ha
hecho en movilidad") no tiene por qué compartir vocabulario con los chunks
relevantes. La búsqueda semántica y la literal miden coincidencia de TEXTO; el
canal temático mide coincidencia de CLASIFICACIÓN. Mismo principio que
`bo:trataTemaAmplio` en GraphRAG, aplicado al lado vectorial.

### 1.2 Roll-up de temas (subtema → tema padre)

`_load_tema_rollup` construye una tabla "tema de nivel 1 ← subtemas de nivel 2"
a partir de la taxonomía SKOS (`themes_skos.ttl`).

**Por qué hace falta:** el tema de nivel 1 "presupuestos y fiscalidad" solo tiene
"fiscalidad" como sinónimo propio. Palabras naturales como "presupuestos" o
"impuestos" son sinónimos de sus SUBTEMAS ("presupuesto municipal", "ordenanzas
fiscales"). Sin el roll-up, una pregunta con esas palabras no detectaba ningún
tema. Además, algunas proposiciones tienen `tema_principal` fijado directamente a
un subtema (ej. "comercio") en vez de al tema padre; sin el roll-up en el FILTRO
(no solo en la detección), esos chunks quedaban fuera de una búsqueda temática
amplia aunque la pregunta sí se detectara.

Se calcula una vez y se cachea a nivel de módulo (parsear el TTL en cada pregunta
sería caro). Un fallo transitorio NO se cachea, para que la siguiente pregunta lo
reintente en vez de dejar el canal temático apagado.

### 1.3 Palabras clave: con y sin tildes

- `_extraer_keywords_pregunta` (para comparar contra texto ya normalizado):
  quita tildes.
- `_extraer_keywords_literales` (para `where_document {"$contains": ...}` contra
  el texto CRUDO del PDF): **conserva tildes**. ChromaDB `$contains` es una
  comparación de substring literal que no ignora acentos. La mayoría del
  vocabulario administrativo español va con tilde ("información", "gestión",
  "situación"), así que sin tildes el canal literal fallaba en silencio para casi
  todo ese vocabulario.
- Mínimo 6 caracteres: reduce falsos positivos por raíces cortas (ej. "presu" de
  "presupuesto" coincidiendo con "presencia").

### 1.4 Corte de candidatos antes de Cohere

Cada canal aporta como máximo ~22 documentos antes de concatenar. Cohere solo
rerankea los primeros ~64 por orden de entrada. Si el canal literal (que puede
devolver 60+) llena el cupo, el canal temático —que es el que trae los años
recientes de un tema— nunca llega a verse. Orden de concatenación: literal
(preciso y escaso) → temático → semántico. El cap de Cohere se subió a 80.

### 1.5 Selección del contexto final: "rescate" de hits de alta confianza

`_select_final_docs` con la "aspiradora" (`_expand_context_by_topic`) expande y
filtra por ASUNTO con título limpio, y descarta chunks de "General / Introducción"
o de puntos de "dación de cuenta" aunque Cohere los haya puntuado muy alto —
justo donde vive parte del contenido de preguntas generales.

Solución: los 8 mejores candidatos según Cohere SIEMPRE sobreviven al filtro, y
además van los PRIMEROS del contexto (por delante del orden cronológico). Si se
dejan mezclados por fecha, el narrador no los distingue del ruido y responde "no
encontrado" pese a tenerlos delante. Los chunks del canal literal (mencionan
textualmente un término de la pregunta) tampoco los puede tirar el filtro de
tema aunque Cohere no los vea.

### 1.6 Chunk representante en la búsqueda temática

`_thematic_search` elige un chunk por proposición. Coger el de `chunk_index` más
alto daba el de la votación ("Votos emitidos: 29...", sin contenido, Cohere lo
puntúa ~0). Se coge el chunk del CUERPO más largo (descartando el primero y los
dos últimos, quitando la línea de votación).

### 1.7 Filtro de partido

`_apply_party_filter`: si la pregunta nombra un grupo, se filtran los chunks a los
de ese grupo. Sobre el corpus ya expandido se usa umbral `min_docs=1` (los docs
ya son del tema correcto, puede ser agresivo).

### 1.8 Concurrencia

`retrieve_context` corre en threads separados por petición (`asyncio.to_thread`
desde el frontend). El roll-up de temas y el singleton del pipeline usan lock:
sin él, dos preguntas concurrentes en el arranque en frío disparaban el parseo
completo del TTL dos veces a la vez.

### 1.9 Orden de keywords no determinista dentro de `_keyword_search`

Bug real encontrado en regresión local (2026-09-12), reproducido y corregido.
`_keyword_search` seleccionaba las keywords con
`sorted(set(keywords), key=len, reverse=True)[:3]`: para dos keywords del mismo
largo (p.ej. "empresa" y "tubacex", ambas 7 caracteres), el orden entre ellas
dependía de la iteración de `set()`, que depende del `PYTHONHASHSEED` aleatorio
de cada proceso Python — la MISMA pregunta podía procesar "tubacex" antes o
después de "empresa" según qué arranque del servidor le tocara.

Esto importa porque el resultado combinado de todas las keywords se corta con
`[:22]` en `_retrieve_and_rank` (ver §1.4) ANTES de dedup/concatenar con los
otros canales. "empresa" es una palabra genérica con ~30 coincidencias en el
corpus; "tubacex" (el nombre de la empresa que preguntaba el usuario) solo
tiene 1. Si "empresa" se procesaba primero, sus 30 hits llenaban el cupo de 22
y el único hit real de "tubacex" nunca llegaba a Cohere ni al contexto final —
el sistema respondía "no hay ninguna referencia" con seguridad, siendo falso
(exactamente el patrón de la Ronda 25, pero introducido de nuevo por esta vía).

Verificado en aislado: en 6 arranques de proceso distintos, el orden
`sorted(set(...))` salió `['empresa','tubacex']` en unos y
`['tubacex','empresa']` en otros — no es un bug ligado a qué LLM narra la
respuesta (Groq/Ollama), es puramente el hash-seed del proceso.

**Fix**: `_keyword_search` ya no concatena los resultados por keyword en una
lista plana; los guarda en `by_kw: Dict[str, List[Document]]` y los devuelve
intercalados en ronda robin (una posición de cada keyword por vuelta). Así los
primeros ~len(keywords) puestos del resultado SIEMPRE tienen representación de
todas las keywords, sin importar cuántas coincidencias totales tenga cada una
ni en qué orden se procesaron. Verificado: "tubacex" sobrevive al corte
`[:22]` en las 4 repeticiones de proceso probadas tras el fix (antes fallaba
en las que "empresa" salía primero).

**Lección**: cualquier truncamiento aguas abajo (aquí `[:22]`) sobre una lista
construida concatenando resultados de fuentes de tamaño muy desigual necesita
una garantía de representación justa (intercalado, cupo por fuente...) — no
basta con que la fuente rara "en teoría" esté en la lista si el orden de
llegada no está garantizado.

---

## 2. RAG vectorial — generación

### 2.1 Fallback de proveedor LLM

Orden: Groq `gpt-oss-120b` (rápido, sin GPU) → Ollama `qwen2.5:7b` local. Si Groq
da error (rate limit del free tier, sin API key), se cae al local automáticamente.
`qwen2.5:7b` se eligió sobre `gemma3:4b` por mejor prosa en la narración.

### 2.2 Timeout explícito en el cliente Ollama

`ChatOllama` sin `timeout` explícito: una conexión colgada (modelo recargándose,
generación estancada por presión de VRAM) bloquea el proceso indefinidamente sin
error. Se fija `client_kwargs={"timeout": 600}`.

### 2.3 Reintento en la narración

En máquinas con poca RAM, Ollama tarda en recargar el modelo LLM después de las
llamadas de embedding y rechaza la conexión durante ese intervalo. Reintentos con
backoff corto (3 intentos, 10 s) lo resuelven.

### 2.4 Reestructuración del prompt de respuesta (Título/Propuesta/Resumen/Votos)

El formato de respuesta original pedía 3 campos (Propuesta/Argumentos/Resultado)
mediante ~9 reglas de texto sin ningún ejemplo concreto de cómo debía verse el
resultado. Con el modelo local (fallback de Groq) la estructura se degradaba: en
pruebas sin Groq, el modelo mezclaba los campos en un párrafo plano o se saltaba
el grupo proponente en la cabecera. Mismo patrón que el estudio de SPARQL (§3.3):
reglas largas sin ejemplo concreto penaliza a los modelos pequeños.

Solución: reglas reducidas al mínimo + un ejemplo concreto ya redactado con el
formato exacto, y el campo "Título" añadido explícitamente (ya existía como
metadato `topic` en el contexto, pero no se pedía como campo propio de la
respuesta). El campo "Fuente" NO se pide al LLM — ya se muestra aparte en la UI
vía `build_sources_data` con el PDF/página exactos; pedírselo al modelo sería
una oportunidad más de que lo invente o lo omita sin necesidad.

### 2.5 `mistral:7b` y `gemma3:4b` probados y descartados como modelo local

A petición del usuario, se probaron como alternativa a `qwen2.5:7b` para la
narración (mismo contexto real recuperado, mismo prompt, solo cambia el modelo):

- **mistral:7b**: calidad de contenido similar, pero sigue peor el formato
  pedido de forma inconsistente (en una de dos preguntas ignora la estructura de
  subapartados) y es un 25-55% más lento en el mismo hardware. Sin ventaja clara.
- **gemma3:4b**: **alucinó una entrada completa** copiando literalmente los
  valores de un ejemplo few-shot *ficticio* del prompt ("GRUPO MUNICIPAL X",
  "Sr. García", "26-10-2010", "29 a favor / 0 en contra") como si fuera un
  pleno real, mezclado entre datos verdaderos. No distinguió "esto es un
  ejemplo de formato" de "esto es un dato que puedo usar". Confirma el mismo
  patrón que en SPARQL (ver §3.4 más abajo): por debajo de 7B el modelo no solo
  pierde precisión, pierde la capacidad de separar plantilla de contenido.

Conclusión: se mantiene `qwen2.5:7b` como fallback local de narración. Ver
`graphrag/graphrag/ESTUDIO_SPARQL_LOCAL.md` para el mismo experimento aplicado a
generación de SPARQL (ahí sí se usa un modelo distinto, `qwen3:8b`, ver §3.3).

### 2.6 Con `qwen2.5:7b`: bloques duplicados y fecha en prosa (2026-09-12)

Probado en vivo (Groq desactivado, solo Ollama+Cohere) tras la regresión local
de la Ronda 35: la narración con el modelo local se aparta de la estructura
pedida de dos formas distintas, ninguna relacionada con la recuperación.

**(a) Bloques repetidos**: en una respuesta multi-sesión, `qwen2.5:7b` copió el
MISMO bloque completo (fecha + grupo + resumen, palabra por palabra) 4 veces
seguidas — y ese bloque en concreto ni siquiera correspondía a ningún documento
recuperado real (fecha+grupo inventados, no presentes en `last_retrieved_docs`).
Falla de repetición/alucinación típica de un modelo de 7B con contexto largo y
muchos fragmentos de asunto parecido, no arreglable solo con prompt.

**(b) Fecha en prosa en sesión única**: `_PROMPT_SINGLE_SESSION` decía
`"...del [fecha]..."` sin especificar formato. `qwen2.5:7b` escribió *"del 26
de enero de 2023"* en vez de *"del 26-01-2023"*. Esto rompe el enganche de
fuentes en `frontend/app.py`, que busca la fecha en formato DD-MM-YYYY LITERAL
dentro del texto para insertar el enlace justo debajo del bloque correspondiente
— si esa cadena exacta no aparece, la fuente cae al lote "no ubicadas" (se sigue
mostrando por la red de seguridad ya existente, pero mal colocada).

**Fix (dos capas, guarda determinista + prompt, mismo principio que el resto
del proyecto — ver "How to apply" al final del documento de memoria del
proyecto)**:
- Prompt: `_PROMPT_SINGLE_SESSION` ahora exige explícitamente DD-MM-YYYY
  ("nunca en palabras"); ambas plantillas añaden la regla "un bloque por
  [fecha]-[proposición], nunca repitas el mismo bloque".
- Código (no depende de que el LLM obedezca): `dedup_answer_blocks()` en
  `backend/rag.py` colapsa párrafos exactamente repetidos (normalizando
  espacios/mayúsculas) antes de insertar fuentes — se aplica en `query()` (CLI)
  y en `frontend/app.py` antes del bloque de inserción de enlaces.
  `find_date_mentions()` reconoce la fecha tanto en DD-MM-YYYY como en prosa
  española ("26 de enero de 2023" → mismo punto de anclaje), así que aunque el
  modelo no obedezca la instrucción nueva del prompt, la fuente sigue
  encajando en su sitio.

Verificado en vivo tras el fix: las mismas preguntas ya no repiten bloques (una
incluso mejoró — el modelo reconoció honestamente "sin bloques relevantes para
Tubacex" en vez de inventar uno) y la fecha de sesión única sale ya en
DD-MM-YYYY. Regresión local completa (`scripts/regression_qa.py`, Groq/Gemini
desactivados) relanzada para confirmar que no rompe nada — no toca la
recuperación, solo post-procesa el texto ya generado.

**Ampliación (mismo día): anclaje por TÍTULO para sesión única con varias
propuestas.** La limitación de arriba SÍ se abordó: `find_item_anchors()` en
`backend/rag.py` segmenta la respuesta por cada línea "Título: ..." (acepta
tanto "- Título: x" como "- **Título:** x", los dos estilos que qwen usa según
la tirada) y devuelve (inicio, fin, texto del título) por propuesta. En
`frontend/app.py`, cuando es sesión única Y hay ≥2 títulos detectados, se
sustituye el anclaje por fecha por anclaje por título: para cada tramo se busca
la fuente cuyo `topic` (el ASUNTO real de la acta) comparte más palabras
significativas con el título que escribió el LLM — igual principio que el
"mejor" match ya existente para el anclaje por fecha, pero aplicado a nivel de
propuesta individual, no de pleno completo. A diferencia de ese caso, aquí NO
se fuerza una asignación con 0 palabras compartidas (`candidatos[0]` de
respaldo) — con un candidato equivocado se citaría la fuente incorrecta bajo un
título concreto, peor que dejarla sin marcar en el lote final.

Verificado en vivo (26-01-2023, 5 propuestas, qwen2.5:7b): 3/5 quedaron con su
fuente correcta justo debajo (antes del fix: solo 1/5); las 2 sin señal
suficiente (overlap 0) se dejaron para el lote final en vez de arriesgar una
atribución errónea. Sesión única de un solo bloque y multi-sesión siguen
exactamente igual que antes (anclaje por fecha, ya verificado en Ronda 35/36).

**Refuerzo con Groq (mismo día)**: probado el mismo caso con Groq (títulos más
limpios, suelen empezar con el número de punto del orden del día tal cual en
el acta, ej. "18. PROPOSICIÓN..."). Encontrado un caso real de falso positivo
por solape de palabras: dos propuestas del PARTIDO POPULAR el mismo día
comparten solo "partido"+"popular" como palabras significativas — el emparejamiento
por solape puro asignó el título "28." a la fuente numerada "27." (misma
familia de palabras, número distinto). Fix: `match_source_by_title()` (nuevo,
`backend/rag.py`) intenta PRIMERO por número de punto del orden del día
(`item_number()`, regex al inicio del texto) si es único entre los candidatos
— más fiable que las palabras cuando varias propuestas comparten proponente;
si no hay número o no es único, cae al solape de palabras pero exigiendo que
cubran ≥34% de las palabras significativas del título (no basta 1 palabra
suelta si el título tiene muchas — así se sigue permitiendo el caso legítimo de
un título de una sola palabra distintiva, como "EH BILDU", que antes ya
funcionaba). Verificado: el mismo caso de Groq pasa de 7/8 enganchadas (con 1
mal asignada) a 6/8 (las 6 con número exacto, ninguna mal asignada) — mejor
enganchar menos y bien que más y con un error de cita.

**Verificación de estabilidad de los fixes (a) y (b)**: repetida la pregunta de
Tubacex 3 veces con qwen2.5:7b — cero bloques duplicados en las 3, y las 3
identificaron y citaron correctamente la única mención real (27-05-2021, EH
BILDU) tanto en el bloque como en la CONCLUSIÓN. Limitación NO relacionada con
estos fixes, ya documentada (Ronda 10/20): el modelo local sigue incluyendo
bloques de plenos semánticamente parecidos pero irrelevantes para la pregunta
(ruido de recuperación, no de formato) — la CONCLUSIÓN sí sintetiza
correctamente a pesar del ruido del cuerpo.

### 2.8 Con `qwen2.5:7b`: proponente "pegajoso" y bloques vacíos por ruido de recuperación (2026-09-12)

Encontrado en la batería de la Ronda 36 (pregunta sobre la empresa Iberdrola,
que solo tiene 2 keywords de búsqueda ["empresa","iberdrola"] — "empresa" trae
ruido de plenos no relacionados, mismo patrón de fondo que Tubacex en §2.6/2.7
pero con un síntoma distinto). Dos problemas de JUICIO del modelo local, no de
recuperación ni el bug de duplicado ya arreglado (aquí cada bloque tenía
contenido DISTINTO, no repetido):

**(a) Grupo proponente "pegajoso"**: el modelo repitió "GRUPO MUNICIPAL
SOCIALISTAS VASCOS" como cabecera en TODOS los bloques de la respuesta, incluso
en uno que en el contexto real era una dación de cuenta de la Alcaldía sin
ningún grupo político asociado (`[PLENO: 30-10-2008] 4. Se da cuenta de la
resolución de la Alcaldía...`). El modelo no leía el `[PLENO: fecha] ASUNTO` de
CADA bloque para determinar el proponente — copiaba el del bloque anterior.

**(b) Bloques vacíos para plenos irrelevantes**: en vez de omitir un pleno
recuperado por ruido de keyword pero sin relación real con la pregunta, el
modelo escribía un bloque completo con "Título: No se encuentra en el contexto
proporcionado." / "Propuesta: No se encuentra..." (o, en otra tirada, colapsaba
todo a la cabecera "**[fecha] — NO HAY INFORMACIÓN RELEVANTE**").

**Fix (prompt + guarda determinista, mismo principio que el resto)**:
- Prompt (`_PROMPT_MULTI_SESSION`): 2 reglas nuevas — "el grupo proponente de
  CADA bloque es el que aparece EN ESE FRAGMENTO CONCRETO, nunca lo copies del
  bloque anterior; si el fragmento no menciona grupo (resolución
  administrativa), escribe 'Ayuntamiento de Bilbao' o el título real" y "si un
  pleno del contexto no tiene relación real con la pregunta, OMÍTELO POR
  COMPLETO, nunca escribas un bloque diciendo que no hay información".
- Código: `strip_empty_blocks()` nuevo en `backend/rag.py`, aplicado junto a
  `dedup_answer_blocks()` en `query()` y `frontend/app.py`. Detecta y elimina
  el párrafo/bloque ENTERO en dos formas: (1) Título Y Propuesta dicen
  "no se encuentra"/"no se proporciona"/"no hay información" a la vez — se
  exige AMBOS campos para no borrar un bloque legítimo que solo tiene el
  Resumen o los Votos escuetos (eso sí es información real); (2) el bloque se
  colapsa a una única línea de cabecera con esa misma frase.

Verificado en vivo, 2 repeticiones consecutivas tras el fix: 0 bloques vacíos
en ambas, atribución de grupo correcta en cada bloque incluyendo el caso de la
Alcaldía ("**[30-10-2008] — ALCALDIA DE BILBAO**", ya no "Socialistas Vascos"),
y un grupo distinto correctamente identificado en el primer bloque ("EZKER
BATUA-BERDEAK" en vez de arrastrar el grupo de otra pregunta). No se relanzó la
regresión completa: `scripts/regression_qa.py` verifica filas de SPARQL y
recuperación (`_retrieve_and_rank`/`_select_final_docs`), nunca llama a
`query()`/`invoke_llm`/`build_answer_prompt` — estos cambios son puramente de
narración y no tocan ningún camino que la regresión cubra; la verificación
correcta es la prueba en vivo ya hecha, no repetir la suite.

### 2.9 Validación contra PDF real (no contra el propio sistema) y mitigación de la mezcla entre puntos del orden del día (2026-09-12)

A petición explícita del usuario ("¿has validado la información contra los
PDF? esto es lo más importante") — hasta este punto la Ronda 36 solo había
comprobado estructura (duplicados, fechas, fuentes), no el CONTENIDO de las
respuestas contra el texto real del acta. Verificado con `pypdf` (extracción
de texto por página, sin OCR) contra dos actas:

- **Tubacex (27-05-2021, pág. 76)**: la cita coincide PALABRA POR PALABRA con
  el PDF ("un abrazo solidario a las plantillas de Tubacex, de PCI-ITP,
  Petronor y UTE Bilboko Argiak, que estos días están llevando adelante unas
  jornadas de movilización...").
- **26-01-2023**: encontrado un caso real de alucinación por MEZCLA entre
  puntos del orden del día. Un bloque atribuido al punto "15." (subvenciones a
  personas mayores, 860.817€, aprobado por UNANIMIDAD 29-0 según pág. 36) tenía
  el título correcto pero la Propuesta hablaba de "comedores sociales" (eso es
  el punto 14, 619.128,55€, pág. 21) y los Votos citados eran "19 a favor, 7 en
  contra, 3 abstenciones" (eso pertenece al punto 20, propuesta de ELKARREKIN
  sobre agua/envases, pág. 119-120) — tres puntos DISTINTOS mezclados en un
  bloque. El enlace de fuente (pág. 35, correcto vía número de punto) sí
  apuntaba al documento correcto; el texto narrado no.

**Comprobado con Groq sobre la MISMA pregunta (2 repeticiones)**: sin ningún
caso de mezcla — cada punto numerado mantiene sus propias cifras de voto de
forma estable entre repeticiones. Verificado además un segundo hecho contra el
PDF real: el punto 21 (VIH/serofobia) — la pág. 128 dice literalmente "esta
proposición sale por UNANIMIDAD y... SIN ENMIENDA", y Groq narró exactamente
eso (29 a favor, aprobada por unanimidad). Conclusión: la mezcla entre puntos
es una limitación específica de los modelos locales pequeños en esta tarea de
síntesis con muchos puntos parecidos en un mismo contexto, no del pipeline de
recuperación (los documentos correctos SÍ estaban en el contexto en el caso
fallido).

**Fix aplicado** (prompt, ambas plantillas): regla nueva — "cuando el contexto
trae VARIAS proposiciones seguidas y parecidas, presta especial atención a NO
mezclar título/propuesta/resumen/votos de una con los de otra; cada bloque
debe venir ÍNTEGRAMENTE del mismo fragmento". No hay guarda determinista
posible aquí (a diferencia de dedup/strip_empty_blocks) porque detectar una
mezcla de contenido requeriría el mismo tipo de comprensión semántica que falla
— es un límite de prompt, no de código.

**Comparativa de 4 modelos sobre el MISMO prompt exacto** (capturado una vez
con `build_answer_prompt`, reutilizado para los 4 para que la comparación sea
justa — solo cambia quién narra, no qué recibe):

| Modelo | Tiempo | Resultado |
|---|---|---|
| Groq `gpt-oss-120b` | ~1-2s | Limpio, verificado 2x contra PDF, sin mezclas |
| `qwen2.5:7b` (actual, tras el fix) | 37-63s | Ya no roba votos de un punto no relacionado; a veces aún FUSIONA 2 propuestas parecidas en un bloque (p.ej. personas mayores + comedores) |
| `qwen3:8b` | 242s (~6x más lento) | Separó comedores/personas-mayores perfectamente en esta prueba (mejor que qwen2.5:7b); una cifra de votos (19/7/3) se repitió sospechosamente entre 2 puntos distintos, sin confirmar si es coincidencia real o error — pendiente de más verificación si se decide perseguirlo |
| `qwen2.5-coder:7b` | 63s | PEOR: tituló un bloque "comedores sociales" con contenido real de otro punto (vestuarios/discapacidad), e INVENTÓ un pasaje entero ("convenio con el Ayuntamiento de Madrid... 35 años de lucha y 21 víctimas mortales") sin ninguna base en el acta — alucinación pura, no solo mezcla; además una cifra de votos con la suma interna incoherente (19+10+10≠29) |

**Decisión**: mantener `qwen2.5:7b` como fallback local por defecto — el fix ya
quitó el fallo más grave (robar cifras de un punto totalmente ajeno) y es
4-6x más rápido que qwen3:8b, relevante para un chat interactivo. `qwen3:8b`
queda anotado como alternativa si se prioriza calidad de narración local sobre
latencia (mismo criterio que llevó a elegirlo para GraphRAG en el estudio
SPARQL). `qwen2.5-coder:7b` descartado para esta tarea de síntesis narrativa —
confirmado peor que el actual, con riesgo de inventar contenido. Ningún cambio
de configuración aplicado (`LLM_MODEL_LOCAL` sigue en `qwen2.5:7b`) — decisión
de PRIORIZAR VELOCIDAD para el fallback de emergencia, dado que Groq es lo que
se usa en producción salvo caída del servicio.

**Lección más importante de esta ronda**: verificar contra el PDF real (no
contra el propio contexto recuperado ni contra la narración de otro LLM) es lo
único que puede destapar una mezcla de contenido — ni la regresión automática
ni las pruebas de estructura de esta sesión la habrían detectado, porque el
enlace de fuente SÍ era correcto y la estructura del bloque era perfecta; solo
el CONTENIDO citado dentro de ese bloque correcto era falso.

### 2.10 Intentos de mejorar qwen2.5:7b / acelerar qwen3:8b, y guarda determinista de votos (2026-09-12)

A raíz de §2.9, se probaron dos vías de PROMPT antes de concluir que hacía
falta una guarda de código:

- **`think=False` en qwen3:8b** (esperando repetir el x4 de velocidad de la
  Ronda 7 en extracción): solo dio 18% de mejora (242s → 198s). La Ronda 7 era
  una tarea CORTA de clasificación donde el bloque de pensamiento dominaba el
  coste; aquí domina la generación de una respuesta larga, así que ahorrarse
  el pensamiento previo apenas cambia el total. Además, en esa misma tirada
  **sí mezcló contenido** entre dos puntos (algo que la primera muestra de
  qwen3:8b no había hecho) — confirma que la buena impresión inicial de
  qwen3:8b en §2.9 era una muestra afortunada (n=1), no una mejora real:
  sigue teniendo el mismo fallo, solo que no siempre.
- **Ejemplo concreto en el prompt** (la técnica que sí funcionó para el
  formato en su día, §2.4): se añadió a `_PROMPT_SINGLE_SESSION` un ejemplo
  con los datos REALES del caso de la Ronda 36 (incorrecto vs correcto,
  mismas cifras de personas mayores/comedores). Probado 2 veces: la mezcla
  SIGUE pasando igual — a veces incluso con más bloques mezclados que antes.
  Conclusión: evitar la mezcla exige "recordar qué cifra va con qué título" a
  lo largo de un contexto largo con muchos ítems parecidos — una capacidad de
  seguimiento, no de formato; un ejemplo no la enseña de forma fiable a un
  modelo de 7B.

**Fix real: guarda determinista, no más prompt.** `replace_votos_line()`
(nuevo, `backend/rag.py`) sustituye el CONTENIDO de la línea "- Votos: ..." de
cada bloque por el `vote_result` real de la fuente ya emparejada vía
`match_source_by_title()` (metadata determinista, no lo que escribió el LLM).
Conectado en `frontend/app.py`, en la misma rama de sesión única con varias
propuestas: tras emparejar cada `(start, fin, título)` con su fuente, se
corrige el bloque ANTES de insertar el enlace, procesando en orden
descendente de posición para no invalidar índices de bloques anteriores (mismo
patrón ya usado para insertar fuentes). Verificado end-to-end con una
respuesta real de qwen2.5:7b: 3 de 8 bloques tenían una fuente lo bastante
fiable para corregir, y las 3 correcciones fueron verificadas correctas contra
el PDF real (antes tenían cifras robadas de otro punto o inconsistentes). Los
otros 5 bloques (títulos duplicados o sin señal suficiente) se dejan tal cual
escribió el modelo — no se fuerza una corrección insegura.

**Decisión final sobre modelos**: ninguno de los 3 modelos probados en la
sesión (§2.9) mejora a qwen2.5:7b lo bastante como para justificar el cambio
de configuración; la mejora real vino de código, no de elegir otro modelo.
`LLM_MODEL_LOCAL` sigue en `qwen2.5:7b`.

---

## 3. GraphRAG — generación de SPARQL

Ver también `graphrag/graphrag/ESTUDIO_SPARQL_LOCAL.md` (estudio de ablación
completo sobre estrategias de prompt y modelos).

### 3.1 Anti-patrones que el LLM genera y producen cifras falsas sin error

`_sanitize_sparql` reescribe patrones que el LLM genera de forma recurrente y que
dan un resultado equivocado SIN dar error de sintaxis (el fallo más peligroso: se
narra como un hecho). Solo actúa cuando el patrón es inequívoco.

| # | Anti-patrón | Efecto | Corrección |
|---|---|---|---|
| 1 | `OPTIONAL` cuyas variables no se usan en ningún agregado ni GROUP BY, en una consulta de COUNT | el OPTIONAL es un no-op, el COUNT cuenta TODO (ej. "euskera rechazadas" devolvía 55 en vez de 3) | si el cuerpo tiene un objeto concreto, era un filtro → se promueve a obligatorio |
| 2 | `?g rdfs:label ?ng` con `?g` sin ligar al resto del patrón | `?g` se cruza con todas las etiquetas del grafo → miles de filas, COUNT inflado | se elimina el triple suelto |
| 3 | roll-up temático a mano con `skos:broader*` + URI inventada | `skos:broader` no tiene cierre transitivo materializado + URI falsa → 0 resultados | → `bo:trataTemaAmplio <URI canónica>` |
| 4 | URI de tema inventada (`br:t_movilidad_y_transporte`, `bo:t_seguridad`) | 0 resultados en silencio | se mapea a la URI canónica real |
| 4a | `bo:anio "2015"^^xsd:int` o `bo:anio "2015"` (string) | tipo distinto de xsd:integer → 0 resultados | se normaliza a entero plano |
| 4b | `?g rdfs:label "Partido Popular"@es` | los labels reales son "PP", "EH BILDU"... y planos → 0 resultados | se mapea al alias canónico |
| 5 | filtrar solo por `bo:Aprobada` cuando la pregunta habla de aprobación en general | infravalora el conteo (una proposición aprobada con enmienda SÍ salió adelante) | se amplía a `IN (bo:Aprobada, bo:AprobadaConEnmienda)` |
| 6 | nombres de propiedad inventados por analogía (`bo:interviene`, `bo:firmadaPor`, `bo:presentadaEn` + `YEAR()`) | propiedades que no existen → 0 | se mapean a las reales; el grafo solo tiene `bo:anio` (entero) y `bo:fecha` (string) |
| 7 | alcalde buscado como `rdfs:label "Alcalde de Bilbao"` o clase `bo:Alcalde` | no existe ese nodo | → `?x bo:esAlcalde true` |
| 7b | pregunta sobre un concejal ("quién votó...") sin mencionar grupo, pero SPARQL con predicado de GRUPO (`bo:votoAFavorDe`) | el LLM confunde el predicado de grupo con el nominal (nombres casi iguales) | → `bo:concejalVotoAFavor` |
| 8 | `CONTAINS(LCASE(?x), "zara")` | casa "Zaragoza", "Zarautz", "Zarandoa" | → `REGEX(STR(?x), "\bzara\b", "i")` (solo términos de una palabra) |
| 9 | `REGEX(..., "díez", "i")` | rdflib REGEX con flag "i" ignora mayúsculas pero NO tildes → "díez" no casa "Diez" | se expanden las vocales del patrón a clases `[aá]` |
| 10 | ratio mal construido: dos variables `?p`/`?r` declaradas por separado con `a bo:Proposicion`, sin vincular, en vez de `OPTIONAL+BIND` sobre la misma variable | el subconjunto (`?r`) deja de estar filtrado por las condiciones de `?p` (p.ej. el grupo) y cuenta de más en silencio — verificado en vivo: "movilidad de EH Bildu rechazadas" daba 74 (todas las rechazadas de movilidad) en vez de 15 (solo las de EH Bildu) | si comparten al menos una condición, se pliega la segunda declaración en `OPTIONAL { ?p <condiciones extra> . BIND(?p AS ?r) }` |

### 3.2 Limpieza del texto generado

- El LLM copia el escape de llaves `{{ }}` de los ejemplos del prompt (son `str.format`) → se normaliza a `{ }`.
- El LLM inventa líneas `PREFIX` con namespaces falsos (`PREFIX bo: <http://example.org/...>` → 0 resultados en silencio) → se quitan todas; rdflib ya tiene los prefijos del grafo.
- El LLM añade prosa después de la consulta ("Esta consulta filtra...") pese a pedirle "solo SPARQL" → se corta contando profundidad de llaves (no `rfind('}')`, porque la prosa cita trozos de la consulta con sus `}`).

### 3.3 El prompt (SCHEMA) estaba sobreajustado — aplicado el cambio

El SCHEMA de producción tenía ~310 líneas / 18,5k caracteres (esquema + listas de
URIs + 20 reglas + 10 ejemplos fijos). El estudio de ablación
(`ESTUDIO_SPARQL_LOCAL.md`) demostró que una versión compacta (~5,6k caracteres,
1/3 del tamaño) con few-shot dinámico (3 ejemplos elegidos por parecido a la
pregunta) + una capa de alias no destructiva **iguala o supera** al prompt
grande, con menos respuestas falsas, y que además `qwen3:8b` supera a
`qwen2.5:7b` sin coste (mismo hardware, 79%→87-94% según cómo se eligen los
ejemplos).

Aplicado a producción en `graph_rag_sparql.py`: `LLM_MODEL_GRAPHRAG` pasa a
`qwen3:8b`; `SCHEMA` pasa a la versión compacta; los 3 ejemplos del few-shot se
eligen por embedding (`bge-m3`, local) cuando el modelo es de la familia
qwen3, y por solape de palabras si no (el estudio encontró que el embedding
solo ayuda con modelos de esa capacidad o mejor); se añade `_alias_rewrite`
(la capa de alias completa del estudio: entidad/persona/grupo inventados como
URI → `label+REGEX`) antes de `_sanitize_sparql`.

### 3.4 `mistral:7b` y `gemma3:4b` probados como alternativa a Qwen para SPARQL

A petición del usuario ("¿mistral no seria mejor?", "¿gemma no es mejor que
qwen?"), se corrieron contra el gold set de 35 preguntas (arm `F_compact`,
metodología más ligera, reps=1 — ver `ESTUDIO_SPARQL_LOCAL.md` Fase 6):

- `mistral:7b`: 66% ok, 0% respuestas falsas pero 23% de errores de sintaxis
  SPARQL — la familia Qwen está más afinada en salida estructurada.
- `gemma3:4b`: 40% ok, 20% respuestas falsas (la tasa más alta de todos los
  modelos locales del estudio) — y en la prueba de narración (§2.5) alucinó
  copiando un ejemplo few-shot ficticio como si fuera un dato real.

Ninguno desplaza a `qwen3:8b`. Confirmado con datos, no solo con intuición.

### 3.5 Fuentes con página exacta en GraphRAG (2026-09-12)

`_fuentes_graphrag` (frontend/app.py) ya enganchaba PDFs por fecha desde antes
de esta sesión, pero solo a la primera página del acta (sin `#page=N`) y solo
si la consulta SPARQL generada por el LLM devolvía un campo de fecha/título con
uno de unos pocos nombres fijos (`fecha`/`fechaProp`/... — frágil, dependía de
cómo nombrara sus variables el LLM esa vez). Comprobado que el grafo tiene
mejor materia prima sin explotar: **el 100% de las 3422 proposiciones** tienen
`bo:fecha`, `bo:pagina`, `bo:fuentePdf` (ruta relativa exacta) y
`bo:tituloTopic` — verificado con SPARQL directo (3422/3422) y con una cita
real contrastada contra el PDF (pág. 44 del acta 28-03-2012 = proposición 14
del PP sobre plusvalía municipal, exacto).

**Cuándo hay una proposición individual que citar**: para preguntas de LISTADO
("qué proposiciones...", "lista las...") el LLM sí vincula `?p` (la URI de la
proposición) en el `SELECT` — verificado con una pregunta real. `graph_sources()`
(nuevo, `graphrag/graph_rag_sparql.py`) detecta cualquier valor de fila que
case con el patrón de URI `br:prop_<hash>`, y con UNA sola consulta por lotes
(`VALUES`) recupera fecha/página/PDF/título para todas las que aparezcan —
igual de preciso que las citas del RAG vectorial (enlace directo a
`/acta/<año>/<fichero>#page=N`).

**Cuándo NO la hay**: preguntas de AGREGADO puro (COUNT/GROUP BY que nunca
devuelven `?p`, la mayoría de las preguntas típicas de GraphRAG — "¿cuántas
proposiciones...", "¿qué grupo ha presentado más...") no tienen ninguna
proposición individual en las filas — es una limitación ESTRUCTURAL de qué se
puede citar desde un número agregado, no un fallo de `graph_sources()`.
Antes esto se traducía en silencio total (sin sección de fuentes). Ahora
`_fuentes_graphrag` explica por qué: *"esta cifra se calcula directamente
sobre el grafo estructurado... al ser un resultado agregado, no corresponde a
un documento concreto que enlazar"* — más honesto que el silencio, y explica
por qué GraphRAG a veces no muestra fuentes y el RAG vectorial casi siempre sí
(diferencia real de arquitectura, no un descuido).

Se mantiene como respaldo la heurística antigua por fecha (sin página) para
filas que traen fecha suelta pero no URI de proposición.

---

## 4. Construcción del grafo

### 4.1 Contexto por proposición para el enriquecimiento LLM

Cuánto texto de proposición pasarle al LLM de enriquecimiento:
- Modelos locales pequeños (Ollama) truncan JSON con textos largos.
- Gemini a 20.000 caracteres = punto dulce coste/calidad (la longitud mediana de
  una proposición es ~19k; con esto casi ninguna se trunca).
- El texto para la capa determinista de voto se parsea del texto COMPLETO sin
  recortar (`extract_proposals.py`), porque la votación va al final de la
  proposición.

### 4.2 json_repair como fallback

Modelos locales pequeños a veces truncan una cadena de texto sin cerrarla dentro
de listas largas (proposiciones con muchas entidades) — JSON casi válido, roto en
un punto. Reproducido con `temperature=0` (determinista, siempre el mismo fallo).
Sin este fallback, la proposición entera se descartaba por un carácter. `json_repair`
reconstruye el resto de la estructura.

Distinción importante: `data=None` (irrecuperable) vs `data={}` (válido pero
vacío, se escribe con valores por defecto). Esa decisión le corresponde a quien
llama (`enrich()`), no a la función de parseo.

### 4.3 Vocabulario controlado de temas

La lista de temas en `build_graph.py` debe coincidir con los `prefLabel` de nivel
1 de `themes_skos.ttl`. `build_rdf.py` normaliza acentos al emparejar, así que un
desajuste no rompería nada, pero mantenerlos iguales evita confundir a quien
audite ambos ficheros.

### 4.4 Roll-up temático en el grafo

Igual principio que el del lado vectorial (§1.2): una pregunta general sobre un
tema debe incluir sus subtemas. En el grafo se materializa con el razonador
OWL-RL en `bo:trataTemaAmplio` (cierre transitivo de `skos:broader`).

### 4.5 Bug de clasificación LLM: "decae" mal etiquetado como "rechazada"/"aprobada con enmienda"

Detectado verificando contra el PDF original (no contra el grafo, que puede
tener sus propios errores) un caso donde el grafo decía `bo:Rechazada` y el
acta decía literalmente "se acepta la enmienda... por lo que **decae** la
proposición" (= `bo:Decae`, una categoría distinta en la ontología). Auditoría
sobre las 3.422 proposiciones: de 1.607 con `vote_result` (texto literal del
acta, extraído por regex, no por LLM) parseable, **912 (57%) no coinciden**
con lo que el LLM (Gemini, en `build_graph.py`) clasificó como `resultado`. El
patrón más grande, con diferencia: 344 casos donde el acta dice literalmente
"se aprueba/acepta la enmienda..., por lo que decae la proposición..." (la
definición exacta de "decae" que se le dio al LLM en el prompt) y el LLM
clasificó "aprobada con enmienda" en su lugar.

**Corrección sin re-enriquecer** (sin volver a llamar al LLM): se construyó un
clasificador determinista que deriva `resultado` directamente del texto
`vote_result` ya extraído (mira qué verbo — acepta/rechaza — actúa sobre qué
objeto — "la enmienda" o "la proposición" — y si aparece la palabra "decae" en
cualquier punto, que es inequívoca). Si el texto es ambiguo (p.ej. "se rechaza
la enmienda" sin mención posterior de la proposición), el clasificador no
fuerza nada y se conserva el valor del LLM — se prioriza no introducir un
error nuevo sobre corregir el 100% de los casos.

Efecto colateral encontrado de paso: `result_re.search()` en
`backend/rag.py::_process_single_pdf` cogía la PRIMERA frase de resultado del
segmento, mientras que las cifras de voto (`vote_re`) ya usaban la ÚLTIMA
(`votes[-1]`) — si una proposición tiene voto de la enmienda y luego voto de
la propuesta, texto y cifras podían venir de votaciones distintas. Se cambió a
`finditer()[-1]` para que ambos usen el mismo criterio (el resultado
conclusivo, el último).

La corrección se aplica editando `proposals_enriched.jsonl` (campo
`resultado`) y `bilbao_reasoned.ttl` (triple `bo:tieneResultado`) directamente,
sin volver a ejecutar el enriquecimiento con LLM ni el razonador OWL-RL
completo — `tieneResultado` no tiene triples derivados aguas abajo que
dependan de su valor exacto (el voto por grupo/nominal se deriva de las listas
nominales del acta, §6, no de esta clasificación).

---

## 5. Capa de concejales y partidos

### 5.1 Fuente del reparto concejal → partido

Las 4 actas de sesión constitutiva (11-06-2011, 13-06-2015, 15-06-2019,
17-06-2023) listan todos los concejales electos agrupados por partido. Las de
2011/2015/2019 se parsean limpiamente (una columna, cuadran 29 escaños). La de
2023 es bilingüe a dos columnas y mezcla nombre↔partido; se reconstruyó desde el
bloque Gobierno (PNV+PSE) vs oposición de un pleno posterior.

### 5.2 Deduplicación de nombres

Los nombres de concejal aparecen con muchas variantes (OCR con espacios sueltos,
euskera vs castellano, apodos, tratamiento a media cadena). La deduplicación usa:
plegado fonético de apellidos (tx→ch, tz→z, v→b...), solape de nombres de pila
(para no fusionar hermanos), regla de prefijo para nombres truncados, y un mapa
explícito de alias eu/es. Resultado: 111 → 94 concejales, 86 con partido (los 8
sin partido son del mandato 2007-2011, sin acta constitutiva en el corpus).

---

## 6. Capa determinista de voto

Las actas escriben el voto nominal literal: "Votos afirmativos: 14
señoras/señores: Madrazo, Sustatxa, Alcalde, …, Ajuria y Abaunza." (etiqueta
euskera "jaun-andre", conector "eta"). `votos_parse.py` parsea esas listas.

Cobertura: 1463/3422 proposiciones (42%) tienen lista nominal, 1387 cuadran con
el recuento declarado (95%). Eso da **el 40% de las proposiciones con voto
verificado palabra por palabra** (`bo:votoFuente "acta"`), frente al ~9% que
cubría el LLM. `build_rdf.py` resuelve apellido → concejal → grupo y emite tanto
el voto nominal (`bo:concejalVoto*`) como el voto por grupo derivado
(`bo:votoAFavorDe`...).
