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
- `_extraer_keywords_literales` (para la búsqueda literal contra el texto CRUDO
  del PDF): conserva la grafía original. La primera versión usaba
  `where_document {"$contains": ...}`, que es substring literal sensible a
  tildes y mayúsculas, y había que lanzar una consulta por cada variante
  (con/sin tilde, singular/plural, capitalizada). Desde 2026-09-27 se usa
  `$regex` con una sola expresión por keyword que ya ignora mayúsculas y
  tildes y cubre singular/plural (ver §1.10).
- ~~Mínimo 6 caracteres~~ (sustituido en §1.10): el mínimo pretendía evitar
  falsos positivos por raíces cortas, pero descartaba siglas y nombres cortos
  con mucho significado (LGTBI, OPE, taxi, Gaza). Ahora el falso positivo se
  evita con límite de palabra (`\b`) en la regex, no con la longitud.

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

### 1.10 Selección de keywords por rareza en el corpus, no por longitud (2026-09-27)

Encontrado gracias a la evaluación RAGAS de 50 preguntas: dos preguntas
sacaban Context Relevance = 0 ("patinetes eléctricos" y "colectivo LGTBI") y
el diagnóstico mostró que no era un fallo de embeddings sino de CÓMO el canal
literal elegía qué buscar. El fix de §1.9 garantizaba representación justa
entre las keywords elegidas, pero no tocaba el criterio de elección
("las 3 más largas de ≥6 letras"), que fallaba de tres formas:

1. **Siglas y palabras cortas descartadas.** "LGTBI" (5 letras) nunca se
   buscaba pese a aparecer en 283 fragmentos de 31 plenos; se buscaban
   "iniciativas", "colectivo" y "derechos" (miles de fragmentos, 0 relevantes
   entre los 22 que pasaban el corte). En las 50 preguntas RAGAS, 21 perdían
   alguna palabra con contenido: OPE, taxi, Gaza, toros, tala, Tour, salud...
   Además, aunque se hubiera admitido, `$contains` con la variante en
   minúscula ("lgtbi") da 0 coincidencias: el texto va en mayúsculas.
2. **Empates resueltos por el hash-seed.** "patinetes", "vehículos" y
   "movilidad" tienen 9 letras; cuál entraba en el top-3 dependía del
   `PYTHONHASHSEED` del proceso. Medido: en 3 arranques, "patinetes" quedó
   fuera en 1 (0 fragmentos relevantes) y dentro en 2 (7 relevantes). En la
   generación del dataset RAGAS tocó el arranque malo. 5/50 preguntas tenían
   un empate en el corte.
3. **Palabras del enunciado ocupando puestos.** "debatido" estaba en el top-3
   de 12 de las 50 preguntas; "iniciativas", "medidas", "debatieron"… traen
   fragmentos al azar.

La longitud era un sustituto pobre de lo que se quería medir: lo distintiva
que es una palabra. **Fix** en `backend/rag.py`:

- `_extraer_keywords_literales`: candidatas de ≥4 letras + siglas de 2-3
  MAYÚSCULAS; nueva lista `_STOPWORDS_ENUNCIADO` (verbos y marcos de pregunta).
- `_regex_literal`: una regex por keyword para `where_document {"$regex"}` —
  `(?i)` (mayúsculas), clases `[aá]`, `[eé]`... (tildes), alternativas
  singular/plural, y `\b` a ambos lados (sin él "tala" coincide dentro de
  "instalación": 500+ coincidencias frente a 72 reales).
- `_keyword_search`: cuenta cuántos fragmentos contiene cada candidata (en
  paralelo, `ThreadPoolExecutor`; cada `$regex` recorre la colección, ~0,4 s)
  y elige las 3 MÁS RARAS, descartando las que no aparecen nunca. Desempate
  determinista (longitud, luego orden alfabético): la misma pregunta
  recupera siempre lo mismo.

Verificado: "LGTBI" pasa de 0 a 8 fragmentos relevantes en el canal literal y
"patinetes" entra siempre (8); Tubacex (§1.9) y desahucios siguen
funcionando; regresión vectorial 13/13 sin reintentos. Efecto colateral
positivo: el canal literal pasa de ~6 s a 0,7–2,6 s por pregunta (antes eran
~20 consultas `$contains` por pregunta, una por variante de grafía; ahora
una de conteo por candidata en paralelo + 3 de búsqueda).

**Lección**: la evaluación con métricas sobre un banco de preguntas amplio
encuentra fallos que la regresión dirigida no ve — los 13 casos de regresión
pasaban todos con el criterio antiguo porque ninguno dependía de una sigla o
de un empate.

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

**Límite de salida (`num_predict=4096`, 2026-09-27).** El timeout NO cubre un
segundo caso, encontrado durante la generación del dataset RAGAS: qwen2.5:7b
entró en un bucle de repetición (el mismo bloque "…(repetido)" una y otra vez)
y generó 52.610 tokens durante 50 minutos en una sola respuesta. El timeout de
lectura nunca salta porque Ollama sí sigue enviando tokens; el proceso Python
aparentaba estar colgado (0 % de CPU, esperando) cuando en realidad era la
generación la que no terminaba — el registro de Ollama (`server.log`) lo mostró
(`POST /api/chat … 50m22s`, `n_decoded = 52610`, `truncated = 1`). Groq ya
tenía `max_tokens=8192` y GraphRAG `num_predict=8192`; el vectorial local era
el único sin límite. La respuesta legítima más larga medida en las 50
preguntas ronda 1.700 tokens, así que 4096 deja margen amplio y acota el peor
caso a ~4 min (a ~17 tokens/s en la RTX 2060). Con el límite, esa misma
pregunta termina y `dedup_answer_blocks` (§2.6) colapsa los bloques repetidos;
la respuesta sigue siendo mala (es un fallo del modelo pequeño), pero ya no
bloquea la aplicación.

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

Se probaron como alternativa a `qwen2.5:7b` para la
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
propuestas.** La limitación de arriba SÍ se abordó: `find_item_anchors()` (hoy en
`backend/fuentes.py`) segmenta la respuesta por cada línea "Título: ..." (acepta
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

Faltaba validar la información contra los
PDF, que es lo más importante: hasta este punto la Ronda 36 solo había
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
(nuevo, hoy en `backend/fuentes.py`) sustituye el CONTENIDO de la línea "- Votos: ..." de
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

Para comprobar si Mistral o Gemma eran mejor opción que
Qwen, se corrieron contra el gold set de 35 preguntas (arm `F_compact`,
metodología más ligera, reps=1 — ver `ESTUDIO_SPARQL_LOCAL.md` Fase 6):

- `mistral:7b`: 66% ok, 0% respuestas falsas pero 23% de errores de sintaxis
  SPARQL — la familia Qwen está más afinada en salida estructurada.
- `gemma3:4b`: 40% ok, 20% respuestas falsas (la tasa más alta de todos los
  modelos locales del estudio) — y en la prueba de narración (§2.5) alucinó
  copiando un ejemplo few-shot ficticio como si fuera un dato real.

Ninguno desplaza a `qwen3:8b`. Confirmado con datos, no solo con intuición.

### 3.5 Fuentes con página exacta en GraphRAG (2026-09-12)

`fuentes_graphrag` (hoy en backend/fuentes.py) ya enganchaba PDFs por fecha desde antes
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
`fuentes_graphrag` explica por qué: *"esta cifra se calcula directamente
sobre el grafo estructurado... al ser un resultado agregado, no corresponde a
un documento concreto que enlazar"* — más honesto que el silencio, y explica
por qué GraphRAG a veces no muestra fuentes y el RAG vectorial casi siempre sí
(diferencia real de arquitectura, no un descuido).

Se mantiene como respaldo la heurística antigua por fecha (sin página) para
filas que traen fecha suelta pero no URI de proposición.

---

### 3.6 Límite de tiempo por consulta SPARQL (2026-09-28)

rdflib no tiene límite de tiempo: una consulta mal generada con variables sin
enlazar (p.ej. `?p a bo:Proposicion ; OPTIONAL { ?t skos:prefLabel ?lab }`,
generada por llama3.2 en el estudio) cruza las 3.422 proposiciones con todas
las etiquetas y estuvo más de 10 minutos sin terminar. En el chat, esa
pregunta se quedaba colgada.

Medido sobre 300 consultas correctas del estudio y las 35 de referencia: la
más lenta tarda 2,8 s (mediana 0,02 s). Límite: **20 s**. Como rdflib no
permite cancelar una consulta, `_ejecutar` la lanza en un hilo y, si se pasa
del límite, le inyecta una excepción (`PyThreadState_SetAsyncExc`); rdflib es
Python puro y la recibe en su siguiente instrucción, así que el hilo muere de
verdad y no sigue consumiendo CPU. El corte cuenta como un error más: entra
en los reintentos con un mensaje que indica la causa probable (producto
cartesiano), y en la prueba el reintento dio la respuesta correcta.

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

---

## 7. Evaluación independiente (21 preguntas)

Todas las cifras de acierto anteriores (p.ej. el 94 % del estudio SPARQL) se
midieron sobre preguntas que se usaron para desarrollar el sistema. Para medir
la generalización, el autor redactó 21 preguntas nuevas sin ver el código, el
banco de ejemplos ni las plantillas, y fijó la respuesta de referencia y el
criterio de calificación antes de ejecutar nada
(el fichero de preguntas de cada medida, que solo se conserva en local). Cubren recuentos, rangos de
años, votaciones, "la última vez", temas sin categoría propia, preguntas con
premisa falsa y preguntas cuyo dato no existe en el corpus. Cada pregunta pasa
por los dos sistemas tal como funcionan en Chainlit (`ejecutar.py`), con la
huella sha256 del código guardada para demostrar que no cambió durante la
ejecución. La calificación es manual (C correcta, P parcial, I incorrecta) y
distingue tres clases de error: **falsa** (afirma un dato falso sin avisar),
**honesta** (no responde pero reconoce que no puede) y **muestra** (calcula
una cifra sobre los fragmentos recuperados avisando de ello).

### 7.1 Antes de corregir (2026-09-28)

| | C | P | I falsa | I honesta | I muestra |
|---|---|---|---|---|---|
| GraphRAG | 9 | 0 | **9** | 3 | – |
| RAG vectorial | 6 | 2 | 4 | 3 | 6 |

El problema principal de GraphRAG no era acertar poco, sino **equivocarse con
seguridad**: 9 de 21 respuestas daban una cifra falsa narrada como cierta.
La causa estaba sobre todo en arm_g (vía determinista con plantillas): resolvió
11 preguntas y acertó 3. En el conjunto de desarrollo tenía un 96 % de
precisión, así que estaba **sobreajustado** a la redacción de esas preguntas:
tomaba "Qué"/"Cuándo" como nombre de entidad, solo el primer año de un rango,
no extraía temas como "violencia machista", y aceptaba cualquier resultado con
filas (un `COUNT` siempre devuelve una). El narrador empeoraba el fallo: tenía
la regla "la relación ya está garantizada por la consulta" y contaba con
seguridad consultas equivocadas.

El vectorial fallaba en lo esperable (recuentos, que solo puede hacer sobre la
muestra recuperada) y en "la última vez": la búsqueda literal cogía siempre
fragmentos de 2007-2008 y la recuperación no tenía en cuenta la fecha.

### 7.2 Correcciones

- **Cobertura de la consulta** (`graphrag/analisis_pregunta.py`): todo lo que
  menciona la pregunta (tema, grupo, rango de años, resultado, "la última",
  unanimidad, oposición, meses) tiene que aparecer en el SPARQL. Probado sobre
  las 2.307 consultas correctas del estudio: no rechaza ninguna.
- **arm_g solo responde si la pregunta está a su alcance** y su plantilla la
  cubre entera; si no, pasa al LLM.
- **Reintento con motivo**: si la consulta no cubre la pregunta, usa
  propiedades que no existen (`bo:proponeMonto`) o filtra por variables sin
  enlazar, el LLM lo vuelve a intentar sabiendo qué falla.
- **Fechas utilizables** en el grafo (`bo:fechaISO`, `bo:mes`) para "la
  última vez", meses y estaciones.
- **Respuestas fijas** para importes y asistencia, que el grafo no guarda.
- **Narrador** sin la regla "confía en la consulta", que explica "Decae" y
  avisa cuando falta un filtro.
- **Vectorial**: búsqueda literal por frases propias y repartida en el tiempo,
  prioridad a lo reciente en "la última vez", estaciones y meses, aviso de
  usar GraphRAG en recuentos y regla de que una enmienda citada no es un
  acuerdo aprobado.

### 7.3 Después de corregir (2026-09-29)

Misma ejecución, mismos criterios (`resultados_despues.json`,
`calificaciones_despues.json`). Regresión tras los cambios: GraphRAG 22/22,
vectorial 13/13 (no se ha roto nada de lo que ya funcionaba).

| | C | P | I falsa | I honesta | I muestra |
|---|---|---|---|---|---|
| GraphRAG | 13 | 2 | **2** | 4 | – |
| RAG vectorial | 8 | 3 | 4 | 0 | 6 |

> **Esta cifra está inflada.** Las correcciones se diseñaron analizando los
> fallos de estas mismas 21 preguntas, así que el "después" ya no es una
> medida independiente: mide que los fallos concretos se han corregido, no
> cuánto generaliza el sistema. Para una medida limpia hace falta un conjunto
> nuevo de preguntas que nadie haya visto. El dato defendible como
> generalización es el de 7.1.

Lo más relevante es el cambio en el **tipo** de error de GraphRAG: las
respuestas falsas bajan de 9 a 2, y tres de las que antes eran falsas
(i09, i10, i12) ahora reconocen que no han encontrado el dato. arm_g ya solo
responde 3 preguntas (i03, i18, i19) y acierta las 3; i20 e i21 van por las
respuestas fijas.

**Fallos nuevos o que persisten** (útiles para la memoria, porque muestran los
límites del enfoque):

- **i04, regresión** (antes C): el LLM relaciona los grupos con el Equipo de
  Gobierno mediante `bo:perteneceA`, obtiene 0 y lo narra como "el equipo de
  gobierno no presentó ninguna". La comprobación de cobertura no lo detecta,
  porque todo lo que pide la pregunta está en la consulta; lo que falla es la
  relación. La cobertura detecta lo que *falta* en el SPARQL, no lo que está
  *mal*.
- **i13**: la consulta reutiliza las variables de la primera subconsulta en
  los `COALESCE`, así que atribuye a 2026 el tema y la cifra de 2008. Mismo
  límite que i04.
- **i07**: la consulta es razonable, pero devuelve 44 filas largas y Groq
  rechaza la narración por tamaño (error 413, límite por minuto del plan
  gratuito); la narra entonces el modelo local, que entra en un bucle
  repetitivo de 26.000 caracteres. **Pendiente**: recortar las filas que se
  pasan al narrador.
- **i06**: los dos "ejemplos" son la misma proposición repetida por tema, y
  aún aparecen fragmentos de URI en la tabla.
- **Vectorial, i16**: la recuperación ya trae fragmentos de 2025-2026 (antes
  solo hasta 2016), pero el modelo sigue concluyendo que la última mención es
  de 2016. La recuperación mejoró; la síntesis no.
- **Vectorial, i13**: con solo fragmentos de 2008, afirma de qué se habla
  "ahora" (antes reconocía que no lo sabía).
- **Vectorial, i06 e i09**: parecía que la conclusión quedaba cortada a media
  frase. No era el modelo: `insertar_fuentes` tomaba una fecha escrita en
  prosa dentro de la conclusión ("26 de junio de 2014") como el inicio de un
  bloque e insertaba ahí el enlace a la fuente, partiendo la frase.

**Nota sobre la ejecución.** La cuota diaria de Groq del plan gratuito
(200.000 tokens) se agotó dos veces durante la re-ejecución. `ejecutar.py`
guarda cada paso de forma atómica y, si Groq agota la cuota, descarta el paso
en vez de guardar la respuesta del modelo local de respaldo, que no sería
comparable; se reanuda con el mismo comando. Los últimos tres pasos y la
regresión se hicieron con una segunda clave de Groq (mismo modelo,
`openai/gpt-oss-120b`), sin cambiar el código: la huella es la misma.

### 7.4 Segunda tanda de mejoras, generales (2026-09-29)

Antes de medir con un conjunto nuevo de 15 preguntas, se corrigieron las
causas **generales** de los fallos que quedaban, no los casos concretos. Cada
cambio se validó con las regresiones (35/35) y, cuando era un filtro nuevo,
contra las 2.307 consultas correctas del estudio.

- **Temas que en realidad son sinónimos.** "zona de bajas emisiones" y
  "violencia machista" no son la etiqueta principal (`skos:prefLabel`) de
  ningún tema, sino sinónimos (`altLabel`) de `t_calidad_aire` y del tema de
  violencia. Una consulta que filtra por la etiqueta principal da 0 aunque haya
  proposiciones. Ahora, antes de ejecutar, se comprueba el patrón contra las
  etiquetas reales del grafo; si no casa, el LLM recibe el motivo y el tema del
  que es sinónimo, y la sugerencia de buscar en el título o las menciones. El
  guard anterior (12) no lo detectaba cuando la consulta encadenaba más
  predicados con `;`. Sobre las 2.307 consultas del estudio solo marca una,
  que hoy devuelve 0 (busca "\bdesahucio\b" y la etiqueta es "desahucios").
- **Asunto libre de la pregunta.** La cobertura solo reconocía temas del
  vocabulario: "¿cuántas sobre la Fórmula 1 aprobó el Pleno en 2018?" la
  resolvía arm_g con todas las aprobadas de 2018 (40). Ahora se extrae el
  asunto ("sobre X", "relacionado con X", "mencionado a X") y, si no es un
  tema ni un grupo, la consulta tiene que buscar alguna de sus palabras. Con
  ese cambio responde 0 buscando en el título. Sobre las consultas del
  estudio marca 6, y las 6 son realmente incorrectas (3 ignoran a EH Bildu y
  3 buscan "zara" en vez de "Mercadona"; el estudio las dio por buenas porque
  la cifra coincidía por casualidad).
- **Un recuento de 0 ya no se acepta a la primera.** Un `COUNT` siempre
  devuelve una fila, así que un 0 no activaba el reintento; casi siempre es
  una relación mal planteada (i04). Ahora se pide otra consulta; si el LLM
  devuelve la misma o la nueva tampoco encuentra nada, se responde con el 0. El mensaje de reintento pone como ejemplo de relación mal planteada un paso intermedio entre la proposición y su grupo, que es lo que falló en i04: es la única pista que sale directamente de un caso de las 21 preguntas.
- **Narrador: tamaño, URIs y filas repetidas.** Las filas y la consulta que
  recibe el narrador tienen un tamaño máximo (7.000 y 2.500 caracteres), y se
  indica cuántas filas se han omitido; así el prompt no supera el límite de
  8.000 tokens por minuto de Groq (i07). Las proposiciones se muestran con
  fecha y título, y las demás URIs con su etiqueta. Las filas de una misma
  proposición se fusionan (con un OPTIONAL de temas salía una por tema y el
  narrador las contaba como ejemplos distintos, i06). Si la narración entra
  en bucle, se eliminan las líneas repetidas.
- **Vectorial, enlaces a las fuentes.** Solo abren bloque las fechas de
  cabecera, no las escritas en prosa, y la conclusión se reconoce también con
  el marcado del LLM ("### Conclusión", "**Conclusión**").
- **Vectorial, preguntas temporales.** Con "la última vez" se le indica al
  modelo cuál es el fragmento más reciente del contexto y que revise primero
  los más nuevos. Con "ahora" o "actualmente", si no hay ningún pleno de 2025
  o posterior en el contexto, se le pide que lo diga en vez de describir la
  situación actual.

No se ha corregido el fallo de i13 (variables reutilizadas en los `COALESCE`
de dos subconsultas): es un error de la consulta concreta y no hay una
comprobación general barata que lo detecte.

**Congelación.** Tras estos cambios se congela el código con
`ejecutar.py --huella` (`congelado.json`). Las 15 preguntas nuevas se
redactan después, sin que el desarrollador las vea antes de congelar, y el
script se niega a ejecutar si el código ha cambiado desde entonces.

### 7.5 Segunda evaluación independiente: 15 preguntas nuevas (2026-09-29)

Las 15 preguntas se redactaron aparte, sin acceso al código ni a las 21
anteriores, y se recibieron **después** de congelar el código
(`congelado.json`). Una de ellas (coste del Mercado de la Ribera) repetía el
asunto de i20 y se sustituyó por otra de importes (luces de Navidad de 2020).
Las referencias se calcularon antes de ejecutar, con consultas SPARQL escritas
a mano, búsqueda literal en el texto de las actas y, cuando el fragmento no
bastaba, el PDF (`preguntas_nuevas.json`). Siete de las 15 resultaron tener
una premisa falsa o parcialmente falsa (la moción del carril bici de la Gran
Vía, el skate park de Miribilla, el ruido de las gaviotas, la okupación en la
calle Zabala, el teleférico al Pagasarri, la ordenanza "de animales
potencialmente peligrosos", la opinión de Podemos sobre Ledesma) y dos piden
datos que no existen (importe de las luces de Navidad de 2020; el 15-01-2015
no hubo pleno). Se ejecutó una sola vez (`resultados_nuevas.json`) y se
calificó con los criterios fijados (`calificaciones_nuevas.json`).

| | C | P | I falsa | I honesta | I muestra |
|---|---|---|---|---|---|
| GraphRAG | 8 | 0 | **4** | 3 | – |
| RAG vectorial | 8 | 1 | 2 | 1 | 3 |

Separando por tipo de pregunta:

| | Respondibles (6) | Premisa falsa / sin dato (9) |
|---|---|---|
| GraphRAG | 3 C (2 por suerte), 2 falsas, 1 honesta | 6 C (1 por suerte), 2 falsas, 2 honestas |
| Vectorial | 2 C (1 por suerte), 1 P, 3 muestra | 6 C, 2 falsas, 1 honesta |

"Por suerte" marca las C en las que la consulta no podía encontrar nada (p.ej.
una etiqueta de tema inventada) y el 0 coincide con la referencia.

**Lectura.** Esta es la medida de generalización más limpia del proyecto, y
es modesta: con preguntas realmente nuevas, GraphRAG acierta 8 de 15 y da
una respuesta falsa en 4 (27 %), frente a 9 falsas de 21 (43 %) en la primera
evaluación independiente. Las mejoras redujeron las respuestas falsas, pero
**no resolvieron las preguntas respondibles con asuntos que no son temas del
vocabulario**: en 5 de las 6 el LLM local no llegó a una consulta correcta en
tres intentos. Los dos sistemas son razonablemente buenos diciendo "no
consta" cuando algo no existe (6 de 9 cada uno), que en un sistema de consulta
de actas es la propiedad más importante.

**Fallos nuevos detectados** (se corrigen después de esta medición; la cifra
de arriba es la del código congelado):

1. **Comparación de fechas como texto.** `bo:fechaISO` es `xsd:date`, y
   `FILTER(?d >= "2023-06-15")` (sin tipo) da falso para todas las filas sin
   ningún error: cualquier rango de fechas devuelve 0. Se detectó al calcular
   las referencias; explica también i15 de la primera evaluación, que se había
   atribuido a otra causa.
2. **Ordenar por `bo:fecha`.** Es texto DD-MM-AAAA, así que `ORDER BY
   DESC(?fecha)` ordena alfabéticamente y "31-01-2019" sale como la fecha más
   reciente (n07).
3. **"Ausentes" no activa la respuesta fija de asistencia** (el detector solo
   reconocía "ausencia", "faltado"...), y la consulta del LLM cuenta
   concejales sin voto en una fecha sin pleno: "0 ausentes" (n15).
4. **Relaciones inventadas pero existentes.** En n06 el LLM encadena
   `bo:enPleno` y `bo:perteneceA` para expresar "rechazada por el equipo de
   gobierno": los términos existen, la consulta es válida y da 0. El reintento
   por recuento 0 se hizo, pero el modelo local no encontró la consulta
   correcta. En n12 toma una proposición de peatonalización de otra calle y la
   narra como si fuera de Ledesma: la cobertura no exige el nombre propio
   ("Ledesma") porque la pregunta no lo introduce con "sobre".
5. **Vectorial:** acepta premisas falsas cuando recupera algo parecido (n04:
   presenta un pleno de 2014 como el de la moción de la Gran Vía; n11:
   atribuye a la calle Zabala un debate general sobre okupación).

**Correcciones posteriores a la medición** (el código ya no coincide con
`congelado.json`; para cualquier evaluación futura hay que volver a congelar):
guard 16 (`_g_fecha_iso_tipada`: tipa como `xsd:date` los literales que se
comparan con `bo:fechaISO`; un rango de ejemplo pasa de 0 a 193 filas),
guard 17 (`_g_orden_fecha_texto`: `ORDER BY ?fecha` sobre `bo:fecha` ordena
por AAAAMMDD; en n07 la fecha más reciente pasa de 31-01-2019 a 26-06-2025) y
"ausente(s)" en el detector de asistencia. Sobre las consultas correctas del
estudio, las dos guardas solo modifican una, que sigue ejecutándose.

### 7.6 Tercera tanda: asuntos fuera del vocabulario y Gemini (2026-09-29)

Tras la segunda evaluación, los fallos que quedaban tenían dos causas
generales: el LLM local no sabe dónde buscar un asunto que no es un tema del
vocabulario, y el vectorial acepta premisas falsas. Estas correcciones se
diseñaron conociendo las 15 preguntas, así que **cualquier mejora medida sobre
ellas estará inflada**; se validaron solo con las regresiones, las consultas
del estudio y preguntas de prueba distintas.

- **Pista de asunto.** Si la pregunta trata de algo que no es un tema
  (arbolado, "Ledesma", el Guggenheim...), antes de llamar al LLM se busca en
  un índice del grafo (título, subtemas con sus sinónimos y entidades
  mencionadas de cada proposición) y se le da el bloque SPARQL ya comprobado,
  con el número de proposiciones que encuentra; si son 0, se le dice que no
  consta. Se exigen todas las palabras específicas del asunto, y las muy
  frecuentes (más del 3 % de las proposiciones: "urbano" casa "urbanismo") se
  descartan. Un número pegado a una palabra va con ella ("Fórmula 1"; antes
  "fórmula" sola encontraba 50 proposiciones de otras fórmulas).
- **El asunto se extrae palabra a palabra.** Antes, si una palabra del asunto
  era un tema, se descartaba entero: en "la peatonalización de la calle
  Ledesma" se perdía "Ledesma". Ahora se quitan solo las palabras cubiertas
  por un tema o un grupo, el asunto sigue tras "de" y acaba en el año, el
  pleno, el grupo o el verbo. Sobre las consultas correctas del estudio sigue
  marcando solo las 6 que son realmente incorrectas.
- **Gemini como segunda opinión para el SPARQL** (Vertex AI,
  `gemini-3.5-flash-lite`, región global). En el estudio, con el mismo
  prompt que usa producción, acertó 102 de 105 (qwen3:8b, 99 de 105). Si el
  modelo local no llega a una consulta válida, completa y con resultados, se
  repite el proceso con Gemini y se usa su consulta si esa sí lo consigue.
- **Gemini como respaldo de la narración y del vectorial**, antes que el
  modelo local: cuando Groq agota la cuota, el modelo local respondía peor y a
  veces entraba en bucle.
- **Vectorial, premisas falsas.** Regla nueva en los dos prompts: si la
  pregunta da por hecho algo concreto que ningún fragmento recoge
  explícitamente, hay que decir que no consta y no presentar otro fragmento
  parecido (otra calle, otro año, un debate general) como si fuera ese.

### 7.7 Tercera evaluación independiente: 15 preguntas más (2026-09-29)

Mismo protocolo que en 7.5: preguntas redactadas aparte, sin acceso al código,
recibidas después de re-congelar el código tras la tercera tanda (7.6);
referencias y criterios fijados antes de ejecutar (`preguntas_tercer.json`);
una sola ejecución (`resultados_tercer.json`, `calificaciones_tercer.json`).
Groq agotó la cuota diaria a mitad y, como hace el sistema desde 7.6, siguió
Gemini (se registra qué modelo respondió cada paso).

| | C | P | I falsa | I honesta | I muestra |
|---|---|---|---|---|---|
| GraphRAG | 4 (1 por suerte) | 2 | **7** | 2 | – |
| RAG vectorial | 8 | 3 | **0** | 1 | 3 |

**El vectorial mejoró** respecto a 7.5 (0 respuestas falsas frente a 2): la
regla de premisas falsas funciona; en las 8 trampas de este conjunto dice
que no consta en 7, y en la octava cita algo relacionado sin afirmar lo que
no hay.

**GraphRAG empeoró** (7 falsas frente a 4), y la causa principal fue una de
las mejoras de 7.6: la **segunda opinión de Gemini se aceptaba cuando la
consulta local daba 0 filas**, bajo la idea de que "devolver filas" indicaba
una consulta mejor. Se comprobó re-generando las consultas solo con el modelo
local: en t04 y t06 daban 0 filas (el sistema habría respondido "no se
encontraron datos"), y Gemini las sustituyó por consultas laxas que unían
las condiciones con OR ("ruido" O "Casco Viejo" → una guía de artistas
callejeros de 2026) y sí devolvían filas. Devolver filas no es evidencia de
que la consulta sea correcta. Otros fallos: el asunto no se detectaba cuando
la pregunta no usa "sobre" ("se ha debatido la ampliación del Metro hasta
Basauri"), "la legislatura" sin años no se filtraba, y un COUNT sobre una
subconsulta con LIMIT 10 se narró como "10 propuestas".

Lectura conjunta de las tres evaluaciones independientes (cada una medida
con el código congelado antes de ver las preguntas):

| | Respuestas falsas de GraphRAG | Respuestas falsas del vectorial |
|---|---|---|
| 1.ª (21 preguntas) | 9 (43 %) | 4 (19 %) |
| 2.ª (15) | 4 (27 %) | 2 (13 %) |
| 3.ª (15) | 7 (47 %) | 0 (0 %) |

El vectorial mejora de forma sostenida porque sus correcciones son reglas
de redacción que no dependen de acertar una consulta. GraphRAG oscila: cada
tanda corrige los fallos vistos, pero las preguntas nuevas encuentran otros,
y una corrección bien intencionada (7.6) introdujo un modo de fallo nuevo.
Con un LLM local de 8B generando SPARQL, la robustez ante preguntas abiertas
sigue siendo el límite del enfoque.

### 7.8 Cuarta tanda, a partir de la tercera evaluación (2026-09-29)

Diseñadas conociendo el tercer conjunto (cualquier medida posterior sobre él
estará inflada):

- **La segunda opinión de Gemini solo sustituye a la consulta local si esta
  es inválida o incompleta** (le falta algo de la pregunta), nunca porque dé
  0 filas; y la de Gemini tiene que ser válida y completa.
- **Cobertura del asunto estricta:** la consulta tiene que contener todas las
  raíces específicas del asunto (las mismas que usa la pista), y juntas: no
  unidas con OR ni en ramas UNION distintas.
- **El asunto se detecta también con "se ha debatido/tratado X"**, y sigue
  tras "hasta" salvo que venga un año ("hasta Basauri").
- **Los nombres propios de la pregunta** (con mayúscula) se exigen aunque
  sean frecuentes en el grafo (Deusto, Otxarkoaga); el filtro de frecuencia
  era para palabras genéricas como "urbano".
- **"La legislatura"/"el mandato"** sin años se interpretan como la actual
  (desde 2023).
- **COUNT sobre una subconsulta con LIMIT** se rechaza como error.

Sobre las consultas correctas del estudio, las reglas nuevas siguen marcando
solo las 6 que son realmente incorrectas.

### 7.9 Consultas montadas sin LLM (2026-09-29)

**Cambio de método de evaluación.** Cada conjunto independiente se "gastaba"
al corregir sus fallos, y la tercera evaluación mostró que así se avanza a
ciegas (una mejora razonable empeoró GraphRAG). A partir de aquí: 40 preguntas
nuevas, redactadas del mismo modo; las impares (20) son de **desarrollo** (se miran y
se itera sobre ellas) y las pares (20) de **prueba** (se guardan
sin mirarlas y se ejecutan una sola vez, al final, con el código congelado).

**Consulta directa** (`graphrag/consulta_directa.py`). Casi todas las
respuestas falsas de GraphRAG venían de consultas mal escritas por el LLM. El
análisis de la pregunta ya extrae tema, asunto, grupo, años, resultado, meses,
oposición y unanimidad; con eso, para las formas más comunes (contar, qué
grupo más, desglose por grupo, qué año más, la última vez, listar) la consulta
se compone sin LLM a partir de piezas fijas. Si la pregunta pide algo que las
piezas no expresan (votaciones de un grupo, personas, porcentajes, "por qué",
dos preguntas en una, un nombre propio no reconocido...), devuelve None y
sigue arm_g y después el LLM. Va antes que arm_g.

- **Validación:** de las 35 preguntas de referencia del estudio, monta 14 y en
  las 14 la primera fila coincide con la consulta de referencia. Sobre los
  tres conjuntos independientes ya usados (solo como comprobación de la
  mecánica, no como medida) resuelve i01, i02, i03, i05, i12, n01, n02, n03,
  n06, n07, t02 con la cifra de la referencia.
- **El asunto entra como VALUES**: las proposiciones que lo contienen se
  calculan en Python con el índice (título, subtemas, menciones). Con el
  bloque UNION + REGEX dentro de una consulta mayor, rdflib tardaba más de
  20 s.
- **Tratar vs. mencionar.** Para contar y ordenar se usan las proposiciones
  que *tratan* del asunto (título o subtemas) y solo si no hay ninguna las que
  lo mencionan. Para "la última vez" y los listados se incluyen ambas con una
  columna `relacion` ("trata"/"menciona"), y el narrador tiene una regla para
  no presentar una mención como si fuera el tema de la proposición.
- **Alternativas con "o"** ("el Athletic o San Mamés"): se buscan las que
  cumplen cualquiera de ellas; dentro de cada una se exigen todas las palabras.
- **Una palabra del asunto sin coincidencias ya no se descarta**: antes la
  búsqueda se ensanchaba sin avisar ("ampliación del Bilbobus hasta
  Otxarkoaga" daba 97); ahora da 0 y la respuesta es que no consta.
- **La legislatura empieza el 15-06-2023**, no el 1 de enero (en t02 la
  diferencia era 27 frente a 19).
- **Fallo de rdflib encontrado:** `FILTER(false)` no filtra nada (devolvía
  las 3.422 proposiciones); se usa `FILTER(1 = 0)`.

**Corrección de una referencia.** Al depurar t07 apareció una propuesta
aprobada el 31-05-2018 sobre el parque de Etxebarria ("conexión del parque
mediante un elevador") que la referencia no recogía: se buscó "Etxebarria" en
el título, y el título solo dice "el Parque"; el subtema sí lo nombra. La
calificación de la tercera evaluación no cambia (ninguno de los dos sistemas
dio esa respuesta), pero la referencia de t07 era incompleta.

### 7.10 Iteración sobre el conjunto de desarrollo y respuesta guiada por el grafo (2026-09-29)

Primera pasada sobre las 20 preguntas de desarrollo (`preguntas_desarrollo.json`,
`resultados_desarrollo_v1.json`), revisada contra los datos. Los fallos de
GraphRAG y su corrección, todas generales:

- **Asunto no detectado** cuando hay palabras entre el verbo y el "de" ("¿se
  ha hablado alguna vez en el Pleno de una plaga de ratas...?" contó 160
  menciones de "Casco"). La ventana pasa a seis palabras, y el "sobre"/"de"
  inicial se limpia antes de dividir en alternativas.
- **Preguntas de sí/no** sin forma directa acababan en consultas laxas del LLM
  (colmenas: 34). Nueva forma `existe`: lista las proposiciones que tratan o
  mencionan el asunto; si no hay ninguna, no consta. En esta forma el asunto
  no se relaja nunca, porque la pregunta es si existe esa combinación.
- **Asunto demasiado estricto** ("el Orgullo LGTBI": 0, cuando 15 tratan de
  LGTBI). Si ninguna proposición contiene todas las palabras, se busca por la
  más específica (un nombre propio, o una palabra de menos del 1 % de las
  proposiciones) y se dice en una columna `criterio`. No se relaja hacia
  palabras genéricas ("instalación").
- **Palabras cortas perdidas** ("la Ría"): se admiten si son nombres propios.
- **Alternativas mal partidas**: "el traslado o la reforma del Mercado de San
  Antón" comparte el complemento entre las dos.
- **Palabras de acción** (implantación, ampliación, reforma...) no cuentan como
  asunto cuando ya hay un tema reconocido.
- **Sinónimos concretos de un tema** ("zona 30" es sinónimo de
  peatonalización; "zona de bajas emisiones", de calidad del aire): se buscan
  también como asunto; si no hay ninguna proposición con ellos, se vuelve al
  tema entero y se dice en `criterio`. Sin esto, "¿qué opinó el PP sobre la
  Zona 30?" recuperaba cualquier proposición de peatonalización.
- **Votaciones**: nueva forma `votos` ("qué grupos votaron en contra...",
  "cómo votó el PP..."), con el voto de cada grupo en las proposiciones del
  asunto; sin fila de voto = no hay voto por grupo registrado (solo unas
  1.500 de 3.422 lo tienen), y el narrador lo sabe.

**Respuesta guiada por el grafo** (`graphrag/guiado.py`). Para "¿qué se
dijo/opinó/debatió sobre X?" el grafo no tiene la respuesta (no guarda lo que
se dijo) y el RAG vectorial busca a ciegas entre casi 100.000 fragmentos. Se
combinan: la consulta directa encuentra las proposiciones del asunto con
precisión, en cualquier año, y se recupera el texto de **sus** debates, porque
cada fragmento de Chroma lleva el `prop_id` de su proposición (enlace exacto,
sin búsqueda por similitud). Si la pregunta es sobre un grupo, se priorizan los
fragmentos donde habla o se le nombra, y en esta forma el grupo no filtra por
proponente (el PP puede opinar sobre una proposición de otro). El LLM responde
solo con esos fragmentos y dice si no aparece la postura pedida. Es la idea de
fondo del TFG llevada a la práctica: el grafo para encontrar, el texto para
responder.

### 7.11 Tercera y cuarta pasada de desarrollo (2026-09-29)

La tercera pasada (código de §7.10) se revisó respuesta por respuesta contra el
grafo y el texto de las actas. Fallos generales encontrados y corregidos:

- **Palabras de acción en el asunto.** Solo se quitaban si había un tema
  reconocido. En "¿qué grupo se abstuvo en la votación sobre la subida de la
  tasa de plusvalía?" ninguna proposición contenía a la vez "subida" y
  "plusvalía" en los campos indexados, y la búsqueda se relajó hacia "subida":
  cualquier subida, con sus votos (respuesta engañosa). Ahora "subida",
  "traslado", "plan", "reforma", "cierre"... se quitan siempre que quede otra
  palabra. Con "el Plan contra la soledad no deseada" se pasa de 1 a las 5
  proposiciones reales sobre la soledad no deseada.
- **Selección en la respuesta guiada.** Se leían las 5 proposiciones más
  recientes. Ahora se ordenan por relevancia (tratan del asunto; fragmentos
  del debate que hablan del asunto y del grupo; presentadas por el grupo) y,
  dentro de cada debate, se eligen primero esos fragmentos. Las fuentes
  citadas son solo los debates que ha leído el LLM.
- **Pleno de una fecha concreta** ("¿qué se aprobó en el pleno del 15 de
  agosto de 2016?"): filtro por `bo:fechaISO`; si ese día no hay ninguna
  proposición, se dice que no consta pleno y se dan el anterior y el
  posterior (7-07-2016 y 29-09-2016). Antes: "no se encontraron datos".
- "Plan" en mayúscula contaba como nombre propio no cubierto e impedía la
  consulta directa.
- La huella de congelado no incluía `consulta_directa.py` ni `guiado.py`;
  ahora sí.

**Corrección a §7.10.** La búsqueda por el sinónimo "zona 30" no aporta nada:
"zona 30" es una etiqueta alternativa del propio tema peatonalización, así que
las 20 proposiciones que "tratan de la Zona 30" son todas las del tema. La
postura del PP sí está en las actas (31-03-2022 y 24-02-2022: propuso retirar
"Bilbao 30 km/h"), pero allí se llama "Bilbao 30" o "30 km/h" y ni el grafo ni
la búsqueda vectorial la relacionan con "Zona 30". Es un límite de vocabulario;
la respuesta es honesta ("no aparece la postura del PP en los debates
encontrados") pero no acierta. No se ha forzado con un sinónimo ad hoc para no
ajustar el sistema a una pregunta de desarrollo.

**Cuarta pasada.** Solo cambian las 6 preguntas afectadas; las otras 14 generan
la misma consulta. Calificación provisional de GraphRAG en las 20: unas 13
correctas u honestas, 4 parciales (búsqueda relajada y dicha en la respuesta,
o 2 falsos positivos de 12 en "Puerto"), 3 débiles (d06 y d14, con dos temas,
siguen escritas por el LLM; d07). Regresión: 35/35. Se para aquí la iteración
sobre desarrollo para no sobreajustar, y se congela el código para la prueba
con las 20 preguntas pares.

### 7.12 Evaluación final sobre el conjunto de prueba (2026-09-30)

**Método.** Las 40 preguntas se generaron aparte, sin acceso al código.
Las 20 impares se usaron para desarrollar (§7.10-7.11); las 20 pares se
recibieron después de congelar el código (`congelado.json`, que ya incluye
`consulta_directa.py` y `guiado.py`), se fijaron sus referencias y criterios
antes de ejecutar (`preguntas_prueba.json`, consultas a mano y texto literal,
sin pasar por el sistema) y se ejecutaron **una sola vez**
(`resultados_prueba.json`, `calificaciones_prueba.json`). Narró sobre todo
Groq; Gemini respondió cuando se agotó su cuota por minuto o por día. Es la
única medida de este trabajo que no está contaminada por el desarrollo.

El conjunto es exigente: 9 de 20 preguntas tienen premisa falsa (una moción
que no existe, un debate que no hubo), 1 pide un importe y 2 son ambiguas.

| Sistema | Correctas | Parciales | Falsas | Honestas (no sabe) | Solo muestra |
|---|---|---|---|---|---|
| GraphRAG | 13 | 1 | 3 (15 %) | 3 | — |
| RAG vectorial | 15 | 1 | 0 | 1 | 3 |

Evolución de las respuestas falsas de GraphRAG en las cuatro evaluaciones
independientes: 43 % → 27 % → 47 % → **15 %**; vectorial: 19 % → 13 % → 0 % → **0 %**.

**Qué funciona.** Las 9 preguntas con premisa falsa se responden bien en
GraphRAG (no consta / no inventa un voto) salvo la de hermanamiento; la consulta
directa responde "la última vez" (funicular, Azkuna Zentroa) y rankings
(bibliotecas). El vectorial es muy fiable en premisas falsas y "la última vez".

**Fallos de GraphRAG** (se documentan, no se corrigen sobre este conjunto):
- p07: la consulta la escribió el LLM (REGEX "joven" en el título) y encontró
  una proposición sobre ocio y botellón que se presentó como la de transporte
  para jóvenes (falsa).
- p12 y p15: relajación del asunto hacia la palabra equivocada ("extranjer" en
  vez de "hermanamiento"; "renfe" sin "soterramiento") y el narrador concluye
  "no se ha tratado", que es falso. La relajación elige la palabra menos
  frecuente, no la que define el asunto.
- p10 y p20: rankings con 0 filas ("administración municipal", "0 a 3 años"
  no casan con el índice): respuesta honesta pero sin dato.
- p13: la respuesta guiada eligió debates poco relevantes y el LLM se inventó
  una fecha ("31 de junio de 2025").

**Fallos del vectorial:** en recuentos y rankings responde sobre la muestra
recuperada (lo avisa) y acierta poco (p10, p15, p20); en p09 presenta menciones
de pasada como debates.

**Conclusión.** Con preguntas nuevas y código congelado, GraphRAG baja al 15 %
de respuestas falsas, y las que quedan vienen de la relajación del asunto y del
camino con LLM; el vectorial no da respuestas falsas pero no sirve para contar.
Los dos perfiles se complementan: el grafo para contar, ordenar y comprobar si
algo existe; el texto para el contenido de los debates.

### 7.13 Cuarto conjunto independiente: 40 preguntas más, mismo código congelado (2026-09-30)

**Método.** 40 preguntas redactadas aparte, sin acceso al código,
evitando repetir las 91 anteriores y etiquetadas por tipo (recuento/ranking,
votación, última vez, tema raro, qué se dijo, premisa falsa). Se recibieron con
el código aún congelado (el mismo de §7.12) y se ejecutaron una vez antes de
tocar nada (`preguntas_cuarto.json`, `resultados_cuarto.json`,
`calificaciones_cuarto.json`). Referencias fijadas antes, a mano sobre el grafo
y el texto de las actas; las cuatro premisas dudosas señaladas al redactarlas
se comprobaron: no hubo pleno el 25-12-2014, no hay acta de octubre de 2012 y
las actas no recogen público asistente, duración de intervenciones ni retrasos.

| Sistema | Correctas | Parciales | Falsas | Honestas (no sabe) | Solo muestra |
|---|---|---|---|---|---|
| GraphRAG | 23 | 4 | 6 (15 %) | 7 | — |
| RAG vectorial | 17 | 3 | 0 | 9 | 11 |

**Acumulado sobre 60 preguntas nuevas con el código congelado (§7.12 + §7.13):**
GraphRAG 36 correctas, 5 parciales, **9 falsas (15 %)**, 10 honestas;
vectorial 32 correctas, 4 parciales, **0 falsas**, 10 honestas, 14 solo muestra.
Que el 15 % se repita en dos conjuntos distintos (20 y 40 preguntas) da
bastante más confianza en la cifra que una sola medida.

**GraphRAG por camino** (qué construyó la consulta en estas 40):

| Camino | Correctas | Parciales | Falsas | Honestas |
|---|---|---|---|---|
| Consulta directa (sin LLM) | 11 | 4 | 1 | 6 |
| SPARQL escrito por el LLM | 11 | 0 | 5 | 1 |
| Respuesta fija (importes) | 1 | 0 | 0 | 0 |

Las falsas vienen casi todas del LLM: 5 de 17 consultas escritas por el LLM
(29 %) frente a 1 de 22 montadas sin él (5 %). Las consultas del LLM que fallan
filtran por una palabra suelta ("mayores", "noria", "euskera") y el narrador
presenta las filas como si respondieran a la pregunta (noria aprobada, el PP se
abstuvo en un plan que no existe), a veces con un aviso que no basta. La
consulta directa casi no inventa, pero se queda sin respuesta más a menudo (6
honestas): asuntos que no están en títulos ni subtemas (ADN de excrementos,
parques caninos, Gautxori), "plenos ordinarios" (el grafo cuenta
proposiciones, no sesiones) y una votación en la que tomó proposiciones que no
eran el presupuesto (única falsa de este camino).

**Otro defecto de datos a la vista:** las proposiciones duplicadas en euskera
cambian rankings (seguridad: 2025 con 36 en bruto frente a 2008 con 23 sin
copias).

**Vectorial:** sin respuestas falsas en 40; acierta premisas falsas, "la última
vez" cuando la mención es reciente y alguna votación del texto (PSE-EE en
contra de bonificar el IBI a familias numerosas, 2017), pero no sirve para
contar ni ordenar (11 respuestas solo sobre la muestra).

**Conclusión.** El camino sin LLM es el que hace fiable a GraphRAG; lo que
queda de falso sale sobre todo del LLM escribiendo SPARQL. Mejoras que
indican los datos: ampliar las formas de consulta directa (y recurrir menos al
LLM), un asunto con núcleo bien elegido (§7.12), deduplicar las proposiciones
en euskera del grafo y, cuando el grafo no encuentra nada, recurrir al texto.

### 7.14 Quinta tanda: mejoras dirigidas a las causas de las respuestas falsas (2026-09-30)

Con las 60 preguntas nuevas de §7.12-7.13 ya calificadas se vio de dónde
venían las respuestas falsas de GraphRAG: 6 de 9 de consultas SPARQL escritas
por el LLM que filtraban por una palabra suelta, casi siempre porque el
análisis por reglas no había encontrado el asunto de la pregunta (solo lo
reconocía detrás de "sobre X", "se ha debatido de X"...). Cambios, todos
generales:

1. **Núcleo del asunto marcado por el LLM** (`graphrag/nucleo.py`). El LLM no
   escribe ninguna consulta: copia de la pregunta las palabras que nombran el
   asunto concreto y dice cuáles son imprescindibles. Cada palabra se comprueba
   contra la pregunta (no puede inventar filtros) y el resultado se guarda en
   caché por pregunta (ejecuciones reproducibles). Se quitan artículos, años,
   grupos, cargos (alcalde, concejal), palabras neutras (Bilbao, Pleno) y
   verbos en infinitivo. Si ninguna proposición tiene todas las palabras, se
   relaja a las imprescindibles ("hermanamiento con alguna ciudad extranjera"
   -> hermanamiento), salvo en votaciones, que se refieren a algo concreto.
2. **El LLM ya no escribe consultas que no filtran lo pedido.** Si su consulta
   no pasa la cobertura, sus filas no se narran; y si el asunto no está en
   ninguna proposición, ni siquiera se le pide.
3. **Búsqueda literal en el texto de las actas** (`graphrag/texto.py`): si el
   grafo no tiene el asunto, se buscan sus palabras (enteras, con plural) en
   los 97.150 fragmentos, se agrupan por debate y se indica al LLM si el asunto
   está en el título del punto (la proposición trata de él) o solo en el debate
   (mención). En "la última vez", "¿existe?" y recuentos se añade además una
   nota con los debates del texto (más recientes que la última proposición).
4. **Voto por grupo desde las votaciones nominales del texto**
   (`graphrag/votos_nominales.py`): para cada proposición se toma la votación
   nominal cuyos recuentos coinciden con los del grafo (la decisiva, no la de
   una enmienda), y cada apellido se asigna a su grupo con los 84 concejales del
   grafo y su periodo de actividad; el grupo recibe el sentido de al menos el
   75 % de sus concejales reconocidos. Completa 572 proposiciones (con voto por
   grupo: 1.455 -> 2.018 de 2.730). Donde el grafo ya tenía el voto, coincide en
   el 92,6 % de 5.513 votos de grupo comparables; las diferencias son sobre todo
   "en contra" (grafo) frente a "a favor" (texto), en parte atribuibles a que
   `build_rdf` tomaba a veces la votación de una enmienda.
5. **Datos:** se quitan al cargar las 692 copias en euskera de puntos que
   también están en castellano (total 3.422 -> 2.730; cambiaban rankings), con
   su equivalencia para no perder los fragmentos enlazados a ellas; nueva forma
   "¿cuántos plenos (ordinarios)...?" (cada acta es una sesión); "el presupuesto
   de 2016" busca el del ejercicio 2016 (votado en 2015); los años fuera del
   periodo de las actas no son fechas ("Bilbao Ría 2000"); "la última vez" recibe
   las últimas proposiciones que tratan del asunto y las últimas que lo mencionan.

La regresión cambió 8 referencias por la corrección de datos (recalculadas con
consultas independientes; ver el comentario en `scripts/regression_qa.py`).

**Resultado sobre las preguntas ya vistas (medida INFLADA: se usaron para
desarrollar):** conjunto de prueba (20) 13 C / 1 P / 3 falsas / 3 honestas ->
17 C / 2 P / 0 falsas / 1 honesta; cuarto conjunto (40) 23 C / 4 P / 6 falsas /
7 honestas -> 34 C / 4 P / 1 falsa / 1 honesta. La cifra válida para la memoria
sigue siendo la de §7.12-7.13 (15 % de falsas) hasta medir con preguntas nuevas.

### 7.15 Evaluación del sistema mejorado con 30 preguntas nuevas (2026-09-30)

**Método.** 30 preguntas redactadas aparte, sin acceso al código, sin
repetir las 131 anteriores, recibidas DESPUÉS de congelar el código de §7.14
(`congelado.json`, 30-09-2026 22:18). Referencias y criterios fijados antes de
ejecutar (`preguntas_quinto.json`), una sola ejecución (`resultados_quinto.json`,
`calificaciones_quinto.json`). Es la medida limpia del efecto de §7.14.

| Sistema | Correctas | Parciales | Falsas | Honestas (no sabe) | Solo muestra |
|---|---|---|---|---|---|
| GraphRAG | 20 (67 %) | 6 | 3 (10 %) | 1 | — |
| RAG vectorial | 19 | 1 | 0 | 1 | 9 |

Comparación con el sistema anterior (60 preguntas de §7.12-7.13, también
nuevas y con código congelado): GraphRAG correctas 60 % -> 67 %, falsas
15 % -> 10 %, "no lo sé" 17 % -> 3 %; parciales 8 % -> 20 %. El vectorial se
mantiene sin respuestas falsas por tercera evaluación seguida. Con 30 preguntas
cada respuesta vale más de 3 puntos: la mejora de las falsas es modesta; la de
"no lo sé" es clara.

**Qué ha mejorado:** ya no hay respuestas falsas por consultas del LLM que
filtran una palabra suelta (antes 5 de 6); las premisas falsas se resuelven
(playa de Podemos en Deusto, 29 de febrero, dietas, resultados electorales,
8 de marzo); "la última vez" distingue la última proposición que trata del
asunto de la última mención en el texto (Doña Casilda, Museo Marítimo, Hospital
de Basurto); la respuesta guiada describe bien los debates (desahucios, Kukutza,
Bilbao La Vieja, polideportivos).

**Qué falla ahora (patrón nuevo y claro):** las preguntas que nombran dos cosas
unidas por "y" ("medio ambiente y zonas verdes", "cultura y festivales",
"contenedores y reciclaje", "el barrio de Otxarkoaga" con el tema "barrios") se
interpretan como intersección (proposiciones que tratan de las dos a la vez)
cuando el usuario pide la unión o una sola de ellas. De ahí salen 2 de las 3
falsas (Otxarkoaga: 1 en vez de 13; cultura y festivales: 1 por año en vez de
2016 con 20) y 3 parciales. La tercera falsa (remunicipalización) presenta
proposiciones de otro asunto. Otras parciales: "legislatura 2019-2023" leída
como años naturales, y dos casos en que dice que no hay proposiciones sobre el
asunto (wifi, estatuas) aunque sí las hay, porque exige todas las palabras de
la pregunta.

**Conclusión para la memoria.** Con preguntas nuevas, GraphRAG pasa del 15 % al
10 % de respuestas falsas y casi deja de responder "no lo sé", a cambio de más
respuestas parciales; el vectorial sigue en 0 % de falsas pero no sirve para
contar. El siguiente paso claro sería interpretar la coordinación "X y Y" como
unión, pero cualquier nueva cifra exigiría otro conjunto de preguntas.

### 7.16 Última tanda: coordinación "X y Y" como unión y legislaturas (2026-10-01)

Última modificación del código antes de la versión final. Ataca solo los fallos
generales que señaló §7.15; las cifras de §7.15 ya no sirven para medirla
(esas preguntas se han usado para comprobar los arreglos), así que la medida
limpia será la de un conjunto nuevo.

1. **"X y Y" son alternativas que se suman.** El núcleo del asunto (`nucleo.py`)
   separa ahora las enumeraciones con "y" igual que las de "o" ("colegios y
   escuelas públicas" -> colegios / escuelas públicas), salvo que formen un
   único nombre ("Parques y Jardines"). La caché pasa a `nucleo_v2.json`.
2. **Un tema reconocido que es una alternativa entera se suma, no filtra.**
   Antes, en "medio ambiente y zonas verdes" el tema medio ambiente se ponía
   como filtro y las zonas verdes como asunto: se contaban las de medio
   ambiente QUE ADEMÁS hablaban de zonas verdes. Ahora la consulta directa usa
   la unión de las proposiciones del tema y las del resto de alternativas, y se
   lo dice al narrador en la columna `criterio`. Con dos temas ("vivienda y
   movilidad") también se suman; antes iba al LLM.
3. **Un tema dentro de un asunto más concreto no se cruza con él.** "El barrio de
   Otxarkoaga" activaba el tema barrios, y se contaban las proposiciones del
   tema barrios que mencionaban Otxarkoaga (1 de 13). Ahora el asunto concreto
   manda. Si el tema va con un complemento común ("vivienda protegida",
   "reciclaje de residuos"), se siguen exigiendo las dos palabras: sumar el
   tema entero contaría toda la vivienda.
4. **Legislaturas.** "La legislatura 2019-2023" o "el mandato de 2015" van de
   su pleno de constitución al de la siguiente (15-06-2019 a 17-06-2023), no
   de enero a diciembre. Se corrige también el inicio de la legislatura
   actual (17-06-2023, no el 15-06-2023, que fue un pleno de la corporación
   saliente).
5. **Sesiones sin proposiciones.** Los plenos se contaban por las actas con
   algún punto en el grafo, y quedaban fuera las sesiones sin proposiciones
   (constitución, actos institucionales). Al cargar el grafo se añade un nodo
   `bo:Sesion` por acta, de las del grafo y de las del texto, con predicados
   propios que ninguna otra consulta ve.

Comprobación sobre las preguntas de §7.15 (desarrollo, no medida):

| Pregunta | Antes | Ahora | Referencia |
|---|---|---|---|
| Otxarkoaga desde 2007 | 1 | 13 | 13 |
| Cultura y festivales, año con más | 1 por año | 2016 (20) | 2016 (20) |
| Medio ambiente y zonas verdes, grupo con más | cifras sin criterio | EH Bildu 50, PP 46 (suma dicha) | unión |
| Plenos extraordinarios legislatura 2019-2023 | 24 (años naturales) | 21 | 22 (una acta no está en el grafo ni en el texto) |
| Wifi / estatuas: ¿se ha tratado? | "ninguna proposición" | 2 / 10 que tratan de ello | 2 / 7 |
| Contenedores y reciclaje | 14 | 15 | 58-59 (criterio más amplio) |

Regresión de GraphRAG: 22/22 con los mismos valores de referencia de §7.14
(incluidas "movilidad y transporte" y "tres grupos con más proposiciones sobre
medio ambiente").

### 7.17 El grafo enriquecido se guarda en el .ttl (2026-10-01)

**Problema.** Desde §7.14 GraphRAG modificaba el grafo en memoria cada vez que
lo cargaba: quitaba las copias en euskera, añadía las fechas ISO, completaba el
voto por grupo con las votaciones nominales y (§7.16) añadía las sesiones. El
grafo que se consultaba no estaba en ningún fichero: lo que se ve al abrir
`bilbao_reasoned.ttl` (o al cargarlo en un visor RDF) no era lo que respondía.

**Solución.** Esos pasos pasan a la construcción del grafo
(`construccion/enriquecer.py`, llamado por `build_rdf.py` antes del razonador)
y el resultado sustituye a `bilbao_reasoned.ttl`. GraphRAG solo lo carga. Se
comprobó antes que `build_rdf.py` reproducía el .ttl anterior triple a triple
(solo cambian 7 nodos anónimos del razonador), y después que el grafo nuevo
coincide con el que se montaba en memoria: 0 diferencias en las proposiciones
fuera de lo previsto. El .ttl anterior queda como
`bilbao_reasoned.ttl.bak_pre_enriquecido` (el estudio de modelos SPARQL se hizo
con el grafo de su fecha; para reproducirlo hay que usar esa copia).

Además de trasladar los pasos, el grafo mejora en:

1. **Un pleno por acta.** Antes había un `bo:Pleno` por fecha (dos sesiones el
   mismo día eran el mismo pleno) y solo existían los plenos con alguna
   proposición. Ahora son 232, uno por acta, con `bo:tipoSesion`
   (ordinaria/extraordinaria), `bo:fechaSesion`, `bo:actaPdf`, incluidas las
   sesiones sin proposiciones (constitución de la corporación...).
2. **Legislaturas.** `bo:Legislatura` (2003-2007 a 2023-2027), de un pleno de
   constitución al siguiente. Cada pleno está en la suya y el razonador la
   propaga a sus proposiciones con un `owl:propertyChainAxiom`
   (`bo:enPleno` ∘ `bo:enLegislatura`), igual que el roll-up de temas.
3. **Procedencia.** El voto por grupo deducido de las votaciones nominales lleva
   `bo:votoFuente "acta_nominal"` (572 proposiciones); la proposición en
   castellano guarda el id de su copia en euskera (`bo:copiaEuskera`, 692),
   porque los fragmentos del texto apuntan a ella.
4. **Limpieza del razonador.** OWL-RL materializa la igualdad reflexiva (cada
   nodo `owl:sameAs` sí mismo) y la aplica también a literales, que quedan como
   sujeto de un triple (no es RDF válido: el grafo no se podía guardar en
   N-Triples). Se quitan 43.507 triples sin información: el .ttl pasa de
   245.181 a 201.674 triples.
5. **Validación.** Formas SHACL nuevas para plenos, legislaturas y la relación
   proposición-pleno: 0 violaciones.

Los plenos y las legislaturas usan propiedades propias (`bo:fechaSesion`, no
`bo:fechaISO`): esas tienen `rdfs:domain bo:Proposicion` y el razonador habría
deducido que cada pleno es una proposición, con lo que cualquier recuento
"de proposiciones" del LLM los habría incluido.

Quedaban cuatro huecos de datos que en una primera revisión se dieron por
irrecuperables (239 proposiciones con grupo desconocido, 327 sin resultado, 12
con el resultado contrario a su recuento y 10 concejales sin grupo); el usuario
pidió revisarlos y sí había mucho que hacer: §7.18.

Regresión de GraphRAG con el .ttl de §7.17: 22/22 (congelado a las 09:22 y
descongelado para §7.18).

### 7.18 Qué es cada punto, proposiciones vecinales, votaciones y concejales (2026-10-01)

**Qué es cada punto (`bo:tipoPunto`).** Todos los puntos del orden del día
estaban en el grafo como `bo:Proposicion`, pero no todos lo son. Por el
encabezado del título (`construccion/puntos.py`): 1.951 proposiciones de grupo,
427 propuestas del gobierno, 92 proposiciones ciudadanas (vecinales), 50
enmiendas al articulado de una ordenanza, 1 declaración institucional, 29
fragmentos sin identificar, y los que **no son proposiciones**: 93 daciones de
cuenta sin votación, 51 trámites (aprobación del acta anterior, urgencia de la
convocatoria, tomas de posesión, renuncias), 32 bloques de preguntas y 4
debates sobre el estado de la ciudad. Estos últimos ya no entran en los
recuentos de la consulta directa: "aprobar por unanimidad el acta de la sesión
anterior" contaba como proposición aprobada por unanimidad. Los patrones se
buscan solo al principio del título (una proposición del PP que en su parte
dispositiva habla de "las actas" salía como trámite en la primera versión).

**Proposiciones vecinales.** Son 92 (presentadas por asociaciones, plataformas,
AMPAs, sindicatos o particulares por el mecanismo de proposición ciudadana).
Antes solo 54 tenían `bo:presentadaPorParticular` y a menudo con la persona que
firmaba "en representación de" en vez de la asociación; una estaba atribuida al
Equipo de Gobierno. Ahora 91 de 92 tienen quién la presenta, con prioridad a la
organización, las variantes del nombre reunidas en un solo nodo ("Pentsionistak
Martxan" / "Asociación Pentsionistak Martxan de Bizkaia") y sin dar nombres que
la propia fuente anonimiza ("#E.G.Z.#", "doña ______"). El sistema entiende
"proposiciones vecinales/ciudadanas" (filtro por tipo, sin tomar "vecinales"
como asunto), "¿qué asociación ha presentado más?" (ranking nuevo) y en los
listados muestra la asociación en vez de "Desconocido". En "subvenciones a
asociaciones vecinales" las asociaciones siguen siendo el asunto: el filtro
solo se activa cuando la pregunta habla de quién presenta.

**Votaciones.** (1) En 150 puntos, el recuento y el voto por grupo guardados son
los de una enmienda ("queda aceptada la enmienda del EQUIPO DE GOBIERNO (Votos
emitidos: 28 | a favor: 21, en contra: 7)"): se marca `bo:votacionRegistrada
"enmienda"` y en las preguntas de votos el narrador lo dice. Esto explica casi
todas las 12 "incoherencias" (rechazada con más votos a favor). (2) 39 puntos
sin resultado lo recuperan: por la frase de su votación ("se rechaza
proposición del Grupo Municipal GOAZEN BILBAO", "queda aprobado el Proyecto de
Presupuestos de Bilbao para el año 2023") o por un recuento completo (los votos
suman los emitidos) de un punto sin enmiendas; con enmiendas no, porque el
recuento puede ser el de una de ellas (el presupuesto de 2024 figura con 4 a
favor y 17 en contra, y se aprobó). Se descartó buscar el resultado en todo el
texto del debate: daba falsos positivos ("Retirada" en presupuestos aprobados).
Los 57 restantes sin votación en el texto se dejan sin resultado (puntos
debatidos junto con otro, texto no enlazado, proposiciones no admitidas).
(3) Dos correcciones comprobadas en el acta, una a una: el 19-09-2008 la
proposición del PSE-EE no se aprobó sino que decayó ("se aprueba la Enmienda
del Equipo de Gobierno, por lo que decae la proposición del Grupo Municipal
PSE-EE", pág. 175), y su recuento era el de la proposición nº 29 del PP; el
24-09-2020 el 6/23 de la proposición de EH Bildu es la enmienda de adición de
Elkarrekin (pág. 113).

**Concejales.** Los 10 sin grupo: 6 se asignan con las votaciones nominales del
texto (coinciden con su grupo en el 99-100 % de más de 100 votaciones y muy
por debajo con cualquier otro): Pontes y Sánchez Sequeros (PP), Sustatxa, De
Castro, Sánchez Robles y Jauregi (EAJ-PNV); Sanz Salamanca (PP) por el acta del
28-11-2007; Lahdou y García Martos figuraban como "GANEMOS", que es la
candidatura de 2015 del grupo Goazen Bilbao (acta constitutiva del 13-06-2015);
y Carlos Granados Pérez no es concejal sino el Presidente de la Junta Electoral
Central que firma las credenciales: se quita. Con los grupos completos, el voto
por grupo de las votaciones nominales alcanza 2 proposiciones más (574).
Cambios en `concejales.jsonl` y `concejales_partido.json` (copias
`.bak_pre_grupos_votos`) y en `grupos.normaliza_grupo` (GANEMOS).

SHACL: formas nuevas para `bo:tipoPunto` y `bo:votacionRegistrada`; 0
violaciones (avisos 31 -> 21, por los concejales que ya tienen grupo).

**Regresión.** 17/22 a la primera: los 5 fallos son recuentos que cambian por
lo anterior, con el mismo año o grupo en cabeza, y se comprobaron uno a uno:
total 2730 -> 2550 (180 puntos que no son proposiciones), seguridad en 2008
23 -> 22 (una resolución de Alcaldía de la que se da cuenta), rechazadas en 2021
10 -> 9 (un bloque de preguntas con el "Rechazada" de la votación vecina),
presupuestos y fiscalidad 507 -> 495 (8 daciones, 3 trámites y un debate del
estado de la ciudad) y rechazadas en 2018 46 -> 47 (la proposición de Goazen
Bilbao del 31-05-2018 que no tenía resultado: "se rechaza proposición del Grupo
Municipal GOAZEN BILBAO"). Con esas referencias actualizadas: 22/22.

**Versión final congelada** el 2026-10-01 09:53 (`congelado.json`, que incluye
`enriquecer.py`, `puntos.py`, `votos_nominales.py`, `build_rdf.py`,
`ontology.ttl`, `grupos.py`, `concejales.jsonl` y el .ttl; la congelación de la
quinta evaluación queda en `congelado_eval_quinto.json`). La medida limpia de
§7.16-7.18 será la de un conjunto de preguntas nuevo, recibido después de esta
fecha.

### 7.19 Medida definitiva: 25 preguntas nuevas con la versión final (2026-10-01)

**Método.** 25 preguntas redactadas aparte, sin acceso al código, sin
repetir las 161 anteriores, recibidas después de congelar la versión final
(09:53). Se centran en lo último que se había arreglado:
4 vecinales, 5 con "X y Y", 4 de legislaturas o plenos (11 R, 3 V, 3 U, 2 T,
3 D, 3 F). Referencias y criterios fijados antes de ejecutar
(`preguntas_sexto.json`), una sola ejecución (`resultados_sexto.json`; Groq sin
cuota diaria, casi todo con Gemini por `--permitir-respaldo`), calificación en
`calificaciones_sexto.json`.

**Referencias corregidas después de ejecutar.** Cuatro referencias (s01, s05,
s12, s24) se habían calculado con el grafo. La respuesta del vectorial a s05
citaba una iniciativa vecinal de 2024 que el grafo no tenía, y al comprobarlo en
el texto apareció un hueco de datos: **las iniciativas vecinales que se debaten
al final de los plenos ordinarios no están en el grafo** (la extracción de las
actas se detuvo antes; el 30-01-2025 el grafo llega al punto 21 y las
iniciativas van después). Solo entre marzo de 2023 y febrero de 2026 son unas
30. La última vecinal aprobada no es la de 24-02-2022 sino la de Hiritarrok del
30-10-2025 (Línea 4 de metro). Las referencias corregidas están junto a las
originales en `preguntas_sexto.json`, con la nota de que son posteriores; se
califica con las corregidas y se da también el resultado con las originales.

| Sistema | Correctas | Parciales | Falsas | Honestas (no sabe) | Solo muestra |
|---|---|---|---|---|---|
| GraphRAG | 14 (56 %) | 4 | 4 (16 %) | 3 | — |
| GraphRAG con las referencias originales | 16 (64 %) | 3 | 3 (12 %) | 3 | — |
| RAG vectorial | 8 | 3 | 1 (4 %) | 4 | 9 |

**Lectura.** Respecto a §7.15 (67 % correctas, 10 % falsas, 3 % "no lo sé")
este conjunto sale peor, pero está hecho a propósito sobre los puntos débiles y
con 25 preguntas cada una vale 4 puntos: la diferencia entre 10 % y 12-16 % de
falsas es de una o dos respuestas. Lo que se arregló en §7.16-7.18 funciona con
preguntas nuevas: "X y Y" se suma (educación PP y PSE-EE 34, Deusto y Basurto
36, limpieza y mantenimiento PP, ruido y terrazas como unión), los plenos por
legislatura (30 ordinarios en 2015-2019), las vecinales como tipo (92), las
premisas falsas (pleno del 1 de enero, Plaza Nueva, baños en el Casco Viejo,
Aste Nagusia 2019). El vectorial da su primera respuesta falsa en cuatro
evaluaciones (una alegación al plan de Zorrotzaurre como "iniciativa vecinal
aprobada").

**Fallos de GraphRAG.** Falsas: (1) el hueco de datos de las vecinales (s05);
(2) dos consultas escritas por el LLM: proposiciones que solo mencionan el
Guggenheim presentadas como "la moción para cerrarlo" (s15, el mismo tipo de
fallo que en §7.12-7.13) y un ranking de barrios cruzado con el tema barrios
(s12); (3) subvenciones sin relación presentadas como proposiciones sobre la
rehabilitación (s11). "No lo sé": dos fallos de la lectura de "legislatura"
introducida en §7.16 ("¿en qué legislatura...?" activa la regla de la
legislatura actual; "legislatura 2023-2027" incluye un año posterior al
corpus y la cobertura rechaza la consulta) y el ranking por pleno, que la
consulta directa no sabe montar.

**Pendiente (no se ha tocado: sería medir otra vez con preguntas usadas).**
Extraer las iniciativas vecinales de los plenos ordinarios (requiere volver a
extraer el final de cada acta); las dos reglas de legislatura; un ranking por
pleno en la consulta directa; y la regla, ya conocida, de no presentar como
respuesta proposiciones que solo mencionan el asunto cuando la pregunta tiene
una premisa concreta.



### 7.20 Grafo reconstruido: los puntos que faltaban y más información (2026-10-01)

**Por qué.** La medida de §7.19 destapó que al grafo le faltaban puntos del
orden del día. El grafo se construía con los fragmentos del indexador del RAG
vectorial, que solo corta en "N. PROPUESTA / PROPOSICIÓN / MOCIÓN / DICTAMEN".
No reconocía "INICIATIVA VECINAL" (el apartado "VII. Participación de
vecinos/as, asociaciones y entidades en el Pleno"), "TOMA DE CONOCIMIENTO" ni
"Se da cuenta" en muchas actas, ni las extraordinarias de punto único sin
numerar; ese texto quedaba pegado al punto anterior. Además 4 actas
extraordinarias eran escaneadas (sin capa de texto) y no estaban ni en el grafo
ni en la base vectorial.

**Cómo** (todo en `graphrag/graphrag/construccion/`, reanudable):

1. `texto_pdf.py`: texto de las 236 actas página a página, en caché.
2. `ocr_gemini.py`: las 4 escaneadas (18-02-2022, 21-03-2024, 15-07-2025,
   30-04-2026; 34 páginas) transcritas página a página con Gemini en Vertex.
   `scripts/indexar_faltantes.py` las añade a la base vectorial con el mismo
   troceo que el resto (97.150 -> 97.263 fragmentos; el RAG vectorial no cambia
   en nada más).
3. `segmentar.py`: puntos de cada acta desde el texto completo. Corta en
   cualquier encabezado de punto al principio de línea, castellano o euskera,
   formato moderno ("23. INICIATIVA VECINAL") o antiguo ("-23-" y en otra línea
   "Proposición..."), y solo si el número sigue la secuencia (las listas
   numeradas de una parte dispositiva no son puntos; un salto grande solo si
   abre como un punto: "Proposición que presenta", "Se da cuenta de"...).
4. `extraer_extra.py`: los puntos que no estaban (no se repiten los que ya
   están con otra numeración; un segmento que se ha tragado puntos del grafo no
   se usa) y las votaciones de todos los puntos. 993 puntos nuevos, enriquecidos
   con el mismo prompt que los demás (Gemini en Vertex, `build_graph.py --extra`).
5. `datos_extra.py` y `enriquecer.py`, antes del razonador:
   - **Votaciones** (`bo:Votacion`): cada "Se somete a votación ... En su
     virtud, ..." con su objeto (enmienda / el punto / por puntos), recuento,
     decisión y voto de cada grupo deducido de las listas nominales (regla del
     75 %); `bo:esDecisiva` en la última. 1.970 votaciones. El narrador las
     recibe en las preguntas de votos y distingue la votación de una enmienda
     de la del punto.
   - **Barrio y distrito** (`bo:enBarrio`, `bo:enDistrito` por cadena de
     propiedades): los barrios que nombran el título, el resumen o las entidades,
     con su distrito oficial (8). 1.368 puntos. Los nombres que también son
     palabras o personas ("merece la pena", "doña Begoña", "Jon Zabala") se
     buscan con mayúscula y sin apellido detrás.
   - **Importes y beneficiarios** (`bo:importe`, `bo:beneficiario`): el de "por
     importe de ..." del título. 174 puntos. Ya no se responde siempre "no
     guardo importes": se dan los del asunto, avisando de que son los que
     aprobó el Pleno y no el gasto total.
   - **Fecha de presentación** de las iniciativas vecinales.
   - **Procedencia** (`bo:origenExtraccion "segmentar"`) de los 993 recuperados.

**`bo:PuntoOrdenDia`.** Con los puntos nuevos, la consulta del LLM de "tasa de
aprobación" pasó de 518 a 640 proposiciones de EH Bildu: contaba los bloques
de preguntas y las daciones de cuenta. Solución de fondo: clase general
`bo:PuntoOrdenDia` (dominio de todas las propiedades de los puntos) y
`bo:Proposicion` subclase suya; las preguntas, daciones de cuenta, trámites,
debates del estado de la ciudad y entradas de la memoria de iniciativas ya no
son `bo:Proposicion`, así que cualquier consulta con `?p a bo:Proposicion`
(también las del LLM, cuyos ejemplos la usan) los deja fuera. 3.723 puntos, de
los que 2.562 son proposiciones.

**Dos hallazgos al revisar los datos.**
- *La memoria de iniciativas.* Las "Proposición de fecha X presentada por ..."
  de los plenos de febrero (2017-2023), y los bloques equivalentes de
  26-09-2024 y 26-03-2026, no son debates: son las entradas del informe anual
  de la Comisión Especial de Sugerencias y Reclamaciones, que repite cada
  iniciativa ya debatida con lo que pasó ("fue tratada en el Pleno de 28 de
  enero de 2021 ... aprobada por unanimidad ... se notificó"). El grafo las
  tenía como proposiciones de la fecha del informe (Belaunaldi Galdua figuraba
  aprobada el 24-02-2022; se aprobó el 28-01-2021) y contaban dos veces. Ahora
  son `informe_iniciativa` (115). Los debates reales son las "INICIATIVA
  VECINAL presentada por ..." (106 recuperadas, 2015-2026). La última aprobada
  es la de Hiritarrok del 30-10-2025 (Línea 4 de metro), la que se le escapaba
  a s05.
- *Las daciones de cuenta no se votan.* En §7.18 una dación con resultado se
  tomaba por propuesta; pero ese resultado era del LLM, o del punto siguiente
  que el troceo antiguo no separó (una "Se da cuenta de la resolución de
  Alcaldía" que designaba representantes salía "aprobada" del PP). Ahora una
  dación es siempre dación, y a un punto que no es proposición no se le asignan
  votaciones (332 descartadas por eso).

**Lo que el sistema sabe responder ahora** (consulta directa nueva): "¿qué
barrio / distrito ...?", "¿cuál ha sido el pleno con más ...?", "¿en qué
legislatura ...?", "¿cuál fue el primer pleno de ...?", importes de
subvenciones y créditos, y las votaciones de cada punto. Se corrigen además los
dos fallos de "legislatura" de §7.19 ("¿en qué legislatura...?" activaba la
actual; "2023-2027" se recortaba a 2026 y la cobertura rechazaba la consulta).

**Comprobación.** SHACL con formas nuevas (votaciones, barrios, importes, tipo
de punto para todo `bo:PuntoOrdenDia`): 0 violaciones. Regresión de GraphRAG
22/22 tras recalcular 11 referencias, cada cambio comprobado punto a punto
contra el grafo anterior (`regression_qa.py` lo detalla): salen las entradas de
la memoria y las daciones, entran los puntos reales que faltaban; ningún
ranking cambia. Copias del grafo anterior: `bilbao_reasoned.ttl.bak_pre_puntos_extra`
(§7.18) y `.bak_pre_enriquecido` (§7.16).

**Límites que quedan.** Algunos debates de iniciativas vecinales de 2016-2021
siguen sin recuperar (sus encabezados no se distinguen en el texto: por
ejemplo, el de Belaunaldi Galdua del 28-01-2021), aunque su entrada en la
memoria sí está; el barrio de un punto es el que nombra el texto, no una
geolocalización; los importes son los del título del punto.

Regresión del RAG vectorial con las 4 actas nuevas: 13/13. **Versión final
congelada** (`congelado.json`, que incluye ya `segmentar.py`, `extraer_extra.py`,
`datos_extra.py`, `build_graph.py` y el .ttl nuevo; la de la sexta evaluación
queda en `congelado_eval_sexto.json`). La medida limpia de §7.20 será la de un
conjunto de preguntas nuevo, recibido después de esta congelación.

## 7.21 Séptima medida (grafo reconstruido, 24 preguntas nuevas)

Conjunto de 24 preguntas redactadas aparte y recibidas después de congelar la versión final (`congelado.json`, 2026-10-01 13:53). Referencias y criterios fijados antes de ejecutar (grafo y texto de las actas) y sin cambios después. Una sola ejecución con `--permitir-respaldo`: Groq agotó la cuota diaria hacia el final y respondió Gemini. Ficheros: `preguntas_septimo.json`, `resultados_septimo.json`, `calificaciones_septimo.json`.

| | Correctas | Parciales | Incorrectas | de ellas falsas |
|---|---|---|---|---|
| GraphRAG | 12 (50 %) | 6 (25 %) | 6 (25 %) | 4 (17 %) |
| Vectorial | 5 (21 %) | 3 (13 %) | 16 (67 %) | 0 |

Las 4 falsas del GraphRAG: t11 (última mención de Errekalde: 28-11-2024 en vez de 26-03-2026), t13 (41 en vez de 6 proposiciones con enmienda), t15 (no desglosa los votos de la última vecinal rechazada) y t24 (da 0 para una legislatura fuera del corpus). Las 2 honestas: t05 y t23. Las preguntas de premisa falsa (t03, t10, t16) las resuelve bien.

Causas conocidas, sin corregir para no invalidar la medida:
- t11: el ranking «último» de un barrio solo usa las iniciativas vecinales, no los puntos con `bo:enBarrio`.
- t13: las enmiendas se asocian a demasiados puntos (215 proposiciones de grupo con alguna enmienda en 2019-2023, frente a 6 aprobadas con enmienda).
- t15: la forma de votos por grupo no se aplica a «la última vecinal rechazada».
- t06: la cifra del Equipo de Gobierno (1218) incluye enmiendas a ordenanzas y presupuestos.

El vectorial no se equivoca (0 falsas), pero responde con muestras de fragmentos en los recuentos y rankings, que su propio aviso remite al GraphRAG.

**Corrección de t08 (vivienda y alquiler).** El grafo contó 34 proposiciones aprobadas por tema y la referencia (por título) era 8. Revisadas las 34, unas 20 tratan realmente de vivienda (pisos vacíos, okupación, Surbisa, pisos turísticos, áreas de rehabilitación) y el resto no (Norma Foral del IBI, aparcamientos, planes especiales). La referencia era demasiado estrecha: la nota pasa de incorrecta a parcial, y `calificaciones_septimo.json` conserva la original. La unión «X y Y» funcionó (`t_vivienda` más `t_alquiler`); no era un fallo de esa regla ni de contar menciones.

## 7.22 Cambio de enfoque: paridad de información entre el grafo y el vectorial

El objetivo del estudio es comparar dos estructuras de datos, así que el grafo no debe tener menos información que el índice vectorial: si la tiene, la diferencia de resultados es de la estructura y no de huecos del grafo. Se prueba en los puntos (3.723 en el grafo frente a 3.652 pares fecha-punto en los fragmentos) y falta el contenido de los debates. Se aparca el perfil mixto (no se usa).

Intervenciones: `construccion/intervenciones.py` extrae de `proposals.jsonl` y `proposals_extra.jsonl` una intervención por orador y punto (17.567, unos 42 millones de caracteres; marcadores «SR./SRA. APELLIDO:» y «APELLIDO AND./JN.:» en euskera). `construccion/resumir_intervenciones.py` resume la postura de cada una con Gemini (Vertex), reanudable. `datos_extra.anadir_intervenciones` crea `bo:Intervencion` con `bo:intervencionEn`, `bo:orador` (concejal por apellido y fecha), `bo:grupoOrador`, `bo:resumenIntervencion` y `bo:posturaIntervencion`. El texto completo no se copia al grafo: queda en el acta (duplicaría el índice vectorial).

### Qué faltaba en el grafo y se ha corregido (§7.22, continuación)

Con el criterio de paridad se auditó el grafo contra el texto de las actas y salieron cuatro problemas de datos, no de consulta:

1. **Proposiciones que no estaban.** En once actas (28-01-2021 a 26-05-2022) el grafo tenía de 0 a 4 proposiciones donde el texto trae de 6 a 10. Dos formatos no los reconocía `segmentar.py`: el punto «-N-» solo en una línea con el título en euskera («…Udal Taldeak aurkezten duen proposamena»), y los puntos sin número («Proposición que presenta el Grupo…»). Un segmento de más de 80.000 caracteres con varias cabeceras se subdivide. Además `extraer_extra.py` descartaba un punto recuperado si el grafo tenía «el mismo título» ese día, y los 40 primeros caracteres de casi todas las proposiciones coinciden: ahora se compara por acta y página de inicio. Métrica de control: cabeceras de proposición en el texto frente a proposiciones del grafo; actas con hueco claro 11 -> 0.
2. **Duplicados.** 17 proposiciones constaban dos veces (punto original con la página como número, «116.», y el recuperado, «16.»). El punto original con la numeración corrupta se enlaza con el segmento que empieza en su página, solo si el «número» coincide con la página y es el único candidato.
3. **Resultados que eran del punto vecino.** `resultados_acta.py` toma la primera decisión de votación del propio punto que lo resuelve de forma explícita («por lo que decae la proposición» -> Decae; «se rechaza la proposición» -> Rechazada) y corrige 66: rechazada -> decae 11, aprobada con enmienda -> decae 12, decae -> rechazada 8, sin resultado -> decae 6... Se compara sin espacios porque los PDF traen «por l o que». No toca los «aprobada».
4. **Enmiendas sin autor.** `autores_enmienda.py` asigna 21 leyendo el acta: ordenanza de igualdad del 22-03-2018 por número de registro (GOAZEN BILBAO 55 y 65, PP 58 a 61, UDALBERRI 69 a 119, EH BILDU 121 a 154) y la del reglamento de distritos del 30-03-2017 (GOAZEN BILBAO). Se quitan 9 enmiendas colgadas de una dación de cuenta: 8 duplicadas de la proposición real ya recuperada y una (cheques de compra básica, 31-10-2013) que no está en el acta. Quedan 3 enmiendas antiguas colgadas de daciones (2008, 2009, 2011) cuya proposición no se ha localizado: límite conocido.

Verificación: rechazadas de 2021 = 15, que coincide pleno a pleno con las fórmulas de rechazo del acta (2+3+1+3+1+1+2+2); SHACL 0 violaciones y 0 avisos; regresión 35/35 (GraphRAG 22/22, vectorial 13/13) con 10 baselines recalculadas y comentadas en `scripts/regression_qa.py`.

**GraphRAG solo con el grafo.** `SOLO_GRAFO` (`GRAFO_CON_TEXTO=1` lo desactiva) quita el respaldo con el texto de las actas, las menciones del texto en «la última vez» y la lectura guiada de debates: «qué se dijo» sale de los resúmenes de las intervenciones del grafo. Los puntos sin dato dicen «no consta en el grafo».

**Cifras del grafo final:** 4.4xx puntos (2.702 proposiciones según la regresión), 18.581 intervenciones (16.467 con grupo y 13.732 con concejal identificado), 1.913 votaciones, unas 120 proposiciones más que antes en 2021-2022. Versión congelada el 2026-10-01 18:43 (`congelado.json`; la de la séptima medida queda en `congelado_eval_septimo.json`). La séptima medida (§7.21) ya no refleja este grafo: la medida limpia será con un conjunto de preguntas nuevo.

**Incidente de método al preparar la octava medida (2026-10-02).** Con las preguntas ya recibidas, al calcular la referencia de «¿qué grupos se abstuvieron en la votación de los presupuestos de 2022?» se vio que el extractor de votaciones no reconocía «Se procede a la votación nominal, siendo el cómputo…» y que los puntos sin resultado podrían resolverse por el recuento de su votación. Se implementaron los dos cambios antes de medir, que es lo que no debe hacerse: el código se congela antes de ver las preguntas. Se revirtieron enteros (la huella de `congelado.json` vuelve a coincidir). La pregunta se mide con el código congelado; si falla, es un límite medido, y el arreglo (genérico) se aplica después de la medida y se documenta como mejora posterior.

## 8. Cambios del 2026-10-04: recorte del contexto, interfaz y reorganización del código

**Recorte de bloques del contexto vectorial.** Con el reparto del límite de caracteres (§7.22, octava medida) cada bloque «[PLENO: fecha] título» se recortaba por el principio; en 47 de las 50 preguntas del RAGAS había bloques recortados (unos 1.260 caracteres de mediana) y en la de Tubacex la única frase con el nombre quedaba fuera (Context Relevance 0). Ahora, si un fragmento destacado (los 8 mejores del reordenador y los del canal literal) no entra en lo recortado, se pone delante una ventana de 600 caracteres alrededor de la palabra buscada (`vectorial/generacion.py`, `_render_bloque`). Regresión vectorial 13/13.

**Interfaz.** Se quita el perfil híbrido (enrutador por reglas): en la única medida con preguntas que no se usaron para diseñarlo (novena) obtuvo 12 correctas frente a 14 del GraphRAG solo. Queda como trabajo futuro; el código está en `_archivo/codigo_anterior/`.

**Reorganización del código, sin cambiar el comportamiento.** `backend/rag.py` (1.744 líneas) y `graphrag/graphrag/graph_rag_sparql.py` (2.480) se parten en módulos por etapa; el código se movió literal, con un script que extrae cada función con sus comentarios. Se retiran el prototipo «grafo con texto» (`GRAFO_CON_TEXTO=1`, desactivado en la versión evaluada: `texto.py`, `guiado.py` y sus ramas) y los scripts superados. Los datos del grafo pasan a `datos/grafo/` y las rutas se definen en un solo sitio (`comun/rutas.py`). Comprobaciones de que no cambia el comportamiento:

- **Comparación estática**: el árbol sintáctico de cada función vieja frente a la nueva (unas 330 idénticas); las distintas son solo los cambios buscados (retirada del prototipo, rutas a `comun/rutas.py`, `query()` de la CLI por el mismo camino que la interfaz, prompt delegado a `construir_prompt`).
- **GraphRAG**: 47 preguntas de las medidas octava y novena, reproduciendo las llamadas a los LLM grabadas antes del cambio: SPARQL, filas y respuesta idénticos en las 47, sin ninguna llamada a un LLM distinta (los prompts son los mismos).
- **Regresión** 35/35.
- **Interfaz**: arranca, carga los dos motores y sirve los PDF.
- **Construcción**: el grafo reconstruido con el `build_rdf` viejo y con el nuevo, sobre los mismos datos: 660.781 triples en los dos; solo cambian los identificadores de los nodos anónimos de la ontología y la etiqueta de 7 asociaciones vecinales, que ya variaba de una reconstrucción a otra (empate entre variantes que solo se diferencian en las mayúsculas, resuelto por el orden en que se recorre el grafo). El desempate es ahora determinista (`enriquecer.py`): la variante que no va toda en mayúsculas y después la alfabética.

La prueba destapó un fallo de la propia reorganización: al juntar `texto.corpus()` y `guiado._coleccion()` en un mismo módulo (hoy `grafo/construccion/texto_actas.py`), las dos compartían el mismo `threading.Lock` (antes cada módulo tenía el suyo) y `corpus()` se bloqueaba esperándose a sí mismo. Corregido con un cerrojo propio para `_coleccion()`.

**La recuperación vectorial no es determinista entre ejecuciones.** Al intentar la misma comparación exacta con el vectorial, dos ejecuciones del mismo código (viejo o nuevo), con las variantes de la pregunta reproducidas y la misma semilla de Python, llevan a la reordenación listas de candidatos distintas en 2 o 3 de cada 4 preguntas. La causa probable es que los *embeddings* de Ollama en la GPU no son idénticos de una llamada a otra, y eso cambia los empates de la búsqueda aproximada. Consecuencia para la evaluación: una misma pregunta puede recuperar contextos algo distintos en dos ejecuciones, otra fuente de variación además del LLM, y por eso las medidas se dan sobre conjuntos de preguntas y no pregunta a pregunta.

| Antes | Ahora (tras la reagrupación del 05-10, §8.1) |
|---|---|
| `backend/rag.py` | `vectorial/` (`indexado`, `recuperacion`, `generacion`, `pipeline`) |
| `backend/providers.py` | `comun/proveedores.py` |
| `backend/fuentes.py` | `vectorial/generacion.py` (fuentes del vectorial) y `grafo/consulta/respuesta.py` (las del GraphRAG) |
| `graphrag/graphrag/grupos.py` | `comun/grupos.py` |
| `graphrag/graphrag/graph_rag_sparql.py` | `grafo/consulta/` (`recursos`, `pregunta`, `guardas`, `generacion`, `respuesta`) |
| `graphrag/graphrag/{analisis_pregunta,nucleo}.py` | `grafo/consulta/pregunta.py` |
| `graphrag/graphrag/consulta_directa.py` | `grafo/consulta/consulta_directa.py` |
| `estudio_sparql/arm_g.py` (plantillas en producción) | `grafo/consulta/plantillas.py` |
| `graphrag/graphrag/construccion/` (22 scripts) | `grafo/construccion/` (`texto_actas`, `extraer`, `build_graph`, `debates`, `votos`, `enriquecer`, `build_rdf`, `validate_shapes`) |
| `graphrag/graphrag/*.ttl`, `*.jsonl`, `cache/` | `datos/grafo/` |
| `scripts/regression_qa.py` | `evaluacion/regresion.py` |
| `scripts/eval_independiente/` | `evaluacion/independiente/` (solo en local) |
| `scripts/eval_ragas_*.py`, `ragas_*.json` | `evaluacion/ragas/` |
| `scripts/diagnostico_recuperacion.py` | `evaluacion/diagnostico/` (solo en local) |
| `estudio_sparql/` (10 scripts) | `evaluacion/estudio_sparql/` (`gold`, `brazos`, `estudio`) |

### 8.1 Reagrupación del 2026-10-05: menos ficheros y las fuentes dentro de cada sistema

La reorganización del día 4 dejó 82 ficheros `.py`: los dos ficheros grandes quedaron partidos en 23 módulos y los 24 scripts de construcción siguieron sueltos. Demasiados para leer el proyecto de un vistazo. Se reagrupan en **38**, uno por etapa completa (los mayores, de unas 850 líneas), con secciones dentro de cada fichero. Se quitan también los `__init__.py`, que solo tenían una línea de descripción: desde Python 3.3 una carpeta se importa como paquete sin ellos.

- `vectorial/` (de 10 a 4): `indexado.py` (de los PDF a los fragmentos), `recuperacion.py` (de la pregunta a los fragmentos), `generacion.py` (de los fragmentos a la respuesta) y `pipeline.py`.
- `grafo/consulta/` (de 13 a 7): `recursos`, `pregunta`, `consulta_directa`, `plantillas`, `generacion`, `guardas` y `respuesta`.
- `grafo/construccion/` (de 24 a 8). Los pasos que antes eran un script cada uno son ahora órdenes de un mismo fichero (`python -m grafo.construccion.extraer concejales|constitutivas|proposiciones|extra`, `... debates intervenciones|resumir`).
- `evaluacion/estudio_sparql/` (de 10 a 3). Su `arm_g.py` era una copia de `grafo/consulta/plantillas.py`; se borra y el estudio usa las plantillas de producción.
- `comun/` (de 5 a 3): solo lo que usan de verdad los dos sistemas (`rutas`, `proveedores`, `grupos`). `comun/texto.py` solo lo usaba el vectorial y vuelve a `vectorial/recuperacion.py`.

**Las fuentes se generan con la respuesta.** Los enlaces a la página del PDF no los escribe el LLM (inventaría páginas): los pone el programa con los metadatos de los fragmentos o de las filas. Hasta ahora lo hacía la interfaz (`comun/fuentes.py`), y las evaluaciones repetían las mismas líneas para imitarla. Ahora cada sistema devuelve la respuesta ya completa: `vectorial.generacion.componer_respuesta` (limpieza, enlaces bajo cada bloque y aviso de perfil; la usan la interfaz, la consola y las evaluaciones) y `grafo.consulta.respuesta.graph_answer`. Desaparecen dos duplicados: el pie «FUENTES UTILIZADAS» en texto de la consola del vectorial, y el del GraphRAG, que en la interfaz salía además de los enlaces (dos bloques de fuentes en la misma respuesta). Una respuesta del GraphRAG sin filas («no consta», fuera de alcance) ya no lleva la nota «esta cifra se calcula sobre el grafo…», que no le correspondía.

La fusión se hizo con un script que mueve cada sentencia de nivel de módulo con sus comentarios, sin tocar el cuerpo de las funciones, y reescribe los imports. Ocho nombres chocaban al juntar módulos y se renombraron (`corregir` de `votos_acta` y de `resultados_acta` pasan a `corregir_recuentos` y `corregir_resultados`, por ejemplo). Comprobaciones, todas contra el código del día 4:

- **Comparación estática**: 971 piezas (funciones, clases y constantes) con el mismo árbol sintáctico; las distintas son solo los cambios buscados.
- **GraphRAG**: las 47 preguntas con las llamadas a los LLM grabadas: SPARQL, filas y respuesta idénticos, sin ninguna llamada nueva.
- **Construcción**: el grafo reconstruido antes y después, 660.781 triples y ninguna diferencia; y los pasos sin LLM (`concejales`, `constitutivas`, `intervenciones`, `extra`) dan ficheros idénticos byte a byte.
- **Estudio SPARQL**: el análisis de los 10 ficheros de resultados y la prueba de las plantillas con las 35 preguntas de referencia dan la misma salida.
- **Interfaz**: arranca, carga los dos motores y sirve los PDF.
- **Regresión** 35/35 (22 del GraphRAG y 13 del vectorial), ya con las fuentes dentro de cada sistema.
- **Fuentes**: en las 49 preguntas del GraphRAG, la respuesta sin el bloque de fuentes es idéntica a la anterior, y hay enlaces a proposiciones exactamente donde antes había pie en texto (19). En el vectorial, `componer_respuesta` da el mismo texto que la secuencia que ejecutaba la interfaz en 5 preguntas de prueba.

Las comprobaciones que dependen del límite de 20 s por consulta SPARQL hay que pasarlas con la máquina libre: con cuatro procesos a la vez, una consulta directa superó el límite y la pregunta siguió por la vía del LLM; repetida sin carga, sale igual.

### 8.2 Revisión de código del 2026-10-05 (`vectorial/`, `grafo/consulta/`, `comun/`, `frontend/`)

Revisión completa de las cuatro carpetas del sistema (unas 7.000 líneas), con una limpieza de código muerto y repeticiones y la corrección de los defectos encontrados. **Las medidas de la memoria (RAGAS, evaluaciones independientes, regresión) se hicieron antes de estos arreglos**; para cada uno se dice abajo a cuántas de las 309 preguntas medidas (las 50 del RAGAS y las 259 de las evaluaciones independientes) puede afectar.

**Limpieza (sin cambio de comportamiento).** Código muerto: `prop_canonica` y `COPIA_DE` (`recursos.py`), `last_retrieved_docs`, tres claves de las fuentes que nadie leía, un aviso inalcanzable en `_responder`, una clave y una rama inalcanzables en `consulta_directa.py` y `guardas.py`, y el filtro que devolvía `_get_temporal_filter` y nadie usaba (ahora `_fechas_pedidas`). Repeticiones: el filtro de fechas de ChromaDB (tres sitios, ahora `_filtro_fechas`), la lista de URIs de grupo (cuatro, `_uris_grupos`) y la similitud coseno (dos). Comprobado con las mismas consultas a ChromaDB en 317 preguntas, el GraphRAG 49/49 con las llamadas grabadas y la regresión 35/35.

**Defectos corregidos.**

1. **Palabras comunes tomadas por grupos políticos.** «populares» se leía como PP, «podemos» como Elkarrekin y «ciudadanos» como Ciudadanos, en el vectorial, en el análisis del GraphRAG y en las plantillas. Ahora «los populares» (no «fiestas populares»), «Podemos» que no es el verbo (no «¿qué podemos saber?») y «Ciudadanos» con mayúscula (no «la seguridad de los ciudadanos»). Además «Bildu» a secas se detecta también en el vectorial, y Aralar y VOX están en todas las tablas. En las 309 preguntas medidas **no cambia ninguna detección de grupo**: las 7 que contienen «Podemos» lo usan como el partido.
2. **La búsqueda literal no encontraba palabras con ñ** (`_regex_literal` quitaba la tilde de la ñ y no admitía la ñ del texto). Ahora una ñ escrita en la pregunta casa con «ñ» y con «n». Afecta a **11 de las 309 preguntas medidas** (ninguna de las 50 del RAGAS): dueños, Begoña, campaña, niños, cigüeñas, Doña Casilda, acompañados y baños recuperan ahora esa palabra. «año(s)» se añade a las palabras del enunciado, porque ahora casaría con todas las actas y antes no se buscaba.
3. **MultiQuery podía dejar fuera la pregunta original.** Ahora se queda con las 5 primeras líneas del LLM y la original va siempre; el canal de proposiciones usa las 3 primeras variantes y la original. Con 5 líneas o menos el resultado es idéntico al anterior, así que no cambia nada con la salida habitual del modelo (3 líneas).
4. **Sin clave de Cohere no se reordenaba nada.** Ahora, sin clave (o con la cuota agotada), se usa el reordenador local (bge-m3). Comprobado: la regresión vectorial sin `COHERE_API_KEY` pasa 13/13 con el reordenador local en las 13 preguntas. Con la clave puesta el comportamiento es el mismo que antes.
5. **El umbral de «pregunta ajena a las actas» no actuaba** (los fragmentos del canal literal que el reordenador deja fuera recibían 0,02, por encima del umbral de 0,01). Ahora esas puntuaciones de relleno no cuentan. Comprobado con Cohere: «¿Cómo se prepara una paella valenciana?» (máximo real 0,005) y «¿Cuántos viajes a Marte…?» (0,004) se rechazan; «¿Qué se debatió sobre el tranvía?» (0,93) se responde. Por la cuenta, con el código anterior las dos primeras se habrían respondido (0,02 > 0,01).
6. **Porcentajes del GraphRAG.** `_augment_ratios` ya no toma como numerador o denominador columnas de año, fecha, mes u orden. Sin cambios en las 49 preguntas del GraphRAG grabadas.
7. **Variables sin valor como texto «None».** `_sin_resultado`, las votaciones del acta y la nota de «no consta ningún pleno ese día» tratan «None» como vacío. En las 49 preguntas grabadas cambia **una** (¿Qué aprobó el Pleno el 29 de febrero de 2022?: el prompt del narrador decía «el None (anterior)» y ahora «—»), que muestra que el defecto sí aparecía en una pregunta medida.
8. **Normalización de grupos (`comun/grupos.py`).** (a) La regla de «abertzaleak» (socialistas) ya no basta sola, solo con «Sozialist»; un nombre como «Euzko Abertzaleak» se leería como PSE-EE, aunque no aparece en ningún fragmento de las actas. (b) `es_grupo_disfrazado` usa `canon_grupo` en lugar de `normaliza_grupo`, que devuelve el texto tal cual cuando no reconoce nada. (c) La regla de «Se da cuenta de…» no reconocía la numeración con guiones («-2- Se da cuenta…», formato de 2010 a 2022) y esos puntos caían en la búsqueda de grupos en el texto. Comprobado sobre los datos reales: (a) y (b) no cambian nada (0 cambios en 4.659 proposiciones, 5.157 entidades distintas y las listas de asistentes de las 236 actas); (c) cambia el grupo de **406 de 4.659 proposiciones**: 302 de «Desconocido» a «Equipo de Gobierno» y 104 de un grupo concreto (39 del PNV, 21 de EH Bildu, 14 del PSE-EE, 10 del PP…) a «Equipo de Gobierno», porque eran resoluciones de Alcaldía y acuerdos de la Junta de Gobierno atribuidos a un grupo por mencionarlo en el texto. Estos puntos no cuentan como `bo:Proposicion` en el grafo (`NO_PROPOSICION`), así que los recuentos por grupo del GraphRAG no cambiarían, pero sí `bo:presentadaPor` y el `grupo_proponente` del vectorial. **Aplicado el 2026-10-05 por la tarde:** grafo reconstruido (`build_rdf`; SHACL sin violaciones; 660.701 triples frente a 660.781; cambian 393 `bo:presentadaPor`, todos a «Equipo de Gobierno», y los nodos anónimos de la ontología) y `enrich_vector_metadata.py` vuelto a ejecutar. Corrección de lo anotado antes: **el índice vectorial no cambia** (0 campos distintos en 97.263 fragmentos). Las 393 proposiciones afectadas son todas de las recuperadas (`proposals_enriched_extra.jsonl`), que no tienen fragmentos con `prop_id` en el índice, y el script solo recorre `proposals_enriched.jsonl`. Copias previas: `bilbao_reasoned.ttl.bak_pre_grupos_0510` y `chroma_db.bak_pre_grupos_0510`.
9. **El grupo del orador se heredaba del orador anterior** (`indexado.py`). Ahora, si el orador nuevo no está en la lista de asistentes, queda «Desconocido». Comprobado con el censo de concejales en 14 actas que tienen etiquetas: de 6.026 fragmentos con orador identificable, los grupos incorrectos pasan del 71,4 % al 25,2 % (4.302 a 1.516) a cambio de perder 40 aciertos casuales (408 a 368). Solo se aplica al reindexar (hecho para las 19 actas, ver abajo).

**Grupo del orador (`party`), corregido con el censo (2026-10-05, tarde).** El campo solo existía en 18 de las 236 actas (8.675 de 97.263 fragmentos) y casi siempre decía «EQUIPO DE GOBIERNO»: el analizador de la lista de asistentes (`_get_party_mapping`) no podía funcionar, porque en el formato de 2011 en adelante la cabecera no dice el grupo de cada concejal (en el acta del 28-06-2018 cogía líneas numeradas de cualquier página, hasta «Diferencias ejecución → Equipo de Gobierno»). Ahora `_get_party_mapping(pages, date)` usa el censo `datos/grafo/concejales.jsonl`: los concejales en activo en la fecha del acta, por apellido; si un apellido lo comparten concejales de grupos distintos solo valen los que figuran en la cabecera del acta, y si sigue ambiguo el orador queda «Desconocido»; «Alcalde/Alcaldesa/Alkate» toma el grupo del alcalde de esa fecha. El grupo es siempre el **partido** (ya no hay «Equipo de Gobierno» como grupo de orador).

Reindexadas solo las 19 actas que lo tenían (las 18 con grupo más una acta de la misma fecha), con la nueva opción `python scripts/full_rebuild.py --actas DD-MM-YYYY,...`, y vuelto a ejecutar `enrich_vector_metadata.py`. Resultado: fragmentos con grupo de orador 8.675 → 9.793, «EQUIPO DE GOBIERNO» 7.197 → 0; cobertura de proposiciones enlazadas 100 %. Limitaciones: (1) el censo no incluye a todos los concejales (faltan, por ejemplo, Ibaibarriaga, Pombo, Lopategi, Barturen y Calvo), que quedan «Desconocido»: es incompleto, no erróneo; (2) el censo solo se aplica a esas 19 actas, el resto sigue sin grupo de orador (reindexar todo costaría horas y el campo apenas se usa: el filtro de grupo usa `grupo_proponente`); (3) el texto de cada fragmento lleva «ORADOR: X (grupo)», así que en los fragmentos reindexados el texto ya trae el grupo nuevo, pero en los del acta del 17-06-2011 se restauró desde la copia el texto original (el troceo actual de esa extraordinaria por página da otro título de punto y rompía el enlace con 2 proposiciones, 176 fragmentos) y solo se corrigió el metadato. El índice queda en 97.217 fragmentos (antes 97.263): los 46 de diferencia vienen de cambios del troceo del código desde la indexación original en otras actas.

**Defectos conocidos que siguen sin corregir** (cambiarían cómo responde el sistema de formas que no se pueden comprobar sin LLM, o piden una decisión):

- **`_generar_y_ejecutar` prefiere una consulta con filas pero incompleta a una completa cuyo recuento es 0**, y `_responder` descarta la incompleta: «¿Cuántas proposiciones presentó Vox en 2012?» puede pasar de «0» a «no he podido montar una consulta».
- **Dos cadenas de proveedores de LLM** (`recursos.py` repite la de `comun/proveedores.py`) y la copia crea Groq sin `reasoning_effort="low"`: al narrar puede devolver la respuesta vacía. Unificarlas cambia la configuración de generación del GraphRAG evaluado.
- **Las variantes de los nombres de grupo siguen en varias tablas** (`comun/grupos.py`, `recuperacion.py`, `pregunta.py`, `guardas.py`, `plantillas.py`); se han alineado las diferencias que importaban (puntos 1 y 8) pero no unificado.
- Menores: una guarda del SPARQL deja `?anio` sin enlazar y fuerza un reintento; la cobertura rechaza consultas con «Agenda 2030» o «Ría 2000» por «sobra un año»; `_fix_degenerate_groupby` colapsa un recuento por año legítimo; en el vectorial, un bloque sin palabras en común con sus fuentes recibe la primera fuente de esa fecha y su línea de votos; las fechas que anuncia el prompt incluyen las de bloques descartados por tamaño.

No aplicado por no compensar el riesgo: unificar los cuatro sitios donde se crea el *embedder* bge-m3, calcular una sola vez la forma de la pregunta y declarar `pista` en la clase `Analisis`.

### 8.3 «¿Cómo votó X la última proposición de Y…?» (2026-10-05, noche)

Al repasar diez preguntas de la novena evaluación por el GraphRAG actual, nueve dieron SPARQL y filas idénticos a los de la medida; la décima («¿Cómo votó el PP la última proposición de Elkarrekin Bilbao y fue aprobada o rechazada?») variaba de una ejecución a otra: ~5 de 8 veces se abstenía y ~3 de 8 contestaba con una proposición equivocada. Causas, todas previas a la revisión del día:

1. `pregunta.py` tomaba `resultado='rechazada'` como filtro aunque la pregunta dijera «aprobada o rechazada», y la guarda de cobertura exigía `bo:Rechazada`: una consulta correcta se rechazaba (abstención) y una con `FILTER IN (Aprobada, AprobadaConEnmienda, Rechazada)`, que excluye las que decaen, pasaba. Ahora, con «aprob*» y «rechaz*» a la vez, no se fija resultado. (Solo esta de 259 preguntas medidas lo contiene.) Por sí solo este cambio empeoraba el caso (8 de 8 respuestas equivocadas), porque dejaba al descubierto las otras dos causas.
2. La consulta que generaba el LLM filtraba por «el PP votó a favor» en vez de mostrar su voto y ordenaba por `DESC(?p)` (el hash de la URI, no la fecha). Nueva consulta determinista en `consulta_directa.py` (`_roles_votos`): un grupo presenta («la proposición de/del Y») y otro vota; con «la última», una sola proposición, la de fecha más reciente y, dentro del mismo pleno, la de número de punto más alto (el título empieza por el número). El voto se toma además de la votación decisiva (`sentidoDecisiva`, `objetoVotacionDecisiva`): cuando la proposición decae, el voto por grupo está en el nodo de votación de la enmienda y no en la proposición. Solo se aplica sin tema ni asunto: con asunto, el filtro por menciones del debate devolvía proposiciones que solo nombran el asunto (en «¿cómo votó el PSE-EE la moción del PP para bajar el IBI en 2023?» salía una del PP sobre bicicletas) y se narraban como si lo trataran. De las 294 preguntas medidas y de regresión, solo z18 activa la plantilla.
3. `enriquecer.py` atribuía a un grupo las «PROPUESTA de resolución de las enmiendas (o alegaciones) presentadas por los grupos municipales ELKARREKIN BILBAO, EH BILDU y PARTIDO POPULAR…»: es la propuesta del gobierno que responde a las enmiendas, y el grupo salía del primer nombre del título (la última «proposición de Elkarrekin» era en realidad el punto 12 del 26-03-2026). También las comunicaciones de Alcaldía sin votación que citan a un grupo (sustituciones de concejales). Ahora van al Equipo de Gobierno: **12 puntos** (8 de Elkarrekin, 2 de EAJ-PNV, 1 de EH Bildu, 1 de PSE-EE); la proposición de EH Bildu del 30-11-2016 con título truncado, que sí es suya, se mantiene. Grafo reconstruido (SHACL sin violaciones, 660.727 triples) y `enrich_vector_metadata.py` con la misma regla (743 fragmentos cambian de `grupo_proponente`, ningún otro campo; cobertura 100 %).

Resultado: z18 responde siempre lo mismo y coincide con la referencia (punto 25 del 26-03-2026, resultado «Decae», el PP en contra de la enmienda decisiva); las otras nueve preguntas siguen con SPARQL y filas idénticos. Regresión 35/35 tras actualizar dos valores esperados de EH Bildu (movilidad 110 → 109; tasa de aprobación 544/46 → 543/45): el punto que desaparece es la «Propuesta de acuerdo relativa a la resolución de las enmiendas… EH BILDU, PARTIDO POPULAR y ELKARREKIN BILBAO al proyecto de Ordenanza reguladora de la zona de bajas emisiones» del 28-12-2023, comprobado en el acta: es del gobierno. Copias previas: `bilbao_reasoned.ttl.bak_pre_gobierno_0510`, `chroma_db.bak_pre_gobierno_0510`. El RAGAS no se repite: la pregunta con EH Bildu no recupera el pleno afectado.

**Lo que explica los otros dos cambios de redacción (z10, z11):** en la medida original narró Gemini (cadena «gemini, groq, gemini»: Groq había agotado su cuota) y hoy narra Groq; con la misma consulta y las mismas filas, forzar Gemini devuelve la redacción de entonces.
