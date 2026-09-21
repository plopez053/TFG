# Estudio: generación de SPARQL con LLM local para GraphRAG

_Bilbao TFG · 2026-09-10 · datos en `estudio_sparql/`_

## Motivación

En la batería comparativa #3 (todo con Ollama `qwen2.5:7b` porque Groq estaba
agotado) el GraphRAG bajó a 3/10 mientras que con Groq acertaba ~8-9/10. Los
datos del grafo son los mismos: lo que se degrada es el paso **texto → SPARQL**.
Este estudio mide **cuánto** se degrada y **qué lo arregla**, con tres palancas:

1. **La estrategia de prompt** (zero-shot mínimo … mega-prompt de producción … few-shot dinámico).
2. **Una capa determinista de post-proceso** ("adaptar la ontología al modelo"):
   traducir el vocabulario que el modelo inventa de forma sistemática al real.
3. **El modelo** (qwen2.5:7b, qwen3:8b, qwen2.5:14b, y Groq gpt-oss-120b como techo).

## Método

- **Conjunto gold: 35 preguntas** con SPARQL de referencia escrita a mano y
  verificada contra el grafo (`estudio_sparql/gold.py`). Cubre 9 formas de
  consulta: conteo, total, ranking, ratio, evolución temporal, persona, voto
  nominal, entidad, coautoría.
- **Puntuación sobre las filas ejecutadas**, no sobre la narración:
  `ok` (filas correctas) · `wrong` (filas con dato equivocado — el fallo peligroso,
  se narra como hecho) · `empty` (consulta válida, 0 filas — el fallo visible) ·
  `error` (SPARQL inválida).
- **3 reps** por celda (el LLM no es determinista ni a temperatura 0). Acierto por
  forma = voto de mayoría entre reps.
- Banco de 18 ejemplos Q→SPARQL **separado del gold** para el few-shot.

### Estrategias de prompt (arms)

| Arm | Esquema | Ejemplos | Reglas | Tamaño |
|---|---|---|---|--:|
| **A_min** | mínimo (solo modelo de datos) | — | — | ~1,8k |
| **B_vocab** | mínimo + listas completas de URIs (grupos, 40 temas, subtemas) | — | — | ~3,2k |
| **C_full** | el prompt de producción actual (`graph_rag_sparql.SCHEMA`) | 8 fijos | ~25 | ~18,6k |
| **D_compact_dyn** | mínimo + URIs + 10 reglas clave | **3 dinámicos** (los más parecidos a la pregunta) | 10 | ~5,6k |
| **E_full_dyn** | producción sin los ejemplos fijos | 3 dinámicos | ~25 | ~15,5k |
| **F_compact** | = D + **capa determinista** (alias + guardas de `_sanitize_sparql`) | 3 dinámicos | 10 | ~5,6k |
| **F_full** | = E + capa determinista | 3 dinámicos | ~25 | ~15,5k |

La **capa determinista** (`estudio_sparql/arms_f.py::post_F`) reescribe, ANTES de
ejecutar, las invenciones sistemáticas medidas en la Fase 1:
- entidad/persona/grupo como URI (`br:Iberdrola`, `br:EHBildu`, `br:yolanda_diez`)
  → patrón `?x rdfs:label ?l . FILTER(REGEX(...))` o URI canónica del grupo
- tema con prefijo equivocado (`bo:t_euskera`, `bo:Euskera`) → `br:t_euskera`
- `rdfs:label "Nombre Apellido"` exacto → REGEX insensible a tildes
- más las guardas de producción (`_sanitize_sparql`: OPTIONAL no-op, año tipado,
  voto de grupo↔nominal, tildes, etc.)

---

## Fase 1 — diagnóstico: ¿qué inventa el modelo?

`qwen2.5:7b`, arms A_min / B_vocab / C_full, 35×2.

| Prompt | OK | vacío | mal | error |
|---|--:|--:|--:|--:|
| A_min (zero-shot) | 30 % | 51 % | 9 % | 10 % |
| B_vocab (+URIs) | 39 % | 29 % | **27 %** | 6 % |
| C_full (producción) | **76 %** | 11 % | 13 % | 0 % |

**Invenciones sistemáticas** (todas las corridas):

| Patrón inventado | Nº | Vocabulario real |
|---|--:|---|
| `br:Iberdrola`, `br:Zorrotzaurre`, `br:Mercadona` (entidad como URI) | 4 c/u | `?e rdfs:label ?n . FILTER(REGEX(...))` |
| `br:XabierOtxandiano`, `br:yolanda_diez` (persona como URI) | 2 c/u | ídem, por label |
| `br:EHBildu`, `br:EH_Bildu` (grupo con URI no canónica) | 3 | `br:grupo_eh_bildu` |
| `bo:t_euskera`, `bo:Euskera` (tema con prefijo `bo:`) | 3 | `br:t_euskera` |
| `?t/rdfs:label`, `YEAR(?fecha)`, `[skos:prefLabel "x"]` | varios | property-paths y funciones que el modelo asume |

**Conclusiones de la Fase 1:**
1. El prompt de producción sube el acierto de **30 % → 76 %**. No es que el 7B
   sea incapaz: es que necesita ver el vocabulario y ejemplos.
2. **Darle solo la lista de URIs (B_vocab) EMPEORA**: sube las respuestas falsas
   de 9 % a 27 %. Con la lista pero sin ejemplos, el modelo elige con seguridad
   la URI equivocada. Los ejemplos importan más que el vocabulario.
3. Las invenciones **no son nombres alternativos razonables** que quisiéramos
   adoptar en la ontología (`br:Iberdrola`, `br:EHBildu`): son el modelo
   saltándose el patrón `label + REGEX`. → **Decisión: capa de alias no
   destructiva, no renombrar la ontología.**

---

## Fase 2 — estrategias de prompt + capa determinista (`qwen2.5:7b`, 35×3)

| Arm | OK | vacío | **mal** | error | prompt | lat. |
|---|--:|--:|--:|--:|--:|--:|
| A_min | 30 % | 51 % | 10 % | 10 % | 1,8k | 5,6s |
| B_vocab | 38 % | 29 % | 28 % | 6 % | 3,2k | 5,4s |
| C_full (producción) | 76 % | 11 % | 12 % | 0 % | 18,6k | 4,8s |
| D_compact_dyn | 73 % | 18 % | **6 %** | 3 % | 5,6k | 6,6s |
| E_full_dyn | 74 % | 11 % | 10 % | 5 % | 15,5k | 7,5s |
| **F_compact** (D + capa determinista) | **79 %** | 10 % | **6 %** | 5 % | 5,6k | 6,5s |
| F_full (E + capa determinista) | 80 % | 3 % | 12 % | 5 % | 15,5k | 7,4s |

### Hallazgos

1. **`D_compact_dyn` (73 %) ≈ `C_full` (76 %) con 1/3 del prompt**, y con **la mitad
   de respuestas falsas** (6 % vs 12 %). El few-shot dinámico (3 ejemplos elegidos
   por parecido a la pregunta) sustituye eficazmente a las 250 líneas de reglas.
2. **La capa determinista añade ~6 puntos**: `F_compact` llega a **79 % OK con
   solo 6 % de respuestas falsas** — mejor que el mega-prompt de producción
   (76 %/12 %) a 1/3 del tamaño. Es la mejor configuración local.
3. `F_full` sube el OK a 80 % pero las guardas de producción a veces sobre-reescriben
   → las respuestas falsas suben a 12 %. El prompt grande + capa determinista se
   pisan. **Menos prompt + capa determinista gana.**

### Acierto por forma de consulta (`qwen2.5:7b`, voto de mayoría)

| forma | A_min | C_full | D_compact | **F_compact** |
|---|:-:|:-:|:-:|:-:|
| conteo | 5/10 | 9/10 | 8/10 | 8/10 |
| total | 1/2 | 1/2 | 1/2 | 1/2 |
| ranking | 1/5 | 5/5 | 5/5 | 5/5 |
| temporal | 1/3 | 3/3 | 3/3 | 3/3 |
| persona | 0/4 | 4/4 | 2/4 | 3/4 |
| ratio | 0/4 | 2/4 | 3/4 | 2/4 |
| **voto_nominal** | 0/3 | 1/3 | 1/3 | **2/3** |
| **entidad** | 1/3 | 1/3 | 2/3 | **3/3** |
| coautoría | 1/1 | 1/1 | 1/1 | 1/1 |

- **entidad**: la capa determinista lo lleva de 1/3 a **3/3** (reescribe la URI
  inventada a `label + REGEX`).
- **voto_nominal**: mejora de 1/3 a 2/3 (arregla la pregunta de "Yolanda Díez" con
  el guard de tildes).
- **ratio y "total simple"**: siguen flojos con cualquier configuración.

### Casos que NINGUNA configuración resuelve nunca

| id | pregunta | por qué |
|---|---|---|
| **g22** | tasa de aprobación de EH Bildu **vs** PP (comparar dos grupos) | el modelo intenta calcular el % dentro del SPARQL, con `UNION` + `BIND` mal → fila errónea. Fallo de comprensión, no de vocabulario. |
| **g27** | ranking del concejal que **más vota en contra** | el modelo confunde la dirección del predicado nominal (`?p bo:concejalVotoEnContra ?c` ↔ `?c ... ?p`). |

Estas dos necesitan un ejemplo dedicado en el few-shot o un modelo mayor — la
capa determinista no puede arreglar una consulta con la estructura equivocada.

---

## Fase 3 — barrido de modelos

C_full, D_compact_dyn, F_compact sobre qwen3:8b y qwen2.5:14b. Groq gpt-oss-120b
como techo — **el límite diario del free tier (200k tok/día) impidió la corrida
completa**; solo da para una referencia parcial.

| Modelo | C_full (18,6k) | D_compact_dyn (5,6k) | **F_compact (5,6k)** | respuestas falsas |
|---|:-:|:-:|:-:|---|
| qwen2.5:7b | 76 % | 73 % | **79 %** | 13 / **6** / **6** |
| qwen3:8b | 82 % | 85 % | **87 %** | 8 / **2** / **2** |
| qwen2.5:14b | 82 % | — ¹ | — ¹ | 4 |
| Groq gpt-oss-120b | — ² | ~90 % ³ | — ² | — |

¹ `qwen2.5:14b` corre a ~50 s/generación en esta máquina (14B parcialmente en CPU,
4 GB de VRAM). `C_full` (82 %) **no mejora a qwen3:8b (82 %)**: en este hardware el
modelo más grande no aporta, y las dos horas que costaría completar D/F no valen la
pena. **La palanca es el método, no el tamaño del modelo.**
² Groq `C_full` = 4,6k tok/petición → revienta el límite por minuto del free tier.
³ Parcial: 18/20 correctas en las preguntas que llegaron a ejecutarse antes de
agotar el límite diario, con `D_compact_dyn`.

**Observaciones Fase 3:**
- En **qwen3:8b**, `D_compact_dyn` y `F_compact` **superan** al mega-prompt de
  producción (85-87 % vs 82 %) con **1/4 de las respuestas falsas** (2 casos vs 8).
  El patrón "menos prompt + few-shot dinámico + capa determinista" se refuerza al
  subir de modelo.
- **qwen3:8b con `F_compact` (87 %) prácticamente alcanza el techo de Groq (~90 %)**,
  local y gratis. El salto grande es 7B→8B; de 8B a 14B no hay ganancia aquí.
- El techo de Groq no está tan lejos: el margen que queda (~3-13 puntos según
  modelo) son sobre todo las 2 formas de consulta duras (ratio de dos grupos,
  ranking de voto nominal), no un problema general.

---

## Conclusiones de las Fases 1-3 (2026-09-10)

1. **El GraphRAG local es tan bueno como el paso texto→SPARQL, y ese paso se
   arregla sin cambiar de modelo ni de hardware:**
   | | qwen2.5:7b | qwen3:8b |
   |---|:-:|:-:|
   | zero-shot mínimo | 30 % | — |
   | mega-prompt de producción | 76 % | 82 % |
   | **compacto + few-shot dinámico + capa determinista** | **79 %** | **87 %** |
   | Groq (techo, referencia) | ~90 % | ~90 % |

2. **Menos prompt, mejor**, si se compensa con few-shot dinámico y post-proceso
   determinista: `F_compact` (5,6k) > `C_full` (18,6k) en acierto Y en tasa de
   respuestas falsas, en los dos modelos locales. El mega-prompt de 310 líneas
   está **sobreajustado** y de hecho estorba.
3. **La lista de vocabulario sola es contraproducente** (B_vocab: 27-28 % de
   respuestas falsas) — el modelo se envalentona y elige la URI equivocada con
   seguridad. Necesita ejemplos, no listas.
4. **"Adaptar la ontología al modelo"** funciona como **capa de alias no
   destructiva** (traducir lo que el modelo inventa: `br:Iberdrola` → `label +
   REGEX`, `bo:t_euskera` → `br:t_euskera`, etc.), NO como renombrado de la
   ontología: +6 puntos en el 7B, y convierte la capa de entidades de 1/3 a 3/3.
5. **El tamaño del modelo importa menos que el método**: 7B→8B sube 8 puntos,
   8B→14B no sube nada (en este hardware). qwen3:8b + F_compact alcanza casi el
   techo de Groq, local y gratis.
6. Quedan 2 formas de consulta duras (comparar la tasa de dos grupos con un ratio;
   ranking del concejal que más vota) que ningún prompt ni la capa determinista
   arreglan — necesitan un ejemplo dedicado en el few-shot o Groq.

---

## Fase 4 — few-shot por EMBEDDING vs. por solape de palabras (2026-09-11)

Pregunta directa del usuario: la Fase 2 eligió los 3 ejemplos dinámicos del banco
por **solape de palabras (Jaccard)** — un prototipo barato. ¿Mejora si se elige
por **similitud de embeddings** de verdad? `estudio_sparql/embed_bank.py` usa
`bge-m3` (local, 1024d, ya estaba instalado en Ollama) para embeber las 18
preguntas del banco (cacheadas en `_bank_embeds.json`) y la pregunta entrante, y
elige por coseno. Arms nuevos `D_embed_dyn`/`F_embed` (`arms_embed.py`) — idénticos
a `D_compact_dyn`/`F_compact` salvo en CÓMO se eligen los 3 ejemplos.

**Primer dato, antes de generar nada**: sobre las 35 preguntas gold, el embedding
elige un trío **distinto** al de Jaccard en **35/35 casos** (`embed_bank.py` como
script suelto lo verifica). No es un cambio marginal — es un mecanismo de selección
genuinamente distinto, no una variación de la misma heurística.

### Resultado: el efecto depende del MODELO, no es universal

| Modelo | arm | por solape (Jaccard) | por embedding (bge-m3) | Δ acierto | latencia |
|---|---|:-:|:-:|:-:|---|
| qwen2.5:7b | D | 73 % (6 % mal) | 71 % (9 % mal) | −2 pts, **peor** | 6,6s → **11,2s** |
| qwen2.5:7b | F | 79 % (6 % mal) | 77 % (6 % mal) | −2 pts, empate en falsas | 6,5s → **9,4s** |
| **qwen3:8b** | D | 86 % (3 % mal) | **91 %** (3 % mal) | **+5 pts** | 11,9s → 14,8s |
| **qwen3:8b** | F | 87 % (2 % mal) | **94 %** (**1 % mal**) | **+7 pts, mitad de falsas** | 11,7s → 12,5s |
| gemini-3.5-flash-lite | D | 96 % (1 % mal) | 95 % (2 % mal) | −1 pt, empate práctico | 0,9s → 1,0s |
| gemini-3.5-flash-lite | F | 96 % (4 % mal) | **97 %** (1 % mal) | +1 pt | 0,9s → 0,9s |

**Hallazgos:**

1. **No hay un ganador universal.** En qwen3:8b el embedding gana claro (+5 a +7
   puntos, la tasa de respuestas falsas de F_embed baja a **1 %**, la mejor cifra
   de todo el estudio en un modelo local). En qwen2.5:7b el embedding pierde
   ligeramente. En Gemini es un empate estadístico en ambas direcciones. La
   hipótesis: un modelo más capaz (qwen3:8b, más reciente) sabe **aprovechar**
   un ejemplo semánticamente parecido aunque no comparta vocabulario literal con
   la pregunta; un modelo más simple (qwen2.5:7b, familia más vieja) se beneficia
   más de ver las MISMAS palabras que va a usar (grupo, año, "aprobadas"...),
   que es justo lo que Jaccard garantiza y el embedding no.
2. **Coste operativo real y no obvio en hardware local limitado**: en la GPU de
   6 GB de este proyecto, embeber con `bge-m3` (1,2 GB) mientras `qwen2.5:7b`
   (4,7 GB) está cargado obliga a Ollama a alternar los dos modelos en VRAM en
   cada pregunta → **+40-70 % de latencia** (6,6s→11,2s) sin ninguna ganancia de
   acierto. En qwen3:8b la ganancia de acierto compensa igualmente el coste. En
   **Gemini no hay coste** porque la generación no toca la GPU local — el embedding
   se computa una vez y ya. Conclusión práctica: el embedding no es gratis en
   local salvo que el modelo de generación sea lo bastante bueno para rentabilizarlo.
3. **Esto encaja con la literatura**, no es un artefacto de este proyecto: un
   estudio de generación de SPARQL few-shot sobre un grafo de aviación
   (*Context-Aware Few-Shot Learning SPARQL Query Generation*, MAKE 2025) probó
   seis estrategias de selección de ejemplos —aleatoria, por similitud, etc.— y
   encontró que **"un prompt simple + ontología + 5 ejemplos aleatorios"** superaba
   a la selección por similitud en su caso. La selección por embedding no es
   estrictamente superior al azar o al solape léxico en tareas de query generation
   con pocos ejemplos; ayuda cuando el modelo es lo bastante capaz de explotar la
   similitud semántica en vez de necesitar el andamiaje léxico literal.
4. **Decisión para producción**: usar embeddings SOLO si el modelo de generación
   es `qwen3:8b` (o mejor). Con `qwen2.5:7b` el solape de palabras sigue siendo
   la opción correcta (igual o mejor acierto, más rápido, sin tocar VRAM). Ver
   Fase 6 para la recomendación final de qué modelo usar.

---

## Fase 5 — Arm G: plantilla por embedding + slot-filling, CERO LLM (2026-09-11)

Idea explícita del usuario en la Fase 3 ("Arm G pendiente"): para la mayoría de
preguntas recurrentes (conteo/ranking/temporal/ratio simples con tema-grupo-año),
¿hace falta un LLM? Prototipo en `estudio_sparql/arm_g.py`:

1. Catálogo de ~13 "formas" de pregunta, cada una con 1-4 frases prototipo en
   castellano (p.ej. *"¿Qué grupo ha presentado más proposiciones sobre un
   tema?"*) y, cuando aplica, una función `builder(slots) -> SPARQL`.
2. Al llegar una pregunta real: se embebe con `bge-m3` y se compara por coseno
   contra los prototipos (misma mecánica que `embed_bank.py`). Si la similitud
   máxima < 0,55 → no hay plantilla, se delega a un LLM (no implementado en este
   prototipo, sería la ruta de producción).
3. Si hay match: los huecos (tema/grupo/año/resultado/entidad/dirección de voto)
   se extraen con **regex y diccionarios de alias deterministas** (reutilizando
   los mismos catálogos de `arms.py`/`arms_f.py` — grupos, temas, subtemas), y se
   monta la SPARQL con f-strings. Sin LLM en ningún punto de esta ruta.
4. Dos guardas deterministas más, descubiertas al iterar sobre los fallos:
   distinguir *"¿en qué año se rechazaron MÁS?"* (ranking temporal) de *"¿cuántas
   se rechazaron EN 2021?"* (conteo con año fijo) por la presencia de la
   interrogativa "qué año"; y distinguir *"cuántas...Y CUÁNTAS..."*/"porcentaje"
   (ratio, dos cifras) de un conteo simple con filtro de resultado — exactamente
   el mismo espíritu que las guardas de `_sanitize_sparql` en producción.

### Resultado sobre las 35 preguntas gold (una sola pasada — es determinista, no hace falta repetir)

| | n | % del total |
|---|--:|--:|
| Generó SPARQL y fue **correcta** | 27 | **77 %** |
| Generó SPARQL pero **equivocada** | 1 (g22 — comparar tasas de 2 grupos) | 3 % |
| Reconoció la forma pero **no la sabe rellenar** → delega a LLM (persona, voto nominal) | 7 | 20 % |
| No reconoció ninguna forma → delega a LLM | 0 | 0 % |

- **Precisión cuando responde: 27/28 = 96 %** — mejor que CUALQUIER configuración
  de `qwen2.5:7b` de la Fase 2-4, y comparable a `qwen3:8b` + F_embed (94 %/1 % mal).
- **Latencia: ~290 ms/pregunta** (solo la llamada de embedding local), sin coste
  de API, **sin variabilidad entre repeticiones** (no hay temperatura ni muestreo).
- El único fallo real (g22, comparar la tasa de aprobación de dos grupos en la
  misma consulta) es **la misma pregunta que ninguna configuración de `qwen2.5:7b`
  resuelve nunca** en las Fases 2-4 (ver "casos nunca resueltos") — confirma que
  es una forma de consulta genuinamente compleja, no una debilidad específica de
  este prototipo.
- Las 7 preguntas que delega (persona/voto nominal, p.ej. "¿qué concejal ha
  intervenido más?") las reconoce correctamente como fuera de su alcance — **cero
  respuestas falsas por sobreconfianza**, que es justo el fallo peligroso que
  hay que evitar (un sistema que no sabe que no sabe).

**Esto confirma que el patrón de "sistema KGQA basado en plantillas" es un
diseño con precedente académico real** (no una ocurrencia de este proyecto): la
literatura lo llama *template-based KGQA* — con la limitación conocida de que
"suelen ser poco flexibles y estar pensados para grafos de propósito general" (lo
que aquí se traduce en: cubre bien las ~10 formas de pregunta recurrentes del
dominio municipal, pero necesita fallback a LLM para lo que no está en el
catálogo — persona, voto nominal, comparaciones de dos grupos).

**Recomendación concreta**: en producción, `graph_answer()` podría intentar
primero `arm_g.answer(pregunta)`; si devuelve `sparql=None`, caer al LLM
(`F_compact`/`F_embed` según el modelo). Esto cubriría de forma instantánea y
gratuita ~la mitad de las preguntas típicas de la batería comparativa del TFG,
reservando el LLM (y su latencia/coste/variabilidad) para lo que de verdad lo
necesita.

---

## Fase 6 — barrido de modelos AMPLIADO: suelo, techo y nubes intermedias (2026-09-11)

Se añaden 4 modelos nuevos a la Fase 3: **`llama3.2:3b`** (suelo deliberadamente
pequeño, para ver dónde deja de servir cualquier prompt), **`gemini-3.5-flash-lite`**
(nube de pago, ~0,30 $/M tok entrada, sin límite diario duro), y dos modelos
Groq nuevos — **`openai/gpt-oss-20b`** (cuota TPD independiente de gpt-oss-120b,
se descubrió al agotarse la del 120b a mitad de esta sesión) y `qwen/qwen3.6-27b`
(catalogado, no evaluado por presupuesto de tiempo). Más tarde se añaden también
**`mistral:7b`** y **`gemma3:4b`**, a petición del usuario (ver subsección
dedicada más abajo, con metodología más ligera que el resto de esta fase).

### Tabla completa (mejor arm de cada modelo, salvo indicación)

| Modelo | tamaño | mejor arm local | OK % | % mal | latencia | coste |
|---|---|---|:-:|:-:|--:|---|
| **llama3.2:3b** | 3B | D_compact_dyn/F_compact | **43 %** | 20-28 % | 2,2s | gratis (local) |
| qwen2.5:7b | 7B | F_compact | 79 % | 6 % | 6,5s | gratis (local) |
| qwen2.5-coder:7b ⁴ | 7B (especializado en código) | D_compact_dyn | 85 % | 7 % | 6,9s | gratis (local) |
| qwen2.5:14b | 14B | C_full ¹ | 82 % | — | ~50s | gratis (local) |
| **qwen3:8b** | 8B | **F_embed** | **94 %** | **1 %** | 12,5s | gratis (local) |
| groq gpt-oss-20b | 20B (MoE) | D_compact_dyn ² | 86 % | 3 % | 13,7s | gratis (cuota 200k tok/día) |
| groq gpt-oss-120b | 120B (MoE) | D_compact_dyn ² ³ | ~90 % | — | 1-2s | gratis (cuota 200k tok/día) |
| **gemini-3.5-flash-lite** | ? (cerrado) | **F_embed** | **97 %** | **1 %** | **0,9s** | ~0,30-2,50 $/M tok |

¹ `qwen2.5:14b` no se re-evaluó con D/E/F en esta ronda (ya se comprobó en la
Fase 3 que no mejora a qwen3:8b y cuesta ~50s/generación en esta máquina).
² Medido con `reps=1` (para caber en la cuota diaria), más ruidoso que el resto
de la tabla (`reps=3`) — tratar como orientativo, no como cifra final.
³ La corrida de hoy para gpt-oss-120b agotó su cupo diario (200 000 tokens) a los
20 minutos — **175 de 205 filas son error de cuota, no señal real**; la cifra de
la tabla es la de la Fase 3 (18/20 correctas, corrida cuando la cuota estaba
fresca). El cupo de 200k tok/día resultó ser **por modelo, no por cuenta**:
`gpt-oss-20b` tiene su propio cupo independiente, lo que permitió evaluarlo hoy
sin esperar a que resetee el del 120b.
⁴ Ver la subsección siguiente — el resultado agregado no cuenta toda la historia.

### `qwen2.5-coder:7b`: confirma la literatura, con un matiz importante

La hipótesis del "Fundamento en la literatura" (modelos afinados a código superan
a generalistas del mismo tamaño en text-to-SQL) se confirma en AGREGADO: al mismo
tamaño (7B), `qwen2.5-coder:7b` con `D_compact_dyn` llega a **85 %/7 % mal**, seis
puntos por encima del mejor resultado de `qwen2.5:7b` genérico (79 %/6 % mal,
`F_compact`) — casi alcanza a `qwen3:8b` con solape de palabras (86 %/3 % mal) **con
un modelo más pequeño y más antiguo**. Pero el detalle por caso es más interesante
que el agregado:

- **Es el ÚNICO modelo/prompt de todo el estudio, aparte de Gemini, que resuelve
  g22** (comparar la tasa de aprobación de EH Bildu y el PP en una sola consulta)
  — pero **solo con `C_full`** (3/3 reps ok); con los prompts compactos (`D`/`F`)
  vuelve a fallar (wrong o incluso error). Es decir: para ESTE caso compositivo
  difícil, la especialización en código compensa la falta de tamaño, pero solo si
  se le da el prompt largo con las 25 reglas explícitas — el patrón "menos prompt
  + few-shot dinámico gana" de las Fases 2-4 tiene una excepción real cuando la
  pregunta exige combinar varios pasos de razonamiento SPARQL a la vez.
- A cambio, **falla sistemáticamente en `g32`** (proposiciones conjuntas entre
  grupos, el caso "gracioso" donde la respuesta correcta es 0) en las 5×3 = 15
  corridas — un caso que `qwen2.5:7b` y `qwen3:8b` resuelven bien. No es
  estrictamente mejor en todo, es mejor en promedio con un perfil de errores
  distinto.
- También falla siempre en g28 (voto nominal de una concejala con tilde en el
  nombre) en las 5 configuraciones — un hueco de cobertura específico de este
  modelo que ni la capa de alias (`F_compact`/`F_embed`) arregla.

**Conclusión**: `qwen2.5-coder:7b` es un segundo candidato local sólido —notablemente
mejor que `qwen2.5:7b` genérico al mismo tamaño y coste, confirmando la
literatura— pero no desplaza a `qwen3:8b` + `F_embed` (94 %/1 % mal) como mejor
opción local de este estudio. Si en el futuro `qwen3:8b` deja de estar disponible
o hay que bajar de tamaño, `qwen2.5-coder:7b` con `D_compact_dyn` es la alternativa
a probar primero, no `qwen2.5:7b`.

### `mistral:7b` y `gemma3:4b`: dos candidatos descartados (2026-09-11, prueba puntual)

Prueba fuera de la corrida principal, motivada por preguntas directas del usuario
("¿por qué no usamos mistral?", "¿gemma no es mejor que qwen?"). Metodología más
ligera que el resto del estudio — **reps=1** (no 3) y solo arm `F_compact` —
tratar como orientativo, no como cifra final comparable punto a punto con el
resto de la tabla.

| Modelo | tamaño | arm | OK % | vacío | mal | error | nota |
|---|---|---|:-:|:-:|:-:|:-:|---|
| mistral:7b | 7B | F_compact | 66 % (23/35) | 11 % | **0 %** | 23 % | 8/35 `ParseException` — SPARQL sintácticamente inválida, algo que qwen2.5:7b casi no comete |
| gemma3:4b | 4B | F_compact | 40 % (14/35) ⁵ | 17 % | **20 %** | 23 % | peor tasa de "mal" (respuesta segura pero incorrecta) de todos los modelos locales del estudio |

⁵ Corrida parcialmente contaminada: los primeros 7 casos (g01-g07) dieron
`ReadTimeout` de 300 s porque en ese momento Ollama tenía cargado `llama3.2` de
otra sesión corriendo en paralelo en la misma máquina — contención real de
VRAM, no fallo del modelo. Descontando esos 7 como no válidos, el resto (28
casos limpios) da 14/28 = 50 % ok, con la misma proporción alta de "mal"
(7/28 = 25 %): el problema de respuestas confiadas-pero-erróneas no se explica
por la contención.

**Por qué se descartan:**
- **mistral:7b**: por debajo de qwen2.5:7b (66 % vs 79 %), y con 8 errores de
  sintaxis SPARQL — la familia Qwen está más afinada en salida estructurada.
  Su único punto a favor es 0 % de "mal" (falla limpio, no con confianza), pero
  el volumen de errores sintácticos lo descarta igualmente.
- **gemma3:4b**: el peor de los modelos locales evaluados en este estudio junto
  con `llama3.2:3b`, con la tasa de "mal" más alta de todas (20 %, incluso
  descontando la contención). Un hallazgo aparte, en una prueba de narración
  del RAG vectorial (prompt distinto, fuera de este estudio SPARQL), refuerza
  la misma conclusión: `gemma3:4b` copió literalmente los valores de un
  ejemplo few-shot ficticio del prompt como si fueran datos reales, alucinando
  una entrada completa entre datos verdaderos. Por debajo de 7B el modelo no
  solo pierde precisión en SPARQL, pierde la capacidad de distinguir "ejemplo
  de formato" de "dato real" — un riesgo más grave que un simple error de
  sintaxis.

---

### Lo que esto confirma (y lo que añade) sobre "tamaño vs. método"

1. **Hay un suelo real por debajo del cual ningún prompt salva la papeleta**:
   `llama3.2:3b` se queda en 37-43 % con LAS MISMAS estrategias de prompt que
   llevan a `qwen2.5:7b` al 76-79 % — la Fase 3 ya decía "el método importa más
   que el tamaño", esto matiza: **por encima de cierto umbral** (~7B en esta
   familia de tareas). Por debajo, ni el mejor prompt compensa la falta de
   capacidad del modelo para seguir instrucciones estructurales complejas
   (36-38 % de sus respuestas son directamente incorrectas con seguridad, el
   doble que qwen2.5:7b).
2. **El techo real de este estudio no es Groq, es Gemini**: `gemini-3.5-flash-lite`
   con `F_embed` llega a 97 %/1 % mal — mejor que cualquier corrida de Groq
   conseguida (con o sin cuota agotada) — y a 0,30 $/M tokens de entrada, sin el
   límite diario duro que ha interrumpido 2 corridas de Groq en esta sesión. Para
   un sistema en producción con presupuesto (aunque sea mínimo), es la opción
   con mejor relación acierto/fiabilidad de cuota de todo el estudio.
3. **`qwen3:8b` + `F_embed` (94 %/1 % mal, local, gratis) está a solo 3 puntos de
   Gemini** y por delante de la referencia parcial de Groq 120B. Confirma la
   recomendación de la Fase 3 (cambiar de qwen2.5:7b a qwen3:8b) y la refina: el
   salto de qwen2.5:7b (79 %) a qwen3:8b (87 % con solape, 94 % con embedding) es
   la palanca más grande y más barata disponible en este proyecto — más que
   cambiar de proveedor cloud.
4. **`gpt-oss-20b` (86 %) rinde parecido a `qwen3:8b` con solape (86-87 %)** pero
   peor que `qwen3:8b` + embedding (94 %) — un modelo cloud de tamaño medio no
   garantiza superar a un modelo local bien evaluado. El tamaño del modelo NO
   predice el resultado tan bien como la combinación (modelo, estrategia de
   prompt) evaluada de verdad.

---

## Fundamento en la literatura (2026-09-11)

Para no depender solo de los datos de este proyecto, se revisó brevemente qué
dice la investigación reciente sobre texto→SPARQL/SQL con LLM:

- **Zero-shot vs. few-shot en KGQA**: el zero-shot puro funciona pero se queda
  muy por debajo de dar ejemplos — coincide exactamente con la Fase 1 de este
  estudio (30 % zero-shot vs 76-82 % con ejemplos). *[Are LLMs adequate SPARQL
  query generators?](https://link.springer.com/article/10.1007/s00521-025-11799-x)*.
- **La selección de ejemplos por similitud NO es automáticamente mejor que
  alternativas más simples**: un estudio de generación SPARQL sobre un grafo de
  aviación probó 6 estrategias (aleatoria, 10/20/30 ejemplos, por similitud) y
  encontró que un prompt simple + 5 ejemplos aleatorios daba el mejor resultado
  — exactamente lo que la Fase 4 de este estudio encuentra para qwen2.5:7b (el
  embedding no gana). *[Context-Aware Few-Shot Learning SPARQL Query Generation
  from Natural Language on an Aviation Knowledge Graph](https://doi.org/10.3390/make7020052)*.
- **La recuperación dinámica por embedding SÍ es el patrón dominante en
  text-to-SQL de producción** (LangChain la ofrece de fábrica) y mejora sobre
  ejemplos fijos en la mayoría de estudios — pero la ganancia depende de la
  calidad del modelo base para explotarla, coincidiendo con el hallazgo
  específico de este estudio (gana en qwen3:8b, no en qwen2.5:7b).
  *[Robust Active Learning for Few-Shot Example Selection in Text-to-SQL](https://arxiv.org/abs/2606.10125)*.
- **El tamaño del modelo tiene rendimientos decrecientes claros** en tareas de
  query generation: un estudio cruzado en BIRD (familias Qwen2.5-Coder,
  CodeLlama, Llama-3.x) concluye que **"la generación importa más que el tamaño
  puro"**, y otro (SLM-SQL) muestra un modelo de 0,6B superando a un DTS-SQL de
  7B con la técnica adecuada — exactamente el patrón 7B→8B (+8 pts) / 8B→14B (0
  pts) de la Fase 3 de este estudio.
  *[How Far Do On-Prem Open LLMs Get on Text-to-SQL?](https://arxiv.org/html/2606.29733)*,
  *[SLM-SQL](https://arxiv.org/html/2507.22478v1)*.
- **Los sistemas KGQA basados en plantillas son un diseño con precedente**, con
  la limitación conocida de que son menos flexibles que un LLM ante fraseos no
  previstos — exactamente el trade-off medido en la Fase 5 (Arm G): 77 % de
  acierto zero-LLM, pero con un 20 % de preguntas que reconoce como fuera de su
  catálogo y delega. *[Interpretable Question Answering with Knowledge
  Graphs](https://arxiv.org/html/2510.19181v1)*.
- Un dato que vale la pena vigilar para este dominio (actas municipales, texto
  administrativo) más que para el general: modelos de código específicos
  (Qwen2.5-Coder) superan a sus equivalentes generalistas del mismo tamaño en
  text-to-SQL — motivó probar `qwen2.5-coder:7b` en este estudio (ver pendientes).

---

## Conclusiones generales (actualizado 2026-09-11)

1. **Pregunta del usuario — few-shot, zero-shot o embeddings**: ninguna gana de
   forma absoluta; lo que gana siempre es **compacto + ejemplos dinámicos + capa
   determinista** frente a zero-shot o al mega-prompt de reglas. Dentro de "cómo
   elegir los ejemplos dinámicos", **el embedding solo aporta si el modelo de
   generación es lo bastante capaz de explotar similitud semántica** (qwen3:8b,
   Gemini); con un modelo más simple (qwen2.5:7b) el solape léxico simple iguala
   o mejora el resultado y es más barato. Esto tiene respaldo directo en la
   literatura de text-to-SQL/SPARQL (ver arriba), no es un artefacto de este grafo.
2. **La alternativa "cero LLM" (Arm G) es viable para un subconjunto real y
   grande de preguntas**: 77 % de acierto exacto, 96 % de precisión cuando
   responde, cero respuestas falsas por sobreconfianza en lo que no cubre, ~300
   veces más rápido que cualquier LLM (300 ms vs 1-15 s) y sin coste de API. No
   sustituye al LLM (no cubre persona/voto nominal/comparaciones entre grupos)
   pero es la opción correcta como PRIMER intento antes de gastar una llamada
   de LLM.
3. **Modelos**: hay un suelo (llama3.2:3b ~40 %, gemma3:4b ~40-50 %, ninguno
   arreglable con prompt) y un techo práctico en este estudio
   (gemini-3.5-flash-lite, 97 %/1 % mal, barato). `mistral:7b` queda en tierra
   de nadie (66 %, sin respuestas "mal" pero con muchos errores de sintaxis) sin
   superar a ninguna variante de Qwen. Entre suelo y techo, **qwen3:8b +
   few-shot dinámico por embedding + capa determinista (94 %/1 % mal) es la
   mejor opción local, gratuita, y a solo 3 puntos del techo de pago** — mejor
   recomendación que cualquier variante de qwen2.5:7b, que mistral o gemma, o
   que depender de la cuota gratuita de Groq (agotada dos veces en esta sola
   sesión).
4. Quedan sin resolver, de forma consistente en TODAS las configuraciones locales
   probadas (Fases 2-6): comparar la tasa de aprobación de dos grupos en la misma
   consulta (g22) y el ranking de voto nominal (g27) — necesitan un ejemplo
   dedicado en el few-shot o un modelo de la clase Gemini/Groq-120B.

## Recomendación para el sistema de producción (actualizada)

1. **`graph_answer()` intenta primero `arm_g.answer(pregunta)`** (Fase 5, cero
   LLM, ~300ms) — si devuelve `sparql=None`, sigue al paso 2. Esto responde de
   forma instantánea y gratuita a las preguntas recurrentes de conteo/ranking/
   temporal/ratio/entidad simples.
2. Si no hay plantilla: generar con LLM usando **`F_embed`** (compacto + pocos
   ejemplos elegidos por embedding bge-m3 + capa de alias/guardas deterministas)
   si el modelo es **`qwen3:8b`** — es la combinación (modelo, estrategia) con
   mejor resultado local de todo el estudio (94 %/1 % mal). Si algún día se usa
   `qwen2.5:7b` (p.ej. por VRAM), usar `F_compact` (solape de palabras) en su
   lugar — el embedding no compensa en ese modelo y además penaliza la latencia
   en esta GPU de 6 GB por el intercambio de modelos.
3. Si hay presupuesto de API y se prioriza fiabilidad sobre coste cero: **Gemini
   3.5 Flash Lite + F_embed** (97 %/1 % mal, 0,9s, ~0,30 $/M tok entrada) es
   superior a cualquier configuración de Groq probada en esta sesión, y no tiene
   el límite diario duro que interrumpió dos corridas de Groq.
4. De las 12 guardas de `_sanitize_sparql`, conservar las ~8 estructurales y
   añadir la capa de alias de `arms_f.py`.
5. Antes de dar el cambio por bueno: ampliar `scripts/regression_qa.py` con
   preguntas NUEVAS (no las que motivaron los parches actuales) — el gold de este
   estudio (`estudio_sparql/gold.py`, 35 casos) sirve de base.

## Pendiente (no ejecutado en esta sesión, por presupuesto de tiempo)

- `qwen/qwen3.6-27b` en Groq (catalogado, cuota propia probablemente disponible,
  no evaluado).
- Repetir `groq-20b` y `groq` (120b) con `reps=3` completos cuando la cuota
  diaria esté fresca desde el principio del día, para cifras menos ruidosas.
- Arm G (Fase 5) es un prototipo de investigación, no está integrado en
  `graph_rag_sparql.py` — si se decide llevarlo a producción, ampliar su catálogo
  de formas con los casos reales que fallen en `regression_qa.py`.

## Reproducir

```
cd estudio_sparql

# Fases 2-3 (word-overlap)
python run_study.py --arms A_min,B_vocab,C_full,D_compact_dyn,E_full_dyn,F_compact,F_full \
                    --model qwen2.5:7b --reps 3 --out study_qwen7b.jsonl

# Fase 4 (embedding bge-m3 en vez de solape de palabras)
python run_study.py --arms D_embed_dyn,F_embed --model qwen3:8b --reps 3 --out study_qwen3_8b.jsonl

# Fase 5 (cero LLM, una sola pasada, determinista)
python arm_g.py

# Fase 6 (modelos nuevos)
python run_study.py --arms D_compact_dyn,F_compact --model llama3.2:latest --reps 3 --out study_llama32.jsonl
python run_study.py --arms A_min,C_full,D_compact_dyn,D_embed_dyn,F_compact,F_embed \
                    --model gemini --reps 3 --out study_gemini.jsonl --sleep 0.6
python run_study.py --arms D_compact_dyn,D_embed_dyn,F_compact,F_embed \
                    --model groq-20b --reps 3 --out study_groq20b.jsonl --sleep 2

# candidatos descartados (reps=1, solo F_compact)
python run_study.py --arms F_compact --model mistral:7b --reps 1 --out study_mistral.jsonl
python run_study.py --arms F_compact --model gemma3:4b --reps 1 --out study_gemma.jsonl

python analyze.py study_qwen7b.jsonl                   # tablas (repetir por fichero)
```
