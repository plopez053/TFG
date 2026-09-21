# GraphRAG vs RAG vectorial — comparativa para el TFG

_Batería ejecutada 2026-09-09 sobre el grafo re-enriquecido con Gemini (3.422
proposiciones, capa de concejales/enmiendas/voto) y el chroma_db reparcheado.
Resultados crudos en `scratchpad/bateria_resultados*.md`._

> **Cómo usar este documento mañana:** cada pregunta de abajo tiene marcado
> **[CAPTURA GRAFO]**, **[CAPTURA VECTOR]** o **[CAPTURA AMBOS]**. Lanza esa
> pregunta en Chainlit, despliega el panel de pasos (SPARQL / fuentes) y captura.
> Debajo de cada una está lo que la captura debe *demostrar* en la memoria.

---

## 1. Síntesis — fortalezas y debilidades

### GraphRAG (rdflib + SPARQL + razonador OWL-RL)

**Fuerte en:**
| Capacidad | Por qué |
|---|---|
| **Recuento y agregación exactos** | "cuántas proposiciones de vivienda por grupo" → tabla con los 10 grupos y cifras exactas, calculadas por `COUNT`/`GROUP BY` sobre TODO el grafo, no sobre una muestra. |
| **Rankings y comparativas** | "qué grupo presentó más enmiendas", "qué concejal firmó más" — imposibles de contestar bien sin estructura. |
| **Evolución temporal** | "proposiciones de euskera año a año" → serie 2008-2026 completa; el vectorial no puede reconstruir una serie. |
| **Roll-up temático (razonador)** | "presupuestos y fiscalidad incluyendo subtemas" → suma automáticamente IBI, subvenciones, ordenanzas fiscales… vía `bo:trataTemaAmplio`. |
| **Trazabilidad** | la consulta SPARQL es auditable: se ve exactamente qué se contó y cómo. |
| **Velocidad** | ~7-15 s en agregaciones (vs 20-60 s del vectorial). |
| **Cifras sin alucinación** | los números salen del SPARQL, no de un LLM leyendo texto. |
| **Voto nominal y por grupo, literal del acta** | `bo:votoAFavor`/`EnContra`/`Abstencion` por concejal (34.178 votos), parseados con regex de las listas "Votos negativos: N señoras/señores: …" — `bo:votoFuente "acta"`. 40% de las proposiciones con voto **verificado palabra por palabra**, no inferido. Habilita "¿cómo votó X?", "¿quién rompe la disciplina de voto?". |

**Débil en:**
| Problema | Ejemplo de la batería |
|---|---|
| **Preguntas de "qué se dijo / qué argumentó"** | "argumentos de EH Bildu contra los presupuestos" → forzó un filtro `votoEnContraDe + tema:presupuestos` y devolvió 2 proposiciones que NO van de eso (una "declaración de urgencia", otra de "taxis eléctricos"). El grafo modela *proposiciones y votos*, no el hilo argumental del debate. |
| **Entidades que el enriquecimiento no extrajo** | "¿se ha mencionado a Mercadona?" → Mercadona no es un nodo del grafo (el enriquecedor solo guardó entidades "relevantes"); el `CONTAINS "zara"` además dio falsos positivos: Zaragoza, Zarautz, Avenida Zarandoa. |
| **No determinismo del generador de SPARQL** | en ~2 de 17 casos de agregación compleja (ratios, dos cifras) el LLM escribe `GROUP BY ?prop` o mete `FILTER` donde va `OPTIONAL` → cifra mal o vacío. Los 3 reintentos lo tapan casi siempre pero no del todo. |
| **Matices y citas literales** | no puede citar la frase textual de un concejal ni el tono del debate. |

### RAG vectorial (ChromaDB + Cohere rerank + Groq)

**Fuerte en:**
| Capacidad | Por qué |
|---|---|
| **"Qué se dijo / qué argumentó / qué tono"** | "argumentos de EH Bildu contra los presupuestos" → relato coherente: criticó la gestión de viviendas vacías y la falta de control de la tasa H sobre empresas, con cifras del propio debate (16 M€). |
| **Detalle por proposición** | reconstruye propuesta + argumentos + resultado + reparto de votos leyendo el acta. |
| **Citas y fuentes textuales** | devuelve el fragmento literal del PDF y la página. |
| **Búsqueda literal de cualquier nombre** | encuentra una mención única de una empresa/persona aunque no esté estructurada (cubre el punto ciego del grafo). |
| **Cobertura del corpus completo** | indexa el 100% del texto, no solo lo que el enriquecedor consideró una "proposición". |

**Débil en:**
| Problema | Ejemplo de la batería |
|---|---|
| **No sabe contar** | "cuántas proposiciones de vivienda por grupo" → contestó "PP 4, EH BILDU 1, PSE 1…" contando SOLO los ~20 fragmentos que recuperó. La cifra real (grafo) es PP 46, EH BILDU 62. **Respuesta con pinta de correcta pero radicalmente falsa.** |
| **No agrega ni compara** | "¿en qué año más rechazos?" → "empate a 1 entre 2008 y 2012" (solo vio esos plenos). Real: 2018 con 56. |
| **Deriva temática en temas recientes** | primera pasada de "pisos turísticos" → recuperó plenos de 2008-2013 y respondió "no se ha debatido". Real: hay proposiciones de tasa turística de EH Bildu y Elkarrekin en 2024. |
| **Riesgo de alucinación en cifras** | al no tener los números, el LLM los infiere del texto que ve y puede equivocarse con total seguridad. |
| **Más lento y depende de Groq** | 20-60 s; con el tier gratis de Groq (8.000 tokens/min) una ráfaga de preguntas lo tumba. |

### La conclusión del TFG en una frase

> **El GraphRAG responde "cuánto / quién / cuándo / cómo se repartió" con precisión
> verificable; el RAG vectorial responde "qué se dijo / por qué / con qué palabras".
> Son complementarios: el grafo da el esqueleto cuantitativo, el vectorial la carne
> argumental. Sus errores también son distintos: el grafo falla de forma visible
> (consulta rara, resultado vacío), el vectorial falla de forma invisible (cifra
> segura y equivocada).**

---

## 2. Guion de capturas — pregunta a pregunta

### Bloque A — donde gana el GRAFO (agregación / estructura)

#### A1. `¿Cuántas proposiciones sobre vivienda ha presentado cada grupo municipal?` — **[CAPTURA AMBOS]**
- **Grafo:** tabla con 10 grupos, EH BILDU 62 / PP 46 / ELKARREKIN 17 / PSE 12 / … Cifras exactas, SPARQL visible con `COUNT + GROUP BY`.
- **Vector:** contesta "PP 4, EH BILDU 1, PSE 1…" a partir de los pocos fragmentos recuperados.
- **Qué demuestra:** el vectorial **no puede contar** — da una respuesta plausible pero falsa por un factor de 10. El grafo cuenta sobre el corpus entero.

#### A2. `¿En qué año se rechazaron más proposiciones en el Pleno de Bilbao?` — **[CAPTURA AMBOS]**
- **Grafo:** 2018 con 56 (ranking por año con `bo:tieneResultado bo:Rechazada`).
- **Vector:** "empate a 1 entre 2008 y 2012".
- **Qué demuestra:** agregación temporal — el vectorial solo ve su ventana de recuperación.

#### A3. `¿Cómo ha evolucionado el número de proposiciones sobre euskera año a año?` — **[CAPTURA GRAFO]**
- **Grafo:** serie 2008→2026. Salto claro: 1-4/año hasta 2020 → 9 en 2022-2024 → 12 en 2025.
- **Qué demuestra:** el grafo reconstruye una **serie temporal completa**; es un gráfico listo para la memoria.

#### A4. `¿Cuántas enmiendas ha presentado cada grupo municipal?` — **[CAPTURA GRAFO]**
- **Grafo:** EQUIPO DE GOBIERNO 1.125 / EAJ-PNV 125 / EH BILDU 66 / … (nodos `bo:Enmienda`, capa nueva del re-enriquecimiento).
- **Qué demuestra:** capacidad nueva tras el re-enriquecimiento; el patrón "el gobierno enmienda para hacer decaer las proposiciones de la oposición" es una conclusión sustantiva del TFG.

#### A5. `¿Qué concejal o concejala ha firmado más proposiciones?` — **[CAPTURA GRAFO]**
- **Grafo:** Cristina Ruiz Bujedo (PP) con 80, seguida de Luis Hermosa (41), Luis Eguiluz (40)… (`bo:proponePersona`).
- **Qué demuestra:** la **capa de personas** — el grafo distingue concejal de grupo; el vectorial no tiene ese concepto.

#### A6. `¿Cuántas proposiciones sobre presupuestos y fiscalidad se han presentado en total, incluyendo subtemas?` — **[CAPTURA GRAFO]**
- **Grafo:** 733 (con `bo:trataTemaAmplio`, que arrastra IBI, subvenciones, ordenanzas fiscales…). Directo sin roll-up: 389.
- **Qué demuestra:** el **razonador OWL-RL** en acción — inferencia real, no solo lookup.

#### A7. `¿Qué concejal o concejala ha votado más veces en contra de las proposiciones?` — **[CAPTURA GRAFO]**
- **Grafo:** Yolanda Díez Saiz (309), Alfonso Gil Invernón (266), Marta Ajuria (264)… (`bo:votoEnContra`, **voto nominal parseado literal del acta** — capa determinista, `bo:votoFuente "acta"`).
- **Qué demuestra:** dato **imposible** para el vectorial y **100% fiel** (copiado de las listas "Votos negativos: N señoras/señores: …" del acta, no inferido por ningún LLM). 34.178 votos nominales en el grafo, 40% de las proposiciones con voto verificado (vs 9% que cubría el enriquecedor LLM).

#### A8. `¿Cómo votó EH Bildu las proposiciones sobre vivienda?` — **[CAPTURA GRAFO]**
- **Grafo:** a favor 46, en contra 19, abstención 11 (`bo:votoAFavorDe`/`EnContraDe`/`seAbstuvo` derivados del voto nominal).
- **Qué demuestra:** análisis de comportamiento de voto por grupo y tema — el núcleo de un análisis político, y ahora con respaldo literal del acta.

### Bloque B — donde gana el VECTORIAL (matiz / texto)

#### B1. `¿Qué argumentos ha dado EH Bildu para oponerse a los presupuestos municipales de Bilbao?` — **[CAPTURA AMBOS]**
- **Vector:** relato con los argumentos reales (viviendas vacías, control de la tasa H, 16 M€ sin supervisar) y citas del acta.
- **Grafo:** 2 proposiciones sin relación clara (declaración de urgencia, taxis eléctricos) — filtro forzado.
- **Qué demuestra:** para "por qué / con qué argumentos", el grafo se queda corto y el vectorial brilla.

#### B2. `¿Qué se debatió sobre el registro de viviendas turísticas y los pisos turísticos en Bilbao?` — **[CAPTURA AMBOS]**
- **Vector:** debe recuperar los debates recientes (tasa turística 2024, regulación de alojamientos). _Ojo: en una pasada derivó a plenos de 2008-2013 y dijo "no encontrado" — si pasa en la captura, reformular a "impuesto al turismo en Bilbao 2024" o repetir._
- **Grafo:** mapea a `bo:trataTemaAmplio br:t_turismo` y lista ~15 proposiciones, incluidas las de EH Bildu y Elkarrekin de 2024.
- **Qué demuestra:** contraste de recuperación — el grafo por tema es robusto, el vectorial puede derivar.

#### B3. `¿Qué se dijo sobre las personas sin hogar en Bilbao?` — **[CAPTURA VECTOR]**
- **Vector:** relato cronológico con propuestas concretas de varios grupos y sus resultados.
- **Qué demuestra:** narrativa temática rica con fuentes — el caso de uso natural del vectorial.

### Bloque C — donde AMBOS fallan (y de forma distinta) → muy útil para "limitaciones"

#### C1. `¿Se ha mencionado a Mercadona o a Zara en algún pleno del Ayuntamiento de Bilbao?` — **[CAPTURA AMBOS]**
- **Grafo:** ahora usa `REGEX(STR(?ent), "\bzara\b", "i")` (límite de palabra) → resultado **vacío** → responde honestamente "no aparecen como entidad del grafo". _(Antes: `CONTAINS "zara"` daba falsos positivos "Zaragoza", "Zarautz", "Avenida Zarandoa" — corregido en `_sanitize_sparql` guard 8 + SCHEMA.)_
- **Vector:** busca el literal; si el término no está en el top-k, responde "no se menciona".
- **Qué demuestra:** el grafo **solo sabe lo que el enriquecedor estructuró** (de hecho Inditex, Eroski y El Corte Inglés SÍ están, pero Mercadona y Zara no). Limitación honesta del enfoque de grafo para el "long tail" de entidades: el enriquecedor no es exhaustivo. Para eso está el vectorial.

#### C2. `¿Qué grupo ha impulsado más iniciativas a favor del comercio local de barrio?` — **[CAPTURA GRAFO]**
- **Grafo:** PP con 59 (por `bo:trataTemaAmplio br:t_comercio`). **Pero** "a favor de" es una valoración que el grafo no modela: cuenta *todas* las de comercio del PP, no solo las favorables. La cifra es correcta para "proposiciones sobre comercio local", no exactamente para la pregunta.
- **Qué demuestra:** el grafo responde la versión *cuantificable* de la pregunta, no el matiz ("a favor de", "más crítico con"). Limitación a señalar.

---

## 3. Checklist para las capturas de mañana

- [ ] Arrancar Chainlit: `chainlit run frontend/app.py --port 8000` desde `TFG/TFG/`
- [ ] Verificar que Ollama está arrancado (embeddings) y que Groq responde (si da 413/429, esperar 1 min entre preguntas — límite 8.000 tok/min del tier gratis)
- [ ] Por cada pregunta del bloque: capturar (a) la pregunta, (b) el panel de pasos desplegado — **SPARQL** en GraphRAG, **fuentes/PDF** en vectorial —, (c) la respuesta completa
- [ ] Para A1 y A2: capturar las DOS respuestas una al lado de otra (es el contraste más claro del TFG)
- [ ] Guardar las capturas en una carpeta `capturas/` con nombres tipo `A1_grafo.png`, `A1_vector.png`
- [ ] Anotar la fecha/hora de cada captura (por si hay que rehacer alguna)

## 4. Preguntas de reserva (por si alguna falla en directo)

| Reserva | Sistema | Resultado esperado |
|---|---|---|
| `¿Cuántas proposiciones hay en total en el grafo?` | Grafo | 3.422 |
| `¿Cuántas proposiciones sobre desahucios se han presentado?` | Grafo | 14 |
| `¿En cuántos debates ha intervenido Xabier Otxandiano?` | Grafo | 80 |
| `¿Qué se ha debatido sobre la Zona de Bajas Emisiones (ZBE)?` | Vector | plenos 2019-2025, propuestas de Elkarrekin/PP |
| `¿Qué se dijo sobre el soterramiento de la estación de Abando?` | Vector | relato con varias sesiones |
| `¿Qué grupo tiene mejor tasa de aprobación, EH Bildu o el PP?` | Grafo | EH Bildu 200/638 (31%) vs PP 276/936 (29%) |

---

## 5. Batería #3 (2026-09-10) — el GraphRAG con LLM local vs con Groq

_Crudos en `bateria_comparativa_cruda_p3.md`. Groq tenía el límite diario agotado,
así que **toda** la batería corrió con Ollama `qwen2.5:7b` (generación de SPARQL
Y narración, en los dos sistemas). Esto convierte la batería en un experimento
útil: **aísla el efecto del modelo que genera la SPARQL**._

### Resultado global

| | GraphRAG | RAG vectorial |
|---|---|---|
| Preguntas con respuesta correcta/útil | **3 / 10** | 5 / 10 |
| Fallos por SPARQL mal generada (0 filas) | **6 / 10** | — |
| Fallos por no agregar / ignorar la pregunta | — | 4 / 10 |

Con Groq, el GraphRAG acertaba ~8-9 de 10 en preguntas de este tipo (ver batería
#1 y regresión 22/22). **Con `qwen2.5:7b` local baja a 3/10.** No es que el grafo
empeore: los datos son los mismos. Es que el modelo de 7B **inventa vocabulario**
—etiquetas, URIs de tema, funciones SPARQL— que no existe, la consulta devuelve
cero filas, y el narrador convierte ese vacío en una **negación rotunda y falsa**.

### Los 6 fallos del grafo, todos el mismo patrón

| Q | Lo que el LLM local inventó | Realidad |
|---|---|---|
| Q2 PP vota a favor de EH Bildu | `bo:proponePersona` = grupo, `votoAFavorDe` al revés, filtro `REGEX(nombre,"popular")` | dato existe en la capa nominal, query irrescatable |
| Q3 votos en contra de Yolanda Díez | `REGEX(...,"\\bdíez\\b")` con tilde | grafo la tiene como "Diez" (sin tilde) → **309 votos**. `REGEX "i"` no ignora tildes. **Corregido** (guard 9) |
| Q5 posición del gobierno sobre metro | `rdfs:label "Equipo de Gobierno de Bilbao"` (literal inventado) | pregunta narrativa, el grafo no la modela |
| Q6 limpieza en Otxarkoaga/San Ignacio | `bo:trataTemaAmplioLabel(...)` (función inexistente), filtro por barrio en los temas | hay `br:t_barrios` (119) y `br:t_reciclaje` (73), pero ningún tema de "limpieza"; y los temas no tienen granularidad de barrio |
| Q8 evolución "emergencia climática" año a año | nodo en blanco `[ skos:prefLabel "emergencia climática" ]` | no supo mapear el texto a `br:t_medioambiente` (269); no existe `t_clima` |
| Q9 grupo más crítico con limpieza viaria | `br:t_limpieza_viaria_y_recogida_basuras` (URI de tema inventado) | no existe ese tema; `br:t_reciclaje` es lo más cercano |

> El grafo tiene **~40 temas amplios** con `skos:prefLabel` en español
> (presupuestos, movilidad, urbanismo, vivienda, euskera, comercio, barrios,
> reciclaje…). El LLM local no los ve todos en el SCHEMA y, en vez de resolver
> el tema con `?t skos:prefLabel ?l . FILTER(REGEX(?l, "..."))`, **adivina el
> URI** — y casi siempre falla. Mejora pendiente: forzar en el SCHEMA la
> resolución de temas por prefLabel, nunca por URI directo.

### Qué demuestra para el TFG

1. **El GraphRAG es tan bueno como su generador de SPARQL.** El cuello de botella
   no es rdflib ni el modelo de datos: es el paso texto→SPARQL. Un modelo grande
   (Groq gpt-oss-120b) lo hace bien; uno de 7B local falla la mitad de las veces.
   El RAG vectorial es **mucho menos sensible al modelo** — con el mismo 7B local
   sigue recuperando y resumiendo de forma razonable.
2. **El modo de fallo del grafo es peligroso:** 0 filas → "no hay datos sobre
   este tema" dicho con seguridad total. Un usuario que no sepa SPARQL se lo cree.
   Mitigación posible: cuando la consulta da 0 filas, el narrador debería decir
   "no he sabido consultarlo" en vez de "no existe".
3. **El modo de fallo del vectorial también:** Q7 (¿se mencionó a Amazon / El
   Corte Inglés?) → recupera 14 plenos sin ninguna mención, y **resume esos
   plenos ignorando la pregunta**, sin decir nunca "no lo he encontrado".

### Preguntas de la batería #3 para capturar

| # | Pregunta | Captura | Qué demuestra |
|---|---|---|---|
| **B3-1** | `¿Cuántas proposiciones ha presentado cada grupo y qué tasa de aprobación tiene?` | **AMBOS** | grafo: tabla exacta 11 grupos. Vector: no puede. |
| **B3-2** | `¿Cuál fue el pleno concreto con más proposiciones rechazadas de toda la serie?` | **AMBOS** | grafo: 22-03-2018, 23 rechazos (exacto). Vector: "28-11-2007, 3" (ve solo 2 plenos → falso). |
| **B3-3** | `¿Qué posición ha defendido el equipo de gobierno sobre la ampliación del metro?` | **VECTOR** | grafo: 0 filas → "no hay datos" (falso). Vector: debate real Línea 4 vs 5. |
| **B3-4** | `¿Se ha mencionado a Amazon o a El Corte Inglés en algún pleno?` | **AMBOS** | los dos fallan: grafo dice "no" sin mirar la capa de entidades; vector resume plenos que no vienen a cuento. |
| **B3-5** | `¿Cuántas veces ha votado en contra la concejala Yolanda Díez?` | **GRAFO** | **tras el fix de tildes**: 309. Capturar para mostrar la capa de voto nominal funcionando. |

> **Antes de capturar B3-5**: reiniciar Chainlit para cargar el guard 9 (tildes).
> Si Groq vuelve a estar disponible, relanzar la batería #3 con Groq para tener
> también la versión "buena" del generador de SPARQL como comparación.
