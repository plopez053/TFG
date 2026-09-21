# Plan de mejora del grafo (GraphRAG) — para ejecutar en sesiones futuras

> Preparado el 2026-09-07. El usuario quiere mejorar el grafo **todo lo posible**,
> sin importar coste ni dificultad. Este documento es el guion de trabajo.

---

## ✅ YA PREPARADO (código listo, 2026-09-08) — solo falta EJECUTAR

Las Tareas **1, 2, 3, 5, 7** y parte de la **4** están implementadas. El código
está en el repo y NO se ha ejecutado la pasada cara todavía. Todo es aditivo:
el grafo actual sigue funcionando hasta que se relance el pipeline.

### Lo que se ha tocado
- **`build_graph.py`**: nuevo `EXTRACT_PROMPT` rico (resumen, votos_por_grupo,
  enmienda, proponente_persona, + temas/entidades/resultado). Rama
  `--model gemini` (Google, 1M contexto, tier gratis). `--rpm` para el ritmo del
  tier gratis. Sin truncado para gemini/groq (`_ctx_chars`). Backoff largo en 429.
- **`extract_proposals.py`**: `MAX_TEXT` 6.000 → 30.000. Campo nuevo `oradores`
  (apellidos que intervinieron, del `speaker` del indexador).
- **`extract_concejales.py`** (NUEVO): saca del encabezado de cada acta:
  - `concejales.jsonl` — **94 concejales** (dedup de OCR: espacios espurios
    "H elena", grafía euskera/castellano "Otxandiano"/"Ochandiano", apodos
    "Goyo"/"Gregorio Zurro", "Txema"/"José María Oleaga", nombres truncados,
    y bug de split "doña X doña Y"). Con `es_alcalde` y variantes de apellido.
  - `personal_tecnico.jsonl` — **5** (Secretario General, Interventor, adjuntos).
    Asisten a todo pero NO son cargos electos; excluidos de `concejales.jsonl`.
- **`concejales_partido.json`** (NUEVO): tabla concejal→grupo por mandato.
  ✅ **RECONSTRUIDA Y VERIFICADA (2026-09-08)**. **86/94 concejales con grupo**;
  los 8 sin grupo son del mandato 2007-2011 (el corpus no tiene su acta
  constitutiva) + 2 sustitutos con <15 actas. Fuentes:
  - 2011/2015/2019: sección "(EAJ-PNV): NOMBRE NOMBRE…" de las **actas de sesión
    constitutiva** (11-06-2011, 13-06-2015, 15-06-2019), parseadas respetando
    saltos de página. Cuadran 29 escaños y los bloques de voto de plenos
    posteriores (acta 2016 etiqueta "D. Xabier Otxandiano Martínez (EAJ-PNV)"
    — corrige el error de bulto: Otxandiano es PNV, NO EH Bildu).
  - 2023: el acta constitutiva es bilingüe a 2 columnas y la extracción mezcla
    nombre↔partido; reconstruido desde el bloque Gobierno(PNV+PSE) vs oposición
    del pleno 25-01-2024. Correcciones sobre el borrador: Renedo Lara y
    Undabarrena son EH BILDU; Garagalza es PP; Bilbao Aldayturriaga es PSE.
  - Sustitutos: bloque de voto del pleno (Guarrotxena EH BILDU, Zubizarreta
    Agirrezabal EAJ-PNV, J.L. Delgado PSE).
  - El `_INSTRUCCIONES` del JSON documenta cada bloque. Un humano solo tendría
    que rellenar los 5 de 2007-2011 (bilbao.eus / Wikipedia) si se quieren.
- **`bo:Alcalde` / `bo:esAlcalde`**: los 3 alcaldes del periodo (Azkuna 2007-
  2014, Areso 2014-2015 interino, Aburto 2015-) con sus fechas EXACTAS curadas
  en `build_rdf.py::_ALCALDES`. El orador "ALCALDE"/"PRESIDENTE" del speaker_regex
  se resuelve a la persona correcta según la fecha (verificado).
- **`entidades.py`** (NUEVO): `canon_entidad()` — filtra basura ("A.A.", "Aita",
  "Ayuntamiento de Bilbao") y agrupa "Iberdrola"/"Iberdrola SA" en un solo nodo.
- **`ontology.ttl`**: clases `bo:Concejal`, `bo:Enmienda`; propiedades
  `bo:perteneceA`, `bo:intervino`, `bo:proponePersona`, `bo:votoAFavorDe`/
  `votoEnContraDe`/`seAbstuvo`, `bo:tieneEnmienda`/`enmiendaPor`/`resumenEnmienda`/
  `tipoEnmienda`, `bo:resumen`, `bo:pagina`, `bo:fuentePdf`.
- **`build_rdf.py`**: `cargar_concejales()` + `resolver_concejal()` (enlaza
  apellido+fecha → concejal correcto, verificado con Ibaibarriaga/Gil/Aburto...).
  Consume TODOS los campos nuevos del enriquecido. Entidades pasan por `canon_entidad`.
- **`graph_rag_sparql.py`**: SCHEMA ampliado con los patrones nuevos (concejal
  más activo, voto por grupo, etc.).
- **`regression_qa.py`**: sin regresión. El grafo se regeneró con el censo nuevo
  (94 Concejal, 84 `bo:perteneceA` con partido correcto, 3 `bo:esAlcalde`, 5
  PersonalTecnico) sobre el enriquecido VIEJO — 3923 props, vivienda 190, euskera
  3, sin mover. `bo:intervino` sigue en 0 hasta correr Fase 1+2 (el enriquecido
  actual aún no tiene el campo `oradores`; se llena en el RUNBOOK).

### RUNBOOK — orden exacto de ejecución (en el equipo con GPU)

```bash
# 0. Preparar Gemini (una vez)
#    - API key en https://aistudio.google.com/apikey  ->  .env:  GOOGLE_API_KEY=AIza...
pip install langchain-google-genai
#    - concejales_partido.json YA reconstruido y verificado (86/94). Opcional:
#      rellenar los 5 del mandato 2007-2011 desde bilbao.eus si se quieren.

# 1. Re-extraer proposiciones SIN truncar (rápido, ~10 min, necesita Ollama)
python graphrag/graphrag/extract_proposals.py          # regenera proposals.jsonl con oradores + texto largo

# 2. Censo de concejales (rápido, ~5 min)
python graphrag/graphrag/extract_concejales.py         # concejales.jsonl  (revisar a mano: nombres, fechas)

# 3. RE-ENRIQUECER TODO con Gemini (la pasada cara)  <-- PASO 3
mv graphrag/graphrag/proposals_enriched.jsonl graphrag/graphrag/proposals_enriched.qwen3_8b.jsonl.bak
#
#    EJECUTADO 2026-09-09: API DE PAGO, gemini-3.5-flash-lite, --rpm 60 --max-eur 12.
#    (OJO: gemini-2.5-flash / 2.5-flash-lite YA NO existen para claves nuevas -> 404.
#     El alias 'gemini' en build_graph.py apunta ya a gemini-3.5-flash-lite, mismo
#     precio 0,30/2,50 $/M que el viejo 2.5-flash. gemini-3.5-flash "grande" = 5x.)
#    Coste medido: ~4.700 tok in + ~190 out por proposición -> tanda entera
#    (3422 props) ~7 €. Billing debe estar activo en aistudio.google.com/apikey.
#    Opcional: presupuesto con aviso email en Google Cloud (solo avisa, no corta).
#
#    a) CALIBRAR con 100 (imprime "gasto estimado ~X EUR"; x34 = tanda entera):
python graphrag/graphrag/build_graph.py --enrich --model gemini --rpm 60 --limit 100
#
#    b) TANDA COMPLETA con tope duro (si se corta, relanza el MISMO comando: lo
#       hecho queda guardado y no se repite):
python graphrag/graphrag/build_graph.py --enrich --model gemini --rpm 60 --max-eur 12
#       --max-eur = para solo al llegar a ese gasto. Sube el número y relanza
#       para continuar.
#
#    Alternativas: tier GRATIS (--model gemini-3.5-flash-lite --rpm 8, sin --max-eur)
#    ~limitado por cuota diaria; Batch API 50% más barato pero NO implementado.

# 4. Reconstruir el grafo
python graphrag/graphrag/build_rdf.py                  # bilbao_reasoned.ttl

# 5. Reparchear el chroma_db (metadatos del RAG vectorial) con lo nuevo
python scripts/enrich_vector_metadata.py

# 6. Verificar
python scripts/regression_qa.py                        # actualizar los expect si alguna cifra
                                                       # de agregación cambió (verificar con SPARQL)
```

### PREGUNTAS NUEVAS que el grafo enriquecido podrá responder (para el TFG)

**Concejales** (hoy: imposible, solo hay grupos):
- ¿En qué proposiciones intervino la concejala X? ¿sobre qué temas interviene más?
- ¿Qué concejal ha sido el más activo en el debate? (nº de intervenciones)
- ¿Qué concejales de EH Bildu intervinieron sobre vivienda?
- ¿Quién firmó la proposición de X? ¿a qué grupo pertenece Y?

**Voto por grupo** (hoy: solo el total favor/contra):
- ¿Cómo votó EH Bildu la proposición de la ZBE del 26-09-2019?
- ¿En qué proposiciones votaron juntos PP y PSE? ¿y sistemáticamente distinto EH Bildu y el equipo de gobierno?
- ¿Proposiciones aprobadas con el voto en contra de EAJ-PNV? ¿qué grupo se abstiene más?
- Matriz de afinidad de voto entre grupos.

**Enmiendas** (hoy: solo el valor "aprobada con enmienda"):
- ¿Qué grupo enmienda más las proposiciones de otros?
- ¿Cuántas proposiciones de EH Bildu acabaron enmendadas por el equipo de gobierno?
- ¿Qué enmiendas se rechazaron en votación, y por cuánto?  (bo:resultadoEnmienda + bo:votos*Enmienda)
- ¿Qué se cambió exactamente en la proposición de X?  (bo:resumenEnmienda)

**Resumen dispositivo + entidades limpias + sin truncado**:
- ¿Qué pedía EXACTAMENTE la proposición de X sobre Y? (respuesta precisa, no el título genérico)
- ¿Qué proposiciones mencionan a Iberdrola? (ahora fiable: "Iberdrola"/"Iberdrola SA" unificadas)
- ¿Qué empresas/organizaciones se citan más en el Pleno? (ranking real; hoy el 59% del texto se trunca antes de extraer)

**Multi-hop (el punto fuerte del grafo, ahora con más aristas)**:
- ¿Qué concejal del PP presentó más proposiciones de seguridad que fueron rechazadas?
- ¿En qué temas EH Bildu y el equipo de gobierno votan sistemáticamente distinto?

**Calidad transversal**: menos error temático (Gemini 2.5-Flash vs qwen3:8b local:
~10% → ~5%), menos "otros" en tema_principal, muchas más entidades capturadas.

### Lo que hay que verificar a mano tras la pasada
- `concejales.jsonl`: nombres bien, sin ruido de OCR en la cola, fechas de mandato.
- 20 `proposals_enriched.jsonl` al azar: `resumen` fiel, `votos_por_grupo` cuadra
  con el texto, `entidades` sin basura, `proponente_persona` correcto.
- Grafo: `bo:intervino` > 0, `bo:votoAFavorDe` en las props con desglose,
  `bo:Concejal` con `bo:perteneceA` para la mayoría.
- Añadir a `regression_qa.py`: "concejal más activo", "en qué intervino X",
  "cómo votó EH Bildu en <fecha>", "% de proposiciones con enmienda".

### COSTE REAL de la pasada de re-enriquecimiento (números reconstruidos del chroma_db)

Longitud REAL de las proposiciones: mediana ~19.000 chars, media ~28.000,
algunas 50k+ (hoy TODO se corta a 6.000). El prompt rico se repite ~3.900 veces.

| Escenario (corte de texto) | Tokens in | out | Gemini 2.5-Flash / 3.5-Flash-Lite ($0,30/$2,50 por M) | 3.6-Flash ($0,75/$3,75) |
|---|---|---|---|---|
| corte a 6k (como hoy) | ~10M | ~1,6M | ~6 € | ~13 € |
| **corte a 20k (implementado: `_ctx_chars` gemini=20000)** | ~18M | ~1,8M | **~10 €** | ~19 € |
| sin corte real (cap 40k) | ~26M | ~2M | ~13 € | ~25 € |
+ ~20% de colchón (reintentos de JSON, un par de pasadas de prueba con `--limit 20`).

- **Punto dulce (ya configurado): corte 20k + Gemini 2.5-Flash ≈ 10-12 €.**
- **Tope de gasto (implementado, Ronda 28): `--max-eur N`** — `enrich()` suma
  los tokens de cada llamada (`usage_metadata`, o estimados a 4 chars/tok si el
  proveedor no los da), calcula el coste con `PRECIO_USD_POR_M`/`USD_A_EUR` y
  PARA solo al superar N, dejando lo hecho guardado. Actualizar `PRECIO_USD_POR_M`
  si Google cambia precios. Calibrar antes con `--limit 100`.
- **Tiempo (API de pago, secuencial):** ~8 s/llamada × 3.900 ≈ 8-9 h de una tirada
  con `--rpm 60`. La latencia de inferencia manda, no el `--rpm`.
- **Batch API de Gemini = mitad de precio** (asíncrono, resultados en <24h) →
  cualquier cifra a la mitad. Encaja perfecto para un proceso de fondo como
  este. NO está implementado (usa la API síncrona con `--rpm`); si se quiere,
  hay que reescribir `enrich()` con el SDK `google-genai` en modo batch
  (submit job + poll). ~5 € con batch.
- El free tier también valdría (gratis) pero con tope diario → 4-16 días por
  lotes. Con billing activo no compensa.

### Lo que NO se ha preparado (queda para después)
- **Tarea 6** (proposiciones conjuntas → varios proponentes): `grupos.py` sin tocar.
- **Tarea 8** (ruegos y preguntas como nodos): sin empezar.
- **Tarea 9** (actas 2002-2006): faltan los PDF.
- **Tarea 10** (cifras de voto en casos raros): `parse_votos` sin tocar.
- **Tarea 4 completa** (clustering de entidades por distancia de edición + gazetteer
  de barrios/calles): `entidades.py` solo hace la limpieza determinista por-entidad.

---

> **Contexto rápido del pipeline actual** (3 fases + 1):
> 1. `extract_proposals.py` → `proposals.jsonl` (3.923 proposiciones, texto
>    truncado a 6.000 chars, agrupadas por `topic` de cada acta; **descarta
>    "General / Introducción"** = ruegos, preguntas, apertura, apartes del debate).
> 2. `build_graph.py --enrich` → `proposals_enriched.jsonl` (LLM por proposición
>    sobre `text[:5000]`: `tema_principal`, `temas`, `resultado`, `entidades` máx 8).
> 3. `build_rdf.py` → `bilbao_reasoned.ttl` (RDF + razonador OWL-RL).
> 4. `scripts/enrich_vector_metadata.py` → parchea el chroma_db con lo mismo.
>
> **Modelo actual**: Proposición —presentadaPor→ Grupo · —enPleno→ Pleno ·
> —tieneResultado→ Resultado · —trataSobre/trataTemaAmplio→ Tema · —menciona→ Entidad.
> Datos: `bo:fecha`, `bo:anio`, `bo:tituloTopic`, `bo:votosFavor/Contra`, `bo:votoTexto`.
>
> **Cobertura real**: ~2007 (parcial) – 2026. Las carpetas `actas/2002..2006`
> están VACÍAS (nunca se consiguieron esos PDF).
>
> **Regla de oro de todo el proyecto**: tras CUALQUIER cambio en el pipeline o
> los prompts, correr `python scripts/regression_qa.py` (baseline 30/30) antes
> de dar el cambio por bueno. Preferir guardas de código sobre reglas de prompt.

---

## RESUMEN DE PRIORIDADES

| # | Mejora | Impacto | Dificultad | Bloqueantes |
|---|--------|---------|-----------|-------------|
| **1** | Concejales individuales (quién presenta / quién interviene / a qué grupo pertenece) | 🔴 Muy alto | Media-alta | Ninguno (la extracción del censo YA existe) |
| **2** | Voto por grupo (a favor / en contra / abstención) | 🔴 Muy alto | Media | LLM o regex sobre el texto del voto |
| **3** | Quitar el truncado a 5.000/6.000 chars + re-enriquecer con un modelo grande | 🟠 Alto | Media | Cuota de Groq (tier de pago) o batching lento |
| **4** | Canonicalización / limpieza de entidades | 🟠 Alto | Media-alta | Ninguno |
| **5** | Resumen de la parte dispositiva + página del PDF como propiedades | 🟡 Medio | Baja | Ninguno |
| **6** | Proposiciones conjuntas → varios grupos proponentes | 🟡 Medio | Baja-media | Ninguno |
| **7** | Enmiendas como entidad de primer nivel | 🟡 Medio | Media | Ninguno |
| **8** | Ruegos y preguntas + apartes del debate como nodos | 🟢 Bajo-medio | Media | Ninguno |
| **9** | Conseguir e indexar las actas 2002-2006 | 🟢 Bajo-medio | Baja (si hay PDF) | Los PDF (portal de transparencia de Bilbao) |
| **10** | Cifras de voto en casos raros (~0,5%) + segmentación modo antiguo | 🟢 Bajo | Media | Ninguno |

Recomendación de orden: **1 → 2 → 3 → 4 → 5/6 → resto.**

---

## TAREA 1 — Concejales individuales

### Objetivo
Poder responder: *"¿qué ha dicho el/la concejal X sobre Y?"*, *"¿qué concejal ha
presentado/intervenido más?"*, *"concejales del grupo Z"*, *"¿en qué proposiciones
intervino X?"*. Hoy el grafo **solo tiene grupos**, ninguna persona como
proponente/interviniente/votante.

### Por qué es viable (no partimos de cero)
- `backend/rag.py::_get_party_mapping()` YA parsea la cabecera de cada acta y
  saca el **censo de concejales de la corporación** (`"En representación del
  grupo municipal X"` + `"N.- DON/DOÑA NOMBRE (partido)"`). Hoy se usa solo para
  el filtro de partido del chat vectorial.
- `backend/rag.py` línea ~610: `speaker_regex` YA captura los turnos
  `SR./SRA. APELLIDO:` y asigna `metadata['speaker']` a cada chunk (81% de
  cobertura en chroma_db).

### Datos nuevos a extraer
1. **Censo por mandato** (5 corporaciones en el rango 2007-2026: 2007-2011,
   2011-2015, 2015-2019, 2019-2023, 2023-2027). El acta de constitución de cada
   corporación (la primera de cada legislatura, o cualquiera con la sección
   "COMPOSICIÓN DE LA CORPORACIÓN") lista los ~29 concejales con nombre completo
   y grupo. → construir `graphrag/graphrag/concejales.jsonl`:
   `{nombre_completo, apellidos, grupo, mandato_desde, mandato_hasta}`.
   - Empezar adaptando `_get_party_mapping` a una función que devuelva la lista
     completa, no solo un dict apellido→partido.
   - Verificar a mano contra 2-3 actas de constitución (Wikipedia tiene la
     composición del Pleno de Bilbao por legislatura para contrastar).
2. **Turnos de intervención por proposición**: en `extract_proposals.py`,
   además del texto, guardar la lista de `speaker` distintos que aparecen en los
   chunks de esa proposición (ya está el dato en `c.metadata['speaker']`).
   → añadir campo `oradores: [apellido, ...]` a `proposals.jsonl`.
3. **Proponente persona** (opcional, más difícil): el título/primer párrafo a
   veces nombra a quien firma ("Proposición que presenta don ... del Grupo ...").
   Regex sobre el `topic`/`text`. Si no se puede, dejar solo `intervino`.

### Ontología (añadir a `ontology.ttl`)
```turtle
bo:Concejal a owl:Class ; rdfs:subClassOf bo:Persona ; rdfs:label "Concejal/a"@es .
bo:perteneceA a owl:ObjectProperty ; rdfs:domain bo:Concejal ; rdfs:range bo:Grupo ;
    rdfs:label "pertenece al grupo"@es .
bo:intervino a owl:ObjectProperty ; rdfs:domain bo:Proposicion ; rdfs:range bo:Concejal ;
    rdfs:label "intervino en"@es .
bo:proponePersona a owl:ObjectProperty ; rdfs:domain bo:Proposicion ; rdfs:range bo:Concejal ;
    rdfs:label "proponente (persona)"@es .   # opcional
```
Nota: `bo:Concejal ⊑ bo:Persona` — así las consultas viejas por `bo:menciona`
+ `bo:Persona` no se rompen, pero los concejales quedan diferenciables.

### build_rdf.py
- Cargar `concejales.jsonl`. Por cada concejal: nodo `br:concejal_<slug>` con
  `rdfs:label` (nombre completo) y `bo:perteneceA br:grupo_<slug>`.
- Enlazar apellido del acta (`oradores`) → concejal del censo de ESE mandato
  (por fecha de la proposición). Cuidado: apellidos repetidos entre grupos y
  legislaturas — desambiguar por (apellido + mandato). Si un apellido no
  resuelve, crear un `bo:Concejal` "suelto" con solo el apellido (mejor que
  perderlo) y marcarlo `bo:sinCensar true`.
- `?prop bo:intervino br:concejal_<slug>` por cada orador.

### SCHEMA de `graph_rag_sparql.py`
Añadir el patrón: *"¿qué dijo el concejal X?"* → `?prop bo:intervino ?c . ?c
rdfs:label ?nombre . FILTER(CONTAINS(LCASE(?nombre), "apellido"))`. Y "concejal
más activo" → `GROUP BY ?c ORDER BY DESC(COUNT)`.

### Verificación
- `concejales.jsonl` tiene ~29×5 ≈ 145 filas (menos si hay repeticiones entre
  legislaturas); contrastar 3 nombres por legislatura con Wikipedia.
- Consulta: "¿en qué proposiciones intervino IBAIBARRIAGA?" vs. contar chunks
  con `speaker=IBAIBARRIAGA` en chroma agrupados por proposición.
- Añadir 2-3 casos a `regression_qa.py` (GraphRAG): "concejal más activo",
  "proposiciones donde intervino X".

### Riesgos
- Apellidos ambiguos ("GARCÍA", "MUÑOZ" aparecen en varios grupos). Mitigar con
  el censo por mandato + el grupo del orador si el acta lo da.
- El `speaker_regex` falla en actas de formato antiguo (turnos sin "SR.:").
  Aceptar cobertura parcial (documentar el %).

---

## TAREA 2 — Voto por grupo

### Objetivo
*"¿Cómo votó EH Bildu en la proposición X?"*, *"¿en qué proposiciones votaron
juntos PP y PSE?"*, *"proposiciones aprobadas con el voto en contra de EAJ-PNV"*,
*"disciplina de voto por grupo"*. Hoy solo hay totales `votosFavor/votosContra`.

### Datos
El texto del voto (`vote_result`, ya en `proposals.jsonl`) y el debate a veces
detallan el sentido por grupo:
> "Sometida a votación... votan a favor los concejales de los grupos EAJ-PNV,
> Socialistas Vascos y EH BILDU; en contra, el Grupo Popular; se abstiene el
> Grupo GOAZEN BILBAO."

Opciones (probar en este orden):
1. **Regex + `grupos.py`**: buscar `"a favor"/"en contra"/"se abstien"` seguido
   de lista de grupos, mapear cada grupo con `extrae_grupo`/`canon_grupo`.
   Cubre el formato moderno estándar. Barato y determinista.
2. **LLM** (una llamada extra por proposición, o reusar la de la Tarea 3):
   añadir al `EXTRACT_PROMPT` un campo `"votos_por_grupo": {"favor": [...],
   "contra": [...], "abstencion": [...]}`. Cubre variantes raras. Coste: +1
   campo en cada llamada (barato si se hace junto a la re-enriquecida de T3).
3. Contraste: la suma de `favor` debe ≈ `votosFavor` (± los concejales no
   adscritos / ausencias). Usar como validación, no como filtro duro.

### Ontología
```turtle
bo:VotoGrupo a owl:Class ; rdfs:label "Voto de un grupo en una proposición"@es .
bo:tieneVoto a owl:ObjectProperty ; rdfs:domain bo:Proposicion ; rdfs:range bo:VotoGrupo .
bo:votoDe    a owl:ObjectProperty ; rdfs:domain bo:VotoGrupo ; rdfs:range bo:Grupo .
bo:sentido   a owl:DatatypeProperty ; rdfs:domain bo:VotoGrupo ; rdfs:range xsd:string .
    # "favor" | "contra" | "abstencion"
```
(Reificación: un nodo `br:voto_<prop>_<grupo>` con `votoDe` + `sentido`.)
Alternativa más simple si no se necesita metadatos del voto: propiedades
directas `bo:votoAFavorDe` / `bo:votoEnContraDe` / `bo:seAbstuvo` (Proposición→Grupo).
**Recomendado: la versión simple** (3 propiedades directas), menos nodos, SPARQL
más fácil para el LLM.

### build_rdf.py
Nueva función `parse_votos_por_grupo(vote_result, texto)` → dict; añadir los
triples. Si la proposición no tiene desglose, no añadir nada (no inventar).

### SCHEMA
Patrón: "cómo votó X en Y" / "coincidencias de voto" con `bo:votoAFavorDe` etc.

### Verificación
- 20 proposiciones a mano: comparar el desglose extraído con el PDF.
- Consulta de coherencia: `votosFavor` vs. nº de concejales de los grupos que
  votaron a favor (necesita la Tarea 1 para ser exacto; sin ella, aproximado).
- Casos nuevos en `regression_qa.py`.

---

## TAREA 3 — Sin truncado + re-enriquecer con modelo grande

### Problema
`extract_proposals.py` trunca a 6.000 chars y `build_graph.py` pasa `text[:5000]`
al LLM. **El 59% de las proposiciones supera 5.000 chars** → entidades y matices
del final se pierden. Y el modelo actual (qwen3:8b local) comete ~10% de error
temático y pierde entidades.

### Plan
1. **Subir `MAX_TEXT`** en `extract_proposals.py` a, p.ej., 20.000 (o sin
   límite y truncar solo en la Fase 2 según el modelo).
2. **Re-enriquecer con Groq `openai/gpt-oss-120b`** (128k de contexto, mucho
   mejor que qwen3:8b). `build_graph.py --enrich --model groq`.
   - Bloqueante: el free tier son 200.000 tokens/día. 3.923 proposiciones ×
     ~4.000 tokens (texto largo + prompt + salida) ≈ 16M tokens → **~80 días** a
     ese ritmo. Opciones:
     - **Dev tier de Groq** (de pago, barato) → se hace en horas.
     - Batching a 200k/día → ~10 semanas (inviable).
     - Modelo local mejor: `qwen2.5:32b` o `llama3.3:70b` si la RTX 2060 6GB lo
       aguanta con cuantización agresiva (lento, ~1-2 min/proposición → ~4-5
       días). Probar primero con `--limit 20` y comparar calidad contra groq.
   - **Recomendado: pagar el Dev tier de Groq para esta pasada.** Es la mejora
     de calidad más grande por menos esfuerzo.
3. Si el texto es demasiado largo aún para una llamada: trocear la proposición
   en 2-3 partes, enriquecer cada una, **unir** (`temas` = unión,
   `entidades` = unión deduplicada, `resultado`/`tema_principal` = de la parte
   con la votación).
4. Guardar el enriquecido viejo como `proposals_enriched.qwen3_8b.jsonl.bak`.
5. Regenerar grafo + parchear chroma.

### Verificación
- Comparar 30 proposiciones: entidades/temas nuevo vs viejo. Debe subir el nº
  de entidades y bajar el "otros" en `tema_principal`.
- `regression_qa.py` completo (las cifras de agregación pueden moverse un poco;
  actualizar los `expect` con los valores nuevos verificados por SPARQL).

---

## TAREA 4 — Canonicalización y limpieza de entidades

### Problema
6.274 nodos `Entidad`, pero:
- Duplicados: "Iberdrola" / "IBERDROLA" / "Iberdrola SA" / "Iberdrola, S.A."
  → `slug()` solo dedup exacto-ish; "iberdrola_sa" ≠ "iberdrola".
- Basura: `"A.A."`, `"#A. T. E."`, `"Aita"` (euskera: "padre"), `"Abbaunza"`
  (mal escrito, es el concejal Abaunza), siglas sueltas.
- Sin tipo fiable (persona/lugar/organización mal asignado a veces).

### Plan
Nuevo `graphrag/graphrag/entidades.py` con `canon_entidad(nombre, tipo) -> (nombre_canon, tipo_canon) | None`:
1. **Filtro de basura**: descartar si len < 4, si es solo mayúsculas+puntos de
   ≤4 chars, si tras normalizar es una stopword, si no tiene ninguna vocal, etc.
2. **Normalización**: quitar sufijos jurídicos (`, S\.?A\.?`, ` OAL`, ` SL`,
   ` SA`, ` S\.COOP`), quitar comillas, colapsar espacios, `strip_accents` para
   la CLAVE (no para el label mostrado).
3. **Clustering por similitud**: agrupar los que comparten la clave normalizada
   o tienen distancia de edición ≤2 sobre ≥6 chars. El representante del clúster
   = la forma más frecuente / más completa.
4. **Gazetteer** (lista curada, `entidades_conocidas.json`): barrios y calles de
   Bilbao (hay listas abiertas del Ayuntamiento), grandes organizaciones
   (Gobierno Vasco, Diputación, Metro Bilbao, BBK, Petronor, Iberdrola...),
   equipamientos (Guggenheim, Azkuna Zentroa/Alhóndiga, San Mamés...). Si un
   nombre casa con el gazetteer → usar la forma canónica del gazetteer + su tipo.
5. **Concejales**: los nombres de `entidades` de tipo persona que coincidan con
   el censo (Tarea 1) → enlazar al `bo:Concejal` en vez de crear un `bo:Persona`
   nuevo (unifica "menciona a un concejal" con "el concejal existe").

### build_rdf.py
Pasar cada entidad por `canon_entidad` antes de crear el nodo. Los `None` se
descartan.

### Verificación
- Antes/después: nº de nodos Entidad (debería BAJAR bastante), nº de
  proposiciones que mencionan "iberdrola" (debería subir al unificar variantes).
- Revisar a mano 50 clústeres al azar.

---

## TAREA 5 — Resumen dispositivo + página PDF

### Objetivo
Mejor narración y poder responder "¿qué pedía exactamente la proposición X?".
Hoy solo está `bo:tituloTopic` (la cabecera ASUNTO, muchas veces genérica:
"29. Proposición que presenta el Grupo Municipal...").

### Plan
- En `build_graph.py --enrich`, añadir campo `"resumen": "<1-2 frases: qué pide
  la parte dispositiva>"` al prompt (coste marginal, va en la misma llamada).
- `build_rdf.py`: `?prop bo:resumen "..."` y `?prop bo:pagina <int>` (el
  `page_ini` ya está en el enriquecido) y `?prop bo:fuentePdf "<ruta>"`.
- Ontología: `bo:resumen`, `bo:pagina` (xsd:integer), `bo:fuentePdf` (xsd:string),
  todos `rdfs:domain bo:Proposicion`.

### Verificación
20 resúmenes a mano. `regression_qa.py` sin cambios (solo añade datos).

---

## TAREA 6 — Proposiciones conjuntas → varios proponentes

### Problema
"Proposición conjunta que presentan los Grupos Municipales EH BILDU y ELKARREKIN"
→ hoy `extrae_grupo` devuelve UNO. La pregunta "¿cuántas conjuntas?" hoy
degrada con gracia (no rompe) pero no se puede contar ni desglosar.

### Plan
- `grupos.py::extrae_grupos_multi(topic, text) -> list[str]` (nueva, no romper
  `extrae_grupo`): detectar `"conjunta"` + `"y"`/`","` entre nombres de grupo.
- `build_rdf.py`: `bo:presentadaPor` pasa a ser multi-valor (ya lo permite RDF).
  Marcar `?prop bo:esConjunta true` cuando hay ≥2.
- Ojo: TODAS las consultas de "por grupo" ya funcionan (un `?prop` con 2
  `presentadaPor` cuenta para los 2 grupos — que es lo correcto).

### Verificación
Contar `bo:esConjunta true` vs. `grep -c "conjunta" proposals.jsonl` (aprox).
Actualizar el caso "proposiciones conjuntas" en `regression_qa.py` (ahora sí
tendrá respuesta real).

---

## TAREA 7 — Enmiendas de primer nivel

### Objetivo
"aprobada con enmienda" hoy es solo un valor de resultado. Se pierde: quién
enmendó, qué cambió. Muchísimas proposiciones (1.382 = 35%) acaban así.

### Plan
- `EXTRACT_PROMPT`: campo `"enmienda": {"por": "<grupo>", "tipo":
  "modificacion|adicion|sustitucion", "resumen": "..."}` o `null`.
- Ontología: `bo:Enmienda`, `bo:tieneEnmienda` (prop→enmienda), `bo:enmiendaPor`
  (enmienda→grupo), `bo:sentidoEnmienda`, `bo:resumenEnmienda`.
- `build_rdf.py`: crear el nodo cuando el enriquecido lo trae.

### Verificación
20 a mano. Consulta: "¿qué grupo enmienda más las proposiciones de otros?".

---

## TAREA 8 — Ruegos, preguntas y apartes del debate

### Problema
`extract_proposals.py` descarta `topic in {"General", "General / Introducción"}`.
Ahí van los ruegos, las preguntas orales, y menciones sueltas (era donde estaba
Tubacex). Son ~19.000 chunks del chroma_db.

### Plan
- Segunda extracción `extract_ruegos.py` (o rama en `extract_proposals.py`):
  agrupar los chunks "General" por (acta, orador) o por marcadores de "RUEGO:"/
  "PREGUNTA:" si el formato lo permite.
- Nodo `bo:RuegoPregunta` (subClassOf de nada, propio): `bo:enPleno`,
  `bo:planteadoPor` (→ Grupo o Concejal), `bo:menciona`, `bo:resumen`.
- NO enriquecer con el vocabulario temático de 19 (son heterogéneos); solo
  entidades + resumen.

### Verificación
Que Tubacex aparezca ahora también en el grafo (nodo RuegoPregunta del 27-05-2021).

---

## TAREA 9 — Actas 2002-2006

- Las carpetas `actas/2002..2006` están vacías. Conseguir los PDF (portal de
  transparencia / archivo del Ayuntamiento de Bilbao; puede requerir petición
  formal). Formato antiguo → probablemente `_process_single_pdf` necesite
  ajustes de regex (ya hay lógica "modo histórico").
- Si se consiguen: `python scripts/full_rebuild.py` (vector) + las 3 fases del
  grafo + `enrich_vector_metadata.py`. Es "solo" volumen.
- Impacto: +5 años, ~1.000 proposiciones más. Bajo esfuerzo SI hay PDF, alto SI
  hay que digitalizarlos.

---

## TAREA 10 — Cifras de voto en casos raros

- `parse_votos` coge el ÚLTIMO bloque de votación del texto; cuando hay votación
  de enmienda + votación final, puede coger la de la enmienda. ~0,5% del corpus.
- `build_rdf.py::parse_votos`: parsear TODOS los bloques, quedarse con el que
  va precedido de "de la proposición" / "en su conjunto" / "texto transaccional".
- Baja prioridad (volumen mínimo), pero fácil de hacer junto a la Tarea 2.

---

## SECUENCIA RECOMENDADA (una sesión por bloque)

1. **Sesión A** — Tarea 1 (concejales). Es la que más desbloquea y no depende
   de nada externo.
2. **Sesión B** — Tarea 2 (voto por grupo) + Tarea 10 (cifras raras), van juntas.
3. **Sesión C** — Tarea 3 (re-enriquecer con modelo grande). Requiere decidir
   antes: ¿se paga el Dev tier de Groq? Si sí, esta sesión también hace T5
   (resumen) y T7 (enmiendas) y T2-opción-LLM en la MISMA pasada de
   enriquecimiento (un solo prompt más rico = una sola pasada cara).
4. **Sesión D** — Tarea 4 (canonicalización de entidades) sobre el enriquecido
   nuevo.
5. **Sesión E** — Tareas 6 y 8 (conjuntas, ruegos).
6. **Cuando haya PDF** — Tarea 9.

Antes de la Sesión C conviene **rehacer el prompt de enriquecimiento entero**
(`EXTRACT_PROMPT`) con TODOS los campos nuevos de golpe (temas, resultado,
entidades, resumen, enmienda, votos_por_grupo, proponente_persona) para no pagar
la pasada cara varias veces.

## COSAS QUE NO CAMBIAR SIN PENSAR
- El roll-up temático OWL-RL (`trataTemaAmplio`, `broaderT`, `propertyChainAxiom`)
  funciona y está verificado — no tocar la mecánica.
- `bo:anio` SIN `rdfs:domain` a propósito (se usa en Pleno y Proposición).
- Los sanitizadores de `graph_rag_sparql.py::_sanitize_sparql` — si se añaden
  propiedades/clases, actualizar `_REAL_TEMA_URIS`, el SCHEMA y añadir guardas
  para los nuevos patrones que el LLM generará mal.
- `scripts/regression_qa.py` es la red de seguridad — ampliarla con cada tarea,
  nunca bajar el baseline.
