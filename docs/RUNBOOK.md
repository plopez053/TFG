# RUNBOOK — reconstruir el sistema desde cero

Guía operativa: cómo montar el grafo y la base vectorial desde los PDF de las
actas, con las decisiones de coste y de modelo. El código solo hace lo que dice;
el "por qué" de las decisiones está aquí y en `../memoria/decisiones_tecnicas.md`.

## Requisitos

- Python 3.11, `pip install -r requirements.txt`
- Ollama en `localhost:11434` con los modelos: `nomic-embed-text` (embeddings),
  `bge-m3` (reordenación local y plantillas), `qwen3:8b` (SPARQL de GraphRAG) y
  `qwen2.5:7b` (narración de respaldo).
- `.env` con `GROQ_API_KEY`, `COHERE_API_KEY` y, para Gemini en Vertex AI,
  `GOOGLE_CLOUD_PROJECT` y `GOOGLE_APPLICATION_CREDENTIALS`.
- Los PDF de las actas en `actas/<año>/*.pdf` (fuera del repo).

## Pipeline completo

```
actas/*.pdf
  │  grafo.construccion.extraer proposiciones  (Fase 1: segmentación determinista)
  ▼
proposals.jsonl
  │  grafo.construccion.build_graph --enrich  (Fase 2: enriquecimiento LLM por proposición)
  ▼
proposals_enriched.jsonl  (+ puntos recuperados, votaciones e intervenciones:
  │                        ver grafo/construccion/README.md)
  │  grafo.construccion.build_rdf             (Fase 3: RDF + enriquecimiento + razonador OWL-RL)
  ▼
datos/grafo/bilbao_reasoned.ttl   ← lo consulta GraphRAG

actas/*.pdf
  │  scripts/full_rebuild.py     (embeddings + metadata básica)
  ▼
chroma_db/
  │  scripts/enrich_vector_metadata.py   (tema_principal, banderas tf_<tema>, grupo_proponente)
  ▼
chroma_db/  (enriquecido)        ← lo consulta el RAG vectorial
  │  scripts/indexar_proposiciones.py    (un resumen por proposición, canal de proposiciones)
  ▼
chroma_db_proposiciones/
```

## Fase 2 — enriquecimiento (`build_graph.py --enrich`)

```
python -m grafo.construccion.build_graph --enrich --model gemini --rpm 8 --max-eur 12
```

### Elección de modelo (`--model`)

| valor | modelo real | cuándo |
|---|---|---|
| `gemini` | `gemini-3.5-flash-lite` | por defecto; mejor relación coste/calidad |
| `gemini-3.5-flash` | — | ~5x más caro; solo si el lite se queda corto |
| `gemini-flash-latest` | — | intermedio |
| `groq` | `openai/gpt-oss-120b` | gratis pero el free tier limita mucho (200k tok/día) |
| un modelo de Ollama | — | local, gratis, más lento y menos preciso (ver `desarrollo/ESTUDIO_SPARQL_LOCAL.md`) |

`gemini-2.5-flash` y `2.5-flash-lite` **ya no están disponibles para claves
nuevas** (404). Si Google descomisiona `3.5-flash-lite`, actualizar
`GEMINI_ALIAS` y `PRECIO_USD_POR_M` en `grafo/construccion/build_graph.py`.

### Ritmo y coste

- `--rpm N`: peticiones por minuto (solo Gemini). El tier gratis admite ~8-10;
  con API de pago se puede subir a 60-120. `0` = sin pausa.
- `--max-eur N`: el script estima el gasto acumulado con la tabla
  `PRECIO_USD_POR_M` y para al superar N. Lo ya hecho queda guardado en
  `proposals_enriched.jsonl` (reanuda solo).
- **Antes de la tanda entera**, calibrar el coste real con `--limit 100`.
- Precios (USD por 1M de tokens, entrada/salida), comprobados en
  https://ai.google.dev/gemini-api/docs/pricing — actualizar si cambian:
  `3.5-flash-lite` 0.30/2.50 · `3.5-flash` 1.50/9.00 · `flash-latest` 0.75/3.75.
- Coste medido de la tanda completa (~3.400 proposiciones) con
  `3.5-flash-lite`: **~5 €**.

### Notas

- El límite de texto por proposición (`CTX_CHARS`) es 20k para Gemini y 30k para
  Groq. Los modelos locales pequeños truncan JSON con textos más largos.
- Un registro solo se escribe si sobrevivió al reintento + `json_repair`. Un
  registro sin temas ni entidades es legítimo (fragmento sin nada que
  clasificar), no un fallo.

## Fase 3 — grafo RDF (`build_rdf.py`)

```
python -m grafo.construccion.build_rdf
```

Combina `proposals_enriched.jsonl` + `concejales.jsonl` + `personal_tecnico.jsonl`
+ `ontology.ttl` + `themes_skos.ttl`, ejecuta el razonador OWL-RL y escribe
`bilbao_reasoned.ttl` con las inferencias materializadas (roll-up temático, tipos
de entidad).

`concejales.jsonl` se genera con `python -m grafo.construccion.extraer concejales` a partir de las 4 actas
de sesión constitutiva (ver `../memoria/decisiones_tecnicas.md` §5).

## Base vectorial

```
python scripts/full_rebuild.py                 # embeddings + metadata básica
python scripts/enrich_vector_metadata.py       # tema_principal, tf_<tema>, grupo_proponente
```

`full_rebuild.py` NO calcula `tema_principal` ni las banderas de tema (eso
implica LLM); hay que correr `enrich_vector_metadata.py` después o el canal de
búsqueda temática ignora las actas nuevas.

## Verificación

```
python evaluacion/regresion.py                 # 35 casos, GraphRAG + vectorial
```
