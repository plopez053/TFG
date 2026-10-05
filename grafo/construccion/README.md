# Construcción del grafo de conocimiento

Scripts que convierten las actas en el grafo RDF que consulta el GraphRAG
(`datos/grafo/bilbao_reasoned.ttl`). Se ejecutan a mano, en este orden, desde la
raíz del proyecto. Todos leen y escriben en `datos/grafo/` (rutas en
`comun/rutas.py`). Los pasos con LLM son reanudables. El porqué de cada decisión
está en `memoria/decisiones_tecnicas.md`; los modelos y costes, en
`docs/RUNBOOK.md`.

| Paso | Orden (`python -m grafo.construccion.…`) | Entrada → salida |
|---|---|---|
| 1 | `extraer concejales` | encabezado de las actas → `concejales.jsonl`, `personal_tecnico.jsonl` (grupo de cada concejal en `concejales_partido.json`, comprobado con `extraer constitutivas`) |
| 2 | `extraer proposiciones` | actas, con el troceo del índice vectorial → `proposals.jsonl` |
| 3 | `build_graph --enrich` | `proposals.jsonl` → `proposals_enriched.jsonl` (LLM: resumen, temas, entidades, resultado, votos) |
| 4 | `texto_actas` | actas escaneadas sin capa de texto → transcripción (OCR con Gemini) en `cache/pdf_texto/` |
| 5 | `extraer extra` | texto completo de cada acta → `proposals_extra.jsonl` (puntos que faltaban) y `votaciones.jsonl` |
| 6 | `build_graph --enrich --extra` | `proposals_extra.jsonl` → `proposals_enriched_extra.jsonl` |
| 7 | `debates intervenciones` | texto de los debates → `intervenciones.jsonl` (una por orador y punto, sin LLM) |
| 8 | `debates resumir` | `intervenciones.jsonl` → `intervenciones_resumen.jsonl` (LLM: resumen y postura) |
| 9 | `build_rdf` | todo lo anterior + `ontology.ttl` + `themes_skos.ttl` → `bilbao_reasoned.ttl` |
| 10 | `validate_shapes` | `bilbao_reasoned.ttl` contra `shapes.ttl` (SHACL) |

Los ficheros:

- `texto_actas.py`: texto de cada PDF página a página (en caché), OCR de las
  actas escaneadas, puntos del orden del día y fragmentos del índice vectorial.
- `extraer.py`: concejales, proposiciones y los puntos y votaciones que faltaban.
- `build_graph.py`: enriquecimiento de cada proposición con un LLM.
- `debates.py`: intervenciones, su resumen y los nombres propios citados.
- `votos.py`: listas nominales, voto por grupo y correcciones de recuentos,
  resultados y autores de enmiendas.
- `enriquecer.py`: lo que `build_rdf` añade al grafo antes del razonador OWL-RL
  (tipo de cada punto, votaciones, barrios, importes, plenos y legislaturas,
  y las correcciones de `votos.py` y `debates.py`).
- `build_rdf.py`: construye el grafo, lo enriquece, lo razona y lo guarda.
- `validate_shapes.py`: validación SHACL.
