# GraphRAG ontológico (RDF/OWL + razonador + SPARQL) vs RAG vectorial

Comparativa para el TFG: un **knowledge graph** modelado con una **ontología OWL** y consultado
por **SPARQL** (con razonador OWL-RL) frente al **RAG vectorial** (ChromaDB) que ya existe.
Foco: preguntas de **agregación / multi-hop**, donde el grafo + razonamiento sacan ventaja.

## Arquitectura / ficheros
| Fichero | Rol |
|---|---|
| `extract_proposals.py` | Extrae las proposiciones de los PDF → `proposals.jsonl` (YA generado: 3923) |
| `build_graph.py --enrich` | LLM extrae tema/entidades/resultado por proposición → `proposals_enriched.jsonl` |
| `ontology.ttl` | Ontología OWL: clases, jerarquías, propiedades, *property chain* para roll-up temático |
| `themes_skos.ttl` | Taxonomía SKOS de temas canónicos |
| `build_rdf.py` | Construye el RDF (ABox) + razonador **owlrl** → `bilbao_reasoned.ttl` |
| `graph_rag_sparql.py` | GraphRAG: pregunta → SPARQL → respuesta |
| `compare.py` | Comparación GraphRAG vs RAG vectorial |
| `build_graph.py --load` / `NEO4J_SETUP.md` | (Opcional) cargar el grafo en Neo4j solo para VISUALIZAR |

## ⚠️ El enriquecimiento LLM hay que hacerlo en el EQUIPO POTENTE
En el portátil (6 GB RAM, sin GPU) una extracción tarda ~200 s → inviable (días).
Groq gratis NO sirve: límite 100k tokens/día (~50 proposiciones/día). **Usar Ollama LOCAL con GPU.**

## Runbook (en el equipo potente)

```bash
# 0. dependencias (una vez)
pip install rdflib owlrl langchain-ollama
ollama pull qwen2.5:7b          # o el modelo que prefieras

# 1. (opcional) regenerar proposals.jsonl si reindexaste actas; si no, ya está
python graphrag/extract_proposals.py

# 2. enriquecer con LLM LOCAL (la parte larga; resumible: si se corta, relánzalo)
python graphrag/build_graph.py --enrich --model qwen2.5:7b
#    - resumible: salta las ya hechas; reintenta las fallidas (no las envenena)
#    - con GPU ~1-3 s/proposición → 3923 en ~2-3 h

# 3. construir el grafo RDF + razonar
python graphrag/build_rdf.py        # → graphrag/bilbao_reasoned.ttl

# 4. probar una consulta GraphRAG
python graphrag/graph_rag_sparql.py "¿Cuántas proposiciones sobre vivienda presentó cada grupo?"

# 5. comparación final GraphRAG vs vectorial
python graphrag/compare.py
```

## Estado actual (lo ya hecho en el portátil)
- `proposals.jsonl`: 3923 proposiciones extraídas (con el código de segmentación corregido).
- `proposals_enriched.jsonl`: 72 enriquecidas (las que entraron antes de agotar Groq); el resto pendiente.
- Ontología + razonador + SPARQL: validados end-to-end sobre datos parciales (el roll-up temático
  funciona: una consulta por "vivienda" agrega "vivienda social", "alquiler", etc. por inferencia).
- Extracción del **grupo proponente** desde el título (el metadato `party` era del orador): ~1800
  proposiciones con grupo (PP, EH BILDU, PSE-EE, UDALBERRI, GOAZEN BILBAO, ELKARREKIN...).

## Notas
- El grafo RDF vive en `bilbao_reasoned.ttl` (rdflib en memoria); no necesita servidor.
- Neo4j es OPCIONAL, solo para visualizar en el browser (`build_graph.py --load` + `NEO4J_SETUP.md`).
- La cuota de Groq es por cuenta/API key: usar Groq en otra máquina NO evita el límite.
