# Evaluación

Todo se ejecuta desde la raíz del proyecto.

| Carpeta o fichero | Qué mide | Cómo |
|---|---|---|
| `regresion.py` | 22 casos GraphRAG (filas del SPARQL) y 13 vectoriales (plenos que llegan al contexto), con el resultado verificado contra las actas. Se pasa tras cada cambio. | `python evaluacion/regresion.py [graph\|vector]` |
| `ragas/` | RAGAS del RAG vectorial: 50 preguntas con respuesta de referencia (`referencias.json`), juez `gemini-2.5-pro`. | `python evaluacion/ragas/generar.py`, después `.venv-ragas\Scripts\python.exe evaluacion\ragas\puntuar.py` |
| `estudio_sparql/` | Estudio de generación de SPARQL con 10 modelos y 35 preguntas de referencia (septiembre de 2026). Se hizo con el grafo y el código de entonces: sus resultados no se reproducen con el grafo actual. | `python evaluacion/estudio_sparql/estudio.py ejecutar\|analizar\|plantillas` |
