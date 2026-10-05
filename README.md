# Chatbot de las actas del Pleno del Ayuntamiento de Bilbao

Dos sistemas que responden en lenguaje natural sobre las actas del Pleno
(2007-2026) a partir de los mismos documentos, organizados en dos estructuras
de datos distintas:

- **RAG vectorial**: fragmentos de las actas en ChromaDB, recuperación híbrida
  (semántica, literal, temática y por proposición), reordenación y un LLM que
  redacta la respuesta citando las actas.
- **GraphRAG**: un grafo RDF de las proposiciones (ontología OWL, razonador
  OWL-RL) consultado con SPARQL; la consulta la monta el programa para los tipos
  de pregunta previstos o la genera un LLM, y otro LLM narra las filas.

## Estructura

```
frontend/app.py         interfaz web (Chainlit): perfiles "RAG Vectorial" y "GraphRAG (SPARQL)"
comun/                  lo que usan los dos sistemas
  rutas.py              rutas de los datos (actas, índices, grafo)
  proveedores.py        LLM (Groq, Gemini, Ollama) y reordenación (Cohere, local)
  grupos.py             grupos políticos del Pleno
vectorial/              RAG vectorial (punto de entrada: pipeline.py)
  indexado.py           de los PDF a los fragmentos en ChromaDB
  recuperacion.py       de la pregunta a los fragmentos: análisis, canales de búsqueda, reordenación y selección
  generacion.py         de los fragmentos a la respuesta: contexto, prompts, limpieza y fuentes
  pipeline.py           RAGPipeline: une las etapas; interfaz y línea de comandos
grafo/                  GraphRAG
  consulta/             respuesta a preguntas (punto de entrada: respuesta.py)
    recursos.py         el grafo RDF en memoria y los modelos de lenguaje
    pregunta.py         análisis de la pregunta, núcleo y asunto
    consulta_directa.py consulta montada por el programa, sin LLM
    plantillas.py       consulta por plantilla, sin LLM
    generacion.py       consulta generada por el LLM, con reintentos
    guardas.py          vocabulario del grafo y correcciones del SPARQL generado
    respuesta.py        graph_answer: orden de las vías, narración y fuentes
  construccion/         construcción del grafo a partir de las actas (ver su README)
datos/                  datos generados: grafo/ (RDF, JSONL y cachés)
chroma_db/              índice vectorial de fragmentos (no se versiona; se regenera)
chroma_db_proposiciones/  índice de resúmenes de proposiciones
scripts/                descarga de las actas e índices vectoriales
evaluacion/             regresión, RAGAS y estudio SPARQL
memoria/                memoria del TFG (LaTeX) y registro de decisiones técnicas
docs/                   guía de reconstrucción (RUNBOOK.md)
_archivo/               copias y código retirado, fuera del sistema
```

## Puesta en marcha

Requisitos: Python 3.11, `pip install -r requirements.txt`, Ollama en
`localhost:11434` con `nomic-embed-text`, `bge-m3`, `qwen3:8b` y `qwen2.5:7b`, y
un `.env` con `GROQ_API_KEY`, `COHERE_API_KEY` y el proyecto de Google Cloud para
Gemini (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`). Las actas en
PDF van en `actas/<año>/` (dentro del proyecto o en la carpeta de al lado).

```
python -m chainlit run frontend/app.py --port 8000   # interfaz web
python -m vectorial.pipeline --query "..."        # RAG vectorial desde la consola
python -m grafo.consulta.respuesta "..."          # GraphRAG desde la consola
```

## Evaluación

```
python evaluacion/regresion.py [graph|vector]     # 35 casos con resultado verificado contra las actas
python evaluacion/ragas/generar.py                # RAGAS, fase 1 (entorno principal)
.venv-ragas\Scripts\python.exe evaluacion\ragas\puntuar.py   # RAGAS, fase 2 (entorno de RAGAS)
```

Para reconstruir los datos desde cero: `docs/RUNBOOK.md`.
