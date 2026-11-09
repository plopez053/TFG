# Demo en un portátil Linux sin LLM locales

Con `LLM_SOLO_NUBE=1` en el `.env`, ningún LLM se ejecuta en Ollama:

| Tarea | Equipo de desarrollo | Demo |
|---|---|---|
| Respuesta del RAG vectorial | Groq → Gemini → Ollama | Groq → Gemini |
| SPARQL del GraphRAG | qwen3:8b (Ollama) → Gemini → Groq | Gemini → Groq |
| Narración del GraphRAG | Groq → Gemini → Ollama | Groq → Gemini |
| Embeddings | nomic-embed-text y bge-m3 (Ollama) | igual |
| Reordenación | Cohere, bge-m3 de respaldo | igual |

Ollama sigue haciendo falta, pero solo para los embeddings (dos modelos
pequeños, ~1 GB de memoria entre los dos).

## Pasos

1. Clonar el repositorio: `git clone https://github.com/plopez053/TFG.git && cd TFG`
2. Copiar a esa carpeta `datos_demo.tar.gz` y `claves_demo.tar.gz` (por USB,
   nunca por GitHub: el segundo lleva las claves) y descomprimirlos:
   `tar -xzf datos_demo.tar.gz && tar -xzf claves_demo.tar.gz`
3. `bash demo/instalar.sh` (comprueba los datos, crea `.venv`, instala las
   dependencias y descarga los dos modelos de embeddings)
4. `bash demo/arrancar.sh` y abrir http://localhost:8000

Requisitos: Python 3.11–3.13 con `venv` (`sudo apt install python3-venv` en
Ubuntu/Debian), Ollama, `curl` y conexión a internet.

## Si algo falla

- **"Failed to connect to Ollama"**: `ollama serve` en otra terminal.
- **Groq 429 / 413** (cuota por minuto): responde Gemini automáticamente.
- **Cohere agotado**: reordena bge-m3 en local; funciona, pero tarda más en
  un portátil sin GPU.
- **Puerto ocupado**: `PUERTO=8001 bash demo/arrancar.sh`
