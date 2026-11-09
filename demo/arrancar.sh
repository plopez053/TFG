#!/usr/bin/env bash
# Arranca la interfaz web de la demo en http://localhost:8000
# Uso (desde la carpeta del proyecto):  bash demo/arrancar.sh
set -e
cd "$(dirname "$0")/.."

# Ollama hace falta para los embeddings (no para los LLM)
if ! curl -s http://localhost:11434/api/tags >/dev/null; then
    echo "[*] Arrancando Ollama..."
    (ollama serve >/dev/null 2>&1 &)
    sleep 3
fi

export LLM_SOLO_NUBE=1
exec .venv/bin/python -m chainlit run frontend/app.py --port "${PUERTO:-8000}"
