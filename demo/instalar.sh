#!/usr/bin/env bash
# Prepara la demo en un portátil Linux sin capacidad para LLM locales.
# Uso (desde la carpeta del proyecto):  bash demo/instalar.sh
set -e
cd "$(dirname "$0")/.."

echo "== 1. Datos que no están en GitHub"
falta=0
for f in chroma_db/chroma.sqlite3 chroma_db_proposiciones/chroma.sqlite3 actas/2024 \
         datos/grafo/bilbao_reasoned.ttl datos/grafo/concejales.jsonl .env; do
    if [ -e "$f" ]; then echo "   ok  $f"; else echo "   FALTA  $f"; falta=1; fi
done
ls gen-lang-client-*.json >/dev/null 2>&1 && echo "   ok  credenciales de Google" \
    || { echo "   FALTA  gen-lang-client-*.json (credenciales de Vertex/Gemini)"; falta=1; }
if [ "$falta" = 1 ]; then
    echo "Descomprime antes los paquetes en esta carpeta:"
    echo "   tar -xzf datos_demo.tar.gz && tar -xzf claves_demo.tar.gz"
    exit 1
fi

echo "== 2. Modo sin LLM locales en el .env"
# el .env viene de Windows: se quitan los retornos de carro
sed -i 's/\r$//' .env
grep -q '^LLM_SOLO_NUBE=' .env || printf '\nLLM_SOLO_NUBE=1\n' >> .env
grep '^LLM_SOLO_NUBE=' .env

echo "== 3. Entorno de Python"
PY=${PYTHON:-python3}
"$PY" --version
[ -d .venv ] || "$PY" -m venv .venv || {
    echo "No se pudo crear el entorno. En Ubuntu/Debian: sudo apt install python3-venv"; exit 1; }
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r demo/requirements.txt

echo "== 4. Modelos de embeddings en Ollama (solo estos dos, ~1,5 GB)"
command -v ollama >/dev/null || { echo "Ollama no está instalado"; exit 1; }
curl -s http://localhost:11434/api/tags >/dev/null || { (ollama serve >/dev/null 2>&1 &); sleep 3; }
ollama pull nomic-embed-text
ollama pull bge-m3

echo
echo "Listo. Para arrancar la demo:  bash demo/arrancar.sh"
