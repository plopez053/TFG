# -*- coding: utf-8 -*-
"""
Índice de proposiciones para el RAG vectorial: un documento corto por punto del
orden del día (título, quién lo presenta, temas, entidades, resumen y resultado).

Motivo: el vectorial solo buscaba entre fragmentos de 1.200 caracteres, y una
pregunta que no usa las palabras del acta ("cierre de oficinas bancarias" frente
a "malas prácticas bancarias y cierre de sucursales") no encontraba la
proposición. En el diagnóstico de recuperación solo el 55 % de las fechas
correctas llegaba a los candidatos. El resumen describe de qué trata la
proposición entera y se busca por significado; la recuperación (vectorial/recuperacion.py,
_propuestas_search) trae después los fragmentos de los debates de las que
salen. Es la misma información que tiene el grafo (resumen, temas, entidades),
sin más texto que el de las actas.

Se guarda en chroma_db_proposiciones/, aparte de la colección de fragmentos, que
no se toca. Si el directorio no existe, el RAG funciona como antes.

Uso:
    python scripts/indexar_proposiciones.py
"""
import sys, os, io, json, re, shutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from comun.rutas import CHROMA_PROPOSICIONES_PATH as DESTINO, dato_grafo  # noqa: E402

ENRIQUECIDAS = dato_grafo("proposals_enriched.jsonl")
# punto en euskera (las actas bilingües traen cada punto dos veces): el sistema es en castellano
_EUSKERA = re.compile(r"^\W*(\d+)\W*(proposizioa|proposamena|proposatzen)", re.I)
# cabeceras de sección: no son una proposición
_CABECERA = re.compile(r"^\d+\.\s*(PROPOSICIONES|MOCIONES DE URGENCIA)\b", re.I)


def _lista(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v) if isinstance(v, str) and v.strip().startswith("[") else []
    except ValueError:
        return []


def documento(d: dict) -> str:
    partes = [re.sub(r"\s+", " ", d.get("topic") or "")[:260]]
    if d.get("resumen"):
        partes.append(d["resumen"])
    if d.get("grupo") and d["grupo"] != "Desconocido":
        partes.append(f"Presenta: {d['grupo']}.")
    temas = [t for t in _lista(d.get("temas")) if isinstance(t, str)]
    if temas:
        partes.append("Temas: " + ", ".join(temas[:6]) + ".")
    ents = [e.get("nombre") for e in _lista(d.get("entidades")) if isinstance(e, dict) and e.get("nombre")]
    if ents:
        partes.append("Menciona: " + ", ".join(ents[:10]) + ".")
    if d.get("resultado") and d["resultado"] != "sin resultado":
        partes.append(f"Resultado: {d['resultado']}.")
    return " ".join(partes)


def main():
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings
    from comun.proveedores import EMBEDDING_MODEL

    docs, ids = [], []
    with open(ENRIQUECIDAS, encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            topic = d.get("topic") or ""
            if _EUSKERA.match(topic) or _CABECERA.match(topic.strip()) or not d.get("resumen"):
                continue
            docs.append(Document(page_content=documento(d), metadata={
                "prop_id": d["id"], "date": d.get("date", ""), "topic": topic[:300],
                "source": os.path.basename(d.get("source") or "")}))
            ids.append(d["id"])
    print(f"proposiciones a indexar: {len(docs)}")

    # directamente en el destino: en Windows no se puede renombrar un directorio con la base de
    # datos abierta. Si se interrumpe, basta volver a lanzarlo
    shutil.rmtree(DESTINO, ignore_errors=True)
    store = Chroma(collection_name="proposiciones", embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
                   persist_directory=DESTINO)
    for i in range(0, len(docs), 100):
        store.add_documents(docs[i:i + 100], ids=ids[i:i + 100])
        print(f"  {min(i + 100, len(docs))}/{len(docs)}", flush=True)
    print(f"[+] índice guardado en {DESTINO}")


if __name__ == "__main__":
    main()
