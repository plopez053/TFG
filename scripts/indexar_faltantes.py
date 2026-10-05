# -*- coding: utf-8 -*-
"""
Añade a la base vectorial (ChromaDB) las actas que no estaban indexadas, con el
mismo troceo y metadatos que el resto (RAGPipeline._process_single_pdf: tema,
orador, página, resultado de la votación). index_missing.py usaba una copia
antigua del troceo, sin página ni resultado.

    python scripts/indexar_faltantes.py          # lista y añade las que faltan
    python scripts/indexar_faltantes.py --ver    # solo lista
"""
import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import vectorial.indexado as _rag  # noqa: E402
from comun.rutas import CACHE_GRAFO, DATA_PATH  # noqa: E402
from vectorial.pipeline import RAGPipeline  # noqa: E402

# Actas escaneadas (sin capa de texto): se usa la transcripción de
# grafo/construccion/texto_actas.py en lugar de PyPDFLoader
_OCR = os.path.join(CACHE_GRAFO, "pdf_texto")
_PyPDFLoader = _rag.PyPDFLoader


class _LectorConOCR:
    def __init__(self, path):
        self.path = path

    def load(self):
        import json
        reg = os.path.join(_OCR, "_ocr.json")
        nombre = os.path.basename(self.path)
        if os.path.exists(reg) and nombre in json.load(open(reg, encoding="utf-8")):
            from langchain_core.documents import Document
            pags = json.load(open(os.path.join(_OCR, nombre + ".json"), encoding="utf-8"))
            return [Document(page_content=t, metadata={"source": self.path, "page": i}) for i, t in enumerate(pags)]
        return _PyPDFLoader(self.path).load()


_rag.PyPDFLoader = _LectorConOCR


def indexadas(store):
    nombres, off = set(), 0
    while True:
        r = store.get(include=["metadatas"], limit=10000, offset=off)
        if not r["ids"]:
            break
        nombres |= {os.path.basename(str(m.get("source", "")).replace("\\", "/")) for m in r["metadatas"] if m}
        off += len(r["ids"])
    return nombres


if __name__ == "__main__":
    rag = RAGPipeline()
    rag.create_vector_store()
    store = rag.vector_store
    ya = indexadas(store)
    pdfs = sorted(glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True))
    faltan = [p for p in pdfs if os.path.basename(p) not in ya]
    print(f"[*] {len(pdfs)} actas, {len(ya)} indexadas, faltan {len(faltan)}:")
    for p in faltan:
        print("   ", os.path.basename(p))
    if "--ver" in sys.argv or not faltan:
        sys.exit(0)
    for p in faltan:
        chunks = rag._process_single_pdf(p)
        if chunks:
            store.add_documents(chunks)
        print(f"[+] {os.path.basename(p)}: {len(chunks)} fragmentos", flush=True)
    print(f"[+] total en la colección: {store._collection.count()}")
