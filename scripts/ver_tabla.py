import os
import sys

# Agregar el directorio backend al PYTHONPATH para poder importar rag
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from collections import Counter

# Forzar UTF-8 en la salida para evitar errores de encoding en Windows
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_final_v11")
DATA_PATH = os.path.join(BASE_DIR, "actas_scalability")
EMBEDDING_MODEL = "nomic-embed-text"

def print_some_metadatas():
    # 1. PDFs on disk with their parent directory name
    pdfs_on_disk = {}
    for root, dirs, files in os.walk(DATA_PATH):
        for f in files:
            if f.endswith(".pdf"):
                parent = os.path.basename(root)
                pdfs_on_disk[f] = parent
                
    print(f"Total PDFs en disco: {len(pdfs_on_disk)}")
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    print("Leyendo fuentes en la BD...")
    batch_size = 10000
    offset = 0
    unique_sources = set()
    
    while True:
        res = vector_store.get(limit=batch_size, offset=offset, include=['metadatas'])
        metadatas = res['metadatas']
        if not metadatas:
            break
        
        for meta in metadatas:
            if meta and 'source' in meta:
                unique_sources.add(meta['source'])
        
        offset += batch_size
            
    print(f"Total fuentes unicas en BD: {len(unique_sources)}")
    
    # Let's count by year folder
    disk_by_year = Counter(pdfs_on_disk.values())
    db_by_year = Counter()
    for s in unique_sources:
        parts = s.replace('/', os.sep).split(os.sep)
        if len(parts) >= 2:
            db_by_year[parts[-2]] += 1
            
    print("\n=== PDFs por año: Disco vs ChromaDB ===")
    total_ok = 0
    total_missing = 0
    for yr in sorted(list(disk_by_year.keys())):
        disk_n = disk_by_year[yr]
        db_n = db_by_year.get(yr, 0)
        if disk_n == db_n:
            status = "[OK]"
            total_ok += disk_n
        elif db_n == 0:
            status = "[FALTA]"
            total_missing += disk_n
        else:
            status = f"[INCOMPLETO falta {disk_n - db_n}]"
            total_missing += (disk_n - db_n)
        print(f"  Año {yr}: Disco={disk_n} | BD={db_n}  {status}")
    
    print(f"\nResumen: {total_ok}/{len(pdfs_on_disk)} PDFs indexados | Faltan: {total_missing}")
        
if __name__ == "__main__":
    print_some_metadatas()
