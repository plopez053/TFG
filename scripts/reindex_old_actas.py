"""
Borra y re-indexa los PDFs de 2007-2010 con el nuevo extractor de topics
para formato antiguo (secciones separadas por "- N -").
"""
import os
import sys
import glob
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
from backend.rag import RAGPipeline, CHROMA_PATH, DATA_PATH, EMBEDDING_MODEL

YEARS_TO_REINDEX = [2007, 2008, 2009, 2010]

def get_dates_for_years(years):
    dates = []
    for year in years:
        year_dir = os.path.join(DATA_PATH, str(year))
        if not os.path.exists(year_dir):
            continue
        for f in glob.glob(os.path.join(year_dir, "*.pdf")):
            m = re.search(r'(\d{2}-\d{2}-\d{4})', os.path.basename(f))
            if m:
                dates.append(m.group(1))
    return list(set(dates))


def delete_chunks_for_dates(vs, dates):
    """Elimina de ChromaDB todos los chunks cuya fecha esté en la lista."""
    total_deleted = 0
    # Procesar en lotes pequeños para no saturar SQLite
    batch = 20
    for i in range(0, len(dates), batch):
        batch_dates = dates[i:i + batch]
        try:
            res = vs.get(where={"date": {"$in": batch_dates}})
            ids = res["ids"]
            if ids:
                vs._collection.delete(ids=ids)
                total_deleted += len(ids)
                print(f"  Eliminados {len(ids)} chunks ({batch_dates[0]}...)")
        except Exception as e:
            print(f"  Error eliminando {batch_dates}: {e}")
    return total_deleted


def main():
    print(f"[*] Re-indexando actas de {YEARS_TO_REINDEX}...")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 1. Obtener fechas a eliminar
    dates = get_dates_for_years(YEARS_TO_REINDEX)
    print(f"[*] Fechas a eliminar: {len(dates)} ({sorted(dates)[:3]}...)")

    # 2. Eliminar de ChromaDB
    print("[*] Borrando chunks antiguos...")
    deleted = delete_chunks_for_dates(vs, dates)
    print(f"[+] {deleted} chunks eliminados.")

    # 3. Verificar conteo actual
    count_before = vs._collection.count()
    print(f"[*] Chunks restantes en BD: {count_before}")

    # 4. Re-indexar con la nueva lógica
    print("[*] Re-indexando con nuevo extractor de topics...")
    rag = RAGPipeline()
    rag.vector_store = vs

    pdf_files = []
    for year in YEARS_TO_REINDEX:
        pdf_files.extend(glob.glob(os.path.join(DATA_PATH, str(year), "*.pdf")))

    print(f"[*] PDFs a procesar: {len(pdf_files)}")

    all_chunks = []
    for path in tqdm(pdf_files, desc="Procesando", unit="pdf"):
        try:
            chunks = rag._process_single_pdf(path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [!] Error en {os.path.basename(path)}: {e}")

    if not all_chunks:
        print("[!] No se generaron chunks. Revisa el log.")
        return

    print(f"\n[*] Chunks generados: {len(all_chunks)}")

    # Verificar muestra de topics
    from collections import Counter
    topics_sample = Counter(c.metadata.get("topic", "")[:50] for c in all_chunks)
    general_count = sum(v for k, v in topics_sample.items() if k.startswith("General"))
    other_count = len(all_chunks) - general_count
    print(f"[*] Topics reales (no General): {other_count} | General/Intro: {general_count}")
    print("[*] Muestra de topics encontrados:")
    for topic, count in topics_sample.most_common(8):
        if not topic.startswith("General"):
            print(f"    ({count}x) {topic}")

    # 5. Añadir a ChromaDB en lotes
    print(f"\n[*] Añadiendo {len(all_chunks)} chunks a la BD...")
    batch_size = 500
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexando", unit="lote"):
        vs.add_documents(all_chunks[i:i + batch_size])

    final_count = vs._collection.count()
    print(f"\n[+] Re-indexado completo. Total en BD: {final_count}")


if __name__ == "__main__":
    main()
