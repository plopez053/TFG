import os
import sys
import shutil
import glob
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import RAGPipeline, DATA_PATH, CHROMA_PATH
from langchain_chroma import Chroma


def run_full_rebuild(year_limit=None):
    print("=== SISTEMA DE RECONSTRUCCIÓN ESCALABLE DE BILBAO ===")

    rag = RAGPipeline()

    if year_limit:
        print(f"[*] Reconstrucción quirúrgica activa: SOLO el año {year_limit}.")
        print(f"[*] Cargando base de datos existente en {CHROMA_PATH}...")
        rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)

        year_dir = os.path.join(DATA_PATH, str(year_limit))
        if os.path.exists(year_dir):
            pdf_files = glob.glob(os.path.join(year_dir, "*.pdf"))
            print(f"[*] Purgando fragmentos antiguos de {len(pdf_files)} actas de {year_limit} en la DB...")
            for pdf_path in pdf_files:
                abs_path = os.path.abspath(pdf_path)
                try:
                    rag.vector_store.delete(where={"source": abs_path})
                except Exception:
                    pass
                try:
                    rag.vector_store.delete(where={"source": abs_path.replace("\\", "/")})
                except Exception:
                    pass
            print("[+] Purga completada.")
    else:
        if os.path.exists(CHROMA_PATH):
            print(f"[*] Borrando base de datos antigua en {CHROMA_PATH}...")
            shutil.rmtree(CHROMA_PATH)
            print("[+] Limpieza completada.")

    if year_limit:
        years = [str(year_limit)]
    else:
        if not os.path.exists(DATA_PATH):
            print(f"[!] No se encontró la ruta de datos: {DATA_PATH}")
            return
        years = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])

    if not years:
        print("[!] No se encontraron carpetas de años para procesar.")
        return

    print(f"[*] DATA_PATH: {DATA_PATH}")
    print(f"[*] Años a procesar: {years}")

    for year in years:
        year_dir = os.path.join(DATA_PATH, year)
        pdf_files = glob.glob(os.path.join(year_dir, "*.pdf"))

        if not pdf_files:
            print(f"[-] Año {year}: No hay PDFs. Saltando...")
            continue

        print(f"\n{'='*50}")
        print(f"[*] PROCESANDO AÑO: {year} ({len(pdf_files)} actas)")
        print("="*50)

        try:
            from tqdm import tqdm
            chunks = []
            for path in tqdm(pdf_files, desc=f"Año {year}", unit="pdf"):
                try:
                    chunks.extend(rag._process_single_pdf(path))
                except Exception as e:
                    print(f"[!] Error cargando {path}: {e}")

            if chunks:
                print(f"[*] Año {year}: {len(chunks)} fragmentos generados. Integrando en la DB...")

                # Lotes pequeños (100, no 1000): Ollama sirve los embeddings a través de
                # un proceso "runner" interno con puerto dinámico; con peticiones muy
                # grandes (1000 textos de golpe) ese runner puede reciclarse a mitad de
                # petición y el cliente se queda apuntando a un puerto ya muerto
                # (ResponseError "dial tcp ...: connectex ... denegó la conexión").
                # Con lotes pequeños + reintento el fallo puntual solo repite ~100 docs.
                sub_batch_size = 100
                for i in range(0, len(chunks), sub_batch_size):
                    sub_batch = chunks[i:i + sub_batch_size]
                    lote_num = i // sub_batch_size + 1
                    total_lotes = (len(chunks) + sub_batch_size - 1) // sub_batch_size
                    print(f"   -> Sub-lote {lote_num}/{total_lotes}: {len(sub_batch)} fragmentos...")

                    for intento in range(3):
                        try:
                            if rag.vector_store is None:
                                rag.vector_store = Chroma.from_documents(
                                    documents=sub_batch,
                                    embedding=rag.embeddings,
                                    persist_directory=CHROMA_PATH
                                )
                            else:
                                rag.vector_store.add_documents(sub_batch)
                            break
                        except Exception as e:
                            if intento < 2:
                                print(f"      [!] Fallo de embeddings (intento {intento+1}/3): "
                                      f"{type(e).__name__}: {str(e)[:150]} — reintentando en 5s...")
                                time.sleep(5)
                            else:
                                raise

                print(f"[+] Año {year} integrado con éxito.")
            else:
                print(f"[!] Año {year}: No se generaron fragmentos.")

        except Exception as e:
            print(f"[ERROR] Fallo crítico procesando el año {year}: {e}")

    print(f"\n{'='*50}")
    print("[¡PROCESO COMPLETADO!]")
    print(f"Base de datos actualizada en: {CHROMA_PATH}")
    print(f"Total de fragmentos en la DB: {rag.vector_store._collection.count() if rag.vector_store else 0}")
    print("="*50)
    # AVISO: este rebuild solo genera embeddings + metadata básica
    # (date/topic/party/vote_result). Los campos que usa el canal de búsqueda
    # temática (_thematic_search) — tema_principal, temas, resultado,
    # grupo_proponente, prop_id — vienen de un pipeline SEPARADO
    # (extract_proposals.py -> build_graph.py --enrich -> build_rdf.py)
    # y de scripts/enrich_vector_metadata.py, que NO se invocan aquí porque
    # implican llamadas a LLM (coste/tiempo que este script no debe decidir
    # por su cuenta). Sin ese segundo paso, las actas nuevas indexadas ahora
    # quedan SIN esos campos y _thematic_search las ignora en silencio para
    # preguntas generales sobre esos temas — no es un error, solo un paso
    # pendiente. Si se han añadido actas nuevas, ejecutar en este orden:
    print("[!] RECORDATORIO: si se han añadido actas NUEVAS, este rebuild NO")
    print("    incluye el enriquecimiento temático del RAG vectorial todavía.")
    print("    Para que el canal de búsqueda por tema (_thematic_search) las")
    print("    cubra, ejecutar en orden:")
    print("      1) python graphrag/graphrag/construccion/extract_proposals.py")
    print("      2) python graphrag/graphrag/construccion/build_graph.py --enrich --model qwen3:8b")
    print("      3) python graphrag/graphrag/construccion/build_rdf.py")
    print("      4) python scripts/enrich_vector_metadata.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstrucción escalable de base de datos vectorial.")
    parser.add_argument("--year", type=int, help="Año específico a procesar (omite para reconstruir todo)")
    args = parser.parse_args()
    run_full_rebuild(year_limit=args.year)
