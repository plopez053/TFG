import os
import sys
import shutil
import glob
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comun.rutas import CHROMA_PATH, DATA_PATH
from vectorial.pipeline import RAGPipeline
from langchain_chroma import Chroma


# purga de la DB los fragmentos de las actas dadas (la ruta se guardó con las dos barras)
def _purgar_actas(rag, pdf_files):
    for pdf_path in pdf_files:
        abs_path = os.path.abspath(pdf_path)
        for source in (abs_path, abs_path.replace("\\", "/")):
            try:
                rag.vector_store.delete(where={"source": source})
            except Exception:
                pass


def run_full_rebuild(year_limit=None, actas=None):
    print("=== SISTEMA DE RECONSTRUCCIÓN ESCALABLE DE BILBAO ===")

    rag = RAGPipeline()

    if actas:
        # actas sueltas por fecha (DD-MM-YYYY): se purgan y se vuelven a indexar solo esas
        pdf_files = sorted(p for fecha in actas
                           for p in glob.glob(os.path.join(DATA_PATH, "**", f"{fecha}_*.pdf"), recursive=True))
        faltan = [f for f in actas if not any(os.path.basename(p).startswith(f) for p in pdf_files)]
        if faltan:
            print(f"[!] Sin PDF para: {', '.join(faltan)}")
            return
        print(f"[*] Reconstrucción de {len(pdf_files)} actas concretas en {CHROMA_PATH}...")
        rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)
        _purgar_actas(rag, pdf_files)
        print("[+] Purga completada.")
        grupos = [("actas seleccionadas", pdf_files)]
    elif year_limit:
        print(f"[*] Reconstrucción quirúrgica activa: SOLO el año {year_limit}.")
        print(f"[*] Cargando base de datos existente en {CHROMA_PATH}...")
        rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)

        year_dir = os.path.join(DATA_PATH, str(year_limit))
        if os.path.exists(year_dir):
            pdf_files = glob.glob(os.path.join(year_dir, "*.pdf"))
            print(f"[*] Purgando fragmentos antiguos de {len(pdf_files)} actas de {year_limit} en la DB...")
            _purgar_actas(rag, pdf_files)
            print("[+] Purga completada.")
    else:
        if os.path.exists(CHROMA_PATH):
            print(f"[*] Borrando base de datos antigua en {CHROMA_PATH}...")
            shutil.rmtree(CHROMA_PATH)
            print("[+] Limpieza completada.")

    if not actas:
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
        grupos = [(f"AÑO {year}", glob.glob(os.path.join(DATA_PATH, year, "*.pdf"))) for year in years]

    for year, pdf_files in grupos:
        if not pdf_files:
            print(f"[-] {year}: No hay PDFs. Saltando...")
            continue

        print(f"\n{'='*50}")
        print(f"[*] PROCESANDO {year} ({len(pdf_files)} actas)")
        print("="*50)

        try:
            from tqdm import tqdm
            chunks = []
            for path in tqdm(pdf_files, desc=year, unit="pdf"):
                try:
                    chunks.extend(rag._process_single_pdf(path))
                except Exception as e:
                    print(f"[!] Error cargando {path}: {e}")

            if chunks:
                print(f"[*] {year}: {len(chunks)} fragmentos generados. Integrando en la DB...")

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

                print(f"[+] {year} integrado con éxito.")
            else:
                print(f"[!] {year}: No se generaron fragmentos.")

        except Exception as e:
            print(f"[ERROR] Fallo crítico procesando {year}: {e}")

    print(f"\n{'='*50}")
    print("[¡PROCESO COMPLETADO!]")
    print(f"Base de datos actualizada en: {CHROMA_PATH}")
    print(f"Total de fragmentos en la DB: {rag.vector_store._collection.count() if rag.vector_store else 0}")
    print("="*50)
    # AVISO: este rebuild solo genera embeddings + metadata básica
    # (date/topic/party/vote_result). Los campos que usa el canal de búsqueda
    # temática (_thematic_search) — tema_principal, temas, resultado,
    # grupo_proponente, prop_id — vienen de un pipeline SEPARADO
    # (extraer.py proposiciones -> build_graph.py --enrich -> build_rdf.py)
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
    print("      1) python -m grafo.construccion.extraer proposiciones")
    print("      2) python -m grafo.construccion.build_graph --enrich --model vertex")
    print("      3) python -m grafo.construccion.build_rdf")
    print("      4) python scripts/enrich_vector_metadata.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstrucción escalable de base de datos vectorial.")
    parser.add_argument("--year", type=int, help="Año específico a procesar (omite para reconstruir todo)")
    parser.add_argument("--actas", help="Actas concretas por fecha DD-MM-YYYY, separadas por comas (solo se reindexan esas)")
    args = parser.parse_args()
    run_full_rebuild(year_limit=args.year, actas=args.actas.split(",") if args.actas else None)
