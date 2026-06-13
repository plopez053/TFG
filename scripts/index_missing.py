import os
import sys
import re
import glob

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
DATA_PATH = os.path.join(BASE_DIR, "actas")
EMBEDDING_MODEL = "nomic-embed-text"


def get_indexed_filenames(vector_store):
    indexed = set()
    offset = 0
    while True:
        res = vector_store.get(limit=10000, offset=offset, include=["metadatas"])
        if not res["metadatas"]:
            break
        for m in res["metadatas"]:
            if m and "source" in m:
                fname = m["source"].replace("\\", "/").split("/")[-1]
                indexed.add(fname)
        offset += 10000
    return indexed


def extract_date(path):
    match = re.search(r"(\d{2}-\d{2}-\d{4})", os.path.basename(path))
    return match.group(1) if match else "Fecha desconocida"


def get_party_mapping(pages):
    party_mapping = {}
    header_text = "\n".join([p.page_content for p in pages[:10]])
    current_party = "Goberno Local/Otros"
    for line in header_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        re_esp = re.search(r"En representación del grupo municipal\s+([A-Z\s-]+)", line, re.IGNORECASE)
        re_eus = re.search(r"([A-Z\s-]+)\s+udal talde politikoaren izenean", line, re.IGNORECASE)
        if re_esp:
            current_party = re_esp.group(1).strip().strip(":")
            continue
        if re_eus:
            current_party = re_eus.group(1).strip().strip(":")
            continue
        re_member = re.search(r"^\d+\.-?\s*(?:DON|DOÑA|SR\.|SRA\.)?\s*([A-ZÁÉÍÓÚÑ]{4,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)", line, re.IGNORECASE)
        if re_member:
            name = re_member.group(1).strip()
            paren = re.search(r"\(([^)]+)\)", line)
            party_mapping[name] = paren.group(1).strip() if paren else current_party
    return party_mapping


def process_pdf(path):
    date = extract_date(path)
    pages = PyPDFLoader(path).load()
    party_map = get_party_mapping(pages)

    full_text = "\n".join([p.page_content for p in pages])
    split_regex = re.compile(
        r"(?=\n(?:[\s]*)(?:\d+)\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA)"
        r"|\n\s*-\d+-\s*\n\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA|Proposición|Propuesta|Moción|Mozio|Dictamen|Irizpen|Asunto|Gaia|Proposamen))",
        re.IGNORECASE,
    )
    segments = split_regex.split(full_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    speaker_regex = re.compile(r"(?:(?:EL|LA)\s+)?(?:SR\.|SRA\.)\s+([A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)\s*[:\.]", re.IGNORECASE)

    chunks = []
    chunk_index = 0
    for segment in segments:
        if not segment.strip():
            continue
        current_speaker, current_party = "Desconocido", "Desconocido"
        topic_match_std = re.search(
            r"^\s*(\d+\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA).{0,400})",
            segment, re.IGNORECASE | re.DOTALL,
        )
        topic_match_hist = re.search(
            r"^\s*-\s*(\d+)\s*-\s*\n\s*((?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA|Proposición|Propuesta|Moción|Mozio|Dictamen|Irizpen|Asunto|Gaia|Proposamen).{0,400})",
            segment, re.IGNORECASE | re.DOTALL,
        )
        if topic_match_std:
            current_topic = topic_match_std.group(1).strip().replace("\n", " ")
        elif topic_match_hist:
            current_topic = f"{topic_match_hist.group(1)}. {topic_match_hist.group(2).strip().replace(chr(10), ' ')}"
        else:
            current_topic = "General / Introducción"

        for chunk_text in text_splitter.split_text(segment):
            match = speaker_regex.search(chunk_text)
            if match:
                current_speaker = match.group(1).strip()
                if len(current_speaker) < 50:
                    for kn, kp in party_map.items():
                        if kn in current_speaker or current_speaker in kn:
                            current_party = kp
                            break
            chunks.append(Document(
                page_content=f"ASUNTO: {current_topic}\nORADOR: {current_speaker} ({current_party})\n\n{chunk_text}",
                metadata={
                    "source": path, "date": date, "speaker": current_speaker,
                    "party": current_party, "topic": current_topic, "chunk_index": chunk_index,
                },
            ))
            chunk_index += 1
    return chunks


if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    print("Buscando PDFs no indexados...")
    indexed = get_indexed_filenames(vector_store)
    all_pdfs = glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True)
    missing = [p for p in all_pdfs if os.path.basename(p) not in indexed]

    if not missing:
        print("No faltan PDFs, la BD está completa.")
        sys.exit(0)

    print(f"PDFs a indexar: {len(missing)}")
    for p in missing:
        print(f"  - {os.path.basename(p)}")

    all_chunks = []
    for path in tqdm(missing, desc="Procesando", unit="pdf"):
        try:
            chunks = process_pdf(path)
            if chunks:
                all_chunks.extend(chunks)
                print(f"  {os.path.basename(path)}: {len(chunks)} chunks")
            else:
                print(f"  {os.path.basename(path)}: sin texto (probablemente escaneado)")
        except Exception as e:
            print(f"  [!] Error en {os.path.basename(path)}: {e}")

    if all_chunks:
        print(f"\nAñadiendo {len(all_chunks)} chunks a la BD...")
        vector_store.add_documents(all_chunks)
        print(f"[+] BD actualizada. Total ahora: {vector_store._collection.count()}")
    else:
        print("No se generaron chunks (todos los PDFs son escaneados).")
