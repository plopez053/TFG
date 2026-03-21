import os
import argparse
import re
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from tqdm import tqdm

# Configuration
DATA_PATH = "actas/pruebas actas_2025"
CHROMA_PATH = "chroma_db_test"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

class RAGPipeline:
    def __init__(self):
        # Initialize Embeddings
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = None

    def _extract_date_from_filename(self, path):
        """Extracts date from filename like '27-02-2025_...pdf'"""
        filename = os.path.basename(path)
        match = re.search(r'(\d{2}-\d{2}-\d{4})', filename)
        if match:
            return match.group(1)
        return "Fecha desconocida"

    def _get_party_mapping(self, pages):
        """Scans the header of the document to map names to parties."""
        party_mapping = {}
        # Take the first 10 pages for safety as headers can be long
        header_text = "\n".join([p.page_content for p in pages[:10]])
        
        current_party = "Goberno Local/Otros"
        lines = header_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Party headers:
            # "En representación del grupo municipal [PARTY]:"
            # "[PARTY] udal talde politikoaren izenean:"
            re_esp = re.search(r"En representación del grupo municipal\s+([A-Z\s-]+)", line, re.IGNORECASE)
            re_eus = re.search(r"([A-Z\s-]+)\s+udal talde politikoaren izenean", line, re.IGNORECASE)
            
            if re_esp:
                current_party = re_esp.group(1).strip().strip(':')
                continue
            if re_eus:
                current_party = re_eus.group(1).strip().strip(':')
                continue
            
            # Member detection: "3.- MARTA AJURIA ARRIBAS" or "8.- DON JOSEBA JAUREGI (EAJ-PNV)"
            re_member = re.search(r"^\d+\.-?\s*(?:DON|DOÑA|SR\.|SRA\.)?\s*([A-ZÁÉÍÓÚÑ]{4,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)", line, re.IGNORECASE)
            if re_member:
                name = re_member.group(1).strip()
                # Party in parentheses e.g. (EAJ-PNV)
                paren = re.search(r'\(([^)]+)\)', line)
                if paren:
                    party_mapping[name] = paren.group(1).strip()
                else:
                    party_mapping[name] = current_party
                    
        return party_mapping

    def load_and_split_documents(self):
        """Loads PDFs, extracts metadata (Date, Speaker, Party, Topic) and splits into chunks."""
        if not os.path.exists(DATA_PATH):
            print(f"Error: Directory '{DATA_PATH}' not found.")
            return []

        file_paths = glob.glob(os.path.join(DATA_PATH, '**', '*.pdf'), recursive=True)
        print(f"Encontrados {len(file_paths)} archivos PDF en {DATA_PATH}.")
        
        if not file_paths:
            return []

        all_chunks = []
        # Speaker interventions: "El Sr. ALCALDE:", "La Sra. AJURIA ARRIBAS:"
        speaker_regex = re.compile(
            r'(?:(?:EL|LA)\s+)?(?:SR\.|SRA\.)\s+([A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)\s*[:.]', 
            re.IGNORECASE
        )
        
        topic_regex = re.compile(
            r'^\s*(\d+\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA).*?)',
            re.IGNORECASE | re.MULTILINE
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )

        for path in tqdm(file_paths, desc="Procesando PDFs", unit="pdf"):
            try:
                date = self._extract_date_from_filename(path)
                pdf_loader = PyPDFLoader(path)
                pages = pdf_loader.load()
                
                party_map = self._get_party_mapping(pages)
                
                current_speaker = "Desconocido"
                current_party = "Desconocido"
                current_topic = "General / Sin Asunto Específico"
                
                for page in pages:
                    content = page.page_content
                    # We check for speaker and topic changes
                    lines = content.split('\n')
                    
                    # Accumulate text for the topic title if it spans multiple lines.
                    # For simplicity, we grab the first line of the proposal string.
                    for line in lines:
                        topic_match = topic_regex.search(line)
                        if topic_match:
                            candidate = topic_match.group(1).strip()
                            if len(candidate) > 10:
                                current_topic = candidate
                    
                    page_chunks = text_splitter.split_text(content)
                    
                    for chunk_text in page_chunks:
                        # Check for speaker change in the chunk
                        match = speaker_regex.search(chunk_text)
                        if match:
                            name_candidate = match.group(1).strip()
                            # Reject if it's too long or has low-case (regex already handles caps)
                            if len(name_candidate) < 50:
                                current_speaker = name_candidate
                                current_party = "Desconocido"
                                for kn in sorted(party_map.keys(), key=len, reverse=True):
                                    if kn in current_speaker or current_speaker in kn:
                                        current_party = party_map[kn]
                                        break
                        
                        # Create a chunk document with enriched metadata
                        all_chunks.append(Document(
                            page_content=chunk_text,
                            metadata={
                                "source": path,
                                "date": date,
                                "speaker": current_speaker,
                                "party": current_party,
                                "topic": current_topic,
                                "page": page.metadata.get("page", 0)
                            }
                        ))
                        
            except Exception as e:
                print(f"Error cargando {path}: {e}")
                
        print(f"Generados {len(all_chunks)} chunks con metadatos enriquecidos.")
        return all_chunks

    def create_vector_store(self, force_rebuild=False):
        """Creates or loads the ChromaDB vector store."""
        if os.path.exists(CHROMA_PATH) and not force_rebuild:
            print(f"Loading existing vector store from {CHROMA_PATH}...")
            self.vector_store = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=self.embeddings
            )
        else:
            if force_rebuild and os.path.exists(CHROMA_PATH):
                print(f"Deleting existing vector store for clean rebuild...")
                import shutil
                shutil.rmtree(CHROMA_PATH)
            
            chunks = self.load_and_split_documents()
            if not chunks:
                return
            print("Creando nuevos embeddings en ChromaDB...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_PATH
            )
            print(f"Vector store guardado en {CHROMA_PATH}")

    def _format_source_name(self, source_path):
        """Formats source path for display."""
        filename = os.path.basename(source_path)
        return os.path.splitext(filename)[0].replace('_', ' ')

    def query(self, question):
        """Queries the RAG system and returns an answer based purely on the documents."""
        if not self.vector_store:
            self.create_vector_store()

        llm = ChatOllama(model=LLM_MODEL, temperature=0)

        system_prompt = """Eres un asistente experto en Actas del Pleno del Ayuntamiento de Bilbao.
Tu objetivo es responder a las preguntas basándote ÚNICAMENTE en el contexto proporcionado.
ES OBLIGATORIO RESPONDER SIEMPRE EN CASTELLANO, independientemente del idioma de la pregunta.

**NORMAS ESTRICTAS:**
1. NO inventes información. Si la respuesta no está en el contexto, di "No hay información en las actas proporcionadas."
2. NO uses conocimientos externos.
3. El contexto puede incluir fragmentos en Euskera ("udalbatza", "proposamena") y Castellano. Traduce y resume la respuesta SIEMPRE al Castellano.
4. Identifica la intención de la pregunta:
   - ¿Es una pregunta sobre el sistema o los documentos disponibles? (CASO A)
   - ¿Es una pregunta sobre el contenido detallado o lo que dijo alguien? (CASO B)
   - ¿Es una pregunta pidiendo un listado de las propuestas o temas tratados? (CASO C)

### CASO A: Preguntas sobre el SISTEMA o DOCUMENTOS DISPONIBLES
(Ejemplos: "¿Qué actas tienes?", "¿De qué fechas son los documentos?", "¿Qué puedes hacer?")
- **ACCIÓN**: Enumera los documentos que ves en el contexto de forma clara y concisa (ej: "Tengo acceso al acta del 27-02-2025").

### CASO B: Preguntas sobre DATO ESPECÍFICO o CONTENIDO detallado
(Ejemplo: "¿Qué dijo X sobre Y?", "¿Dime todo sobre la propuesta Z?")
- **ACCIÓN**: Usa el contexto de abajo para responder OBLIGATORIAMENTE en este formato de ficha:
  - **Acta**: [Nombre del archivo]
  - **Fecha**: [Usar metadato 'fecha']
  - **Asunto/Propuesta**: [Usar metadato 'asunto' si la pregunta es sobre qué se aprobó o votó]
  - **Intervención**: "[Texto literal, si aplica]"
  - **Autor**: [Nombre del orador] ([Partido Político])
  - **Resumen/Contexto**: [Explicación detallada. IMPORTANTE: Si te preguntan por una propuesta completa, estructura tu resumen en 3 partes si el contexto lo permite: 1) De qué trata la propuesta. 2) Principales posturas en el debate. 3) Resultado de la votación (busca 'votos emitidos', 'acuerda', 'aprobado').]

### CASO C: Listado de PROPUESTAS / ASUNTOS
(Ejemplos: "¿Qué propuestas se hicieron el 29 de mayo?", "Enumera los temas tratados")
- **ACCIÓN**: NO uses el formato de ficha. Responde con una introducción amigable y enumera en una lista con viñetas los distintos asuntos o propuestas que encuentres en el contexto (fíjate en el campo [ASUNTO: ...] de cada fragmento). No repitas asuntos idénticos.

Contexto recuperado:
{context}

Pregunta del usuario: {question}"""
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        def format_docs(docs):
            formatted_texts = []
            for doc in docs:
                source = os.path.basename(doc.metadata.get("source", "Desconocido"))
                date = doc.metadata.get("date", "Fecha desconocida")
                speaker = doc.metadata.get("speaker", "Desconocido")
                party = doc.metadata.get("party", "Desconocido")
                topic = doc.metadata.get("topic", "General / Sin Asunto Específico")
                formatted_texts.append(f"[ACTA: {source}]\n[FECHA: {date}]\n[ASUNTO: {topic}]\n[ORADOR: {speaker} ({party})]\n{doc.page_content}")
            return "\n\n".join(formatted_texts)

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 7})
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(f"\nGenerando respuesta...")
        return chain.invoke(question)

def main():
    global DATA_PATH, CHROMA_PATH, LLM_MODEL
    
    parser = argparse.ArgumentParser(description="RAG System for Bilbao Actas")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of vector store")
    parser.add_argument("--query", type=str, help="Ask a specific question and exit")
    parser.add_argument("--path", type=str, default=DATA_PATH, help="Path to documents directory")
    parser.add_argument("--db", type=str, default=CHROMA_PATH, help="Path to chroma database")
    parser.add_argument("--model", type=str, default=LLM_MODEL, help="Ollama model for LLM")
    args = parser.parse_args()

    DATA_PATH = args.path
    CHROMA_PATH = args.db
    LLM_MODEL = args.model

    rag = RAGPipeline()
    
    if args.rebuild:
        rag.create_vector_store(force_rebuild=True)
    else:
        if not os.path.exists(CHROMA_PATH):
             rag.create_vector_store()
        else:
             rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)

    if args.query:
        print("\nRESPUESTA:")
        print(rag.query(args.query))
    else:
        print(f"\n--- RAG Bilbao Ready [Model: {LLM_MODEL}] (Tipo 'exit' para salir) ---")
        while True:
            q = input("\nPregunta: ")
            if q.lower() in ["exit", "quit"]: break
            if not q.strip(): continue
            print(rag.query(q))

if __name__ == "__main__":
    main()
