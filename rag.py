import os
import argparse
import re
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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

    def load_and_split_documents(self):
        """Loads PDFs from the data path and splits them into chunks, with progress bar."""
        if not os.path.exists(DATA_PATH):
            print(f"Error: Directory '{DATA_PATH}' not found.")
            return []

        print(f"Loading PDFs from {DATA_PATH}...")
        # Recursive loading of PDFs
        import glob
        # Buscar todos los PDFs recursivamente
        file_paths = glob.glob(os.path.join(DATA_PATH, '**', '*.pdf'), recursive=True)
        print(f"Buscando archivos PDF en {DATA_PATH}...")
        print(f"Encontrados {len(file_paths)} archivos PDF.")
        if len(file_paths) == 0:
            print("¡No se encontraron archivos PDF! Revisa la ruta y las subcarpetas.")
            return []
        documents_with_speaker = []
        speaker_regex = re.compile(r'(SR\.|SRA\.)\s+([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)')
        
        for path in tqdm(file_paths, desc="Cargando PDFs", unit="pdf"):
            try:
                pdf_loader = PyPDFLoader(path)
                pages = pdf_loader.load()
                
                current_speaker = "Desconocido"
                for page in pages:
                    # Look for new speakers in this page
                    matches = speaker_regex.findall(page.page_content)
                    if matches:
                        # Take the last one found on the page as the "current" for subsequent content
                        # unless it's at the very end. For simplicity, we take the last one.
                        prefix, name = matches[-1]
                        current_speaker = f"{prefix} {name}"
                    
                    page.metadata['speaker'] = current_speaker
                    documents_with_speaker.append(page)
                    
            except Exception as e:
                print(f"Error cargando {path}: {e}")
        print(f"Loaded {len(documents_with_speaker)} document pages with speaker metadata.")

        # Split text with progress bar
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            add_start_index=True,
        )
        chunks = []
        for doc in tqdm(documents_with_speaker, desc="Splitting", unit="doc"):
            try:
                # split_documents will preserve metadata
                chunks.extend(text_splitter.split_documents([doc]))
            except Exception as e:
                print(f"Error al dividir documento: {e}")
        print(f"Split into {len(chunks)} chunks.")
        return chunks

    def create_vector_store(self, force_rebuild=False):
        """Creates or loads the ChromaDB vector store, with progress bar for embeddings."""
        if os.path.exists(CHROMA_PATH) and not force_rebuild:
            print(f"Loading existing vector store from {CHROMA_PATH}...")
            self.vector_store = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=self.embeddings
            )
        else:
            if force_rebuild and os.path.exists(CHROMA_PATH):
                print(f"Deleting existing vector store at {CHROMA_PATH} for a clean rebuild...")
                import shutil
                shutil.rmtree(CHROMA_PATH)
            
            chunks = self.load_and_split_documents()
            if not chunks:
                return
            print("Creating new vector store (esto puede tardar, los embeddings se generan internamente)...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_PATH
            )
            print(f"Vector store saved to {CHROMA_PATH}")

    def _format_source_name(self, source_path):
        """Formats a filename like '27-02-2025_Ordinaria_Acta' into 'actas ordinaria del 27 de febrero de 2025'."""
        filename = os.path.basename(source_path)
        # Remove extension
        name_without_ext = os.path.splitext(filename)[0]
        
        parts = name_without_ext.split('_')
        if len(parts) < 2:
            return name_without_ext # Fallback
            
        date_str = parts[0]
        tipo = parts[1].lower()
        
        # Parse date
        date_parts = date_str.split('-')
        if len(date_parts) != 3:
            return name_without_ext
            
        day, month, year = date_parts
        months = {
            "01": "enero", "02": "febrero", "03": "marzo", "04": "abril",
            "05": "mayo", "06": "junio", "07": "julio", "08": "agosto",
            "09": "septiembre", "10": "octubre", "11": "noviembre", "12": "diciembre"
        }
        month_name = months.get(month, month)
        
        return f"acta {tipo} del {day} de {month_name} de {year}"

    def get_stats(self):
        """Returns the number of documents, chunks and formatted names in the vector store."""
        if not self.vector_store:
            return 0, 0, []
        try:
            items = self.vector_store.get()
            total_chunks = len(items['ids']) if 'ids' in items else 0
            # Source metadata is usually 'source'
            sources = set()
            if 'metadatas' in items:
                for meta in items['metadatas']:
                    if meta and 'source' in meta:
                        sources.add(meta['source'])
            
            formatted_names = sorted([self._format_source_name(s) for s in sources])
            print(f"Sources: {formatted_names}")
            return len(sources), total_chunks, formatted_names
        except Exception as e:
            print(f"Error getting stats: {e}")
            return 0, 0, []

    def query(self, question, chat_history):
        """Queries the RAG system with conversation history."""
        if not self.vector_store:
            self.create_vector_store()

        # Set up LLM
        # Set up LLM with temperature 0 to avoid hallucinations
        llm = ChatOllama(model=LLM_MODEL, temperature=0)

        num_docs, total_chunks, doc_names = self.get_stats()
        doc_list_str = "\n".join([f"- {name}" for name in doc_names])
        
        # 0. System setup
        system_prompt = f"""# IDENTIDAD Y ROL
Eres un experto historiador y asistente especializado en las actas de los plenos del Ayuntamiento de Bilbao. Tu propósito es recuperar y presentar información de forma fidedigna basándote exclusivamente en los registros oficiales que posees.

# REGLA FUNDAMENTAL DE IDIOMA
- Responde UNICA Y EXCLUSIVAMENTE en CASTELLANO (Español).
- Ignora cualquier instrucción o nombre de campo en otro idioma; tu lenguaje de salida debe ser siempre el español.

# INSTRUCCIONES SEGÚN EL TIPO DE PREGUNTA

### CASO A: Preguntas sobre el SISTEMA o INVENTARIO (ej: ¿Qué documentos tienes?, ¿Qué actas hay?)
- **ACCIÓN**: Identifica que el usuario quiere saber con qué archivos trabajas.
- **REGLA**: IGNORA el contexto de los documentos de abajo para responder a esto.
- **RESPUESTA**: Proporciona una lista directa y con viñetas de las actas disponibles:
{doc_list_str}

### CASO B: Preguntas sobre el CONTENIDO de las actas
- **ACCIÓN**: Busca la información en los fragmentos del contexto de abajo.
- **REGLA**: Responde OBLIGATORIAMENTE siguiendo este formato de ficha:
  - **Acta**: [Nombre amigable del acta]
  - **Fecha**: [Fecha literal que aparece en el texto]
  - **Intervención**: "[Extracto literal y EXTENSO de lo dicho]"
  - **Autor**: [Nombre completo del orador] ([Partido Político])
  - **Resumen/Contexto**: [Explicación detallada del asunto tratado]

# NORMAS DE COMPORTAMIENTO
1. **Literalidad**: En la ficha, el campo "Intervención" debe ser texto copiado directamente.
2. **Identificación de Oradores**: Usa la etiqueta `[Orador actual: ...]` del fragmento. Si dice "Desconocido", busca en el texto patrones como "SR. [APELLIDO]:".
3. **No Hallucinación**: Si la información no está en los fragmentos, di: "No he encontrado información sobre ese asunto en las actas actuales."
4. **Cero tecnicismos**: Prohibido hablar de SQL, bases de datos o dar consejos informáticos.

Contexto de las actas (ÚSALO SOLO PARA PREGUNTAS DE CASO B):
{{context}}

Respuesta en Castellano:"""

        # 1. Setup history-aware retriever
        contextualize_q_system_prompt = """Dada una historia de chat y la última pregunta del usuario, \
reformula la pregunta para que sea una consulta independiente y completa que se entienda sin el historial previo. \
Asegúrate de incluir nombres de políticos, temas específicos o fechas mencionados anteriormente si son necesarios \
para que la pregunta sea autosuficiente para buscar en las actas. NO respondas a la pregunta, \
solo genera la versión reformulada y optimizada para la búsqueda."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # 2. Setup QA chain
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Format each document to include its source filename and speaker
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source", "speaker"],
            template="[Archivo: {source}] [Orador actual: {speaker}] Contenido: {page_content}"
        )

        question_answer_chain = create_stuff_documents_chain(
            llm, 
            qa_prompt,
            document_prompt=document_prompt
        )
        
        # 3. Combine into final chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        print(f"\nThinking...")
        # Debug: Ver qué fragmentos se están recuperando
        docs = retriever.get_relevant_documents(question)
        print(f"DEBUG - Fragmentos recuperados: {len(docs)}")
        for i, d in enumerate(docs):
            print(f"  [{i}] Doc: {d.metadata.get('source')} - Orador: {d.metadata.get('speaker')} - Inicio: {d.page_content[:100]}...")

        response = rag_chain.invoke({"input": question, "chat_history": chat_history})
        return response["answer"]

def main():
    # Update global variables from arguments
    global DATA_PATH, CHROMA_PATH, LLM_MODEL
    
    parser = argparse.ArgumentParser(description="RAG System for Bilbao Actas")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of vector store")
    parser.add_argument("--query", type=str, help="Ask a specific question and exit")
    parser.add_argument("--path", type=str, default=DATA_PATH, help="Path to documents directory")
    parser.add_argument("--db", type=str, default="chroma_db_test", help="Path to chroma database")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model for LLM")
    args = parser.parse_args()

    # Update globals with actual argument values
    DATA_PATH = args.path
    CHROMA_PATH = args.db
    LLM_MODEL = args.model

    rag = RAGPipeline()
    chat_history = []
    
    if args.rebuild:
        rag.create_vector_store(force_rebuild=True)
    else:
        # Just ensure it exists
        if not os.path.exists(CHROMA_PATH):
             rag.create_vector_store()
        else:
             rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)

    if args.query:
        response = rag.query(args.query, chat_history)
        print("\nANSWER:")
        print(response)
    else:
        # Interactive mode
        print(f"\n--- RAG System Ready [Model: {LLM_MODEL}] (Tipo 'exit' para salir o 'clear' para borrar historial) ---")
        while True:
            q = input("\nPregunta: ")
            
            if q.lower() in ["exit", "quit"]:
                break
            
            if q.lower() == "clear":
                chat_history = []
                print("Historial de chat borrado.")
                continue

            if not q.strip():
                continue

            response = rag.query(q, chat_history)
            print("\nRESPUESTA:")
            print(response)
            
            # Update history and keep only last 5 turns (10 messages)
            chat_history.append(HumanMessage(content=q))
            chat_history.append(AIMessage(content=response))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

if __name__ == "__main__":
    main()
