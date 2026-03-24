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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Puntero a la raíz del TFG
DATA_PATH = os.path.join(BASE_DIR, "actas", "pruebas actas_2025")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_v4")
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
                
                # 1. Join all pages into a single full text string
                full_text = "\n".join([page.page_content for page in pages])
                
                # 2. Split the text into segments based on proposal headers
                split_regex = re.compile(
                    r'(?=\n(?:[\s]*)(?:\d+)\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA))',
                    re.IGNORECASE 
                )
                segments = split_regex.split(full_text)
                
                current_speaker = "Desconocido"
                current_party = "Desconocido"
                current_topic = "General / Introducción"
                
                for segment in segments:
                    if not segment.strip():
                        continue
                        
                    # Extract the topic from the start of the segment (up to 300 characters to capture the full description)
                    topic_match = re.search(
                        r'^\s*(\d+\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA).{0,400})', 
                        segment, re.IGNORECASE | re.DOTALL
                    )
                    
                    if topic_match:
                        # Clean up newlines in the topic so it reads as a single sentence
                        candidate = topic_match.group(1).strip().replace('\n', ' ')
                        if len(candidate) > 10:
                            current_topic = candidate
                    
                    # 3. Chunk the text within this specific proposal
                    segment_chunks = text_splitter.split_text(segment)
                    
                    for chunk_text in segment_chunks:
                        # Check for speaker change in the chunk
                        match = speaker_regex.search(chunk_text)
                        if match:
                            name_candidate = match.group(1).strip()
                            # Reject if it's too long
                            if len(name_candidate) < 50:
                                current_speaker = name_candidate
                                current_party = "Desconocido"
                                for kn in sorted(party_map.keys(), key=len, reverse=True):
                                    if kn in current_speaker or current_speaker in kn:
                                        current_party = party_map[kn]
                                        break
                        
                        # Inyectar el metadata directamente en el texto para que Chroma lo vectorice
                        enriched_content = f"ASUNTO: {current_topic}\nORADOR: {current_speaker} ({current_party})\n\n{chunk_text}"
                        
                        # Create a chunk document with enriched metadata
                        all_chunks.append(Document(
                            page_content=enriched_content,
                            metadata={
                                "source": path,
                                "date": date,
                                "speaker": current_speaker,
                                "party": current_party,
                                "topic": current_topic
                            }
                        ))
                        
            except Exception as e:
                print(f"Error cargando {path}: {e}")
                
        print(f"Generados {len(all_chunks)} chunks con metadatos enriquecidos enfocados a propuestas.")
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
Tu ÚNICO objetivo es responder a las preguntas basándote ESTRICTAMENTE en la información del texto proporcionado bajo "Contexto recuperado".

**REGLAS CRÍTICAS (DE OBLIGADO CUMPLIMIENTO):**
1. CÍÑETE AL CONTEXTO: Si la respuesta no está explícitamente en el texto proporcionado abajo, di "No hay información suficiente en las actas proporcionadas para responder a esto." ¡NO INVENTES NADA!
2. PROHIBIDO USAR CONOCIMIENTO EXTERNO: No asumas de qué trata una propuesta si el texto no lo dice. Lee el texto con atención.
3. INCLUYE FECHAS Y ORADORES: Menciona en tu respuesta quién (Orador) dice qué y en qué fecha, guiándote por los metadatos de los fragmentos.
4. FORMATO: Si te preguntan por una propuesta o un debate, redacta un resumen claro indicando: 
   - De qué trata el tema.
   - Qué postura tiene cada orador mencionado en el texto.
   - Si el texto menciona el resultado de la votación, indícalo. Si no lo menciona, di que no aparece el resultado.

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

        mq_prompt = f"""Eres un asistente experto. Genera 3 versiones alternativas de la siguiente pregunta en CASTELLANO, enfocadas en palabras clave esenciales (nombres, partidos, temas) para un motor de búsqueda matemático.
Escribe SÓLO las preguntas alternativas, una por línea, sin listas estructuradas ni guiones.

Pregunta original: {question}"""

        print("\nMultiQuery: Expandiendo tu pregunta en búsquedas vectoriales simultáneas...")
        try:
            variations_text = llm.invoke(mq_prompt).content
            variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
        except Exception:
            variations = []
            
        if question not in variations:
            variations.append(question)
            
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
        all_initial_docs = []
        for v in variations:
            try:
                all_initial_docs.extend(retriever.invoke(v))
            except Exception:
                pass
            
        # --- FASE 2: EXPANSIÓN POR TEMA (Topic Expansion) ---
        # Identificar los temas más relevantes (top 2 temas para no saturar memoria)
        topic_counts = {}
        for doc in all_initial_docs:
            source = doc.metadata.get("source")
            topic = doc.metadata.get("topic")
            if source and topic and topic != "General / Introducción":
                key = (source, topic)
                topic_counts[key] = topic_counts.get(key, 0) + 1
        
        # Ordenar temas por frecuencia de Hits
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Filtrado inteligente: prioritizar temas que contengan palabras clave específicas
        # Excluimos palabras genéricas que están en casi todas las propuestas
        generic_words = {"partido", "popular", "propuesta", "proposicion", "mocio", "mocion", "bilbao", "votas", "votacion", "reunio", "reunion", "sobre", "dame", "informacion"}
        important_keywords = [w.lower() for w in question.split() if len(w) > 4 and w.lower() not in generic_words]
        
        prioritized_topics_scores = []
        for topic_key, count in sorted_topics:
            topic_text = topic_key[1].lower()
            # Contar CUÁNTAS palabras clave únicas coinciden
            matches = sum(1 for kw in important_keywords if kw in topic_text)
            prioritized_topics_scores.append((topic_key, matches, count))
            
        # Ordenar primero por número de matches únicos, luego por frecuencia de hits
        prioritized_topics_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # PRUNING: Si el mejor tema tiene matches y el segundo no tiene ninguno, 
        # o el primero tiene muchos más, nos quedamos SOLO con el mejor para evitar ruido.
        if len(prioritized_topics_scores) > 1:
            top_score = prioritized_topics_scores[0][1]
            next_score = prioritized_topics_scores[1][1]
            if top_score > 0 and next_score == 0:
                target_topics = [prioritized_topics_scores[0][0]]
            else:
                target_topics = [t[0] for t in prioritized_topics_scores[:2]]
        elif prioritized_topics_scores:
            target_topics = [prioritized_topics_scores[0][0]]
        else:
            target_topics = []
        
        docs = []
        if target_topics:
            # Detectar fecha de la pregunta para priorizar aún más (opcional, pero ayuda)
            print(f"Propuestas priorizadas (Pruned): {[t[1][:60] for t in target_topics]}")
            for (source, topic) in target_topics:
                # Recuperar fragmentos de ese tema
                topic_docs_data = self.vector_store.get(where={
                    "$and": [
                        {"source": source},
                        {"topic": topic}
                    ]
                })
                
                metadatas = topic_docs_data.get('metadatas', [])
                documents = topic_docs_data.get('documents', [])
                
                metadatas = topic_docs_data.get('metadatas', [])
                documents = topic_docs_data.get('documents', [])
                
                # Para asegurar que el LLM vea el debate completo (intro + votos), 
                # tomamos los primeros 15 y los últimos 15 chunks si el tema es largo.
                if len(documents) > 30:
                    indices = list(range(15)) + list(range(len(documents)-15, len(documents)))
                    for idx in indices:
                         docs.append(Document(page_content=documents[idx], metadata=metadatas[idx]))
                else:
                    for i in range(len(documents)):
                        docs.append(Document(page_content=documents[i], metadata=metadatas[i]))
        else:
            # Si no detectamos un tema claro, usamos los hits iniciales deduplicados
            docs = list({doc.page_content: doc for doc in all_initial_docs}.values())[:15]
        
        chain = prompt | llm | StrOutputParser()

        print(f"\nGenerando respuesta...")
        llm_response = chain.invoke({
            "context": format_docs(docs), 
            "question": question
        })
        
        # ==============================================================
        # Añadir las fuentes (con Orador, Acta y fragmento) directamente
        # a la respuesta final. (Fragmento más largo solicitado por el usuario)
        # ==============================================================
        sources_text = "\n\n" + "="*60 + "\nFUENTES Y EXTRACTOS UTILIZADOS PARA ESTA RESPUESTA:\n"
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "Desconocido"))
            topic = doc.metadata.get("topic", "General / Sin Asunto Específico")
            speaker = doc.metadata.get("speaker", "Desconocido")
            
            topic_short = topic if len(topic) < 70 else topic[:67] + "..."
            
            # Limpiar saltos de línea y mostrar un fragmento mucho más largo (400 caracteres)
            snippet = doc.page_content.replace('\n', ' ')
            snippet = snippet if len(snippet) < 400 else snippet[:397] + "..."
            
            sources_text += f" [{i+1}] {source} | Orador: {speaker} | Asunto: {topic_short}\n"
            sources_text += f"     -> Extracto: \"{snippet}\"\n\n"
        
        return llm_response + sources_text

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
        try:
            res = rag.query(args.query)
            # Intentar imprimir normalmente, pero caer a buffer si falla
            print(res)
        except UnicodeEncodeError:
            # Forzar salida en UTF-8 si el terminal de Windows falla
            sys.stdout.buffer.write(res.encode('utf-8', errors='replace'))
            print("\n")
    else:
        print(f"\n--- RAG Bilbao Ready [Model: {LLM_MODEL}] (Tipo 'exit' para salir) ---")
        while True:
            q = input("\nPregunta: ")
            if q.lower() in ["exit", "quit"]: break
            if not q.strip(): continue
            try:
                res = rag.query(q)
                print(res)
            except UnicodeEncodeError:
                sys.stdout.buffer.write(res.encode('utf-8', errors='replace'))
                print("\n")

if __name__ == "__main__":
    main()
