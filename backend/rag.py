import os
import argparse
import re
import glob
import sys
import time
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Autodetecta la ubicación de las actas: dentro del proyecto (portátil) o fuera (equipo potente).
_ACTAS_DENTRO = os.path.join(BASE_DIR, "actas")
_ACTAS_FUERA = os.path.join(os.path.dirname(BASE_DIR), "actas")
DATA_PATH = _ACTAS_DENTRO if os.path.isdir(_ACTAS_DENTRO) else _ACTAS_FUERA
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL_LOCAL = "gemma3:4b"
LLM_MODEL_GROQ = "llama-3.3-70b-versatile"
COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Umbral de distancia para descartar resultados semánticos irrelevantes.
SIMILARITY_DISTANCE_MAX = 1.4
# Prompt de MultiQuery: genera variantes de la pregunta para ampliar el recall.
MULTIQUERY_PROMPT = "Genera 3 variantes de: '{question}' centradas en el sujeto principal. Una por línea:"
# Relevancia mínima del reranker para considerar que hay algo que responder.
RELEVANCE_FLOOR = 0.01


def _date_sort_key(doc: Document) -> Tuple[int, int, int]:
    """Clave de orden cronológico (año, mes, día) a partir del metadato 'date' (DD-MM-YYYY)."""
    try:
        parts = doc.metadata.get('date', '01-01-1900').split('-')
        if len(parts) == 3:
            return (int(parts[2]), int(parts[1]), int(parts[0]))
    except (ValueError, AttributeError):
        pass
    return (1900, 1, 1)


class RAGPipeline:
    def __init__(self):
        """Inicializa el motor RAG con el modelo de embeddings configurado."""
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = None
        if GROQ_API_KEY:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(model=LLM_MODEL_GROQ, temperature=0, api_key=GROQ_API_KEY)
            print(f"[+] LLM: Groq ({LLM_MODEL_GROQ}) — respuestas rápidas via API")
        else:
            self.llm = ChatOllama(model=LLM_MODEL_LOCAL, temperature=0, num_ctx=8192)
            print(f"[+] LLM: Ollama local ({LLM_MODEL_LOCAL})")

    # --- Métodos de Utilidad ---

    def _extract_date_from_filename(self, path: str) -> str:
        """Extrae la fecha de un nombre de archivo (ej: '27-02-2025_...pdf')."""
        filename = os.path.basename(path)
        match = re.search(r'(\d{2}-\d{2}-\d{4})', filename)
        return match.group(1) if match else "Fecha desconocida"

    def _get_party_mapping(self, pages: List[Any]) -> Dict[str, str]:
        """Escanea las primeras páginas del acta para mapear nombres de concejales a partidos."""
        party_mapping = {}
        header_text = "\n".join([p.page_content for p in pages[:10]])
        current_party = "Goberno Local/Otros"
        
        for line in header_text.split('\n'):
            line = line.strip()
            if not line: continue
            
            re_esp = re.search(r"En representación del grupo municipal\s+([A-Z\s-]+)", line, re.IGNORECASE)
            re_eus = re.search(r"([A-Z\s-]+)\s+udal talde politikoaren izenean", line, re.IGNORECASE)
            
            if re_esp:
                current_party = re_esp.group(1).strip().strip(':')
                continue
            if re_eus:
                current_party = re_eus.group(1).strip().strip(':')
                continue
            
            re_member = re.search(r"^\d+\.-?\s*(?:DON|DOÑA|SR\.|SRA\.)?\s*([A-ZÁÉÍÓÚÑ]{4,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)", line, re.IGNORECASE)
            if re_member:
                name = re_member.group(1).strip()
                paren = re.search(r'\(([^)]+)\)', line)
                party_mapping[name] = paren.group(1).strip() if paren else current_party
                    
        return party_mapping

    # --- Fase de Ingesta (ETL) ---

    def _process_single_pdf(self, path: str) -> List[Document]:
        """Procesa un único PDF y devuelve sus chunks con metadatos (page + vote_result)."""
        speaker_regex = re.compile(r'(?:(?:EL|LA)\s+)?(?:SR\.|SRA\.)\s+([A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)\s*[:.]', re.IGNORECASE)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        # Ventanas amplias (0,500) entre etiquetas: en las actas BILINGÜES (euskera+
        # castellano) entre "Votos emitidos" y "Votos afirmativos" se intercala el
        # bloque en euskera ("Baiezko botoak: N jaun/andre: [nombres]...") + la lista
        # de concejales, que supera con creces los 80 caracteres del patrón antiguo.
        # Con 80 se perdían TODOS los votos de muchos plenos (p.ej. 27-10-2022: 0 de 26).
        vote_re = re.compile(
            r'Votos\s+emitidos[:\s]+(\d+)'
            r'.{0,500}?Votos\s+afirmativos[:\s]+(\d+)'
            r'(?:.{0,500}?Votos\s+negativos[:\s]+(\d+))?'
            r'(?:.{0,500}?Abstenciones?[:\s]+(\d+))?',
            re.IGNORECASE | re.DOTALL
        )
        # [^.] en vez de . para parar en el PUNTO que cierra la frase del resultado
        # (evita tragarse narración posterior como "- Siendo las 14:05 horas, el
        # señor Alcalde anuncia el receso..."). Margen amplio (0,400) para no cortar
        # a media palabra frases largas con varios grupos ("...el Grupo ELKARREKIN...").
        result_re = re.compile(
            r'(?:se\s+(?:acepta|aprueba|rechaza|desestima|deniega)\b[^.]{0,30}?'
            r'(?:enmienda|proposici[óo]n|propuesta|moci[óo]n|mozio|proposamen)[^.]{0,400}|'
            r'queda\s+(?:aprobad[ao]|rechazad[ao]|desestimad[ao])[^.]{0,400}|'
            r'resulta\s+(?:aprobad[ao]|rechazad[ao])[^.]{0,400})',
            re.IGNORECASE | re.DOTALL
        )
        # Resultados SIN cifras (acuerdos unánimes / por asentimiento). Muy frecuentes,
        # sobre todo en actas antiguas: "El Pleno Municipal, por unanimidad de miembros
        # presentes, acuerda...", "Aprobar por unanimidad...". Anclado en marcadores
        # formales del resultado para no confundirlo con discurso del debate.
        unanim_re = re.compile(
            r'(?:(?:el\s+pleno(?:\s+municipal)?|excmo\.?\s+ayuntamiento\s+pleno|aprobar)'
            r'[^.]{0,60}por\s+unanimidad[^.]{0,200}'
            r'|se\s+(?:aprueba|acuerda|aprueban|desestiman?|rechazan?)[^.]{0,60}'
            r'por\s+(?:unanimidad|asentimiento)[^.]{0,200})',
            re.IGNORECASE
        )
        chunks = []

        date = self._extract_date_from_filename(path)
        pdf_loader = PyPDFLoader(path)
        pages = pdf_loader.load()
        party_map = self._get_party_mapping(pages)

        # Mapa de offsets para rastrear nº de página por posición en el texto
        page_offsets = []  # (char_offset, page_number 1-indexed)
        full_text_parts = []
        cursor = 0
        for i, page in enumerate(pages):
            page_offsets.append((cursor, i + 1))
            full_text_parts.append(page.page_content)
            cursor += len(page.page_content) + 1
        full_text = "\n".join(full_text_parts)

        def page_for_offset(offset: int) -> int:
            pg = 1
            for off, num in page_offsets:
                if off <= offset:
                    pg = num
                else:
                    break
            return pg

        split_regex = re.compile(
            r'(?=\n(?:[\s]*)(?:\d+)\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA)'
            r'|\n\s*-\d+-\s*\n\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA|Proposición|Propuesta|Moción|Mozio|Dictamen|Irizpen|Asunto|Gaia|Proposamen))',
            re.IGNORECASE
        )
        segments = split_regex.split(full_text)

        # Tipos de punto del orden del día en el formato ANTIGUO (2002-2009), donde el
        # marcador es "- N - \n\n TIPO ...". Sirve para distinguir un punto real de un
        # simple número de PÁGINA (también escrito "- N -"), que NO debe partir el acta.
        _OLD_TYPES = (r'(?:Proposici[oó]n|Proposizioa|Propuesta|Preguntas?|'
                      r'Se\s+da\s+cuenta|Dar\s+cuenta|Dictamen|Moci[oó]n|'
                      r'Comparecencia|Interpelaci[oó]n|Declaraci[oó]n|Aprobar)')
        split_old_agenda = re.compile(r'(?=\n[\s]*-\s*\d+\s*-\s*\n[\s]*' + _OLD_TYPES + r')', re.IGNORECASE)
        old_topic_re = re.compile(r'^\s*-\s*(\d+)\s*-\s*\n[\s]*(' + _OLD_TYPES + r'.{0,140})', re.IGNORECASE | re.DOTALL)

        is_old_format = len(segments) <= 2
        if is_old_format:
            # 1º intentar partir por los puntos REALES del orden del día (ordinarias
            # antiguas) → topics limpios y un voto por punto. Si hay varios, usarlo.
            agenda_segs = [s for s in split_old_agenda.split(full_text) if s.strip()]
            if len(agenda_segs) > 2:
                segments = agenda_segs
            else:
                # Extraordinarias (p.ej. presupuestos): sin marcadores de orden del día;
                # se mantiene el troceo por página (comportamiento previo, sin regresión).
                split_regex_old = re.compile(r'(?=\n-\s*\d+\s*-\s*\n)')
                segments = split_regex_old.split(full_text)

        chunk_index_in_doc = 0
        seg_search_start = 0
        for segment in segments:
            if not segment.strip():
                continue
            current_speaker, current_party = "Desconocido", "Desconocido"

            # Posición del segmento en el texto completo (para calcular página)
            segment_offset = full_text.find(segment, seg_search_start)
            if segment_offset != -1:
                seg_search_start = segment_offset + max(1, len(segment) - 200)

            # Extraer vote_result a nivel de segmento limpio (sin solapamiento de chunks)
            seg_flat = re.sub(r'\s+', ' ', segment)
            resultado_text = None
            rm = result_re.search(seg_flat)
            if rm:
                resultado_text = re.split(
                    r'\s*-{3,}\s*|\s+-\s+|\s*https?://|\s+Egiaztatzeko|\s+Verificaci|\s+Siendo\s+las\b',
                    rm.group(0).strip()
                )[0].strip()
            resultado_num = None
            votes = list(vote_re.finditer(seg_flat))
            if votes:
                vm = votes[-1]
                emitidos, favor, contra, absten = vm.group(1), vm.group(2), vm.group(3), vm.group(4)
                # Basta con "a favor": las votaciones unánimes no traen "en contra"
                # y antes se descartaban (vote_result quedaba None y se perdía el voto).
                if favor:
                    partes = [f"a favor: {favor}"]
                    if contra:
                        partes.append(f"en contra: {contra}")
                    if absten:
                        partes.append(f"abstenciones: {absten}")
                    cab = f"Votos emitidos: {emitidos} | " if emitidos else ""
                    resultado_num = cab + ", ".join(partes)
            # Prioridad: cifras > texto unánime/asentimiento. Se evita guardar texto de
            # resultado "a secas" sin cifras (podría ser discurso); solo se acepta sin
            # cifras si es claramente un acuerdo unánime o por asentimiento.
            def _limpiar(txt):
                return re.split(
                    r'\s*-{3,}\s*|\s+-\s+|\s*https?://|\s+Egiaztatzeko|\s+Verificaci|\s+Siendo\s+las\b',
                    txt.strip()
                )[0].strip()

            if resultado_num:
                vote_result = f"{resultado_text} ({resultado_num})" if resultado_text else resultado_num
            elif resultado_text and re.search(r'unanimidad|asentimiento', resultado_text, re.I):
                vote_result = resultado_text
            else:
                um = unanim_re.search(seg_flat)
                vote_result = _limpiar(um.group(0)) if um else None

            if is_old_format:
                # 1º: encabezado REAL del punto ("- N - \n TIPO ...") → "N. TIPO ...".
                tm_old = old_topic_re.search(segment.lstrip('\n'))
                if tm_old:
                    texto = re.sub(r'\s+', ' ', tm_old.group(2)).strip()
                    current_topic = f"{tm_old.group(1)}. {texto[:115]}"
                else:
                    # 2º: fallback (extraordinarias troceadas por página): 1ª línea.
                    topic_raw = re.search(r'^-\s*\d+\s*-\s*\n\s*(.{0,200})', segment.lstrip('\n'), re.DOTALL)
                    if topic_raw:
                        first_line = topic_raw.group(1).strip().split('\n')[0].strip()
                        first_line = re.sub(r' {2,}', ' ', first_line)
                        current_topic = first_line[:120] if first_line else "General / Introducción"
                    else:
                        current_topic = "General / Introducción"
            else:
                topic_match_std = re.search(
                    r'^\s*(\d+\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA).{0,400})',
                    segment, re.IGNORECASE | re.DOTALL
                )
                topic_match_hist = re.search(
                    r'^\s*-\s*(\d+)\s*-\s*\n\s*((?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA|Proposición|Propuesta|Moción|Mozio|Dictamen|Irizpen|Asunto|Gaia|Proposamen).{0,400})',
                    segment, re.IGNORECASE | re.DOTALL
                )
                if topic_match_std:
                    current_topic = topic_match_std.group(1).strip().replace('\n', ' ')
                elif topic_match_hist:
                    clean_text = topic_match_hist.group(2).strip().replace('\n', ' ')
                    current_topic = f"{topic_match_hist.group(1)}. {clean_text}"
                else:
                    current_topic = "General / Introducción"

            chunk_search_start = 0
            for chunk_text in text_splitter.split_text(segment):
                match = speaker_regex.search(chunk_text)
                if match:
                    current_speaker = match.group(1).strip()
                    if len(current_speaker) < 50:
                        for kn, kp in party_map.items():
                            if kn in current_speaker or current_speaker in kn:
                                current_party = kp
                                break

                # Página del chunk: buscar su posición en el segmento
                if segment_offset != -1:
                    chunk_pos = segment.find(chunk_text[:60], chunk_search_start)
                    if chunk_pos != -1:
                        page_num = page_for_offset(segment_offset + chunk_pos)
                        chunk_search_start = chunk_pos
                    else:
                        page_num = page_for_offset(segment_offset)
                else:
                    page_num = 1

                metadata = {
                    "source": path, "date": date, "speaker": current_speaker,
                    "party": current_party, "topic": current_topic,
                    "chunk_index": chunk_index_in_doc, "page": page_num,
                }
                if vote_result:
                    metadata["vote_result"] = vote_result

                chunks.append(Document(
                    page_content=f"ASUNTO: {current_topic}\nORADOR: {current_speaker} ({current_party})\n\n{chunk_text}",
                    metadata=metadata
                ))
                chunk_index_in_doc += 1
        return chunks

    def load_and_split_documents(self) -> List[Document]:
        """Carga los PDFs y los divide en fragmentos con metadatos de forma recursiva."""
        pdf_files = glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True)

        if not pdf_files:
            print(f"[!] No se encontraron archivos PDF en {DATA_PATH}")
            return []

        print(f"[*] Encontrados {len(pdf_files)} archivos PDF. Procesando...")

        all_chunks = []
        for path in tqdm(pdf_files, desc="Procesando Actas", unit="pdf"):
            try:
                all_chunks.extend(self._process_single_pdf(path))
            except Exception as e:
                print(f"[!] Error cargando {path}: {e}")

        return all_chunks

    def create_vector_store(self):
        """Crea o carga la base de datos vectorial ChromaDB."""
        if os.path.exists(CHROMA_PATH):
            print(f"[*] Cargando base de datos vectorial desde {CHROMA_PATH}...")
            self.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        else:
            chunks = self.load_and_split_documents()
            if not chunks: return
            print("[*] Generando embeddings...")
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=CHROMA_PATH)
            print(f"[+] Base de datos guardada en {CHROMA_PATH}")

    # --- Fase de Recuperación (Retrieval) ---

    def _get_temporal_filter(self, question: str) -> Optional[Dict[str, Any]]:
        """Analiza la pregunta para aplicar filtros de fecha inteligentes."""
        # Recopilar fechas conocidas del sistema de archivos
        known_dates = []
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files:
                if file.endswith(".pdf"):
                    d = self._extract_date_from_filename(file)
                    if d: known_dates.append(d)
        known_dates = list(set(known_dates))

        # Prioridad 1: fecha exacta DD-MM-YYYY en la pregunta
        exact_match = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', question)
        if exact_match:
            exact_date = exact_match.group(0)
            if exact_date in known_dates:
                print(f"[*] Filtro fecha exacta: {exact_date}")
                return {"date": {"$eq": exact_date}}, [exact_date]
            # Fallback a mes-año si la fecha exacta no existe
            target_pattern = f"{exact_match.group(2)}-{exact_match.group(3)}"
            valid_dates = [d for d in known_dates if target_pattern in d]
            if valid_dates:
                print(f"[*] Filtro Temporal (mes-año): {target_pattern} ({len(valid_dates)} actas)")
                return {"date": {"$in": valid_dates}}, valid_dates
            return None, []

        # Prioridad 2: año + nombre de mes en español
        months = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        year_match = re.search(r'(20\d{2})', question)
        month_found = next((m for m in months if m in question.lower()), None)

        if not year_match: return None, []

        target_year = year_match.group(1)
        target_month = f"{str(months.index(month_found)+1).zfill(2)}" if month_found else None
        target_pattern = f"{target_month}-{target_year}" if target_month else target_year

        valid_dates = [d for d in known_dates if target_pattern in d]
        if valid_dates:
            print(f"[*] Filtro Temporal: {target_pattern} ({len(valid_dates)} actas detectadas)")
            return {"date": {"$in": valid_dates}}, valid_dates
        return None, []

    def _detect_party_in_question(self, question: str) -> Optional[str]:
        """Detecta si la pregunta menciona un grupo político concreto.

        Devuelve el substring normalizado para filtrar metadata['party'] (fuzzy substring),
        o None si la pregunta es genérica (varios grupos o sin mención).
        """
        q = question.lower()
        # Más específico primero para evitar falsos positivos
        if re.search(r'eh\s*bildu|euskal\s+herria\s+bildu', q):
            return "EH BILDU"
        if re.search(r'elkarrekin', q):
            return "ELKARREKIN"
        if re.search(r'pse[\s\-]ee|\bsocialistas?\s+vascos?\b|\bpartido\s+socialista\b|\bpse\b', q):
            return "PSE"
        if re.search(r'eaj[\s\-]pnv|\bpnv\b|\bnacionalistas?\s+vascos?\b', q):
            return "PNV"
        if re.search(r'\bpartido\s+popular\b|\bgrupo\s+(?:municipal\s+)?pp\b|\bel\s+pp\b|\bdel\s+pp\b', q):
            return "POPULAR"
        if re.search(r'udalberri', q):
            return "UDALBERRI"
        if re.search(r'ciudadanos', q):
            return "CIUDADANOS"
        if re.search(r'ezker\s+batua|izquierda\s+unida', q):
            return "EZKER"
        return None

    def _expand_context_by_topic(self, initial_docs: List[Document], question: str) -> List[Document]:
        """Técnica de la 'Aspiradora': Expande los fragmentos semánticos a debates completos.
        
        Usa dos pasadas:
        1. Puntuar los temas que ya llegaron de la búsqueda semántica.
        2. Búsqueda directa en los títulos de tema de la BD para garantizar cobertura total.
        """
        target_topics, seen = [], set()

        # Solo palabras ≥6 chars que llegan al check (las <6 chars y los años numéricos
        # ya se descartan antes por las reglas \w{6,} y not w.isdigit()).
        stopwords = {
            # Omnipresentes en todos los títulos de acta — no aportan señal temática
            "bilbao", "municipal", "municipales", "partido", "popular", "grupos",
            "acuerdo", "acuerdos", "proposicion", "proposizioa", "propuesta", "propuestas",
            "mocion", "mozioa", "debate", "debates", "sesion", "reunion",
            "resultado", "votacion", "propone", "propuso", "propuesto", "presenta", "plantea",
            "siguiente", "adoptado", "tomado", "tomados", "relacionado", "relacionados",
            "dispositiva", "literal", "adopcion", "decidio", "habido", "habida",
            "cuales", "exacto",
            # Meses ≥6 chars (los ≤5 chars ya los filtra la regla \w{6,})
            "febrero", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        }
        # Normalizamos acentos y caracteres especiales para asegurar coincidencia robusta
        def normalize(txt: str) -> str:
            replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
            txt = txt.lower()
            for k, v in replacements.items():
                txt = txt.replace(k, v)
            txt = re.sub(r'[^a-z0-9\s]', '', txt)
            return txt

        q_clean = normalize(question)
        # Mínimo 6 caracteres: reduce falsos positivos por stems cortos
        # (ej: "presu" de "presupuesto" matcheando "presencia" con stem[:5])
        q_keywords = [w for w in re.findall(r'\w{6,}', q_clean) if w not in stopwords and not w.isdigit()]

        # --- PASADA 1: Puntuar temas de la búsqueda semántica ---
        for doc in initial_docs:
            topic = doc.metadata.get("topic")
            source = doc.metadata.get("source")
            if topic and topic not in ["General", "General / Introducción"]:
                source_bn = os.path.basename(source)
                if (source_bn, topic) not in seen:
                    norm_topic = normalize(topic)
                    norm_content = normalize(doc.page_content)
                    # Stem de 6 chars: más específico que 5, menos falsos positivos.
                    # Puntuación: 20 pts título (señal fuerte) / 2 pts contenido (señal débil)
                    score = sum(
                        20 if kw[:6] in norm_topic else 2 if kw[:6] in norm_content else 0
                        for kw in q_keywords
                    )
                    seen.add((source_bn, topic))
                    target_topics.append({"topic": topic, "source": source, "score": score, "basename": source_bn})



        # PENALIZACIÓN DE TEMAS PRESUPUESTARIOS: Los debates de presupuestos son muy largos
        # y mencionan de pasada cualquier tema (medio ambiente, pobreza, movilidad…).
        # Si la pregunta no menciona "presupuesto" ni "modificación", penalizamos esos temas
        # para que no contaminen búsquedas temáticas específicas.
        q_mentions_budget = bool(re.search(r'presupuest|modificaci[oó]n\s+presup|ordenanza', q_clean, re.I))
        if not q_mentions_budget:
            for t in target_topics:
                t_norm = normalize(t['topic'])
                if re.search(r'presupuest|modificac|ordenanza', t_norm):
                    t['score'] *= 0.2

        # FILTRO DE TÍTULO: Si algún tema tiene una keyword en su título (score ≥ 20),
        # descartar los que solo puntúan por menciones de pasada en el contenido.
        # Evita que debates de presupuestos, aparcamientos, etc. entren porque mencionan
        # "pobreza" o "movilidad" una vez en el debate sin que el punto trate ese tema.
        # Si ningún tema supera 20 puntos (todos son coincidencias de contenido), se mantiene
        # el comportamiento actual para no romper consultas sin keywords en los títulos.
        title_matched = [t for t in target_topics if t['score'] >= 20]
        if title_matched:
            target_topics = title_matched

        # DOBLE UMBRAL para eliminar ruido:
        # 1) Umbral absoluto: evita que temas con score bajo pasen cuando todos los scores son bajos.
        #    Con stem 6 chars + peso 20/2: score=4 → al menos 2 contenidos relevantes.
        #    score=20 → una keyword clave en el título (señal muy fuerte).
        # 2) Umbral relativo: descarta temas con menos del 40% del mejor score.
        MIN_ABSOLUTE_SCORE = 4
        target_topics = sorted(
            [t for t in target_topics if t['score'] >= MIN_ABSOLUTE_SCORE],
            key=lambda x: x['score'], reverse=True
        )

        if target_topics:
            max_s = target_topics[0]['score']
            if max_s > 0:
                # Umbral relativo: descartar temas por debajo del 40% del score máximo
                global_threshold = max_s * 0.4
                target_topics = [t for t in target_topics if t['score'] >= global_threshold]

            # Estrategia de diversidad temporal: preferimos 1 tema por año distinto
            # para que preguntas "a lo largo de los años" cubran múltiples sesiones
            # Solo los temas que ya superaron el umbral global pueden ser "representantes de año"
            seen_years = {}
            diverse_topics = []
            for t in target_topics:
                # Extraer año del nombre del fichero fuente (ej: "28-09-2017_Ordinaria...")
                year = ""
                src_bn = t.get("basename", "")
                year_match = re.search(r'(\d{4})', src_bn)
                if year_match:
                    year = year_match.group(1)
                
                if year not in seen_years:
                    seen_years[year] = t['score']
                    diverse_topics.append(t)
                elif t['score'] >= seen_years[year] * 0.9:
                    # Segundo tema del mismo año solo si es casi tan bueno
                    diverse_topics.append(t)
            
            # Para preguntas multi-año, subimos el límite a 8 temas
            # El filtro de 12.000 chars impide que el contexto se desborde
            target_topics = diverse_topics[:8]

        docs, final_seen = [], set()
        if not target_topics: return initial_docs[:15], []

        print(f"[*] Temas priorizados: {[t['topic'][:50] for t in target_topics]}")
        for t_info in target_topics:
            t_name, src = t_info['topic'], t_info['source']
            t_prefix = f"ASUNTO: {t_name[:40]}"
            try:
                res = self.vector_store.get(where={"source": src})
                pairs = sorted(zip(res['metadatas'], res['documents']), key=lambda x: x[0].get('chunk_index', 0))
                
                # ESTRATEGIA DE BYPASS DE ÍNDICE: Buscamos el tema pero saltamos los primeros fragmentos
                # El índice suele estar en los primeros 100 fragmentos. El debate real mucho después.
                topic_indices = [i for i, (m, d) in enumerate(pairs) 
                                 if (m.get('topic') == t_name or d.lstrip().startswith(t_prefix))]
                
                if topic_indices:
                    # El tema suele aparecer DOS veces en el acta:
                    #   1) en el ORDEN DEL DÍA al principio -> una racha de fragmentos cortos (solo títulos)
                    #   2) en el DEBATE real mucho después -> una racha de fragmentos largos (cuerpo + votación)
                    # Agrupamos los índices en rachas contiguas y nos quedamos con la que
                    # tiene MÁS texto: ese es el debate real, no la entrada del índice.
                    runs = []
                    current_run = [topic_indices[0]]
                    for idx in topic_indices[1:]:
                        if idx == current_run[-1] + 1:
                            current_run.append(idx)
                        else:
                            runs.append(current_run)
                            current_run = [idx]
                    runs.append(current_run)

                    best_run = max(runs, key=lambda run: sum(len(pairs[i][1]) for i in run))
                    selected = [pairs[i] for i in best_run]

                    # Filtro de seguridad: descartar fragmentos de otros temas
                    _num = t_name.split('.')[0].strip()
                    t_topic_number = (_num + ".") if _num.isdigit() else t_name[:80]
                    seen_content = set()
                    for m, d in selected:
                        topic_val = m.get('topic', '')
                        if topic_val and not topic_val.startswith(t_topic_number):
                            continue
                            
                        # Usamos los ÚLTIMOS 150 caracteres para la deduplicación
                        # porque los primeros siempre son "ASUNTO: [título largo]" y colisionan
                        content_key = d[-150:].strip() if len(d) > 150 else d.strip()
                        if content_key not in seen_content:
                            seen_content.add(content_key)
                            key = (src, m.get('chunk_index', d[:50]))
                            if key not in final_seen:
                                final_seen.add(key)
                                docs.append(Document(page_content=d, metadata=m))
            except Exception as e:
                print(f"[!] Error en expansión: {e}")
        
        return docs, target_topics



    def _rerank_with_cohere(self, docs: List[Document], question: str, top_n: int = 30) -> List[Document]:
        """Reordena los documentos candidatos usando Cohere rerank-multilingual-v3.0.

        Si la API key no está configurada o hay un error, devuelve los docs originales
        sin modificar (degradación elegante).
        """
        if not COHERE_API_KEY or not docs:
            return docs
        try:
            import cohere
            co = cohere.Client(COHERE_API_KEY)

            # Deduplicar por contenido antes de enviar a la API
            seen_content: set = set()
            unique_docs: List[Document] = []
            for d in docs:
                key = d.page_content[:200]
                if key not in seen_content:
                    seen_content.add(key)
                    unique_docs.append(d)

            # Limitar a 60 candidatos para no desperdiciar cuota de la API
            candidates = unique_docs[:60]

            results = co.rerank(
                model=COHERE_RERANK_MODEL,
                query=question,
                documents=[d.page_content for d in candidates],
                top_n=min(top_n, len(candidates))
            )
            reranked = []
            for r in results.results:
                doc = candidates[r.index]
                # Guardar el score de relevancia (transitorio) para poder aplicar
                # un umbral mínimo y responder "no encontrado" a preguntas sin relación.
                doc.metadata["_rerank_score"] = float(r.relevance_score)
                reranked.append(doc)
            print(f"[*] Cohere reranking: {len(candidates)} candidatos → top {len(reranked)}")
            return reranked
        except Exception as e:
            print(f"[!] Cohere reranking fallido, usando ChromaDB directo: {e}")
            return docs

    def _format_context(self, docs: List[Document]) -> str:
        """Formatea los documentos agrupando por (fecha, tema) para compactar el contexto.
        
        En lugar de un bloque por chunk, genera un bloque por debate completo.
        Esto reduce el tamaño del contexto cuando hay múltiples chunks del mismo debate.
        """
        from collections import OrderedDict
        
        # Agrupar por (fecha, topic) para unir el inicio y el final del debate
        groups = OrderedDict()
        for d in docs:
            date = d.metadata.get("date", "Fecha desconocida")
            topic = d.metadata.get("topic", "General")
            source = os.path.basename(d.metadata.get("source", "Acta"))
            key = (date, topic, source)
            
            content = d.page_content.strip()
            content = content.replace("Udalbatzako Idazkaritza Nagusia", "")
            content = content.replace("Udalbatzarreko Idazkaritza Nagusia", "")
            content = content.replace("Secretar\u00eda General del Pleno", "")
            
            # Quitar las etiquetas ASUNTO:/ORADOR: SIN borrar el cuerpo de la propuesta.
            # OJO: en las actas modernas el chunk entero va en UNA sola línea, así que
            # el antiguo split('\n') + quitar la primera línea borraba toda la propuesta.
            lines = content.split('\n')
            if len(lines) >= 3:
                # Formato antiguo: cabeceras ASUNTO:/ORADOR: en líneas propias.
                if lines[0].startswith('ASUNTO:'):
                    lines = lines[1:]
                if lines and lines[0].startswith('ORADOR:'):
                    lines = lines[1:]
                content = '\n'.join(lines).strip()
            else:
                # Formato moderno (todo en una línea): quitamos solo las ETIQUETAS
                # con regex y conservamos el cuerpo de la propuesta.
                content = re.sub(r'^ASUNTO:\s*', '', content)
                content = re.sub(r'\bORADOR:\s*[^(]*\([^)]*\)\s*', ' ', content)
                content = re.sub(r'\s{2,}', ' ', content).strip()
            
            if key not in groups:
                groups[key] = {"contents": [], "vote_result": None}
            groups[key]["contents"].append(content)
            if not groups[key]["vote_result"] and d.metadata.get("vote_result"):
                groups[key]["vote_result"] = d.metadata["vote_result"]

        final_text = ""
        for (date, topic, source), group_data in groups.items():
            contents = group_data["contents"]
            resultado = group_data["vote_result"]  # extraído del segmento limpio, sin solapamiento

            raw = "\n---\n".join(c for c in contents if c.strip())
            raw_flat = re.sub(r'\s+', ' ', raw)
            
            # Construir el cuerpo: cabecera (texto dispositivo de la propuesta) + primer
            # tramo del DEBATE real. Las actas repiten muchas veces el texto dispositivo
            # antes de las intervenciones, así que saltamos a la primera intervención de
            # un concejal (SR./SRA. NOMBRE) para que los argumentos entren en el contexto
            # sin disparar el nº de tokens. Si no hay intervención localizable, recortamos normal.
            head = raw_flat[:1600]
            m_int = re.search(r'SR[A]?\.\s+[A-ZÁÉÍÓÚÑ]{2,}', raw_flat)
            if m_int and m_int.start() > 1600:
                raw = head + " [...debate...] " + raw_flat[m_int.start():m_int.start() + 2200]
            elif len(raw) > 3800:
                raw = raw[:3800] + "..."

            short_topic = topic[:120]
            block = f"[PLENO: {date}] {short_topic}\n{raw}\n"
            if resultado:
                block += f"RESULTADO: {resultado}\n"
            block += "\n"
            final_text += block
            
        return final_text.strip()



    # --- Pipeline de recuperación compartido (usado por retrieve_context y query) ---

    def _query_variations(self, question: str) -> List[str]:
        """MultiQuery: genera variantes de búsqueda para ampliar el recall (más la original)."""
        try:
            vars_txt = self.llm.invoke(MULTIQUERY_PROMPT.format(question=question)).content
            variations = [v.strip() for v in vars_txt.split('\n') if v.strip()] + [question]
        except Exception:
            variations = [question]
        return variations[:6]

    def _initial_search(self, variations: List[str], valid_dates: list,
                        exact_date: bool, k: int) -> List[Document]:
        """Búsqueda semántica inicial sobre todas las variantes (con o sin filtro de fecha).

        Con fecha exacta se toman todos los chunks de ese día; en el resto se descartan
        los que superan la distancia máxima de similitud.
        """
        docs: List[Document] = []
        for v in variations:
            targets = valid_dates if valid_dates else [None]
            for d_val in targets:
                flt = {"date": {"$eq": d_val}} if d_val else None
                try:
                    if exact_date:
                        # Fecha única: tomar todo el pleno sin filtrar por score
                        docs.extend(self.vector_store.similarity_search(v, k=k, filter=flt))
                    else:
                        res = self.vector_store.similarity_search_with_score(v, k=k, filter=flt)
                        docs.extend([d for d, s in res if s < SIMILARITY_DISTANCE_MAX])
                except Exception:
                    pass
        return docs

    def _apply_party_filter(self, docs: List[Document], question: str,
                            min_docs: int = 3) -> List[Document]:
        """Si la pregunta menciona un grupo concreto, conserva solo los chunks donde ese
        grupo es el ORADOR (no donde otros lo mencionan).

        min_docs: mínimo de docs filtrados para aplicar el filtro. En la búsqueda inicial
        (semántica) se usa 3 para evitar sobreajuste con ruido; después de la aspiradora
        se llama con 1 porque los docs ya son del tema correcto.
        """
        target_party = self._detect_party_in_question(question)
        if not target_party:
            return docs
        party_filtered = [
            d for d in docs
            if target_party.lower() in d.metadata.get("party", "").lower()
            or target_party.lower() in d.page_content.split('\n\n')[0].lower()
        ]
        if len(party_filtered) >= min_docs:
            print(f"[*] Filtro de grupo '{target_party}': {len(party_filtered)} docs")
            return party_filtered
        return docs

    def _dedup_docs(self, docs: List[Document], limit: Optional[int] = None) -> List[Document]:
        """Elimina duplicados por (source, chunk_index) conservando el orden de entrada."""
        seen, out = set(), []
        for d in docs:
            key = (d.metadata.get('source', ''), d.metadata.get('chunk_index', d.page_content[:50]))
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out[:limit] if limit else out

    def _filter_by_target_topics(self, docs: List[Document], target_topics: list) -> List[Document]:
        """Elimina docs cuyo topic no esté entre los temas priorizados (si el filtro no vacía todo)."""
        if not target_topics:
            return docs
        valid_prefixes = []
        for t in target_topics:
            tp = t['topic']
            num = tp.split('.')[0].strip()
            valid_prefixes.append((num + ".") if num.isdigit() else tp[:80])
        filtered = [d for d in docs if any(d.metadata.get('topic', '').startswith(p) for p in valid_prefixes)]
        return filtered if filtered else docs

    def _retrieve_and_rank(self, question: str, k: int) -> Tuple[List[Document], bool]:
        """MultiQuery → búsqueda inicial → filtro de grupo → reranking de Cohere.

        Devuelve (docs candidatos, exact_date). `exact_date` es True cuando la pregunta
        apunta a un único pleno concreto (cambia la estrategia de expansión posterior).
        """
        variations = self._query_variations(question)
        _, valid_dates = self._get_temporal_filter(question)
        exact_date = len(valid_dates) == 1
        docs = self._initial_search(variations, valid_dates, exact_date, k)
        docs = self._apply_party_filter(docs, question)
        if COHERE_API_KEY:
            docs = self._rerank_with_cohere(docs, question, top_n=30)
        return docs, exact_date

    def _select_final_docs(self, candidates: List[Document], question: str,
                           exact_date: bool) -> List[Document]:
        """A partir de los candidatos rankeados, decide el conjunto final de docs.

        Fecha exacta → chunks más relevantes sin expandir (la expansión traería ruido de
        otros temas del mismo pleno). En el resto → expansión "aspiradora" por tema.
        Siempre se devuelve ordenado cronológicamente.
        """
        if exact_date:
            docs = self._dedup_docs(candidates, limit=40)
        else:
            docs, target_topics = self._expand_context_by_topic(candidates, question)
            docs = self._filter_by_target_topics(docs, target_topics)
            # Re-aplicar filtro de partido sobre el corpus completo recuperado por la aspiradora.
            # Umbral min_docs=1: los docs ya son del tema correcto, el filtro puede ser agresivo.
            docs = self._apply_party_filter(docs, question, min_docs=1)
        return sorted(docs, key=_date_sort_key)

    def _dump_debug_context(self, formatted_context: str) -> None:
        """Vuelca el contexto enviado al LLM para inspección técnica."""
        with open("debug_context.txt", "w", encoding="utf-8") as f:
            f.write(formatted_context)

    # --- Punto de Entrada Principal ---

    def retrieve_context(self, question: str) -> dict:
        """Fase de recuperación: devuelve el contexto y el prompt listos para el LLM.

        Separa la recuperación (lenta por embeddings) de la generación (streaming).
        Llamar esto en un thread y luego hacer streaming del LLM en el hilo async.
        """
        if not self.vector_store: self.create_vector_store()

        all_initial_docs, exact_date = self._retrieve_and_rank(question, k=80)

        # Umbral de relevancia: si la pregunta NO menciona una fecha concreta y ni el mejor
        # fragmento alcanza una relevancia mínima, no hay nada que responder. Evita inventar
        # fuentes con preguntas ajenas a las actas (p.ej. "viajes a Marte": score top ~0.002,
        # frente a ~0.89 de una pregunta legítima).
        # Solo aplicar el umbral si Cohere realmente asignó scores (si falló,
        # los docs no tienen '_rerank_score' y no debemos bloquear por eso).
        if COHERE_API_KEY and not exact_date and all_initial_docs:
            rerank_scores = [d.metadata["_rerank_score"] for d in all_initial_docs if "_rerank_score" in d.metadata]
            max_score = max(rerank_scores) if rerank_scores else None
            if max_score is not None and max_score < RELEVANCE_FLOOR:
                print(f"[*] Relevancia máxima {max_score:.4f} < {RELEVANCE_FLOOR}: sin resultados.")
                return {
                    "context": "", "is_multi_session": False,
                    "unique_dates": [], "docs": [], "question": question,
                }

        docs = self._select_final_docs(all_initial_docs, question, exact_date)
        self.last_retrieved_docs = docs

        formatted_context = self._format_context(docs)
        # Límite alto: con Groq el contexto ya no es cuello de botella y así las actas
        # recientes (ordenadas al final) no se cortan. Cada debate ya está acotado a 800 chars.
        if len(formatted_context) > 34000:
            formatted_context = formatted_context[:34000] + "\n\n[...CONTEXTO TRUNCADO POR TAMAÑO...]"

        self._dump_debug_context(formatted_context)

        unique_dates = list(set(d.metadata.get("date", "") for d in docs if d.metadata.get("date")))
        is_multi_session = len(unique_dates) > 1

        return {
            "context": formatted_context,
            "is_multi_session": is_multi_session,
            "unique_dates": unique_dates,
            "docs": docs,
            "question": question,
        }

    def search_related(self, question: str, exclude_keys: set, k: int = 80) -> List[Dict[str, str]]:
        """Búsqueda amplia para descubrir propuestas relacionadas no incluidas en los resultados principales."""
        if not self.vector_store:
            return []
        docs = self.vector_store.similarity_search(question, k=k)
        seen: set = set()
        related = []
        _SKIP = {"General / Introducción", "General", "GENERAL / INTRODUCCIÓN", "GENERAL"}
        for d in docs:
            date = d.metadata.get("date", "")
            topic = d.metadata.get("topic", "")
            if not topic or topic.strip() in _SKIP:
                continue
            t = topic.strip()
            is_numbered = bool(re.match(r'^\d+', t))
            has_keyword = bool(re.search(
                r'[Pp]roposici[oó]n|[Pp]roposamen|PROPOSAMENA|MOZIO|MOC?I[OÓ]N|DICTAMEN|IRIZPENA',
                t
            ))
            # Cabecera formal sin número: >55% de letras en mayúscula con longitud suficiente.
            # Cubre partidos, organizaciones y cualquier cabecera de formato antiguo sin
            # necesidad de hardcodear ningún nombre concreto.
            letters = [c for c in t if c.isalpha()]
            is_formal_header = (
                len(letters) >= 10
                and sum(1 for c in letters if c.isupper()) / len(letters) > 0.55
            )
            if not (is_numbered or has_keyword or is_formal_header):
                continue
            key = (date, t[:60])
            if key in exclude_keys or key in seen:
                continue
            seen.add(key)
            body = d.page_content.split('\n\n', 1)[-1] if '\n\n' in d.page_content else d.page_content
            snippet = re.sub(r'\s+', ' ', body).strip()[:130]
            related.append({"date": date, "topic": topic, "snippet": snippet})
        return related

    def query(self, question: str) -> str:
        """Flujo principal de la CLI: Recuperación, Expansión y Generación con fuentes."""
        if not self.vector_store: self.create_vector_store()

        # 1-2. Recuperación compartida (MultiQuery, búsqueda, filtro de grupo, rerank)
        all_initial_docs, exact_date = self._retrieve_and_rank(question, k=50)

        # 3. Expansión/selección y ordenación cronológica (más antiguo primero)
        docs = self._select_final_docs(all_initial_docs, question, exact_date)
        self.last_retrieved_docs = docs

        formatted_context = self._format_context(docs)

        # FILTRO DE SEGURIDAD CONTRA CUELGUES (MÁXIMO 12000 CARACTERES)
        # Esto asegura que el prompt sea de aprox 3000 tokens como máximo,
        # lo que previene que el modelo local se quede pillado.
        if len(formatted_context) > 12000:
            formatted_context = formatted_context[:12000] + "\n\n[...CONTEXTO TRUNCADO POR TAMAÑO...]"
        
        # Detectar si la pregunta es general (varios años/sesiones) o específica (un pleno concreto)
        unique_dates = list(set(d.metadata.get("date", "") for d in docs if d.metadata.get("date")))
        is_multi_session = len(unique_dates) > 1

        self._dump_debug_context(formatted_context)

        # Guarda contra alucinaciones: si el contexto está vacío no se invoca el LLM
        if not formatted_context.strip():
            return ("Lo siento, no he encontrado fragmentos relevantes en las actas para responder a tu pregunta. "
                    "Puede que el tema no esté cubierto en los documentos indexados, o que el modelo de embeddings "
                    "no haya podido conectarse. Prueba a reformular la pregunta.")

        if is_multi_session:
            dates_found = ', '.join(sorted(unique_dates))
            sys_prompt = f"""Eres el Cronista Oficial de Bilbao, experto en historia municipal. RESPONDE SIEMPRE EN ESPAÑOL.

        INSTRUCCION: Se te proporcionan fragmentos de MULTIPLES plenos del Ayuntamiento de Bilbao.
        Las fechas de los plenos en este contexto son: {dates_found}
        Responde a la pregunta haciendo un RESUMEN CRONOLOGICO de los debates y propuestas encontrados.

        REGLAS CRUCIALES - DEBES SEGUIRLAS TODAS:
        - USA SOLO la informacion que esta explicitamente en las actas proporcionadas abajo.
        - NUNCA inventes fechas, cifras, nombres, resultados o detalles que no esten en el texto.
        - DEBES cubrir TODAS las fechas listadas arriba que tengan informacion relevante.
        - Si no sabes el resultado de una votacion porque no esta en el acta, escribe exactamente: [Sin resultado en acta]
        - Para cada pleno relevante que encuentres, usa este formato:
          FECHA: [fecha del pleno segun el acta]
          Grupo: [quien presento la propuesta]
          Propuesta: [que pedia exactamente]
          Resultado: [resultado de la votacion si consta, si no: [Sin resultado en acta]]
        - Ordena de mas antiguo a mas reciente.
        - Si un acta no contiene informacion relacionada con la pregunta, ignorala.

        ACTAS DE BILBAO PARA ANALIZAR:
        {{context}}

        PREGUNTA: {{question}}
        RESUMEN CRONOLOGICO:"""
        else:
            sys_prompt = """Eres el Cronista Oficial de Bilbao. Tu misión es relatar lo ocurrido en el Pleno. RESPONDE SIEMPRE EN ESPAÑOL.

        INSTRUCCIÓN: Basándote en el ACTA de abajo, responde a: {question}
        
        REGLAS CRUCIALES:
        - Empieza directamente con: "En la sesión del Pleno de Bilbao..."
        - No digas "el texto menciona" ni "según el acta". Habla como si estuvieras allí.
        - Detalla los puntos de la propuesta (qué se pide exactamente).
        - Indica el resultado final de la votación (votos a favor y en contra).

        ACTA DE BILBAO PARA ANALIZAR:
        {context}

        PREGUNTA: {question}
        CRÓNICA:"""
        
        chain = ChatPromptTemplate.from_template(sys_prompt) | self.llm | StrOutputParser()
        print(f"[*] Generando crónica detallada (Fuerza Bruta de Contexto)...")

        # Reintento automático: en máquinas con poca RAM ollama puede tardar en recargar
        # el modelo LLM después de las llamadas de embedding, rechazando la conexión
        # durante ese breve intervalo. Reintentos con backoff corto solucionan el problema.
        response = None
        for attempt in range(3):
            try:
                response = chain.invoke({"context": formatted_context, "question": question})
                break
            except Exception as e:
                if attempt < 2 and any(msg in str(e) for msg in ("Connection refused", "RemoteProtocolError", "Server disconnected", "ConnectError")):
                    print(f"[!] Ollama cargando modelo, reintentando ({attempt+2}/3)...")
                    time.sleep(10)
                else:
                    raise
        
        # 5. Añadir fuentes enriquecidas (deduplicadas por acta+tema)
        sources = "\n\n" + "="*60 + "\nFUENTES UTILIZADAS:\n"
        seen_src_keys: set = set()
        src_idx = 1
        for d in docs:
            src = os.path.basename(d.metadata.get("source", "Acta"))
            top = d.metadata.get("topic", "")
            key = (src, top[:60])
            if key in seen_src_keys:
                continue
            seen_src_keys.add(key)
            spk = d.metadata.get("speaker", "")
            snippet = d.page_content.replace('\n', ' ')[:400] + "..."
            sources += f" [{src_idx}] {src} | {spk} | {top[:60]}\n     -> \"{snippet}\"\n\n"
            src_idx += 1
        
        return response + sources

# Singleton compartido para que app.py pueda importarlo sin re-inicializar la BD
_rag_instance: Optional[RAGPipeline] = None

def get_rag() -> RAGPipeline:
    """Devuelve la instancia global del RAG (la crea la primera vez que se llama)."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
        _rag_instance.vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=_rag_instance.embeddings
        )
        # Precalentar el LLM para que el primer query no espere la carga del modelo
        try:
            _rag_instance.llm.invoke("ok")
        except Exception:
            pass
    return _rag_instance

def main():
    parser = argparse.ArgumentParser(description="RAG System for Bilbao Actas")
    parser.add_argument("--query", type=str, help="Pregunta directa")
    args = parser.parse_args()

    rag = RAGPipeline()
    if not os.path.exists(CHROMA_PATH):
        rag.create_vector_store()
    else:
        rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)

    # Precalentar mistral ANTES de que empiece cualquier query para evitar que
    # el primer LLM call (MultiQuery) falle por RAM y corrompa el pool HTTP de httpx
    print("[*] Cargando modelo LLM...")
    for attempt in range(3):
        try:
            rag.llm.invoke("ok")
            print("[+] Modelo listo.")
            break
        except Exception as e:
            if attempt < 2:
                print(f"[!] Modelo cargando, reintentando ({attempt+2}/3)...")
                time.sleep(10)

    if args.query:
        print("\nRESPUESTA:")
        try:
            print(rag.query(args.query))
        except UnicodeEncodeError:
            sys.stdout.buffer.write(rag.query(args.query).encode('utf-8'))
    else:
        print(f"\n--- RAG Bilbao Ready [Model: {LLM_MODEL_GROQ if GROQ_API_KEY else LLM_MODEL_LOCAL}] ---")
        while True:
            try:
                q = input("\nPregunta: ")
                if not q.strip(): continue
                if q.lower() in ['exit', 'quit', 'salir']: break
                
                res = rag.query(q)
                try:
                    print(res)
                except UnicodeEncodeError:
                    sys.stdout.buffer.write(res.encode('utf-8'))
                    
            except (KeyboardInterrupt, EOFError): break
            except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    main()
