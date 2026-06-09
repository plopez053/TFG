import os
import argparse
import re
import glob
import sys
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "actas_scalability")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_final_v11")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

# --- Diccionario Temático Global ---
# Cada entrada mapea triggers (palabras del usuario) a:
#   "search"   -> variantes de búsqueda semántica en la BD vectorial
#   "synonyms" -> stems de 6 chars para enriquecer el scoring de relevancia de temas
# Añade aquí nuevos temas conforme sea necesario.
TOPIC_VARIANTS_MAP = [
    {"triggers": ["tasa", "turis", "ecotasa", "turism"],
     "search":   ["tasa turística bilbao", "ecotasa turística", "impuesto turismo bilbao"],
     "synonyms": ["turism", "ecotasa", "alojam", "impost"]},
    {"triggers": ["vivienda", "alquiler", "piso", "inmobil", "habitac"],
     "search":   ["vivienda social bilbao", "vivienda proteccion oficial",
                  "zona tensionada bilbao", "viviendas municipales bilbao"],
     "synonyms": ["vivien", "alquil", "protec", "habita", "tensan", "inmovi"]},
    {"triggers": ["transporte", "metro", "autobus", "autobús", "bizi", "movilidad", "bilbobus"],
     "search":   ["transporte público bilbao", "bilbobus lineas autobus", "movilidad urbana bilbao"],
     "synonyms": ["transp", "autobu", "movili", "bilbob", "viajero", "trafico"]},
    {"triggers": ["medio ambiente", "sostenib", "ecolog", "verde", "contaminacion",
                  "clima", "residuo", "energia", "emisio"],
     "search":   ["medio ambiente bilbao", "cambio climático bilbao",
                  "zonas verdes bilbao", "contaminación bilbao",
                  "residuos urbanos bilbao", "sostenibilidad municipal bilbao",
                  "transición ecológica bilbao", "emisiones CO2 bilbao",
                  "energía renovable bilbao", "arbolado urbano bilbao"],
     "synonyms": ["ambien", "sosten", "ecolog", "climat", "residu",
                  "contam", "parque", "emisio", "energi", "verdes", "arbolad"]},
    {"triggers": ["presupuesto", "gasto", "inversion", "financ", "deficit"],
     "search":   ["presupuesto municipal bilbao", "presupuesto general villa bilbao",
                  "aprobacion presupuestos bilbao", "gasto publico municipal bilbao"],
     "synonyms": ["presup", "invers", "financ", "partid", "ejercic", "deficit"]},
    {"triggers": ["servicio social", "depend", "infanci", "bienestar"],
     "search":   ["servicios sociales bilbao", "personas mayores bilbao",
                  "dependencia municipal bilbao", "menores infancia bilbao"],
     "synonyms": ["social", "depend", "infanc", "bienes", "menor"]},
    {"triggers": ["urbanismo", "obra", "edificio", "rehabilit", "construc"],
     "search":   ["plan urbanístico bilbao", "obras municipales bilbao",
                  "rehabilitación edificios bilbao", "plan general ordenacion"],
     "synonyms": ["urbani", "rehabi", "edific", "constr", "planea", "zorrot"]},
    {"triggers": ["cultura", "deporte", "festival", "museo", "teatro"],
     "search":   ["cultura bilbao", "deporte municipal bilbao", "actividades culturales bilbao"],
     "synonyms": ["cultur", "deport", "festiv", "museo", "teatro", "instal"]},
    {"triggers": ["seguridad", "policia", "convivencia", "delincuencia"],
     "search":   ["seguridad ciudadana bilbao", "policia municipal bilbao", "convivencia vecinal bilbao"],
     "synonyms": ["seguri", "polici", "conviv", "delict"]},
    {"triggers": ["empleo", "trabajo", "desempleo", "paro", "economia", "comercio"],
     "search":   ["empleo bilbao", "desempleo paro bilbao", "economia municipal bilbao"],
     "synonyms": ["empleo", "trabaj", "desempl", "econom", "comerc"]},
    {"triggers": ["euskera", "euskara", "idioma", "lengua"],
     "search":   ["euskera bilbao", "normalizacion linguistica bilbao"],
     "synonyms": ["eusker", "euskar", "idioma", "lingua"]},
]

class RAGPipeline:
    def __init__(self):
        """Inicializa el motor RAG con el modelo de embeddings configurado."""
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = None
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=4096)

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

    def load_and_split_documents(self) -> List[Document]:
        """Carga los PDFs y los divide en fragmentos con metadatos de forma recursiva."""
        all_chunks = []
        # Buscamos PDFs en DATA_PATH y todas sus subcarpetas (2023, 2024, 2025...)
        pdf_files = glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True)
        
        if not pdf_files:
            print(f"[!] No se encontraron archivos PDF en {DATA_PATH}")
            return []
        
        print(f"[*] Encontrados {len(pdf_files)} archivos PDF. Procesando...")
        
        all_chunks = []
        speaker_regex = re.compile(r'(?:(?:EL|LA)\s+)?(?:SR\.|SRA\.)\s+([A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)\s*[:.]', re.IGNORECASE)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

        for path in tqdm(pdf_files, desc="Procesando Actas", unit="pdf"):
            try:
                date = self._extract_date_from_filename(path)
                pdf_loader = PyPDFLoader(path)
                pages = pdf_loader.load()
                party_map = self._get_party_mapping(pages)
                
                full_text = "\n".join([page.page_content for page in pages])
                split_regex = re.compile(
                    r'(?=\n(?:[\s]*)(?:\d+)\.-?\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA)'
                    r'|\n\s*-\d+-\s*\n\s*(?:PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA|Proposición|Propuesta|Moción|Mozio|Dictamen|Irizpen|Asunto|Gaia|Proposamen))',
                    re.IGNORECASE
                )
                segments = split_regex.split(full_text)
                
                chunk_index_in_doc = 0
                for segment in segments:
                    if not segment.strip(): continue
                    
                    current_speaker, current_party = "Desconocido", "Desconocido"
                    
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
                    
                    for chunk_text in text_splitter.split_text(segment):
                        match = speaker_regex.search(chunk_text)
                        if match:
                            current_speaker = match.group(1).strip()
                            if len(current_speaker) < 50:
                                for kn, kp in party_map.items():
                                    if kn in current_speaker or current_speaker in kn:
                                        current_party = kp
                                        break
                        
                        all_chunks.append(Document(
                            page_content=f"ASUNTO: {current_topic}\nORADOR: {current_speaker} ({current_party})\n\n{chunk_text}",
                            metadata={
                                "source": path, "date": date, "speaker": current_speaker,
                                "party": current_party, "topic": current_topic, "chunk_index": chunk_index_in_doc
                            }
                        ))
                        chunk_index_in_doc += 1
            except Exception as e:
                print(f"[!] Error cargando {path}: {e}")
                
        return all_chunks

    def create_vector_store(self, force_rebuild: bool = False):
        """Crea o carga la base de datos vectorial ChromaDB."""
        if os.path.exists(CHROMA_PATH) and not force_rebuild:
            print(f"[*] Cargando base de datos vectorial desde {CHROMA_PATH}...")
            self.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        else:
            if force_rebuild and os.path.exists(CHROMA_PATH):
                import shutil
                shutil.rmtree(CHROMA_PATH)
            
            chunks = self.load_and_split_documents()
            if not chunks: return
            print("[*] Generando nuevos embeddings (esto puede tardar unos minutos)...")
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=CHROMA_PATH)
            print(f"[+] Base de datos guardada en {CHROMA_PATH}")

    # --- Fase de Recuperación (Retrieval) ---

    def _get_temporal_filter(self, question: str) -> Optional[Dict[str, Any]]:
        """Analiza la pregunta para aplicar filtros de fecha inteligentes."""
        months = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        year_match = re.search(r'(20\d{2})', question)
        month_found = next((m for m in months if m in question.lower()), None)
        
        if not year_match: return None, []
        
        target_year = year_match.group(1)
        target_month = f"{str(months.index(month_found)+1).zfill(2)}" if month_found else None
        target_pattern = f"{target_month}-{target_year}" if target_month else target_year
        
        # Obtener fechas únicas desde el sistema de archivos (Escalabilidad Real)
        # Es mucho más rápido que consultar 60k metadatos en la DB
        known_dates = []
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files:
                if file.endswith(".pdf"):
                    d = self._extract_date_from_filename(file)
                    if d: known_dates.append(d)
        
        known_dates = list(set(known_dates))
        valid_dates = [d for d in known_dates if target_pattern in d]
        
        if valid_dates:
            print(f"[*] Filtro Temporal: {target_pattern} ({len(valid_dates)} actas detectadas)")
            return {"date": {"$in": valid_dates}}, valid_dates
        return None, []

    def _expand_context_by_topic(self, initial_docs: List[Document], question: str) -> List[Document]:
        """Técnica de la 'Aspiradora': Expande los fragmentos semánticos a debates completos.
        
        Usa dos pasadas:
        1. Puntuar los temas que ya llegaron de la búsqueda semántica.
        2. Búsqueda directa en los títulos de tema de la BD para garantizar cobertura total.
        """
        target_topics, seen = [], set()

        stopwords = {
            "partido", "popular", "grupo", "municipal", "municipales", "proposicion", "proposizioa",
            "acuerdo", "bilbao", "pleno", "sobre", "mayo", "2024", "resultado",
            "exacto", "votacion", "propone", "presenta", "mocion", "mozioa",
            "junio", "abril", "marzo", "enero", "febrero", "septiembre", "octubre",
            "noviembre", "diciembre", "reunion", "sesion",
            "decidio", "propuso", "dime", "quien", "cual",
            "este", "esta", "estos", "estas", "2023", "2025", "2026",
            # Palabras estructurales de los títulos de actas
            "propuesta", "propuestas", "debate", "debates", "habido", "habida",
            "habia", "cuales", "hubo", "hay", "sido", "han",
            # Términos que aparecen en TODOS los títulos y no aportan señal
            "presenta", "cuya", "parte", "dispositiva", "tenor", "literal",
            "siguiente", "adoptado", "acuerdo", "acuerdos", "tomado", "tomados",
            "relacionado", "relacionados", "plantea", "adopcion",
            # Artículos, preposiciones y palabras funcionales cortas del castellano
            "que", "los", "las", "del", "con", "por", "para", "una", "uno",
            "mas", "pero", "como", "sus", "son", "fue", "era", "les", "nos",
            "largo", "anos", "ano", "vez", "todo", "toda", "todos"
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

        # MEJORA: Ampliar q_keywords con sinónimos temáticos del diccionario global.
        # Permite que el scoring detecte temas relacionados aunque el título del acta
        # use terminología distinta a la de la pregunta del usuario.
        # Ej: "medio ambiente" → añade ["contam", "residu", "parque", "verdes"...]
        # así un tema "Proposición sobre zonas verdes" obtiene score alto.
        q_lower_for_syn = normalize(question)
        for entry in TOPIC_VARIANTS_MAP:
            if any(normalize(tr) in q_lower_for_syn for tr in entry["triggers"]):
                for syn in entry["synonyms"]:
                    if syn not in q_keywords:
                        q_keywords.append(syn)

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
                    # NUEVA LÓGICA: Extraer solo el fragmento inicial (donde suele estar la propuesta)
                    # y el fragmento final (donde suele estar la votación)
                    start_idx = min(topic_indices)
                    start_chunks = [pairs[start_idx]]
                    
                    # Buscamos el final del tema (donde cambia el topic o se acaba el acta)
                    topic_end_idx = start_idx
                    for i in range(start_idx, len(pairs)):
                        if pairs[i][0].get('topic') == t_name:
                            topic_end_idx = i
                        else:
                            break
                    
                    end_chunks = [pairs[topic_end_idx]] if topic_end_idx != start_idx else []
                    selected = start_chunks + end_chunks
                    
                    # Extraemos el número del tema para el filtro de seguridad (ej: "25.")
                    t_topic_number = t_name.split('.')[0].strip() + "."
                    seen_content = set()
                    for m, d in selected:
                        # FILTRO DE SEGURIDAD: Si el fragmento tiene un metadato de OTRO tema, lo saltamos
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
            
            # Quitar las lineas ASUNTO: y ORADOR: del inicio (el titulo ya lo tenemos)
            lines = content.split('\n')
            if lines and lines[0].startswith('ASUNTO:'):
                lines = lines[1:]
            if lines and lines[0].startswith('ORADOR:'):
                lines = lines[1:]
            content = '\n'.join(lines).strip()
            
            if key not in groups:
                groups[key] = []
            groups[key].append(content)
        
        # Formatear cada grupo como un bloque compacto de debate
        final_text = ""
        # Regex para extraer resultados de votación del texto
        # vote_re: captura resultados numéricos de votación.
        # Usa DOTALL para no cortarse en saltos de línea del PDF.
        vote_re = re.compile(
            r'(?:Votos\s+emitidos[:\s]+(\d+)[.\s]*)?'
            r'Votos\s+afirmativos[:\s]+(\d+).{0,60}'
            r'Votos\s+negativos[:\s]+(\d+)',
            re.IGNORECASE | re.DOTALL
        )
        # result_re: captura la frase de resultado textual.
        # FIX: usa re.DOTALL para no cortarse en saltos de línea del PDF.
        # Antes: [^\n]{0,100} → se cortaba en mitad de la frase (ej: "se aprueba la\nenmienda")
        # Ahora: .{0,200} con DOTALL captura la frase completa.
        result_re = re.compile(
            r'(?:se\s+(?:acepta|aprueba|rechaza|desestima|deniega).{0,200}|'
            r'queda\s+(?:aprobad[ao]|rechazad[ao]|desestimad[ao]).{0,200}|'
            r'resulta\s+(?:aprobad[ao]|rechazad[ao]).{0,150})',
            re.IGNORECASE | re.DOTALL
        )

        for (date, topic, source), contents in groups.items():
            raw = "\n---\n".join(c for c in contents if c.strip())
            # Texto aplanado: sustituye saltos de línea por espacios para que las frases
            # cortadas por el OCR del PDF no rompan las coincidencias de los regex.
            raw_flat = re.sub(r'\s+', ' ', raw)

            # Extraer resultado de votación si está en el texto (búsqueda sobre texto aplanado)
            resultado = None
            vm = vote_re.search(raw_flat)
            if vm:
                total = vm.group(1) or "?"
                favor = vm.group(2)
                contra = vm.group(3)
                resultado = f"Votos emitidos: {total} | A favor: {favor} | En contra: {contra}"
            else:
                rm = result_re.search(raw_flat)
                if rm:
                    resultado = rm.group(0).strip()
            
            # Limitar el texto del debate a 800 chars
            if len(raw) > 800:
                raw = raw[:800] + "..."
            
            short_topic = topic[:120]
            block = f"[PLENO: {date}] {short_topic}\n{raw}\n"
            if resultado:
                block += f"RESULTADO: {resultado}\n"
            block += "\n"
            final_text += block
            
        return final_text.strip()



    # --- Punto de Entrada Principal ---

    def query(self, question: str) -> str:
        """Flujo principal: Recuperación, Expansión y Generación."""
        if not self.vector_store: self.create_vector_store()

        # 1. MultiQuery: Generar variantes de búsqueda
        mq_prompt = "Genera 3 variantes de: '{question}' centradas en el sujeto principal. Una por línea:"
        try:
            vars_txt = self.llm.invoke(mq_prompt.format(question=question)).content
            variations = [v.strip() for v in vars_txt.split('\n') if v.strip()] + [question]
        except Exception: variations = [question]

        # Enriquecer proactivamente con variantes de búsqueda adicionales basadas en keywords
        # (esto mejora la cobertura semántica sin depender del modelo MultiQuery)
        q_lower = question.lower()
        # Usar el diccionario temático global para enriquecer proactivamente la búsqueda.
        # Antes: solo 3 temas hardcoded (turismo, vivienda, transporte).
        # Ahora: 11 temas cubriendo toda la agenda municipal (medio ambiente, presupuestos, etc.)
        # y usando elif → ahora usa extend() para acumular variantes de múltiples temas.
        extra_variants = []
        for entry in TOPIC_VARIANTS_MAP:
            if any(kw in q_lower for kw in entry["triggers"]):
                extra_variants.extend(entry["search"])

        if extra_variants:
            seen_vars: set = set()
            variations = [v for v in (variations + extra_variants) if not (v in seen_vars or seen_vars.add(v))]

        # 2. Búsqueda Inicial con Filtros
        filter_dict, valid_dates = self._get_temporal_filter(question)
        all_initial_docs = []
        for v in variations:
            if filter_dict:
                for d_val in valid_dates:
                    try:
                        res = self.vector_store.similarity_search_with_score(v, k=150, filter={"date": {"$eq": d_val}})
                        # Pre-filtrar: descartar docs con distancia semántica muy alta (>1.4)
                        # Esto elimina ruido antes de que llegue al scoring de temas.
                        all_initial_docs.extend([d for d, s in res if s < 1.4])
                    except: pass
            else:
                try:
                    res = self.vector_store.similarity_search_with_score(v, k=150)
                    # Pre-filtrar: descartar docs con distancia semántica muy alta (>1.4)
                    all_initial_docs.extend([d for d, s in res if s < 1.4])
                except: pass

        # 3. Expansión y Formateo
        docs, target_topics = self._expand_context_by_topic(all_initial_docs, question)
        
        # Filtro de ruido: eliminamos docs de temas que NO están en la lista de target_topics
        # (ya pasaron el umbral del 40% en _expand_context_by_topic)
        if target_topics:
            valid_prefixes = [t['topic'].split('.')[0].strip() + "." for t in target_topics]
            docs = [d for d in docs if any(d.metadata.get('topic', '').startswith(p) for p in valid_prefixes)]
        # Ordenar docs cronológicamente (más antiguo primero)
        # Esto garantiza que el contexto represente todos los años de forma equilibrada
        # en lugar de meter todos los docs de un año juntos al principio
        def _date_sort_key(d):
            date_str = d.metadata.get('date', '01-01-1900')
            try:
                parts = date_str.split('-')
                if len(parts) == 3:
                    return (int(parts[2]), int(parts[1]), int(parts[0]))
            except:
                pass
            return (1900, 1, 1)
        docs = sorted(docs, key=_date_sort_key)
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

        # Guardar para inspección técnica
        with open("debug_context.txt", "w", encoding="utf-8") as f: f.write(formatted_context)

        if is_multi_session:
            dates_found = ', '.join(sorted(unique_dates))
            sys_prompt = f"""Eres el Cronista Oficial de Bilbao, experto en historia municipal.

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
            sys_prompt = """Eres el Cronista Oficial de Bilbao. Tu misión es relatar lo ocurrido en el Pleno.
        
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
        
        response = chain.invoke({"context": formatted_context, "question": question})
        
        # 5. Añadir fuentes enriquecidas
        sources = "\n\n" + "="*60 + "\nFUENTES UTILIZADAS:\n"
        for i, d in enumerate(docs):
            src, top, spk = os.path.basename(d.metadata.get("source", "Acta")), d.metadata.get("topic", ""), d.metadata.get("speaker", "")
            snippet = d.page_content.replace('\n', ' ')[:400] + "..."
            sources += f" [{i+1}] {src} | {spk} | {top[:60]}\n     -> \"{snippet}\"\n\n"
        
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
    return _rag_instance

def main():
    parser = argparse.ArgumentParser(description="RAG System for Bilbao Actas")
    parser.add_argument("--rebuild", action="store_true", help="Reconstruir base de datos")
    parser.add_argument("--query", type=str, help="Pregunta directa")
    args = parser.parse_args()

    rag = RAGPipeline()
    if args.rebuild:
        rag.create_vector_store(force_rebuild=True)
    elif not os.path.exists(CHROMA_PATH):
        rag.create_vector_store()
    else:
        rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)

    if args.query:
        print("\nRESPUESTA:")
        try:
            print(rag.query(args.query))
        except UnicodeEncodeError:
            sys.stdout.buffer.write(rag.query(args.query).encode('utf-8'))
    else:
        print(f"\n--- RAG Bilbao Ready [Model: {LLM_MODEL}] ---")
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
