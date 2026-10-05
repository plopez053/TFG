"""Indexado: cada acta PDF se trocea por puntos del orden del día y después en
fragmentos de 1.200 caracteres, con metadatos (fecha, orador, grupo, tema,
página y resultado de la votación), y se guarda en ChromaDB."""
import glob
import os
import re
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from comun.grupos import normaliza_grupo as _normaliza_grupo_partido
from comun.rutas import CHROMA_PATH, DATA_PATH


_TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)


_SPEAKER_RE = re.compile(
    r'(?:(?:EL|LA)\s+)?(?:SR\.|SRA\.)\s+([A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)\s*[:.]', re.IGNORECASE)


# tipos de punto del orden del día (castellano y euskera)
_TIPOS = "PROPUESTA|PROPOSAMENA|MOCIÓN|MOZIOA|DICTAMEN|IRIZPENA|ASUNTO|GAIA|PROPOSICIÓN|PROPOSIZIOA"


_TIPOS_HIST = _TIPOS + "|Proposición|Propuesta|Moción|Mozio|Dictamen|Irizpen|Asunto|Gaia|Proposamen"


# formato antiguo (2002-2009): "- N -" marca tanto puntos como números de página;
# solo es un punto real si le sigue uno de estos tipos
_TIPOS_ANTIGUOS = (r'(?:Proposici[oó]n|Proposizioa|Propuesta|Preguntas?|'
                   r'Se\s+da\s+cuenta|Dar\s+cuenta|Dictamen|Moci[oó]n|'
                   r'Comparecencia|Interpelaci[oó]n|Declaraci[oó]n|Aprobar)')


# troceo: formato moderno ("N. PROPUESTA" o "-N- PROPUESTA")
_SPLIT_RE = re.compile(
    r'(?=\n(?:[\s]*)(?:\d+)\.-?\s*(?:' + _TIPOS + r')'
    r'|\n\s*-\d+-\s*\n\s*(?:' + _TIPOS_HIST + r'))',
    re.IGNORECASE
)


# troceo: formato antiguo por orden del día, o por página si no hay puntos (extraordinarias)
_SPLIT_OLD_AGENDA_RE = re.compile(r'(?=\n[\s]*-\s*\d+\s*-\s*\n[\s]*' + _TIPOS_ANTIGUOS + r')', re.IGNORECASE)


_SPLIT_OLD_PAGE_RE = re.compile(r'(?=\n-\s*\d+\s*-\s*\n)')


# tema del segmento según el formato
_TOPIC_STD_RE = re.compile(r'^\s*(\d+\.-?\s*(?:' + _TIPOS + r').{0,400})', re.IGNORECASE | re.DOTALL)


_TOPIC_HIST_RE = re.compile(r'^\s*-\s*(\d+)\s*-\s*\n\s*((?:' + _TIPOS_HIST + r').{0,400})', re.IGNORECASE | re.DOTALL)


_TOPIC_OLD_RE = re.compile(r'^\s*-\s*(\d+)\s*-\s*\n[\s]*(' + _TIPOS_ANTIGUOS + r'.{0,140})', re.IGNORECASE | re.DOTALL)


_TOPIC_OLD_PAGE_RE = re.compile(r'^-\s*\d+\s*-\s*\n\s*(.{0,200})', re.DOTALL)


# cifras de votación; ventanas amplias porque en las actas bilingües entre
# "Votos emitidos" y "Votos afirmativos" se intercala el bloque en euskera
_VOTE_RE = re.compile(
    r'Votos\s+emitidos[:\s]+(\d+)'
    r'.{0,500}?Votos\s+afirmativos[:\s]+(\d+)'
    r'(?:.{0,500}?Votos\s+negativos[:\s]+(\d+))?'
    r'(?:.{0,500}?Abstenciones?[:\s]+(\d+))?',
    re.IGNORECASE | re.DOTALL
)


# frase de resultado: [^.] para parar en el punto que la cierra. Incluye
# "queda aceptada/retirada" y "decae", que aparecen sin votación previa
# (p.ej. enmienda aceptada "sin necesidad de someterla a votación").
_RESULT_RE = re.compile(
    r'(?:se\s+(?:acepta|aprueba|rechaza|desestima|deniega)\b[^.]{0,30}?'
    r'(?:enmienda|proposici[óo]n|propuesta|moci[óo]n|mozio|proposamen)[^.]{0,400}|'
    r'queda\s+(?:aprobad[ao]|rechazad[ao]|desestimad[ao]|aceptad[ao]|retirad[ao])[^.]{0,400}|'
    r'resulta\s+(?:aprobad[ao]|rechazad[ao])[^.]{0,400}|'
    r'decaen?\b[^.]{0,400})',
    re.IGNORECASE | re.DOTALL
)


# fórmulas de cierre propias de secretaría, que no aparecen en el debate
_STRONG_RESULT_RE = re.compile(
    r'\b(?:queda\s+(?:aprobad[ao]|rechazad[ao]|desestimad[ao]|aceptad[ao]|retirad[ao])'
    r'|resulta\s+(?:aprobad[ao]|rechazad[ao])|decaen?\b)',
    re.IGNORECASE
)


# resultados sin cifras (unanimidad o asentimiento), frecuentes en actas antiguas
_UNANIM_RE = re.compile(
    r'(?:(?:el\s+pleno(?:\s+municipal)?|excmo\.?\s+ayuntamiento\s+pleno|aprobar)'
    r'[^.]{0,60}por\s+unanimidad[^.]{0,200}'
    r'|se\s+(?:aprueba|acuerda|aprueban|desestiman?|rechazan?)[^.]{0,60}'
    r'por\s+(?:unanimidad|asentimiento)[^.]{0,200})',
    re.IGNORECASE
)


# lo que sigue al resultado y no forma parte de él (pies de página, URL, hora...)
_FIN_RESULTADO_RE = re.compile(
    r'\s*-{3,}\s*|\s+-\s+|\s*https?://|\s+Egiaztatzeko|\s+Verificaci|\s+Siendo\s+las\b')


# extrae la fecha del nombre del fichero (ej: '27-02-2025_...pdf')
def _fecha_de_fichero(path: str) -> str:
    match = re.search(r'(\d{2}-\d{2}-\d{4})', os.path.basename(path))
    return match.group(1) if match else "Fecha desconocida"


# parte el texto del acta en segmentos; devuelve (segmentos, es_formato_antiguo)
def _trocear_acta(full_text: str) -> Tuple[List[str], bool]:
    segments = _SPLIT_RE.split(full_text)
    if len(segments) > 2:
        return segments, False
    agenda_segs = [s for s in _SPLIT_OLD_AGENDA_RE.split(full_text) if s.strip()]
    if len(agenda_segs) > 2:
        return agenda_segs, True
    return _SPLIT_OLD_PAGE_RE.split(full_text), True


def _limpiar_resultado(txt: str) -> str:
    return _FIN_RESULTADO_RE.split(txt.strip())[0].strip()


# resultado de la votación de un segmento, o None.
# Prioridad: cifras > unanimidad/asentimiento > fórmula de cierre. Un "se
# aprueba" suelto sin cifras se descarta porque puede ser una cita del debate.
def _extraer_resultado(segment: str) -> Optional[str]:
    seg_flat = re.sub(r'\s+', ' ', segment)
    # última coincidencia: si antes se vota una enmienda, el resultado que
    # cuenta es el de la proposición, al final del segmento
    rms = list(_RESULT_RE.finditer(seg_flat))
    resultado_text = _limpiar_resultado(rms[-1].group(0)) if rms else None

    resultado_num = None
    votes = list(_VOTE_RE.finditer(seg_flat))
    if votes:
        emitidos, favor, contra, absten = votes[-1].groups()
        # basta con "a favor": las votaciones unánimes no traen "en contra"
        if favor:
            partes = [f"a favor: {favor}"]
            if contra:
                partes.append(f"en contra: {contra}")
            if absten:
                partes.append(f"abstenciones: {absten}")
            cab = f"Votos emitidos: {emitidos} | " if emitidos else ""
            resultado_num = cab + ", ".join(partes)

    if resultado_num:
        return f"{resultado_text} ({resultado_num})" if resultado_text else resultado_num
    if resultado_text and (re.search(r'unanimidad|asentimiento', resultado_text, re.I)
                           or _STRONG_RESULT_RE.search(resultado_text)):
        return resultado_text
    um = _UNANIM_RE.search(seg_flat)
    return _limpiar_resultado(um.group(0)) if um else None


# título del punto del orden del día al que pertenece el segmento
def _extraer_tema(segment: str, is_old_format: bool) -> str:
    if is_old_format:
        tm_old = _TOPIC_OLD_RE.search(segment.lstrip('\n'))
        if tm_old:
            texto = re.sub(r'\s+', ' ', tm_old.group(2)).strip()
            return f"{tm_old.group(1)}. {texto[:115]}"
        # extraordinarias troceadas por página: primera línea
        topic_raw = _TOPIC_OLD_PAGE_RE.search(segment.lstrip('\n'))
        if topic_raw:
            first_line = topic_raw.group(1).strip().split('\n')[0].strip()
            first_line = re.sub(r' {2,}', ' ', first_line)
            return first_line[:120] if first_line else "General / Introducción"
        return "General / Introducción"

    topic_match_std = _TOPIC_STD_RE.search(segment)
    if topic_match_std:
        return topic_match_std.group(1).strip().replace('\n', ' ')
    topic_match_hist = _TOPIC_HIST_RE.search(segment)
    if topic_match_hist:
        clean_text = topic_match_hist.group(2).strip().replace('\n', ' ')
        return f"{topic_match_hist.group(1)}. {clean_text}"
    return "General / Introducción"


# métodos de RAGPipeline (vectorial/pipeline.py) para construir el índice
class Indexado:

    # nombres de concejales -> grupo, a partir de la lista de asistentes de las primeras páginas
    def _get_party_mapping(self, pages: List[Any]) -> Dict[str, str]:
        party_mapping = {}
        header_text = "\n".join([p.page_content for p in pages[:10]])
        current_party = "Gobierno Local/Otros"

        for line in header_text.split('\n'):
            line = line.strip()
            if not line:
                continue

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
                raw_party = paren.group(1).strip() if paren else current_party
                normalized = _normaliza_grupo_partido(raw_party)
                # si no se puede normalizar se conserva el texto original
                party_mapping[name] = normalized if normalized != "Desconocido" else raw_party

        return party_mapping

    # acta PDF -> chunks con metadatos (fecha, orador, grupo, tema, página, resultado)
    def _process_single_pdf(self, path: str) -> List[Document]:
        date = _fecha_de_fichero(path)
        pages = PyPDFLoader(path).load()
        party_map = self._get_party_mapping(pages)

        full_text = "\n".join(p.page_content for p in pages)
        page_starts, cursor = [], 0
        for p in pages:
            page_starts.append(cursor)
            cursor += len(p.page_content) + 1

        def page_for_offset(offset: int) -> int:
            return max(1, bisect_right(page_starts, offset))

        segments, is_old_format = _trocear_acta(full_text)
        chunks: List[Document] = []
        seg_search_start = 0
        for segment in segments:
            if not segment.strip():
                continue
            segment_offset = full_text.find(segment, seg_search_start)
            if segment_offset != -1:
                seg_search_start = segment_offset + max(1, len(segment) - 200)

            vote_result = _extraer_resultado(segment)
            current_topic = _extraer_tema(segment, is_old_format)
            current_speaker, current_party = "Desconocido", "Desconocido"

            chunk_search_start = 0
            for chunk_text in _TEXT_SPLITTER.split_text(segment):
                match = _SPEAKER_RE.search(chunk_text)
                if match:
                    current_speaker = match.group(1).strip()
                    if len(current_speaker) < 50:
                        for kn, kp in party_map.items():
                            if kn in current_speaker or current_speaker in kn:
                                current_party = kp
                                break

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
                    "chunk_index": len(chunks), "page": page_num,
                }
                if vote_result:
                    metadata["vote_result"] = vote_result
                chunks.append(Document(
                    page_content=f"ASUNTO: {current_topic}\nORADOR: {current_speaker} ({current_party})\n\n{chunk_text}",
                    metadata=metadata
                ))
        return chunks

    def load_and_split_documents(self) -> List[Document]:
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

    # carga ChromaDB o, si no existe, la crea desde los PDF
    def create_vector_store(self):
        if os.path.exists(CHROMA_PATH):
            print(f"[*] Cargando base de datos vectorial desde {CHROMA_PATH}...")
            self.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        else:
            chunks = self.load_and_split_documents()
            if not chunks:
                return
            print("[*] Generando embeddings...")
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=CHROMA_PATH)
            print(f"[+] Base de datos guardada en {CHROMA_PATH}")
