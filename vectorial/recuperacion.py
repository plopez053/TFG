"""
Recuperación: de la pregunta a los fragmentos de las actas que la responden.

  1. análisis de la pregunta (palabras clave, nombres propios, grupo, fechas)
  2. cuatro canales de búsqueda: semántico (MultiQuery), literal por palabras
     poco frecuentes, temático (taxonomía SKOS) y de proposiciones
  3. filtro de grupo y reordenación (Cohere, o el reordenador local)
  4. selección final: los debates que responden se recuperan completos
"""
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from comun.grupos import extrae_grupo as _extrae_grupo_partido
from comun.proveedores import COHERE_API_KEY, rerank as _cohere_rerank
from comun.rutas import CHROMA_PROPOSICIONES_PATH, DATA_PATH, ONTOLOGIA, TEMAS_SKOS
from vectorial.indexado import _fecha_de_fichero


# =============================================================================
# Utilidades de texto y fechas
# =============================================================================

# quita acentos y diacríticos
def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()


# minúsculas, sin acentos ni signos
def normalizar(txt: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', strip_accents(txt.lower()))


ESTACIONES = {"primavera": (3, 4, 5, 6), "verano": (6, 7, 8, 9),
               "otoño": (9, 10, 11, 12), "invierno": (12, 1, 2, 3)}


MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11,
    "diciembre": 12,
}


# clave de orden cronológico (año, mes, día) a partir de una fecha 'DD-MM-YYYY'
def clave_fecha(date_str: str) -> Tuple[int, int, int]:
    try:
        parts = (date_str or '01-01-1900').split('-')
        if len(parts) == 3:
            return (int(parts[2]), int(parts[1]), int(parts[0]))
    except (ValueError, AttributeError):
        pass
    return (1900, 1, 1)


# clave de orden cronológico a partir del metadato 'date' de un Document
def clave_fecha_doc(doc: Document) -> Tuple[int, int, int]:
    return clave_fecha(doc.metadata.get('date', ''))


# =============================================================================
# 1. Análisis de la pregunta
# =============================================================================

# palabras presentes en casi cualquier título o pregunta, sin valor temático
_STOPWORDS_TEMATICAS = {
    "bilbao", "municipal", "municipales", "partido", "popular", "grupos",
    "acuerdo", "acuerdos", "proposicion", "proposizioa", "propuesta", "propuestas",
    "mocion", "mozioa", "debate", "debates", "sesion", "reunion",
    "resultado", "votacion", "propone", "propuso", "propuesto", "presenta", "plantea",
    "siguiente", "adoptado", "tomado", "tomados", "relacionado", "relacionados",
    "dispositiva", "literal", "adopcion", "decidio", "habido", "habida",
    "cuales", "exacto",
    "febrero", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
}


# palabras del enunciado de la pregunta, no del tema (sin tildes)
_STOPWORDS_ENUNCIADO = {
    "debatido", "debatida", "debatidos", "debatio", "debatieron", "debatir", "debatiendo",
    "iniciativa", "iniciativas", "medida", "medidas", "tema", "temas", "asunto", "asuntos",
    "dicho", "dijo", "dijeron", "hecho", "hicieron", "ocurrio", "ocurrido",
    "presentado", "presentada", "presentados", "presentadas", "planteado", "planteada",
    "aprobado", "aprobada", "aprobados", "aprobadas", "rechazado", "rechazada",
    "ayuntamiento", "pleno", "plenos", "concejal", "concejales", "grupo",
    "sobre", "para", "como", "cual", "cuales", "cuando", "donde", "quien", "quienes",
    "este", "esta", "estos", "estas", "todo", "todos", "toda", "todas", "entre",
    "desde", "hasta", "hacia", "contra", "segun", "tras", "ante", "cada", "algun",
    "alguna", "algunos", "algunas", "otro", "otra", "otros", "otras", "mismo", "misma",
    "dado", "dada", "sido", "sera", "habia", "tiene", "tienen", "hace", "hecha",
    "apoyo", "relacion", "respecto", "hablado", "habla", "hablar", "decir",
    "voto", "votaron", "votado", "votar", "aprobo", "aprobaron", "rechazo", "rechazaron",
    "presento", "presentaron", "trato", "trataron", "tratado", "tratar", "ultima", "ultimo",
    "ultimas", "ultimos", "reciente", "cuanto", "cuanta", "cuantos", "cuantas",
}


# palabras >=6 letras de la pregunta, sin acentos ni términos omnipresentes (para la aspiradora)
def _extraer_keywords_pregunta(question: str) -> List[str]:
    q_clean = re.sub(r'[^a-z0-9\s]', '', strip_accents(question.lower()))
    return [w for w in re.findall(r'\w{6,}', q_clean) if w not in _STOPWORDS_TEMATICAS and not w.isdigit()]


# candidatas para la búsqueda literal: palabras de >=4 letras y siglas en
# mayúsculas (OPE, ZBE, LGTBI), con la grafía original
def _extraer_keywords_literales(question: str) -> List[str]:
    out, vistos = [], set()
    for w in re.findall(r'[^\W\d_]+', question, flags=re.UNICODE):
        es_sigla = len(w) >= 2 and w.isupper()
        if len(w) < 4 and not es_sigla:
            continue
        norm = strip_accents(w.lower())
        if norm in _STOPWORDS_TEMATICAS or norm in _STOPWORDS_ENUNCIADO or norm in vistos:
            continue
        vistos.add(norm)
        out.append(w)
    return out


_VOCAL_CLASE = {"a": "[aá]", "e": "[eé]", "i": "[ií]", "o": "[oó]", "u": "[uúü]"}


# regex para `$regex` de ChromaDB: palabra (o frase) completa, sin distinguir
# mayúsculas ni tildes, en singular o plural; en una frase, espacios flexibles
# y plural solo en la última palabra
def _regex_literal(keyword: str) -> str:
    *inicio, base = strip_accents(keyword.lower()).split()
    formas = {base}
    if base.endswith("es") and len(base) >= 6:
        formas.update({base[:-1], base[:-2]})
    elif base.endswith("s") and len(base) >= 5:
        formas.add(base[:-1])
    else:
        formas.update({base + "s", base + "es"})

    def clases(palabra):
        return "".join(_VOCAL_CLASE.get(c, re.escape(c)) for c in palabra)

    prefijo = "".join(clases(w) + r"\s+" for w in inicio)
    alts = [clases(f) for f in sorted(formas, key=len, reverse=True)]
    return r"(?i)\b" + prefijo + "(" + "|".join(alts) + r")\b"


_CONECTORES = r"(?:de|del|la|las|los|el|y|en)"


_FRASE_PROPIA_RE = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+(?:" + _CONECTORES + r"\s+)*[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)+")


# nombres propios de varias palabras ("Zona de Bajas Emisiones", "San Mamés"):
# se buscan como frase, no como palabras sueltas
def _frases_propias(question: str) -> List[str]:
    out = []
    for m in _FRASE_PROPIA_RE.finditer(question):
        palabras = [strip_accents(w.lower()) for w in m.group(0).split()
                    if not re.fullmatch(_CONECTORES, w.lower())]
        if any(w not in _STOPWORDS_TEMATICAS and w not in _STOPWORDS_ENUNCIADO for w in palabras):
            out.append(m.group(0))
    return out


# "la última vez", "lo más reciente": la recuperación prima lo más nuevo
def _pide_reciente(question: str) -> bool:
    return bool(re.search(r"\búltim[ao]s?\b|\bultim[ao]s?\b|más reciente|mas reciente|recientemente",
                          question.lower()))


# grupo político mencionado en la pregunta; el primero que coincide gana,
# por eso van de más a menos específico
_PARTIDOS_PREGUNTA = [
    (r'eh\s*bildu|euskal\s+herria\s+bildu|herri\s+batasuna|\bhb\b', "EH BILDU"),
    (r'elkarrekin|bilbao\s+en\s+com[uú]n|\bpodemos\b', "ELKARREKIN BILBAO"),
    (r'\bgoazen\b', "GOAZEN BILBAO"),
    (r'pse[\s\-]ee|\bsocialistas?\s+vascos?\b|\bpartido\s+socialista\b|\bpse\b', "PSE-EE"),
    (r'eaj[\s\-]pnv|\bpnv\b|\bnacionalistas?\s+vascos?\b', "EAJ-PNV"),
    (r'\bpartido\s+popular\b|\bgrupo\s+(?:municipal\s+)?pp\b|\bel\s+pp\b|\bdel\s+pp\b|\bpopulares\b', "PP"),
    (r'udalberri', "UDALBERRI"),
    (r'\bciudadanos\b', "CIUDADANOS"),
    (r'ezker\s+batua|izquierda\s+unida', "EZKER BATUA-IU"),
    (r'\baralar\b', "ARALAR"),
    (r'\bvox\b', "VOX"),
    (r'equipo\s+de\s+gobierno|gobierno\s+municipal', "EQUIPO DE GOBIERNO"),
    (r'grupo\s+mixto', "GRUPO MIXTO"),
]


# =============================================================================
# Taxonomía de temas (canal temático)
# =============================================================================

# Taxonomía de temas del grafo (SKOS) para el canal temático: qué etiquetas
# de la pregunta corresponden a cada tema de primer nivel.

_TEMA_ROLLUP_CACHE: Optional[Dict[str, Any]] = None

_TEMA_ROLLUP_LOCK = threading.Lock()


def _load_tema_rollup() -> Dict[str, Any]:
    global _TEMA_ROLLUP_CACHE
    if _TEMA_ROLLUP_CACHE is not None:
        return _TEMA_ROLLUP_CACHE
    # el frontend atiende cada pregunta en un hilo distinto
    with _TEMA_ROLLUP_LOCK:
        if _TEMA_ROLLUP_CACHE is not None:
            return _TEMA_ROLLUP_CACHE
        result = _build_tema_rollup()
        if result is not None:
            # un fallo no se cachea, para reintentarlo en la siguiente pregunta
            _TEMA_ROLLUP_CACHE = result
            return result
        return {"detect": [], "filter_values": {}}


# construye el roll-up desde la taxonomía SKOS; None si falla
def _build_tema_rollup() -> Optional[Dict[str, Any]]:
    try:
        from rdflib import Graph, Namespace, RDF
        from rdflib.namespace import SKOS
        g = Graph()
        BO = Namespace("http://bilbao.tfg/ontology#")
        g.parse(ONTOLOGIA, format="turtle")
        g.parse(TEMAS_SKOS, format="turtle")

        def es_labels(concept) -> List[str]:
            # en cualquier idioma, igual que canon_theme_map() en build_rdf.py
            labs = [g.value(concept, SKOS.prefLabel)]
            labs += list(g.objects(concept, SKOS.altLabel))
            return [str(l) for l in labs if l]

        toplevel = {}  # uri -> prefLabel
        for c, _, _ in g.triples((None, RDF.type, BO.Tema)):
            if not list(g.triples((c, SKOS.broader, None))):
                lbl = g.value(c, SKOS.prefLabel)
                if lbl:
                    toplevel[c] = str(lbl)

        detect: List[Tuple[str, str]] = []
        filter_values: Dict[str, set] = {pref: {pref} for pref in toplevel.values()}
        for c, pref in toplevel.items():
            for lbl in es_labels(c):
                detect.append((strip_accents(lbl.lower()), pref))

        for c, _, _ in g.triples((None, RDF.type, BO.Tema)):
            parent = g.value(c, SKOS.broader)
            if parent is None or parent not in toplevel:
                continue
            parent_pref = toplevel[parent]
            sub_pref = g.value(c, SKOS.prefLabel)
            if sub_pref:
                filter_values[parent_pref].add(str(sub_pref))
            for lbl in es_labels(c):
                detect.append((strip_accents(lbl.lower()), parent_pref))

        # slug de nivel 1 (br:t_medioambiente -> "medioambiente") para filtrar
        # por la bandera tf_<slug> que escribe enrich_vector_metadata.py
        parent_slug = {}
        for uri, pref in toplevel.items():
            local = str(uri).rsplit("/", 1)[-1].split("#")[-1]
            parent_slug[pref] = local[2:] if local.startswith("t_") else local

        # más largas primero, para que "medio ambiente" gane a un término más corto
        detect.sort(key=lambda x: len(x[0]), reverse=True)
        return {
            "detect": detect,
            "filter_values": {p: sorted(v) for p, v in filter_values.items()},
            "parent_slug": parent_slug,
        }
    except Exception as e:
        print(f"[!] No se pudo cargar la taxonomía de temas: {type(e).__name__}: {e}", flush=True)
        return None


# =============================================================================
# 2 y 3. Canales de búsqueda, filtro de grupo y reordenación
# =============================================================================

# distancia máxima para aceptar un resultado de la búsqueda semántica
SIMILARITY_DISTANCE_MAX = 1.4


MULTIQUERY_PROMPT = "Genera 3 variantes de: '{question}' centradas en el sujeto principal. Una por línea:"


# De los fragmentos que coinciden, k repartidos a lo largo del tiempo (uno por
# debate) o, si se pide lo más reciente, los k más nuevos. Antes se tomaban los
# k primeros en orden de inserción, que para cualquier palabra frecuente eran
# siempre las actas de 2007-2008.
def _elegir_por_fecha(ids: List[str], metas: List[dict], k: int, reciente: bool) -> List[str]:
    por_debate: "OrderedDict[tuple, str]" = OrderedDict()
    for cid, m in zip(ids, metas):
        por_debate.setdefault((m.get("date", ""), m.get("topic", "")), cid)
    claves = sorted(por_debate, key=lambda c: clave_fecha(c[0]))
    if reciente:
        claves = claves[::-1][:k]
    elif len(claves) > k:
        paso = (len(claves) - 1) / (k - 1) if k > 1 else 0
        claves = [claves[round(i * paso)] for i in range(k)]
    return [por_debate[c] for c in claves]


# métodos de RAGPipeline (vectorial/pipeline.py) para recuperar candidatos
class Recuperacion:

    # fechas de todos los plenos, a partir de los nombres de los PDF (se calcula una vez)
    def _fechas_conocidas(self) -> List[str]:
        if self._known_dates is None:
            dates = []
            for _, _, files in os.walk(DATA_PATH):
                for file in files:
                    if file.endswith(".pdf"):
                        d = _fecha_de_fichero(file)
                        if d:
                            dates.append(d)
            self._known_dates = list(set(dates))
        return self._known_dates

    # filtro de fechas: fecha exacta DD-MM-YYYY, o año con mes opcional
    def _get_temporal_filter(self, question: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        known_dates = self._fechas_conocidas()

        exact_match = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', question)
        if exact_match:
            exact_date = exact_match.group(0)
            if exact_date in known_dates:
                print(f"[*] Filtro fecha exacta: {exact_date}")
                return {"date": {"$eq": exact_date}}, [exact_date]
            # si ese día no hubo pleno, se usa el mes
            target_pattern = f"{exact_match.group(2)}-{exact_match.group(3)}"
            valid_dates = [d for d in known_dates if target_pattern in d]
            if valid_dates:
                print(f"[*] Filtro Temporal (mes-año): {target_pattern} ({len(valid_dates)} actas)")
                return {"date": {"$in": valid_dates}}, valid_dates
            return None, []

        year_match = re.search(r'(20\d{2})', question)
        if not year_match:
            return None, []
        q_low = question.lower()
        # \b: sin él "mayoría" activaba el filtro de mayo
        month_found = next((m for m in MESES_ES if re.search(rf'\b{m}\b', q_low)), None)
        estacion = next((ms for nombre, ms in ESTACIONES.items() if re.search(rf'\b{nombre}\b', q_low)), None)
        target_year = year_match.group(1)
        if month_found:
            target_pattern = f"{MESES_ES[month_found]:02d}-{target_year}"
            valid_dates = [d for d in known_dates if target_pattern in d]
        elif estacion:
            target_pattern = f"{target_year} (meses {estacion})"
            valid_dates = [d for d in known_dates
                           if d.endswith(target_year) and d[3:5].isdigit() and int(d[3:5]) in estacion]
        else:
            target_pattern = target_year
            valid_dates = [d for d in known_dates if target_pattern in d]
            # "los presupuestos de 2022" se aprueban a finales de 2021: sin los plenos de
            # octubre a diciembre del año anterior, el filtro dejaba fuera la votación (octava
            # evaluación: todo lo recuperado era de 2022 y nunca se encontró la de noviembre de 2021)
            if re.search(r'presupuest|ejercicio', q_low):
                previo = str(int(target_year) - 1)
                valid_dates += [d for d in known_dates
                                if d.endswith(previo) and d[3:5].isdigit() and int(d[3:5]) >= 10]
                target_pattern += f" + oct-dic {previo}"
        if valid_dates:
            print(f"[*] Filtro Temporal: {target_pattern} ({len(valid_dates)} actas detectadas)")
            return {"date": {"$in": valid_dates}}, valid_dates
        return None, []

    def _detect_party_in_question(self, question: str) -> Optional[str]:
        q = question.lower()
        return next((grupo for patron, grupo in _PARTIDOS_PREGUNTA if re.search(patron, q)), None)

    # MultiQuery: variantes de la pregunta para ampliar el recall (más la original)
    def _query_variations(self, question: str) -> List[str]:
        try:
            vars_txt = self.invoke_llm(MULTIQUERY_PROMPT.format(question=question)).content
            variations = [v.strip() for v in vars_txt.split('\n') if v.strip()] + [question]
        except Exception as e:
            print(f"[!] MultiQuery falló, usando pregunta original: {e}", flush=True)
            variations = [question]
        return variations[:6]

    # canal semántico: búsqueda por embeddings de cada variante
    def _initial_search(self, variations: List[str], valid_dates: list,
                        exact_date: bool, k: int) -> List[Document]:
        docs: List[Document] = []
        for v in variations:
            targets = valid_dates if valid_dates else [None]
            for d_val in targets:
                flt = {"date": {"$eq": d_val}} if d_val else None
                try:
                    if exact_date:
                        # un pleno concreto: todo el pleno, sin umbral de distancia
                        docs.extend(self.vector_store.similarity_search(v, k=k, filter=flt))
                    else:
                        res = self.vector_store.similarity_search_with_score(v, k=k, filter=flt)
                        docs.extend([d for d, s in res if s < SIMILARITY_DISTANCE_MAX])
                except Exception as e:
                    print(f"[!] ChromaDB búsqueda fallida: {type(e).__name__}: {e}", flush=True)
        return docs

    # canal literal: fragmentos que contienen textualmente las palabras más raras
    # de la pregunta (o sus nombres propios de varias palabras, como frase)
    def _keyword_search(self, question: str, valid_dates: list, k_per_kw: int = 15,
                        count_cap: int = 3000) -> List[Document]:
        frases = _frases_propias(question)
        candidatas = frases + _extraer_keywords_literales(question)
        if not candidatas:
            return []
        reciente = _pide_reciente(question)

        # una sola consulta por keyword con $in, no una por cada fecha
        if len(valid_dates) == 1:
            where = {"date": {"$eq": valid_dates[0]}}
        elif valid_dates:
            where = {"date": {"$in": valid_dates}}
        else:
            where = None

        # se buscan las 3 keywords más raras del corpus (menos fragmentos),
        # con desempate fijo; se consultan en paralelo porque cada $regex
        # recorre toda la colección
        def coincidencias(kw: str):
            try:
                res = self.vector_store.get(
                    where=where, where_document={"$regex": _regex_literal(kw)},
                    limit=count_cap, include=["metadatas"])
                return res["ids"], res["metadatas"]
            except Exception as e:
                print(f"[!] Búsqueda literal fallida para '{kw}': {type(e).__name__}: {e}", flush=True)
                return [], []

        with ThreadPoolExecutor(max_workers=min(8, len(candidatas))) as ex:
            hits = dict(zip(candidatas, ex.map(coincidencias, candidatas)))
        # si un nombre propio aparece como frase, sus palabras sueltas sobran
        sueltas_de_frase = {strip_accents(w.lower()) for f in frases if hits[f][0] for w in f.split()}
        candidatas = [kw for kw in candidatas
                      if kw in frases or strip_accents(kw.lower()) not in sueltas_de_frase]
        keywords = sorted((kw for kw in candidatas if hits[kw][0]),
                          key=lambda kw: (len(hits[kw][0]), -len(kw), kw.lower()))[:3]

        # resultados por keyword y luego intercalados, para que una keyword
        # frecuente no ocupe todo el cupo que se pasa a Cohere
        by_kw: Dict[str, List[Document]] = {kw: [] for kw in keywords}

        def buscar(kw: str) -> List[Document]:
            elegidos = _elegir_por_fecha(*hits[kw], k_per_kw, reciente)
            try:
                res = self.vector_store.get(ids=elegidos, include=["metadatas", "documents"])
            except Exception as e:
                print(f"[!] Búsqueda literal fallida para '{kw}': {type(e).__name__}: {e}", flush=True)
                return []
            orden = {cid: i for i, cid in enumerate(elegidos)}
            docs_kw = []
            for _, content, meta in sorted(zip(res["ids"], res["documents"], res["metadatas"]),
                                             key=lambda x: orden[x[0]]):
                m = dict(meta)
                m["_kw"] = kw  # marca de canal literal (ver _select_final_docs)
                docs_kw.append(Document(page_content=content, metadata=m))
            return docs_kw

        if keywords:
            with ThreadPoolExecutor(max_workers=len(keywords)) as ex:
                for kw, docs_kw in zip(keywords, ex.map(buscar, keywords)):
                    by_kw[kw] = docs_kw

        # round robin entre keywords
        docs: List[Document] = []
        i = 0
        while any(i < len(by_kw[kw]) for kw in keywords):
            for kw in keywords:
                if i < len(by_kw[kw]):
                    docs.append(by_kw[kw][i])
            i += 1
        return docs

    # canal de proposiciones: el resumen de cada proposición se busca por significado (encuentra
    # "cierre de sucursales" cuando se pregunta por "oficinas bancarias") y de las mejores se
    # traen los primeros fragmentos de su debate, en orden de relevancia. Los fragmentos
    # compiten después con los demás candidatos en el reranker.
    def _indice_proposiciones(self):
        if not self._indice_props_cargado:
            self._indice_props_cargado = True
            if os.path.isdir(CHROMA_PROPOSICIONES_PATH):
                try:
                    self._indice_props = Chroma(collection_name="proposiciones", embedding_function=self.embeddings,
                                                persist_directory=CHROMA_PROPOSICIONES_PATH)
                except Exception as e:
                    print(f"[!] Índice de proposiciones no disponible: {type(e).__name__}: {e}", flush=True)
        return self._indice_props

    def _propuestas_search(self, variations: List[str], valid_dates: list, k: int = 10,
                           max_props: int = 12, por_prop: int = 2) -> List[Document]:
        store = self._indice_proposiciones()
        if store is None:
            return []
        flt = None
        if len(valid_dates) == 1:
            flt = {"date": {"$eq": valid_dates[0]}}
        elif valid_dates:
            flt = {"date": {"$in": valid_dates}}
        rondas = []
        for v in variations[:4]:
            try:
                res = store.similarity_search_with_score(v, k=k, filter=flt)
            except Exception as e:
                print(f"[!] Búsqueda de proposiciones fallida: {type(e).__name__}: {e}", flush=True)
                continue
            rondas.append([d.metadata["prop_id"] for d, s in res if s < SIMILARITY_DISTANCE_MAX])
        # una variante tras otra: la mejor de cada una antes que la segunda de ninguna
        ids = []
        for i in range(k):
            for r in rondas:
                if i < len(r) and r[i] not in ids:
                    ids.append(r[i])
        ids = ids[:max_props]
        if not ids:
            return []
        try:
            r = self.vector_store.get(where={"prop_id": {"$in": ids}}, include=["documents", "metadatas"])
        except Exception as e:
            print(f"[!] Fragmentos de proposiciones no disponibles: {type(e).__name__}: {e}", flush=True)
            return []
        por_id = {}
        for doc, meta in zip(r["documents"], r["metadatas"]):
            por_id.setdefault(meta.get("prop_id"), []).append((meta.get("chunk_index", 0), doc, meta))
        out = []
        for pid in ids:
            for _, doc, meta in sorted(por_id.get(pid, []), key=lambda x: x[0])[:por_prop]:
                out.append(Document(page_content=doc, metadata=dict(meta)))
        return out

    # canal temático: proposiciones clasificadas con el tema de la pregunta,
    # aunque su texto no repita la palabra (ver memoria 1.1)
    def _thematic_search(self, question: str, valid_dates: list,
                         max_proposals: int = 40, pool_limit: int = 20000) -> List[Document]:
        rollup = _load_tema_rollup()
        if not rollup["detect"]:
            return []
        q_norm = strip_accents(question.lower())
        matched_label = None
        for label_norm, label_display in rollup["detect"]:
            if re.search(rf'\b{re.escape(label_norm)}\b', q_norm):
                matched_label = label_display
                break
        if not matched_label:
            return []

        # la bandera tf_<slug> cubre el tema principal y los secundarios; si el
        # chunk no tiene banderas se filtra por tema_principal y sus subtemas
        slug = rollup.get("parent_slug", {}).get(matched_label)
        filter_values = rollup["filter_values"].get(matched_label, [matched_label])
        tema_clause = (
            {"$or": [{f"tf_{slug}": True}, {"tema_principal": {"$in": filter_values}}]}
            if slug else {"tema_principal": {"$in": filter_values}}
        )
        clauses = [tema_clause]
        if len(valid_dates) == 1:
            clauses.append({"date": {"$eq": valid_dates[0]}})
        elif valid_dates:
            clauses.append({"date": {"$in": valid_dates}})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}

        # primero solo metadatos (un tema grande supera los 10.000 chunks); con
        # un pool_limit bajo se perdían los años más recientes de los temas grandes
        try:
            res = self.vector_store.get(where=where, limit=pool_limit, include=["metadatas"])
        except Exception as e:
            print(f"[!] Búsqueda temática fallida: {type(e).__name__}: {e}", flush=True)
            return []

        # muestreo repartido en el tiempo: muchas proposiciones antes que pocas en detalle
        by_prop: Dict[str, List[Tuple[str, int]]] = {}
        prop_date: Dict[str, str] = {}
        for cid, meta in zip(res["ids"], res["metadatas"]):
            pid = meta.get("prop_id")
            if not pid:
                continue
            by_prop.setdefault(pid, []).append((cid, meta.get("chunk_index", 0)))
            prop_date.setdefault(pid, meta.get("date", ""))

        pids = sorted(by_prop, key=lambda p: clave_fecha(prop_date.get(p, "")))
        if len(pids) > max_proposals:
            if _pide_reciente(question):
                pids = pids[-max_proposals:]
            else:
                # paso (n-1)/(k-1) para que entren la primera y la última
                step = (len(pids) - 1) / (max_proposals - 1)
                pids = [pids[round(i * step)] for i in range(max_proposals)]

        print(f"[*] Tema detectado en la pregunta: '{matched_label}' "
              f"-> {len(by_prop)} proposiciones distintas (muestra de {len(pids)})")
        if not pids:
            return []

        # de cada proposición, el chunk más largo del cuerpo (sin el primero ni
        # los dos últimos, que suelen ser el orden del día y el recuento de votos)
        ids_cuerpo = []
        for p in pids:
            orden = [cid for cid, _ in sorted(by_prop[p], key=lambda ci: ci[1])]
            ids_cuerpo += (orden[1:-2] or orden)[:8]
        res2 = self.vector_store.get(ids=ids_cuerpo, include=["metadatas", "documents"])
        mejor: Dict[str, Tuple[str, dict, int]] = {}
        for c, m in zip(res2["documents"], res2["metadatas"]):
            pid = m.get("prop_id")
            if not pid:
                continue
            cuerpo = re.sub(r"votos\s+emitidos.*", "", c, flags=re.I | re.S)
            n = len(cuerpo.strip())
            if pid not in mejor or n > mejor[pid][2]:
                mejor[pid] = (c, m, n)
        return [Document(page_content=c, metadata=m) for c, m, _ in mejor.values()]

    @staticmethod
    def _doc_key(d: Document) -> tuple:
        return (d.metadata.get("source", ""), d.metadata.get("chunk_index", d.page_content[:50]))

    # elimina duplicados por (source, chunk_index) conservando el orden
    def _dedup_docs(self, docs: List[Document], limit: Optional[int] = None) -> List[Document]:
        seen, out = set(), []
        for d in docs:
            key = self._doc_key(d)
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out[:limit] if limit else out

    # si la pregunta menciona un grupo, se queda con sus chunks (si quedan al menos min_docs)
    def _apply_party_filter(self, docs: List[Document], question: str,
                            min_docs: int = 3) -> List[Document]:
        target_party = self._detect_party_in_question(question)
        if not target_party:
            return docs

        def es_proponente(d: Document) -> bool:
            # grupo_proponente viene de la indexación; si falta, se deduce del topic
            grupo_meta = d.metadata.get("grupo_proponente")
            if grupo_meta:
                return grupo_meta == target_party
            return _extrae_grupo_partido(d.metadata.get("topic", "")) == target_party

        party_filtered = [
            d for d in docs
            if target_party.lower() in d.metadata.get("party", "").lower()
            or es_proponente(d)
        ]
        if len(party_filtered) >= min_docs:
            print(f"[*] Filtro de grupo '{target_party}': {len(party_filtered)} docs")
            return party_filtered
        return docs

    def _rerank_with_cohere(self, docs: List[Document], question: str, top_n: int = 30) -> List[Document]:
        # 100 candidatos (una sola petición de Cohere): con el canal de proposiciones hay más
        return _cohere_rerank(docs, question, top_n, max_candidates=100)

    # MultiQuery -> canales literal, temático y semántico -> filtro de grupo -> Cohere
    def _retrieve_and_rank(self, question: str, k: int) -> Tuple[List[Document], bool]:
        variations = self._query_variations(question)
        _, valid_dates = self._get_temporal_filter(question)
        exact_date = len(valid_dates) == 1
        docs = self._initial_search(variations, valid_dates, exact_date, k)
        # máx. 22 por canal (24 el de proposiciones): el reranker solo reordena los primeros candidatos
        docs = self._dedup_docs(
            self._keyword_search(question, valid_dates)[:22]
            + self._thematic_search(question, valid_dates)[:22]
            + self._propuestas_search(variations, valid_dates)
            + docs
        )
        docs = self._apply_party_filter(docs, question)
        if COHERE_API_KEY:
            docs = self._rerank_with_cohere(docs, question, top_n=50)
        return docs, exact_date


# =============================================================================
# 4. Selección final
# =============================================================================

# Selección final: de los candidatos reordenados se eligen los debates que
# responden a la pregunta y se recuperan completos ("aspiradora"); los mejores
# fragmentos de la reordenación y del canal literal se conservan siempre.

# puntúa cada tema (topic) distinto de los candidatos: 20 puntos por keyword en
# el título, 2 por keyword solo en el contenido (raíz de 6 letras)
def _puntuar_temas(initial_docs: List[Document], question: str) -> List[dict]:
    q_keywords = _extraer_keywords_pregunta(question)
    target_topics, seen = [], set()
    for doc in initial_docs:
        topic = doc.metadata.get("topic")
        source = doc.metadata.get("source")
        if not topic or topic in ["General", "General / Introducción"]:
            continue
        source_bn = os.path.basename(source)
        if (source_bn, topic) in seen:
            continue
        norm_topic = normalizar(topic)
        norm_content = normalizar(doc.page_content)
        score = sum(
            20 if kw[:6] in norm_topic else 2 if kw[:6] in norm_content else 0
            for kw in q_keywords
        )
        seen.add((source_bn, topic))
        target_topics.append({"topic": topic, "source": source, "score": score, "basename": source_bn})
    return target_topics


# se queda con los temas relevantes (hasta 15), repartidos entre años
def _elegir_temas(target_topics: List[dict], question: str) -> List[dict]:
    # los presupuestos mencionan de pasada cualquier tema: se penalizan salvo
    # que la pregunta trate de ellos
    q_clean = normalizar(question)
    if not re.search(r'presupuest|modificaci[oó]n\s+presup|ordenanza', q_clean, re.I):
        for t in target_topics:
            if re.search(r'presupuest|modificac|ordenanza', normalizar(t['topic'])):
                t['score'] *= 0.2

    # si algún tema tiene una keyword en el título, fuera los que solo la
    # mencionan en el contenido
    title_matched = [t for t in target_topics if t['score'] >= 20]
    if title_matched:
        target_topics = title_matched

    # umbral absoluto (4 = keyword en al menos dos contenidos) y relativo (40 % del mejor)
    MIN_ABSOLUTE_SCORE = 4
    target_topics = sorted(
        [t for t in target_topics if t['score'] >= MIN_ABSOLUTE_SCORE],
        key=lambda x: x['score'], reverse=True
    )
    if not target_topics:
        return []
    max_s = target_topics[0]['score']
    if max_s > 0:
        target_topics = [t for t in target_topics if t['score'] >= max_s * 0.4]

    # diversidad temporal: un tema por año, y un segundo del mismo año solo si
    # es casi tan bueno (90 %)
    seen_years = {}
    diverse_topics = []
    for t in target_topics:
        year_match = re.search(r'(\d{4})', t.get("basename", ""))
        year = year_match.group(1) if year_match else ""
        if year not in seen_years:
            seen_years[year] = t['score']
            diverse_topics.append(t)
        elif t['score'] >= seen_years[year] * 0.9:
            diverse_topics.append(t)
    return diverse_topics[:15]


# prefijo con el que empiezan los topics de un mismo punto ("12." o el título)
def _prefijo_tema(topic: str) -> str:
    num = topic.split('.')[0].strip()
    return (num + ".") if num.isdigit() else topic[:80]


# métodos de RAGPipeline (vectorial/pipeline.py) para elegir los fragmentos finales
class Seleccion:

    # todos los chunks del debate de un tema, saltando su entrada en el orden del día
    def _recuperar_debate(self, t_info: dict, final_seen: set) -> List[Document]:
        t_name, src = t_info['topic'], t_info['source']
        t_prefix = f"ASUNTO: {t_name[:40]}"
        docs = []
        try:
            res = self.vector_store.get(where={"source": src})
            pairs = sorted(zip(res['metadatas'], res['documents']), key=lambda x: x[0].get('chunk_index', 0))
            topic_indices = [i for i, (m, d) in enumerate(pairs)
                             if (m.get('topic') == t_name or d.lstrip().startswith(t_prefix))]
            if not topic_indices:
                return docs

            # El tema aparece dos veces: en el orden del día (chunks cortos, solo
            # títulos) y en el debate (chunks largos). Se agrupan los índices en
            # rachas contiguas y se elige la de más texto.
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

            t_topic_number = _prefijo_tema(t_name)
            seen_content = set()
            for i in best_run:
                m, d = pairs[i]
                topic_val = m.get('topic', '')
                if topic_val and not topic_val.startswith(t_topic_number):
                    continue
                # los últimos 150 caracteres: los primeros son siempre "ASUNTO: ..."
                content_key = d[-150:].strip() if len(d) > 150 else d.strip()
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                key = (src, m.get('chunk_index', d[:50]))
                if key not in final_seen:
                    final_seen.add(key)
                    docs.append(Document(page_content=d, metadata=m))
        except Exception as e:
            print(f"[!] Error en expansión: {e}")
        return docs

    # aspiradora: de los fragmentos sueltos a los debates completos de los temas elegidos
    def _expand_context_by_topic(self, initial_docs: List[Document],
                                 question: str) -> Tuple[List[Document], list]:
        target_topics = _elegir_temas(_puntuar_temas(initial_docs, question), question)
        if not target_topics:
            return initial_docs[:15], []

        print(f"[*] Temas priorizados: {[t['topic'][:50] for t in target_topics]}")
        docs, final_seen = [], set()
        for t_info in target_topics:
            docs += self._recuperar_debate(t_info, final_seen)
        return docs, target_topics

    # quita los docs cuyo tema no está entre los elegidos (si no los quita todos)
    def _filter_by_target_topics(self, docs: List[Document], target_topics: list) -> List[Document]:
        if not target_topics:
            return docs
        valid_prefixes = [_prefijo_tema(t['topic']) for t in target_topics]
        filtered = [d for d in docs if any(d.metadata.get('topic', '').startswith(p) for p in valid_prefixes)]
        return filtered if filtered else docs

    # pleno concreto: sus fragmentos tal cual. Pregunta general: aspiradora + rescate (ver memoria 1.5)
    def _select_final_docs(self, candidates: List[Document], question: str,
                           exact_date: bool) -> List[Document]:
        if exact_date:
            return sorted(self._dedup_docs(candidates, limit=40), key=clave_fecha_doc)

        docs, target_topics = self._expand_context_by_topic(candidates, question)
        docs = self._filter_by_target_topics(docs, target_topics)
        # rescate: los 8 mejores de Cohere y los del canal literal se
        # conservan siempre, aunque la aspiradora los haya descartado
        ranked = [d for d in candidates if "_rerank_score" in d.metadata]
        kw_hits = [d for d in candidates if d.metadata.get("_kw")][:10]
        lead = self._dedup_docs(ranked[:8] + kw_hits)
        for d in lead:
            d.metadata["_lead"] = True       # _format_context los coloca primero y no los descarta
        lead_keys = {self._doc_key(d) for d in lead}
        # el resto, por relevancia del tema: si el contexto no cabe, se descartan los últimos
        # (_format_context los muestra después en orden cronológico)
        tail = [d for d in self._dedup_docs(docs) if self._doc_key(d) not in lead_keys]
        # los docs ya son del tema correcto, el filtro de grupo puede ser estricto
        return self._apply_party_filter(lead + tail, question, min_docs=1)
