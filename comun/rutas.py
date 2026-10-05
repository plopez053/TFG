"""Rutas de los datos del proyecto: actas en PDF, índices vectoriales y grafo."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# actas dentro del proyecto (portátil) o en la carpeta de al lado (sobremesa)
_ACTAS_DENTRO = os.path.join(BASE_DIR, "actas")
_ACTAS_FUERA = os.path.join(os.path.dirname(BASE_DIR), "actas")
DATA_PATH = _ACTAS_DENTRO if os.path.isdir(_ACTAS_DENTRO) else _ACTAS_FUERA

# índice de fragmentos del RAG vectorial
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
# resumen por proposición (scripts/indexar_proposiciones.py); si no existe, el canal no se usa
CHROMA_PROPOSICIONES_PATH = os.path.join(BASE_DIR, "chroma_db_proposiciones")

# grafo de conocimiento: ontología, taxonomía de temas, grafo final y datos intermedios
GRAFO_DATOS = os.path.join(BASE_DIR, "datos", "grafo")
GRAFO_TTL = os.path.join(GRAFO_DATOS, "bilbao_reasoned.ttl")
ONTOLOGIA = os.path.join(GRAFO_DATOS, "ontology.ttl")
TEMAS_SKOS = os.path.join(GRAFO_DATOS, "themes_skos.ttl")
FORMAS_SHACL = os.path.join(GRAFO_DATOS, "shapes.ttl")
# cachés del grafo (núcleo de las preguntas, texto de las actas, embeddings de plantillas)
CACHE_GRAFO = os.path.join(GRAFO_DATOS, "cache")


# fichero de datos del grafo por su nombre ("proposals.jsonl", "votaciones.jsonl"...)
def dato_grafo(nombre: str) -> str:
    return os.path.join(GRAFO_DATOS, nombre)
