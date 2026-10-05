"""
Qué pregunta la pregunta, antes de escribir ninguna consulta:

  1. análisis determinista: temas, grupos, años, resultado y lo que pide
  2. núcleo del asunto, marcado por un LLM y comprobado contra la pregunta
  3. asunto: las proposiciones del grafo que lo contienen y la pista para el LLM
"""
import json
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from comun.rutas import CACHE_GRAFO, TEMAS_SKOS as _THEMES_TTL
from grafo.consulta.recursos import _ejecutar, _llm_invoke, _PREFIXES


# =============================================================================
# 1. Análisis determinista de la pregunta
# =============================================================================

# Análisis determinista de una pregunta para GraphRAG: qué temas, grupos, años y
# resultado menciona, y si pide algo especial (lo más reciente, unanimidad,
# oposición, un mes o estación, comparar años) o algo que el grafo no guarda
# (importes, asistencia).
#
# Sirve para comprobar que una consulta SPARQL cubre lo que pregunta la pregunta
# (`cobertura`): la causa principal de respuestas falsas en la evaluación
# independiente eran consultas que se ejecutaban bien pero ignoraban el tema, el
# grupo o parte del rango de años.

ANIO_MIN, ANIO_MAX = 2007, 2026
LEGISLATURA_ACTUAL = 2023
# pleno de constitución de cada corporación (actas/): una legislatura va de
# una constitución a la siguiente, no de enero a diciembre
CONSTITUCION = {2007: "2007-06-16", 2011: "2011-06-11", 2015: "2015-06-13", 2019: "2019-06-15", 2023: "2023-06-17"}
LEGISLATURA_INICIO = CONSTITUCION[LEGISLATURA_ACTUAL]


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    # variantes ortográficas de una misma palabra: las actas escriben "quioscos" y la pregunta
    # "kioscos" (novena evaluación: el grafo respondía que no constaba, y sí había una proposición)
    s = re.sub(r"\bquiosc", "kiosc", s)
    return re.sub(r"\s+", " ", s)


# --- temas: etiquetas de themes_skos.ttl -> slug (cualquier nivel) ---

_TEMAS_CACHE: Optional[List[Tuple[re.Pattern, str, str]]] = None
_SINONIMOS: set = set()   # (slug, etiqueta) que son altLabel (sinónimos) de un tema


# cada palabra admite plural ("carril bici" casa "carriles bici")
def _patron_etiqueta(etiqueta: str) -> re.Pattern:
    partes = []
    for w in _norm(etiqueta).split():
        w = re.escape(w)
        partes.append(w + r"(?:e?s)?" if len(w) > 2 else w)
    return re.compile(r"\b" + r"\s+".join(partes) + r"\b")


def _temas():
    global _TEMAS_CACHE
    if _TEMAS_CACHE is None:
        import rdflib
        from rdflib.namespace import SKOS
        g = rdflib.Graph()
        g.parse(_THEMES_TTL, format="turtle")
        bo = rdflib.Namespace("http://bilbao.tfg/ontology#")
        entradas = []
        for t in g.subjects(rdflib.RDF.type, bo.Tema):
            slug = str(t).rsplit("/", 1)[-1]
            if slug == "t_otros":
                continue
            for pred in (SKOS.prefLabel, SKOS.altLabel):
                for lab in g.objects(t, pred):
                    entradas.append((_patron_etiqueta(str(lab)), slug, _norm(str(lab))))
                    if pred == SKOS.altLabel:
                        _SINONIMOS.add((slug, _norm(str(lab))))
        # más largas primero: "zona de bajas emisiones" antes que "emisiones"
        entradas.sort(key=lambda e: len(e[2]), reverse=True)
        _TEMAS_CACHE = entradas
    return _TEMAS_CACHE


# --- grupos ---

_GRUPOS = [  # (patrón sobre el texto normalizado, [uris aceptadas], nombre)
    (r"partido popular|\bpp\b|\bpopulares\b", ["grupo_pp"], "PP"),
    (r"\beh[ -]?bildu\b|\bbildu\b", ["grupo_eh_bildu"], "EH Bildu"),
    (r"pse[ -]?ee|\bpse\b|socialistas?|psoe", ["grupo_pse_ee"], "PSE-EE"),
    (r"elkarrekin", ["grupo_elkarrekin_bilbao"], "Elkarrekin Bilbao"),
    (r"\bpodemos\b", ["grupo_elkarrekin_bilbao", "grupo_udalberri"], "Podemos (Elkarrekin/Udalberri)"),
    (r"goazen", ["grupo_goazen_bilbao"], "Goazen Bilbao"),
    (r"udalberri|bilbao en comun", ["grupo_udalberri"], "Udalberri"),
    (r"eaj[ -]?pnv|\bpnv\b|nacionalistas? vascos?", ["grupo_eaj_pnv"], "EAJ-PNV"),
    (r"\bciudadanos\b", ["grupo_ciudadanos"], "Ciudadanos"),
    (r"\bvox\b", ["grupo_vox"], "Vox"),
    (r"\baralar\b", ["grupo_aralar"], "Aralar"),
    (r"ezker batua|izquierda unida", ["grupo_ezker_batua_iu"], "Ezker Batua-IU"),
    (r"equipo de gobierno|gobierno municipal", ["grupo_equipo_de_gobierno"], "Equipo de Gobierno"),
    (r"grupo mixto", ["grupo_grupo_mixto"], "Grupo Mixto"),
]

# Nombres de grupo que también son palabras corrientes: solo cuentan como grupo en su uso de
# grupo ("los populares", "Podemos" que no es el verbo, "Ciudadanos" con mayúscula). El segundo
# valor dice si se comprueba sobre la pregunta tal cual (mayúsculas) o normalizada.
_USO_DE_GRUPO = {
    "PP": (r"partido popular|\bpp\b|\b(?:los|las)\s+populares\b", False),
    "Podemos (Elkarrekin/Udalberri)": (r"\bpodemos\b(?!\s+\w+(?:ar|er|ir)\b)", False),
    "Ciudadanos": (r"\b(?-i:Ciudadanos|Cs)\b", True),
}

# palabra de la etiqueta real de cada grupo, por si la consulta filtra por nombre
_ETIQUETA_GRUPO = {
    "grupo_pp": "pp", "grupo_eh_bildu": "bildu", "grupo_pse_ee": "pse",
    "grupo_elkarrekin_bilbao": "elkarrekin", "grupo_goazen_bilbao": "goazen",
    "grupo_udalberri": "udalberri", "grupo_eaj_pnv": "pnv", "grupo_ciudadanos": "ciudadanos",
    "grupo_vox": "vox", "grupo_ezker_batua_iu": "ezker", "grupo_equipo_de_gobierno": "gobierno",
    "grupo_grupo_mixto": "mixto",
}

_MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
          "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
_ESTACIONES = {"primavera": (3, 4, 5, 6), "verano": (6, 7, 8, 9),
               "otono": (9, 10, 11, 12), "invierno": (12, 1, 2, 3)}


@dataclass
class Analisis:
    temas: List[Tuple[str, str]] = field(default_factory=list)   # (slug, etiqueta encontrada)
    grupos: List[Tuple[List[str], str]] = field(default_factory=list)
    anios: Optional[Tuple[int, int]] = None                        # rango pedido (incluido)
    anio_ahora: bool = False
    resultado: Optional[str] = None                                # aprobada/rechazada/decae/retirada
    reciente: bool = False
    unanimidad: bool = False
    oposicion: bool = False
    # proposiciones vecinales/ciudadanas: las presenta una asociación o un
    # particular (bo:tipoPunto "proposicion_ciudadana"), no un grupo
    ciudadana: bool = False
    # pregunta por un importe en euros (bo:importe de los puntos)
    importe: bool = False
    meses: Tuple[int, ...] = ()
    compara_anios: bool = False
    fuera_de_alcance: Optional[str] = None
    asunto: List[str] = field(default_factory=list)                # palabras de un asunto que no es un tema
    # raíces del asunto que la consulta tiene que exigir todas a la vez; las
    # calcula GraphRAG con la frecuencia real en el grafo (_preparar_asunto)
    asunto_raices: List[str] = field(default_factory=list)
    # alternativas unidas por "o" en la pregunta ("el Athletic o San Mamés"):
    # palabras y raíces por alternativa; dentro de cada una se exigen todas
    asunto_alternativas: List[List[str]] = field(default_factory=list)
    asunto_grupos: List[List[str]] = field(default_factory=list)
    desde_fecha: Optional[str] = None                              # "2023-06-17" para la legislatura actual
    hasta_fecha: Optional[str] = None                              # fin de una legislatura pasada
    fecha: Optional[str] = None                                    # un pleno concreto: "2016-08-15"
    ejercicio: Optional[int] = None                                # "el presupuesto de 2016" -> 2016
    # por alternativa, las palabras imprescindibles del asunto (núcleo del LLM):
    # si no hay ninguna proposición con todas, se relaja a estas
    asunto_clave: List[List[str]] = field(default_factory=list)
    # temas reconocidos por un sinónimo más concreto que el tema ("zona 30" ->
    # peatonalización): el sinónimo también se busca como asunto
    temas_sinonimo: List[Tuple[str, str]] = field(default_factory=list)
    # palabras escritas con mayúscula en la pregunta (sin contar la primera):
    # nombres de lugares o entidades que se exigen aunque sean frecuentes
    nombres_propios: List[str] = field(default_factory=list)
    # "medio ambiente y zonas verdes": temas que son una alternativa entera del
    # asunto y se SUMAN a sus proposiciones (no se cruzan con ellas)
    asunto_temas: List[str] = field(default_factory=list)
    # temas reconocidos que el núcleo del asunto ya incluye ("el barrio de
    # Otxarkoaga" contiene el tema barrios): la consulta directa no los añade
    # como filtro aparte
    temas_absorbidos: List[Tuple[str, str]] = field(default_factory=list)

    # algo que las plantillas sin LLM no saben expresar
    def requiere_llm(self) -> bool:
        rango = self.anios is not None and self.anios[0] != self.anios[1]
        return (self.reciente or self.unanimidad or self.oposicion or bool(self.meses)
                or self.compara_anios or rango or len(self.temas) > 1)


# "sobre X", "relacionado con X"...: X es el asunto de la pregunta. Si no es un
# tema del vocabulario ni un grupo, la consulta tiene que buscarlo (título,
# menciones); si no, se responde sin filtrar por él ("¿cuántas sobre la
# Fórmula 1 aprobó el Pleno en 2018?" -> todas las aprobadas de 2018).
_ASUNTO_RE = re.compile(
    r"\b(?:sobre|acerca de|relacionad[ao]s? con|referentes? a|en materia de|respecto (?:a|de)|en torno a|mencionad[ao]s? a|nombrad[ao]s? a"
    # "¿se ha hablado alguna vez en el Pleno de una plaga...?": hasta cuatro
    # palabras entre el verbo y el "de"
    r"|(?:hablad[oa]|comentad[oa]|debatid[oa]|tratad[oa]|discutid[oa])(?:\s+\w+){0,6}?\s+de"
    # "¿cuántas veces se ha debatido la ampliación del Metro...?", "se trató el ruido..."
    r"|(?:debatid[oa]|tratad[oa]|discutid[oa]|debati[oó]|trat[oó]|discuti[oó])(?:\s+en\s+el\s+pleno)?)"
    r"\s+(?:(?:el|la|los|las|lo|un|una)\s+)?(.+?)"
    # el asunto sigue tras "de" y "hasta" ("la ampliación del Metro hasta
    # Basauri") y acaba en el año, el pleno, el grupo o el verbo
    r"(?=\s+(?:en\s+(?:el\s+|la\s+)?(?:pleno|legislatura|mandato|ano|\d{4})|(?:entre|desde|hasta)\s+(?:el\s+)?\d{4}"
    r"|durante|por|que|se|ha|han|aprob\w*|rechaz\w*|present\w*|vot[oa]\w*|debat\w*|opin\w*)\b|[?.,;:]|$)")
_NO_ASUNTO = {"pleno", "plenos", "bilbao", "ayuntamiento", "proposicion", "proposiciones", "mocion", "mociones",
              "iniciativa", "iniciativas", "tema", "temas", "asunto", "asuntos", "algo", "esto", "eso", "ello",
              "cada", "grupo", "grupos", "municipal", "votacion", "votaciones", "calle", "calles", "barrio",
              "barrios", "zona", "zonas", "plaza", "avenida", "villa", "ciudad", "propuesta", "propuestas",
              "para", "sobre", "como", "entre", "desde", "hasta", "este", "esta", "estos", "estas",
              "algun", "alguna", "alguno", "algunos", "algunas", "vez", "veces", "nunca", "favor", "contra",
              "hubo", "habido", "hecho", "tratado", "dicho", "sido", "estado", "todo", "toda", "todos", "todas",
              "ultima", "ultimo", "ultimas", "ultimos", "primera", "primer", "legislatura", "mandato", "anos",
              "relacionado", "relacionada", "relacionados", "relacionadas", "cosa", "cosas", "cuestion",
              "consta", "constan", "existe", "existen", "debatido", "debatida", "hablado", "comentado",
              "persona", "personas", "gente", "vecinos", "vecinas", "ciudadania", "centro", "municipio",
              # "atención a la infancia": el asunto es la infancia; exigir también "atención" dejaba 6
              # proposiciones de las ~40 y cambiaba el ranking de grupos (octava evaluación)
              "atencion"}


# Palabras del asunto que no cubre ningún tema ni grupo reconocido. Se quitan
# palabra a palabra: en "la peatonalización de la calle Ledesma" el tema
# (peatonalización) está en el vocabulario, pero "Ledesma" no, y la consulta
# tiene que buscarlo (antes se descartaba el asunto entero y se respondió con
# una peatonalización de otra calle).
def _palabras_asunto(frase: str, a: "Analisis") -> List[str]:
    # un número pegado a una palabra va con ella ("formula 1", "bilbobus 38")
    # las palabras cortas solo si son nombres propios de la pregunta ("la Ría")
    palabras = [w for w in re.findall(r"[a-z]+(?: \d+\b)?|\d+", frase) if w not in _NO_ASUNTO
                and not (a.ciudadana and w in _PALABRAS_CIUDADANA)
                and (len(w) >= 4 or w.isdigit() or (len(w) == 3 and w in a.nombres_propios))]
    palabras = [w for w in palabras if not w.isdigit() or len(palabras) == 1]
    # los verbos en infinitivo ("construir un parque acuático") no son el asunto
    sin_verbos = [w for w in palabras if not re.fullmatch(r"[a-z]{4,}(ar|er|ir)", w)]
    palabras = sin_verbos or palabras
    reconocido = " ".join(e for _, e in a.temas) + " " + " ".join(n.lower() for _, n in a.grupos)
    palabras = [w for w in palabras if w[:4] not in reconocido
                and not any(re.fullmatch(pat, w) for pat, _, _ in _GRUPOS)]
    # las palabras de acción no identifican el asunto ("la implantación de la
    # Zona 30", "la subida de la plusvalía"): se quitan si queda alguna otra.
    # Antes solo se quitaban con un tema reconocido, y "subida de la tasa de
    # plusvalía" se relajó hacia "subida" (cualquier subida, con sus votos)
    sin_acciones = [w for w in palabras if w not in _ACCIONES]
    if sin_acciones or a.temas:
        palabras = sin_acciones
    return list(dict.fromkeys(palabras))


_ACCIONES = {"implantacion", "instalacion", "ampliacion", "creacion", "mejora", "mejoras", "construccion",
             "puesta", "reforma", "renovacion", "aplicacion", "desarrollo", "gestion", "promocion", "fomento",
             "impulso", "situacion", "problema", "problemas", "medidas", "plan", "ayudas", "ayuda",
             "subida", "bajada", "aumento", "reduccion", "traslado", "cierre", "apertura", "rehabilitacion",
             "remodelacion", "eliminacion", "modificacion", "cambio", "cambios", "uso", "utilizacion"}


def _asunto_libre(q: str, a: "Analisis") -> List[str]:
    m = _ASUNTO_RE.search(q)
    if not m:
        return []
    frase = re.sub(r"^(?:sobre|acerca de|de)\s+(?:(?:el|la|los|las|un|una)\s+)?", "", m.group(1))
    partes = [p for p in re.split(r"\s+(?:o|u|o bien)\s+(?:con\s+|sobre\s+)?", frase) if p.strip()]
    # "el traslado o la reforma del Mercado de San Antón": el complemento de la
    # segunda alternativa es de las dos ("traslado del Mercado de San Antón")
    if len(partes) == 2 and not re.search(r"\bdel?\b", partes[0]):
        comp = re.search(r"\b(del?\b.*)$", partes[1])
        if comp and len(partes[0].split()) <= 2:
            partes[0] = partes[0] + " " + comp.group(1)
    alternativas = [w for w in (_palabras_asunto(p, a) for p in partes) if w]
    if len(alternativas) > 1:
        a.asunto_alternativas = alternativas
    return list(dict.fromkeys(w for alt in alternativas for w in alt))


_CIUDADANA_RE = re.compile(
    r"\b(proposicion|propuesta|mocion|iniciativa)(es|s)? (vecinal|vecinales|ciudadana|ciudadanas|de (los )?vecinos)\b"
    r"|\bpresentad[ao]s? por (una |las |los |alguna |algun )?(asociacion|asociaciones|colectivo|vecin|particular|plataforma|entidad)"
    r"|\b(que|cual|cuales) (asociacion|asociaciones|colectivo|colectivos|entidad|entidades|plataforma|plataformas|organizacion)"
    r"\w* .{0,40}\b(presentad|propuest|llevad)")
# palabras que dicen quién presenta (no son el asunto) en una pregunta "ciudadana"
_PALABRAS_CIUDADANA = {"vecinal", "vecinales", "ciudadana", "ciudadanas", "ciudadano", "ciudadanos", "vecinos",
                       "vecinas", "particular", "particulares", "asociacion", "asociaciones", "colectivo", "colectivos",
                       "entidad", "entidades", "plataforma", "plataformas", "organizacion", "organizaciones"}


def analizar(pregunta: str) -> Analisis:
    q = _norm(pregunta)
    a = Analisis()

    if re.search(r"cuanto dinero|\bimporte|\beuros?\b|€|cuanto (costo|cuesta|se invirti|se gasto|se destino)"
                 r"|\bpartida presupuestaria|\bcuantia", q):
        a.fuera_de_alcance = "importes"
        # el grafo guarda el importe que figura en el título de algunos puntos
        # (subvenciones, créditos): graph_answer lo intenta antes de responder
        # que no consta
        a.importe = True
    elif re.search(r"\bfalt(ado|o|aron|an)\b|asistenci|ausenci|ausentes?\b|no (acudio|asistio)", q):
        a.fuera_de_alcance = "asistencia"

    vistos = set()
    for patron, slug, etiqueta in _temas():
        if slug in vistos:
            continue
        m = patron.search(q)
        if m and not any(etiqueta in e for _, e in a.temas):
            a.temas.append((slug, etiqueta))
            vistos.add(slug)

    for patron, uris, nombre in _GRUPOS:
        if re.search(patron, q):
            uso, original = _USO_DE_GRUPO.get(nombre, (None, False))
            if uso and not re.search(uso, pregunta if original else q, re.I if original else 0):
                continue
            a.grupos.append((uris, nombre))

    # solo años dentro del periodo de las actas: en "Bilbao Ría 2000" el número
    # es parte del nombre, no una fecha
    anios = [int(x) for x in re.findall(r"\b(19\d{2}|20\d{2})\b", q) if ANIO_MIN - 1 <= int(x) <= ANIO_MAX + 1]
    m_rango = re.search(r"(?:entre|de(?:sde)?)\s+(?:el\s+)?(\d{4})\s+(?:y|a|hasta)\s+(?:el\s+)?(\d{4})"
                        r"|\b(\d{4})\s*[-/]\s*(\d{4})\b", q)
    # "la legislatura 2019-2023", "el mandato de 2015": de su pleno de
    # constitución al de la siguiente (antes, de enero de 2019 a diciembre de
    # 2023, con plenos de las corporaciones anterior y siguiente)
    m_leg = re.search(r"\b(?:legislatura|mandato|corporacion)\s+(?:de\s+|del?\s+)?(\d{4})(?:\s*[-/]\s*(\d{4}))?\b", q)
    if m_leg and int(m_leg.group(1)) in CONSTITUCION and (
            not m_leg.group(2) or int(m_leg.group(2)) == int(m_leg.group(1)) + 4):
        ini = int(m_leg.group(1))
        # sin recortar a ANIO_MAX: "legislatura 2023-2027" nombra 2027 y la
        # cobertura rechazaba una consulta que lo incluía (sexta evaluación)
        a.anios = (ini, ini + 4)
        a.desde_fecha = CONSTITUCION[ini]
        if ini + 4 in CONSTITUCION:
            a.hasta_fecha = CONSTITUCION[ini + 4]
    elif m_rango:
        x, y = [int(v) for v in m_rango.groups() if v]
        a.anios = (min(x, y), max(x, y))
    elif re.search(r"\bdesde\s+(?:el\s+)?(\d{4})", q) and anios:
        desde = int(re.search(r"\bdesde\s+(?:el\s+)?(\d{4})", q).group(1))
        a.anios = None if desde <= ANIO_MIN else (desde, ANIO_MAX)
    elif re.search(r"\bhasta\s+(?:el\s+)?(\d{4})", q) and anios:
        a.anios = (ANIO_MIN, int(re.search(r"\bhasta\s+(?:el\s+)?(\d{4})", q).group(1)))
    elif len(set(anios)) == 1:
        a.anios = (anios[0], anios[0])
    elif len(set(anios)) > 1:
        a.compara_anios = True
        a.anios = (min(anios), max(anios))
    elif (re.search(r"\b(?:legislatura|mandato)\b", q) and not re.search(r"\b(?:anterior|pasad[ao])\b", q)
          and not re.search(r"\b(?:que|cual|cuales|cada|por|varias)\s+(?:legislatura|mandato)s?\b|\blegislaturas\b", q)):
        # "en toda la legislatura", "este mandato": la actual, que empieza en
        # junio de 2023 (los plenos de enero a mayo son de la anterior). No en
        # "¿en qué legislatura...?" ni "por legislatura", que las comparan (en la
        # sexta evaluación activaba la actual y la cobertura rechazaba la consulta)
        a.anios = (LEGISLATURA_ACTUAL, ANIO_MAX)
        a.desde_fecha = LEGISLATURA_INICIO
    # "los presupuestos municipales de 2016": el año es el del ejercicio, no el
    # del pleno (el presupuesto de 2016 se votó en noviembre y diciembre de 2015)
    m_ej = re.search(r"\bpresupuestos?\b(?:\s+\w+){0,3}?\s+(?:de|del|para)\s+(?:el\s+)?(?:ano\s+|ejercicio\s+)?(\d{4})\b", q)
    if m_ej:
        a.ejercicio = int(m_ej.group(1))
        a.anios, a.compara_anios = None, False
    if re.search(r"\bahora\b|\bactualmente\b|\ben la actualidad\b|\bhoy en dia\b", q):
        a.anio_ahora = True
        if anios:
            a.compara_anios = True

    if re.search(r"\bdecae|\bdecaid|\bdecayer", q):
        a.resultado = "decae"
    elif re.search(r"\bretirad", q):
        a.resultado = "retirada"
    elif re.search(r"\brechaz", q) and re.search(r"\baprob", q):
        # "¿fue aprobada o rechazada?" pregunta el resultado, no pide filtrar por uno
        a.resultado = None
    elif re.search(r"\brechaz", q):
        a.resultado = "rechazada"
    elif re.search(r"\baprob", q):
        a.resultado = "aprobada"

    a.reciente = bool(re.search(r"\bultim[ao]s?\b|mas reciente|recientemente", q))
    a.unanimidad = bool(re.search(r"unanimidad|unanime", q))
    a.oposicion = bool(re.search(r"\boposicion\b", q))
    # solo cuando habla de quién presenta: en "subvenciones a asociaciones
    # vecinales" las asociaciones son el asunto
    a.ciudadana = bool(_CIUDADANA_RE.search(q))
    meses = [n for nombre, n in _MESES.items() if re.search(rf"\b{nombre}\b", q)]
    for nombre, ms in _ESTACIONES.items():
        if re.search(rf"\b{nombre}\b", q):
            meses.extend(ms)
    a.meses = tuple(sorted(set(meses)))
    # "el pleno del 15 de agosto de 2016", "el 24-09-2015": un pleno concreto
    m_fecha = re.search(rf"\b(\d{{1,2}}) de ({'|'.join(_MESES)}) (?:de|del) (\d{{4}})\b", q)
    if m_fecha:
        a.fecha = f"{m_fecha.group(3)}-{_MESES[m_fecha.group(2)]:02d}-{int(m_fecha.group(1)):02d}"
    else:
        m_fecha = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", q)
        if m_fecha:
            a.fecha = f"{m_fecha.group(3)}-{int(m_fecha.group(2)):02d}-{int(m_fecha.group(1)):02d}"
    palabras_orig = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", pregunta)
    a.nombres_propios = [_norm(w) for w in palabras_orig[1:] if w[0].isupper() and len(w) >= 3]
    a.asunto = _asunto_libre(q, a)
    # sinónimos de varias palabras o con número ("zona 30", "zona de bajas
    # emisiones", "carril bici"): son más concretos que su tema
    a.temas_sinonimo = [(s, e) for s, e in a.temas if (s, e) in _SINONIMOS and (" " in e or re.search(r"\d", e))]
    return a


# --- cobertura de una consulta ---

def _anios_en_sparql(sparql: str) -> set:
    # años en filtros o triples (bo:anio 2019, ?anio >= 2015, "24-09-2015", "2020-03-01")
    return {int(y) for y in re.findall(r"(?<!\w)((?:19|20)\d{2})(?!\w)", sparql)}


def cobertura(a: Analisis, sparql: str) -> List[str]:
    """Lo que la pregunta pide y la consulta no filtra (o filtra de más)."""
    s = sparql or ""
    # los literales con clases de tildes ("m[oó]v[ií]l...") se reducen a la
    # primera letra de cada clase para poder buscar la palabra
    s_norm = _norm(re.sub(r"\[([^\]])[^\]]*\]", r"\1", s))
    literales = " ".join(re.findall(r'"([^"]*)"', s_norm))
    faltan = []

    for slug, etiqueta in a.temas:
        # al menos la mitad de las palabras de la etiqueta: "zona" sola no
        # cubre "zona de bajas emisiones"
        palabras = [w for w in etiqueta.split() if len(w) >= 4]
        presentes = sum(1 for w in palabras if w[:5] in literales)
        por_literal = bool(palabras) and presentes * 2 >= len(palabras)
        if f"br:{slug}" not in s and not por_literal:
            faltan.append(f'el tema "{etiqueta}" (br:{slug})')

    uris = " ".join(re.findall(r"br:(\w+)", s_norm)).replace("_", " ")
    if a.asunto_raices:
        # todas las raíces específicas, y a la vez: con OR o en ramas UNION
        # distintas la consulta devuelve lo que tiene cualquiera de ellas
        # ("ruido" O "Casco Viejo" -> una guía de artistas callejeros)
        texto = literales + " " + uris
        ausentes = [r for r in a.asunto_raices if r not in texto]
        if ausentes:
            faltan.append(f'el asunto: faltan {", ".join(ausentes)} (tienen que aparecer todas; usa el bloque de la PISTA)')
        elif len(a.asunto_raices) > 1 and len(a.asunto_grupos) <= 1:
            ramas = re.split(r"\bunion\b", s_norm)
            juntas = any(all(r in rama for r in a.asunto_raices) for rama in ramas)
            con_or = any("||" in e and sum(r in _norm(e) for r in a.asunto_raices) > 1
                         for e in re.findall(r"filter\s*\((.*?)\)\s*(?:\.|}|filter|$)", s_norm, re.S))
            if not juntas or con_or:
                faltan.append(f'el asunto: {", ".join(a.asunto_raices)} deben cumplirse todas a la vez (AND), '
                              'no con OR ni en ramas UNION distintas; usa el bloque de la PISTA')
    elif a.asunto and not any(w[:5] in literales or w[:5] in uris for w in a.asunto):
        faltan.append(f'el asunto "{" ".join(a.asunto)}" (no es un tema: búscalo en el título o en las menciones)')

    for uris, nombre in a.grupos:
        por_etiqueta = any(re.search(rf"\b{_ETIQUETA_GRUPO[u]}\b", literales) for u in uris)
        if not any(f"br:{u}" in s for u in uris) and not por_etiqueta:
            faltan.append(f"el grupo {nombre} (" + " o ".join(f"br:{u}" for u in uris) + ")")

    en_sparql = _anios_en_sparql(s)
    if a.anios:
        ini, fin = a.anios
        if ini == fin:
            if ini not in en_sparql:
                faltan.append(f"el año {ini}")
        elif a.compara_anios:
            if ini not in en_sparql or fin not in en_sparql:
                faltan.append(f"los años {ini} y {fin} (se comparan)")
        else:
            if ini not in en_sparql or (fin != ANIO_MAX and fin not in en_sparql):
                faltan.append(f"todo el rango de años {ini}-{fin} (no solo uno)")
        if a.anio_ahora and a.compara_anios and not any(y >= ANIO_MAX - 1 for y in en_sparql):
            faltan.append(f"el año actual ({ANIO_MAX - 1} o {ANIO_MAX}) para comparar con 'ahora'")
        fuera = {y for y in en_sparql if not ini <= y <= fin and not (a.anio_ahora and y >= ANIO_MAX - 1)}
        if fuera:
            faltan.append(f"sobran años que la pregunta no menciona: {sorted(fuera)}")
    elif en_sparql and not a.anio_ahora:
        faltan.append(f"sobra un filtro de año que la pregunta no pide: {sorted(en_sparql)}")

    individuo = {"aprobada": "bo:Aprobada", "rechazada": "bo:Rechazada",
                 "decae": "bo:Decae", "retirada": "bo:Retirada"}.get(a.resultado)
    if individuo and individuo not in s:
        faltan.append(f"el resultado {individuo}")

    if a.reciente and not re.search(r"\bDESC\b|\bMAX\s*\(", s, re.I):
        faltan.append("ordenar por fecha para quedarse con lo más reciente (bo:fechaISO + ORDER BY DESC)")
    if a.unanimidad and not re.search(r"votoEnContraDe|seAbstuvo|unanim", s, re.I):
        faltan.append("la unanimidad (sin grupos en contra ni abstenciones)")
    if a.oposicion and not ("grupo_equipo_de_gobierno" in s and re.search(r"NOT\s+IN|!=|NOT\s+EXISTS", s, re.I)):
        faltan.append("excluir a los grupos de gobierno para quedarse con la oposición")
    if a.meses and not re.search(r"bo:mes|bo:fechaISO|bo:fecha|bo:mesSesion", s):
        faltan.append(f"los meses {list(a.meses)} (bo:mes)")
    return faltan


# =============================================================================
# 2. Núcleo del asunto
# =============================================================================

# Núcleo del asunto de una pregunta, marcado por el LLM.
#
# El análisis por reglas solo encontraba el asunto detrás de "sobre X",
# "relacionado con X", "se ha debatido de X"... y en las evaluaciones
# independientes se le escapaban formulaciones corrientes ("la instalación de
# una noria en el Arenal", "la modificación del Plan General", "rebajar el IBI a
# las familias numerosas") o elegía mal la palabra ("hermanamiento con una
# ciudad extranjera" -> "extranjera"). Sin asunto, la consulta la escribía el
# LLM filtrando por una palabra suelta, y de ahí salían la mayoría de las
# respuestas falsas.
#
# Aquí el LLM NO escribe ninguna consulta: solo copia de la pregunta las
# palabras que nombran el asunto concreto y dice cuáles son imprescindibles
# (para relajar la búsqueda sin perder el sentido). Todo lo que devuelve se
# comprueba contra la pregunta: una palabra que no esté en ella se descarta, así
# que no puede inventar filtros. El resultado se guarda en caché por pregunta
# para que las ejecuciones sean reproducibles.

# v2: las enumeraciones con "y" se separan en alternativas (§7.16); con otra
# versión del prompt no se reutilizan las respuestas guardadas
_CACHE = os.path.join(CACHE_GRAFO, "nucleo_v2.json")
_LOCK = threading.Lock()
_MEMO = None

PROMPT = """Te doy una pregunta sobre las actas del Pleno del Ayuntamiento de Bilbao. Identifica el ASUNTO CONCRETO
del que pregunta: la cosa, lugar, proyecto, servicio, entidad o medida concreta.

NO son asunto (no los incluyas): grupos políticos o partidos, concejales y personas, años y fechas, resultados
(aprobada, rechazada...), las palabras "pleno", "Bilbao", "ayuntamiento", "moción", "proposición", "propuesta",
"iniciativa", "votación", y las acciones genéricas (instalación, implantación, modificación, subida, rebaja,
creación, mejora, gestión, construcción...). Si la pregunta solo pide un recuento o ranking general sobre un tema
amplio (vivienda, movilidad, educación, seguridad...) sin nada más concreto, el asunto es ese tema.
Si la pregunta no trata de ningún asunto (p. ej. "¿cuántas proposiciones se rechazaron en 2021?"), devuelve listas vacías.

Copia las palabras TAL CUAL aparecen en la pregunta. Si la pregunta enumera varios asuntos unidos por "o" o por
"y" ("colegios y escuelas", "parques o jardines"), pon cada uno aparte: valen las proposiciones de CUALQUIERA de
ellos, no hace falta que traten de todos a la vez. Solo van juntos si forman un único nombre (un organismo, un
proyecto, una calle: "Parques y Jardines", "Bilbao Ría 2000").
En "clave" pon, de cada alternativa, la palabra o palabras sin las que el asunto pierde el sentido (lo mínimo que
debe aparecer en un texto para que hable de eso).

Ejemplos:
- "¿Se ha tratado en el Pleno el hermanamiento de Bilbao con alguna ciudad extranjera?"
  {{"alternativas": ["hermanamiento con alguna ciudad extranjera"], "clave": [["hermanamiento"]]}}
- "¿Por qué aprobó el Pleno la instalación de una noria gigante en el Arenal?"
  {{"alternativas": ["noria gigante en el Arenal"], "clave": [["noria"]]}}
- "¿Cuántas mociones sobre el soterramiento de las vías de Renfe se han presentado?"
  {{"alternativas": ["soterramiento de las vías de Renfe"], "clave": [["soterramiento", "Renfe"]]}}
- "¿Qué dijo el PP sobre el Athletic o San Mamés?"
  {{"alternativas": ["Athletic", "San Mamés"], "clave": [["Athletic"], ["San Mamés"]]}}
- "¿En qué año se presentaron más proposiciones sobre deporte y polideportivos municipales?"
  {{"alternativas": ["deporte", "polideportivos municipales"], "clave": [["deporte"], ["polideportivos"]]}}
- "¿Qué grupo ha presentado más proposiciones sobre vivienda?"
  {{"alternativas": ["vivienda"], "clave": [["vivienda"]]}}
- "¿En qué año se rechazaron más proposiciones?"
  {{"alternativas": [], "clave": []}}

Responde SOLO con el JSON, sin explicaciones.

PREGUNTA: {pregunta}
JSON:"""


def _cargar():
    global _MEMO
    if _MEMO is None:
        try:
            with open(_CACHE, encoding="utf-8") as f:
                _MEMO = json.load(f)
        except (OSError, ValueError):
            _MEMO = {}
    return _MEMO


def _guardar():
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    tmp = _CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_MEMO, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, _CACHE)


def _en_pregunta(frase: str, q: str) -> bool:
    # todas las palabras de la frase están en la pregunta (sin tildes ni mayúsculas)
    palabras = re.findall(r"[a-z0-9]+", _norm(frase))
    return bool(palabras) and all(re.search(rf"\b{re.escape(w)}\b", q) for w in palabras)


def _validar(d, pregunta: str):
    q = _norm(pregunta)
    alternativas, claves = [], []
    for i, alt in enumerate(d.get("alternativas") or []):
        if not isinstance(alt, str) or not _en_pregunta(alt, q):
            continue
        clave = (d.get("clave") or [])
        clave = clave[i] if i < len(clave) and isinstance(clave[i], list) else []
        clave = [c for c in clave if isinstance(c, str) and _en_pregunta(c, _norm(alt))]
        alternativas.append(_norm(alt))
        claves.append([_norm(c) for c in clave])
    return {"alternativas": alternativas, "clave": claves}


# "el Metro de Bilbao y sus estaciones": lo que sigue a "y" con posesivo es una parte de lo
# anterior, no otro asunto. Como alternativa sumaba a los recuentos todo lo que contuviera
# "estaciones" (autobuses, tren): 83 proposiciones en vez de las ~42 del Metro (octava evaluación).
def _sin_posesivos(d):
    if not d or len(d.get("alternativas") or []) < 2:
        return d
    manten = [i for i, alt in enumerate(d["alternativas"]) if i == 0 or not re.match(r"(su|sus)\b", alt)]
    if len(manten) == len(d["alternativas"]):
        return d
    return {"alternativas": [d["alternativas"][i] for i in manten],
            "clave": [d["clave"][i] for i in manten if i < len(d["clave"])]}


# {"alternativas": [frase, ...], "clave": [[palabra, ...], ...]} o None si el LLM falla
def nucleo(pregunta: str, invocar):
    with _LOCK:
        memo = _cargar()
        if pregunta in memo:
            return _sin_posesivos(memo[pregunta])
    try:
        texto = invocar(PROMPT.format(pregunta=pregunta))
        m = re.search(r"\{.*\}", texto, re.S)
        d = _validar(json.loads(m.group(0)), pregunta) if m else None
    except Exception as e:
        print(f"[!] Núcleo del asunto: el LLM falló — {type(e).__name__}: {e}", flush=True)
        return None
    if d is None:
        return None
    with _LOCK:
        _cargar()[pregunta] = d
        _guardar()
    return _sin_posesivos(d)


# =============================================================================
# 3. Asunto de la pregunta
# =============================================================================

# El asunto de la pregunta ("arbolado", "Ledesma", "precio del agua"): sus
# palabras, las proposiciones del grafo que lo contienen y la pista para el LLM.

# nombres propios que no son el asunto de la pregunta ni la identifican
_PROPIOS_NEUTROS = {"bilbao", "pleno", "ayuntamiento", "bilboko", "udala"}


_INDICE_ASUNTO = None


# campos buscables de cada proposición: título, subtemas (etiquetas y
# sinónimos) y entidades mencionadas, sin tildes y en minúsculas
def _indice_asunto(g) -> dict:
    global _INDICE_ASUNTO
    if _INDICE_ASUNTO is None:
        textos, tratan = {}, {}
        for q, de_que_trata in (
                ("SELECT ?p ?t WHERE { ?p a bo:Proposicion ; bo:tituloTopic ?t }", True),
                ("SELECT ?p ?t WHERE { ?p a bo:Proposicion ; bo:trataSobre ?s . ?s skos:prefLabel|skos:altLabel ?t }", True),
                ("SELECT ?p ?t WHERE { ?p a bo:Proposicion ; bo:menciona ?e . ?e rdfs:label ?t }", False)):
            for row in g.query(_PREFIXES + q):
                t = _norm(str(row.t))
                textos.setdefault(str(row.p), []).append(t)
                if de_que_trata:
                    tratan.setdefault(str(row.p), []).append(t)
        # los encabezados de sección ("18. PROPOSICIONES", "MOCIONES DE URGENCIA") no son
        # proposiciones: sin título propio, grupo ni resultado, pero con las menciones de toda la
        # sección. En la octava evaluación uno salió como "la última vez que se debatió" (o09).
        # nombres propios del debate (construccion/debates.py): solo como menciones, en las
        # formas que distinguen tratar de mencionar (ver _props_asunto)
        debate = {}
        for row in g.query(_PREFIXES + "SELECT ?p ?t WHERE { ?p a bo:Proposicion ; bo:nombradoEnDebate ?t }"):
            debate.setdefault(str(row.p), []).append(str(row.t))
        for row in g.query(_PREFIXES + "SELECT ?p ?t WHERE { ?p a bo:Proposicion ; bo:tituloTopic ?t }"):
            if _ENCABEZADO_RE.match(str(row.t).strip()):
                textos.pop(str(row.p), None)
                tratan.pop(str(row.p), None)
                debate.pop(str(row.p), None)
        _INDICE_DEBATE.update(debate)
        _INDICE_ASUNTO = textos
        _INDICE_TRATA.update(tratan)
    return _INDICE_ASUNTO


# solo título y subtemas: de qué TRATA la proposición (sin las menciones)
_INDICE_TRATA: dict = {}


# nombres propios del texto del debate (bo:nombradoEnDebate)
_INDICE_DEBATE: dict = {}


_ENCABEZADO_RE = re.compile(r"^(\d+\.\s*(PROPOSICIONES(-PROPOSIZIOAK)?|MOCIONES DE URGENCIA)|Udalbatzarreko "
                            r"Idazkaritza Nagusia)$", re.I)


# raíz de una palabra para buscarla con REGEX: sin la terminación de plural
# ni de género ("arbolado" -> "arbolad", "piscinas" -> "piscin")
def _raiz(w: str) -> str:
    return re.sub(r"(es|as|os|s|a|o|e)$", "", w) if len(w) > 5 else w


# Cuando la pregunta trata de un asunto que no es un tema del vocabulario
# ("arbolado", "Ledesma", "precio del agua"), el LLM local no sabía dónde
# buscarlo: inventaba etiquetas de tema y la consulta daba 0. Se le da el
# bloque ya comprobado sobre el grafo (título, subtemas o menciones) con el
# número de proposiciones que encuentra. Se exigen todas las palabras
# específicas del asunto; las muy frecuentes ("urbano" casa "urbanismo") se
# descartan si queda alguna otra.
# proposiciones con algún campo (título, subtema, mención) que contiene todas
# las raíces del asunto; misma regla que la pista
# grupos: alternativas de raíces ("el Athletic o San Mamés" -> [[athletic],
# [mames]]); dentro de cada una se exigen todas. Primero se buscan las que
# TRATAN del asunto (título o subtemas); solo si no hay ninguna, las que lo
# mencionan: en la tercera evaluación, una proposición sobre las piscinas de
# Txurdinaga que solo mencionaba Etxebarria salió como "aprobada sobre el
# parque de Etxebarria".
def _props_asunto(g, grupos, solo_trata=False, con_menciones=False) -> list:
    if grupos and isinstance(grupos[0], str):
        grupos = [grupos]
    alternativas = [[re.compile(rf"\b{re.escape(r)}") for r in grupo] for grupo in grupos if grupo]

    def buscar(indice):
        return sorted(p for p, campos in indice.items()
                      if any(any(all(x.search(c) for x in alt) for c in campos) for alt in alternativas))
    todo = _indice_asunto(g)
    if solo_trata:
        return buscar(_INDICE_TRATA)
    if con_menciones:
        # también lo nombrado en el debate (Mercado del Ensanche, Palacio de Justicia: octava evaluación).
        # Las palabras de una alternativa tienen que estar juntas en un mismo campo: repartidas entre
        # campos ("seguridad" en el título, "Bilbao la Vieja" en el debate) entraban proposiciones sin
        # relación, porque "Seguridad" es también un nombre propio (el área municipal)
        return sorted(set(buscar(todo)) | set(buscar(_INDICE_DEBATE)))
    return buscar(_INDICE_TRATA) or buscar(todo)


def _norm_q(pregunta: str) -> str:
    return _norm(pregunta)


# calcula la pista y las raíces del asunto que exigirá la cobertura; se hace
# una vez por pregunta, antes de las plantillas, para que las dos vías usen lo mismo
# cargos y personas no son el asunto de una proposición ("¿quién ha sido
# alcalde?" se responde con los datos de personas, no buscando "alcalde")
_STOP_NUCLEO = {"alcalde", "alcaldesa", "alcaldia", "concejal", "concejala", "concejales", "concejalas", "portavoz",
                "portavoces", "teniente", "corporativo", "corporativos",
                "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "en", "con", "por",
                "para", "sobre", "y", "o", "u", "e", "que", "su", "sus", "lo", "se", "mas", "muy", "alguna", "alguno",
                "algun", "algunas", "algunos", "toda", "todo", "todas", "todos", "otra", "otro", "otras", "otros"}


# palabras de una frase del núcleo que se exigirán en el índice: sin artículos
# ni preposiciones, sin años ni números sueltos, sin grupos; las palabras
# genéricas (plaza, calle, zona...) solo si van en mayúscula en la pregunta
# ("la Plaza Circular") o pegadas a un número ("zona 30")
def _palabras_nucleo(frase: str, analisis) -> list:
    propios = set(analisis.nombres_propios) - _PROPIOS_NEUTROS
    out = []
    for w in re.findall(r"[a-z]+(?: \d+\b)?|\d+", frase):
        # "de 0", "a 3": un número pegado a una preposición no es un nombre ("zona 30" sí)
        if w.split(" ")[0] in _STOP_NUCLEO:
            continue
        # "proposiciones vecinales": dice quién las presenta, no de qué tratan
        if analisis.ciudadana and w in _PALABRAS_CIUDADANA:
            continue
        if w.isdigit() or any(re.fullmatch(pat, w) for pat, _, _ in _GRUPOS):
            continue
        # verbos en infinitivo que se cuelan en el núcleo ("rebajar el IBI")
        if len(w) > 5 and re.fullmatch(r"[a-z]+(ar|er|ir)", w) and w not in propios:
            continue
        if w in _NO_ASUNTO and w not in propios:
            continue
        if len(w) < 4 and " " not in w and w not in propios:
            continue
        out.append(w)
    return list(dict.fromkeys(out))


# El núcleo marcado por el LLM (sección 2) sustituye al asunto de las reglas.
# Si todas sus palabras son las de un tema reconocido, el tema ya lo filtra y
# no hay asunto aparte; si el LLM falla, se quedan las reglas.
def _asunto_desde_nucleo(analisis, pregunta: str) -> None:
    n = nucleo(pregunta, lambda p: _llm_invoke(p, prefer="gemini"))
    if n is None:
        return
    reconocido = " ".join(e for _, e in analisis.temas)
    alternativas, claves = [], []
    temas_alt, dentro = [], set()
    for frase, clave in zip(n["alternativas"], n["clave"]):
        palabras = _palabras_nucleo(frase, analisis)
        # temas reconocidos cuyas palabras están en esta alternativa
        suyos = [s for s, e in analisis.temas if _tema_en_frase(e, frase)]
        dentro.update(suyos)
        if not palabras or all(w[:4] in reconocido for w in palabras):
            temas_alt.extend(suyos)
            continue
        alternativas.append(palabras)
        claves.append([w for c in clave for w in _palabras_nucleo(c, analisis)] or palabras)
    analisis.asunto = list(dict.fromkeys(w for alt in alternativas for w in alt))
    analisis.asunto_alternativas = alternativas if len(alternativas) > 1 else []
    analisis.asunto_clave = claves
    # Las alternativas se SUMAN. "medio ambiente y zonas verdes": medio ambiente
    # es un tema y zonas verdes no; antes el tema se ponía como filtro y se
    # contaban las de medio ambiente QUE ADEMÁS hablaban de zonas verdes. Y un
    # tema dentro de un asunto más concreto ("el barrio de Otxarkoaga") no se
    # cruza con él: el asunto ya lo concreta (salían 1 de 13).
    temas_alt = list(dict.fromkeys(temas_alt))
    if dentro and (alternativas or len(temas_alt) > 1):
        analisis.asunto_temas = temas_alt
        analisis.temas_absorbidos = [t for t in analisis.temas if t[0] in dentro]


# todas las palabras de la etiqueta de un tema (de 4 letras o más) en la frase
def _tema_en_frase(etiqueta: str, frase: str) -> bool:
    palabras = [w for w in etiqueta.split() if len(w) >= 4] or etiqueta.split()
    return bool(palabras) and all(re.search(rf"\b{re.escape(w[:4])}", frase) for w in palabras)


# proposiciones de los temas que son alternativas del asunto
def _props_temas(g, slugs) -> set:
    return {r["p"] for s in slugs
            for r in _ejecutar(g, _PREFIXES + f"SELECT DISTINCT ?p WHERE {{ ?p bo:trataTemaAmplio br:{s} }}")}


def _preparar_asunto(g, analisis, pregunta: str = "") -> None:
    if pregunta:
        _asunto_desde_nucleo(analisis, pregunta)
    # "el presupuesto de 2016": el del ejercicio 2016, que se aprueba a finales
    # de 2015; se busca por el título, no por la fecha del pleno
    if analisis.ejercicio:
        analisis.asunto = ["presupuesto general", str(analisis.ejercicio)]
        analisis.asunto_alternativas, analisis.asunto_temas, analisis.temas_absorbidos = [], [], []
        analisis.asunto_clave = [["presupuesto", str(analisis.ejercicio)]]
    # un tema reconocido por un sinónimo concreto ("zona 30" -> peatonalización)
    # se busca también por ese sinónimo; si no hay ninguna proposición con él,
    # la consulta directa vuelve al tema entero y lo dice
    if analisis.temas_sinonimo and not analisis.asunto:
        palabras = [w for _, e in analisis.temas_sinonimo for w in re.findall(r"[a-z]+(?: \d+\b)?", e)
                    if len(w) >= 4 and w not in _NO_ASUNTO or re.search(r"\d", w)]
        analisis.asunto = list(dict.fromkeys(palabras))
    analisis.pista = _pista_asunto(g, analisis)


def _pista_asunto(g, analisis) -> str:
    if not analisis.asunto:
        return ""
    indice = _indice_asunto(g)
    total = len(indice)
    def cumplen(rxs):  # proposiciones con algún campo que contiene todas las raíces
        return sum(1 for campos in indice.values() if any(all(re.search(x, c) for x in rxs) for c in campos))
    propios = {_raiz(w) for w in getattr(analisis, "nombres_propios", [])}

    # Se exigen las palabras específicas y los nombres propios; las muy
    # frecuentes ("urbano") se descartan. Una palabra sin ninguna coincidencia
    # NO se descarta: antes se quitaba y la búsqueda se ensanchaba sin avisar
    # ("ampliación del Bilbobus hasta Otxarkoaga" -> 97 de Bilbobus y
    # Otxarkoaga); ahora da 0 y la respuesta es que no consta.
    def especificas_de(palabras):
        raices = [(_raiz(w), cumplen([rf"\b{re.escape(_raiz(w))}"])) for w in palabras]
        esp = ([r for r, n in raices if n <= 0.03 * total or r in propios or r.isdigit()] or [r for r, n in raices if n])
        return esp or [r for r, _ in raices]

    alternativas = analisis.asunto_alternativas or [analisis.asunto]
    analisis.asunto_grupos = [especificas_de(alt) for alt in alternativas]
    analisis.asunto_raices = [r for grupo in analisis.asunto_grupos for r in grupo]
    rx_grupos = [[rf"\b{re.escape(r)}" for r in grupo] for grupo in analisis.asunto_grupos]
    n = len(_props_asunto(g, analisis.asunto_grupos))
    if n == 0 and not any(cumplen([x]) for grupo in rx_grupos for x in grupo):
        palabras = ", ".join(analisis.asunto)
        return (f"\n\nPISTA: ninguna proposición del grafo menciona ({palabras}) ni en el título, ni en sus "
                "subtemas, ni en sus entidades. Si la pregunta depende de eso, la respuesta es que no consta: "
                "genera igualmente la consulta buscando en el título, y devolverá 0 filas.")
    doble = lambda x: x.replace(chr(92), chr(92) * 2)
    filtros = " || ".join(
        "(" + " && ".join(f'REGEX(STR(?_txt), "{doble(x)}", "i")' for x in grupo) + ")" for grupo in rx_grupos)
    bloque = ("{ ?p bo:tituloTopic ?_txt } UNION { ?p bo:trataSobre ?_s . ?_s skos:prefLabel|skos:altLabel ?_txt }"
              f" UNION {{ ?p bo:menciona ?_e . ?_e rdfs:label ?_txt }} FILTER({filtros})")
    return (f"\n\nPISTA: el asunto de la pregunta ({', '.join(analisis.asunto)}) no es un tema del vocabulario. "
            f"Para encontrar sus proposiciones usa este bloque tal cual (comprobado: {n} proposiciones "
            f"distintas lo cumplen; cuenta con COUNT(DISTINCT ?p)) y añade el resto de filtros de la "
            f"pregunta (grupo, años, resultado, orden):\n  ?p a bo:Proposicion . {bloque}"
            + ("\nComo son 0, la respuesta es que no consta ninguna." if n == 0 else ""))
