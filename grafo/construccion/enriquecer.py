"""
Enriquecimiento del grafo: se aplica en build_rdf.py después de cargar las
proposiciones y ANTES del razonador, y queda guardado en bilbao_reasoned.ttl.

  - qué es cada punto del orden del día y quién lo presenta
  - votaciones, barrio y distrito, importes e intervenciones
  - copias en euskera, fechas, plenos y legislaturas, y las correcciones de
    votos.py y debates.py (función enriquecer)
"""
import json
import os
import re
import unicodedata
from collections import defaultdict

from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from comun.rutas import dato_grafo
from grafo.construccion import debates as D, texto_actas as T, votos as V


# =============================================================================
# Tipo de cada punto y quién lo presenta
# =============================================================================

# Qué es cada punto del orden del día y quién lo presenta, a partir de su título
# y del texto de la votación. Lo usa enriquecer.py.
#
# Todos los puntos de las actas están en el grafo como bo:Proposicion, pero no
# todos son proposiciones: también hay preguntas, daciones de cuenta, la
# aprobación del acta anterior, la urgencia de la convocatoria... Contados como
# proposiciones inflaban los recuentos ("aprobar por unanimidad el acta de la
# sesión anterior" salía como proposición aprobada por unanimidad). Y las
# proposiciones vecinales, que presentan asociaciones o particulares, quedaban
# como "grupo desconocido" y en 44 de 94 sin saber quién las presentaba.

TIPOS = ("proposicion_grupo", "propuesta_gobierno", "proposicion_ciudadana", "informe_iniciativa", "enmienda_articulado",
         "pregunta", "dacion_cuenta", "tramite", "declaracion_institucional", "debate_estado_ciudad", "otro")
# puntos que no son proposiciones: no entran en los recuentos de proposiciones
NO_PROPOSICION = ("pregunta", "dacion_cuenta", "tramite", "debate_estado_ciudad", "informe_iniciativa")


def _n(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn"))


_INICIO = r"^\W*\d*\W*\s*"
_INFORME = re.compile(_INICIO + r"(proposici\s?on|iniciativa\s+vecinal)\s+(vecinal\s+)?(d\s?e\s+f\s?e\s?c\s?h\s?a\b"
                      r"|presentada\s+(el|en)\b)")
_GRUPO_EN_TITULO = re.compile(r"grupo\s*(municipal|politico)|equipo de gobierno|udal\s*talde")
# propuestas del gobierno que resuelven las enmiendas o alegaciones de los grupos
_RESOLUCION_GOBIERNO = re.compile(r"resoluci\w+ de las\s+(enmiendas|alegaciones)")
# "21. Proposición que presenta el Grupo Municipal...", "PROPOSICIÓN del grupo político municipal..."
_DE_GRUPO = re.compile(_INICIO + r"(proposici\s?on|mocion|proposizioa)\s+(que\s+(presenta|formula)n?\s+(el|los)\s+|del?\s+(la\s+|los\s+)?)"
                       r"(grupos?|equipo de gobierno)")
_CIUDADANA = re.compile(
    _INICIO + r"(iniciativa\s+vecinal|herri\s+ekimena)\b"
    r"|" + _INICIO + r"proposici\s?on\s+(vecinal|ciudadana)\b"
    r"|" + _INICIO + r"proposici\s?on\s+(d\s?e\s+f\s?e\s?c\s?h\s?a\s+.{0,45}?\s+)?"
    r"(presentada\s+.{0,60}?\s*por\s+|de\s+(la\s+|el\s+|los\s+)?[\"“#]|de\s+(la\s+|el\s+)?(asociacion|plataforma|federacion|"
    r"fundacion|coordinadora|comunidad|ampa)|de\s+(don|dona|d\.|dna)\s)")
# (tipo, patrón, hasta qué carácter del título se busca): solo al principio,
# porque la parte dispositiva de una proposición puede mencionar "las actas" o
# "el estado de la ciudad"
_REGLAS = [
    ("pregunta", _INICIO + r"(preguntas?|galder)", 60),
    ("declaracion_institucional", r"declaracion\s+institucional|adierazpen\s+instituzional", 120),
    ("debate_estado_ciudad", r"(debate|eztabaida).{0,40}(estado de la ciudad|hiriaren egoera)|hiriaren egoerari buruzko", 120),
    ("tramite", _INICIO + r"(aprobar.{0,90}\ba\s?c\s?t\s?as?\b|aprobacion\s+de(l| las?)?\s+actas?\b|ratificacion de la urgencia|declaracion de (la )?ur\s?gencia"
                r"|toma de posesion|toma de conocimiento de la renuncia|se da lectura a la comunicacion de renuncia|sorteo)"
                r"|^se aprueba por unanimidad la declaracion de urgencia|premiazkoa|fallecimiento del", 160),
    # (el OCR corta palabras: "TOMA DE C ONOCIMIENTO", "DECLARACIÓN de ur gencia")
    ("dacion_cuenta", _INICIO + r"(se da cuenta|dar cuenta|toma\s+de\s+c\s?o\s?n\s?o\s?c|se da lectura)", 60),
    ("enmienda_articulado", _INICIO + r"propuesta de (enmienda|modificacion) (de \w+ )?n(u|º|o)", 80),
]


def tipo_punto(titulo: str, grupo: str) -> str:
    t = _n(titulo)[:300]
    # Entrada de la memoria anual de la Comisión Especial de Sugerencias y
    # Reclamaciones: "Proposición de fecha 4 de enero de 2021 presentada por ...
    # fue tratada en el Pleno de 28 de enero de 2021 y aprobada ... se notificó".
    # No es un debate: repite una iniciativa ya debatida, con la fecha del
    # informe (Belaunaldi Galdua figuraba aprobada el 24-02-2022; se aprobó el
    # 28-01-2021). Los debates son "INICIATIVA VECINAL presentada por ...".
    if _INFORME.match(t):
        return "informe_iniciativa"
    # una iniciativa vecinal nunca la presenta un grupo, aunque el título nombre
    # al Equipo de Gobierno ("que el Pleno inste al Equipo de Gobierno a...")
    if re.match(_INICIO + r"(iniciativa\s+vecinal|herri\s+ekimena)\b", t):
        return "proposicion_ciudadana"
    if _DE_GRUPO.search(t):
        return "proposicion_grupo" if grupo not in ("Desconocido", "EQUIPO DE GOBIERNO") else "propuesta_gobierno"
    if _CIUDADANA.search(t) and not _GRUPO_EN_TITULO.search(t[:160]):
        return "proposicion_ciudadana"
    for tipo, rx, hasta in _REGLAS:
        if re.search(rx, t[:hasta]):
            return tipo
    if grupo not in ("Desconocido", "EQUIPO DE GOBIERNO") and re.search(r"proposici|mocion|proposizio", t):
        return "proposicion_grupo"
    if grupo != "Desconocido" or re.search(r"propuesta|proposamen|dictamen|ordenanza|aprobacion (inicial|definitiva|provisional)", t):
        return "propuesta_gobierno"
    return "otro"


# --- quién presenta una proposición ciudadana ---

# ("comisión" no: "la Comisión Especial de referencia" es del cuerpo del texto;
# "la Comisión Promotora por una Asociación..." va entre comillas y entra por ahí)
_ORG = (r"asociaci[oó]n(?:es)?|plataforma|federaci[oó]n|fundaci[oó]n|coordinadora|sindicato|sociedad"
        r"|ampa|colectivo|elkartea|comunidad de propietarios|entidad social")
_FIN = (r"(?=\s*[,.;(]|\s+(?:que|para|relativ\w*|relacionad\w*|solicit\w*|mediante|en la que|instando|insta|propon\w*"
        r"|sobre|y en nombre|denominad\w*)\b|$)")


def _limpia(s: str) -> str:
    s = re.sub(r"[“”\"«»]", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;-")
    return re.sub(r"^(la|el|los|las|del|de la|de los|todas las)\s+", "", s, flags=re.I)


# (nombre, tipo) del que presenta, o None. Prefiere la organización a la
# persona que firma "en representación de" (y no da nombres anonimizados en la
# propia fuente: "#E.G.Z.#", "doña ______")
def entidad_ciudadana(titulo: str):
    t = re.sub(r"\s+", " ", titulo or "")[:400]
    # 1) nombre entre comillas, con la palabra de organización delante o dentro
    for m in re.finditer(r"[“\"]\s*([^”\"]{3,120}?)\s*[”\"]", t):
        dentro = m.group(1)
        antes = t[max(0, m.start() - 70):m.start()]
        # 'la asociación BIEL... denominada “Moción al Ayuntamiento...”': lo
        # entrecomillado es el título de la proposición
        if re.search(r"denominad\w*\s*$", antes, re.I):
            continue
        if re.search(_ORG, dentro, re.I):
            return _limpia(dentro), "organizacion"
        k = list(re.finditer(rf"\b({_ORG})\b[^“\",.;]{{0,60}}$", antes, re.I))
        if k:
            return _limpia(antes[k[-1].start():] + " " + dentro), "organizacion"
        if re.search(r"(representaci[oó]n|nombre|presentada por|presenta)\s+(de\s+)?(la\s+|el\s+|del\s+)?$", antes, re.I):
            return _limpia(dentro), "organizacion"
    # 2) organización sin comillas
    m = re.search(rf"\b(?:{_ORG})\b\s+[^,.;“\"]{{3,90}}?{_FIN}", t, re.I)
    if m and not re.search(r"^asociaci[oó]n de municipios", m.group(0), re.I):
        return _limpia(m.group(0)), "organizacion"
    # 2b) "en representación de Ekologistak Martxan", "en nombre de la Asamblea de Jóvenes..."
    m = re.search(rf"\ben\s+(?:represent\s?aci[oó]n|nombre)\s+de(?:l|\s+la|\s+los|\s+las)?\s+"
                  rf"((?-i:[A-ZÁÉÍÓÚÑ])[^,.;“\"]{{2,90}}?){_FIN}", t, re.I)
    if m and not re.search(r"#|__|^(vecindario|vecinos|vecinas)\b", m.group(1), re.I):
        return _limpia(m.group(1)), "organizacion"
    # 3) un colectivo vecinal ("el vecindario del barrio de Betolaza", "los Vecinos del Distrito 2")
    m = re.search(rf"\b(vecindario|vecinos|vecinas)\s+de(?:l| la)?\s+[^,.;]{{3,60}}?{_FIN}", t, re.I)
    if m:
        return _limpia(m.group(0)), "organizacion"
    # 4) la persona que la presenta, si su nombre no está anonimizado
    m = re.search(r"\b(?:don|doña|dña\.?)\s+((?:[A-ZÁÉÍÓÚÑ][\wáéíóúñ'\-]+\s*){2,5})", t)
    if m and not re.search(r"#|__", m.group(0)):
        return _limpia(m.group(1)), "persona"
    return None


# Clave para reunir las variantes de una misma organización ("Pentsionistak
# Martxan" / "Asociación Pentsionistak Martxan de Bizkaia", "Fundación MUNDUBAT"
# / "Fundación Mundubat- Mundubat Fundazioa"): sin las palabras genéricas, sin
# repetir palabras y sin el lugar del final
_GENERICAS = {"asociacion", "asociaciones", "elkartea", "elkarteen", "fundacion", "fundazioa", "plataforma",
              "vecinal", "vecinales", "vecinos", "vecinas", "as", "auzo", "de", "del", "la", "el", "los", "las", "y",
              "en", "a", "por", "bizkaia", "bilbao", "bilboko", "sociedad", "entidad", "social"}


def clave_organizacion(nombre: str) -> str:
    # (las letras sueltas son cortes de OCR: "J ubilados", "Colectiv o")
    palabras = [w for w in re.findall(r"[a-z0-9]+", _n(nombre)) if w not in _GENERICAS and len(w) > 1]
    return "_".join(dict.fromkeys(palabras))[:80] or "_".join(re.findall(r"[a-z0-9]+", _n(nombre)))[:80]


# las claves que empiezan por la misma palabra distintiva son la misma
# organización ("nagusiak" / "nagusiak_jubilados", "vida_independiente_enebizia"
# / "..._zorrotza"); se queda la más corta
_NO_DISTINTIVAS = {"federacion", "coordinadora", "sindicato", "colectivo", "comunidad", "ampa", "padres", "familias",
                   "familiares", "afectados", "vecindario", "jubilados", "comision", "asamblea", "union"}


def unificar_claves(claves) -> dict:
    canon = {}
    for k in sorted(set(claves), key=lambda k: (len(k), k)):
        primera = k.split("_")[0]
        distintiva = len(primera) >= 4 and primera not in _NO_DISTINTIVAS
        previa = next((c for c in set(canon.values()) if distintiva and c.split("_")[0] == primera), None)
        canon[k] = previa or k
    return canon


# --- de qué votación es el recuento guardado ---

# El recuento (bo:votosFavor/Contra) y el voto por grupo se extrajeron de "la"
# votación del punto, pero cuando hay enmiendas a veces es la de una enmienda
# ("queda aceptada la enmienda del EQUIPO DE GOBIERNO (Votos emitidos: 28 | a
# favor: 21, en contra: 7)"), y la proposición figura como rechazada con más
# votos a favor que en contra.
def recuento_de(voto_texto: str):
    v = _n(voto_texto)
    frase = v.split("(votos emitidos")[0] if "(votos emitidos" in v else ""
    if not frase.strip():
        return None
    if re.search(r"\benmiendas?\b", frase) and not re.search(r"\b(proposicion|propuesta|mocion)\b.{0,40}\b(presentada|formulada|del grupo)", frase):
        return "enmienda"
    if re.search(r"\b(proposicion|propuesta|mocion|proyecto|dictamen|presupuesto)", frase):
        return "proposicion"
    return None


# Resultado de un punto que quedó sin resultado, a partir de su texto de
# votación. Solo con frases explícitas sobre el propio punto, o con un recuento
# completo (los votos suman los emitidos) de un punto SIN enmiendas: con
# enmiendas, el recuento puede ser el de una de ellas.
def resultado_recuperado(voto_texto: str, tiene_enmienda: bool):
    v = _n(voto_texto)
    frase = v.split("(votos emitidos")[0]
    if re.search(r"\bdeca(e|en)\b.{0,60}\b(proposicion|mocion)", frase):
        return "Decae", "votoTexto"
    if not re.search(r"\benmiendas?\b", frase):
        if re.search(r"(se rechaza|queda rechazad[ao]|rechazad[ao])\s+(la |el )?(propuesta|proposicion|proyecto|mocion)", frase):
            return "Rechazada", "votoTexto"
        if re.search(r"(se aprueba|queda aprobad[ao]|se acepta|queda aceptad[ao])\s+(la |el )?"
                     r"(propuesta|proposicion|proyecto|dictamen|mocion|modificacion|presupuesto)", frase):
            return "Aprobada", "votoTexto"
    m = re.match(r"\s*votos emitidos:\s*(\d+)\s*\|\s*a favor:\s*(\d+)(?:,\s*en contra:\s*(\d+))?(?:,\s*abstenciones:\s*(\d+))?", v)
    if m and not tiene_enmienda:
        emitidos, fav, con, abst = (int(x or 0) for x in m.groups())
        if fav + con + abst == emitidos and fav != con:
            return ("Aprobada" if fav > con else "Rechazada"), "recuento"
    return None


# =============================================================================
# Votaciones, barrios, importes e intervenciones
# =============================================================================

# Información que el grafo no tenía, añadida en enriquecer.py antes del razonador:
#
#   - Votaciones (bo:Votacion): cada "Se somete a votación ... En su virtud, ..."
#     de cada punto, con su objeto (enmienda, el punto, por puntos), recuento,
#     decisión y voto de cada grupo deducido de las listas nominales. Antes solo
#     había una votación por punto, y en 150 puntos era la de una enmienda.
#   - Barrio y distrito (bo:enBarrio, bo:enDistrito): los barrios de Bilbao que
#     nombran el título, el resumen o las entidades del punto, con su distrito
#     oficial (8 distritos).
#   - Importes (bo:importe, bo:beneficiario): la cantidad en euros y quién la
#     recibe, cuando el título la da ("subvención nominativa a favor de Cruz Roja
#     Española por importe de ... (275.000,00 €)").
#   - Fecha de presentación de las iniciativas vecinales (bo:fechaPresentacion).

BO = Namespace("http://bilbao.tfg/ontology#")
BR = Namespace("http://bilbao.tfg/resource/")
VOTACIONES = dato_grafo("votaciones.jsonl")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _n(s)).strip("_")


# --- votaciones ---

def anadir_votaciones(g, ejecutar, prefijos) -> dict:
    if not os.path.exists(VOTACIONES):
        return {"sin_fichero": 1}
    conc = V._concejales(g, ejecutar, prefijos)
    alcalde = "http://bilbao.tfg/resource/grupo_eaj_pnv"
    cuenta = defaultdict(int)
    with open(VOTACIONES, encoding="utf-8") as f:
        filas = [json.loads(l) for l in f if l.strip()]
    for fila in filas:
        p = BR[f"prop_{fila['id']}"]
        if (p, RDF.type, BO.PuntoOrdenDia) not in g or not fila["votaciones"]:
            continue
        # lo que no es una proposición no se vota: si su texto trae votaciones,
        # son de un punto siguiente que el segmentador no separó
        if (p, RDF.type, BO.Proposicion) not in g:
            cuenta["descartadas_no_proposicion"] += len(fila["votaciones"])
            continue
        fecha = str(g.value(p, BO.fechaISO) or "")
        vs = fila["votaciones"]
        for v in vs:
            nodo = BR[f"votacion_{fila['id']}_{v['orden']}"]
            g.add((nodo, RDF.type, BO.Votacion))
            g.add((p, BO.tieneVotacion, nodo))
            g.add((nodo, BO.ordenVotacion, Literal(v["orden"])))
            g.add((nodo, BO.objetoVotacion, Literal(v["objeto"])))
            g.add((nodo, RDFS.label, Literal(v["descripcion"][:300])))
            if v.get("decision"):
                g.add((nodo, BO.decision, Literal(v["decision"][:400])))
            for k, pred in (("emitidos", BO.votosEmitidosVotacion), ("favor", BO.votosFavorVotacion),
                            ("contra", BO.votosContraVotacion), ("abst", BO.abstencionesVotacion)):
                if v.get(k) is not None:
                    g.add((nodo, pred, Literal(int(v[k]), datatype=XSD.integer)))
            # voto de cada grupo: el sentido de al menos el 75 % de sus concejales
            # reconocidos en las listas nominales (misma regla que votos.py)
            listas = v.get("listas") or {}
            por_grupo = defaultdict(lambda: defaultdict(int))
            for sentido in ("favor", "contra", "abstencion"):
                for nombre in listas.get(sentido) or []:
                    gr = V._grupo(V._norm(nombre), fecha, conc, alcalde)
                    if gr:
                        por_grupo[gr][sentido] += 1
            pred = {"favor": BO.grupoVotaAFavor, "contra": BO.grupoVotaEnContra, "abstencion": BO.grupoSeAbstiene}
            for gr, c in por_grupo.items():
                sentido, n = max(c.items(), key=lambda x: x[1])
                if n >= 0.75 * sum(c.values()):
                    g.add((nodo, pred[sentido], URIRef(gr)))
            cuenta["votaciones"] += 1
            cuenta[f"objeto_{v['objeto']}"] += 1
        # la que decide la suerte del punto: la última
        g.add((BR[f"votacion_{fila['id']}_{vs[-1]['orden']}"], BO.esDecisiva, Literal(True)))
        cuenta["puntos_con_votaciones"] += 1
    return dict(cuenta)


# --- barrios y distritos (los 8 distritos oficiales de Bilbao) ---

DISTRITOS = {
    1: ("Deusto", ["Deusto", "Arangoiti", "Ibarrekolanda", "San Ignacio", "Elorrieta", "San Pedro de Deusto",
                   "La Ribera de Deusto", "Zorrotzaurre", "Zorrozaurre", "Sarriko"]),
    2: ("Uribarri", ["Uribarri", "Castaños", "Matiko", "Ciudad Jardín", "Zurbaran", "Arabella", "Artxanda"]),
    3: ("Otxarkoaga-Txurdinaga", ["Otxarkoaga", "Txurdinaga"]),
    4: ("Begoña", ["Begoña", "Santutxu", "Bolueta"]),
    5: ("Ibaiondo", ["Casco Viejo", "Zazpikaleak", "Siete Calles", "Bilbao la Vieja", "San Francisco", "Zabala",
                     "Miribilla", "San Adrián", "La Peña", "Atxuri", "Solokoetxe", "Iturralde", "Abusu"]),
    6: ("Abando", ["Abando", "Indautxu", "Uribitarte"]),
    7: ("Errekalde", ["Errekalde", "Rekalde", "Errekaldeberri", "Larraskitu", "Iralabarri", "Irala", "Amezola",
                      "Ametzola", "Iturrigorri", "Peñascal", "Uretamendi", "Betolaza", "Torre Urizar"]),
    8: ("Basurto-Zorrotza", ["Basurto", "Zorrotza", "Zorroza", "Altamira", "Masustegi", "Monte Caramelo",
                             "Olabeaga"]),
}
# nombre del barrio en el grafo (las variantes van al mismo nodo)
_CANON = {"Zorrozaurre": "Zorrotzaurre", "Zazpikaleak": "Casco Viejo", "Siete Calles": "Casco Viejo",
          "Rekalde": "Errekalde", "Ametzola": "Amezola", "Zorroza": "Zorrotza", "Irala": "Iralabarri",
          "La Ribera de Deusto": "San Pedro de Deusto"}


# Nombres de barrio que también son palabras o nombres de persona: se buscan en
# el texto original, con mayúscula y sin un apellido detrás ("merece la pena",
# "doña Begoña Marbán", "Jon Zabala Basterra", "calle San Francisco Javier")
_AMBIGUOS = {
    "La Peña": r"\b(?:La|la) Peña\b(?!\s+[A-ZÁÉÍÓÚ])",
    "Begoña": r"(?<!doña )(?<!Dña\. )\bBegoña\b(?!\s+[A-ZÁÉÍÓÚ][a-záéíóúñ])",
    "Zabala": r"(?<!Jon )\bZabala\b(?!\s+[A-ZÁÉÍÓÚ][a-záéíóúñ])",
    "Iturralde": r"\bIturralde\b(?!\s+[A-ZÁÉÍÓÚ][a-záéíóúñ])",
    "San Francisco": r"\bSan Francisco\b(?!\s+(?:Javier|de Asís|[A-ZÁÉÍÓÚ][a-záéíóúñ]))",
    "Castaños": r"\bCastaños\b",
    "Abando": r"\bAbando\b",
}


def _patrones_barrios():
    out = []
    for num, (dist, barrios) in DISTRITOS.items():
        for b in barrios:
            canon = _CANON.get(b, b)
            if b in _AMBIGUOS:
                out.append((re.compile(_AMBIGUOS[b]), canon, num, dist, True))
            else:
                out.append((re.compile(rf"\b{re.escape(_n(b))}\b"), canon, num, dist, False))
    return out


def anadir_barrios(g) -> dict:
    pats = _patrones_barrios()
    cuenta = defaultdict(int)
    for num, (dist, _) in DISTRITOS.items():
        d = BR[f"distrito_{num}"]
        g.add((d, RDF.type, BO.Distrito))
        g.add((d, RDFS.label, Literal(f"Distrito {num} {dist}")))
    for p in set(g.subjects(RDF.type, BO.PuntoOrdenDia)):
        partes = [str(g.value(p, BO.tituloTopic) or ""), str(g.value(p, BO.resumen) or "")]
        partes += [str(g.value(e, RDFS.label) or "") for e in g.objects(p, BO.menciona)]
        original = re.sub(r"\s+", " ", " | ".join(partes))
        texto = _n(original)
        vistos = set()
        for rx, canon, num, dist, en_original in pats:
            if canon in vistos or not rx.search(original if en_original else texto):
                continue
            # "Universidad de Deusto", "Hospital de Basurto", "Metro Bilbao Abando"...
            # siguen estando en su barrio: no se excluyen
            vistos.add(canon)
            b = BR[f"barrio_{_slug(canon)}"]
            g.add((b, RDF.type, BO.Barrio))
            g.set((b, RDFS.label, Literal(canon)))
            g.set((b, BO.barrioEnDistrito, BR[f"distrito_{num}"]))
            g.add((p, BO.enBarrio, b))
            cuenta["enlaces"] += 1
        if vistos:
            cuenta["puntos"] += 1
    return dict(cuenta)


# --- importes y beneficiarios ---

_IMPORTE = re.compile(r"(\d{1,3}(?:[.\s]\d{3})+|\d{4,})(?:,(\d{1,2}))?\s?(?:€|euros?\b|eur\b)", re.I)
_BENEF = re.compile(r"(?:a\s+favor\s+de(?:l)?|nominativas?\s+a(?:l)?|subvenci[óo]n\s+(?:directa\s+)?a(?:l)?)\s+"
                    r"(?:la\s+|el\s+|los\s+|las\s+)?([A-ZÁÉÍÓÚÑ\"“][^,(]{2,140}?)\s*(?:,|\(|por\s+(?:un\s+)?importe|para|con\s+destino|$)")


def _euros(m) -> float:
    entero = re.sub(r"[.\s]", "", m.group(1))
    return float(entero + "." + (m.group(2) or "0"))


def anadir_importes(g) -> dict:
    cuenta = defaultdict(int)
    for p in set(g.subjects(RDF.type, BO.PuntoOrdenDia)):
        titulo = re.sub(r"\s+", " ", str(g.value(p, BO.tituloTopic) or ""))
        cifras = [_euros(m) for m in _IMPORTE.finditer(titulo)]
        cifras = [c for c in cifras if c >= 100]
        if not cifras:
            continue
        # la del "importe de ..." si la hay; si no, la mayor
        tras = re.search(r"importe\s+(?:total\s+)?de", titulo, re.I)
        principal = next((_euros(m) for m in _IMPORTE.finditer(titulo) if tras and m.start() > tras.start()), max(cifras))
        g.add((p, BO.importe, Literal(round(principal, 2), datatype=XSD.decimal)))
        cuenta["con_importe"] += 1
        b = _BENEF.search(titulo)
        if b:
            nombre = b.group(1).strip(" \"“”.")
            if 3 <= len(nombre) <= 140:
                g.add((p, BO.beneficiario, Literal(nombre)))
                cuenta["con_beneficiario"] += 1
    return dict(cuenta)


# --- iniciativas vecinales: fecha de presentación ---

_MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
          "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}


def anadir_fecha_presentacion(g) -> int:
    n = 0
    for p in set(g.subjects(BO.tipoPunto, Literal("proposicion_ciudadana"))) | set(g.subjects(BO.tipoPunto, Literal("informe_iniciativa"))):
        t = _n(str(g.value(p, BO.tituloTopic) or ""))
        m = re.search(r"(?:de fecha|presentada (?:el(?: dia)?|en fecha))\s+(\d{1,2}) de (\w+)(?: de (\d{4}))?", t)
        if not m or m.group(2) not in _MESES:
            continue
        anio = m.group(3) or str(g.value(p, BO.anio) or "")
        if not anio.isdigit():
            continue
        g.add((p, BO.fechaPresentacion, Literal(f"{anio}-{_MESES[m.group(2)]:02d}-{int(m.group(1)):02d}",
                                                 datatype=XSD.date)))
        n += 1
    return n


# Una bo:Intervencion por cada intervención de intervenciones.jsonl, con el resumen
# y la postura de intervenciones_resumen.jsonl (debates.py). El orador
# se resuelve a un concejal por apellido (y por fecha si el apellido es ambiguo);
# el alcalde figura con su grupo.
def anadir_intervenciones(g, ejecutar, prefijos, copia_de) -> dict:
    ruta, ruta_res = dato_grafo("intervenciones.jsonl"), dato_grafo("intervenciones_resumen.jsonl")
    if not (os.path.exists(ruta) and os.path.exists(ruta_res)):
        return {"intervenciones": 0}
    conc = V._concejales(g, ejecutar, prefijos)
    resumen = {}
    with open(ruta_res, encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                r = json.loads(linea)
                resumen[r["id"]] = r
    fechas = {str(p): str(d) for p, d in g.subject_objects(BO.fechaISO)}
    cuenta = defaultdict(int)
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            r = json.loads(linea)
            res = resumen.get(r["id"])
            if not res:
                cuenta["sin_resumen"] += 1
                continue
            pid = copia_de.get(r["punto"], r["punto"])
            pr = BR[f"prop_{pid}"]
            if (pr, RDF.type, BO.PuntoOrdenDia) not in g:
                cuenta["punto_ausente"] += 1
                continue
            iv = BR[f"interv_{r['id']}"]
            g.add((iv, RDF.type, BO.Intervencion))
            g.add((iv, BO.intervencionEn, pr))
            g.add((iv, BO.oradorEtiqueta, Literal(r["orador"])))
            g.add((iv, BO.ordenIntervencion, Literal(int(r["orden"]), datatype=XSD.integer)))
            g.add((iv, BO.longitudIntervencion, Literal(len(r["texto"]), datatype=XSD.integer)))
            g.add((iv, BO.resumenIntervencion, Literal(res["resumen"])))
            g.add((iv, BO.posturaIntervencion, Literal(res["postura"])))
            ap, fecha = r["apellidos"], fechas.get(str(pr), "")
            grupo = None
            if ap == "alcalde":
                g.add((iv, BO.rolOrador, Literal("alcalde")))
                grupo = BR.grupo_eaj_pnv
            elif ap == "secretario":
                g.add((iv, BO.rolOrador, Literal("secretario")))
            else:
                cands = [u for u, c in conc.items() if ap in c["claves"]]
                if len(cands) > 1:
                    cands = [u for u in cands if conc[u]["ini"] <= fecha <= conc[u]["fin"]]
                if len(cands) == 1:
                    g.add((iv, BO.orador, URIRef(cands[0])))
                    grupo = URIRef(conc[cands[0]]["grupo"])
                else:
                    grupos = {conc[u]["grupo"] for u in cands}
                    if len(grupos) == 1:
                        grupo = URIRef(grupos.pop())
            if grupo is not None:
                g.add((iv, BO.grupoOrador, grupo))
                cuenta["con_grupo"] += 1
            if (iv, BO.orador, None) in g:
                cuenta["con_concejal"] += 1
            cuenta["intervenciones"] += 1
    return dict(cuenta)


# =============================================================================
# Enriquecimiento del grafo
# =============================================================================

# Enriquecimiento del grafo: se aplica en build_rdf.py después de cargar las
# proposiciones y ANTES del razonador, y queda guardado en bilbao_reasoned.ttl.
#
# Hasta §7.16 estos pasos se hacían en memoria cada vez que GraphRAG cargaba el
# grafo, así que el grafo que se consultaba no estaba en ningún fichero. Ahora el
# .ttl es exactamente el grafo que se consulta:
#
#   1. Copias en euskera: las actas bilingües traen cada punto dos veces y el
#      grafo guardaba las dos como proposiciones distintas (692 copias que
#      inflaban los recuentos). Se quita la copia cuando existe la versión en
#      castellano del mismo punto; la castellana guarda el id de la copia
#      (bo:copiaEuskera) porque los fragmentos del texto apuntan a ella.
#   2. Fechas: bo:fecha es un texto "DD-MM-AAAA" que no se puede ordenar; se
#      añaden bo:fechaISO (xsd:date) y bo:mes.
#   3. Voto por grupo desde las votaciones nominales del texto
#      (votos.py), marcado con bo:votoFuente "acta_nominal".
#   4. Plenos: un bo:Pleno por acta (antes, uno por fecha: dos sesiones del
#      mismo día eran el mismo pleno), con su tipo (ordinaria/extraordinaria),
#      fecha y acta, incluidas las sesiones sin ninguna proposición
#      (constitución de la corporación, actos institucionales).
#   5. Legislaturas: bo:Legislatura, de un pleno de constitución al siguiente,
#      y cada pleno en la suya (el razonador la propaga a las proposiciones).
#
# Los plenos y las legislaturas usan propiedades propias: bo:fechaISO,
# bo:fuentePdf... tienen rdfs:domain bo:Proposicion, y el razonador deduciría
# que cada pleno es una proposición.

# pleno de constitución de cada corporación (actas/); la de 2003 es anterior al
# corpus, pero los plenos de enero a junio de 2007 son de esa legislatura
CONSTITUCION = {2003: "2003-06-14", 2007: "2007-06-16", 2011: "2011-06-11", 2015: "2015-06-13",
                2019: "2019-06-15", 2023: "2023-06-17"}

_EUSKERA_RE = re.compile(r"^\W*(\d+)\W*(proposizioa|proposamena|proposatzen)", re.I)


def _pid(uri) -> str:
    return str(uri).rsplit("prop_", 1)[-1]


# 1. copia en euskera -> versión en castellano del mismo punto (misma fecha y número)
def quitar_copias_euskera(g) -> dict:
    puntos = defaultdict(list)
    for p, t in g.subject_objects(BO.tituloTopic):
        m = re.match(r"^\W*(\d+)", str(t))
        puntos[(str(g.value(p, BO.fecha)), m.group(1) if m else None)].append((p, str(t)))
    copia_de = {}
    for grupo in puntos.values():
        castellano = [p for p, t in grupo if not _EUSKERA_RE.search(t)]
        if not castellano:
            continue
        for p, t in grupo:
            if _EUSKERA_RE.search(t):
                copia_de[_pid(p)] = _pid(castellano[0])
                g.add((castellano[0], BO.copiaEuskera, Literal(_pid(p))))
                g.remove((p, None, None))
                g.remove((None, None, p))
    return copia_de


# 2. bo:fechaISO y bo:mes de cada proposición
def anadir_fechas(g):
    for s, f in list(g.subject_objects(BO.fecha)):
        m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", str(f))
        if m:
            d, mes, a = m.groups()
            g.add((s, BO.fechaISO, Literal(f"{a}-{mes}-{d}", datatype=XSD.date)))
            g.add((s, BO.mes, Literal(int(mes))))


def _ejecutar(g, q):
    return [{str(k): (str(v) if v is not None else None) for k, v in r.asdict().items()} for r in g.query(q)]


_PREFIJOS = """PREFIX bo: <http://bilbao.tfg/ontology#>
PREFIX br: <http://bilbao.tfg/resource/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


# Correcciones comprobadas leyendo el acta, una a una (mismo criterio que
# build_rdf.py::_RESULTADO_ENMIENDA_NORM): solo las que contradicen el propio
# recuento del grafo y se han verificado en el texto.
CORRECCIONES = {
    # 19-09-2008, punto 26 (PSE-EE): "se aprueba la Enmienda del Equipo de
    # Gobierno, por lo que decae la proposición del Grupo Municipal PSE-EE"
    # (acta, pág. 175). Figuraba Aprobada, con el recuento 14/15 de la
    # proposición nº 29 del PP, que se votó a continuación.
    "0acc2ae9c71abb6d": {"resultado": "Decae", "quitar_votos": True},
    # 24-09-2020, punto 21 (EH BILDU): el 6 a favor / 23 en contra es la
    # votación de la enmienda de adición de ELKARREKIN BILBAO (acta, pág. 113),
    # no la de la proposición.
    "d2aeb50df46caae3": {"votacion": "enmienda"},
}
_VOTOS = ("votosFavor", "votosContra", "votoTexto", "votoAFavorDe", "votoEnContraDe", "seAbstuvo",
          "concejalVotoAFavor", "concejalVotoEnContra", "concejalVotoAbstencion", "votoFuente")


# Qué es cada punto (bo:tipoPunto), quién presenta las proposiciones
# ciudadanas, de qué votación es el recuento guardado y el resultado de los
# puntos sin resultado que lo dicen en su texto de votación (primera sección)
def clasificar_puntos(g) -> dict:
    from rdflib.namespace import RDFS as _RDFS
    cuenta = defaultdict(int)
    desconocido = BR.grupo_desconocido
    ciudadanas = {}
    for p in set(g.subjects(RDF.type, BO.Proposicion)):
        titulo = str(g.value(p, BO.tituloTopic) or "")
        grupo_uri = g.value(p, BO.presentadaPor)
        grupo = str(g.value(grupo_uri, _RDFS.label) or "Desconocido") if grupo_uri else "Desconocido"
        resultado = g.value(p, BO.tieneResultado)
        tipo = tipo_punto(titulo, grupo)
        recuperado = (p, BO.origenExtraccion, Literal("segmentar")) in g
        # Una dación de cuenta ("Se da cuenta de la resolución de Alcaldía...",
        # "Toma de conocimiento...") es informativa: no se vota. El resultado o
        # el recuento que traía eran del LLM o de la votación del punto
        # siguiente, que el troceo no había separado (en §7.18 las que tenían
        # resultado se tomaban por propuestas; salían "aprobadas" del PP o de EH
        # Bildu resoluciones de Alcaldía que solo designaban representantes).
        if tipo == "dacion_cuenta" and resultado is not None and resultado != BO.SinResultado:
            g.set((p, BO.tieneResultado, BO.SinResultado))
            resultado = BO.SinResultado
        g.add((p, BO.tipoPunto, Literal(tipo)))
        g.add((p, RDF.type, BO.PuntoOrdenDia))
        # una propuesta recuperada es del gobierno salvo que el título nombre a
        # otro grupo (el grupo salía del texto: "designar representante del
        # Grupo Municipal EH BILDU" en una resolución de Alcaldía)
        if recuperado and tipo == "propuesta_gobierno" and not _GRUPO_EN_TITULO.search(_n(titulo)[:200]):
            g.remove((p, BO.presentadaPor, None))
            g.add((p, BO.presentadaPor, BR.grupo_equipo_de_gobierno))
            cuenta["gobierno_reasignada"] += 1
        # "PROPUESTA de resolución de las enmiendas presentadas por los grupos municipales
        # ELKARREKIN BILBAO, EH BILDU y PARTIDO POPULAR": es la respuesta del gobierno a las
        # enmiendas, no una proposición del primer grupo que nombra el título; igual las
        # comunicaciones de Alcaldía sin votación (sustituciones de concejales) que citan a un grupo
        elif (tipo == "propuesta_gobierno" and grupo not in ("Desconocido", "EQUIPO DE GOBIERNO")
              and (_RESOLUCION_GOBIERNO.search(_n(titulo)[:200]) or resultado == BO.SinResultado)):
            g.remove((p, BO.presentadaPor, None))
            g.add((p, BO.presentadaPor, BR.grupo_equipo_de_gobierno))
            cuenta["gobierno_reasignada"] += 1
        # una pregunta, una dación de cuenta o un trámite no son proposiciones:
        # cualquier consulta con "?p a bo:Proposicion" (también las del LLM) los deja fuera
        if tipo in NO_PROPOSICION:
            g.remove((p, RDF.type, BO.Proposicion))
        cuenta[tipo] += 1
        # (también las entradas de la memoria de la Comisión de Sugerencias)
        if tipo in ("proposicion_ciudadana", "informe_iniciativa"):
            # la presenta una asociación o un particular, no un grupo
            if grupo_uri != desconocido:
                g.remove((p, BO.presentadaPor, None))
                g.add((p, BO.presentadaPor, desconocido))
                cuenta["ciudadana_reasignada"] += 1
            e = entidad_ciudadana(titulo)
            if e:
                ciudadanas[p] = e
        # de qué votación es el recuento guardado
        voto = str(g.value(p, BO.votoTexto) or "")
        vot = recuento_de(voto)
        if vot:
            g.add((p, BO.votacionRegistrada, Literal(vot)))
            cuenta[f"votacion_{vot}"] += 1
        # resultado de los puntos que lo dicen en su votación
        if resultado == BO.SinResultado and tipo not in NO_PROPOSICION:
            rec = resultado_recuperado(voto, (p, BO.tieneEnmienda, None) in g)
            if rec:
                g.set((p, BO.tieneResultado, getattr(BO, rec[0])))
                g.set((p, BO.resultadoFuente, Literal(rec[1])))
                cuenta["resultado_recuperado"] += 1

    # quién presenta las ciudadanas: la organización (con sus variantes
    # reunidas en un nodo) o, si no hay, la persona
    claves = {p: clave_organizacion(e[0]) for p, e in ciudadanas.items() if e[1] == "organizacion"}
    canon = unificar_claves(claves.values())
    nombres = defaultdict(list)
    for p, k in claves.items():
        nombres[canon[k]].append(ciudadanas[p][0])
    for p, (nombre, tipo) in ciudadanas.items():
        if tipo == "organizacion":
            k = canon[claves[p]]
            ent = BR[f"org_{k}"]
            g.add((ent, RDF.type, BO.Organizacion))
            # la variante más completa como nombre; a igual longitud, la que no va toda en
            # mayúsculas y después la alfabética (antes el empate dependía del orden en que
            # se recorría el grafo y la etiqueta cambiaba de una reconstrucción a otra)
            g.set((ent, RDFS.label, Literal(max(nombres[k], key=lambda n: (len(n.split()) <= 12, len(n),
                                                                           not n.isupper(), n)))))
        else:
            ent = BR["persona_" + clave_organizacion(nombre)]
            g.add((ent, RDF.type, BO.Persona))
            g.set((ent, RDFS.label, Literal(nombre)))
        g.remove((p, BO.presentadaPorParticular, None))
        g.add((p, BO.presentadaPorParticular, ent))
        cuenta["ciudadana_con_quien"] += 1

    for pid, c in CORRECCIONES.items():
        p = BR[f"prop_{pid}"]
        if (p, RDF.type, BO.PuntoOrdenDia) not in g:
            print(f"[!] corrección sin proposición: {pid}")
            continue
        if c.get("quitar_votos"):
            for pred in _VOTOS:
                g.remove((p, BO[pred], None))
            g.remove((p, BO.votacionRegistrada, None))
        if c.get("resultado"):
            g.set((p, BO.tieneResultado, getattr(BO, c["resultado"])))
            g.set((p, BO.resultadoFuente, Literal("acta_verificada")))
        if c.get("votacion"):
            g.set((p, BO.votacionRegistrada, Literal(c["votacion"])))
        cuenta["correcciones"] += 1
    return dict(cuenta)


# 3. voto por grupo de las votaciones nominales del texto
def anadir_votos_nominales(g, corpus, copia_de) -> int:
    from grafo.construccion.votos import anadir
    return anadir(g, _ejecutar, _PREFIJOS, corpus, lambda pid: copia_de.get(pid, pid))


def _legislatura(iso: str) -> int:
    return max(a for a, f in CONSTITUCION.items() if f <= iso)


# 4 y 5. un bo:Pleno por acta y las legislaturas
def anadir_plenos(g, corpus) -> int:
    actas = {}   # nombre del pdf -> ruta como la guarda el grafo (o el nombre, si solo está en el texto)
    for p, ruta in g.subject_objects(BO.fuentePdf):
        actas.setdefault(os.path.basename(str(ruta).replace("\\", "/")), str(ruta))
    for d in corpus:
        if d[8]:
            actas.setdefault(d[8], d[8])
    por_fecha = defaultdict(list)
    for pdf in actas:
        m = re.match(r"(\d{2})-(\d{2})-(\d{4})", pdf)
        if m:
            por_fecha[m.group(0)].append(pdf)

    # las legislaturas
    anios = sorted(CONSTITUCION)
    for a in anios:
        lg = BR[f"legislatura_{a}_{a + 4}"]
        g.add((lg, RDF.type, BO.Legislatura))
        g.add((lg, RDFS.label, Literal(f"Legislatura {a}-{a + 4}")))
        g.add((lg, BO.inicioLegislatura, Literal(CONSTITUCION[a], datatype=XSD.date)))
        if a + 4 in CONSTITUCION:
            g.add((lg, BO.finLegislatura, Literal(CONSTITUCION[a + 4], datatype=XSD.date)))

    uri_de = {}
    for fecha, pdfs in por_fecha.items():
        # el pleno ordinario conserva el URI por fecha de siempre (pleno_DD_MM_AAAA);
        # si ese día hubo otra sesión, va con sufijo
        pdfs.sort(key=lambda x: ("Extraordinaria" in x, x))
        for i, pdf in enumerate(pdfs):
            base = "pleno_" + fecha.replace("-", "_")
            if i:
                base += "_" + ("extraordinaria" if "Extraordinaria" in pdf else "sesion") + (f"_{i}" if i > 1 else "")
            uri_de[pdf] = BR[base]
    for pdf, pl in uri_de.items():
        d, mes, a = re.match(r"(\d{2})-(\d{2})-(\d{4})", pdf).groups()
        iso = f"{a}-{mes}-{d}"
        tipo = "extraordinaria" if "Extraordinaria" in pdf else "ordinaria"
        g.set((pl, RDFS.label, Literal(f"{d}-{mes}-{a}" + (" (extraordinaria)" if pl.endswith("extraordinaria") else ""))))
        g.add((pl, RDF.type, BO.Pleno))
        g.set((pl, BO.anio, Literal(int(a), datatype=XSD.integer)))
        g.add((pl, BO.fechaSesion, Literal(iso, datatype=XSD.date)))
        g.add((pl, BO.mesSesion, Literal(int(mes))))
        g.add((pl, BO.tipoSesion, Literal(tipo)))
        g.add((pl, BO.actaPdf, Literal(pdf)))
        g.add((pl, BO.enLegislatura, BR[f"legislatura_{_legislatura(iso)}_{_legislatura(iso) + 4}"]))

    # cada proposición, al pleno de SU acta
    for p, ruta in list(g.subject_objects(BO.fuentePdf)):
        pl = uri_de.get(os.path.basename(str(ruta).replace("\\", "/")))
        if pl is not None:
            g.remove((p, BO.enPleno, None))
            g.add((p, BO.enPleno, pl))
    # plenos por fecha que se han quedado sin proposiciones ni acta (no debería haber)
    for pl in set(g.subjects(RDF.type, BO.Pleno)):
        if g.value(pl, BO.actaPdf) is None and not any(g.subjects(BO.enPleno, pl)):
            g.remove((pl, None, None))
    return len(uri_de)


# Después del razonador. OWL-RL materializa la igualdad reflexiva: cada nodo y
# cada literal "es igual a sí mismo" (owl:sameAs), y trata los literales como
# sujetos ("portal e-zinegotzi" owl:sameAs "portal e-zinegotzi"), que no es RDF
# válido (no se puede guardar en N-Triples). Eran unos 60.000 triples, un 25 %
# del fichero, sin ninguna información.
def limpiar_razonado(g) -> int:
    from rdflib.namespace import OWL
    basura = [t for t in g if isinstance(t[0], Literal) or (t[1] == OWL.sameAs and t[0] == t[2])]
    for t in basura:
        g.remove(t)
    return len(basura)


def enriquecer(g):
    corpus = T.corpus()
    copia_de = quitar_copias_euskera(g)
    print(f"[*] copias en euskera quitadas: {len(copia_de)}")
    anadir_fechas(g)
    print(f"[*] puntos: {clasificar_puntos(g)}")
    print(f"[*] resultado corregido con la votación del acta: {V.corregir_resultados(g, copia_de)}")
    print(f"[*] autor de enmiendas sin grupo: {V.asignar(g)}")
    n = anadir_votos_nominales(g, corpus, copia_de)
    print(f"[*] voto por grupo completado desde votaciones nominales: {n} proposiciones")
    print(f"[*] votaciones separadas: {anadir_votaciones(g, _ejecutar, _PREFIJOS)}")
    print(f"[*] recuentos de votación corregidos con el acta: "
          f"{V.corregir_recuentos(g, corpus, lambda pid: copia_de.get(pid, pid), _ejecutar, _PREFIJOS)}")
    print(f"[*] barrios y distritos: {anadir_barrios(g)}")
    print(f"[*] importes: {anadir_importes(g)}")
    print(f"[*] fecha de presentación de iniciativas vecinales: {anadir_fecha_presentacion(g)}")
    print(f"[*] intervenciones: {anadir_intervenciones(g, _ejecutar, _PREFIJOS, copia_de)}")
    print(f"[*] nombres propios de los debates (bo:nombradoEnDebate): {D.anadir(g, corpus, copia_de)}")
    n = anadir_plenos(g, corpus)
    print(f"[*] plenos (uno por acta): {n}")
    g.add((BR.grafo, BO.enriquecido, Literal(True)))
