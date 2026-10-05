"""
Corrección del SPARQL que genera el LLM:

  1. vocabulario real del grafo (resultados, grupos, temas), para corregir lo
     que el LLM se inventa y detectar propiedades o temas que no existen
  2. guardas deterministas contra los errores que se repiten, cada una
     documentada con su caso
"""
import re
import threading
import unicodedata

from comun.rutas import TEMAS_SKOS


# =============================================================================
# 1. Vocabulario real del grafo
# =============================================================================

_RESULT_INDIVIDUALS = ("bo:Aprobada", "bo:Rechazada", "bo:Decae", "bo:Retirada",
                       "bo:AprobadaConEnmienda", "bo:SinResultado")


# rdfs:label reales de los grupos (literales planos, sin lang tag).
_GRUPO_LABELS = {
    "PP", "EH BILDU", "PSE-EE", "ELKARREKIN BILBAO", "GOAZEN BILBAO", "UDALBERRI",
    "EZKER BATUA-IU", "EAJ-PNV", "CIUDADANOS", "EQUIPO DE GOBIERNO", "GRUPO MIXTO",
}


_GRUPO_ALIAS = {
    "partido popular": "PP", "popular": "PP", "pp": "PP",
    "partido socialista": "PSE-EE", "socialistas vascos": "PSE-EE", "pse": "PSE-EE",
    "pse-ee": "PSE-EE", "psoe": "PSE-EE",
    "partido nacionalista vasco": "EAJ-PNV", "pnv": "EAJ-PNV", "eaj": "EAJ-PNV",
    "eaj-pnv": "EAJ-PNV", "jeltzale": "EAJ-PNV",
    "bildu": "EH BILDU", "eh bildu": "EH BILDU", "euskal herria bildu": "EH BILDU",
    "elkarrekin": "ELKARREKIN BILBAO", "elkarrekin bilbao": "ELKARREKIN BILBAO",
    "podemos": "ELKARREKIN BILBAO",
    "goazen": "GOAZEN BILBAO", "goazen bilbao": "GOAZEN BILBAO",
    "udalberri": "UDALBERRI", "bilbao en comun": "UDALBERRI", "bilbao en común": "UDALBERRI",
    "ezker batua": "EZKER BATUA-IU", "izquierda unida": "EZKER BATUA-IU", "iu": "EZKER BATUA-IU",
    "ciudadanos": "CIUDADANOS", "equipo de gobierno": "EQUIPO DE GOBIERNO",
    "gobierno municipal": "EQUIPO DE GOBIERNO", "grupo mixto": "GRUPO MIXTO",
}


_CANON_TEMAS = (
    "vivienda", "urbanismo", "movilidad", "medioambiente", "euskera", "cultura",
    "deporte", "educacion", "igualdad", "serviciossociales", "empleoeconomia",
    "presupuestos", "seguridad", "participacion", "turismo", "sanidad",
    "memoriahistorica", "derechoshumanos", "otros",
)


# URIs de tema REALES (nivel 1 + subtemas) de themes_skos.ttl — para detectar
# cuándo el LLM inventa un slug (br:t_movilidad_y_transporte, bo:t_seguridad...)
# que da 0 resultados en silencio. Si themes_skos.ttl cambia, este set también.
_REAL_TEMA_URIS = frozenset((
    "t_accesibilidad", "t_alquiler", "t_aparcamiento", "t_arte", "t_barrios",
    "t_bibliotecas", "t_bicicleta", "t_bilbobus", "t_calidad_aire",
    "t_peatonalizacion", "t_comercio", "t_cultura",
    "t_deporte", "t_derechoshumanos", "t_desahucios", "t_discapacidad",
    "t_educacion", "t_empleo", "t_empleoeconomia", "t_empresas", "t_energia",
    "t_espacio_publico", "t_euskera", "t_exclusion_social", "t_fiestas",
    "t_financiacion", "t_hosteleria", "t_igualdad", "t_juventud",
    "t_medioambiente", "t_memoriahistorica", "t_metro_bilbao", "t_movilidad",
    "t_museos", "t_ordenacion_urbana", "t_ordenanzas_fiscales", "t_otros",
    "t_participacion", "t_personas_mayores", "t_policia_municipal",
    "t_presupuesto_municipal", "t_presupuestos", "t_reciclaje",
    "t_rehabilitacion", "t_retribuciones", "t_sanidad", "t_seguridad",
    "t_serviciossociales", "t_subvenciones", "t_transparencia",
    "t_transporte_publico", "t_tranvia", "t_turismo", "t_urbanismo",
    "t_violencia_genero", "t_vivienda", "t_vivienda_social", "t_vivienda_vacia",
))


# etiqueta de tema (prefLabel/altLabel, en minúsculas) -> tema de nivel 1 al que
# pertenece, según themes_skos.ttl. Lo usan las guardas 11 y 12 de más abajo.
_LABEL_TOPLEVEL = {
    "accesibilidad": "t_movilidad", "alquiler": "t_vivienda",
    "alquiler social": "t_vivienda", "alquileres": "t_vivienda",
    "aparcamiento": "t_movilidad", "aparcamientos": "t_movilidad",
    "arte": "t_cultura", "ascensores": "t_movilidad", "aste nagusia": "t_cultura",
    "aurrekontuak eta fiskalitatea": "t_presupuestos", "autobuses": "t_movilidad",
    "autobús": "t_movilidad", "ayudas": "t_presupuestos",
    "ayudas económicas": "t_presupuestos", "ayudas sociales": "t_serviciossociales",
    "barrio": "t_urbanismo", "barrios": "t_urbanismo", "basuras": "t_medioambiente",
    "berdintasuna eta feminismoa": "t_igualdad", "besteak": "t_otros",
    "biblioteca": "t_cultura", "bibliotecas": "t_cultura", "bicicleta": "t_movilidad",
    "bicicletas": "t_movilidad", "bidebarrieta": "t_cultura", "bidegorri": "t_movilidad",
    "bidegorris": "t_movilidad", "bilbobus": "t_movilidad", "bonificaciones": "t_presupuestos",
    "buen gobierno": "t_participacion", "calidad del aire": "t_medioambiente",
    "calles peatonales": "t_movilidad", "carril bici": "t_movilidad",
    "circulación": "t_movilidad", "ciudad 30": "t_movilidad", "comercio": "t_empleoeconomia",
    "comercio local": "t_empleoeconomia", "contaminación atmosférica": "t_medioambiente",
    "contaminación del aire": "t_medioambiente", "cultura": "t_cultura",
    "deporte": "t_deporte", "deportes": "t_deporte", "derechos humanos": "t_derechoshumanos",
    "desahucio": "t_vivienda", "desahucios": "t_vivienda", "desempleo": "t_empleoeconomia",
    "discapacidad": "t_serviciossociales", "distritos": "t_urbanismo",
    "diversidad funcional": "t_serviciossociales", "economía": "t_empleoeconomia",
    "educación": "t_educacion", "eficiencia energética": "t_medioambiente",
    "emisiones": "t_medioambiente", "empleo": "t_empleoeconomia",
    "empleo público": "t_empleoeconomia", "empleo y economía": "t_empleoeconomia",
    "empresa": "t_empleoeconomia", "empresas": "t_empleoeconomia",
    "energética": "t_medioambiente", "energético": "t_medioambiente",
    "energía": "t_medioambiente", "energías renovables": "t_medioambiente",
    "enplegua eta ekonomia": "t_empleoeconomia", "envejecimiento": "t_serviciossociales",
    "ertzaintza": "t_seguridad", "espacio público": "t_urbanismo", "etxebizitza": "t_vivienda",
    "euskara": "t_euskera", "euskera": "t_euskera", "exclusión social": "t_serviciossociales",
    "feminismo": "t_igualdad", "fiestas": "t_cultura", "fiestas populares": "t_cultura",
    "financiación": "t_presupuestos", "fiscalidad": "t_presupuestos",
    "gaztedia": "t_serviciossociales", "giza eskubideak": "t_derechoshumanos",
    "gizarte zerbitzuak": "t_serviciossociales", "herritarren parte-hartzea": "t_participacion",
    "hezkuntza": "t_educacion", "hirigintza": "t_urbanismo", "hostelería": "t_empleoeconomia",
    "igualdad": "t_igualdad", "igualdad de género": "t_igualdad",
    "igualdad de mujeres y hombres": "t_igualdad", "igualdad y feminismo": "t_igualdad",
    "impuestos": "t_presupuestos", "industria": "t_empleoeconomia",
    "ingurumena": "t_medioambiente", "inversiones": "t_presupuestos",
    "joven": "t_serviciossociales", "juvenil": "t_serviciossociales",
    "juveniles": "t_serviciossociales", "juventud": "t_serviciossociales",
    "jóvenes": "t_serviciossociales", "kirola": "t_deporte", "kultura": "t_cultura",
    "limitación de velocidad": "t_movilidad", "limpieza": "t_medioambiente",
    "limpieza viaria": "t_medioambiente", "mayores": "t_serviciossociales",
    "medio ambiente": "t_medioambiente", "medioambiente": "t_medioambiente",
    "memoria historikoa": "t_memoriahistorica", "memoria histórica": "t_memoriahistorica",
    "mercados": "t_empleoeconomia", "metro": "t_movilidad", "metro bilbao": "t_movilidad",
    "movilidad": "t_movilidad", "movilidad y transporte": "t_movilidad",
    "mugikortasuna eta garraioa": "t_movilidad", "mujeres": "t_igualdad",
    "museo": "t_cultura", "museos": "t_cultura", "ordenación urbana": "t_urbanismo",
    "ordenanza fiscal": "t_presupuestos", "ordenanzas fiscales": "t_presupuestos",
    "osasuna": "t_sanidad", "ota": "t_movilidad", "otros": "t_otros",
    "pacificación del tráfico": "t_movilidad", "parking": "t_movilidad",
    "paro": "t_empleoeconomia", "parques y jardines": "t_urbanismo",
    "participación": "t_participacion", "participación ciudadana": "t_participacion",
    "peatonalizaciones": "t_movilidad", "peatonalización": "t_movilidad",
    "personas mayores": "t_serviciossociales", "pgou": "t_urbanismo",
    "plan general": "t_urbanismo", "planeamiento": "t_urbanismo", "plazas": "t_urbanismo",
    "pobreza": "t_serviciossociales", "pobreza energética": "t_medioambiente",
    "policía": "t_seguridad", "policía municipal": "t_seguridad",
    "presupuesto": "t_presupuestos", "presupuesto municipal": "t_presupuestos",
    "presupuestos": "t_presupuestos", "presupuestos municipales": "t_presupuestos",
    "presupuestos y fiscalidad": "t_presupuestos", "pymes": "t_empleoeconomia",
    "reciclaje": "t_medioambiente", "reducción de emisiones": "t_medioambiente",
    "regeneración urbana": "t_urbanismo", "rehabilitación": "t_urbanismo",
    "residuos": "t_medioambiente", "retribuciones": "t_presupuestos",
    "sala de estudio": "t_cultura", "salarios": "t_presupuestos",
    "salas de estudio": "t_cultura", "salud": "t_sanidad", "salud pública": "t_sanidad",
    "sanidad": "t_sanidad", "seguridad": "t_seguridad", "seguridad ciudadana": "t_seguridad",
    "segurtasuna": "t_seguridad", "servicios sociales": "t_serviciossociales",
    "sostenibilidad": "t_medioambiente", "subvenciones": "t_presupuestos",
    "subvención": "t_presupuestos", "sueldos": "t_presupuestos",
    "tasas": "t_presupuestos", "tercera edad": "t_serviciossociales",
    "transición energética": "t_medioambiente", "transparencia": "t_participacion",
    "transporte": "t_movilidad", "transporte público": "t_movilidad",
    "tranvía": "t_movilidad", "tráfico": "t_movilidad", "turismo": "t_turismo",
    "turismoa": "t_turismo", "urbanismo": "t_urbanismo",
    "violencia de género": "t_igualdad", "violencia machista": "t_igualdad",
    "vivienda": "t_vivienda", "vivienda deshabitada": "t_vivienda",
    "vivienda en alquiler": "t_vivienda", "vivienda protegida": "t_vivienda",
    "vivienda pública": "t_vivienda", "vivienda social": "t_vivienda",
    "vivienda vacía": "t_vivienda", "viviendas": "t_vivienda",
    "viviendas deshabitadas": "t_vivienda", "viviendas sociales": "t_vivienda",
    "viviendas vacías": "t_vivienda", "vpo": "t_vivienda", "zbe": "t_medioambiente",
    "zona 30": "t_movilidad", "zona de bajas emisiones": "t_medioambiente",
    "zonas de bajas emisiones": "t_medioambiente",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# mismo diccionario, pero con las claves sin acentos -- el LLM genera "tranvia"
# tan a menudo como "tranvía", y el lookup debe casar los dos.
_LABEL_TOPLEVEL_NORM = {_strip_accents(k): v for k, v in _LABEL_TOPLEVEL.items()}


# Índice de embeddings (bge-m3) de todas las etiquetas reales de temas y
# subtemas, para corregir slugs inventados que no comparten texto con ningún
# tema real (br:t_ciclovia -> bicicleta). Mismo enfoque que Sharma et al.
# (arXiv:2502.13369): recuperar las entidades reales del grafo en vez de
# añadir una regla por cada invención.
_THEMES_TTL = TEMAS_SKOS
_tema_embed_cache = None
_tema_embed_lock = threading.Lock()


def _tema_embed_index():
    global _tema_embed_cache
    if _tema_embed_cache is not None:
        return _tema_embed_cache
    with _tema_embed_lock:
        if _tema_embed_cache is not None:
            return _tema_embed_cache
        import rdflib
        from rdflib.namespace import SKOS
        g = rdflib.Graph()
        g.parse(_THEMES_TTL, format="turtle")
        BO_ = rdflib.Namespace("http://bilbao.tfg/ontology#")
        textos, slugs = [], []
        for t in g.subjects(rdflib.RDF.type, BO_.Tema):
            slug = str(t).rsplit("/", 1)[-1]
            for pred in (SKOS.prefLabel, SKOS.altLabel):
                for lab in g.objects(t, pred):
                    textos.append(str(lab))
                    slugs.append(slug)
        from langchain_ollama import OllamaEmbeddings
        embedder = OllamaEmbeddings(model="bge-m3")
        vecs = embedder.embed_documents(textos)
        _tema_embed_cache = (textos, slugs, vecs, embedder)
        return _tema_embed_cache


# Umbral 0,70 calibrado con 8 slugs inventados: las equivalencias reales
# puntuaron 0,71-1,00 y las forzadas 0,62-0,68. Por debajo no se corrige (la
# consulta dará 0 filas y se reintentará).
def _fix_tema_uri_semantic(uri_slug: str, umbral: float = 0.70):
    try:
        textos, slugs, vecs, embedder = _tema_embed_index()
        qv = embedder.embed_query(uri_slug.replace("_", " ").replace("-", " "))

        def _cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            return dot / (na * nb + 1e-9)

        mejor_sim, mejor_slug = -1.0, None
        for i, v in enumerate(vecs):
            s = _cos(qv, v)
            if s > mejor_sim:
                mejor_sim, mejor_slug = s, slugs[i]
        if mejor_sim >= umbral:
            return f"br:{mejor_slug}"
    except Exception:
        pass
    return None


# slug de tema inventado (br:t_presupuestos_fiscalidad...) -> tema real: por
# texto compartido con un tema de nivel 1 y, si no, por embeddings
def _fix_tema_uri(uri_slug: str) -> str:
    s = uri_slug.replace("-", "").replace("_", "")
    for canon in _CANON_TEMAS:
        if s.startswith(canon) or canon.startswith(s) or canon in s:
            return f"br:t_{canon}"
    semantico = _fix_tema_uri_semantic(uri_slug)
    return semantico or f"br:t_{uri_slug}"


_VOCAB = None


_VOCAB_LOCK = threading.Lock()


def _vocabulario_real(g):
    global _VOCAB
    if _VOCAB is None:
        with _VOCAB_LOCK:
            if _VOCAB is None:
                bo, br = "http://bilbao.tfg/ontology#", "http://bilbao.tfg/resource/"
                terminos, grupos = set(), set()
                for triple in g:
                    for x in triple:
                        x = str(x)
                        if x.startswith(bo):
                            terminos.add(x[len(bo):])
                        elif x.startswith(br + "grupo_"):
                            grupos.add(x[len(br):])
                _VOCAB = (terminos, grupos)
    return _VOCAB


# contenidos de los FILTER(...) de la consulta (paréntesis equilibrados)
def _expresiones_filter(sparql: str) -> list:
    out = []
    for m in re.finditer(r"\bFILTER\s*\(", sparql, re.I):
        depth, i = 1, m.end()
        while i < len(sparql) and depth:
            depth += {"(": 1, ")": -1}.get(sparql[i], 0)
            i += 1
        out.append((m.start(), i, sparql[m.end():i - 1]))
    return out


# Variables usadas en un FILTER que no aparecen en ningún triple, BIND ni
# VALUES: el filtro descarta todas las filas sin dar error (visto con
# "FILTER(?anio >= 2019 && ...)" sin "?p bo:anio ?anio").
def _variables_sin_enlazar(sparql: str) -> list:
    m_where = re.search(r"\bWHERE\s*\{", sparql, re.I)
    if not m_where:
        return []
    cuerpo = sparql[m_where.end():]
    exprs = _expresiones_filter(cuerpo)
    en_filter = {v for _, _, e in exprs for v in re.findall(r"\?(\w+)", e)}
    resto = cuerpo
    for ini, fin, _ in reversed(exprs):
        resto = resto[:ini] + resto[fin:]
    enlazadas = set(re.findall(r"\?(\w+)", resto))
    return sorted(en_filter - enlazadas)


_ETIQUETAS_TEMA = None


# {"prefLabel": {etiqueta: [uris]}, "altLabel": {...}} de todo el grafo
def _etiquetas_tema(g) -> dict:
    global _ETIQUETAS_TEMA
    if _ETIQUETAS_TEMA is None:
        from rdflib.namespace import SKOS
        with _VOCAB_LOCK:
            if _ETIQUETAS_TEMA is None:
                out = {}
                for nombre, pred in (("prefLabel", SKOS.prefLabel), ("altLabel", SKOS.altLabel)):
                    d = {}
                    for s_, o in g.subject_objects(pred):
                        d.setdefault(str(o), []).append(str(s_))
                    out[nombre] = d
                _ETIQUETAS_TEMA = out
    return _ETIQUETAS_TEMA


# REGEX sobre skos:prefLabel que no casa con NINGUNA etiqueta principal del
# grafo: la consulta devolverá 0 aunque haya proposiciones sobre el asunto.
# Pasaba con "zona de bajas emisiones", que es un sinónimo (altLabel) del tema
# t_calidad_aire, no la etiqueta de un tema. Devuelve (patrón, sinónimo de).
def _temas_inexistentes(g, sparql: str) -> list:
    vars_pref = set(re.findall(r"skos:prefLabel\s+\?(\w+)", sparql))
    etiquetas = _etiquetas_tema(g)
    malos = []
    for var, patron in re.findall(r'REGEX\s*\(\s*(?:STR\s*\(\s*)?\?(\w+)\s*\)?\s*,\s*"([^"]+)"', sparql, re.I):
        if var not in vars_pref:
            continue
        try:
            rx = re.compile(patron.replace("\\\\", "\\"), re.I)
        except re.error:
            continue
        if any(rx.search(lab) for lab in etiquetas["prefLabel"]):
            continue
        sinonimo = sorted({u for lab, uris in etiquetas["altLabel"].items() if rx.search(lab) for u in uris})
        malos.append((patron, sinonimo))
    return malos


def _mensaje_temas(malos: list) -> str:
    partes = []
    for patron, sinonimo in malos:
        txt = f'"{patron}" no es la etiqueta de ningún tema'
        if sinonimo:
            txt += " (es un sinónimo del tema " + ", ".join("br:" + u.rsplit("/", 1)[-1] for u in sinonimo[:3]) + ", más amplio)"
        partes.append(txt)
    return ("filtra por la etiqueta de un tema, pero " + "; ".join(partes) + ". Si el asunto es concreto, "
            "búscalo en el título (?p bo:tituloTopic ?titulo . FILTER(REGEX(STR(?titulo), \"palabra\", \"i\"))) o en "
            "las menciones (?p bo:menciona ?e . ?e rdfs:label ?n . FILTER(REGEX(STR(?n), \"palabra\", \"i\"))); "
            "si basta con el tema amplio, usa su URI con bo:trataTemaAmplio")


# términos bo: y grupos br:grupo_ de la consulta que no existen en el grafo
def _vocabulario_inexistente(g, sparql: str) -> list:
    terminos, grupos = _vocabulario_real(g)
    malos = {f"bo:{t}" for t in re.findall(r"\bbo:(\w+)", sparql) if t not in terminos}
    malos |= {f"br:{t}" for t in re.findall(r"\bbr:(grupo_\w+)", sparql) if t not in grupos}
    return sorted(malos)


# =============================================================================
# 2. Limpieza y guardas del SPARQL
# =============================================================================

# texto del LLM -> consulta: sin ``` ni PREFIX ni la prosa que añade detrás
def _clean_sparql(txt: str) -> str:
    txt = re.sub(r"```(?:sparql)?", "", txt).strip()
    m = re.search(r"\b(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)\b", txt, re.IGNORECASE)
    if m:
        txt = txt[m.start():]
    # El LLM copia el escape de llaves de los ejemplos del SCHEMA ({{ }} de
    # str.format) — rdflib lo tolera pero rompe los regex de _sanitize_sparql.
    txt = txt.replace("{{", "{").replace("}}", "}")
    # Quitar las líneas PREFIX que escribe el LLM: rdflib ya tiene bo:/br:/skos:/
    # rdfs:/xsd: enlazados desde el grafo, y el LLM a veces inventa el namespace
    # (PREFIX bo: <http://example.org/...> -> 0 resultados en silencio). Vale
    # cualquier nombre de prefijo, también mal escrito ("PREFIX sk. <...>").
    txt = re.sub(r"(?im)^\s*PREFIX\b[^<\n]*<[^>]*>\s*\.?\s*$\n?", "", txt)
    # Cortar la prosa que el LLM a veces añade DESPUÉS de la consulta ("Esta
    # consulta filtra...", "### Explicación:") pese a pedirle "solo SPARQL":
    # rompía g.query() con "Expected end of text". Se recorre contando llaves y
    # se corta cuando la profundidad vuelve a 0 (fin del patrón WHERE) + los
    # modificadores de solución que sigan. Contar llaves —y no rfind('}')—
    # porque la prosa suele CITAR trozos de la consulta, con sus '}' incluidos.
    depth, seen_open = 0, False
    for i, ch in enumerate(txt):
        if ch == "{":
            depth += 1
            seen_open = True
        elif ch == "}":
            depth -= 1
            if seen_open and depth == 0:
                rest = txt[i + 1:]
                mm = re.match(
                    r"\s*(?:(?:GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET|HAVING|VALUES)\b[^\n]*\s*)*",
                    rest, re.IGNORECASE)
                txt = txt[:i + 1] + (mm.group(0) if mm else "")
                break
    return txt.strip()


# --- capa de alias: invenciones sistemáticas del LLM -> patrón real del grafo ---
# (entidades, personas o grupos puestos como URI en vez de resolverse con
# rdfs:label + REGEX; ver ESTUDIO_SPARQL_LOCAL.md, fases 1-2)
_ALIAS_GRUPOS = {
    "pp": "grupo_pp", "partidopopular": "grupo_pp",
    "ehbildu": "grupo_eh_bildu", "bildu": "grupo_eh_bildu", "ehb": "grupo_eh_bildu",
    "pseee": "grupo_pse_ee", "pse": "grupo_pse_ee", "socialistas": "grupo_pse_ee",
    "elkarrekinbilbao": "grupo_elkarrekin_bilbao", "elkarrekin": "grupo_elkarrekin_bilbao",
    "goazenbilbao": "grupo_goazen_bilbao", "goazen": "grupo_goazen_bilbao",
    "udalberri": "grupo_udalberri", "eajpnv": "grupo_eaj_pnv", "pnv": "grupo_eaj_pnv",
    "eaj": "grupo_eaj_pnv", "ciudadanos": "grupo_ciudadanos", "cs": "grupo_ciudadanos",
    "vox": "grupo_vox", "ezkerbatuaiu": "grupo_ezker_batua_iu", "ezkerbatua": "grupo_ezker_batua_iu",
    "equipodegobierno": "grupo_equipo_de_gobierno", "gobierno": "grupo_equipo_de_gobierno",
    "grupomixto": "grupo_grupo_mixto", "mixto": "grupo_grupo_mixto",
}


_ALIAS_VOTO_PERSONA = ("proponePersona", "intervino", "concejalVotoAFavor",
                       "concejalVotoEnContra", "concejalVotoAbstencion")


_ALIAS_STOP = {"concejal", "concejala", "concejales", "sr", "sra", "don", "dona", "doña",
              "grupo", "municipal", "el", "la", "los", "las", "de", "del", "senor",
              "senora", "señor", "señora", "persona", "entidad"}


def _alias_canon_grupo(raw: str):
    k = re.sub(r"[^a-z0-9]", "", raw.lower())
    if k in _ALIAS_GRUPOS:
        return "br:" + _ALIAS_GRUPOS[k]
    if k.startswith("grupo"):
        return "br:" + re.sub(r"^grupo_?", "grupo_", raw.lower())
    return None


def _alias_tokens(uri: str):
    u = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", uri)
    u = re.sub(r"[_\-]+", " ", u)
    toks = [w for w in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+", u.lower()) if len(w) > 1]
    keep = [w for w in toks if w not in _ALIAS_STOP]
    return keep or toks


def _alias_regex_and(var: str, uri: str) -> str:
    toks = _alias_tokens(uri) or [uri.lower()]
    return " && ".join(f'REGEX(STR({var}), "{t}", "i")' for t in toks)


def _alias_tail(body: str, subj: str, term: str) -> str:
    if term == ";":
        return body + f" . {subj} "
    return body + " " + term


def _alias_per_grp(m):
    pre, pred, uri = m.group(1), m.group(2), m.group(3)
    term = m.group(4) if (m.lastindex or 0) >= 4 else "."
    nom = {"votoAFavorDe": "concejalVotoAFavor", "votoEnContraDe": "concejalVotoEnContra",
           "seAbstuvo": "concejalVotoAbstencion"}[pred]
    body = (f'{pre} bo:{nom} ?per_f . ?per_f rdfs:label ?per_fl . '
            f'FILTER({_alias_regex_and("?per_fl", uri)})')
    return _alias_tail(body, pre, term)


def _alias_rewrite(sparql: str) -> str:
    if not sparql:
        return sparql
    s = sparql

    # <br:x> -> br:x (el modelo a veces mete la URI prefijada entre <>)
    s = re.sub(r"<(br:[A-Za-z_]\w*)>", r"\1", s)
    s = re.sub(r"<(bo:[A-Za-z_]\w*)>", r"\1", s)

    # tema con prefijo bo: -> br: (bo:t_euskera -> br:t_euskera)
    s = re.sub(r"\bbo:(t_[a-z_]+)\b", r"br:\1", s)

    # grupo con URI no canonica: br:EHBildu / br:EH_Bildu -> br:grupo_eh_bildu
    s = re.sub(r"\bbr:([A-Za-z][A-Za-z_]*)\b", lambda m: (
        _alias_canon_grupo(m.group(1)) or m.group(0)) if re.sub(r"[^a-z0-9]", "", m.group(1).lower()) in _ALIAS_GRUPOS else m.group(0), s)

    # entidad como URI: ?p bo:menciona br:Iberdrola -> ?p bo:menciona ?ent_f . ?ent_f rdfs:label ?ent_fl . FILTER(...)
    def _ent(m):
        pre, uri = m.group(1), m.group(2)
        uri = re.sub(r"^(entidad_|ent_|lugar_|org_|organizacion_)", "", uri, flags=re.I)
        body = (f'{pre} bo:menciona ?ent_f . ?ent_f rdfs:label ?ent_fl . '
                f'FILTER({_alias_regex_and("?ent_fl", uri)})')
        return _alias_tail(body, pre, m.group(3))
    s = re.sub(r"(\?\w+)\s+bo:menciona\s+br:([A-Za-z_]\w*)\s*([;.])", _ent, s)

    # persona como URI con predicados de persona / voto nominal
    def _per(m):
        pre, pred, uri = m.group(1), m.group(2), m.group(3)
        body = (f'{pre} bo:{pred} ?per_f . ?per_f rdfs:label ?per_fl . '
                f'FILTER({_alias_regex_and("?per_fl", uri)})')
        return _alias_tail(body, pre, m.group(4))
    s = re.sub(r"(\?\w+)\s+bo:(" + "|".join(_ALIAS_VOTO_PERSONA) + r")\s+br:([A-Za-z_]\w*)\s*([;.])", _per, s)

    # voto de GRUPO con un br:<algo> que NO es un grupo canonico -> es persona
    s = re.sub(r"(\?\w+)\s+bo:(votoAFavorDe|votoEnContraDe|seAbstuvo)\s+br:(?!grupo_)([A-Za-z_]\w*)\s*([;.])",
               _alias_per_grp, s)

    # rdfs:label "Nombre Apellido" EXACTO -> REGEX (rara vez coincide letra a letra con el grafo)
    def _lbl(m):
        var, lit, term = m.group(1), m.group(2), m.group(3)
        if len(lit.split()) > 4 or len(lit) < 3:
            return m.group(0)
        body = f'{var} rdfs:label ?lbl_f . FILTER({_alias_regex_and("?lbl_f", lit)})'
        return _alias_tail(body, var, term)
    s = re.sub(r'(\?\w+)\s+rdfs:label\s+"([^"]{3,50})"\s*([;.)}])', _lbl, s)

    return s


# 0. COUNT(DISTINCT ?x) con ?x sin ligar en el WHERE da siempre 0: se cuenta la
#    variable de la proposición
def _g_count_sin_ligar(sparql: str, pregunta: str) -> str:
    m0 = re.search(r"COUNT\s*\(\s*DISTINCT\s+\?(\w+)\s*\)", sparql, re.I)
    if m0:
        cvar = m0.group(1)
        where_m = re.search(r"WHERE\s*\{(.*)\}\s*$", sparql, re.I | re.S)
        body = where_m.group(1) if where_m else sparql
        if not re.search(rf"(?<!\w)\?{cvar}\b", body):
            subj_m = re.search(r"\?(\w+)\s+a\s+bo:Proposicion\b", sparql)
            if subj_m and subj_m.group(1) != cvar:
                sparql = re.sub(rf"(?<!\w)\?{cvar}\b", f"?{subj_m.group(1)}", sparql)
    return sparql


# 1. OPTIONAL que no aporta nada al resultado pero lleva un valor concreto: el
#    LLM lo quería como filtro, se saca del OPTIONAL
def _g_optional_noop(sparql: str, pregunta: str) -> str:
    if not re.search(r"\bCOUNT\s*\(", sparql, re.I):
        return sparql
    agg_vars = set(re.findall(r"(?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?(\?\w+)", sparql, re.I))
    gb_match = re.search(r"GROUP\s+BY\s+([^\n]+)", sparql, re.I)
    gb_vars = set(re.findall(r"\?\w+", gb_match.group(1))) if gb_match else set()
    select_clause = re.search(r"SELECT\b(.*?)\bWHERE", sparql, re.I | re.S)
    select_vars = set(re.findall(r"\?\w+", select_clause.group(1))) if select_clause else set()

    def _promote(m):
        body = m.group(1)
        opt_vars = set(re.findall(r"\?\w+", body))
        bind_vars = set(re.findall(r"\bAS\s+(\?\w+)", body, re.I))
        # si el OPTIONAL introduce una variable que se usa en el SELECT, los
        # agregados o el GROUP BY, sí influye en el resultado: no se toca
        introduced = bind_vars | (opt_vars - set(re.findall(r"\?\w+", sparql[:m.start()])))
        if introduced & (agg_vars | gb_vars | select_vars):
            return m.group(0)
        if any(ind in body for ind in _RESULT_INDIVIDUALS) or re.search(r"\bbr:\w+", body):
            return body.strip().rstrip(".").strip() + " .\n  "
        return m.group(0)

    return re.sub(r"OPTIONAL\s*\{([^{}]*)\}", _promote, sparql)


# 2. "?x rdfs:label ?l" con ?x suelto: producto cartesiano con todas las etiquetas
def _g_join_label(sparql: str, pregunta: str) -> str:
    for subj, obj in re.findall(r"\?(\w+)\s+rdfs:label\s+\?(\w+)", sparql):
        if len(re.findall(rf"\?{subj}\b", sparql)) == 1:
            sparql = re.sub(rf"\?{subj}\s+rdfs:label\s+\?{obj}\s*\.?", "", sparql)
            sparql = re.sub(rf"\(\s*[^()]*\?{obj}[^()]*\)\s*", "", sparql)  # agregados con ?obj
            sparql = re.sub(rf"(?<!\w)\?{obj}\b", "", sparql)               # ?obj suelto en SELECT
            gb = re.search(r"GROUP\s+BY\s+([^\n]*)", sparql, re.I)
            if gb and not re.search(r"\?\w+", gb.group(1)):
                sparql = re.sub(r"GROUP\s+BY\s*[^\n]*\n?", "", sparql, flags=re.I)
    return sparql


# 2b. Ratio con dos variables "a bo:Proposicion" sin vincular: el subconjunto
#     deja de filtrarse por las condiciones del total ("movilidad de EH Bildu
#     rechazadas" daba 74 en vez de 15). Se reescribe como OPTIONAL + BIND si
#     comparten alguna condición; si no, son dos conteos independientes.
def _g_ratio_sin_vincular(sparql: str, pregunta: str) -> str:
    try:
        decl_vars = []
        for m in re.finditer(r"\?(\w+)\s+a\s+bo:Proposicion\b", sparql):
            if m.group(1) not in decl_vars:
                decl_vars.append(m.group(1))
        if len(decl_vars) >= 2:
            counted = set(re.findall(r"COUNT\s*\(\s*(?:DISTINCT\s+)?\?(\w+)\s*\)", sparql, re.I))
            primary = decl_vars[0]

            def _bloque(var):
                return re.search(rf"\?{var}\s+a\s+bo:Proposicion\s*;\s*(.*?)\s*\.\s*(?=\?|\}}|$)", sparql, re.S)

            m_p = _bloque(primary)
            if primary in counted and m_p:
                primary_preds = {p.strip() for p in m_p.group(1).split(";")}
                for var in decl_vars[1:]:
                    if var not in counted:
                        continue
                    m_r = _bloque(var)
                    if not m_r:
                        continue
                    r_preds = [p.strip() for p in m_r.group(1).split(";")]
                    compartido = set(r_preds) & primary_preds
                    extra = [p for p in r_preds if p not in primary_preds]
                    if not compartido or not extra:
                        continue
                    nuevo = f"OPTIONAL {{ ?{primary} {' ; '.join(extra)} . BIND(?{primary} AS ?{var}) }} "
                    sparql = sparql.replace(m_r.group(0), nuevo, 1)
    except Exception:
        pass  # si no se puede analizar, la consulta se deja como está
    return sparql


# 3. Roll-up de temas a mano con skos:broader (sin cierre transitivo en el
#    grafo y con la URI a menudo inventada): bo:trataTemaAmplio ya lo incluye
def _g_broader_manual(sparql: str, pregunta: str) -> str:
    m_bro = re.search(
        r"(?:bo:trataSobre|bo:trataTemaAmplio)\s+\?(\w+)\s*\.\s*"
        r"\?\1\s+skos:broader[*+]?\s+br:t_(\w+)\b\s*(?:;[^.}]*)?\.",
        sparql)
    if m_bro:
        tema_var, tema_slug = m_bro.groups()
        canon = _fix_tema_uri(tema_slug)
        sparql = sparql[:m_bro.start()] + f"bo:trataTemaAmplio {canon} ." + sparql[m_bro.end():]
        # referencias sueltas al ?tema que ya no existe
        sparql = re.sub(rf"\?{tema_var}\s+\w+:\w+\s+\?\w+\s*[;.]?", "", sparql)
        sparql = re.sub(rf"(?<!\w)\?{tema_var}\b", "", sparql)
    return sparql


# 4. URI de tema inventada (br:t_movilidad_y_transporte) o con prefijo bo:
def _g_uri_tema(sparql: str, pregunta: str) -> str:
    def _fix_uri(m):
        slug = m.group(2)
        if f"t_{slug}" in _REAL_TEMA_URIS:
            return f"br:t_{slug}"  # tema real: solo se corrige el prefijo
        canon = _fix_tema_uri(slug)
        return canon if canon != f"br:t_{slug}" else m.group(0)
    return re.sub(r"\b(bo|br):t_(\w+)\b", _fix_uri, sparql)


# 4a. Tipos de literal equivocados: bo:anio es un entero plano y bo:fecha un
#     texto "DD-MM-AAAA" (con ^^xsd:date rdflib ni siquiera construye la consulta)
def _g_tipos_anio_fecha(sparql: str, pregunta: str) -> str:
    sparql = re.sub(r'"(\d{4})"\s*\^\^\s*xsd:\w+', r"\1", sparql)
    sparql = re.sub(r'(bo:anio\s+)"(\d{4})"', r"\1\2", sparql)
    return re.sub(r'"(\d{2}-\d{2}-\d{4})"\s*\^\^\s*xsd:\w+', r'"\1"', sparql)


# 4c. Variable comparada con un año (o un mes) en un FILTER sin enlazar a
#     bo:anio (bo:mes): el filtro descarta todo. Se añade el triple que falta.
def _g_anio_sin_enlazar(sparql: str, pregunta: str) -> str:
    sujeto = re.search(r"\?(\w+)\s+a\s+bo:Proposicion\b", sparql)
    m_where = re.search(r"\bWHERE\s*\{", sparql, re.I)
    if not sujeto or not m_where:
        return sparql
    sin_enlazar = set(_variables_sin_enlazar(sparql))
    nuevos = []
    for var in sorted(sin_enlazar):
        exprs = " ".join(e for _, _, e in _expresiones_filter(sparql))
        if re.search(rf"\?{var}\s*(?:>=|<=|=|>|<|!=)\s*(?:19|20)\d{{2}}\b", exprs):
            nuevos.append(f"?{sujeto.group(1)} bo:anio ?{var} .")
        elif re.search(rf"\?{var}\s+IN\s*\(\s*\d{{1,2}}\s*[,)]", exprs, re.I):
            nuevos.append(f"?{sujeto.group(1)} bo:mes ?{var} .")
    if not nuevos:
        return sparql
    return sparql[:m_where.end()] + " " + " ".join(nuevos) + sparql[m_where.end():]


# 4b. Etiqueta de grupo con nombre no canónico o con @lang ("Partido Popular"@es -> "PP")
def _g_label_grupo(sparql: str, pregunta: str) -> str:
    def _fix_label(m):
        nombre, canon_directo = m.group(1), m.group(1).strip().upper()
        if canon_directo in _GRUPO_LABELS:
            return f'rdfs:label "{canon_directo}"'
        alias = _GRUPO_ALIAS.get(re.sub(r"\s+", " ", nombre.strip().lower()))
        return f'rdfs:label "{alias}"' if alias else m.group(0)
    return re.sub(r'rdfs:label\s+"([^"]+)"(?:@\w+)?', _fix_label, sparql)


# 5. "aprobadas" en general = bo:Aprobada + bo:AprobadaConEnmienda (PP 2015: 8
#    frente a 31). El FILTER se inserta antes del "}" del bloque que contiene
#    cada aparición, para no sacarlo de un OPTIONAL (rompería los ratios).
def _g_aprobadas(sparql: str, pregunta: str) -> str:
    pl = pregunta.lower()
    if not re.search(r"aprob", pl) or re.search(r"sin enmienda|con enmienda|estrict|tal cual|sin modificar", pl):
        return sparql
    for mm in reversed(list(re.finditer(r"bo:tieneResultado\s+bo:Aprobada\b", sparql))):
        depth = 0
        cierre = None
        for i in range(mm.end(), len(sparql)):
            if sparql[i] == "{":
                depth += 1
            elif sparql[i] == "}":
                if depth == 0:
                    cierre = i
                    break
                depth -= 1
        if cierre is None:
            continue
        filtro = " FILTER(?_rApr IN (bo:Aprobada, bo:AprobadaConEnmienda)) "
        sparql = (sparql[:mm.start()] + "bo:tieneResultado ?_rApr"
                  + sparql[mm.end():cierre] + filtro + sparql[cierre:])
    return sparql


_FECHA_FAKE = r"bo:(?:presentadaEn|presentadaEl|fechaPresentacion|fechaProposicion|fechaDePresentacion)"


# 6. Propiedades inventadas por analogía (bo:interviene, bo:firmadaPor,
#    bo:presentadaEn + YEAR()...): el grafo solo tiene bo:anio y bo:fecha
def _g_predicados_inventados(sparql: str, pregunta: str) -> str:
    sparql = re.sub(r"\bbo:intervien\w*\b", "bo:intervino", sparql)
    sparql = re.sub(r"\bbo:(?:firmadaPor|proponente|propuestaPor)\b", "bo:proponePersona", sparql)
    sparql = re.sub(r";\s*" + _FECHA_FAKE + r"\s+\?\w+(?=\s*[.;])", "", sparql)   # en medio de lista ";"
    sparql = re.sub(_FECHA_FAKE + r"\s+\?\w+\s*;\s*", "", sparql)                  # al principio de lista
    sparql = re.sub(_FECHA_FAKE + r"\s+\?\w+\s*(?=\})", "", sparql)               # triple suelto
    # YEAR(?x) -> ?anio, salvo si ?x es una bo:fechaISO real (ahí YEAR funciona)
    sparql = re.sub(r"YEAR\s*\(\s*\?(\w+)\s*\)",
                    lambda m: m.group(0) if re.search(rf"bo:fechaISO\s+\?{m.group(1)}\b", sparql) else "?anio",
                    sparql)
    return re.sub(r"\?anio\s*=\s*(\d{4})\s*&&\s*\?anio\s*=\s*\1", r"?anio = \1", sparql)


# 7. El alcalde no es un nodo "Alcalde de Bilbao": es un bo:Concejal con bo:esAlcalde true
def _g_alcalde(sparql: str, pregunta: str) -> str:
    sparql = re.sub(
        r"\?\w+\s+rdfs:label\s+\"[^\"]*[Aa]lcalde[^\"]*\"\s*[.;]",
        "?_alc bo:esAlcalde true .", sparql)
    return re.sub(r"\?\w+\s+a\s+bo:Alcalde\b\s*[.;]", "?_alc bo:esAlcalde true .", sparql)


# 7b. Pregunta sobre un concejal (sin mencionar grupo) con el predicado de voto
#     de GRUPO: el LLM confunde los dos, se usa el voto nominal
def _g_voto_persona(sparql: str, pregunta: str) -> str:
    if re.search(r"concejal|concejala|qu[ée]\s+persona|\bqui[ée]n\b", pregunta, re.I) \
       and not re.search(r"\bgrupo\b|\bpartido\b", pregunta, re.I):
        sparql = re.sub(r"\bbo:votoAFavorDe\b", "bo:concejalVotoAFavor", sparql)
        sparql = re.sub(r"\bbo:votoEnContraDe\b", "bo:concejalVotoEnContra", sparql)
        sparql = re.sub(r"\bbo:seAbstuvo\b", "bo:concejalVotoAbstencion", sparql)
    return sparql


# 8. CONTAINS de una sola palabra -> REGEX con límite de palabra ("zara" no
#    debe casar "Zaragoza"); los términos de varias palabras se dejan
def _g_contains(sparql: str, pregunta: str) -> str:
    def _cont2regex(m):
        var, term = m.group(1), m.group(2)
        if not re.fullmatch(r"[0-9A-Za-zñáéíóúü]{3,}", term):
            return m.group(0)
        return f'REGEX(STR({var}), "\\\\b{term}\\\\b", "i")'
    return re.sub(
        r'CONTAINS\s*\(\s*LCASE\s*\(\s*(?:STR\s*\(\s*)?(\?\w+)\s*\)?\s*\)\s*,\s*"([^"]+)"\s*\)',
        _cont2regex, sparql, flags=re.I)


_REGEX_LITERAL_RE = r'(REGEX\s*\(\s*(?:LCASE\s*\(\s*)?(?:STR\s*\(\s*)?\?\w+\s*\)*\s*,\s*")([^"]+)"'


_ACC = {"a": "[aá]", "á": "[aá]", "e": "[eé]", "é": "[eé]", "i": "[ií]",
        "í": "[ií]", "o": "[oó]", "ó": "[oó]", "u": "[uúü]", "ú": "[uúü]",
        "ü": "[uúü]", "n": "[nñ]", "ñ": "[nñ]"}


# 9. REGEX insensible a tildes: el flag "i" de rdflib no las ignora ("díez" no
#    casa "Diez"), así que cada vocal y la ñ pasan a una clase [aá]
def _g_regex_tildes(sparql: str, pregunta: str) -> str:
    def _acc_insens(m):
        head, pat = m.group(1), m.group(2)
        if "[" in pat:  # ya trae clases
            return m.group(0)
        return head + "".join(_ACC.get(c.lower(), c) if c.isalpha() else c
                              for c in pat) + '"'
    return re.sub(_REGEX_LITERAL_RE, _acc_insens, sparql, flags=re.I)


# 10. "\b" con una sola barra en un literal SPARQL es un retroceso (\x08), no
#     un límite de palabra: el REGEX deja de casar sin error. Hacen falta dos.
def _g_regex_b(sparql: str, pregunta: str) -> str:
    def _fix_lone_b_escape(m):
        head, pat = m.group(1), m.group(2)
        return head + re.sub(r"(?<!\\)\\b", r"\\\\b", pat) + '"'
    return re.sub(_REGEX_LITERAL_RE, _fix_lone_b_escape, sparql, flags=re.I)


# 11. Roll-up a mano con un OR de etiquetas: se queda corto si falta algún
#     subtema ("movilidad por año": 24 frente a 50). Si todas las etiquetas son
#     del mismo tema de nivel 1, se usa bo:trataTemaAmplio sobre ese tema.
def _g_rollup_or(sparql: str, pregunta: str) -> str:
    def _collapse_rollup(m):
        lvar, disj = m.group(2), m.group(3)
        labels = re.findall(r'STR\s*\(\s*\?' + re.escape(lvar) + r'\s*\)\s*=\s*"([^"]+)"', disj, re.I)
        if len(labels) < 2:
            return m.group(0)
        tops = [_LABEL_TOPLEVEL_NORM.get(_strip_accents(lab.lower())) for lab in labels]
        if any(t is None for t in tops) or len(set(tops)) != 1:
            return m.group(0)
        return f"bo:trataTemaAmplio br:{tops[0]} ."
    return re.sub(
        r"bo:trataTemaAmplio\s+\?(\w+)\s*\.\s*\?\1\s+skos:prefLabel\s+\?(\w+)\s*\.\s*"
        r'FILTER\s*\(\s*((?:STR\s*\(\s*\?\2\s*\)\s*=\s*"[^"]+"\s*(?:\|\|\s*)?)+)\)\s*\.?',
        _collapse_rollup, sparql, flags=re.I)


# 12. bo:trataTemaAmplio con una etiqueta que no es ningún tema (un nombre
#     propio: "BBVA"): se busca como mención con bo:menciona
def _g_tema_vs_entidad(sparql: str, pregunta: str) -> str:
    def _fix_tema_vs_entidad(m):
        tvar, lvar, pat = m.group(1), m.group(2), m.group(3)
        # la guarda 9 ya ha expandido las vocales a clases [eé]: se deja la
        # primera letra de cada clase para recuperar la palabra
        sin_clases = re.sub(r"\[([^\]]+)\]", lambda cm: cm.group(1)[0], pat)
        sin_clases = re.sub(r"\\+b", "", sin_clases)
        # alternancia "presupuestos|fiscalidad": basta con que un término sea un tema
        terminos = [_strip_accents(re.sub(r"[^a-záéíóúñ]", "", t.lower())) for t in sin_clases.split("|")]
        if any(term and any(term in lab or lab in term for lab in _LABEL_TOPLEVEL_NORM) for term in terminos):
            return m.group(0)
        return (f'bo:menciona ?{tvar} . ?{tvar} rdfs:label ?{lvar} . '
                f'FILTER(REGEX(STR(?{lvar}), "{pat}", "i"))')
    return re.sub(
        r'bo:trataTemaAmplio\s+\?(\w+)\s*\.\s*\?\1\s+skos:prefLabel\s+\?(\w+)\s*\.\s*'
        r'FILTER\s*\(\s*REGEX\s*\(\s*STR\s*\(\s*\?\2\s*\)\s*,\s*"([^"]+)"\s*,\s*"i"\s*\)\s*\)\s*\.?',
        _fix_tema_vs_entidad, sparql, flags=re.I)


_MANDATO_INICIO = (2007, 2011, 2015, 2019, 2023)


# 13. "desde el inicio del mandato actual (2023) hasta ahora" es un rango, no
#     un único año: FILTER(?anio = 2023) -> FILTER(?anio >= 2023)
def _g_mandato(sparql: str, pregunta: str) -> str:
    if not (re.search(r"\bmandato\b|\blegislatura\b", pregunta, re.I) and
            re.search(r"actual|en curso|hasta (la fecha|ahora|hoy)", pregunta, re.I)):
        return sparql

    def _mandato_range(m):
        var, anio = m.group(1), int(m.group(2))
        if anio in _MANDATO_INICIO:
            return f"FILTER(?{var} >= {anio})"
        return m.group(0)
    return re.sub(r"FILTER\s*\(\s*\?(\w+)\s*=\s*(\d{4})\s*\)", _mandato_range, sparql)


_KNOWN_PREFIXES = {"bo", "br", "skos", "rdfs", "xsd", "rdf", "owl"}


_RDFS_TERMS = {"label", "comment", "seeAlso", "subClassOf", "domain", "range"}


# 14. Prefijo inventado ("r3:label"): rdflib falla con "Unknown namespace
#     prefix". Se usa el prefijo real con el que aparece ese nombre en la misma
#     consulta y, si no aparece, rdfs: para label/comment/...
def _g_prefijo_inventado(sparql: str, pregunta: str) -> str:
    declared_used = {}
    for pfx, local in re.findall(r"\b([a-zA-Z][\w-]*):([a-zA-Z_][\w-]*)", sparql):
        if pfx in _KNOWN_PREFIXES:
            declared_used.setdefault(local, pfx)

    def _fix_unknown_prefix(m):
        pfx, local = m.group(1), m.group(2)
        if pfx in _KNOWN_PREFIXES:
            return m.group(0)
        real = declared_used.get(local) or ("rdfs" if local in _RDFS_TERMS else None)
        return f"{real}:{local}" if real else m.group(0)
    return re.sub(r"\b([a-zA-Z][\w-]*):([a-zA-Z_][\w-]*)", _fix_unknown_prefix, sparql)


# 15. Fecha con barras ("24/09/2015"): bo:fecha se guarda como "24-09-2015"
def _g_fecha_barras(sparql: str, pregunta: str) -> str:
    def _fix_date_slash(m):
        d, mth, y = m.group(1), m.group(2), m.group(3)
        return f'"{int(d):02d}-{int(mth):02d}-{y}"'
    return re.sub(r'"(\d{1,2})/(\d{1,2})/(\d{4})"', _fix_date_slash, sparql)


# 16. bo:fechaISO es xsd:date: comparada con un literal sin tipo
#     (FILTER(?d >= "2023-06-15")) da falso en todas las filas, sin error, y
#     cualquier rango de fechas devolvía 0. Se tipa el literal.
def _g_fecha_iso_tipada(sparql: str, pregunta: str) -> str:
    for v in set(re.findall(r"bo:fechaISO\s+\?(\w+)", sparql)):
        sparql = re.sub(rf'(\?{v}\s*(?:>=|<=|!=|=|>|<)\s*)"(\d{{4}}-\d{{2}}-\d{{2}})"(?!\^\^)',
                        r'\1"\2"^^xsd:date', sparql)
        sparql = re.sub(rf'"(\d{{4}}-\d{{2}}-\d{{2}})"(?!\^\^)(\s*(?:>=|<=|!=|=|>|<)\s*\?{v}\b)',
                        r'"\1"^^xsd:date\2', sparql)
    return sparql


# 17. bo:fecha es texto DD-MM-AAAA: ORDER BY ?fecha ordena alfabéticamente
#     ("31-01-2019" antes que "26-02-2026"). Se ordena por AAAAMMDD.
def _g_orden_fecha_texto(sparql: str, pregunta: str) -> str:
    m = re.search(r"\bORDER\s+BY\s+(.+?)(?=\s+LIMIT\b|\s+OFFSET\b|\s*$)", sparql, re.I | re.S)
    if not m:
        return sparql
    clausula = m.group(1)
    for v in set(re.findall(r"bo:fecha\s+\?(\w+)", sparql)):
        clave = f"CONCAT(SUBSTR(?{v}, 7, 4), SUBSTR(?{v}, 4, 2), SUBSTR(?{v}, 1, 2))"
        clausula = re.sub(rf"\b(ASC|DESC)\s*\(\s*\?{v}\s*\)", r"\1(@CLAVE@)", clausula, flags=re.I)
        clausula = re.sub(rf"\?{v}\b", "ASC(@CLAVE@)", clausula)
        clausula = clausula.replace("@CLAVE@", clave)
    return sparql[:m.start(1)] + clausula + sparql[m.end(1):]


_GUARDAS = (
    _g_count_sin_ligar, _g_optional_noop, _g_join_label, _g_ratio_sin_vincular,
    _g_broader_manual, _g_uri_tema, _g_tipos_anio_fecha, _g_anio_sin_enlazar, _g_label_grupo,
    _g_aprobadas, _g_predicados_inventados, _g_alcalde, _g_voto_persona,
    _g_contains, _g_regex_tildes, _g_regex_b, _g_rollup_or,
    _g_tema_vs_entidad, _g_mandato, _g_prefijo_inventado, _g_fecha_barras,
    _g_fecha_iso_tipada, _g_orden_fecha_texto,
)


# aplica todas las guardas en orden
def _sanitize_sparql(sparql: str, verbose: bool = False, pregunta: str = "") -> str:
    original = sparql
    for guarda in _GUARDAS:
        sparql = guarda(sparql, pregunta)
    sparql = re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", sparql)).strip()
    if verbose and sparql != original:
        print(f"[SPARQL saneado]\n{sparql}\n")
    return sparql


# texto del LLM -> consulta lista para ejecutar
def _preparar_sparql(texto_llm: str, verbose: bool, pregunta: str) -> str:
    return _sanitize_sparql(_alias_rewrite(_clean_sparql(texto_llm)), verbose, pregunta)
