"""
Consulta SPARQL montada directamente a partir del análisis de la pregunta, sin
LLM, para las formas de pregunta más comunes: contar, qué grupo más, desglose
por grupo, qué año más, la última vez, listar, "¿se ha hablado alguna vez
de...?", "¿qué se dijo/opinó sobre...?" y cómo votaron los grupos.

Casi todas las respuestas falsas de las evaluaciones independientes venían de
consultas mal escritas por el LLM (condiciones unidas con OR, LIMIT dentro de
un recuento, relaciones inventadas). Pero el análisis ya extrae el tema, el
asunto, el grupo, los años, el resultado y los meses: con eso la consulta se
compone de piezas fijas y comprobadas. Si la pregunta pide algo que estas
piezas no saben expresar (personas, porcentajes, "por qué", dos preguntas en
una...), se devuelve None y sigue el camino de siempre (plantillas y LLM).
"""
import re

from grafo.consulta.pregunta import _ACCIONES, _norm, _PROPIOS_NEUTROS, Analisis


_RESULTADOS = {"aprobada": "bo:Aprobada, bo:AprobadaConEnmienda", "rechazada": "bo:Rechazada",
               "decae": "bo:Decae", "retirada": "bo:Retirada"}
_GOBIERNO = "br:grupo_equipo_de_gobierno, br:grupo_eaj_pnv, br:grupo_pse_ee, br:grupo_desconocido"

# lo que estas piezas no saben expresar: se deja al LLM
_NO_SOPORTADO = re.compile(
    r"\bintervin|interven|concejal|alcald|\benmienda|porcentaje|\btasa de aprob|%|\bproporcion|conjunta"
    r"|\bpor que\b|\by cuant|\bmiembros\b|firmad"
    r"|\bno (llegaron|llego|tuvieron|tienen|tiene|han tenido)\b|\bsin (resultado|votacion)")
# votaciones: solo en la forma "votos"
_VOTO = re.compile(r"\bvot(o|a|aron|ado|os|acion|aciones)\b|abstuv|abstenci")

# puntos del orden del día que no son proposiciones (bo:tipoPunto, ver
# construccion/enriquecer.py): preguntas, daciones de cuenta, la aprobación del
# acta anterior, la urgencia... No entran en los recuentos de proposiciones
# ("aprobar por unanimidad el acta de la sesión anterior" contaba como
# proposición aprobada por unanimidad)
_NO_PROPOSICION = ("pregunta", "dacion_cuenta", "tramite", "debate_estado_ciudad", "informe_iniciativa")
_FILTRO_PROPOSICION = ("  FILTER NOT EXISTS { ?p bo:tipoPunto ?_tp . FILTER(?_tp IN ("
                       + ", ".join(f'"{t}"' for t in _NO_PROPOSICION) + ")) }\n")


# Formas sobre las intervenciones (bo:Intervencion): quién habló, cuántas veces
_FORMAS_INTERV = ("ranking_orador", "ranking_grupo_interv", "conteo_interv")


def forma(q: str) -> str:
    # "¿quién/qué concejal habló más sobre X?", "¿qué grupo intervino más...?", "¿cuántas veces habló X...?"
    if re.search(r"\b(que|cual) (grupo|partido)s?\b.*\b(habl|interv)\w*\b.*\bmas\b|\b(grupo|partido)s? que mas (habl|interv)", q):
        return "ranking_grupo_interv"
    if re.search(r"\b(quien(es)?|que concejal\w*|que orador\w*|que persona)\b.*\b(habl|interv)\w*\b.*\bmas\b|\bquien mas (habl|interv)", q):
        return "ranking_orador"
    if re.search(r"\bcuant\w+ (veces|intervenciones)\b.*\b(interv|habl)|\bcuant\w+ intervenciones\b", q):
        return "conteo_interv"
    # "¿cuántos plenos (ordinarios) hubo en 2016?": sesiones, no proposiciones
    if re.search(r"^\W*cuant[oa]s\s+(plenos|sesiones)\b", q):
        return "plenos"
    if _VOTO.search(q) and re.search(r"\b(que grupos?|que partidos?|quien(es)?|como)\b", q):
        return "votos"
    if re.search(r"\bque (grupo|partido|grupo politico|grupo municipal)s?\b.*\bmas\b", q):
        return "ranking_grupo"
    # "¿qué asociación ha presentado más proposiciones vecinales?"
    if re.search(r"\b(que|cual|cuales) (asociacion|colectivo|entidad|plataforma|organizacion)(es|s)?\b.*\bmas\b", q):
        return "ranking_particular"
    # "¿qué barrio / distrito ha tenido más...?" (bo:enBarrio, bo:enDistrito)
    if re.search(r"\b(que|cual|cuales) (barrio|distrito)s?\b.*\bmas\b", q):
        return "ranking_distrito" if re.search(r"\bdistritos?\b", q) else "ranking_barrio"
    # "¿cuál ha sido el pleno con más proposiciones?"
    if re.search(r"\b(que|cual) (ha sido |fue |es )?(el )?pleno\b.*\bmas\b|\bpleno con mas\b", q):
        return "ranking_pleno"
    # "¿en qué legislatura se presentaron más...?"
    if re.search(r"\b(en )?(que|cual) (legislatura|mandato)\b.*\bmas\b", q):
        return "ranking_legislatura"
    # "¿cuál fue el primer pleno de la legislatura 2023-2027?"
    if re.search(r"\bprimer pleno\b|\bprimera sesion\b", q):
        return "primer_pleno"
    if re.search(r"\b(cada grupo|por grupos?|cada partido|por partidos?)\b", q):
        return "desglose_grupo"
    if re.search(r"\b(en )?que ano\b.*\bmas\b|\bcual (fue|ha sido) el ano\b.*\bmas\b|\bque ano tuvo\b", q):
        return "ranking_anio"
    if re.search(r"\bultim[ao]s?\b|\bmas reciente", q) and re.search(r"^\W*(cuando|cual|que)\b", q):
        return "ultima"
    if re.search(r"^\W*cuant[ao]s\b", q):
        return "conteo"
    if re.search(r"^\W*(que|cuales)\s+(propuestas|mociones|proposiciones|iniciativas)\b", q):
        return "lista"
    # "¿se ha hablado/debatido alguna vez de...?", "¿hay alguna moción sobre...?"
    if re.search(r"^\W*(se ha|se han|ha habido|hay|han|existe|consta)\b|\balguna vez\b", q):
        return "existe"
    # "¿qué se dijo/opinó/debatió/comentó sobre...?": el grafo no guarda lo que se
    # dijo; se listan las proposiciones sobre el asunto y el narrador lo avisa
    if re.search(r"^\W*que\b.*\b(opin|dijo|dijeron|se dijo|coment|debati|postura|penso)", q):
        return "contenido"
    return ""


# props: lista de URIs, o de (URI, relación) con relación "trata" (título o
# subtemas) o "menciona" (solo entre las entidades mencionadas)
def _bloque_asunto(raices, props) -> str:
    bloque = f"  # asunto (en el título, los subtemas o las menciones): {', '.join(raices)}\n"
    if not props:
        # ninguna proposición: FILTER(false) no filtra nada en rdflib (devolvía
        # las 3.422 proposiciones); 1 = 0 sí
        return bloque + "  FILTER(1 = 0)\n"
    if isinstance(props[0], tuple):
        valores = " ".join(f'(<{p}> "{rel}")' for p, rel in props)
        return bloque + f"  VALUES (?p ?relacion) {{ {valores} }}\n"
    return bloque + f"  VALUES ?p {{ {' '.join(f'<{p}>' for p in props)} }}\n"


def _grupos_votantes(q: str, a: Analisis) -> bool:
    # "¿cómo votó el PP...?", "¿votó el PNV...?": el grupo es quien vota, no quien
    # presenta, salvo que el grupo vaya detrás de "la moción del/presentada por"
    # ("la propuesta de rebajar el IBI" no es "la propuesta del PSE-EE")
    from grafo.consulta.pregunta import _GRUPOS
    if not a.grupos or not re.search(r"\bcomo vot|\bvot\w*\s+(el|la|los)?\s*(grupo\s+)?\w+", q):
        return False
    nombres = "|".join(f"(?:{pat})" for pat, _, _ in _GRUPOS)
    return not re.search(rf"\b(mocion|propuesta|proposicion|iniciativa)\w*\s+(?:del?|presentad\w*(?:\s+por)?)\s+"
                         rf"(?:el\s+|la\s+|los\s+)?(?:grupo\s+(?:municipal\s+)?)?(?:{nombres})", q)


# una legislatura: de su pleno de constitución al de la siguiente
def _filtro_fechas(a: Analisis) -> str:
    out = f'  FILTER(?d >= "{a.desde_fecha}"^^xsd:date)\n' if a.desde_fecha else ""
    return out + (f'  FILTER(?d < "{a.hasta_fecha}"^^xsd:date)\n' if a.hasta_fecha else "")


# Sesiones del Pleno: cada acta (PDF) es una sesión, ordinaria o extraordinaria
# (bo:Pleno, uno por acta, con su tipo de sesión: también las sesiones sin
# proposiciones; construccion/enriquecer.py).
def _consulta_plenos(q: str, a: Analisis):
    otras = [w for w in a.asunto if not re.match(r"ordinari|extraordinari|sesion|celebr", w)]
    if a.temas or otras or a.grupos:
        return None
    cuerpo = "  ?s a bo:Pleno ; bo:actaPdf ?pdf ; bo:fechaSesion ?d ; bo:tipoSesion ?tipo .\n"
    if a.anios:
        cuerpo += f"  ?s bo:anio ?anio . FILTER(?anio >= {a.anios[0]} && ?anio <= {a.anios[1]})\n"
    cuerpo += _filtro_fechas(a)
    if a.meses:
        cuerpo += f"  ?s bo:mesSesion ?mes . FILTER(?mes IN ({', '.join(str(m) for m in a.meses)}))\n"
    if re.search(r"\bextraordinari", q):
        cuerpo += '  FILTER(?tipo = "extraordinaria")\n'
    elif re.search(r"\bordinari", q):
        cuerpo += '  FILTER(?tipo = "ordinaria")\n'
    return f"SELECT ?tipo (COUNT(DISTINCT ?pdf) AS ?plenos) WHERE {{\n{cuerpo}}} GROUP BY ?tipo"


# El primer pleno de un periodo (una legislatura, un año) y los puntos que se
# trataron en él; una sesión sin proposiciones (la de constitución) sale igual
def _consulta_primer_pleno(q: str, a: Analisis):
    if a.temas or a.asunto or a.grupos:
        return None
    cuerpo = "    ?s a bo:Pleno ; bo:fechaSesion ?d ; bo:tipoSesion ?tipo .\n"
    if a.anios:
        cuerpo += f"    ?s bo:anio ?anio . FILTER(?anio >= {a.anios[0]} && ?anio <= {a.anios[1]})\n"
    cuerpo += _filtro_fechas(a).replace("  FILTER", "    FILTER")
    return ("SELECT ?fecha ?tipo ?titulo ?resumen WHERE {\n"
            f"  {{ SELECT ?s ?d ?tipo WHERE {{\n{cuerpo}  }} ORDER BY ?d LIMIT 1 }}\n"
            "  ?s rdfs:label ?fecha .\n"
            "  OPTIONAL { ?p bo:enPleno ?s ; bo:tituloTopic ?titulo . OPTIONAL { ?p bo:resumen ?resumen } }\n"
            "} ORDER BY ?titulo")


# Consulta para la pregunta, o None si no se puede montar con seguridad.
# props_asunto: proposiciones del asunto; criterio: texto que explica al
# narrador cómo se han elegido si no se pudieron exigir todas las palabras.
def consulta_directa(pregunta: str, a: Analisis, props_asunto=None, criterio: str = ""):
    q = _norm(pregunta)
    f = forma(q)
    # "¿qué se aprobó/debatió en el pleno del 15 de agosto de 2016?": lo de ese día
    if not f and a.fecha:
        f = "lista"
    # "¿cuánto dinero / por qué importe...?": los puntos del asunto con importe
    # en el título (subvenciones, créditos); sin asunto no se sabe de qué importe
    if a.importe:
        if not (a.asunto_raices or a.asunto_temas or a.temas):
            return None
        f = "importe"
    if not f or a.fuera_de_alcance or a.compara_anios or a.anio_ahora or (
            _NO_SOPORTADO.search(q) and f not in _FORMAS_INTERV):
        return None
    if f == "plenos":
        return _consulta_plenos(q, a)
    if f == "primer_pleno":
        return _consulta_primer_pleno(q, a)
    if f != "votos" and _VOTO.search(q):
        return None
    if len(a.temas) > 1:
        return None
    # un nombre propio que no se ha reconocido como tema, grupo ni asunto
    # (una persona, un lugar sin "sobre") dejaría la consulta sin su filtro
    cubierto = _norm(" ".join(a.asunto) + " " + " ".join(e for _, e in a.temas + a.temas_absorbidos) + " " +
                     " ".join(n.lower() for _, n in a.grupos))
    # ("el Plan contra la soledad": "Plan" va en mayúscula pero es una palabra de acción)
    for w in a.nombres_propios:
        if w not in _PROPIOS_NEUTROS and w not in _ACCIONES and w[:5] not in cubierto and len(w) >= 4:
            return None
    if a.asunto and not a.asunto_raices:
        return None
    if f in ("ranking_grupo", "desglose_grupo") and a.grupos:
        return None
    # sin tema ni asunto, "existe", "contenido" y "votos" no tienen de qué tratar
    if f in ("existe", "contenido", "votos") and not a.temas and not a.asunto_raices and not a.asunto_temas:
        return None
    # en "¿qué opinó/votó el PP...?" el grupo es quien habla o vota, no quien presenta
    votantes = (f == "votos" and _grupos_votantes(q, a)) or f == "contenido" or f in _FORMAS_INTERV

    cuerpo = "  ?p a bo:Proposicion ; bo:fechaISO ?d ; bo:fecha ?fecha .\n" + _FILTRO_PROPOSICION
    if a.ciudadana or f == "ranking_particular":
        cuerpo += '  ?p bo:tipoPunto "proposicion_ciudadana" .\n'
    if a.temas:
        cuerpo += f"  ?p bo:trataTemaAmplio br:{a.temas[0][0]} .\n"
    if a.asunto_raices or a.asunto_temas:
        if props_asunto is None:
            return None
        cuerpo += _bloque_asunto(a.asunto_raices + [f"tema {t}" for t in a.asunto_temas], props_asunto)
    if criterio:
        cuerpo += f'  BIND("{criterio}" AS ?criterio)\n'
    if a.grupos and not votantes:
        uris = sorted({u for us, _ in a.grupos for u in us})
        cuerpo += f"  ?p bo:presentadaPor ?gf . VALUES ?gf {{ {' '.join('br:' + u for u in uris)} }}\n"
    if a.oposicion:
        cuerpo += f"  ?p bo:presentadaPor ?go . FILTER(?go NOT IN ({_GOBIERNO}))\n"
    if a.anios:
        ini, fin = a.anios
        cuerpo += f"  ?p bo:anio ?anio . FILTER(?anio >= {ini} && ?anio <= {fin})\n"
    cuerpo += _filtro_fechas(a)
    if a.fecha:
        cuerpo += f'  FILTER(?d = "{a.fecha}"^^xsd:date)\n'
    if a.meses:
        cuerpo += f"  ?p bo:mes ?mes . FILTER(?mes IN ({', '.join(str(m) for m in a.meses)}))\n"
    if a.resultado:
        cuerpo += f"  ?p bo:tieneResultado ?r . FILTER(?r IN ({_RESULTADOS[a.resultado]}))\n"
    if a.unanimidad:
        # con voto por grupo registrado y sin ningún grupo en contra ni abstenido
        cuerpo += ("  FILTER EXISTS { ?p bo:votoAFavorDe ?_gv }\n"
                   "  FILTER NOT EXISTS { ?p bo:votoEnContraDe ?_gc }\n"
                   "  FILTER NOT EXISTS { ?p bo:seAbstuvo ?_ga }\n")
    crit = " ?criterio" if criterio else ""

    if f == "conteo":
        return f"SELECT (COUNT(DISTINCT ?p) AS ?n){crit} WHERE {{\n{cuerpo}}}" + (" GROUP BY ?criterio" if crit else "")
    if f in ("ranking_grupo", "desglose_grupo"):
        return (f"SELECT ?grupo (COUNT(DISTINCT ?p) AS ?n){crit} WHERE {{\n" + cuerpo +
                "  ?p bo:presentadaPor ?g . ?g rdfs:label ?grupo . FILTER(?g != br:grupo_desconocido)\n"
                f"}} GROUP BY ?grupo{crit} ORDER BY DESC(?n) LIMIT 20")
    if f == "ranking_particular":
        # quién presenta las proposiciones ciudadanas: asociación o particular
        return (f"SELECT ?presentadaPor (COUNT(DISTINCT ?p) AS ?n){crit} WHERE {{\n" + cuerpo +
                "  ?p bo:presentadaPorParticular ?e . ?e rdfs:label ?presentadaPor .\n"
                f"}} GROUP BY ?presentadaPor{crit} ORDER BY DESC(?n) LIMIT 15")
    if f in ("ranking_barrio", "ranking_distrito"):
        # barrio o distrito que nombra el punto (título, resumen o entidades)
        pred, var = ("bo:enBarrio", "barrio") if f == "ranking_barrio" else ("bo:enDistrito", "distrito")
        return (f"SELECT ?{var} (COUNT(DISTINCT ?p) AS ?n){crit} WHERE {{\n" + cuerpo +
                f"  ?p {pred} ?_lugar . ?_lugar rdfs:label ?{var} .\n"
                f"}} GROUP BY ?{var}{crit} ORDER BY DESC(?n) LIMIT 15")
    if f == "ranking_pleno":
        return (f"SELECT ?pleno ?tipo (COUNT(DISTINCT ?p) AS ?n){crit} WHERE {{\n" + cuerpo +
                "  ?p bo:enPleno ?_s . ?_s rdfs:label ?pleno ; bo:tipoSesion ?tipo .\n"
                f"}} GROUP BY ?pleno ?tipo{crit} ORDER BY DESC(?n) LIMIT 5")
    if f == "ranking_legislatura":
        return (f"SELECT ?legislatura (COUNT(DISTINCT ?p) AS ?n){crit} WHERE {{\n" + cuerpo +
                "  ?p bo:enLegislatura ?_l . ?_l rdfs:label ?legislatura .\n"
                f"}} GROUP BY ?legislatura{crit} ORDER BY DESC(?n) LIMIT 6")
    if f in _FORMAS_INTERV:
        # los grupos de la pregunta son quien habla, no quien presenta el punto
        grupo_orador = ""
        if a.grupos:
            uris = sorted({u for us, _ in a.grupos for u in us})
            grupo_orador = f"  ?i bo:grupoOrador ?_go . VALUES ?_go {{ {' '.join('br:' + u for u in uris)} }}\n"
        base = cuerpo + "  ?i bo:intervencionEn ?p .\n" + grupo_orador
        # "¿cuántas intervenciones hizo el alcalde...?": quien habla es el alcalde. Sin este
        # filtro se contaban las de todos los oradores (en la octava evaluación: 932 en vez
        # de unas 80). La etiqueta del orador es "SR. ALCALDE" o, en euskera, "ALKATE JN.";
        # "ALKATE ORDEA" (teniente de alcalde) no es el alcalde.
        if f == "conteo_interv" and re.search(r"\balcald(?:e|es|ia)\b", q) and not re.search(
                r"\b(sin|excepto|salvo|menos)\b.*\balcald", q):
            base += ('  ?i bo:oradorEtiqueta ?_etq . '
                     'FILTER(REGEX(?_etq, "alcald|alkate jn|alkate jauna", "i"))\n')
        if f == "ranking_orador":
            return (f"SELECT ?orador (COUNT(DISTINCT ?i) AS ?n){crit} WHERE {{\n" + base +
                    "  ?i bo:oradorEtiqueta ?etq . OPTIONAL { ?i bo:orador ?_c . ?_c rdfs:label ?cl }\n"
                    "  BIND(COALESCE(?cl, ?etq) AS ?orador)\n"
                    f"}} GROUP BY ?orador{crit} ORDER BY DESC(?n) LIMIT 15")
        if f == "ranking_grupo_interv":
            return (f"SELECT ?grupo (COUNT(DISTINCT ?i) AS ?n){crit} WHERE {{\n" + base +
                    "  ?i bo:grupoOrador ?g . ?g rdfs:label ?grupo .\n"
                    f"}} GROUP BY ?grupo{crit} ORDER BY DESC(?n) LIMIT 15")
        return f"SELECT (COUNT(DISTINCT ?i) AS ?n){crit} WHERE {{\n{base}}}" + (" GROUP BY ?criterio" if crit else "")
    if f == "contenido":
        # lo que dijeron en los puntos del asunto: resumen y postura de cada intervención
        # (las 8 proposiciones más recientes del asunto)
        filtro = ""
        if a.grupos:
            uris = sorted({u for us, _ in a.grupos for u in us})
            filtro = f"  ?i bo:grupoOrador ?_go . VALUES ?_go {{ {' '.join('br:' + u for u in uris)} }}\n"
        return (f"SELECT ?p ?fecha ?orador ?grupo ?postura ?resumen{crit} WHERE {{\n"
                f"  {{ SELECT DISTINCT ?p ?d ?fecha{crit} WHERE {{\n{cuerpo}  }} ORDER BY DESC(?d) LIMIT 8 }}\n"
                "  ?i bo:intervencionEn ?p ; bo:resumenIntervencion ?resumen ; bo:posturaIntervencion ?postura ;\n"
                "     bo:ordenIntervencion ?orden .\n" + filtro +
                "  OPTIONAL { ?i bo:oradorEtiqueta ?orador }\n"
                "  OPTIONAL { ?i bo:grupoOrador ?_g . ?_g rdfs:label ?grupo }\n"
                "} ORDER BY DESC(?d) ?p ?orden LIMIT 60")
    if f == "importe":
        return (f"SELECT ?p ?fecha ?titulo ?importe ?beneficiario ?resultado{crit} WHERE {{\n" + cuerpo +
                "  ?p bo:importe ?importe ; bo:tituloTopic ?titulo .\n"
                "  OPTIONAL { ?p bo:beneficiario ?beneficiario }\n"
                "  OPTIONAL { ?p bo:tieneResultado ?r2 . BIND(STRAFTER(STR(?r2), \"#\") AS ?resultado) }\n"
                f"}} ORDER BY DESC(?d) LIMIT 25")
    if f == "ranking_anio":
        if "?anio" not in cuerpo:
            cuerpo += "  ?p bo:anio ?anio .\n"
        return (f"SELECT ?anio (COUNT(DISTINCT ?p) AS ?n){crit} WHERE {{\n{cuerpo}}} "
                f"GROUP BY ?anio{crit} ORDER BY DESC(?n) LIMIT 10")
    rel = " ?relacion" if "?relacion" in cuerpo else ""
    # en las proposiciones ciudadanas, quién las presenta en vez de "Desconocido"
    extra = ("  OPTIONAL { ?p bo:presentadaPor ?g2 . ?g2 rdfs:label ?grupo0 }\n"
             "  OPTIONAL { ?p bo:presentadaPorParticular ?pp . ?pp rdfs:label ?particular }\n"
             "  BIND(COALESCE(?particular, ?grupo0) AS ?grupo)\n"
             "  OPTIONAL { ?p bo:tieneResultado ?r2 . BIND(STRAFTER(STR(?r2), \"#\") AS ?resultado) }\n")
    if f == "votos":
        # las 5 proposiciones más recientes del asunto y el voto de cada grupo;
        # sin fila de voto = no hay voto por grupo registrado para esa proposición
        sentido = []
        if re.search(r"en contra", q):
            sentido.append('{ ?p bo:votoEnContraDe ?gv . BIND("en contra" AS ?sentido) }')
        if re.search(r"a favor", q):
            sentido.append('{ ?p bo:votoAFavorDe ?gv . BIND("a favor" AS ?sentido) }')
        if re.search(r"abstuv|abstenc", q):
            sentido.append('{ ?p bo:seAbstuvo ?gv . BIND("abstención" AS ?sentido) }')
        if not sentido:
            sentido = ['{ ?p bo:votoAFavorDe ?gv . BIND("a favor" AS ?sentido) }',
                       '{ ?p bo:votoEnContraDe ?gv . BIND("en contra" AS ?sentido) }',
                       '{ ?p bo:seAbstuvo ?gv . BIND("abstención" AS ?sentido) }']
        filtro_votante = ""
        if votantes:
            uris = sorted({u for us, _ in a.grupos for u in us})
            filtro_votante = f" FILTER(?gv IN ({', '.join('br:' + u for u in uris)}))"
        return (f"SELECT ?p ?fecha ?grupo ?resultado{rel}{crit} ?votacion ?grupoVoto ?sentido WHERE {{\n"
                f"  {{ SELECT DISTINCT ?p ?d ?fecha{rel}{crit} WHERE {{\n{cuerpo}  }} ORDER BY DESC(?d) LIMIT 5 }}\n"
                f"{extra}"
                "  OPTIONAL { ?p bo:votacionRegistrada ?votacion }\n"
                f"  OPTIONAL {{ {' UNION '.join(sentido)} ?gv rdfs:label ?grupoVoto .{filtro_votante} }}\n"
                "} ORDER BY DESC(?d)")
    # "la última vez": las 3 más recientes que TRATAN del asunto y las 3 que solo
    # lo mencionan. Con las 5 más recientes a secas salían solo menciones y el
    # narrador concluía que ninguna proposición trataba del Teatro Arriaga
    if f == "ultima" and rel:
        sub = lambda r: (f"  {{ SELECT DISTINCT ?p ?d ?fecha ?relacion{crit} WHERE {{\n{cuerpo}"
                         f"  FILTER(?relacion = \"{r}\") }} ORDER BY DESC(?d) LIMIT 3 }}\n")
        return (f"SELECT DISTINCT ?p ?fecha ?grupo ?resultado ?relacion{crit} WHERE {{\n"
                f"{{\n{sub('trata')}}} UNION {{\n{sub('menciona')}}}\n{extra}}} ORDER BY DESC(?d)")
    limite = {"ultima": 5, "existe": 10, "contenido": 15}.get(f, 40)
    return (f"SELECT DISTINCT ?p ?fecha ?grupo ?resultado{rel}{crit} WHERE {{\n{cuerpo}{extra}}} "
            f"ORDER BY DESC(?d) LIMIT {limite}")
