"""
Votaciones: lo que el acta dice al votar cada punto y cómo se lleva al grafo.
Lo usan extraer.py (listas nominales) y enriquecer.py (lo demás).

  - listas nominales de voto del texto de un acta
  - voto de cada grupo a partir de esas listas
  - corrección de los recuentos que no suman
  - corrección del resultado con la fórmula que usa el acta
  - autor de las enmiendas sin grupo
"""
import json
import os
import re
import unicodedata
from collections import defaultdict

from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from comun.rutas import dato_grafo


# =============================================================================
# Listas nominales de voto
# =============================================================================

# Listas nominales de voto del texto de un acta ("votos afirmativos: señoras y
# señores X, Y...") separadas por sentido del voto. Las usa extraer.py.

# etiqueta de cada bloque -> sentido normalizado
_ETIQUETAS = [
    (r"[Vv]otos?\s+(?:afirmativos?|positivos?|a\s+favor|baiezkoak|aldekoak)", "favor"),
    (r"[Vv]otos?\s+(?:negativos?|en\s+contra|ezezkoak|kontrakoak)", "contra"),
    (r"(?:[Aa]bsten(?:ciones?|tzioak))", "abstencion"),
]
# "señoras/señores:" o "jaun-andre:" tras el número
_TRAS_NUM = r"(?:se[ñn]or(?:as|es)?(?:\s*[/y]\s*se[ñn]or(?:as|es)?)?|jaun[\s./-]*andre(?:ak)?)\s*:?"
# la lista termina en el PRIMERO de estos. NO se usa "fin de frase" genérico
# (un apellido no lleva punto; el punto que aparezca es OCR o abreviatura).
_FIN_LISTA = re.compile(
    r"\bEn\s+su\s+virtud\b|\bSobre\s+la\s+base\b|\bEl\s+Pleno\b|\bLa\s+Presidencia\b"
    r"|\bProducido\b|\bMediante\b|\bAs[ií]\s|\bA\s+la\s+vista\b|\bCon\s+el\s"
    r"|\bEl\s+Ayuntamiento\b|\bY\s+dado\b|\bQueda\b|\bEsta\s+proposici[oó]n\b"
    r"|[Vv]otos?\s+(?:afirmativos?|negativos?|a\s+favor|en\s+contra|emitidos)"
    r"|[Aa]bsten(?:ciones?|tzioak)|\bDado\b|\bResultando\b|\bConsiderando\b"
    r"|Udalbatza\w*\s+Idazkaritza|Secretar[ií]a\s+General\s+del\s+Pleno",
)
# basura que se cuela en medio de la lista (cabecera bilingüe, nº de página)
_BASURA = re.compile(
    r"Udalbatza\w*\s+Idazkaritza\s+Nagusia|Secretar[ií]a\s+General\s+del\s+Pleno"
    r"|-\s*\d+\s*-|\bP[áa]gina\s+\d+")
_PARENT = re.compile(r"\([^)]*\)")           # "(emitido por delegación, ...)"
_OCR_SP = re.compile(r"\b([A-Za-zÀ-ÿ])\s+(?=[a-zà-ÿ])")   # "Ma drazo" -> "Madrazo"


def _limpia_token(tok: str) -> str:
    tok = _PARENT.sub("", tok)
    tok = re.sub(r"\s+", " ", tok).strip(" .,;:-")
    # unir letras sueltas de OCR dentro de una palabra ("Delgad o", "Dí ez")
    tok = re.sub(r"(\w)\s+(\w)(?=\s|$)", lambda m: m.group(1) + m.group(2)
                 if len(m.group(2)) <= 2 or len(m.group(1)) <= 2 else m.group(0), tok)
    tok = re.sub(r"\b([A-Za-zÀ-ÿ]{1,2})\s+([a-zà-ÿ]{2,})\b", r"\1\2", tok)
    return tok.strip(" .,;:-")


def _split_lista(lista: str):
    lista = _BASURA.sub(" ", lista)
    lista = _PARENT.sub(" ", lista)
    # separadores: ",", ";", " y ", " e ", " eta "
    trozos = re.split(r"\s*[,;]\s*|\s+y\s+|\s+e\s+|\s+eta\s+", lista)
    out = []
    for t in trozos:
        t = _limpia_token(t)
        if not t or re.search(r"\d", t):
            continue
        # nombre válido: 1-4 palabras, empieza en mayúscula, sin coletillas
        pal = t.split()
        if not (1 <= len(pal) <= 4):
            continue
        if not re.match(r"^[A-ZÑÁÉÍÓÚ]", t):
            continue
        low = t.lower()
        if low in ("del pleno", "general del pleno", "idazkaritza nagusia",
                   "secretaria general", "nagusia"):
            continue
        out.append(t)
    return out


def parse_listas_voto(texto: str):
    if not texto:
        return None
    t = re.sub(r"[ \t]*\n[ \t]*", " ", texto)
    res = {"favor": [], "contra": [], "abstencion": [],
           "_conteo_declarado": {}, "_cuadra": None}
    encontrado = False
    for rx, sentido in _ETIQUETAS:
        m = re.search(rx + r"\s*:?\s*([\d ]{1,4})?\s*" + _TRAS_NUM + r"\s*(.+)", t)
        if not m:
            continue
        n_decl = None
        if m.group(1) and m.group(1).strip():
            n_decl = int(re.sub(r"\s+", "", m.group(1)))
        resto = m.group(2)
        fin = _FIN_LISTA.search(resto)
        lista = resto[:fin.start()] if fin else resto[:600]
        nombres = _split_lista(lista)
        if not nombres:
            continue
        # sobre-captura: la lista se comió parte de otra (revote, enmienda,
        # proposición siguiente). Si hay conteo declarado y extrajimos bastantes
        # más, recorta a los primeros n (el arranque de la lista sí es fiable).
        if n_decl is not None and len(nombres) > n_decl + 1:
            nombres = nombres[:n_decl]
        encontrado = True
        res[sentido] = nombres
        if n_decl is not None:
            res["_conteo_declarado"][sentido] = n_decl
    if not encontrado:
        return None
    # ¿cuadran los conteos declarados con los nombres extraídos?
    cuadra = True
    for sen, n in res["_conteo_declarado"].items():
        if abs(len(res[sen]) - n) > 1:      # tolerancia 1 (OCR puede partir un nombre)
            cuadra = False
    res["_cuadra"] = cuadra
    return res


# =============================================================================
# Voto por grupo a partir de las votaciones nominales
# =============================================================================

# Voto por grupo a partir de las votaciones nominales del texto de las actas.
#
# El grafo tiene el voto por grupo en algo más de la mitad de las proposiciones.
# En otras muchas, el acta trae la votación nominal ("votos afirmativos: 17
# señoras/señores: gil, diez, ...; abstenciones: 4 señoras/señores: eguiluz,
# marcos, carrón y fernández") y el grafo solo guardó el recuento: en la cuarta
# evaluación, "¿cómo votó el PP los presupuestos de 2016?" no tenía respuesta
# aunque el acta dice que el PP se abstuvo.
#
# Para cada proposición sin voto por grupo se busca, entre los fragmentos de su
# debate, una votación nominal cuyos recuentos coincidan con los que el grafo ya
# tiene (así se toma la votación decisiva y no la de una enmienda intermedia).
# Cada apellido se asigna a su concejal (y a su grupo) con los 84 concejales del
# grafo; si un apellido es ambiguo, por el periodo en que ese concejal aparece en
# las actas. Un grupo solo recibe voto si todos sus concejales reconocidos votaron
# lo mismo.

BO = Namespace("http://bilbao.tfg/ontology#")
PREFIJO_RECURSO = "http://bilbao.tfg/resource/"

# "señoras/señores" puede llegar partido por el OCR ("seno res") y la lista
# puede quedar cortada al final del fragmento, sin punto
_LISTA = r"(\d+)\s*seno\s*r\w*\s*/\s*seno\s*r\w*\s*:\s*(.+?)(?:\.|$)"
_AFIRMATIVOS = re.compile(rf"votos afirmativos:\s*{_LISTA}", re.S)
_NEGATIVOS = re.compile(rf"votos negativos:\s*{_LISTA}", re.S)
_ABSTENCIONES = re.compile(rf"abstenciones:\s*{_LISTA}", re.S)


# votaciones nominales de un fragmento: [(n_favor, favor, n_contra, contra, n_abst, abst)].
# Entre una lista y la siguiente puede haber la versión en euskera
# ("abstentzioak: 12 jaun-andre: ...") o una cabecera de página
def _votaciones(norm):
    out = []
    for m in _AFIRMATIVOS.finditer(norm):
        tramo = norm[m.end():m.end() + 1500]
        corte = re.search(r"votos (afirmativos|emitidos)", tramo)
        tramo = tramo[:corte.start()] if corte else tramo
        neg, abst = _NEGATIVOS.search(tramo), _ABSTENCIONES.search(tramo)
        out.append((m.group(1), m.group(2), neg.group(1) if neg else "0", neg.group(2) if neg else "",
                    abst.group(1) if abst else "0", abst.group(2) if abst else ""))
    return out


_PREDICADO = {"favor": BO.votoAFavorDe, "contra": BO.votoEnContraDe, "abst": BO.seAbstuvo}


def _norm(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _concejales(g, ejecutar, prefijos):
    conc = {}
    for r in ejecutar(g, prefijos + "SELECT ?c ?l ?g WHERE { ?c a bo:Concejal ; rdfs:label ?l ; bo:perteneceA ?g }"):
        tokens = re.findall(r"[a-z]+", _norm(r["l"]))
        claves = {"".join(tokens[i:j]) for i in range(len(tokens)) for j in range(i + 1, len(tokens) + 1)}
        conc[r["c"]] = {"grupo": r["g"], "claves": claves, "ini": "0000", "fin": "9999"}
    for r in ejecutar(g, prefijos + """SELECT ?c (MIN(?d) AS ?ini) (MAX(?d) AS ?fin) WHERE { ?p bo:fechaISO ?d .
        { ?p bo:intervino ?c } UNION { ?p bo:concejalVotoAFavor ?c } UNION { ?p bo:concejalVotoEnContra ?c }
        UNION { ?p bo:concejalVotoAbstencion ?c } } GROUP BY ?c"""):
        if r["c"] in conc:
            conc[r["c"]]["ini"], conc[r["c"]]["fin"] = r["ini"], r["fin"]
    return conc


# grupo de un apellido de la lista en una fecha, o None si no se sabe
def _grupo(nombre, fecha, conc, alcalde_grupo):
    clave = re.sub(r"[^a-z]", "", nombre)
    if clave == "alcalde":
        return alcalde_grupo
    cands = [c for c in conc.values() if clave in c["claves"]]
    if len(cands) > 1:
        cands = [c for c in cands if c["ini"] <= fecha <= c["fin"]]
    grupos = {c["grupo"] for c in cands}
    return grupos.pop() if len(grupos) == 1 else None


def _nombres(lista):
    return [n.strip() for n in re.split(r",|\by\b", lista) if n.strip()]


def anadir(g, ejecutar, prefijos, corpus, canonica=lambda pid: pid) -> int:
    conc = _concejales(g, ejecutar, prefijos)
    # todas las proposiciones con recuento; a las que ya tienen voto por grupo
    # solo se les añaden los grupos que faltan (en el presupuesto de 2016 el
    # grafo tenía los votos a favor y en contra, pero no la abstención del PP)
    sin = {}
    for r in ejecutar(g, prefijos + """SELECT ?p ?f ?c ?d WHERE {
            ?p a bo:Proposicion ; bo:votosFavor ?f ; bo:fechaISO ?d . OPTIONAL { ?p bo:votosContra ?c } }"""):
        if r.get("c") in (None, "None", ""):
            r["c"] = "0"
        sin[r["p"].rsplit("prop_", 1)[-1]] = r
    ya = defaultdict(set)
    for r in ejecutar(g, prefijos + """SELECT ?p ?gv WHERE { { ?p bo:votoAFavorDe ?gv } UNION { ?p bo:votoEnContraDe ?gv }
            UNION { ?p bo:seAbstuvo ?gv } }"""):
        ya[r["p"]].add(r["gv"])
    alcalde = PREFIJO_RECURSO + "grupo_eaj_pnv"
    # la votación más completa de cada proposición (una lista puede quedar
    # cortada al final de un fragmento y seguir en el siguiente)
    mejor = {}
    for norm, _o, _iso, _f, _punto, pid, *_ in corpus:
        pid = canonica(pid)
        if pid not in sin or "votos afirmativos" not in norm:
            continue
        info = sin[pid]
        for n_fav, fav, n_contra, contra, n_abst, abst in _votaciones(norm):
            if n_fav != info["f"] or n_contra != info["c"]:
                continue
            listas = {"favor": _nombres(fav), "contra": _nombres(contra), "abst": _nombres(abst)}
            completa = sum(len(v) for v in listas.values()) - abs(int(n_abst) - len(listas["abst"]))
            if pid not in mejor or completa > mejor[pid][0]:
                mejor[pid] = (completa, listas)
    completadas = 0
    for pid, (_, listas) in mejor.items():
        info = sin[pid]
        # el grafo no tiene a todos los concejales: un apellido común puede caer en
        # el concejal equivocado ("García" de Goazen -> un García del PP). El grupo
        # recibe el sentido de al menos el 75 % de sus concejales reconocidos
        por_grupo = defaultdict(lambda: defaultdict(int))
        for sentido, nombres in listas.items():
            for nombre in nombres:
                gr = _grupo(nombre, info["d"], conc, alcalde)
                if gr:
                    por_grupo[gr][sentido] += 1
        p = URIRef(info["p"])
        nuevos = 0
        for gr, cuenta in por_grupo.items():
            sentido, n = max(cuenta.items(), key=lambda x: x[1])
            if n >= 0.75 * sum(cuenta.values()) and gr not in ya[info["p"]]:
                g.add((p, _PREDICADO[sentido], URIRef(gr)))
                nuevos += 1
        if nuevos:
            # procedencia: votación nominal del acta (texto), no extracción del LLM
            g.add((p, BO.votoFuente, Literal("acta_nominal")))
        completadas += bool(nuevos)
    return completadas


# =============================================================================
# Recuentos de votación incoherentes
# =============================================================================

# Corrige los recuentos de votación incoherentes con lo que dice el acta.
#
# votaciones.jsonl (extraer.py) sale de un LLM, y en 158 de las 1.862 votaciones
# los recuentos no suman: 29 votos emitidos, 0 a favor y 25 en contra. Era el caso de la
# proposición del PP sobre el Palacio de Justicia (28-09-2023): el acta dice "Votos
# afirmativos: 4 señoras/señores: Martínez, Rodrigo, Goti y Garagalza", el grafo tenía
# 0 a favor y daba al PP como grupo en contra (octava evaluación, o22).
#
# Las actas traen el recuento y la lista nominal de cada votación ("Votos emitidos: 29",
# "Votos afirmativos: 4 señoras/señores: ...", "Votos negativos: ...", "Abstenciones:
# ..."), pero el OCR parte las palabras ("Votos af irmativos", "s eñores") y la sección
# anterior, que busca el texto exacto, no las lee. Aquí se buscan sin espacios.
#
# Para cada votación con recuentos que no suman se toma, del texto de su proposición, el
# recuento con los mismos votos emitidos cuyos números coincidan con los que el grafo ya
# tiene (un 0 en el grafo se considera "no extraído"). Solo se corrige si hay UN candidato
# y suma bien; las dudosas se dejan como están. Se rehacen los recuentos y el voto de
# cada grupo de la votación (misma regla que la sección anterior: el sentido de al menos el
# 75 % de sus concejales reconocidos); si es la votación decisiva del punto, también los
# recuentos y el voto por grupo de la proposición, con bo:votoFuente = "acta_nominal".

def _t(s: str) -> str:
    # la palabra o frase con espacios opcionales entre letras
    return r"\s*".join(re.escape(c) for c in s.replace(" ", ""))


_EMITIDOS = re.compile(_t("votos emitidos") + r"\s*:\s*(\d+)", re.I)
_LISTA_OCR = r"\s*:?\s*(\d+)\s*" + _t("señoras/señores") + r"\s*:\s*(.+?)(?:\.|$)"
_AFIRM = re.compile(_t("votos afirmativos") + _LISTA_OCR, re.I | re.S)
_NEGAT = re.compile(_t("votos negativos") + _LISTA_OCR, re.I | re.S)
_ABSTE = re.compile(_t("abstenciones") + _LISTA_OCR, re.I | re.S)
# versión sin lista ("Votos afirmativos: 4" a secas)
_SOLO = {k: re.compile(_t(p) + r"\s*:\s*(\d+)", re.I)
         for k, p in (("favor", "votos afirmativos"), ("contra", "votos negativos"), ("abst", "abstenciones"))}


def _votaciones_texto(texto):
    """[(emitidos, favor, contra, abst, {'favor': [...], 'contra': [...], 'abstencion': [...]})]"""
    out = []
    for m in _EMITIDOS.finditer(texto):
        seg = texto[m.end(): m.end() + 2500]
        corte = _EMITIDOS.search(seg)
        seg = seg[:corte.start()] if corte else seg
        n, listas = {}, {}
        for clave, rx, sentido in (("favor", _AFIRM, "favor"), ("contra", _NEGAT, "contra"),
                                   ("abst", _ABSTE, "abstencion")):
            lm = rx.search(seg)
            if lm:
                n[clave] = int(lm.group(1))
                listas[sentido] = [x.strip() for x in re.split(r",|\by\b", lm.group(2)) if x.strip()]
            else:
                sm = _SOLO[clave].search(seg)
                n[clave] = int(sm.group(1)) if sm else 0
        out.append((int(m.group(1)), n["favor"], n["contra"], n["abst"], listas))
    return out


def corregir_recuentos(g, corpus, canonica, ejecutar, prefijos) -> dict:
    conc = _concejales(g, ejecutar, prefijos)
    alcalde = "http://bilbao.tfg/resource/grupo_eaj_pnv"
    texto_de = defaultdict(list)
    for _n, orig, _iso, _f, _punto, pid, *_ in corpus:
        if pid:
            texto_de[canonica(pid)].append(orig)

    filas = ejecutar(g, prefijos + """SELECT ?v ?p ?e ?f ?c ?a ?fecha WHERE {
        ?v a bo:Votacion ; bo:votosEmitidosVotacion ?e ; bo:votosFavorVotacion ?f ;
           bo:votosContraVotacion ?c ; bo:abstencionesVotacion ?a . ?p bo:tieneVotacion ?v ; bo:fechaISO ?fecha }""")
    cuenta = defaultdict(int)
    pred_nodo = {"favor": BO.grupoVotaAFavor, "contra": BO.grupoVotaEnContra, "abstencion": BO.grupoSeAbstiene}
    pred_prop = {"favor": BO.votoAFavorDe, "contra": BO.votoEnContraDe, "abstencion": BO.seAbstuvo}
    for r in filas:
        e, f, c, a = (int(r[k]) for k in ("e", "f", "c", "a"))
        if e == f + c + a:
            continue
        cuenta["incoherentes"] += 1
        pid = r["p"].rsplit("prop_", 1)[-1]
        cand = {}
        for t in texto_de.get(pid, []):
            for te, tf, tc, ta, listas in _votaciones_texto(t):
                if te == e and tf + tc + ta == e and all(gv in (0, tv) for gv, tv in ((f, tf), (c, tc), (a, ta))):
                    ant = cand.get((tf, tc, ta))
                    # la lista más completa de los fragmentos que traen el mismo recuento
                    if ant is None or sum(map(len, listas.values())) > sum(map(len, ant.values())):
                        cand[(tf, tc, ta)] = listas
        if len(cand) != 1:
            cuenta["sin_candidato_unico"] += 1
            continue
        (tf, tc, ta), listas = next(iter(cand.items()))
        nodo, p = URIRef(r["v"]), URIRef(r["p"])
        for pred, valor in ((BO.votosFavorVotacion, tf), (BO.votosContraVotacion, tc), (BO.abstencionesVotacion, ta)):
            g.set((nodo, pred, Literal(valor, datatype=XSD.integer)))
        cuenta["recuentos_corregidos"] += 1
        # voto de cada grupo, solo si el acta trae las listas completas (nombres = recuento)
        completas = all(len(listas.get(s, [])) == n for s, n in (("favor", tf), ("contra", tc), ("abstencion", ta)))
        if not completas:
            continue
        por_grupo = defaultdict(lambda: defaultdict(int))
        for sentido, nombres in listas.items():
            for nombre in nombres:
                gr = _grupo(_norm(nombre), r["fecha"], conc, alcalde)
                if gr:
                    por_grupo[gr][sentido] += 1
        nuevos = {}
        for gr, cu in por_grupo.items():
            sentido, n = max(cu.items(), key=lambda x: x[1])
            if n >= 0.75 * sum(cu.values()):
                nuevos[gr] = sentido
        for pr in pred_nodo.values():
            for o in list(g.objects(nodo, pr)):
                g.remove((nodo, pr, o))
        for gr, sentido in nuevos.items():
            g.add((nodo, pred_nodo[sentido], URIRef(gr)))
        cuenta["grupos_rehechos"] += 1
        # la votación decisiva fija el voto de la proposición
        if (nodo, BO.esDecisiva, Literal(True)) in g:
            for pr in pred_prop.values():
                for o in list(g.objects(p, pr)):
                    g.remove((p, pr, o))
            for gr, sentido in nuevos.items():
                g.add((p, pred_prop[sentido], URIRef(gr)))
            g.set((p, BO.votosFavor, Literal(tf, datatype=XSD.integer)))
            g.set((p, BO.votosContra, Literal(tc, datatype=XSD.integer)))
            g.set((p, BO.votoFuente, Literal("acta_nominal")))
            cuenta["proposiciones_corregidas"] += 1
    return dict(cuenta)


# =============================================================================
# Resultado según la fórmula del acta
# =============================================================================

# Corrige el resultado de las proposiciones con lo que dice el acta al votarlas.
#
# El resultado de muchos puntos se sacó con un LLM de un texto que a veces incluía la
# votación del punto siguiente: constaban "rechazadas" proposiciones que habían decaído
# por una enmienda del Equipo de Gobierno ("se acepta la enmienda ..., por lo que decae
# la proposición ..."). Cada punto guarda en votaciones.jsonl las decisiones de sus
# votaciones (extraer.py); la última que resuelve el punto con una fórmula
# explícita fija su resultado:
#
#   "por lo que decae(n) la proposición/iniciativa/moción/propuesta"  -> Decae
#   "se rechaza / queda rechazada la proposición/..."                  -> Rechazada
#
# No se tocan los resultados "aprobada": distinguir Aprobada de AprobadaConEnmienda
# exige leer la enmienda y la fórmula varía. Solo se corrige lo que el acta dice sin
# ambigüedad.

BR = Namespace("http://bilbao.tfg/resource/")

# Los PDF traen espacios dentro de las palabras ("por l o que decae", "se rec haza"):
# se compara sin ningún espacio.
_PROP = r"(?:proposicion|iniciativa|mocion|propuesta|dictamen)"
_DECAE = re.compile(rf"porloquedecaen?(?:tanto)?(?:la|el){_PROP}")
_RECHAZA = re.compile(rf"(?:serechazan?|quedarechazad[ao]|quedadesestimad[ao])(?:la|el){_PROP}")
_APRUEBA = re.compile(rf"(?:seaprueban?|seacepta|quedaaprobad[ao])(?:porunanimidad)?(?:la|el){_PROP}")
TIPOS = ("proposicion_grupo", "proposicion_ciudadana", "propuesta_gobierno")


def _sin_espacios(s):
    t = "".join(c for c in unicodedata.normalize("NFD", (s or "").lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", "", t)


def resultado_explicito(decisiones):
    """'Decae' | 'Rechazada' según la última decisión que resuelve el punto de forma explícita;
    None si la última es una aprobación o ninguna resuelve."""
    # la primera decisión que resuelve la proposición es la del propio punto: las
    # siguientes, si el segmento se ha tragado puntos sin cabecera reconocible, son
    # de esos puntos (el 27-05-2021 el punto 30 se rechaza y a continuación el segmento
    # trae la decisión "decae" del 31)
    for d in decisiones:
        t = _sin_espacios(d)
        if _DECAE.search(t):
            return "Decae"
        if _RECHAZA.search(t):
            return "Rechazada"
        if _APRUEBA.search(t):
            return None
    return None


def corregir_resultados(g, copia_de=None) -> dict:
    ruta = dato_grafo("votaciones.jsonl")
    if not os.path.exists(ruta):
        return {}
    copia_de = copia_de or {}
    cuenta = defaultdict(int)
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            v = json.loads(linea)
            pr = BR[f"prop_{copia_de.get(v['id'], v['id'])}"]
            if (pr, RDF.type, BO.PuntoOrdenDia) not in g or str(g.value(pr, BO.tipoPunto) or "") not in TIPOS:
                continue
            nuevo = resultado_explicito([a.get("decision", "") for a in v["votaciones"]])
            actual = g.value(pr, BO.tieneResultado)
            if nuevo is None or actual is None or str(actual).endswith("#" + nuevo):
                continue
            g.remove((pr, BO.tieneResultado, actual))
            g.add((pr, BO.tieneResultado, BO[nuevo]))
            g.set((pr, BO.resultadoFuente, Literal("votacion_acta")))
            cuenta[f"{str(actual).split('#')[-1]}->{nuevo}"] += 1
    return dict(cuenta)


# =============================================================================
# Autor de las enmiendas sin grupo
# =============================================================================

# Autor de las enmiendas que la extracción no atribuyó a ningún grupo.
#
# Comprobado leyendo el acta (no por inferencia): en el pleno del 22-03-2018
# (ordenanza de igualdad) las enmiendas van agrupadas por grupo y por número de
# registro: "I.- Enmiendas del Grupo GOAZEN BILBAO. Registro 55 y 65", "II.- PARTIDO
# POPULAR, 58 a 61", "III.- UDALBERRI-BILBAO EN COMÚN, 69 a 119", "IV.- EH BILDU,
# 121 a 154" (acta, págs. 39-62). En el pleno del 30-03-2017 (reglamento de los
# distritos), la enmienda "de adición del siguiente texto en el artículo 2" es de
# GOAZEN BILBAO (sección III, acta pág. 36).
#
# En el pleno del 31-10-2013 hay una "enmienda" (condicionar los cheques de compra
# básica) colgada de una dación de cuenta cuyo texto no figura en el acta y sin
# autor: se quita. Las demás enmiendas colgadas de daciones (10) son reales: la
# extracción las unió a la dación porque el texto del punto absorbe las
# proposiciones siguientes; se dejan.

# fecha -> [(número desde, hasta, grupo)]
RANGOS = {
    "2018-03-22": [(55, 55, "goazen_bilbao"), (65, 65, "goazen_bilbao"), (58, 61, "pp"),
                   (69, 119, "udalberri"), (121, 154, "eh_bildu")],
}
# fecha -> [(principio del título, grupo)]
POR_TITULO = {
    "2017-03-30": [("1. Propuesta de enmienda de adición de l siguiente texto en el artículo 2", "goazen_bilbao")],
}
# Enmiendas de 2021-2022 que la extracción original colgó de una dación de cuenta
# porque el texto del punto absorbía las proposiciones siguientes. Al recuperar
# esas proposiciones (texto_actas.py) cada una tiene ya su enmienda: la de la dación
# es un duplicado. (fecha, principio del resumen de la enmienda), verificadas una a una.
DUPLICADAS_EN_DACION = [
    ("2021-10-28", "Solicita analizar los futuros usos de la esta"),
    ("2022-05-26", "Establece directrices de uso prioritario y pr"),
    ("2022-02-24", "Se sustituye la proposición original por un t"),
    ("2022-03-31", "Acuerda que el Ayuntamiento y la Hermandad de"),
    ("2021-02-25", "Modifica los plazos para finalizar el estudio"),
    ("2021-06-24", "Insta a colaborar con el Gobierno Vasco en la"),
    ("2021-03-25", "Presentada conjuntamente por EAJ-PNV, PSE-EE "),
    ("2021-11-25", "Ajusta la petición a la Diputación para armon"),
]
_NUMERO = re.compile(r"n[uú]mero\s+(\d+)")


def _titulo(g, punto):
    return re.sub(r"\s+", " ", str(g.value(punto, BO.tituloTopic) or "")).strip()


def asignar(g) -> dict:
    cuenta = {"autor_asignado": 0, "dacion_quitadas": 0}
    for punto in set(g.subjects(BO.tieneEnmienda, None)):
        tipo = str(g.value(punto, BO.tipoPunto) or "")
        for enm in list(g.objects(punto, BO.tieneEnmienda)):
            resumen = str(g.value(enm, BO.resumenEnmienda) or "")
            fecha0 = str(g.value(punto, BO.fechaISO) or "")
            sin_autor = (enm, BO.enmiendaPor, None) not in g
            if tipo == "dacion_cuenta" and ((sin_autor and resumen.startswith("Condicionar los cheques de compra básica"))
                                            or any(fecha0 == f and resumen.startswith(r) for f, r in DUPLICADAS_EN_DACION)):
                for t in list(g.triples((enm, None, None))) + list(g.triples((None, None, enm))):
                    g.remove(t)
                cuenta["dacion_quitadas"] += 1
                continue
            if (enm, BO.enmiendaPor, None) in g:
                continue
            fecha = str(g.value(punto, BO.fechaISO) or "")
            titulo = _titulo(g, punto)
            grupo = None
            for inicio, gr in POR_TITULO.get(fecha, []):
                if titulo.startswith(inicio):
                    grupo = gr
            if grupo is None and fecha in RANGOS:
                m = _NUMERO.search(titulo)
                if m:
                    n = int(m.group(1))
                    grupo = next((gr for a, b, gr in RANGOS[fecha] if a <= n <= b), None)
            if grupo:
                g.add((enm, BO.enmiendaPor, BR[f"grupo_{grupo}"]))
                g.add((enm, BO.autorFuente, Literal("acta_verificada")))
                cuenta["autor_asignado"] += 1
    return cuenta
