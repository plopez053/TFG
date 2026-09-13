# -*- coding: utf-8 -*-

# shape: conteo | ranking | ratio | temporal | persona | voto_nominal | entidad | coautoria | total
GOLD = [
    # ---------------- conteos simples con tema ----------------
    dict(id="g01", shape="conteo",
         q="¿Cuántas proposiciones sobre movilidad y transporte se presentaron en 2019?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_movilidad ; bo:anio 2019 . }""",
         must=["24"], must_not=["0"]),
    dict(id="g02", shape="conteo",
         q="¿Cuántas proposiciones sobre euskera se han rechazado en total?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_euskera ; bo:tieneResultado bo:Rechazada . }""",
         must=["4"], must_not=["55", "60"]),
    dict(id="g03", shape="conteo",
         q="¿Cuántas proposiciones sobre desahucios se han presentado?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_desahucios . }""",
         must=["14"], must_not=["0"]),
    dict(id="g04", shape="conteo",
         q="¿Cuántas proposiciones sobre presupuestos y fiscalidad se han presentado en total, incluyendo subtemas?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_presupuestos . }""",
         must=["733"], must_not=["0"]),
    dict(id="g05", shape="conteo",
         q="¿Cuántas proposiciones sobre la bicicleta se han presentado en el Pleno?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_bicicleta . }""",
         must=["56"], must_not=["0"]),
    dict(id="g06", shape="conteo",
         q="¿Cuántas proposiciones se rechazaron en 2021?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:anio 2021 ; bo:tieneResultado bo:Rechazada . }""",
         must=["11"], must_not=["0"]),

    # ---------------- total sin desglose ----------------
    dict(id="g07", shape="total",
         q="¿Cuántas proposiciones hay en total en el grafo?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE { ?p a bo:Proposicion . }""",
         must=["3422"], must_not=["3923"]),
    dict(id="g08", shape="total",
         q="¿Cuántas proposiciones no llegaron a tener un resultado registrado?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:tieneResultado bo:SinResultado . }""",
         must=["_ANY_"], must_not=["0"]),

    # ---------------- conteo con grupo + año ----------------
    dict(id="g09", shape="ratio",
         q="¿Cuántas proposiciones presentó el Partido Popular en 2015 y cuántas se aprobaron?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?ap) AS ?aprob) WHERE {
           ?p a bo:Proposicion ; bo:presentadaPor br:grupo_pp ; bo:anio 2015 .
           OPTIONAL { ?p bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) . BIND(?p AS ?ap) } }""",
         must=["37", "18"]),
    dict(id="g10", shape="conteo",
         q="¿Cuántas proposiciones aprobó EH Bildu sobre vivienda?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:presentadaPor br:grupo_eh_bildu ; bo:trataTemaAmplio br:t_vivienda ;
              bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) }""",
         must=["23"], must_not=["0"]),
    dict(id="g11", shape="ratio",
         q="¿Cuántas proposiciones sobre movilidad ha presentado EH Bildu y cuántas se han rechazado?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?re) AS ?rech) WHERE {
           ?p a bo:Proposicion ; bo:presentadaPor br:grupo_eh_bildu ; bo:trataTemaAmplio br:t_movilidad .
           OPTIONAL { ?p bo:tieneResultado bo:Rechazada . BIND(?p AS ?re) } }""",
         must=["120", "15"]),
    dict(id="g12", shape="conteo",
         q="¿Cuántas proposiciones sobre turismo ha presentado EH Bildu?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:presentadaPor br:grupo_eh_bildu ; bo:trataTemaAmplio br:t_turismo . }""",
         must=["_ANY_"], must_not=["0"]),

    # ---------------- rankings por grupo ----------------
    dict(id="g13", shape="ranking",
         q="¿Qué tres grupos han presentado más proposiciones sobre medio ambiente?",
         ref="""SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_medioambiente ; bo:presentadaPor ?g .
           ?g rdfs:label ?ng . FILTER(?g != br:grupo_desconocido) }
           GROUP BY ?ng ORDER BY DESC(?n) LIMIT 3""",
         must=["EH BILDU"], top="EH BILDU"),
    dict(id="g14", shape="ranking",
         q="¿Qué grupo ha presentado más proposiciones sobre educación?",
         ref="""SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_educacion ; bo:presentadaPor ?g .
           ?g rdfs:label ?ng . FILTER(?g != br:grupo_desconocido) }
           GROUP BY ?ng ORDER BY DESC(?n) LIMIT 5""",
         must=["EH BILDU"], top="EH BILDU"),
    dict(id="g15", shape="ranking",
         q="¿Cuántas proposiciones sobre vivienda ha presentado cada grupo?",
         ref="""SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_vivienda ; bo:presentadaPor ?g .
           ?g rdfs:label ?ng . FILTER(?g != br:grupo_desconocido) }
           GROUP BY ?ng ORDER BY DESC(?n) LIMIT 20""",
         must=["EH BILDU", "62"], top="EH BILDU"),
    dict(id="g16", shape="ranking",
         q="¿Qué grupo ha presentado más enmiendas en total?",
         ref="""SELECT ?ng (COUNT(DISTINCT ?e) AS ?n) WHERE {
           ?e a bo:Enmienda ; bo:enmiendaPor ?g . ?g rdfs:label ?ng .
           FILTER(?g != br:grupo_desconocido) }
           GROUP BY ?ng ORDER BY DESC(?n) LIMIT 20""",
         must=["EQUIPO DE GOBIERNO"], top="EQUIPO DE GOBIERNO"),
    dict(id="g17", shape="conteo",
         q="¿Cuántas enmiendas ha presentado el Partido Popular en total?",
         ref="""SELECT (COUNT(DISTINCT ?e) AS ?n) WHERE { ?e a bo:Enmienda ; bo:enmiendaPor br:grupo_pp . }""",
         must=["24"], must_not=["936", "0"]),

    # ---------------- máximos temporales ----------------
    dict(id="g18", shape="temporal",
         q="¿En qué año se presentaron más proposiciones sobre seguridad?",
         ref="""SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_seguridad ; bo:anio ?anio . }
           GROUP BY ?anio ORDER BY DESC(?n) LIMIT 1""",
         must=["2025"], top="2025"),
    dict(id="g19", shape="temporal",
         q="¿En qué año se rechazaron más proposiciones en el Pleno de Bilbao?",
         ref="""SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:tieneResultado bo:Rechazada ; bo:anio ?anio . }
           GROUP BY ?anio ORDER BY DESC(?n) LIMIT 1""",
         must=["2018"], top="2018"),
    dict(id="g20", shape="temporal",
         q="¿Cómo ha evolucionado el número de proposiciones sobre euskera año a año?",
         ref="""SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_euskera ; bo:anio ?anio . }
           GROUP BY ?anio ORDER BY ASC(?anio)""",
         must=["2025"], min_rows=8),

    # ---------------- ratios / porcentajes ----------------
    dict(id="g21", shape="ratio",
         q="¿Qué porcentaje de las proposiciones sobre vivienda se han aprobado?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?ap) AS ?aprob) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_vivienda .
           OPTIONAL { ?p bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) . BIND(?p AS ?ap) } }""",
         must=["200"], must_not=["0"]),
    dict(id="g22", shape="ratio",
         q="¿Qué grupo tiene mejor tasa de aprobación de sus proposiciones, EH Bildu o el PP?",
         ref="""SELECT ?g (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?ap) AS ?aprob) WHERE {
           ?p a bo:Proposicion ; bo:presentadaPor ?g . FILTER(?g IN (br:grupo_eh_bildu, br:grupo_pp))
           OPTIONAL { ?p bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) . BIND(?p AS ?ap) } }
           GROUP BY ?g""",
         must=["638", "936"]),

    # ---------------- capa personas ----------------
    dict(id="g23", shape="persona",
         q="¿Qué concejal o concejala ha presentado (firmado) más proposiciones?",
         ref="""SELECT ?n (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p bo:proponePersona ?con . ?con rdfs:label ?n . } GROUP BY ?n ORDER BY DESC(?c) LIMIT 1""",
         must=["Cristina Ruiz Bujedo"], top="Cristina Ruiz Bujedo"),
    dict(id="g24", shape="persona",
         q="¿En cuántos debates del Pleno ha intervenido Xabier Otxandiano?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p bo:intervino ?con . ?con rdfs:label ?n . FILTER(REGEX(STR(?n), "otxandiano", "i")) }""",
         must=["80"], must_not=["0"]),
    dict(id="g25", shape="persona",
         q="¿Quién ha sido alcalde de Bilbao en el periodo de las actas?",
         ref="""SELECT ?n WHERE { ?con bo:esAlcalde true ; rdfs:label ?n . }""",
         must=["Aburto"]),
    dict(id="g26", shape="persona",
         q="¿Cuántas proposiciones ha firmado el concejal Luis Hermosa?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p bo:proponePersona ?con . ?con rdfs:label ?n . FILTER(REGEX(STR(?n), "hermosa", "i")) }""",
         must=["41"], must_not=["0"]),

    # ---------------- capa voto nominal ----------------
    dict(id="g27", shape="voto_nominal",
         q="¿Qué concejal o concejala ha votado más veces en contra de las proposiciones?",
         ref="""SELECT ?n (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p bo:concejalVotoEnContra ?con . ?con rdfs:label ?n . } GROUP BY ?n ORDER BY DESC(?c) LIMIT 1""",
         must=["Yolanda Diez Saiz"], top="Yolanda Diez Saiz"),
    dict(id="g28", shape="voto_nominal",
         q="¿Cuántas veces ha votado en contra la concejala Yolanda Díez?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p bo:concejalVotoEnContra ?con . ?con rdfs:label ?n .
           FILTER(REGEX(STR(?n), "yolanda", "i") && REGEX(STR(?n), "diez", "i")) }""",
         must=["309"], must_not=["0"]),
    dict(id="g29", shape="voto_nominal",
         q="¿En cuántas proposiciones sobre vivienda votó a favor el grupo EH Bildu?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_vivienda ; bo:votoAFavorDe br:grupo_eh_bildu . }""",
         must=["_ANY_"], must_not=["0"]),

    # ---------------- capa entidad ----------------
    dict(id="g30", shape="entidad",
         q="¿Se ha mencionado a Iberdrola en algún pleno del Ayuntamiento de Bilbao?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p bo:menciona ?e . ?e rdfs:label ?n . FILTER(REGEX(STR(?n), "\\\\biberdrola\\\\b", "i")) }""",
         must=["_ANY_"], must_not=["0"]),
    dict(id="g31", shape="entidad",
         q="¿Se ha mencionado a Zorrotzaurre en el Pleno?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p bo:menciona ?e . ?e rdfs:label ?n . FILTER(REGEX(STR(?n), "zorrotzaurre", "i")) }""",
         must=["_ANY_"], must_not=["0"]),

    # ---------------- degradación esperada ----------------
    dict(id="g32", shape="coautoria",
         q="¿Cuántas proposiciones conjuntas entre varios grupos ha habido?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p bo:presentadaPor ?a, ?b . FILTER(?a != ?b) }""",
         must=["0"], graceful=True),
    dict(id="g33", shape="entidad",
         q="¿Se ha mencionado a Mercadona en algún pleno del Ayuntamiento de Bilbao?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
           ?p bo:menciona ?e . ?e rdfs:label ?n . FILTER(REGEX(STR(?n), "\\\\bmercadona\\\\b", "i")) }""",
         must=["0"], graceful=True),

    # ---------------- desglose por tema ----------------
    dict(id="g34", shape="ranking",
         q="¿Cuál es el tema más tratado en las proposiciones del Pleno?",
         ref="""SELECT ?lab (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataSobre ?t . ?t skos:prefLabel ?lab . }
           GROUP BY ?lab ORDER BY DESC(?n) LIMIT 1""",
         must=["_ANY_"], min_rows=1),
    dict(id="g35", shape="conteo",
         q="¿Cuántas proposiciones sobre vivienda social ha habido?",
         ref="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
           ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_vivienda_social . }""",
         must=["_ANY_"], must_not=["0"]),
]

PREFIXES = """PREFIX bo: <http://bilbao.tfg/ontology#>
PREFIX br: <http://bilbao.tfg/resource/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
