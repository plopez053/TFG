# -*- coding: utf-8 -*-
BANK = [
    dict(q="¿Cuántas proposiciones sobre cultura se han presentado?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_cultura . }"""),

    dict(q="¿Cuántas proposiciones presentó el PSE-EE en 2018?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_pse_ee ; bo:anio 2018 . }"""),

    dict(q="¿Cuántas proposiciones sobre sanidad se han aprobado?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_sanidad ; bo:tieneResultado ?r .
  FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) }"""),

    dict(q="¿Qué grupo ha presentado más proposiciones sobre seguridad?",
         sparql="""SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_seguridad ; bo:presentadaPor ?g .
  ?g rdfs:label ?ng . FILTER(?g != br:grupo_desconocido) }
GROUP BY ?ng ORDER BY DESC(?n) LIMIT 10"""),

    dict(q="¿Cuántas proposiciones ha presentado cada grupo en total?",
         sparql="""SELECT ?ng (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor ?g . ?g rdfs:label ?ng .
  FILTER(?g != br:grupo_desconocido) }
GROUP BY ?ng ORDER BY DESC(?n) LIMIT 20"""),

    dict(q="¿En qué año hubo más proposiciones sobre cultura?",
         sparql="""SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_cultura ; bo:anio ?anio . }
GROUP BY ?anio ORDER BY DESC(?n) LIMIT 1"""),

    dict(q="¿Cómo ha evolucionado el número de proposiciones sobre movilidad por año?",
         sparql="""SELECT ?anio (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_movilidad ; bo:anio ?anio . }
GROUP BY ?anio ORDER BY ASC(?anio)"""),

    dict(q="¿Cuántas proposiciones presentó el PP en 2016 y cuántas se aprobaron?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?ap) AS ?aprob) WHERE {
  ?p a bo:Proposicion ; bo:presentadaPor br:grupo_pp ; bo:anio 2016 .
  OPTIONAL { ?p bo:tieneResultado ?r . FILTER(?r IN (bo:Aprobada, bo:AprobadaConEnmienda)) . BIND(?p AS ?ap) } }"""),

    dict(q="¿Qué porcentaje de las proposiciones sobre seguridad se rechazaron?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?total) (COUNT(DISTINCT ?re) AS ?rech) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_seguridad .
  OPTIONAL { ?p bo:tieneResultado bo:Rechazada . BIND(?p AS ?re) } }"""),

    dict(q="¿Qué concejal o concejala ha intervenido en más debates del Pleno?",
         sparql="""SELECT ?n (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:intervino ?con . ?con rdfs:label ?n . }
GROUP BY ?n ORDER BY DESC(?c) LIMIT 1"""),

    dict(q="¿Cuántas proposiciones ha firmado el concejal Gorka Otxandiano?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:proponePersona ?con . ?con rdfs:label ?n .
  FILTER(REGEX(STR(?n), "gorka", "i") && REGEX(STR(?n), "otxandiano", "i")) }"""),

    dict(q="¿Qué concejal o concejala ha votado a favor de más proposiciones?",
         sparql="""SELECT ?n (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:concejalVotoAFavor ?con . ?con rdfs:label ?n . }
GROUP BY ?n ORDER BY DESC(?c) LIMIT 1"""),

    dict(q="¿Cuántas veces se abstuvo el grupo EH Bildu en proposiciones sobre urbanismo?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_urbanismo ; bo:seAbstuvo br:grupo_eh_bildu . }"""),

    dict(q="¿Se ha mencionado a Petronor en algún pleno?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?c) WHERE {
  ?p bo:menciona ?e . ?e rdfs:label ?n . FILTER(REGEX(STR(?n), "\\\\bpetronor\\\\b", "i")) }"""),

    dict(q="¿Cuántas enmiendas ha presentado el grupo EH Bildu?",
         sparql="""SELECT (COUNT(DISTINCT ?e) AS ?n) WHERE {
  ?e a bo:Enmienda ; bo:enmiendaPor br:grupo_eh_bildu . }"""),

    dict(q="¿Cuál es el tema menos tratado en las proposiciones del Pleno?",
         sparql="""SELECT ?lab (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataSobre ?t . ?t skos:prefLabel ?lab . }
GROUP BY ?lab ORDER BY ASC(?n) LIMIT 1"""),

    dict(q="¿Cuántas proposiciones sobre el alquiler se han presentado?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_alquiler . }"""),

    dict(q="¿Cuántas proposiciones hay sobre igualdad y feminismo presentadas por el PP?",
         sparql="""SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p a bo:Proposicion ; bo:trataTemaAmplio br:t_igualdad ; bo:presentadaPor br:grupo_pp . }"""),
]


def _tok(s):
    import re
    return set(re.findall(r"[a-záéíóúñ0-9]+", s.lower()))


def nearest(pregunta, k=3):
    qt = _tok(pregunta)
    scored = []
    for ex in BANK:
        et = _tok(ex["q"])
        j = len(qt & et) / max(1, len(qt | et))
        scored.append((j, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]


def fixed(k=6):
    return [BANK[0], BANK[4], BANK[6], BANK[7], BANK[10], BANK[3]][:k]
