"""Fase 3b: construye el grafo RDF (ABox) desde las proposiciones enriquecidas,
lo une con la ontología (TBox) + taxonomía SKOS, y ejecuta el razonador OWL-RL
(owlrl) para materializar inferencias (roll-up temático, tipos de entidad, etc.).

Salida: graphrag/bilbao_reasoned.ttl  (grafo con triples inferidos incluidos)

Uso:  python graphrag/build_rdf.py
"""
import os
import re
import sys
import json
import hashlib
import unicodedata

from rdflib import Graph, Namespace, Literal, RDF, RDFS, URIRef
from rdflib.namespace import XSD, SKOS
import owlrl

def normaliza_grupo(p: str) -> str:
    s = re.sub(r"\s+", " ", (p or "")).upper()
    d = s.replace(" ", "")
    if "GOAZEN" in d:
        return "GOAZEN BILBAO"
    if "ELKARREKIN" in d or "PODEMOS" in d or "EZKERANITZA" in d or "EQUO" in d:
        return "ELKARREKIN BILBAO"
    if "UDALBERRI" in d:
        return "UDALBERRI"
    if "BILBAOENCOMUN" in d or "BILBAOENCOMÚN" in d or "BILBAOENKUMUN" in d:
        return "ELKARREKIN BILBAO"
    if "BILDU" in d or "EUSKALHERRIA" in d or "EAE-ANV" in d or "EAEANV" in d:
        return "EH BILDU"
    if re.search(r"\bHB\b", s) or "HERRIBATASUNA" in d or "HERRIKO\s*BATASUNA" in d.replace(" ", ""):
        return "EH BILDU"
    if "SOCIALIST" in d or "PSE" in d or "PSOE" in d:
        return "PSE-EE"
    if "SOZIALISTAK" in d or "SOZIALIST" in d or "ABERTZALEAK" in d:
        return "PSE-EE"
    if "PNV" in d or "EAJ" in d or "NACIONALIST" in d or "JELTZALE" in d:
        return "EAJ-PNV"
    if "POPULAR" in d or "P.P" in d or re.search(r"\bP\s?P\b", s):
        return "PP"
    if "EZKERBATUA" in d or "IZQUIERDAUNIDA" in d or "BERDEAK" in d or re.search(r"\bI\s?U\b", s):
        return "EZKER BATUA-IU"
    if "ARALAR" in d:
        return "ARALAR"
    if "CIUDADANOS" in d or re.search(r"\bC\s?S\b", s):
        return "CIUDADANOS"
    if "VOX" in d:
        return "VOX"
    if "GOBIERNO" in d or "GOBERNUTAL" in d or "GORBERNU" in d or "ALCALD" in d:
        return "EQUIPO DE GOBIERNO"
    if "MIXTO" in d:
        return "GRUPO MIXTO"
    if not p or p.strip() == "" or p == "Desconocido":
        return "Desconocido"
    return p.strip()[:40]


_GRUPO_RE = re.compile(
    r"(?:que\s+presenta[n]?\s+(?:el|los)|presentad[ao]\s+por\s+(?:el|los)|del|de\s+la)\s+grupo[s]?\s+"
    r"(?:pol[ií]tico[s]?\s+)?(?:municipal(?:es)?\s+)?(.{3,60}?)(?:\s*,|\.|cuya|que\s+su|presenta|$)",
    re.IGNORECASE,
)
_GRUPO_EUSKERA_RE = re.compile(
    r"^(.{3,60}?)\s+udal\s+tald(?:eak|e(?:ak)?)\s+aurkezten",
    re.IGNORECASE,
)
_PARTIDO_DIRECTO_RE = re.compile(
    r"\b("
    r"EH\s+BILDU|EH-BILDU|EHBILDU|HERRI\s+BATASUNA|EUSKAL\s+HERRIA\s+BILDU"
    r"|ELKARREKIN\s+BILBAO|ELKARREKIN"
    r"|GOAZEN\s+BILBAO|GOAZEN"
    r"|EZKER\s+BATUA[- ]IU|EZKER\s+BATUA|IZQUIERDA\s+UNIDA"
    r"|PSE[- ]EE|PSE\s*EE|PARTIDO\s+SOCIALISTA|SOZIALISTA\s+ABERTZALEAK"
    r"|EAJ[- ]PNV|EAJ\s*PNV|PARTIDO\s+NACIONALISTA\s+VASCO"
    r"|PARTIDO\s+POPULAR"
    r"|UDALBERRI"
    r"|BILBAO\s+EN\s+COM[ÚU]N"
    r"|ARALAR"
    r"|CIUDADANOS"
    r"|VOX"
    r"|EQUIPO\s+DE\s+GOBIERNO|GOBIERNO\s+MUNICIPAL|GORBERNU\s+TALDEA"
    r"|GRUPO\s+MIXTO"
    r")\b",
    re.IGNORECASE,
)


def extrae_grupo(topic: str, text: str = "") -> str:
    for src in (topic or "", (text or "")[:600]):
        m = _GRUPO_RE.search(src)
        if m:
            raw = m.group(1).strip()
            raw = re.split(r"\s*[,y]\s+(?:EH|PP|PSE|EAJ|ELK|GOA|UDA)", raw)[0]
            g = normaliza_grupo(raw)
            if g != "Desconocido":
                return g
    for src in (topic or "", (text or "")[:600]):
        m = _GRUPO_EUSKERA_RE.search(src)
        if m:
            g = normaliza_grupo(m.group(1))
            if g != "Desconocido":
                return g
    for src in (topic or "", (text or "")[:900]):
        m = _PARTIDO_DIRECTO_RE.search(src)
        if m:
            g = normaliza_grupo(m.group(1))
            if g != "Desconocido":
                return g
    return "Desconocido"


def prop_id(p: dict) -> str:
    return hashlib.md5((p["date"] + "||" + p["topic"]).encode("utf-8")).hexdigest()[:16]


GRUPOS_CANONICOS = {
    "EH BILDU", "PSE-EE", "EAJ-PNV", "PP", "ELKARREKIN BILBAO",
    "GOAZEN BILBAO", "UDALBERRI", "EZKER BATUA-IU", "ARALAR",
    "CIUDADANOS", "VOX", "EQUIPO DE GOBIERNO", "GRUPO MIXTO", "Desconocido",
}

def canon_grupo(grupo: str) -> str:
    """Segunda pasada de normalización: si el nombre no es canónico, descartarlo."""
    g2 = normaliza_grupo(grupo)
    return g2 if g2 in GRUPOS_CANONICOS else "Desconocido"

HERE = os.path.dirname(os.path.abspath(__file__))
ENRICHED = os.path.join(HERE, "proposals_enriched.jsonl")
ONTOLOGY = os.path.join(HERE, "ontology.ttl")
THEMES = os.path.join(HERE, "themes_skos.ttl")
OUT = os.path.join(HERE, "bilbao_reasoned.ttl")

BO = Namespace("http://bilbao.tfg/ontology#")
BR = Namespace("http://bilbao.tfg/resource/")

RESULTADO_IND = {
    "aprobada": BO.Aprobada, "rechazada": BO.Rechazada, "decae": BO.Decae,
    "retirada": BO.Retirada, "aprobada con enmienda": BO.AprobadaConEnmienda,
    "sin resultado": BO.SinResultado,
}
# Variantes de resultados (encoding corrupto, Basque, typos, sinónimos) → valor canónico
_RESULTADO_NORM = {
    "desestimada": "rechazada",
    "desestimacion": "rechazada",
    "desestimació": "rechazada",
    "desestimación": "rechazada",
    "desestimada con enmienda": "rechazada",
    "inadmitida a tramite": "rechazada",
    "inadmitida a trámite": "rechazada",
    "estimada": "aprobada con enmienda",
    "estimacion parcial": "aprobada con enmienda",
    "estimación parcial": "aprobada con enmienda",
    "estimada parcialmente": "aprobada con enmienda",
    "aprobada inicialmente": "aprobada",
    "aprobada inicialmente con condiciones": "aprobada con enmienda",
    "aprobada con enmiendas": "aprobada con enmienda",
    "aprobadada con enmienda": "aprobada con enmienda",
    "decayó": "decae",
    "decay": "decae",
    "onestea": "aprobada",      # Basque: aceptar/aprobar
    "onartu": "aprobada",       # Basque: aprobado
    "ez onartua": "rechazada",  # Basque: no aprobado
    "bozkatu": "sin resultado", # Basque: votar (indeterminado)
    "proponer": "sin resultado",
    "abstención": "sin resultado",
    "abstencion": "sin resultado",
}
ENTIDAD_CLS = {"persona": BO.Persona, "lugar": BO.Lugar, "organizacion": BO.Organizacion}


def normaliza_resultado(r: str) -> str:
    r2 = unicodedata.normalize("NFKD", (r or "").lower().strip()).encode("ascii", "ignore").decode()
    r2 = re.sub(r"\s+", " ", r2).strip()
    # Busca primero el valor exacto normalizado
    for k, v in _RESULTADO_NORM.items():
        k2 = unicodedata.normalize("NFKD", k).encode("ascii", "ignore").decode()
        if r2 == k2 or r2.startswith(k2):
            return v
    # Fallback a los canónicos directos
    if r2 in RESULTADO_IND:
        return r2
    return "sin resultado"


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.lower()).strip("_")
    return s[:60] or "x"


def canon_theme_map(g: Graph) -> dict:
    """label (str) -> URI del concepto canónico, leído de themes_skos.ttl."""
    m = {}
    for c, _, lab in g.triples((None, SKOS.prefLabel, None)):
        if (c, RDF.type, BO.Tema) in g:
            m[str(lab).lower()] = c
    return m


def parse_votos(vote_text):
    if not vote_text:
        return None, None
    f = re.search(r"a favor:\s*(\d+)", vote_text)
    c = re.search(r"en contra:\s*(\d+)", vote_text)
    return (int(f.group(1)) if f else None), (int(c.group(1)) if c else None)


def build():
    g = Graph()
    g.bind("bo", BO); g.bind("br", BR); g.bind("skos", SKOS)
    g.parse(ONTOLOGY, format="turtle")
    g.parse(THEMES, format="turtle")
    canon = canon_theme_map(g)

    recs = [json.loads(l) for l in open(ENRICHED, encoding="utf-8")]
    PROPOSALS = os.path.join(HERE, "proposals.jsonl")
    textmap = {}
    if os.path.exists(PROPOSALS):
        for l in open(PROPOSALS, encoding="utf-8"):
            p = json.loads(l)
            textmap[prop_id(p)] = p.get("text", "")
    print(f"[*] {len(recs)} proposiciones enriquecidas")

    for r in recs:
        pr = BR[f"prop_{r['id']}"]
        g.add((pr, RDF.type, BO.Proposicion))
        g.add((pr, BO.tituloTopic, Literal(r["topic"][:200])))
        g.add((pr, BO.fecha, Literal(r["date"])))
        año = (r["date"].split("-")[-1] if r.get("date") else "")
        if año.isdigit():
            g.add((pr, BO.anio, Literal(int(año), datatype=XSD.integer)))
        if r.get("vote_result"):
            g.add((pr, BO.votoTexto, Literal(r["vote_result"][:300])))
            vf, vc = parse_votos(r["vote_result"])
            if vf is not None: g.add((pr, BO.votosFavor, Literal(vf, datatype=XSD.integer)))
            if vc is not None: g.add((pr, BO.votosContra, Literal(vc, datatype=XSD.integer)))

        # Grupo PROPONENTE: extraído del título del punto (más fiable que el metadato).
        grupo = extrae_grupo(r.get("topic", ""), textmap.get(r["id"], ""))
        grupo = canon_grupo(grupo)
        if grupo == "Desconocido" and r.get("grupo") and r["grupo"] != "Desconocido":
            grupo = canon_grupo(r["grupo"])
        gr = BR[f"grupo_{slug(grupo)}"]
        g.add((gr, RDF.type, BO.Grupo)); g.add((gr, RDFS.label, Literal(grupo)))
        g.add((pr, BO.presentadaPor, gr))

        # Pleno
        pl = BR[f"pleno_{slug(r['date'])}"]
        g.add((pl, RDF.type, BO.Pleno)); g.add((pl, RDFS.label, Literal(r["date"])))
        if año.isdigit(): g.add((pl, BO.anio, Literal(int(año), datatype=XSD.integer)))
        g.add((pr, BO.enPleno, pl))

        # Resultado (normalizado para corregir variantes Basque, encoding, typos)
        res_raw = normaliza_resultado(r.get("resultado", ""))
        g.add((pr, BO.tieneResultado, RESULTADO_IND.get(res_raw, BO.SinResultado)))

        # Tema principal (canónico) + temas libres como subtemas (skos:broader)
        tp = (r.get("tema_principal") or "otros").lower().strip()
        tp_uri = canon.get(tp, BR[f"t_{slug(tp)}"])
        if tp not in canon:  # tema_principal no canónico → cuélgalo de "otros"
            g.add((tp_uri, RDF.type, BO.Tema)); g.add((tp_uri, SKOS.prefLabel, Literal(tp)))
            g.add((tp_uri, SKOS.broader, canon.get("otros")))
        g.add((pr, BO.trataSobre, tp_uri))

        for t in (r.get("temas") or [])[:4]:
            if not isinstance(t, str) or not t.strip():
                continue
            t_clean = t.lower().strip()
            if t_clean in canon:
                # Tema canónico secundario → enlace directo, sin crear nodo extra
                if canon[t_clean] != tp_uri:  # evitar duplicar el tema principal
                    g.add((pr, BO.trataSobre, canon[t_clean]))
            else:
                # Subtema libre: URI PROPOSICIÓN-ESPECÍFICA para evitar que el mismo
                # slug acumule varios skos:broader a canonicos distintos (causa de la
                # inflacion de trataTemaAmplio que afectaba al grafo anterior).
                sub = BR[f"t_{r['id']}_{slug(t)}"]
                g.add((sub, RDF.type, BO.Tema))
                g.add((sub, SKOS.prefLabel, Literal(t_clean)))
                g.add((sub, SKOS.broader, tp_uri))
                g.add((pr, BO.trataSobre, sub))

        # Entidades
        for e in (r.get("entidades") or [])[:8]:
            if not isinstance(e, dict):
                continue
            nom = (e.get("nombre") or "").strip()
            if not nom:
                continue
            ent = BR[f"ent_{slug(nom)}"]
            cls = ENTIDAD_CLS.get((e.get("tipo") or "").lower(), BO.Entidad)
            g.add((ent, RDF.type, cls))
            g.add((ent, RDFS.label, Literal(nom)))
            g.add((pr, BO.menciona, ent))

    n_before = len(g)
    print(f"[*] triples antes de razonar: {n_before}")

    # Razonador OWL-RL: materializa subClassOf, subPropertyOf, transitividad y
    # propertyChainAxiom (roll-up temático trataTemaAmplio).
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    n_after = len(g)
    print(f"[*] triples tras razonar:   {n_after}  (+{n_after - n_before} inferidos)")

    g.serialize(destination=OUT, format="turtle")
    print(f"[+] grafo razonado -> {OUT}")


if __name__ == "__main__":
    build()
