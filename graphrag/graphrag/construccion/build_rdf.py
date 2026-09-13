import os
import re
import sys
import json
import unicodedata

from rdflib import Graph, Namespace, Literal, RDF, RDFS, URIRef
from rdflib.namespace import XSD, SKOS
import owlrl

HERE = os.path.dirname(os.path.abspath(__file__))            # .../graphrag/graphrag/construccion
GRAPHRAG = os.path.dirname(HERE)                               # .../graphrag/graphrag (datos + utils compartidos)
sys.path.insert(0, GRAPHRAG)
from grupos import normaliza_grupo, extrae_grupo, canon_grupo, prop_id, es_grupo_disfrazado  # noqa: E402
from jsonl_utils import load_jsonl, iter_jsonl  # noqa: E402
from entidades import canon_entidad  # noqa: E402

ENRICHED = os.path.join(GRAPHRAG, "proposals_enriched.jsonl")
CONCEJALES = os.path.join(GRAPHRAG, "concejales.jsonl")
PERSONAL_TEC = os.path.join(GRAPHRAG, "personal_tecnico.jsonl")
ONTOLOGY = os.path.join(GRAPHRAG, "ontology.ttl")
THEMES = os.path.join(GRAPHRAG, "themes_skos.ttl")
OUT = os.path.join(GRAPHRAG, "bilbao_reasoned.ttl")

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


# Patrones deterministas sobre vote_result (texto extraído por regex de
# backend/rag.py, más fiable que la clasificación libre del LLM de
# enriquecimiento) para el desenlace de LA PROPOSICIÓN en sí — nunca el de
# una enmienda ("se rechaza LA ENMIENDA..." no dispara nada aquí, a propósito.
_RE_RECHAZADA_PROP = re.compile(
    r"(?:queda|resulta)\s+rechazad[ao]\s+la\s+proposici[oó]n|se\s+rechaza\s+la\s+proposici[oó]n",
    re.IGNORECASE,
)
_RE_APROBADA_PROP = re.compile(
    r"se\s+(?:acepta|aprueba)\s+la\s+proposici[oó]n(?!.*enmienda)", re.IGNORECASE
)
_RE_DECAE_PROP = re.compile(r"\bdecae\b", re.IGNORECASE)


# usa vote_result cuando indica el desenlace sin ambigüedad, en vez de la clasificación del LLM.
# "decae" va PRIMERO y SIN gate: es la frase legal más inequívoca de las tres
# (bug real corregido 2026-09-11 — el LLM clasificaba sistemáticamente "se aprueba
# la enmienda..., por lo que decae la proposición" como "aprobada con enmienda" o
# "rechazada" en cientos de casos; el gate anterior solo confiaba en "decae" cuando
# el LLM no había dado ninguna respuesta, así que nunca corregía una respuesta
# equivocada con confianza, ver memoria/decisiones_tecnicas.md §4.5).
def resultado_cruzado(vote_result: str, resultado_llm: str) -> str:
    if not vote_result:
        return resultado_llm
    if _RE_DECAE_PROP.search(vote_result):
        return "decae"
    if _RE_RECHAZADA_PROP.search(vote_result):
        return "rechazada"
    if _RE_APROBADA_PROP.search(vote_result):
        if resultado_llm in (None, "", "sin resultado", "rechazada", "decae", "retirada"):
            return "aprobada"
        return resultado_llm
    return resultado_llm


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.lower()).strip("_")
    return s[:60] or "x"


# normaliza un label para matching: sin acentos, minúsculas, espacios colapsados
def norm_label(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower()).strip()


# label normalizado -> URI del concepto SKOS canónico (temas y subtemas)
def canon_theme_map(g: Graph) -> dict:
    m = {}
    for pred in (SKOS.prefLabel, SKOS.altLabel):
        for c, _, lab in g.triples((None, pred, None)):
            if (c, RDF.type, BO.Tema) in g:
                m.setdefault(norm_label(str(lab)), c)
    return m


# busca el concepto canónico para un subtema libre del LLM
def find_canonical(t_clean: str, canon: dict):
    if t_clean in canon:
        return canon[t_clean]
    mejor_label, mejor_uri = None, None
    for label, uri in canon.items():
        if len(label) >= 5 and label in t_clean:
            if mejor_label is None or len(label) > len(mejor_label):
                mejor_label, mejor_uri = label, uri
    return mejor_uri


def _fecha_key(f):
    d = (f or "").split("-")
    return (int(d[2]), int(d[1]), int(d[0])) if len(d) == 3 and all(x.isdigit() for x in d) else (0, 0, 0)


# crea los nodos :Concejal (+ bo:perteneceA, + bo:esAlcalde) y devuelve
def cargar_concejales(g):
    idx, alcaldes = {}, []
    if not os.path.exists(CONCEJALES):
        print("[!] no hay concejales.jsonl -> sin capa de concejales "
              "(corre extract_concejales.py primero)")
        return idx, alcaldes
    n = 0
    for c in iter_jsonl(CONCEJALES):
        uri = BR[f"concejal_{slug(c['nombre'])}"]
        g.add((uri, RDF.type, BO.Concejal))
        g.add((uri, RDFS.label, Literal(c["nombre"])))
        grupo = canon_grupo(c.get("grupo", "Desconocido"))
        if grupo != "Desconocido":
            gr = BR[f"grupo_{slug(grupo)}"]
            g.add((gr, RDF.type, BO.Grupo)); g.add((gr, RDFS.label, Literal(grupo)))
            g.add((uri, BO.perteneceA, gr))
        d0, d1 = _fecha_key(c.get("desde")), _fecha_key(c.get("hasta"))
        if c.get("es_alcalde"):
            g.add((uri, BO.esAlcalde, Literal(True, datatype=XSD.boolean)))
            alcaldes.append((uri, d0, d1))
        for ape in c.get("apellidos", []):
            idx.setdefault(norm_label(ape).upper(), []).append((uri, d0, d1, grupo))
        n += 1
    # Personal técnico (Secretario General, Interventor): nodos aparte, NO entran
    # en el índice de oradores (no debaten políticamente).
    t = 0
    if os.path.exists(PERSONAL_TEC):
        for c in iter_jsonl(PERSONAL_TEC):
            uri = BR[f"tecnico_{slug(c['nombre'])}"]
            g.add((uri, RDF.type, BO.PersonalTecnico))
            g.add((uri, RDFS.label, Literal(c["nombre"])))
            if c.get("cargo"):
                g.add((uri, BO.cargoTecnico, Literal(c["cargo"])))
            t += 1
    print(f"[*] {n} concejales cargados ({len(alcaldes)} han sido alcalde) + {t} personal técnico")
    return idx, alcaldes


# uri del concejal -> uri de su grupo (o None)
def grupo_de(g, conc_uri):
    return g.value(conc_uri, BO.perteneceA) if conc_uri else None


_ORADOR_ALCALDE = re.compile(r"^(?:sr\.?\s*|sra\.?\s*)?(?:alcalde|alcaldesa|presidente|presidenta)$", re.I)

# Alcaldes de Bilbao en el rango de datos, con su periodo EXACTO (curado a mano:
# solo son 3, y la detección automática confunde "Alcalde" con "Teniente de
# Alcalde en funciones"). Se usa para resolver el orador "ALCALDE".
_ALCALDES = [
    ("Inaki Azkuna Urreta",      (2007, 6, 1),  (2014, 1, 20)),   # fallece en el cargo
    ("Ibon Areso Mendiguren",    (2014, 1, 21), (2015, 6, 13)),   # interino
    ("Juan Maria Aburto Rique",  (2015, 6, 13), (2099, 1, 1)),
]


# apellido (del speaker_regex) + fecha -> URI del concejal en ese mandato
def resolver_concejal(idx, alcaldes, apellido, fecha, grupo_hint=None):
    if not apellido:
        return None
    fk = _fecha_key(fecha)
    if _ORADOR_ALCALDE.match(apellido.strip()):
        for nombre, ini, fin in _ALCALDES:
            if ini <= fk <= fin:
                return BR[f"concejal_{slug(nombre)}"]
        return None
    cands = idx.get(norm_label(apellido).upper(), [])
    if not cands:
        return None
    activos = [c for c in cands if c[1] <= fk <= c[2]] or cands
    if len(activos) > 1 and grupo_hint:
        g2 = canon_grupo(grupo_hint)
        activos = [c for c in activos if c[3] == g2] or activos
    return activos[0][0] if activos else None


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
    canon_uris = set(canon.values())  # para el chequeo de colisión de slug (ver más abajo)

    recs = load_jsonl(ENRICHED)
    PROPOSALS = os.path.join(HERE, "proposals.jsonl")
    textmap = {}
    if os.path.exists(PROPOSALS):
        for p in iter_jsonl(PROPOSALS):
            textmap[prop_id(p)] = p.get("text", "")
    print(f"[*] {len(recs)} proposiciones enriquecidas")

    conc_idx, alcaldes = cargar_concejales(g)

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

        # Resultado (normalizado para corregir variantes Basque, encoding, typos),
        # con verificación cruzada determinista contra vote_result cuando el LLM
        # se equivocó o no dio resultado (ver resultado_cruzado()).
        res_llm = normaliza_resultado(r.get("resultado", ""))
        res_raw = resultado_cruzado(r.get("vote_result", ""), res_llm)
        g.add((pr, BO.tieneResultado, RESULTADO_IND.get(res_raw, BO.SinResultado)))
        g.add((pr, BO.resultadoFuente, Literal(
            "acta" if (res_raw != res_llm or _RE_APROBADA_PROP.search(r.get("vote_result", "") or "")
                       or _RE_RECHAZADA_PROP.search(r.get("vote_result", "") or "")) else "llm")))

        # Tema principal (canónico) + temas libres como subtemas (skos:broader)
        tp = norm_label(r.get("tema_principal") or "otros")
        tp_match = find_canonical(tp, canon)
        if tp_match:
            tp_uri = tp_match
        else:
            tp_uri = BR[f"t_{slug(tp)}"]
            if tp_uri not in canon_uris:
                # Solo crear nodo libre si la URI generada no coincide ya con una
                # canónica (ej: slug("movilidad")="movilidad" → BR["t_movilidad"]
                # = canónica de "movilidad y transporte")
                g.add((tp_uri, RDF.type, BO.Tema)); g.add((tp_uri, SKOS.prefLabel, Literal(tp)))
                g.add((tp_uri, SKOS.broader, canon.get("otros")))
        g.add((pr, BO.trataSobre, tp_uri))

        for t in (r.get("temas") or [])[:4]:
            if not isinstance(t, str) or not t.strip():
                continue
            t_clean = norm_label(t)
            t_match = find_canonical(t_clean, canon)
            if t_match:
                # Tema o SUBTEMA canónico → enlace al nodo compartido, sin crear nodo
                # extra. Si es un subtema (nivel 2 de themes_skos.ttl), el razonador
                # infiere bo:trataTemaAmplio hacia su padre (roll-up temático real).
                if t_match != tp_uri:  # evitar duplicar el tema principal
                    g.add((pr, BO.trataSobre, t_match))
            else:
                # Subtema libre: URI PROPOSICIÓN-ESPECÍFICA para evitar que el mismo
                # slug acumule varios skos:broader a canonicos distintos (causa de la
                # inflacion de trataTemaAmplio que afectaba al grafo anterior).
                sub = BR[f"t_{r['id']}_{slug(t)}"]
                g.add((sub, RDF.type, BO.Tema))
                g.add((sub, SKOS.prefLabel, Literal(t_clean)))
                g.add((sub, SKOS.broader, tp_uri))
                g.add((pr, BO.trataSobre, sub))

        # Entidades — pasan por canon_entidad (limpia basura, quita sufijos
        # jurídicos, agrupa "Iberdrola"/"Iberdrola SA"). Ver entidades.py.
        for e in (r.get("entidades") or [])[:12]:
            if not isinstance(e, dict):
                continue
            nom_raw = (e.get("nombre") or "").strip()
            if not nom_raw or es_grupo_disfrazado(nom_raw):
                continue
            can = canon_entidad(nom_raw, e.get("tipo"))
            if not can:
                continue
            etiqueta, tipo_n, clave = can
            ent = BR[f"ent_{clave}"]
            g.add((ent, RDF.type, ENTIDAD_CLS.get(tipo_n, BO.Entidad)))
            g.add((ent, RDFS.label, Literal(etiqueta)))
            g.add((pr, BO.menciona, ent))

        # ---- CAMPOS RICOS (de la pasada de enriquecimiento LLM) ----
        if r.get("resumen"):
            g.add((pr, BO.resumen, Literal(r["resumen"][:500])))
        if r.get("page_ini"):
            try:
                g.add((pr, BO.pagina, Literal(int(r["page_ini"]), datatype=XSD.integer)))
            except (TypeError, ValueError):
                pass
        if r.get("source"):
            g.add((pr, BO.fuentePdf, Literal(r["source"])))

        # ---- VOTO ----
        # Capa 1 (determinista): si el acta trae la lista nominal y los conteos
        # cuadran, se resuelve apellido -> concejal -> grupo y se emiten los
        # votos NOMINALES + los de grupo derivados. bo:votoFuente = "acta".
        # Capa 2 (fallback): si no hay lista fiable, se usa votos_por_grupo del
        # LLM. bo:votoFuente = "llm".
        VOTO_NOM = {"favor": BO.concejalVotoAFavor, "contra": BO.concejalVotoEnContra, "abstencion": BO.concejalVotoAbstencion}
        VOTO_GRP = {"favor": BO.votoAFavorDe, "contra": BO.votoEnContraDe, "abstencion": BO.seAbstuvo}
        vn = r.get("votos_nominales")
        if isinstance(vn, dict) and vn.get("_cuadra"):
            grupos_sent = {"favor": set(), "contra": set(), "abstencion": set()}
            for sentido in ("favor", "contra", "abstencion"):
                for ape in (vn.get(sentido) or []):
                    cu = resolver_concejal(conc_idx, alcaldes, ape, r.get("date"))
                    if not cu:
                        continue
                    g.add((pr, VOTO_NOM[sentido], cu))
                    gv = grupo_de(g, cu)
                    if gv is not None:
                        grupos_sent[sentido].add(gv)
            for sentido, gvs in grupos_sent.items():
                for gv in gvs:
                    g.add((pr, VOTO_GRP[sentido], gv))
            g.add((pr, BO.votoFuente, Literal("acta")))
        else:
            vpg = r.get("votos_por_grupo") or {}
            emitido = False
            for sentido, pred in VOTO_GRP.items():
                for gnom in (vpg.get(sentido) or []):
                    gc = canon_grupo(gnom)
                    if gc == "Desconocido":
                        continue
                    gv = BR[f"grupo_{slug(gc)}"]
                    g.add((gv, RDF.type, BO.Grupo)); g.add((gv, RDFS.label, Literal(gc)))
                    g.add((pr, pred, gv))
                    emitido = True
            if emitido:
                g.add((pr, BO.votoFuente, Literal("llm")))

        # Enmienda
        enm = r.get("enmienda")
        if isinstance(enm, dict):
            en_uri = BR[f"enmienda_{r['id']}"]
            g.add((en_uri, RDF.type, BO.Enmienda))
            g.add((pr, BO.tieneEnmienda, en_uri))
            if enm.get("resumen"):
                g.add((en_uri, BO.resumenEnmienda, Literal(str(enm["resumen"])[:400])))
            if enm.get("tipo"):
                g.add((en_uri, BO.tipoEnmienda, Literal(str(enm["tipo"])[:40])))
            if enm.get("resultado"):
                g.add((en_uri, BO.resultadoEnmienda, Literal(str(enm["resultado"])[:40])))
            ev = enm.get("votos") or {}
            for k, pred in (("favor", BO.votosFavorEnmienda), ("contra", BO.votosContraEnmienda)):
                try:
                    if ev.get(k) is not None:
                        g.add((en_uri, pred, Literal(int(ev[k]), datatype=XSD.integer)))
                except (TypeError, ValueError):
                    pass
            gc = canon_grupo(enm.get("por", ""))
            if gc != "Desconocido":
                ge = BR[f"grupo_{slug(gc)}"]
                g.add((ge, RDF.type, BO.Grupo)); g.add((ge, RDFS.label, Literal(gc)))
                g.add((en_uri, BO.enmiendaPor, ge))

        # Concejales que intervinieron (oradores) + proponente persona
        for ape in (r.get("oradores") or []):
            cu = resolver_concejal(conc_idx, alcaldes, ape, r.get("date"), grupo_hint=grupo)
            if cu:
                g.add((pr, BO.intervino, cu))
        pp = r.get("proponente_persona")
        if pp:
            # último apellido "grande" del nombre completo del LLM
            token = next((w for w in reversed(norm_label(pp).split()) if len(w) >= 4), None)
            cu = resolver_concejal(conc_idx, alcaldes, token, r.get("date"), grupo_hint=grupo) if token else None
            if cu:
                g.add((pr, BO.proponePersona, cu))

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
