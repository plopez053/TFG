"""Normalización y extracción del GRUPO MUNICIPAL proponente/orador.

Módulo compartido por build_graph.py (fase 2: enriquecimiento LLM) y
build_rdf.py (fase 3: construcción del grafo RDF) — antes este código vivía
duplicado en ambos ficheros; unificarlo evita que diverjan silenciosamente.

Es resolución de entidades de dominio cerrado: el Pleno de Bilbao ha tenido
un número finito y conocido de grupos municipales en 2002-2026, con variantes
por OCR, bilingüismo (castellano/euskera) y cambios de sigla a lo largo de los
años (Bildu → EH Bildu, Bilbao en Común → Elkarrekin Bilbao...). No existe un
algoritmo "general" que infiera estas equivalencias sin conocimiento del
dominio: por eso el mapeo es una tabla explícita, no heurística estadística.
"""
import re

GRUPOS_CANONICOS = {
    "EH BILDU", "PSE-EE", "EAJ-PNV", "PP", "ELKARREKIN BILBAO",
    "GOAZEN BILBAO", "UDALBERRI", "EZKER BATUA-IU", "ARALAR",
    "CIUDADANOS", "VOX", "EQUIPO DE GOBIERNO", "GRUPO MIXTO", "Desconocido",
}


# colapsa las variantes de un nombre de grupo a su forma canónica
def normaliza_grupo(p: str) -> str:
    s = re.sub(r"\s+", " ", (p or "")).upper()
    d = s.replace(" ", "")  # despaciado: tolera cortes de OCR ("BIL DU" -> "BILDU")
    if "GOAZEN" in d:
        return "GOAZEN BILBAO"
    if "ELKARREKIN" in d or "PODEMOS" in d or "EZKERANITZA" in d or "EQUO" in d:
        return "ELKARREKIN BILBAO"
    # UDALBERRI antes de BILBAO EN COMÚN para no confundir
    if "UDALBERRI" in d:
        return "UDALBERRI"
    # "Bilbao en Común" fue la coalición pre-Elkarrekin (mismos integrantes)
    if "BILBAOENCOMUN" in d or "BILBAOENCOMÚN" in d or "BILBAOENKUMUN" in d:
        return "ELKARREKIN BILBAO"
    if "BILDU" in d or "EUSKALHERRIA" in d or "EAE-ANV" in d or "EAEANV" in d:
        return "EH BILDU"
    # HB (Herri Batasuna) — predecesor de EH BILDU
    if re.search(r"\bHB\b", s) or "HERRIBATASUNA" in d:
        return "EH BILDU"
    if "SOCIALIST" in d or "PSE" in d or "PSOE" in d:
        return "PSE-EE"
    # Sozialista Abertzaleak (nombre histórico en euskera de los socialistas)
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
    # Gobierno/Equipo: "Gorbernu Taldeak" (euskera), "Junta de Gobierno", "Alcaldía"
    if "GOBIERNO" in d or "GOBERNUTAL" in d or "GORBERNU" in d or "ALCALD" in d:
        return "EQUIPO DE GOBIERNO"
    if "MIXTO" in d:
        return "GRUPO MIXTO"
    if not p or p.strip() == "" or p == "Desconocido":
        return "Desconocido"
    return p.strip()[:40]


# segunda pasada: si el nombre normalizado no es uno de los canónicos
def canon_grupo(grupo: str) -> str:
    g2 = normaliza_grupo(grupo)
    return g2 if g2 in GRUPOS_CANONICOS else "Desconocido"


# Extrae el GRUPO PROPONENTE del título/texto del punto ("...que presenta el
# Grupo Municipal EH BILDU..."). Más fiable que el metadato `party` del chunk
# (que identifica al ORADOR que interviene, no a quien presenta la proposición).
_GRUPO_RE = re.compile(
    r"(?:que\s+presenta[n]?\s+(?:el|los)|presentad[ao]\s+por\s+(?:el|los)|del|de\s+la)\s+grupo[s]?\s+"
    r"(?:pol[ií]tico[s]?\s+)?(?:municipal(?:es)?\s+)?(.{3,60}?)(?:\s*,|\.|cuya|que\s+su|presenta|$)",
    re.IGNORECASE,
)

# Proposiciones en euskera: "X udal taldeak aurkezten duen"
_GRUPO_EUSKERA_RE = re.compile(
    r"^(.{3,60}?)\s+udal\s+tald(?:eak|e(?:ak)?)\s+aurkezten",
    re.IGNORECASE,
)

# Fallback: busca el nombre del partido directamente en el texto cuando no
# aparece el patrón "Grupo Municipal X". Alternativas más largas/específicas
# primero para evitar capturas parciales.
_PARTIDO_DIRECTO_RE = re.compile(
    r"\b("
    r"EH\s+BILDU|EH-BILDU|EHBILDU|HERRI\s+BATASUNA|EUSKAL\s+HERRIA\s+BILDU"
    r"|ELKARREKIN\s+BILBAO|ELKARREKIN"
    r"|GOAZEN\s+BILBAO|GOAZEN"
    r"|EZKER\s+BATUA[- ]IU|EZKER\s+BATUA|IZQUIERDA\s+UNIDA"
    r"|PSE[- ]EE|PSE\s*EE|PARTIDO\s+SOCIALISTA|SOZIALISTA\s+ABERTZALEAK|GRUPO\s+SOCIALISTA"
    r"|EAJ[- ]PNV|EAJ\s*PNV|PARTIDO\s+NACIONALISTA\s+VASCO"
    r"|PARTIDO\s+POPULAR|GRUPO\s+POPULAR"
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


# determina el grupo proponente probando, en orden: patrón estructurado
def extrae_grupo(topic: str, text: str = "") -> str:
    for src in (topic or "", (text or "")[:600]):
        m = _GRUPO_RE.search(src)
        if m:
            raw = m.group(1).strip()
            # Para conjuntas "PSE-EE, EH BILDU y PP" tomar solo el primero
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


# ID estable y determinista de una proposición a partir de (fecha, tema)
def prop_id(p: dict) -> str:
    import hashlib
    return hashlib.md5((p["date"] + "||" + p["topic"]).encode("utf-8")).hexdigest()[:16]


# true si una 'entidad' del LLM es en realidad un grupo político o el Ayuntamiento
def es_grupo_disfrazado(nombre_entidad: str) -> bool:
    nom = (nombre_entidad or "").strip()
    if not nom or len(nom) > 70:
        return False
    if re.search(r"ayuntamiento\s+de\s+bilbao", nom, re.IGNORECASE):
        return True
    for rex in (_GRUPO_RE, _GRUPO_EUSKERA_RE, _PARTIDO_DIRECTO_RE):
        m = rex.search(nom)
        if m and (m.end() - m.start()) / len(nom) > 0.4:
            grupo_texto = m.group(1) if rex is not _PARTIDO_DIRECTO_RE else m.group(0)
            if normaliza_grupo(grupo_texto) != "Desconocido":
                return True
    return False
