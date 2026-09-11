"""Limpieza y canonicalización de entidades del enriquecimiento LLM, para
build_rdf.py. Objetivo: menos nodos basura y menos duplicados ("Iberdrola" /
"Iberdrola SA" / "IBERDROLA" -> un solo nodo).

- `es_basura(nombre)` -> True si hay que descartarla.
- `clave_entidad(nombre)` -> clave normalizada para agrupar (slug estable).
- `TIPOS` normaliza el campo 'tipo'.

No hace clustering por distancia de edición (eso sería una segunda pasada
offline sobre TODO el enriquecido); esto es lo que se puede hacer por-entidad
al vuelo, de forma determinista.
"""
import re
import unicodedata

TIPOS = {
    "persona": "persona", "person": "persona", "concejal": "persona", "concejala": "persona",
    "lugar": "lugar", "place": "lugar", "barrio": "lugar", "calle": "lugar", "equipamiento": "lugar",
    "organizacion": "organizacion", "organización": "organizacion", "organization": "organizacion",
    "empresa": "organizacion", "institucion": "organizacion", "institución": "organizacion",
}

# Sufijos jurídicos y coletillas que NO deben crear un nodo distinto.
_SUFIJOS = re.compile(
    r"[,\s]+(?:s\.?\s?a\.?|s\.?\s?l\.?|s\.?\s?coop\.?|s\.?\s?c\.?|oal|o\.?a\.?l\.?|"
    r"a\.?i\.?e\.?|ute|sociedad an[oó]nima|sociedad limitada|fundaz?i?oa?|fundaci[oó]n)\.?$",
    re.I)

# Basura conocida / demasiado genérica para ser una entidad útil.
_STOP_ENT = {
    "ayuntamiento", "ayuntamiento de bilbao", "pleno", "pleno municipal", "corporacion",
    "junta de gobierno", "junta de gobierno local", "gobierno municipal", "equipo de gobierno",
    "comision", "comisiones", "comisiones municipales", "grupo municipal", "grupos municipales",
    "area", "secretaria general del pleno", "alcaldia", "intervencion general",
    "villa de bilbao", "bilbao", "ciudad", "municipio", "estado", "gobierno", "administracion",
    "aita", "ama", "sr", "sra", "don", "dona",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


def es_basura(nombre: str) -> bool:
    n = _norm(nombre).lower().strip(" .,-\"'")
    if len(n) < 4:
        return True
    if not re.search(r"[aeiou]", n):            # sin vocales -> siglas sueltas ("a.t.e")
        return True
    if re.fullmatch(r"[a-z]\.?( ?[a-z]\.?){0,3}", n):  # "a.a", "a. t. e."
        return True
    if n in _STOP_ENT:
        return True
    if re.fullmatch(r"(?:el |la |los |las )?(?:se[nñ]or[ae]?s?|conceja?l[ae]?s?|grupos?|"
                    r"[aá]reas?|servicios?|planes?|proyectos?)", n):
        return True
    return False


# slug estable: minúsculas, sin acentos, sin sufijo jurídico, sin signos
def clave_entidad(nombre: str) -> str:
    n = _norm(nombre).lower().strip(" .,-\"'")
    prev = None
    while prev != n:                            # quitar sufijos apilados ("..., S.A.")
        prev = n
        n = _SUFIJOS.sub("", n).strip(" .,-")
    n = re.sub(r"[^a-z0-9]+", "_", n).strip("_")
    return n[:60]


_TRATO_PERSONA = re.compile(r"^(?:el |la )?(?:sr\.?|sra\.?|d\.?|d[nñ]a\.?|don|do[nñ]a|se[nñ]or[ae]?)\s", re.I)


# devuelve (etiqueta_mostrada, tipo_norm, clave) o None si es basura
def canon_entidad(nombre: str, tipo: str):
    nombre = _norm(nombre)
    if es_basura(nombre):
        return None
    tipo_n = TIPOS.get((tipo or "").lower().strip(), "entidad")
    # Persona con tratamiento ("Sr. Rodrigo") o de una sola palabra ("Basagoiti"):
    # casi siempre es un concejal que interviene en el debate -> se descarta como
    # entidad "mencionada" (esa relación la lleva bo:intervino).
    if tipo_n == "persona":
        limpio = _TRATO_PERSONA.sub("", nombre).strip()
        if len(limpio.split()) < 2:
            return None
    clave = clave_entidad(nombre)
    if not clave or len(clave) < 3:
        return None
    return nombre, tipo_n, clave
