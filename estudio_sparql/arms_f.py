# -*- coding: utf-8 -*-
import re
import arms as A

# ---- mapa de grupos (forma laxa -> URI canónica) -------------------------
_GRUPOS = {
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
_TEMAS = {"vivienda", "urbanismo", "movilidad", "medioambiente", "euskera", "cultura",
          "deporte", "educacion", "igualdad", "serviciossociales", "empleoeconomia",
          "presupuestos", "seguridad", "participacion", "turismo", "sanidad",
          "memoriahistorica", "derechoshumanos", "otros"}

_VOTO_PERSONA = ("proponePersona", "intervino", "concejalVotoAFavor",
                 "concejalVotoEnContra", "concejalVotoAbstencion")


def _canon_grupo(raw):
    k = re.sub(r"[^a-z0-9]", "", raw.lower())
    if k in _GRUPOS:
        return "br:" + _GRUPOS[k]
    # ya es canónica (grupo_xxx)
    if k.startswith("grupo"):
        return "br:" + re.sub(r"^grupo_?", "grupo_", raw.lower())
    return None


def alias_rewrite(sparql: str) -> str:
    if not sparql:
        return sparql
    s = sparql

    # 0) <br:x> -> br:x  (el modelo a veces mete la URI prefijada entre <>)
    s = re.sub(r"<(br:[A-Za-z_]\w*)>", r"\1", s)
    s = re.sub(r"<(bo:[A-Za-z_]\w*)>", r"\1", s)

    # 1) tema con prefijo bo: -> br:   (bo:t_euskera -> br:t_euskera)
    s = re.sub(r"\bbo:(t_[a-z_]+)\b", r"br:\1", s)
    # 1b) bo:<TemaCapitalizado> -> br:t_<minuscula>   (bo:Euskera -> br:t_euskera)
    def _bo_tema(m):
        w = m.group(1).lower()
        return f"br:t_{w}" if w in _TEMAS else m.group(0)
    s = re.sub(r"\bbo:([A-ZÁÉÍÓÚ][a-záéíóú]+)\b", _bo_tema, s)

    # 2) grupo con URI no canónica: br:EHBildu / br:EH_Bildu -> br:grupo_eh_bildu
    def _grp(m):
        c = _canon_grupo(m.group(1))
        return c or m.group(0)
    s = re.sub(r"\bbr:([A-Za-z][A-Za-z_]*)\b", lambda m: (
        _canon_grupo(m.group(1)) or m.group(0)) if re.sub(r"[^a-z0-9]", "", m.group(1).lower()) in _GRUPOS else m.group(0), s)

    # 3) entidad como URI:  ?p bo:menciona br:Iberdrola   /  bo:menciona br:entidad_mercadona
    #    -> ?p bo:menciona ?ent_f . ?ent_f rdfs:label ?ent_fl . FILTER(REGEX(...))
    def _ent(m):
        pre, uri = m.group(1), m.group(2)
        uri = re.sub(r"^(entidad_|ent_|lugar_|org_|organizacion_)", "", uri, flags=re.I)
        body = (f'{pre} bo:menciona ?ent_f . ?ent_f rdfs:label ?ent_fl . '
                f'FILTER({_regex_and("?ent_fl", uri)})')
        return _tail(body, pre, m.group(3))
    s = re.sub(r"(\?\w+)\s+bo:menciona\s+br:([A-Za-z_]\w*)\s*([;.])", _ent, s)

    # 4) persona como URI con predicados de persona / voto nominal (term = ; o .)
    def _per(m):
        pre, pred, uri = m.group(1), m.group(2), m.group(3)
        body = (f'{pre} bo:{pred} ?per_f . ?per_f rdfs:label ?per_fl . '
                f'FILTER({_regex_and("?per_fl", uri)})')
        return _tail(body, pre, m.group(4))
    s = re.sub(r"(\?\w+)\s+bo:(" + "|".join(_VOTO_PERSONA) + r")\s+br:([A-Za-z_]\w*)\s*([;.])", _per, s)

    # 4b) voto de GRUPO con un br:<algo> que NO es un grupo canónico -> es persona
    s = re.sub(r"(\?\w+)\s+bo:(votoAFavorDe|votoEnContraDe|seAbstuvo)\s+br:(?!grupo_)([A-Za-z_]\w*)\s*([;.])",
               _per_grp, s)

    # 5) rdfs:label "Nombre Apellido" EXACTO  ->  REGEX (el modelo pone el nombre
    #    literal, casi nunca coincide con el label completo del grafo + tildes).
    def _lbl(m):
        var, lit, term = m.group(1), m.group(2), m.group(3)
        if len(lit.split()) > 4 or len(lit) < 3:
            return m.group(0)
        body = f'{var} rdfs:label ?lbl_f . FILTER({_regex_and("?lbl_f", lit)})'
        return _tail(body, var, term)
    s = re.sub(r'(\?\w+)\s+rdfs:label\s+"([^"]{3,50})"\s*([;.)}])', _lbl, s)

    return s


def _tail(body, subj, term):
    if term == ";":
        return body + f" . {subj} "
    return body + " " + term


def _per_grp(m):
    pre, pred, uri = m.group(1), m.group(2), m.group(3)
    term = m.group(4) if (m.lastindex or 0) >= 4 else "."
    nom = {"votoAFavorDe": "concejalVotoAFavor", "votoEnContraDe": "concejalVotoEnContra",
           "seAbstuvo": "concejalVotoAbstencion"}[pred]
    body = (f'{pre} bo:{nom} ?per_f . ?per_f rdfs:label ?per_fl . '
            f'FILTER({_regex_and("?per_fl", uri)})')
    return _tail(body, pre, term)


_STOP = {"concejal", "concejala", "concejales", "sr", "sra", "don", "dona", "doña",
         "grupo", "municipal", "el", "la", "los", "las", "de", "del", "senor",
         "senora", "señor", "señora", "persona", "entidad"}


def _tokens(uri: str):
    u = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", uri)      # camelCase
    u = re.sub(r"[_\-]+", " ", u)
    toks = [w for w in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+", u.lower()) if len(w) > 1]
    keep = [w for w in toks if w not in _STOP]
    return keep or toks


def _regex_and(var: str, uri: str) -> str:
    toks = _tokens(uri) or [uri.lower()]
    return " && ".join(f'REGEX(STR({var}), "{t}", "i")' for t in toks)


def _sanitize(sparql, pregunta):
    try:
        from graph_rag_sparql import _sanitize_sparql
        return _sanitize_sparql(sparql, verbose=False, pregunta=pregunta)
    except Exception:
        return sparql


def post_F(sparql, pregunta):
    return _sanitize(alias_rewrite(sparql), pregunta)


# --- prompts de F: base compacta (D) y base completa-dinámica (E) ---------
def F_compact(pregunta):
    return A.D_compact_dyn(pregunta)


def F_full(pregunta):
    return A.E_full_dyn(pregunta)


ARMS_F = {
    "F_compact": F_compact,   # D_compact_dyn + post_F
    "F_full": F_full,         # E_full_dyn + post_F
}
POST = {"F_compact": post_F, "F_full": post_F}
POST_ALIAS = set(ARMS_F)     # compat
