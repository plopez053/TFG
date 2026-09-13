import os
import re
import sys
import glob
import json
import unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))            # .../graphrag/graphrag/construccion
GRAPHRAG = os.path.dirname(HERE)                               # .../graphrag/graphrag (datos + utils compartidos)
ROOT = os.path.dirname(os.path.dirname(GRAPHRAG))              # .../TFG/TFG (proyecto)
sys.path.insert(0, ROOT)
from backend.rag import DATA_PATH  # noqa: E402

OUT = os.path.join(GRAPHRAG, "concejales.jsonl")
OUT_TEC = os.path.join(GRAPHRAG, "personal_tecnico.jsonl")
PARTIDO_JSON = os.path.join(GRAPHRAG, "concejales_partido.json")

_TRAT = r"(?:don|do[nñ]a|d\.|d[nñ]a\.|excmo\.?\s*sr\.?\s*don|sr\.?\s*don|sra\.?\s*do[nñ]a)"


def _fecha_de_ruta(path: str) -> str:
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", os.path.basename(path))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


# clave para agrupar variantes de OCR del mismo nombre (sin espacios ni acentos)
def _clave_ocr(nombre: str) -> str:
    return re.sub(r"[^a-z]", "", _norm(nombre).lower())


# El Secretario General, el Interventor y los adjuntos a la Secretaría asisten a
# TODOS los plenos pero NO son concejales. Se excluyen por nombre y por contexto.
_NO_CONCEJALES = {
    "jon zabala basterra", "jesus alboniga iturbe", "jose luis burgos ibarguren",
    "javier balparda arregui", "imanol bilbao eizagirre",
    "mikel astorkiza abasolo", "mikel astorquiza abasolo",
    "alvaro maestro corrales", "aizbea atela uriarte",
}

# Fusión explícita de variantes de grafía que las heurísticas no colapsan solas
# (euskera/castellano, apodos, partículas). Clave = _clave_ocr(variante rara) ->
# _clave_ocr(nombre canónico que se conserva). Se aplica al recolectar.
_ALIAS_FUSION = {
    "xabierochandianomartinez": "xabierotxandianomartinez",
    "davidlopateguiescudero": "davidlopategiescudero",
    "inigozubizarretaaguirrezabal": "inigozubizarretaagirrezabal",
    "aitziberibaibarriagadeechevarrieta": "aitziberibaibarriagaetxebarrieta",
    "juanfelixmadariagabrizuela": "felixmadariagabrizuela",
    "txemaoleagazalvidea": "josemariaoleagazalvidea",
    "goyozurrotobajas": "gregoriozurrotobajas",
    "mariacarmenmunozlopez": "carmenmunozlopez",
    "patxixabierfernandezmonje": "xabierfernandezmonje",
    "mirenitxasoerrotetasagastagoya": "itxasoerrotetasagastagoya",
    "inigozubizarretagirrezabal": "inigozubizarretaagirrezabal",
}


def _limpia_nombre(nom: str) -> str:
    nom = nom.strip(" .,-")
    # un tratamiento a media cadena => empieza otra persona: cortar ahí
    nom = re.split(r"\s+(?:do[nñ]a?|d\.|d[nñ]a\.)\s+[A-ZÁÉÍÓÚ]", nom, 1)[0]
    # cortar coletillas que se cuelan al final
    nom = re.split(r"\b(asistid|a\s*sistid|estando|Interventor|Secretario|para celebrar|"
                   r"bajo la|que ser|siendo las|en el sal[oó]n)\b", nom, 1, re.I)[0]
    nom = re.split(r"\.\s+[A-Z]", nom, 1)[0]                    # "... Lopez. A sistidos..."
    nom = re.sub(r"\s*-\s*", "-", nom)                          # "Diez -Andino" -> "Diez-Andino"
    nom = re.sub(r"\s+y(\s+el)?\s*$", "", nom, flags=re.I)      # "... Rique y" / "... y el"
    # unir espacios espurios de OCR dentro de una palabra: la continuación
    # empieza SIEMPRE en minúscula ("H elena"->Helena, "Etx arte"->Etxarte,
    # "Fe lix"->Felix, "Alvare z"->Alvarez, "Lorenz o"->Lorenzo).
    nom = re.sub(r"\b(\w{1,3})\s+(?!de\b|del\b|la\b|las\b|los\b|y\b)([a-zà-ÿ]{2,})\b", r"\1\2", nom)
    nom = re.sub(r"\b(\w{2,})\s+([a-zà-ÿ])\b", r"\1\2", nom)
    return nom.strip(" .,-")


# de 'doña Nekane Alonso Santamaría, don Alfonso Gil Invernón y doña ...'
def _split_nombres(bloque: str):
    bloque = re.sub(r"\s+", " ", bloque)
    trozos = re.split(rf"\s*[,;]\s*|\s+y\s+(?={_TRAT}\b)", bloque, flags=re.I)
    nombres = []
    for t in trozos:
        m = re.match(rf"^{_TRAT}\s+(.+)$", t.strip(), re.I)
        if not m:
            continue
        nom = _limpia_nombre(m.group(1))
        if not (4 <= len(nom) <= 55 and " " in nom and not re.search(r"\d", nom)):
            continue
        if _norm(nom).lower() in _NO_CONCEJALES:
            continue
        nombres.append(nom)
    return nombres


# variantes de apellido (eu/es, plegado fonético) de un nombre completo
def _apellidos_variantes(nombre_completo: str):
    palabras = _norm(nombre_completo).split()
    if len(palabras) < 2:
        return {nombre_completo.upper()}
    # apellidos = las últimas 2 palabras salvo que haya partículas (de, del, la...)
    corte = 1 if len(palabras) <= 3 else 2
    apes = palabras[corte:]
    v = {" ".join(apes).upper(), apes[0].upper()}
    if len(apes) >= 2:
        v.add(apes[-1].upper())
        v.add(" ".join(apes[:2]).upper())
    return {x for x in v if len(x) >= 3}


def extraer():
    partido_tab = {}
    if os.path.exists(PARTIDO_JSON):
        raw = json.load(open(PARTIDO_JSON, encoding="utf-8"))
        # {mandato: {NOMBRE: grupo}} -> buscamos por nombre normalizado.
        # Se ignora cualquier clave que empiece por "_" (comentarios/instrucciones).
        for mandato, gente in raw.items():
            if mandato.startswith("_") or not isinstance(gente, dict):
                continue
            for nom, grp in gente.items():
                partido_tab[_norm(nom).upper()] = grp
        print(f"[*] tabla de partidos: {len(partido_tab)} concejales curados")
    else:
        print(f"[!] no existe {os.path.basename(PARTIDO_JSON)} -> todos los grupos 'Desconocido'. "
              "Créalo (plantilla en PLAN_MEJORA_GRAFO.md, Tarea 1).")

    pdfs = sorted(glob.glob(os.path.join(DATA_PATH, "**", "*.pdf"), recursive=True))
    print(f"[*] {len(pdfs)} actas")

    # persona (clave sin espacios ni acentos) -> {variantes:Counter, fechas:set, alcalde:bool}
    from collections import Counter as _C
    personas = defaultdict(lambda: {"variantes": _C(), "fechas": set(), "alcalde": False})
    # personal técnico: {clave -> {variantes, fechas, cargos:set}}
    tecnicos = defaultdict(lambda: {"variantes": _C(), "fechas": set(), "cargos": set()})

    for path in pdfs:
        fecha = _fecha_de_ruta(path)
        try:
            from pypdf import PdfReader
            txt = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages[:4])
        except Exception:
            continue
        txt = re.sub(r"\s+", " ", txt)

        # Alcalde/sa
        m_alc = re.search(rf"[Pp]residencia del .{{0,60}}?Alcalde[^:]{{0,40}}?:?\s*{_TRAT}\s+([A-ZÁÉÍÓÚÑa-záéíóúñ' ]{{6,50}}?)(?:\s+(?:En el|En la|bajo|,|para))",
                          txt)
        if not m_alc:
            m_alc = re.search(rf"bajo la presidencia del .{{0,40}}?Alcalde[^,]{{0,30}}?{_TRAT}\s+([A-Za-záéíóúñ' ]{{6,50}}?)\s+y,?\s+para", txt)
        if m_alc:
            nom = _norm(_limpia_nombre(m_alc.group(1)))
            k = _clave_ocr(nom)
            if len(k) >= 8:
                personas[k]["variantes"][nom] += 1
                personas[k]["alcalde"] = True
                if fecha:
                    personas[k]["fechas"].add(fecha)

        # Personal técnico (Secretario General, Interventor)
        for m in re.finditer(rf"(Secretari[oa] General del Pleno|Interventor[a]? General(?: Municipal)?)[,]?\s+{_TRAT}\s+([A-Za-záéíóúñ' ]{{6,45}}?)(?:\s*(?:y|,|\.|estando|$))",
                             txt, re.I):
            cargo = "Secretario General" if "secretari" in m.group(1).lower() else "Interventor General"
            nom = _norm(_limpia_nombre(m.group(2)))
            k = _clave_ocr(nom)
            if len(k) >= 8:
                tecnicos[k]["variantes"][nom] += 1
                tecnicos[k]["cargos"].add(cargo)
                if fecha:
                    tecnicos[k]["fechas"].add(fecha)

        # Tenientes de Alcalde + Concejales/as
        for m in re.finditer(r"(?:Tenientes de Alcalde|Concejal(?:es)?/?a?s?)\s*:?\s*(.{20,1500}?)(?:asistid[oa]s? por|estando presente|Existe,? en consecuencia|Secretario General del Pleno)",
                             txt, re.I):
            for nom in _split_nombres(m.group(1)):
                nom = _norm(nom)
                k = _clave_ocr(nom)
                if len(k) < 8:
                    continue
                personas[k]["variantes"][nom] += 1
                if fecha:
                    personas[k]["fechas"].add(fecha)

    # fusión explícita de variantes de grafía (euskera/castellano, apodos)
    for var, canon in _ALIAS_FUSION.items():
        if var in personas and var != canon:
            dst = personas[canon]
            src = personas.pop(var)
            dst["variantes"].update(src["variantes"])
            dst["fechas"] |= src["fechas"]
            dst["alcalde"] = dst["alcalde"] or src["alcalde"]

    # volcado
    def _key_fecha(f):
        d = f.split("-")
        return (int(d[2]), int(d[1]), int(d[0])) if len(d) == 3 else (0, 0, 0)

    _PART = ("de", "del", "la", "las", "los", "y", "ma", "mª")

    def _fold(s):
        # pliegue fonético euskera<->castellano + a-z0-9 para casar apellidos
        for a, b in (("tx", "ch"), ("tz", "z"), ("gui", "gi"), ("gue", "ge"),
                     ("qu", "k"), ("gu", "g"), ("v", "b"), ("ss", "s"), ("ii", "i"),
                     ("y", "i"), ("z", "s")):
            s = s.replace(a, b)
        return re.sub(r"[^a-z0-9]", "", s)

    def _partes(nom):
        w = [x for x in _norm(nom).lower().split() if x not in _PART]
        corte = 1 if len(w) <= 3 else 2
        return set(w[:corte] or w[:1]), _fold("".join(w[corte:] or w[-1:]))

    def _fwords(nom):
        return [_fold(x) for x in _norm(nom).lower().split() if x not in _PART]

    def _es_prefijo(a, b):
        # a y b son listas de palabras plegadas; ¿una es prefijo de la otra
        # (nombre truncado, p.ej. "jose lorenzo delgado" c "...delgado vicente")?
        s, l = (a, b) if len(a) <= len(b) else (b, a)
        return len(s) >= 2 and 0 < len(l) - len(s) <= 2 and l[:len(s)] == s

    # tabla de partidos también indexada por apellidos plegados (fallback para
    # variantes de grafía: "Goyo/Gregorio Zurro Tobajas", "Diez-Andino"...)
    part_by_ape = {}
    for nom_t, grp_t in list(partido_tab.items()):
        part_by_ape[_partes(nom_t)[1]] = grp_t

    def _grupo_de(nombre):
        g = partido_tab.get(_norm(nombre).upper())
        if g:
            return g
        return part_by_ape.get(_partes(nombre)[1], "Desconocido")

    filas = []
    for k, v in personas.items():
        fechas = sorted(v["fechas"], key=_key_fecha)
        cand = sorted(v["variantes"].items(),
                      key=lambda kv: (-kv[1], sum(1 for w in kv[0].split() if len(w) <= 2)))
        nombre = cand[0][0]
        if len(nombre.split()) < 2 or len(fechas) < 2:
            continue  # ruido: solo aparece 1 vez, o nombre incompleto
        given, apes = _partes(nombre)
        filas.append({
            "nombre": nombre,
            "apellidos": sorted(_apellidos_variantes(nombre)),
            "grupo": _grupo_de(nombre),
            "desde": fechas[0],
            "hasta": fechas[-1],
            "n_actas": len(fechas),
            "es_alcalde": v["alcalde"],
            "_given": given,
            "_apes": apes,
            "_fw": _fwords(nombre),
        })

    # Segunda pasada de fusión: mismos apellidos plegados y nombres de pila que
    # se solapan -> misma persona ("Itziar Miren Urtasun Jimeno" == "Itziar
    # Urtasun Jimeno"; "Xabier Otxandiano" == "Xabier Ochandiano"). Hermanos con
    # los mismos apellidos NO se fusionan (nombres de pila disjuntos: Ángel vs
    # Gabriel Rodrigo Izquierdo). Gana la variante con más n_actas.
    filas.sort(key=lambda x: -x["n_actas"])
    fusion = []
    for r in filas:
        for f0 in fusion:
            misma = (f0["_apes"] == r["_apes"] and (f0["_given"] & r["_given"])) or \
                    (_es_prefijo(f0["_fw"], r["_fw"]) and next(iter(f0["_given"] or [""])) == next(iter(r["_given"] or [""])))
            if misma:
                f0["n_actas"] += r["n_actas"]
                f0["desde"] = min(f0["desde"], r["desde"], key=_key_fecha)
                f0["hasta"] = max(f0["hasta"], r["hasta"], key=_key_fecha)
                f0["es_alcalde"] = f0["es_alcalde"] or r["es_alcalde"]
                f0["_given"] |= r["_given"]
                if len(r["_fw"]) > len(f0["_fw"]):
                    f0["_fw"] = r["_fw"]
                if f0["grupo"] == "Desconocido":
                    f0["grupo"] = r["grupo"]
                break
        else:
            fusion.append(r)
    for r in fusion:
        for kk in ("_given", "_apes", "_fw"):
            r.pop(kk, None)
    filas = sorted(fusion, key=lambda x: (-x["n_actas"], x["nombre"]))

    with open(OUT, "w", encoding="utf-8") as f:
        for r in filas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sin_grupo = sum(1 for r in filas if r["grupo"] == "Desconocido")
    print(f"[+] {len(filas)} concejales -> {OUT}")
    print(f"    con grupo: {len(filas) - sin_grupo} | sin grupo: {sin_grupo}")

    # personal técnico
    tfilas = []
    for k, v in tecnicos.items():
        fechas = sorted(v["fechas"], key=_key_fecha)
        if len(fechas) < 2:
            continue
        cand = sorted(v["variantes"].items(),
                      key=lambda kv: (-kv[1], sum(1 for w in kv[0].split() if len(w) <= 2)))
        tfilas.append({
            "nombre": cand[0][0],
            "apellidos": sorted(_apellidos_variantes(cand[0][0])),
            "cargo": " / ".join(sorted(v["cargos"])),
            "desde": fechas[0], "hasta": fechas[-1], "n_actas": len(fechas),
        })
    tfilas.sort(key=lambda x: -x["n_actas"])
    with open(OUT_TEC, "w", encoding="utf-8") as f:
        for r in tfilas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] {len(tfilas)} personal técnico -> {OUT_TEC}")
    print("    revisa a mano: nombres, partidos, fechas de mandato, "
          "y los que salgan con pocas n_actas (posible ruido de OCR).")
    return filas


if __name__ == "__main__":
    extraer()
