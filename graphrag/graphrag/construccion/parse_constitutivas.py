import os, re, json, sys, unicodedata
from pypdf import PdfReader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # .../TFG/TFG
sys.path.insert(0, ROOT)
from backend.rag import DATA_PATH  # noqa: E402

ACTAS = {
    "2011-2015": os.path.join(DATA_PATH, "2011", "11-06-2011_Extraordinaria_Acta.pdf"),
    "2015-2019": os.path.join(DATA_PATH, "2015", "13-06-2015_Extraordinaria_Acta.pdf"),
    "2019-2023": os.path.join(DATA_PATH, "2019", "15-06-2019_Extraordinaria_Acta.pdf"),
    "2023-2027": os.path.join(DATA_PATH, "2023", "17-06-2023_Extraordinaria_Acta.pdf"),
}

PARTIDOS = [
    (r"nacionalista vasco|eaj-?pnv|euzko alderdi", "EAJ-PNV"),
    (r"socialista de euskadi|pse-?ee|psoe", "PSE-EE"),
    (r"euskal herria bildu|eh ?bildu|bildu-eusko|alternatiba eraikitzen", "EH BILDU"),
    (r"partido popular|\(pp\)", "PP"),
    (r"elkarrekin|podemos|ezker anitza|equo", "ELKARREKIN BILBAO"),
    (r"udalberri", "UDALBERRI"),
    (r"goazen", "GOAZEN BILBAO"),
    (r"ganemos", "GANEMOS"),
    (r"ezker batua|ezker anitza-iu", "EZKER BATUA-IU"),
    (r"vox", "VOX"),
]

SKIP = re.compile(
    r"secretar[ií]a general|udalbatzarreko|idazkari|^-\s*\d+\s*-$|jaun/andreok|"
    r"sr\.?\s*secretario|acto seguido|mesa de edad|hitza ematen|zin egiten|"
    r"presidencia|constituci[oó]n|^-+$|hautetsi|^\d+\.|electoral", re.I)


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


def lines_in_order(path, npages=4):
    r = PdfReader(path)
    out = []
    for pi in range(min(npages, len(r.pages))):
        parts = []
        r.pages[pi].extract_text(
            visitor_text=lambda t, cm, tm, f, s: parts.append((round(tm[5], 1), round(tm[4], 1), t))
            if t.strip() else None)
        parts.sort(key=lambda z: (-z[0], z[1]))
        cur, buf = None, []
        for y, x, tx in parts:
            if cur is None or abs(y - cur) < 3:
                buf.append((x, tx)); cur = y if cur is None else cur
            else:
                out.append("".join(t for _, t in sorted(buf))); buf = [(x, tx)]; cur = y
        if buf:
            out.append("".join(t for _, t in sorted(buf)))
    return out


def es_nombre(ln):
    ln = ln.strip(" .,-:")
    if not (6 <= len(ln) <= 55) or " " not in ln:
        return None
    letras = re.sub(r"[^A-Za-zÁÉÍÓÚÑÜ' -]", "", ln)
    if len(letras) < len(ln) - 1:
        return None
    # mayoritariamente mayusculas
    up = sum(1 for c in ln if c.isupper())
    lo = sum(1 for c in ln if c.islower())
    if up < 4 or lo > up:
        return None
    if SKIP.search(ln):
        return None
    return norm(ln).upper()


def detecta_partido(ln):
    low = norm(ln).lower()
    if ":" not in ln and "(" not in ln:
        return None
    for rx, canon in PARTIDOS:
        if re.search(rx, low):
            return canon
    return None


def main():
    HARD_STOP = re.compile(r"sr\.?\s*secretario|mesa de edad|acto seguido|"
                           r"comprobaci[oó]n de las credenciales|juramento o promesa", re.I)
    res = {}
    for mandato, path in ACTAS.items():
        tabla = {}
        cur = None
        for ln in lines_in_order(path):
            if HARD_STOP.search(ln):
                cur = None
                continue
            p = detecta_partido(ln)
            if p:
                cur = p
                continue
            if cur is None:
                continue
            nom = es_nombre(ln)
            if nom:
                tabla.setdefault(cur, []).append(nom)
        res[mandato] = tabla
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
