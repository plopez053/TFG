import re

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


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    EJEMPLOS = [
        "Votos emitidos: 27 Votos afirmativos: 14 señoras/señores: Ma drazo, Sustatxa, Alcalde, Areso, Sabas, Sánchez, De Castro, Barkala, Maíz, Alonso, Anuzita, Urtasun, Ajuria y Abaunza. Votos negativos: 6 señoras/señores: Oleaga, Gil Invernon, Díez, Zurro, Gardiazabal y Delgado. Abstenciones: 7 señoras/señores: Basagoiti, Ruiz, Marcos, García, Hermosa, Rodrigo y Pontes. En base al informe",
        "Votos afirmativos: 16 señoras/señores: Arregi, Ajuria, Abaunza, Ibarretxe, Urtasun, Olabarria, Erroteta, Ochandiano, Claver, Odriozola, Inunciaga, Abete, Dí ez, Pérez, Bilbao Aldayturriaga y Bilbao Urquijo. Udalbatzako Idazkaritza Nagusia Secretaría General del Pleno 5 Abstentzioak: 12 jaun-andre: Del Río, Renedo, Perea, Orozco, Fernández, Undabarrena, Martínez, Rodrigo, Goti, Garagalza, Viñals eta Jiménez.",
        "Votos afirmativos: 1 9 señoras/ señores: Gil, Díez, Pérez, Bilbao, Abete, Alcalde, Abaunza, Arregi, Ibarretxe, Urtasun, Olabarria, Ajuria, Agirregoitia, Odriozola, Otxandiano, Alonso, Narbaiza, Erroteta y Zubizarreta Agirrezabal. Abstenciones: 10 señoras/señores: González, García, Rodrig o, Viñals, Jiménez, Muñoz, Goirizelaia, González, Fatuarte y Zubizarreta Unanue. Sobre la base",
        "Votos afirmativos: 27 señoras/señores: Aburto, Arregi, Ajuria ; Abaunza, Ibarretxe, Urtasun, Olabarria, Erroteta, Zubizarreta, Claver, Odriozola, Inunciaga, Abete, Díez, Pérez, Bilbao Aldayturriaga, Bilbao Urquijo, Del Río, Renedo, Perea, Orozco, Fernández, Undabarrena, Martínez, Rodrigo, Goti, Garagalza. Udalbatzako",
        "una proposición sin votación nominal, queda aprobada por unanimidad.",
    ]
    if len(sys.argv) > 1:
        EJEMPLOS = [open(sys.argv[1], encoding="utf-8").read()]
    for e in EJEMPLOS:
        r = parse_listas_voto(e)
        print(json.dumps(r, ensure_ascii=False, indent=1) if r else "None")
        print("-" * 60)
