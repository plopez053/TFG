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
# Cada palabra clave se escribe con un espacio opcional entre cada letra
# ("g\s?r\s?u\s?p\s?o") para tolerar el corte de OCR que mete un espacio
# suelto en cualquier punto de la palabra ("Gru po", "Gr upo", "pre senta",
# "presenta e l"...) -- sin esta tolerancia el patrón estructurado falla y
# el código cae al fallback de "nombre de grupo en cualquier parte del
# texto", que a veces encuentra un grupo distinto mencionado de forma
# incidental más adelante en la página y atribuye la proposición al grupo
# equivocado (verificado con varios casos reales del corpus).
_QUE = r"q\s?u\s?e"                          # que
_P = r"p\s?r\s?e\s?s\s?e\s?n\s?t\s?a"       # presenta
_F = r"f\s?o\s?r\s?m\s?u\s?l\s?a"           # formula
_EL = r"(?:e\s?l|l\s?o\s?s)"                # el/los
_GRUPO_RE = re.compile(
    rf"(?:{_QUE}\s+(?:{_P}|{_F})[n]?\s+{_EL}|{_P}d[ao]\s+por\s+{_EL}|del|de\s+la|de\s+los)\s+"
    r"g\s?r\s?u\s?p\s?o[s]?\s+"
    # Terminador del nombre capturado: coma/punto/"cuya"/"presenta", o "que"
    # como palabra suelta -- "...PARTIDO POPULAR que plantea/propone/insta..."
    # es un patrón habitual y, sin el "\bque\b" genérico (antes solo se
    # aceptaba "que su"), la captura perezosa no encontraba terminador
    # dentro del límite de 60 caracteres, la coincidencia fallaba entera en
    # esa posición y el motor de regex probaba más adelante en el texto,
    # devolviendo una captura completamente distinta y equivocada.
    r"(?:pol[ií]tico[s]?\s+)?(?:municipal(?:es)?\s+)?(.{3,60}?)(?:\s*,|\.|cuya|presenta|\bque\b|$)",
    re.IGNORECASE,
)

# Proposiciones en euskera: "X udal taldeak aurkezten duen"
_GRUPO_EUSKERA_RE = re.compile(
    r"^(.{3,60}?)\s+udal\s+tald(?:eak|e(?:ak)?)\s+aurkezten",
    re.IGNORECASE,
)

# Variante en euskera: "PROPOSIZIOA, X udal taldearena" / "PROPOSIZIOA X udal
# talde politikoarena" (posesivo: "propuesta DEL grupo X"), en vez de "X udal
# taldeak aurkezten duen" (X presenta). No anclado a ^ porque casi siempre
# viene precedido del número de punto y la palabra PROPOSIZIOA.
_GRUPO_EUSKERA2_RE = re.compile(
    r"proposizioa[,:]?\s+(.{3,60}?)\s+udal\s+tald",
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
    r"|PARTIDO\s+POPULAR|GRUPO\s+POPULAR|P\.?P\.?"
    r"|UDALBERRI"
    r"|BILBAO\s+EN\s+COM[ÚU]N"
    r"|ARALAR"
    # Único término de la lista que también es una palabra común del español
    # ("los ciudadanos" = "the citizens", no el partido Ciudadanos/Cs). Se
    # excluye la variante todo-minúscula para no confundirla con el sustantivo
    # genérico -- verificado: las 7 proposiciones que esta regex atribuía al
    # partido CIUDADANOS en todo el corpus eran en realidad falsos positivos
    # de "los ciudadanos"/"a los ciudadanos" en el cuerpo del texto.
    r"|(?-i:CIUDADANOS|Ciudadanos)"
    r"|VOX"
    r"|EQUIPO\s+DE\s+GOBIERNO|GOBIERNO\s+MUNICIPAL|GORBERNU\s+TALDEA"
    r"|GRUPO\s+MIXTO"
    r")\b",
    re.IGNORECASE,
)


# Los puntos "Se da cuenta de..." (resoluciones de Alcaldía, acuerdos de la
# Junta de Gobierno, órdenes forales...) son actos administrativos que el
# Alcalde/Equipo de Gobierno informa al Pleno -- nunca una proposición
# presentada por un grupo. Se resuelven aparte, sin escanear el texto
# completo del punto: si no se hiciera así, el heurístico de "nombre de
# grupo mencionado en el texto" termina cogiendo cualquier grupo que
# aparezca de forma incidental en el resto de la página (censos de
# Consejos de Distrito, listas de voto de otro punto...), atribuyendo el
# punto a un grupo que no tiene nada que ver con él. Verificado con un caso
# real: acta 28-11-2007, punto 10 ("Se da cuenta de la resolución de la
# Alcaldía..."), que el heurístico de texto completo atribuía a EAJ-PNV
# solo porque esa página incluye el censo de Consejos de Distrito.
_DA_CUENTA_RE = re.compile(r"^\s*\d+[.\-]?\s*se\s+da\s+cuenta", re.IGNORECASE)
# Sobre texto despaciado (mismo truco que arriba, tolera cortes de OCR tipo
# "Al cald�a" -> "ALCALD�A" al quitar espacios).
_DA_CUENTA_EJECUTIVO_RE = re.compile(
    r"ALCALD|GOBIERN|GORBERNU|DELEGA|ORDENFORAL",
)

# Otros puntos del orden del día que NUNCA los presenta un grupo político,
# sino un particular, una asociación vecinal, o son trámites del propio
# Pleno (aprobar el acta anterior, tomar posesión/conocimiento de altas y
# bajas de Concejales, declarar la urgencia de la convocatoria...).
# Verificado contra el corpus completo: el heurístico de texto completo los
# estaba atribuyendo a grupos políticos que solo aparecían mencionados de
# forma incidental en el resto de la página (p.ej. una "Proposición vecinal"
# acababa atribuida al Equipo de Gobierno porque su respuesta en el debate
# la firmaba ese grupo, no porque la hubiera presentado).
_SIN_GRUPO_RE = re.compile(
    r"aprobar.{0,90}actas?\s+de\s+l\s?a\s?s?\s+sesi[oó]n"
    r"|toma\s+de\s+posesi[oó]n"
    r"|toma\s+de\s+conocimiento.{0,60}renuncia"
    r"|se\s+da\s+lectura.{0,60}renuncia"
    r"|(?:declaraci[oó]n|ratificaci[oó]n)\s+de\s+(?:la\s+)?u\s?r\s?g\s?e\s?n\s?c\s?i\s?a"
    r"|proposici[oó]n\s+vecinal"
    r"|proposici[oó]n.{0,80}presentad[ao]\s+(?:por\s+)?(?:don|do[ñn]a)\b"
    r"|proposici[oó]n.{0,80}de\s+la\s+asociaci[oó]n"
    r"|proposici[oó]n.{0,80}presentad[ao].{0,60}asociaci[oó]n"
    # Mecanismo de "proposición ciudadana" del ROM: siempre encabezada
    # "Proposición de fecha [DATE]..." (particular o asociación vecinal,
    # nunca un grupo político) -- verificado: de las 41 proposiciones con
    # este encabezado en todo el corpus, ninguna menciona la palabra
    # "grupo". Cubre también los casos de asociaciones no detectadas por
    # el patrón anterior (AMPA, comunidades de propietarios...).
    r"|^\s*\d+[.\-]?\s*proposici[oó]n\s+d\s?e\s+fecha"
    r"|resultado\s+de\s+la[s]?\s+.{0,20}elecciones\s+municipales"
    r"|premiazko",  # euskera: "deialdia premiazkoa" = declaración de urgencia
    re.IGNORECASE,
)

_PROPUESTA_EJECUTIVA_RE = re.compile(r"^\s*[.\-]?\s*\d*[.\-]*\s*(?:propuesta|proposamena)\b", re.IGNORECASE)
# Enmienda/modificación NUMERADA suelta a un artículo concreto de una
# ordenanza en tramitación -- verificado contra el PDF (ver más abajo): el
# texto disponible para estos puntos nunca incluye quién la presentó, solo
# el número de enmienda y su contenido, así que no se le atribuye Equipo de
# Gobierno por defecto (podría ser de cualquier grupo, o del propio
# Gobierno) -- se deja como Desconocido.
_PROPUESTA_ENMIENDA_RE = re.compile(
    r"(?:propuesta|proposamena)\s*[,:]?\s+de\s+enmienda\b"
    r"|(?:enmienda|modificaci\s?[oó]n)\s*n\s?[uú]\s?mero",
    re.IGNORECASE,
)

# Otras aperturas institucionales sin grupo: cita legal ("El artículo 123 de
# la Ley 7/1985..."), ratificación de un acuerdo de un Pleno anterior ("El
# Excmo./Excelentísimo Ayuntamiento Pleno, en sesión celebrada..."), o un
# acuerdo de la Junta de Gobierno citado directamente sin pasar por "Se da
# cuenta de...". Mismo caso que _DA_CUENTA_RE, solo que con otra redacción.
_LEGAL_EJECUTIVO_RE = re.compile(
    r"^\s*el\s+art[ií]culo\s+\d"
    r"|^\s*el\s+exc(?:m|elent[ií]sim)o\.?\s+ayuntamiento\s+pleno"
    r"|^\s*la\s+junta\s+de\s+gobierno"
    r"|^\s*\d*[.\-]?\s*la\s+alcald"
    r"|^\s*\d*[.\-]?\s*mediante\s+acuerdo"
    r"|^\s*\d*[.\-]?\s*el\s+[aá]rea\s+de"
    r"|^\s*\d*[.\-]?\s*daci[oó]n\s+de\s+cuenta"
    r"|^\s*\d*[.\-]?\s*el\s+pleno\s+de\s+la\s+corporaci[oó]n",
    re.IGNORECASE,
)


# determina el grupo proponente probando, en orden: patrón estructurado
def extrae_grupo(topic: str, text: str = "") -> str:
    if _SIN_GRUPO_RE.search(topic or ""):
        return "Desconocido"
    if _DA_CUENTA_RE.search(topic or ""):
        candidato = re.sub(r"\s+", "", ((topic or "") + (text or "")[:300]).upper())
        if _DA_CUENTA_EJECUTIVO_RE.search(candidato):
            return "EQUIPO DE GOBIERNO"
        return "Desconocido"
    # Las tres comprobaciones de abajo usan canon_grupo(), no normaliza_grupo()
    # directamente: normaliza_grupo() devuelve el texto crudo tal cual (sin
    # mapear a "Desconocido") cuando no reconoce ningún grupo -- si aquí se
    # aceptara ese texto crudo como "encontrado", una captura basura (p.ej.
    # un fragmento de OCR roto como "m unicipal") cortaría la búsqueda antes
    # de tiempo e impediría que los patrones siguientes (que sí habrían
    # dado con el grupo correcto) llegaran a probarse. canon_grupo() sí
    # distingue "es un grupo canónico real" de "no se reconoce nada".
    for src in (topic or "", (text or "")[:600]):
        m = _GRUPO_RE.search(src)
        if m:
            raw = m.group(1).strip()
            # Para conjuntas "PSE-EE, EH BILDU y PP" tomar solo el primero
            raw = re.split(r"\s*[,y]\s+(?:EH|PP|PSE|EAJ|ELK|GOA|UDA)", raw)[0]
            g = canon_grupo(raw)
            if g != "Desconocido":
                return g
    for src in (topic or "", (text or "")[:600]):
        m = _GRUPO_EUSKERA_RE.search(src)
        if m:
            g = canon_grupo(m.group(1))
            if g != "Desconocido":
                return g
    for src in (topic or "", (text or "")[:600]):
        m = _GRUPO_EUSKERA2_RE.search(src)
        if m:
            g = canon_grupo(m.group(1))
            if g != "Desconocido":
                return g
    for src in (topic or "", (text or "")[:900]):
        m = _PARTIDO_DIRECTO_RE.search(src)
        if m:
            g = canon_grupo(m.group(1))
            if g != "Desconocido":
                return g
    # Último recurso: los puntos "Propuesta.../PROPOSAMENA..." que llegan
    # hasta aquí sin que ningún patrón anterior encontrara un grupo son,
    # verificado contra varias actas (presupuesto general, créditos
    # adicionales, resolución de alegaciones, modificación de Estatutos...),
    # acuerdos de la Junta de Gobierno elevados al Pleno para su aprobación
    # -- la propia acta los agrupa bajo el epígrafe "PROPUESTAS-PROPOSAMENAK",
    # distinto de "PROPOSICIONES" (que sí son de un grupo). Se excluye
    # explícitamente "propuesta de enmienda" (una enmienda numerada suelta a
    # un artículo, sin autor identificable en el texto disponible -- se deja
    # como Desconocido, no se le atribuye el Gobierno por defecto).
    if _PROPUESTA_EJECUTIVA_RE.search(topic or "") and not _PROPUESTA_ENMIENDA_RE.search(topic or ""):
        return "EQUIPO DE GOBIERNO"
    if _LEGAL_EJECUTIVO_RE.search(topic or ""):
        return "EQUIPO DE GOBIERNO"
    return "Desconocido"


# Terminador común para las dos extracciones de abajo: coma, punto, el
# carácter de viñeta suelto que deja el OCR de las actas ("�"), o alguna de
# las palabras que suelen venir justo después del nombre del particular o
# la asociación en este corpus ("mediante la cual...", "para pedir...",
# "en la que...", "relativa a...", "denominada...", "en representación de").
_TERM_PARTICULAR = (
    r"(?:\s*,|\.|�|\s+mediante|\s+para\s+(?:pedir|solicitar)"
    r"|\s+en\s+la\s+que|\s+relativa\s+a|\s+denominada"
    r"|\s+en\s+representaci[oó]n|\s+correspondiente|$)"
)
# (?-i:[A-ZÁÉÍÓÚÑ0-9]...): la primera letra tras "asociación/ampa/..." tiene
# que ser mayúscula (nombre propio) -- si no, es porque el nombre real no
# contenía esa palabra clave (p.ej. "Etxerat Elkartea", sin "asociación") y
# esta regex habría saltado a una mención genérica posterior tipo "la
# Asociación presentó escrito..." capturando un fragmento de frase, no un
# nombre. Verificado con un caso real del corpus.
_PARTICULAR_ASOCIACION_RE = re.compile(
    rf"((?:asociaci[oó]n|ampa)\s+(?-i:[A-ZÁÉÍÓÚÑ0-9].{{1,68}}?)"
    rf"|comunidad\s+de\s+propietarios\s+(?-i:(?:de\s+)?[A-ZÁÉÍÓÚÑ0-9].{{1,68}}?)){_TERM_PARTICULAR}",
    re.IGNORECASE,
)
_PARTICULAR_PERSONA_RE = re.compile(
    rf"(?:don|do[ñn]a)\s+((?-i:[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ.'\-]+"
    rf"(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ.'\-]+){{0,4}})){_TERM_PARTICULAR}"
)


# Para proposiciones SIN grupo político (mecanismo de "proposición
# ciudadana" del ROM): intenta identificar a quién la presentó de verdad
# -- un particular ("don/doña X") o una asociación/AMPA/comunidad de
# propietarios -- para que quede como dato consultable en el grafo en vez
# de perderse dentro de "Desconocido". Devuelve (nombre, tipo) o None.
# No se intenta con topics ya anonimizados en la propia fuente ("#...#",
# redactado por privacidad antes de que llegara a este pipeline) ni se
# inventa nada cuando el texto disponible no nombra a nadie.
def extrae_particular(topic: str):
    t = topic or ""
    m = _PARTICULAR_ASOCIACION_RE.search(t)
    # "#" marca un dato ya anonimizado en la propia fuente (redactado por
    # privacidad antes de este pipeline) -- se comprueba sobre el propio
    # nombre capturado, no sobre todo el topic: es habitual que el nombre
    # de la asociación (público) aparezca junto a la persona firmante
    # (anonimizada) en el mismo punto, y descartar el topic entero perdería
    # también el nombre público que sí se puede recuperar.
    if m and "#" not in m.group(1):
        return m.group(1).strip(), "organizacion"
    m = _PARTICULAR_PERSONA_RE.search(t)
    if m and "#" not in m.group(1):
        return m.group(1).strip(), "persona"
    return None


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
