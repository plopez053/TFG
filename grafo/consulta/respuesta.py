"""
GraphRAG: responde a una pregunta con el grafo RDF de las proposiciones del Pleno.

graph_answer() sigue este orden:
  1. análisis de la pregunta (pregunta.py)
  2. consulta montada por el programa, sin LLM (consulta_directa.py), si la
     pregunta es de un tipo previsto
  3. si no, plantillas (plantillas.py) y, si tampoco, SPARQL generado por el LLM
     (generacion.py), corregido por las guardas (guardas.py)
  4. narración de las filas por el LLM, con las fuentes

Uso desde la línea de comandos:
    python -m grafo.consulta.respuesta "¿Cuántas proposiciones sobre vivienda ha presentado cada grupo?"
"""
import glob
import os
import re
import sys
import urllib.parse
from collections import defaultdict
from dataclasses import replace

from comun.proveedores import GOOGLE_CLOUD_PROJECT
from comun.rutas import DATA_PATH
from grafo.consulta.consulta_directa import consulta_directa, forma
from grafo.consulta.generacion import (_fix_degenerate_groupby, _generar_y_ejecutar, _RESPUESTA_IRRECUPERABLE,
                                       _try_arm_g)
from grafo.consulta.guardas import _sanitize_sparql
from grafo.consulta.pregunta import (_indice_asunto, _norm_q, _preparar_asunto, _props_asunto, _props_temas, _raiz,
                                     analizar, cobertura)
from grafo.consulta.recursos import _ejecutar, _llm_invoke, _load_graph, _PREFIXES


# =============================================================================
# Narración de las filas y fuentes
# =============================================================================

# Narración de la respuesta: las filas del grafo, con etiquetas legibles y las
# notas que necesita el LLM, se convierten en texto; y las fuentes (página del PDF).

# Tamaño máximo de las filas y de la consulta en el prompt del narrador. El
# plan gratuito de Groq rechaza (413) cualquier petición de más de 8.000
# tokens por minuto: con 44 títulos largos se pasó y la narró el modelo local,
# que entró en un bucle.
NARRAR_MAX_CHARS_FILAS = 7000
NARRAR_MAX_CHARS_SPARQL = 2500


# etiqueta legible de cada URI del grafo (rdfs:label o skos:prefLabel)
def _etiquetas_uris(g, uris: list) -> dict:
    if not uris:
        return {}
    values = " ".join(f"<{u}>" for u in uris)
    q = _PREFIXES + f"""
    SELECT ?u ?l WHERE {{ VALUES ?u {{ {values} }}
      {{ ?u rdfs:label ?l }} UNION {{ ?u skos:prefLabel ?l }} }}"""
    out = {}
    for row in g.query(q):
        out.setdefault(str(row.u), str(row.l))
    return out


# Filas tal como las ve el narrador. Una proposición se sustituye por su fecha
# y título (antes se quitaba y el narrador no podía citar el asunto); las
# demás URIs, por su etiqueta. Las filas de la misma proposición se fusionan
# (con un OPTIONAL de temas salía una fila por tema, y el narrador las contaba
# como ejemplos distintos). Se corta al llegar a NARRAR_MAX_CHARS_FILAS y se
# dice cuántas se han omitido.
def _filas_para_narrar(g, rows: list, max_chars: int = 160) -> str:
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return "(sin resultados en el grafo)"
    props = list(dict.fromkeys(v for r in rows[:200] for v in r.values() if _PROP_URI_RE.match(v)))
    otras = list(dict.fromkeys(v for r in rows[:200] for v in r.values()
                               if v.startswith("http://bilbao.tfg/") and not _PROP_URI_RE.match(v)))
    info = {}
    if props:
        q = _PREFIXES + f"""
        SELECT ?p ?fecha ?titulo WHERE {{ VALUES ?p {{ {" ".join(f"<{u}>" for u in props)} }}
          ?p bo:fecha ?fecha ; bo:tituloTopic ?titulo . }}"""
        for row in g.query(q):
            info[str(row.p)] = f"{row.fecha} — {str(row.titulo)[:max_chars]}"
    etiquetas = _etiquetas_uris(g, otras)

    def legible(v):
        if _PROP_URI_RE.match(v):
            return info.get(v, "proposición sin título")
        if v.startswith("http://bilbao.tfg/"):
            return etiquetas.get(v) or re.sub(r"^t_[0-9a-f]+_", "", re.split(r"[/#]", v)[-1]).replace("_", " ")
        return v if len(v) <= max_chars else v[:max_chars] + "..."

    # fusión por proposición: una fila por ?p, con los valores distintos del resto de columnas
    fusion, orden = {}, []
    for r in rows:
        clave = next((v for v in r.values() if _PROP_URI_RE.match(v)), None) or repr(sorted(r.items()))
        if clave not in fusion:
            fusion[clave] = {}
            orden.append(clave)
        for k, v in r.items():
            vals = fusion[clave].setdefault(k, [])
            if (lv := legible(v)) not in vals:
                vals.append(lv)

    lineas, usados = [], 0
    for clave in orden:
        fila = {k: "; ".join(v) for k, v in fusion[clave].items()}
        linea = str(fila)
        if usados + len(linea) > NARRAR_MAX_CHARS_FILAS and lineas:
            lineas.append(f"(... y {len(orden) - len(lineas)} filas más que no se muestran por espacio; "
                          f"en total la consulta devolvió {len(orden)} resultados distintos)")
            break
        lineas.append(linea)
        usados += len(linea)
    return "\n".join(lineas)


# Quita las líneas repetidas de una narración en bucle (el modelo local, sin
# límite de repetición, repitió la misma viñeta cientos de veces).
def _quitar_bucles(texto: str) -> str:
    vistas, out = set(), []
    for linea in texto.split("\n"):
        norm = re.sub(r"\s+", " ", linea).strip().lower()
        if len(norm) > 25 and norm in vistas and not re.fullmatch(r"[|:\- ]+", norm):
            continue
        vistas.add(norm)
        out.append(linea)
    return "\n".join(out)


# si la pregunta pide un %, calcula el ratio en código y lo inyecta al contexto
def _augment_ratios(rows: list, pregunta: str) -> str:
    if not re.search(r"porcentaje|proporci[oó]n|\btasa\b|ratio|\bpor ?ciento\b|%", pregunta, re.I):
        return ""
    out = []
    for r in rows[:20]:
        if not isinstance(r, dict):
            continue
        nums = [(k, int(v)) for k, v in r.items()
                if isinstance(v, str) and re.fullmatch(r"\d+", v)
                and not re.match(r"(?:anio|ano|year|fecha|mes|orden)", k.lower())]
        if len(nums) < 2:
            continue
        nums.sort(key=lambda kv: kv[1])
        (kn, n), (kd, d) = nums[0], nums[-1]
        if d > 0 and n <= d and kn != kd:
            etiqueta = " ".join(str(v) for k, v in r.items()
                                if isinstance(v, str) and not re.fullmatch(r"\d+", v)) or "total"
            out.append(f"  {etiqueta}: {n}/{d} = {100 * n / d:.1f}%")
    return ("\nPORCENTAJES YA CALCULADOS (usa EXACTAMENTE estos, no recalcules):\n"
            + "\n".join(out)) if out else ""


ANSWER_PROMPT = """Eres un analista político experto en el Ayuntamiento de Bilbao.
Basándote ÚNICAMENTE en los datos del grafo que te proporciono, genera una respuesta en español que sea:
- Narrativa y clara: no solo números, explica qué significan
- Precisa: cita las cifras exactas del grafo
- Contextual: si hay datos temporales, describe la evolución; si hay varios grupos, compáralos
- Completa: menciona los casos más destacados y cualquier patrón interesante

REGLA CRÍTICA: si los datos están vacíos ("sin resultados en el grafo"), responde honestamente que
no se encontraron datos para esa consulta. NUNCA inventes cifras hipotéticas ni pongas ejemplos
ilustrativos: cualquier cifra que no aparezca en los datos es una alucinación.

REGLA CRÍTICA sobre qué son las cifras: TODOS los números de DATOS DEL GRAFO cuentan
PROPOSICIONES (iniciativas presentadas en el Pleno) — NUNCA personas, "miembros",
concejales ni votantes, aunque la fila hable de un grupo político. Si una columna se
llama "n", "total" o similar junto a un grupo/tema/año, significa "número de
proposiciones", no "número de miembros del grupo".

REGLA CRÍTICA al comparar filas ("quién tiene más", "el más activo", rankings): lee las
cifras de TODAS las filas con cuidado antes de concluir cuál es la mayor — no asumas que
la segunda fila es la primera en importancia. Si vas a nombrar un "máximo" o "mínimo",
verifica que su cifra sea realmente la más alta/baja de todas las que ves en los datos.

REGLA CRÍTICA sobre qué cuenta cada cifra: la consulta SPARQL de abajo define exactamente qué
se ha contado. Describe cada cifra con los filtros que aparecen en ella (tema, grupo, años,
resultado). Si coinciden con lo que pide la pregunta, la cifra ES la respuesta: no pidas más
datos para confirmarla. Si la pregunta menciona algo por lo que la consulta NO filtra, dilo en
vez de atribuir la cifra a eso. Si en los datos aparece un AVISO, explícaselo al usuario.

REGLA CRÍTICA sobre los resultados: "Decae" significa que la proposición no llegó a votarse
porque se aprobó una enmienda que la sustituye (no que fuera rechazada). Un recuento de 0
significa que no consta ninguna en el grafo, que cubre de 2007 a 2026. No muestres URIs
(http://bilbao.tfg/...) ni identificadores internos: usa fechas y títulos.

REGLA CRÍTICA sobre la columna "relacion": "trata" significa que la proposición trata de ese
asunto (está en su título o en sus subtemas); "menciona" significa que solo aparece entre las
entidades que se mencionan en ella, pero la proposición trata de OTRA cosa. Nunca presentes una
fila "menciona" como si tratara del asunto: di que lo menciona y de qué trata en realidad.

REGLA CRÍTICA sobre valores "None"/vacíos en una fila: si una fila tiene una columna numérica
(COUNT, total...) con un valor real mayor que 0 pero OTRAS columnas de esa misma fila salen
"None" o vacías, NO significa que no haya datos — solo significa que esas columnas concretas
no se enlazaron en el SPARQL (variable sin usar en el WHERE). El número sigue siendo válido y
es la respuesta. Solo trata una pregunta como "sin datos" si TODAS las filas están vacías o
si la lista de filas está vacía del todo — nunca por ver "None" en una columna aislada.

REGLA CRÍTICA sobre LIMIT: si la consulta SPARQL de abajo termina en "LIMIT N", las filas
que ves son SOLO las N primeras de un ranking, NO todas. NUNCA digas que suman "el total",
"la totalidad", "todas las proposiciones del tema" ni "no hay más grupos/años relevantes":
hay más filas que la consulta no ha traído. Describe solo lo que ves ("los 3 grupos que
más han presentado son...") sin afirmar nada sobre el resto.

REGLA CRÍTICA sobre aritmética: NO calcules restas de años, porcentajes ni sumas que no
estén ya en los datos, salvo que la pregunta lo pida explícitamente y los números necesarios
estén los dos en las filas. Si mencionas dos años (p.ej. 2019 y 2022), NO añadas "X años
después" — limítate a nombrar los años. Un cálculo mental mal hecho es una alucinación.

CONSULTA SPARQL YA EJECUTADA (para que entiendas qué significan las filas, no para repetirla):
{sparql}

PREGUNTA: {pregunta}

DATOS DEL GRAFO (ya filtrados según la consulta de arriba):
{filas}

RESPUESTA:"""


# URI de una proposición (br:prop_<id>); todas tienen fecha, página, PDF y título
_PROP_URI_RE = re.compile(r'^http://bilbao\.tfg/resource/prop_[0-9a-f]+$')


# Datos de cita (fecha, página, PDF, título) de las proposiciones que aparecen
# en las filas, en una sola consulta. Un agregado puro (COUNT/GROUP BY sin ?p)
# no trae ninguna proposición que citar y devuelve [] (memoria 2.10).
def graph_sources(rows: list, limit: int = 15) -> list:
    uris = []
    seen = set()
    for row in rows:
        for v in row.values():
            if v not in seen and _PROP_URI_RE.match(v):
                seen.add(v)
                uris.append(v)
                if len(uris) >= limit:
                    break
        if len(uris) >= limit:
            break
    if not uris:
        return []

    g = _load_graph()
    values = " ".join(f"<{u}>" for u in uris)
    q = _PREFIXES + f"""
    SELECT ?p ?fecha ?pagina ?pdf ?titulo WHERE {{
      VALUES ?p {{ {values} }}
      ?p bo:fecha ?fecha ; bo:pagina ?pagina ; bo:fuentePdf ?pdf ; bo:tituloTopic ?titulo .
    }}"""
    out = []
    for row in g.query(q):
        d = {str(v): str(row[v]) for v in row.labels}
        out.append(d)
    # mismo orden que aparecieron en las filas originales, no el de la VALUES
    orden = {u: i for i, u in enumerate(uris)}
    out.sort(key=lambda d: orden.get(d.get("p", ""), 999))
    return out


# "¿qué se aprobó en el pleno del 15 de agosto de 2016?" sin ninguna
# proposición ese día: probablemente no hubo pleno; se dan los más cercanos
def _nota_fecha_sin_pleno(g, fecha: str) -> str:
    # si ese día sí hubo proposiciones, la consulta ha filtrado de más (resultado, tema...): no es esto
    if _ejecutar(g, _PREFIXES + f'SELECT ?p WHERE {{ ?p bo:fechaISO "{fecha}"^^xsd:date }} LIMIT 1'):
        return ""
    q = _PREFIXES + ("SELECT (MAX(?d) AS ?antes) (MIN(?d2) AS ?despues) WHERE {{ "
                     "{{ ?p bo:fechaISO ?d FILTER(?d < \"{f}\"^^xsd:date) }} UNION "
                     "{{ ?p2 bo:fechaISO ?d2 FILTER(?d2 > \"{f}\"^^xsd:date) }} }}").format(f=fecha)
    try:
        r = _ejecutar(g, q)[0]
    except Exception:
        return ""
    return (f"\nNOTA: no consta ningún pleno el {fecha} en las actas. Los plenos más cercanos son el "
            f"{_valor(r.get('antes')) or '—'} (anterior) y el {_valor(r.get('despues')) or '—'} (posterior). Dilo así.")


# valor de una fila, o None si no se enlazó ("None" es como llega una variable sin valor)
def _valor(x):
    return None if x in (None, "", "None") else x


# Todas las votaciones que el acta registra de cada proposición de las filas
# (bo:Votacion): la de cada enmienda y la del punto, con su resultado y el voto
# de cada grupo. Antes el narrador solo veía una votación por punto, a veces la
# de una enmienda.
def _votaciones_del_acta(g, rows) -> str:
    props = list(dict.fromkeys(str(r.get("p")) for r in rows if r.get("p")))[:5]
    if not props:
        return ""
    out = ""
    for p in props:
        # (sin GROUP_CONCAT: en rdflib falla con variables de un OPTIONAL sin valor)
        filas_v = _ejecutar(g, _PREFIXES + f"""SELECT ?v ?o ?obj ?desc ?dec ?f ?c ?a ?dc WHERE {{
            <{p}> bo:tieneVotacion ?v . ?v bo:ordenVotacion ?o ; bo:objetoVotacion ?obj ; rdfs:label ?desc .
            OPTIONAL {{ ?v bo:decision ?dec }} OPTIONAL {{ ?v bo:votosFavorVotacion ?f }}
            OPTIONAL {{ ?v bo:votosContraVotacion ?c }} OPTIONAL {{ ?v bo:abstencionesVotacion ?a }}
            OPTIONAL {{ ?v bo:esDecisiva ?dc }} }} ORDER BY ?o""")
        if not filas_v:
            continue
        grupos_v = defaultdict(lambda: defaultdict(list))
        for r in _ejecutar(g, _PREFIXES + f"""SELECT ?v ?s ?gl WHERE {{ <{p}> bo:tieneVotacion ?v .
                {{ ?v bo:grupoVotaAFavor ?gr BIND("favor" AS ?s) }} UNION {{ ?v bo:grupoVotaEnContra ?gr BIND("contra" AS ?s) }}
                UNION {{ ?v bo:grupoSeAbstiene ?gr BIND("abst" AS ?s) }} ?gr rdfs:label ?gl }}"""):
            grupos_v[r["v"]][r["s"]].append(r["gl"])
        vs = []
        for v in filas_v:
            v = dict(v)
            for k in ("favor", "contra", "abst"):
                v[k] = ", ".join(sorted(grupos_v[v["v"]][k]))
            vs.append(v)
        fecha = (_ejecutar(g, _PREFIXES + f"SELECT ?f WHERE {{ <{p}> bo:fecha ?f }}") or [{}])[0].get("f", "")
        out += f"\nVOTACIONES DEL ACTA de la proposición del {fecha}:"
        for v in vs:
            cifras = (f"{_valor(v.get('f')) or '?'} a favor, {_valor(v.get('c')) or 0} en contra, "
                      f"{_valor(v.get('a')) or 0} abstenciones")
            grupos = "; ".join(f"{k}: {v[k]}" for k in ("favor", "contra", "abst") if v.get(k) not in (None, "", "None"))
            out += (f"\n  - votación {v['o']} ({v['obj']}{', la que decide' if v.get('dc') in ('true', 'True') else ''}): "
                    f"{v['desc'][:160]} -> {cifras}" + (f" [{grupos}]" if grupos else "")
                    + (f" -> {v['dec'][:200]}" if v.get("dec") not in (None, "None") else ""))
    if out:
        out += ("\nUsa estas votaciones para decir cómo votó cada grupo: distingue la votación de una enmienda de la "
                "del punto (si se aprueba una enmienda de sustitución, la proposición decae y no se vota).")
    return out


# =============================================================================
# Fuentes: enlaces a las actas
# =============================================================================

# PDF del acta de una fecha 'DD-MM-YYYY', o "" si no se encuentra
def _find_pdf_by_date(fecha: str) -> str:
    parts = fecha.split("-")
    if len(parts) != 3:
        return ""
    matches = glob.glob(os.path.join(DATA_PATH, parts[-1], f"{fecha}_*.pdf"))
    return matches[0] if matches else ""


# Enlaces de una respuesta GraphRAG:
#   1. filas con la URI de una proposición: página exacta (graph_sources)
#   2. filas con solo una fecha: el PDF de ese pleno, sin página
#   3. resultado agregado (COUNT, GROUP BY): no hay documento que citar, se explica
# Sin filas (no consta, fuera de alcance) no se añade nada.
def fuentes_graphrag(rows: list) -> str:
    if not rows:
        return ""

    citas = graph_sources(rows)
    if citas:
        links = []
        for c in citas:
            pdf_rel = c["pdf"].replace("\\", "/")
            year_dir = os.path.basename(os.path.dirname(pdf_rel))
            fname = os.path.basename(pdf_rel)
            pagina = c.get("pagina", "")
            anchor = f"#page={pagina}" if pagina else ""
            url = f"/acta/{year_dir}/{urllib.parse.quote(fname)}{anchor}"

            label = f"Ver PDF — Acta {c['fecha']}"
            if pagina:
                label += f" (Pág. {pagina})"
            titulo = c.get("titulo", "")
            if titulo:
                short = titulo[:70] + "..." if len(titulo) > 70 else titulo
                label += f" | {short}"
            links.append(f"- [{label}]({url})")
        return "\n\n---\n**Proposiciones del grafo citadas:**\n" + "\n".join(links)

    DATE_KEYS = ("fecha", "fechaProp", "fechaPleno", "date")
    TITLE_KEYS = ("titulo", "tituloProp", "tituloTopic", "label")
    vistas: set = set()
    links = []
    for row in rows[:50]:
        fecha = next((row[k] for k in DATE_KEYS if k in row and re.match(r"\d{2}-\d{2}-\d{4}", row[k])), None)
        if not fecha or fecha in vistas:
            continue
        vistas.add(fecha)

        pdf = _find_pdf_by_date(fecha)
        if not pdf:
            continue
        year_dir = os.path.basename(os.path.dirname(pdf))
        fname = os.path.basename(pdf)
        url = f"/acta/{year_dir}/{urllib.parse.quote(fname)}"

        titulo = next((row[k] for k in TITLE_KEYS if k in row and row[k]), "")
        label = f"Ver PDF — Acta {fecha}"
        if titulo:
            short = titulo[:70] + "..." if len(titulo) > 70 else titulo
            label += f" | {short}"
        links.append(f"- [{label}]({url})")

    if links:
        return "\n\n---\n**Actas del grafo consultadas:**\n" + "\n".join(links)

    return ("\n\n---\n*Esta cifra se calcula directamente sobre el grafo "
            "estructurado del Pleno (proposiciones, grupos y temas ya "
            "extraídos de las actas); al ser un resultado agregado, no "
            "corresponde a un documento concreto que enlazar.*")


# =============================================================================
# Respuesta a una pregunta
# =============================================================================

_RESPUESTA_FUERA_DE_ALCANCE = {
    "importes": ("El grafo de proposiciones no guarda importes económicos, así que no puedo "
                 "calcular cuánto dinero se aprobó. Si la cifra aparece en el texto de alguna acta, "
                 "el perfil RAG Vectorial puede encontrarla; compruébala en el acta que cite."),
    "asistencia": ("El grafo no guarda la asistencia de los concejales a los plenos, así que no "
                   "puedo decir quién ha faltado más. Solo registra quién presenta, interviene y "
                   "vota cada proposición, y no votar no equivale a haber faltado."),
}


# Consulta montada sin LLM a partir del análisis (consulta_directa.py), ya
# saneada, ejecutada y comprobada con la cobertura; None si la pregunta no
# encaja en sus formas o algo falla, y entonces siguen las plantillas / el LLM.
def _consulta_directa_ejecutada(g, pregunta: str, analisis, verbose: bool):
    # los temas que ya están dentro del asunto no van como filtro aparte
    if analisis.temas_absorbidos:
        analisis = replace(analisis, temas=[t for t in analisis.temas if t not in analisis.temas_absorbidos])
    # "¿qué barrio ha tenido más...?": "barrio" es la agrupación, no el tema
    # barrios (en la sexta evaluación se filtró por él y salió 1 por entidad)
    if forma(_norm_q(pregunta)) in ("ranking_barrio", "ranking_distrito"):
        analisis = replace(analisis, temas=[t for t in analisis.temas if not re.match(r"barrio|distrito", t[1])])
    props, criterio = None, ""
    etiquetadas = forma(_norm_q(pregunta)) in ("ultima", "lista", "existe", "contenido", "votos")
    if analisis.asunto_raices:
        grupos = analisis.asunto_grupos or [analisis.asunto_raices]

        def buscar(gs):
            if etiquetadas:
                # "la última vez", los listados y las votaciones incluyen también
                # las menciones, etiquetadas, para distinguir tratar de mencionar
                trata = set(_props_asunto(g, gs, solo_trata=True))
                return [(p, "trata" if p in trata else "menciona") for p in _props_asunto(g, gs, con_menciones=True)]
            return _props_asunto(g, gs)
        props = buscar(grupos)
        # Ninguna contiene todas las palabras: se relaja a las imprescindibles
        # que marcó el LLM ("hermanamiento con alguna ciudad extranjera" ->
        # hermanamiento), en todas las formas, y se dice en "criterio". Antes se
        # relajaba a la palabra menos frecuente ("extranjera", "renfe").
        clave = [[_raiz(w) for w in c] for c in getattr(analisis, "asunto_clave", []) if c]
        # (no en votaciones: "¿qué grupo votó en contra de la ordenanza de
        # civismo?" pide el voto de algo concreto; relajar a "civismo" dio los
        # votos de otras proposiciones como si fueran los de esa ordenanza)
        if not props and clave and clave != grupos and forma(_norm_q(pregunta)) != "votos":
            props = buscar(clave)
            if props:
                criterio = (f"ninguna proposición contiene a la vez {', '.join(r for gr in grupos for r in gr)}; "
                            f"se buscan las que contienen {' / '.join(' + '.join(c) for c in clave)}")
        # en "¿se ha hablado alguna vez de X?" la respuesta ES si existe esa
        # combinación: no se relaja (con "plaga de ratas en el Casco Viejo" se
        # listaban proposiciones cualesquiera del Casco Viejo)
        if not props and forma(_norm_q(pregunta)) != "existe":
            # Ninguna contiene todas las palabras a la vez ("el Orgullo LGTBI":
            # hay 15 sobre LGTBI, ninguna dice "orgullo"). Se busca por la más
            # específica (un nombre propio si lo hay, si no la menos frecuente)
            # y se le dice al narrador en la columna "criterio": antes se
            # ensanchaba la búsqueda sin avisar.
            propios = {_raiz(w) for w in analisis.nombres_propios}
            cuenta = {r: len(_props_asunto(g, [[r]])) for grupo in grupos for r in grupo}
            # solo hacia un nombre propio o una palabra rara (< 1 %): relajar
            # hacia "instalación" devolvía cualquier instalación
            total = len(_indice_asunto(g))
            validas = [r for r, n in cuenta.items() if n and (r in propios or n <= 0.01 * total)]
            if validas and len(cuenta) > 1 and not clave:
                r = min(validas, key=lambda x: (x not in propios, cuenta[x]))
                props = buscar([[r]])
                criterio = (f"ninguna proposición contiene a la vez {', '.join(cuenta)}; "
                            f"se cuentan las que contienen '{r}'")
            elif analisis.temas and not clave and forma(_norm_q(pregunta)) in ("conteo", "ranking_grupo", "desglose_grupo", "ranking_anio"):
                # hay un tema reconocido y el asunto no añade nada que exista:
                # se cuenta el tema, diciéndolo ("implantación de la Zona 30"). Solo
                # en recuentos y solo sin núcleo del LLM: con él, las palabras que
                # quedan fuera del tema son el asunto de verdad ("subvenciones a
                # asociaciones vecinales" no son todas las subvenciones)
                criterio = (f"ninguna proposición del tema contiene {', '.join(cuenta)}; "
                            "se cuentan todas las del tema")
                sparql = consulta_directa(pregunta, replace(analisis, asunto=[], asunto_raices=[],
                                                            asunto_grupos=[]), None, criterio)
                return _ejecutar_directa(g, pregunta, analisis, sparql, verbose)
    if analisis.asunto_temas:
        # se suman las proposiciones de los temas que son alternativas del asunto
        de_temas = _props_temas(g, analisis.asunto_temas)
        if etiquetadas:
            rel = dict(props or [])
            rel.update({p: "trata" for p in de_temas})
            props = sorted(rel.items())
        else:
            props = sorted(set(props or []) | de_temas)
        etiquetas = [e for s, e in analisis.temas_absorbidos if s in analisis.asunto_temas]
        suma = (f"se suman las proposiciones del tema {' / '.join(etiquetas)}"
                + (f" y las que contienen {' / '.join(' + '.join(gr) for gr in analisis.asunto_grupos)}"
                   if analisis.asunto_raices else ""))
        criterio = f"{criterio}; {suma}" if criterio else suma
    sparql = consulta_directa(pregunta, analisis, props, criterio)
    return _ejecutar_directa(g, pregunta, analisis, sparql, verbose)


def _ejecutar_directa(g, pregunta: str, analisis, sparql, verbose: bool):
    if not sparql:
        return None
    sparql = _sanitize_sparql(sparql, False, pregunta)
    try:
        rows = _ejecutar(g, sparql)
    except Exception as e:
        print(f"[!] Consulta directa falló — {type(e).__name__}: {e}", flush=True)
        return None
    # el asunto va como VALUES (garantizado por construcción); el resto se comprueba.
    # En "¿qué opinó el PP...?" el grupo es quien habla, no un filtro de la consulta
    sin_grupo = forma(_norm_q(pregunta)) == "contenido"
    comprobar = replace(analisis, asunto=[], asunto_raices=[], grupos=[] if sin_grupo else analisis.grupos)
    # sin el comentario del asunto: sus palabras ("presupuesto general, 2016")
    # no son filtros de la consulta
    if cobertura(comprobar, re.sub(r"#[^\n]*", "", sparql)):
        return None
    if verbose:
        print(f"[Consulta directa, sin LLM]\n{sparql}\n")
    return sparql, rows


# sparql_provider: "ollama" (qwen3:8b, local; por defecto) o "groq"
# (gpt-oss-120b) para GENERAR la SPARQL. La narración usa siempre Groq.
_FORMAS_SIN_ASUNTO = ("existe", "ultima", "contenido", "lista", "votos", "conteo", "ranking_grupo", "ranking_anio",
                 "desglose_grupo")
_RESPUESTA_NO_FIABLE = ("No he podido montar sobre el grafo una consulta que responda a todo lo que pides "
                        "(no he podido filtrar por: {faltan}), así que prefiero no darte cifras que no respondan "
                        "a la pregunta. Prueba a reformularla o usa el perfil RAG Vectorial.")


# una consulta sin filas, o un recuento que da 0
def _sin_resultado(rows) -> bool:
    if not rows:
        return True
    if len(rows) == 1:
        vals = [str(v) for k, v in rows[0].items() if k != "criterio" and v is not None]
        return all(v in ("0", "", "None") for v in vals)
    return False


def _hay_props_asunto(g, analisis) -> bool:
    grupos = analisis.asunto_grupos or [analisis.asunto_raices]
    if _props_asunto(g, grupos, con_menciones=True):
        return True
    clave = [[_raiz(w) for w in c] for c in getattr(analisis, "asunto_clave", []) if c]
    return bool(clave and _props_asunto(g, clave, con_menciones=True))


# el grafo no tiene el asunto o no se ha podido filtrar todo lo que pide la pregunta
def _no_consta(sparql: str, situacion: str) -> dict:
    return {"sparql": sparql, "rows": [], "answer": f"No consta en el grafo. {situacion}"}


def _responder(pregunta: str, verbose=True, sparql_provider: str = "ollama") -> dict:
    g = _load_graph()
    analisis = analizar(pregunta)
    if analisis.fuera_de_alcance == "importes" and analisis.importe:
        # el grafo guarda el importe del título de algunos puntos (subvenciones,
        # créditos): se busca el del asunto; si no hay, no consta
        _preparar_asunto(g, analisis, pregunta)
        directa = _consulta_directa_ejecutada(g, pregunta, replace(analisis, fuera_de_alcance=None), verbose)
        if not directa or _sin_resultado(directa[1]):
            return {"sparql": directa[0] if directa else "", "rows": [],
                    "answer": _RESPUESTA_FUERA_DE_ALCANCE["importes"]}
        sparql, rows = directa
        filas = _filas_para_narrar(g, rows) + (
            "\nNOTA: estos importes son los que figuran en el TÍTULO de cada punto del Pleno (una subvención, un "
            "crédito, un contrato), no el gasto total del Ayuntamiento en el asunto: el grafo no guarda "
            "presupuestos ni gastos ejecutados. Si la pregunta pide un total o un gasto exacto, di que no consta y da "
            "estos importes como lo que sí aprobó el Pleno, con su fecha.")
        answer = _quitar_bucles(_llm_invoke(ANSWER_PROMPT.format(pregunta=pregunta, filas=filas,
                                                                 sparql=sparql[:NARRAR_MAX_CHARS_SPARQL]), prefer="groq"))
        return {"sparql": sparql, "rows": rows, "answer": answer}
    if analisis.fuera_de_alcance:
        return {"sparql": "", "rows": [], "answer": _RESPUESTA_FUERA_DE_ALCANCE[analisis.fuera_de_alcance]}

    _preparar_asunto(g, analisis, pregunta)
    f_q = forma(_norm_q(pregunta))
    hay_asunto = bool(analisis.asunto_raices)
    faltan = []
    directa = _consulta_directa_ejecutada(g, pregunta, analisis, verbose)
    armg = None if directa else _try_arm_g(pregunta, g, verbose, analisis)
    if directa:
        sparql, rows = directa
        # el grafo no tiene nada del asunto: no consta (solo si el asunto no está
        # en ninguna proposición: si está pero los demás filtros, grupo o años,
        # dejan 0, la respuesta es 0)
        if hay_asunto and _sin_resultado(rows) and f_q in _FORMAS_SIN_ASUNTO and not _hay_props_asunto(g, analisis):
            return _no_consta(sparql, "Ninguna proposición del grafo trata ni menciona el asunto (en su "
                                      "título, sus subtemas o sus entidades).")
    elif armg:
        sparql, rows = armg
    else:
        # un asunto que no está en ninguna proposición del grafo: el LLM solo
        # podría filtrar por otra cosa (en la cuarta evaluación, "noria" ->
        # cualquier título con "Arenal"); se responde que no consta
        if hay_asunto and not _hay_props_asunto(g, analisis):
            return _no_consta("", "Ninguna proposición del grafo trata ni menciona el asunto.")
        sparql, rows, error, faltan = _generar_y_ejecutar(g, pregunta, sparql_provider, verbose, analisis)
        # Segunda opinión: si el modelo local no llega a una consulta válida
        # (error) o completa (le falta algo de la pregunta), se repite el
        # proceso con Gemini y se usa su consulta si esa sí es válida y
        # completa. NO se hace cuando la consulta local es completa pero da 0
        # filas: en la tercera evaluación, "preferir la que devuelve filas"
        # sustituyó respuestas honestas ("no consta") por consultas laxas (OR
        # entre palabras) que sí devolvían filas, y todas eran falsas.
        dudosa = error is not None or bool(faltan)
        if dudosa and sparql_provider != "gemini" and GOOGLE_CLOUD_PROJECT:
            if verbose:
                print("[~] Consulta local inválida o incompleta: segunda opinión con Gemini", flush=True)
            try:
                alt = _generar_y_ejecutar(g, pregunta, "gemini", verbose, analisis)
            except Exception as e:
                print(f"[!] Segunda opinión con Gemini falló — {type(e).__name__}: {e}", flush=True)
                alt = None
            if alt and alt[2] is None and (error is not None or not alt[3]):
                sparql, rows, error, faltan = alt
        if error is not None:
            # la pregunta pide algo que el modelo no sabe expresar sobre este grafo
            if verbose:
                print(f"[!] SPARQL irrecuperable tras 3 intentos: {error}", flush=True)
            return {"sparql": sparql, "rows": [], "answer": _RESPUESTA_IRRECUPERABLE}
        # La consulta del LLM no filtra algo que pide la pregunta: ya no se
        # narran sus filas con un aviso. En las evaluaciones independientes el
        # narrador acababa presentándolas como la respuesta (5 de las 6 falsas
        # de la cuarta venían de aquí).
        if faltan:
            if verbose:
                print(f"[!] Consulta del LLM incompleta ({'; '.join(faltan)}): no se usa", flush=True)
            if hay_asunto:
                return _no_consta(sparql, "No se ha podido montar sobre el grafo una consulta que filtre "
                                          "todo lo que pide la pregunta.")
            return {"sparql": sparql, "rows": [], "answer": _RESPUESTA_NO_FIABLE.format(faltan="; ".join(faltan))}

    rows = _fix_degenerate_groupby(rows, sparql)
    filas = _filas_para_narrar(g, rows)
    filas += _augment_ratios(rows, pregunta)
    if directa:
        if f_q == "contenido" and rows and "resumen" in rows[0]:
            filas += ("\nNOTA: cada fila es una intervención de un orador en un punto: resumen de lo que dice y su "
                      "postura (a_favor, en_contra, abstencion, neutra) ante la propuesta. Organiza la respuesta por "
                      "punto y por grupo, cita la fecha, y no añadas nada que no esté en los resúmenes. Si faltan "
                      "intervenciones de algún grupo, di que no constan.")
        elif f_q == "contenido":
            filas += ("\nAVISO: el grafo no guarda lo que se dijo ni las opiniones de los grupos; estas son "
                      "las proposiciones sobre el asunto (quién las presentó y su resultado). Dilo, descríbelas "
                      "y recomienda el perfil RAG Vectorial para conocer el contenido del debate.")
        elif f_q == "votos":
            filas += ("\nNOTA: una proposición sin grupoVoto no tiene voto por grupo registrado en el grafo "
                      "(solo lo tiene algo más de la mitad); no lo interpretes como que nadie votó así.")
            if any(str(r.get("votacion")) == "enmienda" for r in rows):
                filas += ("\nNOTA: en las filas con votacion = enmienda, el voto registrado es el de una enmienda a "
                          "esa proposición, no el de la proposición: dilo así.")
            if any(r.get("objetoVotacionDecisiva") not in (None, "None") for r in rows):
                filas += ("\nNOTA: sentidoDecisiva es el voto de ese grupo en la votación decisiva del punto "
                          "(objetoVotacionDecisiva: si es una enmienda, no la proposición; si la proposición decae, "
                          "es la votación de la enmienda que la sustituye). Si no hay fila de voto del grupo, "
                          "no consta cómo votó.")
            filas += _votaciones_del_acta(g, rows)
        elif f_q == "ultima" and any(r.get("relacion") == "trata" for r in rows) \
                and any(r.get("relacion") == "menciona" for r in rows):
            # en la octava evaluación se encabezó con una proposición sobre el tranvía que solo
            # nombraba Zabalburu, como si fuera "la última vez que se habló de la plaza"
            filas += ("\nNOTA: hay filas que TRATAN del asunto y filas que solo lo MENCIONAN. Responde primero con "
                      "la fecha de la última proposición que lo trata (es la respuesta a la pregunta) y después, "
                      "aparte, la mención más reciente, diciendo de qué trataba esa proposición.")
        elif f_q == "existe" and not rows:
            filas += "\nNOTA: ninguna proposición del grafo trata ni menciona ese asunto: la respuesta es que no consta."
        if analisis.fecha and not rows:
            filas += _nota_fecha_sin_pleno(g, analisis.fecha)
        # búsqueda relajada: la consulta no pudo exigir todas las palabras del asunto. En la octava
        # evaluación, "¿se ha mencionado la pintura de pasos de cebra en tres dimensiones?" se
        # respondió "sí" con una proposición de pasos de cebra, pese a ver este mismo criterio
        relajado = next((str(r["criterio"]) for r in rows
                         if str(r.get("criterio", "")).startswith("ninguna proposición contiene a la vez")), "")
        if relajado:
            filas += (f"\nAVISO: {relajado}. Las filas NO cumplen todo lo que pide la pregunta. Empieza diciendo, "
                      "sin rodeos, que no consta nada que reúna todo el asunto (nombra las palabras que no aparecen) "
                      "y presenta después las filas aparte, como lo más cercano que hay, sin decir que traten de lo "
                      "que se pregunta ni responder 'sí' a la pregunta original.")
    if verbose:
        print(f"[FILAS] {len(rows)}")

    sparql_narrar = sparql if len(sparql) <= NARRAR_MAX_CHARS_SPARQL else sparql[:NARRAR_MAX_CHARS_SPARQL] + " ..."
    answer = _llm_invoke(ANSWER_PROMPT.format(pregunta=pregunta, filas=filas, sparql=sparql_narrar), prefer="groq")
    answer = _quitar_bucles(answer)

    return {"sparql": sparql, "rows": rows, "answer": answer}


# Punto de entrada: {"sparql", "rows", "answer"}, con los enlaces a las actas ya en la respuesta
def graph_answer(pregunta: str, verbose=True, sparql_provider: str = "ollama") -> dict:
    res = _responder(pregunta, verbose, sparql_provider)
    res["answer"] += fuentes_graphrag(res["rows"])
    return res


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "¿Cuántas proposiciones sobre vivienda ha presentado cada grupo?"
    res = graph_answer(q)
    print("\n=== RESPUESTA ===\n" + res["answer"])
