# Plan — capa determinista de votos y resultado

> **✅ IMPLEMENTADO 2026-09-10.** `votos_parse.py` nuevo; `extract_proposals.py`,
> `build_graph.py`, `build_rdf.py`, `ontology.ttl`, `graph_rag_sparql.py` tocados.
> Cobertura de voto FIEL: **40% (1.387 props)** vs 9% del LLM. 34.178 votos
> nominales. `bo:votoFuente "acta"|"llm"` marca la procedencia. Detalle en
> `code_review_findings.txt` §RONDA 30. Pendiente: re-correr `extract_proposals.py`
> en el rebuild de producción (735 props con texto truncado a 30k podrían haber
> perdido su lista de votos; el backfill se hizo sobre el texto ya recortado).


## Objetivo

Separar el grafo en dos capas según la **procedencia** de cada dato:

- **Capa 1 (hechos literales del acta)** → parser/regex, ~99% fiel, sin LLM.
- **Capa 2 (interpretación)** → LLM, marcada como derivada.

Este plan implementa el mayor salto disponible de la Capa 1: **el voto,
nominal y por grupo, parseado directamente de las listas del acta** en vez de
adivinado por el LLM.

### Por qué merece la pena (medido sobre `proposals.jsonl`, 3.422 props)

| | Hoy (LLM `votos_por_grupo`) | Con parser determinista |
|---|---|---|
| Cobertura voto por grupo | **9%** (305 props en el grafo) | **~42%** (1.439 props con lista nominal) |
| Fiabilidad | el LLM infiere del texto | copia literal de "Votos afirmativos: N señoras/señores: …" |
| Voto nominal (por concejal) | **no existe** | **nuevo** — ~1.400 props × ~20 personas |

Formato real en el acta (siempre listas de apellidos, nunca "votan los grupos X e Y"):
```
Votos emitidos: 29
Votos afirmativos: 14 señoras/señores: Madrazo, Sustatxa, Alcalde, Areso, Sabas,
  Sánchez, De Castro, Barkala, Maíz, Alonso, Anuzita, Urtasun, Ajuria y Abaunza.
Votos negativos: 6 señoras/señores: Oleaga, Gil Invernon, Díez, Zurro, Gardiazabal y Delgado.
Abstenciones: 7 señoras/señores: Basagoiti, Ruiz, Marcos, García, Hermosa, Rodrigo y Pontes.
```

---

## Qué se añade al grafo

**Ontología (`ontology.ttl`):**
```turtle
bo:votoAFavor     a owl:ObjectProperty ; rdfs:domain bo:Proposicion ; rdfs:range bo:Concejal .
bo:votoEnContra   a owl:ObjectProperty ; rdfs:domain bo:Proposicion ; rdfs:range bo:Concejal .
bo:votoAbstencion a owl:ObjectProperty ; rdfs:domain bo:Proposicion ; rdfs:range bo:Concejal .
bo:votoFuente     a owl:DatatypeProperty ; rdfs:range xsd:string .   # "acta" | "llm"
bo:resultadoFuente a owl:DatatypeProperty ; rdfs:range xsd:string .
```
`bo:votoAFavorDe` / `bo:votoEnContraDe` / `bo:seAbstuvo` (nivel GRUPO) ya existen —
se seguirán usando, pero **derivados de la agregación de los votos nominales**.

**Triples nuevos por proposición con lista nominal:**
- `br:prop_X bo:votoAFavor br:concejal_madrazo_lavin` … (uno por persona)
- `br:prop_X bo:votoAFavorDe br:grupo_eaj_pnv` (derivado: ≥1 miembro del grupo votó a favor;
  si el grupo se dividió, se emiten los dos y `bo:votoFuente "acta"`)
- `br:prop_X bo:votoFuente "acta"`

**Capacidades nuevas de consulta:**
- "¿cómo votó [concejal] la proposición [Y]?"
- "¿qué concejales votaron en contra de los presupuestos de 2023?"
- "¿algún concejal rompió la disciplina de voto de su grupo?" (miembro que vota distinto al resto)
- voto por grupo ahora en 42% en vez de 9%, y fiel.

---

## Cambios por fichero

### 1. `extract_proposals.py` — extracción de las listas nominales (Fase 1)

Nuevo helper `_parse_listas_voto(texto) -> dict | None`:

```python
_SENT_VOTO = {
    "favor":      r"[Vv]otos?\s+(?:afirmativos?|a\s+favor|positivos?)",
    "contra":     r"[Vv]otos?\s+(?:negativos?|en\s+contra)",
    "abstencion": r"[Aa]bsten(?:ciones|tzioak)",
}
# tras la etiqueta: "N señoras/señores:" (o "jaun-andre:") y luego la lista
# hasta el "." final. Separadores: "," y " y " (última). OCR mete espacios:
# "Ma drazo", "Gil Invernon" -> normalizar como en extract_concejales._limpia_nombre.
```

Reglas del parser:
- Cada bloque `<etiqueta> <n> (señoras/señores|jaun-andre): <lista>.`
- `<lista>` se corta en el primer `.` que va seguido de espacio+mayúscula o salto,
  o en la siguiente etiqueta de voto.
- Trocear por `,` y por ` y ` (solo el último); quitar espacios espurios de OCR
  (regex de `extract_concejales._limpia_nombre`), quitar tildes NO (se resuelven
  con acento en `resolver_concejal`).
- Conservar "Alcalde"/"Presidente"/"Presidenta" tal cual (los resuelve `resolver_concejal`).
- Validación: `len(lista) == n` declarado → si no cuadra, marcar `"_dudoso": True`
  (se emite igual pero con `bo:votoFuente "acta?"`).
- Nombres de 2 palabras ("Gil Invernon", "De Castro", "Bilbao Urquijo"): la lista
  los trae juntos, se pasan enteros a `resolver_concejal`.

Nuevo campo en cada fila de `proposals.jsonl`:
```json
"votos_nominales": {
  "favor": ["Madrazo", "Sustatxa", "Alcalde", ...],
  "contra": ["Oleaga", "Gil Invernon", ...],
  "abstencion": [...],
  "_conteo_declarado": {"favor": 14, "contra": 6, "abstencion": 7},
  "_cuadra": true
}
```
Si no hay ninguna lista → `"votos_nominales": null`.

**OJO truncado:** la lista de votos va al FINAL de la proposición. Con `MAX_TEXT=30000`
alguna proposición larga podría cortarla. Mitigación: buscar las listas en el
conjunto COMPLETO de chunks del topic (`"".join(all_chunk_texts)`), no en el
`text[:MAX_TEXT]` ya recortado. (Cambio pequeño en `extract_proposals`.)

### 2. `build_rdf.py` — resolución y agregación (Fase 3)

En `build()`, para cada `r` (proposición enriquecida — hay que arrastrar
`votos_nominales` desde `proposals.jsonl` a `proposals_enriched.jsonl` en
`build_graph.py::enrich`, campo passthrough como ya se hace con `oradores`):

```python
vn = r.get("votos_nominales")
if vn:
    votos_grupo = {"favor": set(), "contra": set(), "abstencion": set()}
    for sentido, apellidos in (("favor", vn["favor"]), ("contra", vn["contra"]),
                               ("abstencion", vn["abstencion"])):
        for ape in apellidos:
            c_uri = resolver_concejal(conc_idx, alcaldes, ape, r.get("date"))
            if not c_uri:
                continue
            g.add((pr, VOTO_NOM_PRED[sentido], c_uri))          # voto nominal
            grupo = grupo_de_concejal(c_uri)                     # del censo
            if grupo:
                votos_grupo[sentido].add(grupo)
    for sentido, grupos in votos_grupo.items():
        for gu in grupos:
            g.add((pr, VOTO_GRP_PRED[sentido], gu))
    g.add((pr, BO.votoFuente, Literal("acta" if vn.get("_cuadra") else "acta?")))
else:
    # fallback: usar votos_por_grupo del LLM (como hoy) con bo:votoFuente "llm"
    ...
```

`grupo_de_concejal(uri)`: dict `br:concejal_X -> br:grupo_Y` construido en
`cargar_concejales` (ya tenemos el `grupo` de cada uno en `concejales.jsonl`).

### 3. `resultado` determinista (rápido, hazlo también)

En `build_rdf.py`, antes de usar `r["resultado"]` del LLM, intentar parsearlo del
string `r["vote_result"]` (que ya trae la frase de resultado + tally):

```python
def resultado_det(vote_text):
    t = (vote_text or "").lower()
    if "decae" in t or "por lo que decae" in t:            return "decae"
    if "queda rechazad" in t or "es rechazad" in t:        return "rechazada"
    if "con enmienda" in t or "aprobada con la incorporación": return "aprobada con enmienda"
    if "queda aprobad" in t or "es aprobad" in t or "por unanimidad" in t: return "aprobada"
    if "se retira" in t or "queda retirad" in t:            return "retirada"
    return None
```
Precedencia: `resultado_det(...) or r["resultado_llm"] or "sin resultado"`, y
`bo:resultadoFuente "acta"|"llm"`.

---

## Verificación

1. **Cobertura:** contar props con `bo:votoFuente "acta"` (esperado ~1.400) vs `"llm"`.
2. **Muestra manual de 25:** abrir el PDF de 25 proposiciones al azar con lista
   nominal, comprobar que los grupos a favor/contra/abstención del grafo coinciden.
3. **Resolución:** % de apellidos de las listas que `resolver_concejal` resuelve
   (esperado >90%; los no resueltos suelen ser OCR muy roto o concejales fuera del
   censo — el mandato 2007-2011 tiene 8 sin partido).
4. **Regresión:** `regression_qa.py` — los conteos por grupo pueden moverse un poco
   (ahora hay más voto por grupo); actualizar baselines. Añadir 2 casos:
   "¿cómo votó EH Bildu las proposiciones de vivienda?" y "¿qué concejal ha votado
   más veces en contra?".
5. **Contraste con el LLM:** en las props que tienen las DOS fuentes, ¿cuántas veces
   discrepan LLM vs acta? (dato para la memoria: cuantifica el error del enriquecimiento).

---

## Esfuerzo estimado

| Tarea | Tiempo |
|---|---|
| `_parse_listas_voto` en extract_proposals + buscar en chunks completos | 2 h |
| passthrough `votos_nominales` en build_graph.enrich | 15 min |
| resolución + agregación + triples en build_rdf | 1,5 h |
| `resultado_det` | 45 min |
| ontología + SCHEMA de graph_rag_sparql (ejemplos de voto nominal) | 45 min |
| regenerar `proposals.jsonl` (Fase 1, ~15 min) + build_rdf + verificación | 1,5 h |
| **Total** | **~7 h** |

**No hace falta re-enriquecer con Gemini** (los `votos_nominales` salen de Fase 1,
que es rápida y gratis; el `proposals_enriched.jsonl` solo necesita el campo
passthrough — se puede reconstruir con un script de merge sin volver a llamar al LLM).

---

## Riesgos

- **Desambiguación de apellido suelto:** "Gil" (¿Gil Invernón, Gil Llanos, Alfonso Gil?),
  "Rodrigo" (Ángel vs Gabriel). Para el voto por GRUPO casi siempre da igual (mismo
  partido); para el voto NOMINAL, `resolver_concejal` coge el primero activo en esa
  fecha → puede acertar el grupo y fallar la persona. Mitigación: si hay varios
  candidatos del MISMO grupo, emitir solo `bo:votoAFavorDe` (grupo) y NO el nominal.
- **Listas en euskera** ("jaun-andre") en actas recientes bilingües — el regex ya
  las cubre pero verificar con una de 2024-2026.
- **"Alcalde" en la lista** cuando el alcalde no vota (preside) — en la práctica sí
  aparece votando; `resolver_concejal` ya lo mapea por fecha.
- **Unanimidad sin lista:** "aprobada por unanimidad de los 29 miembros" → NO hay
  lista pero SÍ se puede derivar "todos los grupos a favor". Opcional (fase 2 del plan):
  si `resultado == aprobada` y `"unanimidad"` en el texto y no hay lista → emitir
  `bo:votoAFavorDe` para todos los grupos con concejales activos en esa fecha,
  `bo:votoFuente "acta-unanimidad"`.
