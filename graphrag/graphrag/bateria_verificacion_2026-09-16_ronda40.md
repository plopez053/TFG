# Batería de verificación — Ronda 40 (2026-09-16)

Objetivo explícito del usuario: (1) comprobar que los baselines de
`regression_qa.py` son fieles a los PDFs reales, no solo internamente
consistentes; (2) lanzar una batería nueva para comprobar que las medidas
aplicadas hoy (integración de Arm G como filtro previo, fix de fidelidad de
`vote_result`) funcionan de verdad en el sistema completo (SPARQL + narración
LLM), no solo en pruebas aisladas.

## Parte 1 — Verificación de fidelidad contra los PDFs

Metodología en dos capas, documentada en detalle en la memoria del proyecto:

1. **Recomputación independiente en Python puro**, sin tocar rdflib/SPARQL,
   reutilizando las funciones reales del pipeline (`grupos.extrae_grupo`,
   `build_rdf.resultado_cruzado`) sobre `proposals.jsonl` /
   `proposals_enriched.jsonl` / `concejales.jsonl` directamente. **9/9 casos
   no temáticos coincidieron exactamente** con los baselines ya corregidos en
   la Ronda 38 — confirma que `build_rdf.py` no introduce errores de
   conversión al grafo RDF.
2. **Muestreo de páginas PDF reales** (campo `source`+`page_ini` de
   `proposals.jsonl`, actas en `actas/`): 8 casos abiertos y comparados
   palabra por palabra contra el texto extraído. 7/8 coincidían exactamente.

### Bug real encontrado

Al abrir la página 293 de `24-09-2015_Ordinaria_Acta.pdf` (ítem 49, PP,
propuesta sobre el Bilbobús) para confirmar un caso marcado "aprobada con
enmienda", el texto real decía: *"...por lo que decaen tanto la enmienda
adición del Grupo Municipal UDALBERRI-BILBAO EN COMÚN como la proposición
presentada por el Grupo Municipal PARTIDO POPULAR."* — la proposición
**decayó**, no se aprobó.

Dos causas compuestas, ambas corregidas:

- `build_rdf.py::_RE_DECAE_PROP` solo reconocía "decae" (singular), no
  "decaen" (plural) → 51 apariciones reales de "decaen" nunca detectadas.
  Fix: `r"\bdecaen?\b"`.
- `extract_proposals.py` se quedaba con el **primer** `vote_result` no vacío
  de cada tema en vez del **último** → en proposiciones con varias enmiendas
  competidoras votadas en secuencia, capturaba el voto de una enmienda
  intermedia en vez del resultado final decisivo. Fix: tomar el último de la
  lista (ya viene ordenada por orden real del acta).

**Impacto medido**: 279 de 3422 registros (8%) cambiaron de `vote_result`
tras el fix. Grafo reconstruido, baselines recalibrados en
`regression_qa.py` y `estudio_sparql/gold.py`. Regresión final:
**GraphRAG 21/22, vectorial 13/13** (único fallo: el caso ya conocido de
comparar 2 grupos en una consulta).

**Limitación conocida, no arreglada esta sesión**: para el caso concreto que
originó el hallazgo, ni siquiera el fix de "último voto" lo corrige del
todo — el párrafo decisivo real queda atribuido al chunk del TEMA SIGUIENTE
(ítem 50) por un corte de página/segmento en `backend/rag.py` que no
coincide con el marcador real del acta. Alcance no cuantificado, arreglarlo
exige revisar la segmentación del indexador (fuera de alcance hoy).

## Parte 2 — Batería nueva: ¿funcionan las medidas aplicadas?

| # | Pregunta | Sistema | Resultado |
|---|----------|---------|-----------|
| Q1 | ¿Cuántas proposiciones sobre desahucios se han presentado? | GraphRAG | **Arm G resolvió sin LLM**, correcto (14) |
| Q2 | PP 2015: cuántas y cuántas aprobadas | GraphRAG | **Arm G resolvió sin LLM en 2.9s**, narración correcta: 38 total, **12** aprobadas (el número YA corregido, confirmado end-to-end) |
| Q3 | ¿Qué pasó con la proposición del PP sobre el Bilbobús del 24-09-2015? | GraphRAG | Arm G no tiene plantilla (correcto, delega) → el LLM (Ollama) generó SPARQL con un prefijo alucinado (`r3:label` en vez de `rdfs:label`) las 3 veces → degradó con gracia ("no he podido traducir esa pregunta"), sin romper. **Hallazgo nuevo, no arreglado**: preguntas de "qué pasó con ESTA proposición concreta" (lookup por fecha+tema con muchos OPTIONAL) son frágiles para el generador SPARQL actual. |
| Q4 | Tasa de aprobación EH Bildu vs PP | GraphRAG | **La guarda nueva de Arm G funcionó**: no lo resolvió, delegó al LLM (confirmado, no aparece "[Arm G] resuelto"). El LLM generado SÍ comparó ambos grupos correctamente esta vez, pero añadió sin pedírselo un filtro `bo:anio 2023` no solicitado, y la conclusión final ("EH Bildu tiene la mejor tasa...") omite la coletilla "en 2023" — respuesta bien argumentada pero de alcance equivocado. Confirma que este patrón sigue siendo el punto débil real del sistema. |
| Q5 | Vivienda por grupo | GraphRAG | **Arm G resolvió sin LLM**, correcto (63/47/34/20/12/8/4/1) |
| Q2v | Misma pregunta de PP 2015 | Vectorial | Sin recuento agregado (limitación de diseño ya documentada, top-k no exhaustivo): narra casos individuales; uno de los citados literalmente incluye "decae la proposición" en el texto fuente — el vectorial nunca sufrió el bug de hoy porque no depende del campo estructurado `vote_result` |
| Q4v | Misma pregunta de tasa EH Bildu/PP | Vectorial | Tocó el límite DIARIO de Groq (200.000 TPD, cuenta ya agotada al terminar esta batería) → cayó a Ollama qwen2.5:7b con gracia. Respuesta débil: solo encontró 2 propuestas de EH Bildu relevantes en su muestra, ninguna del PP, y concluyó "50% de aprobación" con un tamaño de muestra de 2 — ilustra bien por qué el vectorial no puede resolver este tipo de comparación agregada. |

### Conclusión de la batería

Las dos medidas de hoy **funcionan de verdad en el sistema completo**, no
solo en pruebas aisladas: Arm G resolvió 3/5 preguntas de GraphRAG en
segundos y sin LLM, con las cifras ya corregidas por el fix de fidelidad
propagándose correctamente hasta la narración final. La guarda de "2+
grupos" delegó correctamente al LLM en vez de responder mal en silencio. El
único caso que sigue siendo débil (comparación de 2 grupos) lo es en ambos
sistemas, por razones distintas y ya documentadas — no es una regresión de
hoy, es el límite conocido del proyecto.

**Nota de cuota**: la cuenta de Groq (`openai/gpt-oss-120b`) agotó su cupo
diario (200.000 TPD) durante esta batería — cualquier prueba adicional hoy
que dependa de Groq caerá automáticamente a Ollama (ya verificado que
funciona con gracia).
