# Batería de verificación GraphRAG vs RAG vectorial — 2026-09-13

_Preguntas centradas en los arreglos de esta sesión: atribución de grupo (Equipo de Gobierno, ausencia de Ciudadanos), presentadaPorParticular (asociaciones/particulares), deduplicación de entidades (Bilbao Kirolak), y censo de concejales homónimos (Gil)._

_GraphRAG: LLM de generación Ollama qwen3:8b (fallback a Groq openai/gpt-oss-120b si falla/tarda); narrador Groq. RAG vectorial: recuperación híbrida + rerank Cohere sobre chroma_db ya existente; narrador Groq/Ollama._

---

## 1. ¿Cuántas proposiciones ha presentado el Equipo de Gobierno y cuántas el grupo Ciudadanos?

### GraphRAG

```sparql
SELECT ?g ?nombreGrupo (COUNT(DISTINCT ?p) AS ?count)
WHERE {
  ?p a bo:Proposicion .
  ?p bo:presentadaPor ?g .
  ?g a bo:Grupo .
  ?g rdfs:label ?nombreGrupo .
  FILTER(?g = br:grupo_equipo_de_gobierno || ?g = br:grupo_ciudadanos)
}
GROUP BY ?g ?nombreGrupo
```

**Filas (1):** solo Equipo de Gobierno; Ciudadanos no aparece (0 filas).

**Respuesta:**

> Según los datos del grafo, el **Equipo de Gobierno** ha presentado **769 proposiciones**.
>
> En cuanto al grupo **Ciudadanos**, el conjunto de resultados no contiene ninguna fila que lo incluya; por lo tanto, no se dispone de datos sobre cuántas proposiciones ha presentado ese grupo.

**Verificación:** 769 coincide con el recuento directo sobre el grafo. Ciudadanos en 0 confirma que el bug de "ciudadanos"/"Ciudadanos" quedó resuelto (antes 7 falsos positivos por confundir la palabra genérica con el partido).

### RAG vectorial

**Respuesta:**

> - Proposiciones presentadas por el **Equipo de Gobierno**: **0**.
> - Proposiciones presentadas por el **grupo Ciudadanos**: **0**.
>
> **Conclusión:** En el conjunto de actas proporcionado no aparece ninguna proposición formalmente presentada ni por el Equipo de Gobierno ni por el grupo Ciudadanos; únicamente se registran enmiendas o decisiones sobre propuestas de otros grupos. Por lo tanto, el recuento de proposiciones de ambos actores es nulo.
>
> ============================================================
> FUENTES UTILIZADAS: 6 fragmentos, todos proposiciones del PARTIDO POPULAR con enmienda del EQUIPO DE GOBIERNO (p.ej. 28-03-2019, 28-01-2016, 31-05-2012) — ninguno es una proposición *presentada* por el Equipo de Gobierno o Ciudadanos.

**Nota de comparación:** ambos sistemas coinciden en que Ciudadanos = 0. Para Equipo de Gobierno discrepan (769 vs 0) porque son preguntas distintas de facto: GraphRAG cuenta *todas* las proposiciones de la Junta de Gobierno elevadas al Pleno (presupuestos, resoluciones, etc. — la mayoría del corpus real de "Equipo de Gobierno"), mientras que el vectorial solo recuperó, con esta pregunta concreta, fragmentos de proposiciones de partidos con *enmienda* del Equipo de Gobierno (no encontró ninguna con el Equipo de Gobierno como *presentador* directo en su muestra de 6 fragmentos). Es la limitación esperada del vectorial (retrieval por similitud semántica, no recuento exhaustivo), no una discrepancia de datos.

---

## 2. ¿Qué proposiciones ha presentado alguna asociación vecinal, AMPA o particular (no un grupo político)?

### GraphRAG

**Bug real encontrado durante esta batería:** el esquema pasado al LLM generador de SPARQL (`graph_rag_sparql.py::SCHEMA`) no mencionaba la propiedad nueva `bo:presentadaPorParticular`. Primer intento: el LLM generó un filtro sobre `bo:presentadaPor NOT IN (grupos conocidos)` sin excluir `br:grupo_desconocido`, dando un resultado inflado y sin sentido (239 filas, campos vacíos). **Arreglado**: añadida la propiedad al esquema y un ejemplo few-shot.

**Consulta tras el arreglo:**

```sparql
SELECT ?p ?nombre WHERE {
  ?p a bo:Proposicion ; bo:presentadaPorParticular ?ent .
  ?ent rdfs:label ?nombre . } LIMIT 50
```

**Filas (50).**

**Respuesta:**

> Resumen de las proposiciones presentadas por asociaciones vecinales, AMPA o particulares (es decir, por agentes que no son grupos políticos):
>
> | Proposición | Presentada por |
> |---|---|
> | prop_0304e74ca2f384ad | Asociación BIEL Kontsumitzaileen Bizitza Elkartea de Bilbao |
> | prop_09606b78d6009788 | Jaime Puente Romero (particular) |
> | prop_0ccf1e36b2d9fe5a | Aitziber Sagarminaga Abad (particular) |
> | prop_1af5aa6809540cc7 | Comunidad de Propietarios de Santa Monica 2 y 4 (particular) |
> | prop_50bff21bc2ae6979 | Asociación Errondagane Guraso Elkartea del BHI Karmelo-Solokoetxe |
> | prop_63ec16522ab06216 | Asociación Europeista Ambiental AERA |
> | prop_8f723d5b2ae83140 | Eneko Gerrikabeitia (particular) |
> | ... | (50 filas en total, límite de la consulta; nombres reales, ninguno vacío) |

### RAG vectorial

**Respuesta:**

> No se ha encontrado en los fragmentos de actas proporcionados ninguna proposición presentada por una asociación vecinal, una AMPA o un particular; todas las proposiciones citadas corresponden a iniciativas del Grupo Político Municipal Partido Popular.
>
> **Conclusión:** En el conjunto de actas analizado, las únicas propuestas registradas provienen de un partido político, sin que aparezca ninguna iniciativa formalmente presentada por asociaciones de vecinos, AMPA u otros particulares.
>
> ============================================================
> FUENTES UTILIZADAS: 25 fragmentos recuperados, casi todos "PROPOSICIÓN del grupo político municipal PARTIDO POPULAR..." — ninguno de una asociación/particular real.

**Nota de comparación:** discrepancia fuerte y reveladora. GraphRAG encuentra 50 casos reales (verificados uno a uno contra el PDF en esta misma sesión); el vectorial, con esta formulación de pregunta, no recuperó ninguno de esos 50 documentos en su top-k — probablemente porque "proposición vecinal presentada por Fulano/Asociación X" no es semánticamente muy parecido a la pregunta genérica "asociación vecinal, AMPA o particular", y el recuperador prioriza los muchos cientos de "Proposición del grupo político municipal PARTIDO POPULAR" que sí contienen las palabras de la pregunta de forma más literal. Es un caso concreto y real donde GraphRAG (dato estructurado) encuentra información que el vectorial (búsqueda semántica) no recupera con esa formulación.

---

## 3. ¿Cuántas veces se menciona a Bilbao Kirolak en las proposiciones?

### GraphRAG

Dos variantes, diferencia esperada (no contradicción):

- **Coincidencia exacta** (`rdfs:label "Bilbao Kirolak"`): **76** — solo el nodo canónico `ent_bilbao_kirolak` tras la deduplicación de esta sesión.
- **Coincidencia por subcadena** (regex case-insensitive): **79** — incluye además los nodos "Bilbao Kirolak - Instituto Municipal de Deportes" y variantes con "S.A." completo, dejados deliberadamente como entidades distintas (nombre institucional más largo, no un simple sufijo jurídico).

**Respuesta (variante B):**

> En el conjunto de proposiciones del Ayuntamiento de Bilbao, el término "Bilbao Kirolak" aparece **79 veces**.

**Verificación:** coincide con el recuento directo sobre el grafo deduplicado (antes fragmentado en 4 nodos: "BILBAO KIROLAK", "Bilbao Kirolak", "Bilbao Kirolak S.A.", "Bilbao Kirolak SA").

### RAG vectorial

**Respuesta:**

> En los cuatro plenos analizados (26-05-2016, 31-01-2019, 31-03-2011, 29-06-2017), el nombre **Bilbao Kirolak** aparece un total de **13 veces** dentro de las proposiciones recuperadas, en contextos de conciliación, deporte, seguridad en piscinas y oferta de actividades adaptadas.
>
> ============================================================
> FUENTES UTILIZADAS: 23 fragmentos recuperados, de fechas muy dispersas (2007-2025), la mayoría con "Bilbao Kirolak" mencionado en el cuerpo de la proposición.

**Nota de comparación:** la diferencia (79 vs 13) es la esperada entre los dos sistemas: GraphRAG cuenta *todas* las proposiciones del grafo completo que mencionan la entidad (recuento exhaustivo); el vectorial narra solo sobre los 4 plenos "más relevantes" según similitud semántica de su primera pasada de análisis narrativo, aunque recuperó 23 fragmentos en total. No es una discrepancia de datos, es la diferencia entre "contar todo" (grafo) y "resumir lo más relevante de una muestra" (vectorial).

---

## 4. ¿Qué concejales apellidados Gil ha habido en el Ayuntamiento de Bilbao?

### GraphRAG

```sparql
SELECT ?n WHERE {
  ?c a bo:Concejal ;
     rdfs:label ?n .
  FILTER(REGEX(STR(?n), "G[ií]l", "i"))
}
```

**Filas (2).**

**Respuesta:**

> Según los datos del grafo, en el Ayuntamiento de Bilbao han existido dos concejales cuyo apellido incluye "Gil":
> - Begoña Gil Llanos
> - Alfonso Gil Invernón

**Nota:** esta consulta lee el censo directamente (`bo:Concejal`), sin ambigüedad. El fix de esta sesión (omitir en vez de adivinar cuando dos concejales activos comparten apellido) afecta solo a la resolución de menciones sueltas tipo "SR./SRA. GIL:" dentro del texto de un debate, no al censo en sí.

### RAG vectorial

**Respuesta:**

> En los fragmentos de actas que se proporcionan solo aparece un concejal con el apellido **Gil**:
> - **Alfonso Gil** – Concejal de Movilidad y Sostenibilidad (referido como "SR. GIL" en el pleno del 25-01-2018, "concejal de Movilidad y Sostenibilidad, Alfonso Gil" en el 22-02-2018, y mencionado también en 28-06-2018 y 14-09-2021).
>
> ============================================================
> FUENTES UTILIZADAS: 23 fragmentos. **Nota:** la fuente [9] de su propia lista cita literalmente "...la Concejala del Grupo Municipal Socialistas Vascos, doña Begoña Gil Llanos..." — el sistema recuperó ese fragmento pero no lo usó al narrar la respuesta, así que se le "escapó" la segunda Gil que sí tenía delante.

**Nota de comparación:** GraphRAG encuentra a los dos (censo estructurado, sin depender de que el LLM narrador use bien cada fragmento recuperado); el vectorial solo narra uno, aunque tenía la prueba de la segunda en sus propias fuentes citadas. Ejemplo claro de una ventaja estructural del grafo sobre la narración libre para este tipo de pregunta ("listar todos los X").
