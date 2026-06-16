# Plan de reconstrucción / actualización de la base de datos

> Documento de traspaso. Pégalo como contexto a una nueva sesión antes de reindexar.
> **Fecha de este plan: 2026-06-16.**

## 0. Estado actual (qué YA está en la BD, no rehacer)

- **`page` (nº de página)** YA es metadato de cada chunk (verificado contra el PDF). El frontend lo usa.
- **`vote_result`** YA existe como metadato (rebuild anterior), PERO con dos defectos de calidad
  y un fallo de cobertura que ESTE rebuild corrige (ver §1).
- Metadata por chunk: `chunk_index, date, page, party, source, speaker, topic, vote_result`.
- LLM: Groq `llama-3.3-70b-versatile` (`GROQ_API_KEY` en `.env`). Embeddings: `nomic-embed-text` (Ollama).
- `DATA_PATH` autodetecta la carpeta de actas (dentro del proyecto o fuera). Arranque:
  `source venv/bin/activate && chainlit run frontend/app.py --port 8000`.

## 1. Objetivo de ESTE rebuild — mejorar `vote_result` (3 arreglos, ya en el código)

Todo en `backend/rag.py`, método `_process_single_pdf`. Ya implementado y validado; el rebuild
solo hace falta para que la BD recoja los textos/cifras corregidos.

1. **Cobertura: ventana de la regex de votos `vote_re` 80 → 500.**
   Las actas BILINGÜES (euskera+castellano) intercalan entre "Votos emitidos" y "Votos afirmativos"
   el bloque en euskera (`Baiezko botoak: N jaun/andre: [nombres]`) + lista de concejales (>80 chars).
   Con 80 se perdían TODOS los votos de muchos plenos. Caso real: **27-10-2022 pasó de 0 → 795 chunks
   con voto** (1 → 26 votaciones detectadas). Afectaba a ~67 fechas que estaban a cero.

2. **Votaciones unánimes: `if favor and contra:` → `if favor:`.**
   Las votaciones sin "en contra" (unánimes) se descartaban (vote_result quedaba None). Ahora se
   guardan con las cifras que haya (`a favor` obligatorio; `en contra`/`abstenciones` opcionales).

3. **Texto del resultado limpio: `result_re` `.{0,200}` → `[^.]{0,400}` + recorte de narración.**
   Antes el texto del resultado (a) se cortaba a media palabra a los 200 chars (`...Grupo ELKA`) y
   (b) se tragaba narración posterior (`...EH BILDU. - Siendo las 14:05 horas... receso para comer`).
   Ahora para en el PUNTO que cierra la frase y el `re.split` corta también en ` - ` y ` Siendo las`.

4. **Resultados SIN cifras (unánimes / por asentimiento) — `unanim_re` (PENDIENTE de reindexar).**
   Antes solo se guardaba el voto si había cifras numéricas → se perdían TODOS los acuerdos
   "El Pleno Municipal, por unanimidad de miembros presentes, acuerda..." / "Aprobar por unanimidad"
   / "por asentimiento" (179 en solo 25 actas antiguas). Ahora, si no hay cifras, se intenta capturar
   ese resultado unánime/por asentimiento. Validado: 27-11-2008 pasó de 0 → 3; 27-10-2022 de 795 → 854;
   12-11-2007 (con cifras) intacto. **Este arreglo NO está aún en la BD: requiere un nuevo rebuild.**

5. **Segmentación del FORMATO ANTIGUO (2007-2009) — topics limpios (PENDIENTE de reindexar).**
   El corpus empieza en 2007 (carpetas 2002-2006 vacías). En esas actas el marcador "- N -" se usa
   tanto para PUNTOS del orden del día ("- 21 - Proposición que presenta...") como para NÚMEROS DE
   PÁGINA. El código partía por TODOS → topics basura (fragmentos de discurso). Ahora, en formato
   antiguo, se intenta primero partir SOLO por "- N - <TIPO>" (Proposición/Propuesta/Preguntas/Se da
   cuenta/Dictamen/Moción/Aprobar...) → topics tipo "21. Proposición que presenta...". Si no hay tales
   marcadores (Extraordinarias de presupuestos: un solo punto con varias votaciones) se mantiene el
   troceo por página previo (sin regresión). Validado en 40 actas 2007-2010: 0 errores, TODAS las
   ordinarias antiguas con topics limpios; las dudosas restantes son solo Extraordinarias (esperado).

## 2. Cómo lanzar el rebuild (usar `scripts/full_rebuild.py`, NO `rag.py`)

1. **Parar Chainlit** si está abierto (tiene la BD abierta; el rebuild completo borra `chroma_db/`).
2. **Ollama** corriendo con `nomic-embed-text`.
3. **Validar primero un año** (rápido, ~minutos): purga e reindexa solo ese año.
   ```
   python scripts/full_rebuild.py --year 2022
   ```
4. **Rebuild COMPLETO** (borra `chroma_db/` y reindexa todas las actas; en el portátil ~varias horas):
   ```
   python scripts/full_rebuild.py
   ```

## 3. Validación (no saltársela)

Caso de cobertura (el que motivó el rebuild):
```python
import sqlite3
con=sqlite3.connect('chroma_db/chroma.sqlite3'); cur=con.cursor()
cur.execute("SELECT id,key,string_value FROM embedding_metadata WHERE key IN ('date','vote_result')")
# 27-10-2022 debe tener MUCHOS chunks con vote_result (antes 0).
```
Casos de texto limpio (en el navegador, pleno 30-09-2021):
- Resultado NO truncado: debe verse `...Grupo ELKARREKIN BILBAO-PODEMOS/EZKER ANITZA-IU/EQUO BERDEAK`
  completo (no `...Grupo ELKA`).
- Resultado SIN narración: no debe aparecer `Siendo las 14:05 horas... receso para comer`.

Caso histórico clave (sigue válido): la proposición de las **176 viviendas (26-10-2010,
Socialistas Vascos)** debe mostrar SU voto real, no el del tema anterior.

## 4. Cambios de frontend YA hechos (no dependen del rebuild, ya en vivo)

`frontend/app.py`, `build_sources_data(retrieved_docs, answer_text)`:
- Agrupa por **(fecha, topic)** = una fuente por DEBATE (antes por acta → rango de páginas absurdo).
- Excluye la sección `General / Introducción` como fuente.
- **Filtro de relevancia** solo si todos los debates son de la MISMA fecha: conserva los que la
  respuesta trata (≥4 palabras de asunto compartidas, restando `_STOP_PROCEDIMENTAL`).

## 5. Opcional (decidir AHORA para no reindexar otra vez)

- **Enriquecer `topic` con el ASUNTO.** Hoy el `topic` es genérico ("34. Proposición que presenta
  el Grupo Municipal EH BILDU, cuya parte dispositiva...") sin el tema real (vivienda, IBI...). Por
  eso el filtro de fuentes usa palabras del CONTENIDO, no del topic. NO está roto, pero si algún día
  se quiere topic con asunto (mejor emparejado/recuperación), conviene meterlo en ESTE rebuild para
  no reindexar de nuevo. Coste: cambio de extracción con riesgo. Recomendación: dejarlo fuera salvo
  que se quiera explícitamente.

## 6. Datos pesados (no subir a git)

`actas/` (~251 MB), `chroma_db/` (~2.5 GB) y `.env` (API keys) están en `.gitignore`.
Los PDF se sirven por la ruta propia `/acta/{año}/{fichero}` (no hay carpeta `public/`).
