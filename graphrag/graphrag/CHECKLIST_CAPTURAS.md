# Checklist de capturas — verificación 2026-09-10 23:41

> ⚠️ **Esta verificación corrió con Groq AGOTADO** (límite diario, lo consumió el
> estudio). SPARQL y narración salieron de `qwen2.5:7b` local. Por eso C2/D2/D3
> fallan aquí — **con Groq funcionan** (comprobado en la batería #1/#3). Repetir
> esta verificación cuando Groq resetee, ANTES de capturar.

## Veredicto por pregunta

| # | Grafo | Listo para capturar | Nota |
|---|---|---|---|
| A1 vivienda por grupo | ✅ EH BILDU 62 / PP 46 | **sí** (con Groq para narración) | |
| A2 año más rechazos | ✅ 2018 (56) | **sí** | |
| A3 evolución euskera | ✅ serie por año | **sí** | |
| A4 enmiendas por grupo | ✅ EQ. GOBIERNO 1125 / PP 24 | **sí** | |
| A5 concejal + firmas | ✅ Cristina Ruiz Bujedo (80) | **sí** | |
| A6 presupuestos +subtemas | ✅ 733 | **sí** | |
| A7 concejal +votos en contra | ✅ Yolanda Diez Saiz (309) | **sí** | |
| A8 cómo votó EH Bildu vivienda | ✅ 46 a favor / 19 contra / 11 abst | **sí** — capa de voto nueva | |
| D1 tasa aprobación por grupo | ✅ tabla 11 grupos | **sí** | PP 936/276, EHB 638/200 |
| B1 argumentos EH Bildu presup. | ⬜ 0 filas (esperado) | captura el **vectorial** | grafo no modela argumentos |
| B2 pisos turísticos | ⬜ 0 filas (esperado) | captura el **vectorial** | |
| B3 personas sin hogar | — | captura el **vectorial** | 11 plenos, relato con fuentes |
| C1 Mercadona/Zara | ✅ 0 menciones (honesto) | **sí** — demuestra "solo lo estructurado" | con Groq narra bien el "no está" |
| C2 comercio local | ❌ aquí 0, **con Groq da PP ~59** | **NO sin Groq** | |
| D2 pleno con más rechazos | ❌ aquí matchea la ontología, **con Groq da 22-03-2018 (23)** | **NO sin Groq** | |
| D3 Yolanda Díez veces en contra | ❌ qwen escribió "yolanda de" con CONTAINS, **con Groq da 309** | **NO sin Groq** | |

**Conclusión operativa:** A1–A8 + D1 + C1 son sólidas hasta sin Groq. B1/B2/B3
son del vectorial. C2/D2/D3 necesitan Groq — repetir la verificación cuando
resetee y capturar entonces.

---

## A1. ¿Cuántas proposiciones sobre vivienda ha presentado cada grupo municipal?

**Capturar:** ambos  
**Debe salir:** grafo: tabla, EH BILDU 62 / PP 46 arriba. vector: no sabe contar (da 4-5 de lo que ve)

### GraphRAG (43s, 10 filas)

```sparql
SELECT ?grupo ?nombreGrupo (COUNT(?prop) AS ?nProposiciones)
WHERE {
  ?prop a bo:Proposicion ;
        bo:trataTemaAmplio br:t_vivienda ;
        bo:presentadaPor ?grupo .
  ?grupo rdfs:label ?nombreGrupo .
  FILTER(?grupo != br:grupo_desconocido)
}
GROUP BY ?grupo ?nombreGrupo
ORDER BY DESC(?nProposiciones) LIMIT 20
```

filas: `[{"grupo": "http://bilbao.tfg/resource/grupo_eh_bildu", "nombreGrupo": "EH BILDU", "nProposiciones": "62"}, {"grupo": "http://bilbao.tfg/resource/grupo_pp", "nombreGrupo": "PP", "nProposiciones": "46"}, {"grupo": "http://bilbao.tfg/resource/grupo_elkarrekin_bilbao", "nombreGrupo": "ELKARREKIN BILBAO", "nProposiciones": "17"}, {"grupo": "http://bilbao.tfg/resource/grupo_pse_ee", "nombreGrupo": "PSE-EE", "nProposiciones": "12"}, {"grupo": "http://bilbao.tfg/resource/grupo_goazen_bilbao", "nombreGrupo": "GOAZEN BILBAO", "nProposiciones": "8"}, {"grupo": "http://bilbao.tfg/resource/grupo_equipo_de_gobierno", "nombreGrupo": "EQUIPO DE GOBIERNO", "nProposiciones": "7"}, {"grupo": "http://bilbao.tfg/resource/grupo_udalberri", "nombreGrupo": "UDALBERRI", "nProposiciones": "4"}, {"grupo": "http://bilbao.tfg/resource/grupo_eaj_pnv", "nombreGrupo": "EAJ-PNV", "nProposiciones": "2"}]`

> **Resumen de las proposiciones sobre vivienda presentadas por cada grupo municipal (según los datos obtenidos)**  
> 
> | Grupo | Proposiciones presentadas |
> |-------|---------------------------|
> | **EH BILDU** | **62** |
> | **PP** | **46** |
> | **ELKARREKIN BILBAO** | **17** |
> | **PSE‑EE** | **12** |
> | **GOAZEN BILBAO** | **8** |
> | **EQUIPO DE GOBIERNO** | **7** |
> | **UDALBERRI** | **4** |
> | **EAJ‑PNV** | **2** |
> | **CIUDADANOS** | **1** |
> | **EZKER BATUA‑IU** | **1** |
> 
> ### Narrativa y análisis
> 
> 1. **Liderazgo claro de EH BILDU**  
>    - Con **62** proposiciones, EH BILDU es el grupo que más ha impulsado iniciativas sobre vivienda. Este número supera en **16** proposiciones al segundo puesto (PP) y representa casi **39 %** del total de 160 proposiciones que aparecen en la tabla.
> 
> 2. **PP como segundo actor más activo**  
>    - El Partido Popular (PP) ha presentado **46** propuestas, lo que lo sitúa como el segundo grupo con mayor actividad en este tema. Su cifra, aunque inferior a la de EH BILDU, sigue siendo significativa y muestra que la cuestión de la vivienda es también una prioridad para la fuerza de centro‑derecha.
> 
> 3. **Participación de la izquierda progresista**  
>    - Los grupos de corte progresista (ELKARREKIN BILBAO, PSE‑EE, EAJ‑PNV) suman **31** proposiciones (17 + 12 + 2). Esto indica que, más allá de EH BILDU, la izquierda

### RAG vectorial (32s, 23 plenos)

plenos: 01-04-2014, 18-06-2008, 22-02-2018, 24-11-2015, 25-04-2013, 25-06-2020, 25-09-2013, 25-11-2021, 26-01-2012, 26-10-2010, 27-01-2022, 27-05-2010, 27-09-2012, 28-05-2020, 28-06-2012, 28-11-2007, 28-11-2013, 28-11-2024, 29-01-2015, 29-10-2020, 29-11-2012, 30-09-2021, 31-10-2024

> **26-10-2010 — Grupo Municipal Socialistas Vascos (PSE‑EE)**  
> - **Propuesta:** Instar a la Junta de Gobierno Local a habilitar los fondos necesarios para cumplir el contrato de compra de **176 viviendas** en Gaztelondo y ofertarlas en **régimen de alquiler social**.  
> - **Argumentos:** Los interventores del grupo criticaron la falta de prioridad que el Ayuntamiento había dado a la vivienda, la renuncia al proyecto de Gaztelondo y la escasa producción de viviendas por parte del Grupo Ezker Batua, señalando que la política de vivienda del Gobierno municipal había fracasado y que la vivienda es un instrumento básico de cohesión social y de emancipación de los jóvenes.  
> - **Resultado:** Se aceptó la enmienda del Equipo de Gobierno, por lo que **decayó** la proposición del Grupo Municipal PSE‑EE. (Votos emitidos: 29 | a favor: 15, en contra: 14)
> 
> **25-04-2013 — Grupo Municipal Bildu**  
> - **Propuesta:** Instar al Equipo de Gobierno municipal a **realizar un estudio de viabilidad** para iniciar promociones de **vivienda pública en régimen de alquiler** en parcelas de equipamiento público que no tengan su destino comprometido, citando la necesidad de ampliar el parque público ante el aumento del 50 % de solicitantes desde 2008 y el escaso crecimiento (13 %) del parque existente.  
> - **Argumentos:** Se expuso que el parque de viviendas municipales había pasado de 7 500 


---

## A2. ¿En qué año se rechazaron más proposiciones en el Pleno de Bilbao?

**Capturar:** ambos  
**Debe salir:** grafo: 2018 (56). vector: solo ve unos plenos -> año equivocado con seguridad

### GraphRAG (32s, 1 filas)

```sparql
SELECT ?anio (COUNT(?prop) AS ?nRechazadas)
WHERE {
  ?prop a bo:Proposicion ;
        bo:tieneResultado bo:Rechazada ;
        bo:anio ?anio .
}
GROUP BY ?anio
ORDER BY DESC(?nRechazadas)
LIMIT 1
```

filas: `[{"anio": "2018", "nRechazadas": "56"}]`

> Según los datos proporcionados, en el año 2018 se rechazaron 56 proposiciones en el Pleno del Ayuntamiento de Bilbao. Este es el único año que aparece en la consulta realizada, por lo que no podemos hacer comparaciones con otros años.
> 
> El número exacto de 56 proposiciones rechazadas en 2018 representa el máximo registrado durante el período analizado. Sin datos adicionales para años posteriores o anteriores, no podemos determinar si este año fue un pico atípico o parte de una tendencia general.
> 
> Esta cifra puede indicar varios aspectos del funcionamiento del Pleno:
> - Puede reflejar debates más intensos o controversiales en ciertos períodos.
> - Podría estar relacionado con la agenda política dominante en ese año.
> - También podría ser influenciado por factores externos a la ciudad, como crisis económicas o sociales.
> 
> Es importante notar que este dato se refiere específicamente al número de proposiciones rechazadas y no a todas las presentadas. Sin información adicional sobre el total de proposiciones presentadas en ese año, no podemos determinar el porcentaje de rechazo ni hacer comparaciones con otros años o grupos políticos.
> 
> En resumen, 2018 fue el año con mayor número de proposiciones rechazadas, con un total de 56, lo que representa la única información disponible en este análisis.

### RAG vectorial (101s, 17 plenos)

plenos: 01-04-2014, 07-07-2016, 18-06-2008, 24-11-2015, 25-01-2018, 25-10-2018, 26-10-2017, 27-05-2010, 27-09-2012, 28-06-2012, 28-06-2018, 30-03-2017, 30-05-2013, 30-06-2016, 30-11-2017, 31-01-2013, 31-03-2011

> **28-06-2012 — Grupo Municipal Socialistas Vascos**
> - Propuesta: El pleno insta al Gobierno de España a modificar leyes y revisar acuerdos vigentes desde 1979, relacionados con la Iglesia Católica. También se solicita un censo municipal de inmuebles rústicos y urbanos registrados a nombre de la Iglesia para actualizar el Padrón de Bienes Inmuebles.
> - Argumentos: La portavoz critica que el PSOE haya tenido 21 años para hacer cambios pero no los ha realizado, y señala que votaron en contra de dos proposiciones de ley presentadas por Izquierda Unida. El portavoz del Grupo Municipal Socialistas Vascos defiende la propuesta como una medida solidaria durante la crisis.
> - Resultado: Se acepta la enmienda de modificación presentada por el Equipo de Gobierno, por lo que decae la proposición (Votos emitidos: 29 | a favor: 21, en contra: 8).
> 
> **25-01-2018 — Grupo Municipal GOAZEN BILBAO**
> - Propuesta: El pleno solicita al Gobierno Vasco que haga pública la titularidad de las licencias de VTC ya concedidas y insta a las Administraciones Públicas vascas a crear Mesas de Trabajo para un control efectivo de las actividades de las licencias VTC.
> - Argumentos: Se defiende la necesidad de proteger el servicio público del taxi, que está regulado por la Ordenanza del Taxi. El portavoz del Grupo Municipal GOAZEN BILBAO explica que el sector del taxi se enfrenta a empresas de Vehículos


---

## A3. ¿Cómo ha evolucionado el número de proposiciones sobre euskera año a año?

**Capturar:** grafo  
**Debe salir:** serie por año, pico 2025

### GraphRAG (35s, 15 filas)

```sparql
SELECT ?anio (COUNT(?prop) AS ?nProposiciones)
WHERE {
  ?prop a bo:Proposicion ;
        bo:trataTemaAmplio br:t_euskera ;
        bo:anio ?anio .
}
GROUP BY ?anio
ORDER BY ASC(?anio)
```

filas: `[{"anio": "2008", "nProposiciones": "1"}, {"anio": "2010", "nProposiciones": "2"}, {"anio": "2011", "nProposiciones": "1"}, {"anio": "2013", "nProposiciones": "2"}, {"anio": "2014", "nProposiciones": "1"}, {"anio": "2016", "nProposiciones": "3"}, {"anio": "2017", "nProposiciones": "2"}, {"anio": "2018", "nProposiciones": "4"}]`

> La evolución del número de proposiciones sobre temas relacionados con el euskera en el Ayuntamiento de Bilbao ha mostrado una variación interesante a lo largo de los años. Analizando los datos proporcionados, podemos observar que:
> 
> - En 2008 se presentaron solo 1 proposición.
> - Entre 2010 y 2013, el número de proposiciones osciló entre 1 y 2.
> - En 2014 volvió a caer a 1.
> - A partir de 2016, se observa un incremento gradual: en 2016 fueron 3, en 2017 también 2, pero en 2018 el número subió significativamente hasta alcanzar los 4.
> - En 2019 volvió a caer a 1.
> - A partir de 2020, se observa un aumento constante: 3 en 2020, 9 en 2022 y 2023, y finalmente 12 en 2025.
> 
> Es importante destacar que en 2026, el número de proposiciones bajó a 1. Este patrón sugiere un ciclo de actividad variable en torno al tema del euskera, con picos significativos en 2018 y 2023-2025.
> 
> En general, la tendencia muestra que los años 2023-2025 fueron los más activos en cuanto a presentación de proposiciones sobre temas relacionados con el euskera, con un total de 9 propuestas cada uno.


---

## A4. ¿Cuántas enmiendas ha presentado cada grupo municipal?

**Capturar:** grafo  
**Debe salir:** EQUIPO DE GOBIERNO ~1125 muy por encima; PP 24

### GraphRAG (35s, 8 filas)

```sparql
SELECT ?grupoMunicipal (COUNT(DISTINCT ?enm) AS ?nEnmiendas)
WHERE {
  ?enm a bo:Enmienda ;
       bo:enmiendaPor ?grupoMunicipal .
}
GROUP BY ?grupoMunicipal
ORDER BY DESC(?nEnmiendas) LIMIT 20
```

filas: `[{"grupoMunicipal": "http://bilbao.tfg/resource/grupo_equipo_de_gobierno", "nEnmiendas": "1125"}, {"grupoMunicipal": "http://bilbao.tfg/resource/grupo_eaj_pnv", "nEnmiendas": "125"}, {"grupoMunicipal": "http://bilbao.tfg/resource/grupo_eh_bildu", "nEnmiendas": "66"}, {"grupoMunicipal": "http://bilbao.tfg/resource/grupo_elkarrekin_bilbao", "nEnmiendas": "37"}, {"grupoMunicipal": "http://bilbao.tfg/resource/grupo_pse_ee", "nEnmiendas": "33"}, {"grupoMunicipal": "http://bilbao.tfg/resource/grupo_pp", "nEnmiendas": "24"}, {"grupoMunicipal": "http://bilbao.tfg/resource/grupo_udalberri", "nEnmiendas": "14"}, {"grupoMunicipal": "http://bilbao.tfg/resource/grupo_goazen_bilbao", "nEnmiendas": "8"}]`

> Basándonos en los datos proporcionados, podemos observar que el grupo municipal "Equipo de Gobierno" ha presentado un total de 1125 enmiendas al Pleno del Ayuntamiento de Bilbao. Este número es significativamente mayor que el segundo lugar, que corresponde al grupo "EAJ-PNV", con 125 enmiendas.
> 
> El tercer grupo más activo en la presentación de enmiendas es "EH Bildu" con 66 enmiendas, seguido por "Elkarrekin Bilbao" con 37 y "PSE-EE" con 33. El grupo "PP" ha presentado 24 enmiendas, mientras que "Udalberri" y "Goazén Bilbao" han presentado 14 y 8 enmiendas respectivamente.
> 
> Esta distribución muestra una clara predominancia del grupo municipal "Equipo de Gobierno", seguido por "EAJ-PNV". Los grupos "EH Bildu", "Elkarrekin Bilbao" y "PSE-EE" también son relevantes, pero presentan un número significativamente menor de enmiendas. Por otro lado, los grupos "Udalberri" y "Goazén Bilbao" han presentado el menor número de enmiendas entre los analizados.
> 
> En resumen, los tres grupos más activos en la presentación de enmiendas son:
> 1. Equipo de Gobierno: 1125 enmiendas
> 2. EAJ-PNV: 125 enmiendas
> 3. EH Bildu: 66 enmiendas
> 
> Estas cifras reflejan la participación política y el nivel de actividad en el Pleno del Ayuntamiento de Bilbao, con un claro liderazgo del grupo municipal "Equipo de Gobierno".


---

## A5. ¿Qué concejal o concejala ha firmado más proposiciones?

**Capturar:** grafo  
**Debe salir:** Cristina Ruiz Bujedo (80)

### GraphRAG (37s, 20 filas)

```sparql
SELECT ?nombre (COUNT(DISTINCT ?prop) AS ?nFirmadas)
WHERE {
  ?prop bo:proponePersona ?c .
  ?c rdfs:label ?nombre .
}
GROUP BY ?c ?nombre
ORDER BY DESC(?nFirmadas)
LIMIT 20
```

filas: `[{"nombre": "Cristina Ruiz Bujedo", "nFirmadas": "80"}, {"nombre": "Luis Hermosa Santamaria", "nFirmadas": "41"}, {"nombre": "Luis Eguiluz Zobaran", "nFirmadas": "40"}, {"nombre": "Francisco Samir Lahdou del Arco", "nFirmadas": "32"}, {"nombre": "Angel Rodrigo Izquierdo", "nFirmadas": "29"}, {"nombre": "Jose Maria Oleaga Zalvidea", "nFirmadas": "24"}, {"nombre": "Oscar Fernandez Monroy", "nFirmadas": "23"}, {"nombre": "Esteban Goti Bueno", "nFirmadas": "22"}]`

> Basándonos en los datos proporcionados, podemos observar que el concejal o concejala que ha firmado más proposiciones es Cristina Ruiz Bujedo con un total de 80 propuestas presentadas en el Pleno del Ayuntamiento de Bilbao. 
> 
> En segundo lugar se encuentra Luis Hermosa Santamaria con 41 propuestas, seguido de cerca por Luis Eguiluz Zobaran con 40 propuestas. Francisco Samir Lahdou del Arco ocupa la cuarta posición con 32 propuestas.
> 
> El ranking continúa con Angel Rodrigo Izquierdo en quinto lugar con 29 propuestas, y Jose Maria Oleaga Zalvidea en sexto con 24 propuestas. Oscar Fernandez Monroy se sitúa en séptimo lugar con 23 propuestas, seguido de Esteban Goti Bueno con 22.
> 
> Las cifras disminuyen gradualmente a medida que avanzamos por la lista: Carmen Carron Galende con 19 propuestas, Ana Vinals Blanco con 18, Carlos David Garcia Fernandez con 17, Beatriz Marcos Gonzalez con 16, y Carmen Munoz Lopez con 15. Xabier Jimenez Lopez presenta el menor número de propuestas en este ranking con solo 11.
> 
> Es importante destacar que Paula Garagalza Munoz, Alfonso Gil Invernon, Esther Martinez Fernandez y Alba Fatuarte Gonzalez presentan la misma cantidad de propuestas, con un total de 7 cada una. 
> 
> En resumen, Cristina Ruiz Bujedo es el concejal o concejala más activo en términos de número de proposiciones presentadas, seguido por Luis Hermosa Santamaria y Luis Eguilu


---

## A6. ¿Cuántas proposiciones sobre presupuestos y fiscalidad se han presentado en total, incluyendo subtemas?

**Capturar:** grafo  
**Debe salir:** 733

### GraphRAG (13s, 1 filas)

```sparql
SELECT (COUNT(?prop) AS ?totalProposiciones)
WHERE {
  ?prop a bo:Proposicion ;
        bo:trataTemaAmplio br:t_presupuestos .
}
```

filas: `[{"totalProposiciones": "733"}]`

> Según los datos del grafo, se han presentado **733 proposiciones** que tratan el tema amplio de presupuestos y fiscalidad, abarcando también los subtemas relacionados.  
> 
> Esta cifra representa el número total de iniciativas legislativas que han sido introducidas en el Pleno del Ayuntamiento de Bilbao bajo la categoría de presupuestos, sin distinguir entre los diferentes subtemas que puedan estar incluidos dentro de esa área. En otras palabras, en todo el periodo cubierto por el registro del grafo, los concejales han presentado siete cientos treinta y tres propuestas cuyo objetivo es abordar cuestiones presupuestarias y fiscales.  
> 
> No se dispone de desgloses temporales ni de comparaciones con otros temas en este conjunto de datos; la información disponible se limita a este total acumulado.


---

## A7. ¿Qué concejal o concejala ha votado más veces en contra de las proposiciones?

**Capturar:** grafo  
**Debe salir:** Yolanda Diez Saiz (309)

### GraphRAG (27s, 20 filas)

```sparql
SELECT ?n (COUNT(*) AS ?veces)
WHERE {
  ?prop bo:concejalVotoEnContra ?c .
  ?c rdfs:label ?n .
}
GROUP BY ?c ?n
ORDER BY DESC(?veces)
LIMIT 20
```

filas: `[{"n": "Yolanda Diez Saiz", "veces": "309"}, {"n": "Alfonso Gil Invernon", "veces": "266"}, {"n": "Marta Ajuria Arribas", "veces": "264"}, {"n": "Goyo Zurro Tobajas", "veces": "261"}, {"n": "Itziar Urtasun Jimeno", "veces": "261"}, {"n": "Asier Abaunza Robles", "veces": "256"}, {"n": "Angel Rodrigo Izquierdo", "veces": "237"}, {"n": "Beatriz Marcos Gonzalez", "veces": "204"}]`

> Según los datos proporcionados, la concejala Yolanda Diez Saiz ha votado en contra de las proposiciones más veces, con un total de 309 votos. Esto la sitúa claramente como la miembro del equipo que más se opone a las iniciativas presentadas en el Pleno.
> 
> En segundo lugar, Alfonso Gil Invernon ha votado en contra 266 veces, seguido de cerca por Marta Ajuria Arribas y Goyo Zurro Tobajas, con 264 y 261 votos respectivamente. Estos tres concejales son los que más han votado en contra, destacando su posición contraria a las propuestas presentadas.
> 
> El resto del ranking muestra una cierta similitud en el número de votos en contra, con valores entre 139 y 158. Notablemente, estos números indican que la oposición es generalizada pero no uniforme entre los miembros del equipo.
> 
> Es importante destacar que este análisis se basa únicamente en las proposiciones presentadas en el Pleno, y no en otros aspectos como la asistencia a sesiones o votaciones sobre temas específicos.


---

## A8. ¿Cómo votó EH Bildu las proposiciones sobre vivienda?

**Capturar:** grafo  
**Debe salir:** reparto a favor/contra/abstención por la capa de voto del acta

### GraphRAG (26s, 1 filas)

```sparql
SELECT (COUNT(DISTINCT ?pf) AS ?aFavor) (COUNT(DISTINCT ?pc) AS ?enContra) (COUNT(DISTINCT ?pa) AS ?abst)
WHERE {
  ?prop bo:trataTemaAmplio br:t_vivienda .
  OPTIONAL { ?prop bo:votoAFavorDe   br:grupo_eh_bildu . BIND(?prop AS ?pf) }
  OPTIONAL { ?prop bo:votoEnContraDe br:grupo_eh_bildu . BIND(?prop AS ?pc) }
  OPTIONAL { ?prop bo:seAbstuvo      br:grupo_eh_bildu . BIND(?prop AS ?pa) }
}
```

filas: `[{"aFavor": "46", "enContra": "19", "abst": "11"}]`

> EH Bildu votó en 76 ocasiones sobre las proposiciones relacionadas con vivienda presentadas en el Pleno del Ayuntamiento de Bilbao. De estas, 46 fueron a favor, 19 estuvieron en contra y 11 se abstuvieron.
> 
> Esta distribución muestra que EH Bildu tuvo una participación significativa en la votación sobre temas de vivienda, con un total de 76 proposiciones tratadas. La proporción más alta fue el voto a favor, con 46 casos, seguido del voto en contra (19) y las abstenciones (11).
> 
> Es importante destacar que estos datos representan la postura de EH Bildu sobre los temas de vivienda presentados durante el período cubierto por esta información. Sin embargo, para obtener una visión más completa, sería necesario comparar estas cifras con otros grupos políticos o con las votaciones en años anteriores o posteriores.


---

## B1. ¿Qué argumentos ha dado EH Bildu para oponerse a los presupuestos municipales de Bilbao?

**Capturar:** ambos  
**Debe salir:** grafo: ruido / poco. vector: argumentos reales con cifras del debate

### GraphRAG (31s, 0 filas)

```sparql
SELECT ?enmienda ?argumento (COUNT(?prop) AS ?totalOposiciones)
WHERE {
  # Buscamos las enmiendas presentadas por EH Bildu
  ?enmienda a bo:Enmienda ;
            bo:enmiendaPor br:grupo_eh_bildu .

  # Filtramos para que la enmienda se oponga al presupuesto municipal de Bilbao
  FILTER(REGEX(STR(?enmienda), "\\bpr[eé]s[uúü]p[uúü][eé]st[oó]\\b", "i") && REGEX(STR(?enmienda), "\\bb[ií]lb[aá][oó]\\b", "i"))

  # Buscamos los argumentos de la enmienda
  ?enmienda bo:tieneArgumento ?argumento .
}
GROUP BY ?enmienda ?argumento
ORDER BY DESC(?totalOposiciones)
LIMIT 20
```

filas: `[]`

> No se encontraron datos que correspondan a los argumentos dados por EH Bildu para oponerse a los presupuestos municipales de Bilbao. Según la consulta SPARQL realizada, no hay registros de enmiendas presentadas por EH Bildu que se opongan al presupuesto municipal de Bilbao ni de los argumentos asociados a estas enmiendas.
> 
> Si necesitas información sobre otros aspectos relacionados con las enmiendas de EH Bildu o cualquier otro tema, estaré encantado de ayudarte.

### RAG vectorial (63s, 7 plenos)

plenos: 24-11-2016, 26-05-2016, 26-10-2017, 28-01-2016, 29-06-2017, 29-10-2015, 30-10-2025

> **28-01-2016 — Grupo Municipal EH BILDU**
> - Propuesta: Presentaron una propuesta sobre la participación ciudadana en la tramitación de los Presupuestos. Señalaron que la participación previa podría corregir o mejorar el presupuesto, evitando su posterior corrección.
> - Argumentos: Defendieron la importancia de abrir cauces de participación para que la ciudadanía pudiera presentar propuestas y sugerencias antes del proceso de elaboración del Anteproyecto. Consideraron que esta medida representaba un avance en el sistema de gobierno.
> - Resultado: Se aceptó la enmienda del EQUIPO DE GOBIERNO, por lo que la proposición original se vio rechazada (Votos emitidos: 29 | a favor: 27, abstenciones: 2).
> 
> **26-10-2017 — Grupo Municipal EH BILDU**
> - Propuesta: Solicitaron al Gobierno Vasco que provea los presupuestos de 2018 con el dinero necesario para conceder las Ayudas de Emergencia Social, con el objetivo de hacer frente a la pobreza en la ciudad. También se comprometieron a apoyar la ampliación de la partida presupuestaria si fuera necesario.
> - Argumentos: Expusieron que no era casualidad esta iniciativa y la relacionaron con la presentación del presupuesto del Gobierno Vasco, destacando la necesidad de una buena gestión para solicitar más fondos. Sugirieron que el presupuesto para las ayudas de emergencia social podría mantenerse en 27 millones de euros.
> - Resultado: S


---

## B2. ¿Qué se debatió sobre el registro de viviendas turísticas y los pisos turísticos en Bilbao?

**Capturar:** ambos  
**Debe salir:** vector: propuestas 2024 tasa turística. grafo: cuenta pero sin matiz

### GraphRAG (32s, 0 filas)

```sparql
SELECT ?anio (COUNT(?prop) AS ?n)
WHERE {
  ?prop a bo:Proposicion ;
        bo:trataTemaAmplio br:t_vivienda ;
        bo:presentadaPor ?grupo .
  ?grupo rdfs:label ?nombreGrupo .
  FILTER(?grupo != br:grupo_desconocido)
  FILTER(?anio = ?anio)
}
GROUP BY ?anio
ORDER BY ASC(?anio)
LIMIT 20
```

filas: `[]`

> No se encontraron datos relacionados con el registro de viviendas turísticas o pisos turísticos en Bilbao para los años consultados. Según la consulta realizada, no hubo ninguna proposición que tratara estos temas durante el período analizado.
> 
> Esta ausencia sugiere que, al menos en los años contemplados por la consulta (que se limita a mostrar las 20 primeras filas), no hubo iniciativas presentadas en el Pleno del Ayuntamiento de Bilbao relacionadas con el registro de viviendas turísticas o pisos turísticos.

### RAG vectorial (63s, 17 plenos)

plenos: 01-04-2014, 23-02-2012, 23-09-2010, 25-01-2018, 25-01-2024, 25-09-2013, 26-01-2017, 26-03-2015, 26-06-2025, 28-09-2017, 28-10-2011, 28-11-2007, 29-01-2015, 29-10-2009, 29-11-2018, 30-01-2025, 30-10-2008

> **30-01-2025 — ELKARREKIN BILBAO**
> - Propuesta: Se propone la aprobación inicial de la modificación del Plan General de Ordenación Urbana (PGOU) en relación con las viviendas turísticas, principalmente concentradas en el Casco Viejo y otras áreas.
> - Argumentos: Se argumenta que las viviendas turísticas representan un 0,6% de las viviendas en Bilbao y no son responsables del aumento de precios del alquiler. Se critica la falta de regulación y registro de estas viviendas.
> - Resultado: Aprobada. Votos a favor: 17, abstenciones: 12
> 
> **26-06-2025 — ELKARREKIN BILBAO**
> - Propuesta: Se propone la adopción de un acuerdo plenario para garantizar la función social de la vivienda en Bilbao.
> - Argumentos: Se discute sobre el alto porcentaje de viviendas vacías (5,37%) y se critica la falta de regulación. Se menciona que más del 80% de las personas que utilizan Etxebide ya están en alquiler.
> - Resultado: Se aprueba la enmienda de modificación formulada por el EQUIPO DE GOBIERNO, por lo que decae la proposición presentada por ELKARREKIN BILBAO. Votos emitidos: 29 | a favor: 23, en contra: 6
> 
> **26-03-2015 — GRUPO MUNICIPAL PP**
> - Propuesta: Se propone la agilización de los trámites para la futura Ley de Turismo de Euskadi.
> - Argumentos: Se menciona que el Decreto de enero ya regula la inscripción de pisos turísticos, pero se considera necesario agilizar los trámites de


---

## B3. ¿Qué se dijo sobre las personas sin hogar en Bilbao?

**Capturar:** vector  
**Debe salir:** relato con varias sesiones, fuentes

### RAG vectorial (58s, 11 plenos)

plenos: 12-11-2007, 13-09-2024, 15-09-2025, 21-12-2007, 22-03-2018, 25-03-2021, 26-10-2017, 27-09-2018, 28-06-2018, 28-11-2007, 29-06-2017

> **13-09-2024 — General / Introducción**
> - Propuesta: No se presentó una propuesta específica, pero se destacaron los datos sobre el sin hogarismo en Bilbao y Bizkaia.
> - Argumentos: Se subrayó que Bilbao es líder en políticas sociales de atención al sin hogarismo, con un 74% de las plazas y el 93% del gasto total. Sin embargo, se reconoció la necesidad de mejorar.
> - Resultado: [Sin resultado en acta]
> 
> **26-10-2017 — 14. Proposición que presenta el Grupo Municipal UDALBERRI -BILBAO EN COMÚN**
> - Propuesta: Se propuso garantizar el acceso al empadronamiento de las personas residentes en Bilbao cumpliendo con ciertas obligaciones.
> - Argumentos: Se valoraron los avances logrados en servicios sociales, pero se subrayó la necesidad de seguir avanzando y no dejar que los éxitos pasen desapercibidos. También se mencionó el problema de la pobreza y la brecha social en Euskadi.
> - Resultado: Se acepta la enmienda del EQUIPO DE GOBIERNO, por lo que decae la proposición (Votos emitidos: 29 | a favor: 21, en contra: 2, abstenciones: 6)
> 
> **27-09-2018 — 44. Proposición que presenta el Grupo Municipal GOAZEN BILBAO**
> - Propuesta: Se solicitó al EQUIPO DE GOBIERNO realizar un estudio sobre la situación de las personas mayores de 65 años en materia de vivienda y establecer programas de actuación para paliar deficiencias.
> - Argumentos: Se presentaron datos sobre el empadronam


---

## C1. ¿Se ha mencionado a Mercadona o a Zara en algún pleno del Ayuntamiento de Bilbao?

**Capturar:** ambos  
**Debe salir:** grafo: vacío -> 'no es entidad del grafo' (honesto). vector: busca literal

### GraphRAG (31s, 1 filas)

```sparql
SELECT (COUNT(DISTINCT ?prop) AS ?nMenciones)
WHERE {
  ?prop bo:trataTemaAmplio ?tema .
  FILTER(REGEX(STR(?tema), "\\bm[eé]rc[aá]d[oó][nñ][aá]\\b", "i") || REGEX(STR(?tema), "\\bz[aá]r[aá]\\b", "i"))
}
```

filas: `[{"nMenciones": "0"}]`

> No se han registrado ninguna proposición en el Pleno del Ayuntamiento de Bilbao que mencione a Mercadona o a Zara. La búsqueda realizada no ha encontrado referencias a estos temas, lo cual indica que durante los plenos analizados no se han presentado iniciativas relacionadas con Mercadona ni con Zara.

### RAG vectorial (75s, 10 plenos)

plenos: 10-05-2024, 12-11-2007, 17-06-2023, 23-09-2010, 24-02-2011, 24-11-2015, 28-01-2016, 28-10-2011, 28-11-2007, 28-11-2013

> **28-10-2011 — Grupo Municipal EH BILDU**
> - Propuesta: Modificar la figura concreta de gestión directa mediante sociedad mercantil, sustituyéndola por una única Entidad Pública Empresarial Local que asuma la totalidad del objeto social.
> - Argumentos: El cambio se justifica para unificar y optimizar la gestión de las sociedades existentes. Se aprobó inicialmente el contenido de sus estatutos y se sometió a información pública y audiencia de los interesados.
> - Resultado: Aprobada. Votos a favor: 15, en contra: 7, abstenciones: 6
> 
> **23-09-2010 — Grupo Popular**
> - Propuesta: Debates sobre el Impuesto sobre Bienes Inmuebles y la Tasa de Alcantarillado.
> - Argumentos: El grupo popular argumenta que la única política fiscal posible es la del Gobierno, pero finalmente se permitió debatir estos temas. Sin embargo, hubo muchos rodeos en este proceso.
> - Resultado: Aprobada. Votos a favor: 22, abstenciones: 7
> 
> **28-11-2013 — Grupo Popular**
> - Propuesta: Discusión sobre la gestión de organismos autónomos locales y entidades empresariales municipales.
> - Argumentos: Se critica que estas entidades no son gestionadas directamente por el Ayuntamiento, lo cual dificulta su transparencia. También se menciona un proceso de selección sospechoso.
> - Resultado: Aprobada. Votos a favor: 23, abstenciones: 5
> 
> **17-06-2023 — Alcalde**
> - Propuesta: Creación de un polo audiovis


---

## C2. ¿Qué grupo ha impulsado más iniciativas a favor del comercio local de barrio?

**Capturar:** grafo  
**Debe salir:** cuenta comercio (PP arriba) pero no distingue 'a favor de' -> limitación

### GraphRAG (38s, 0 filas)

```sparql
SELECT ?grupo ?nombreGrupo (COUNT(?prop) AS ?n)
WHERE {
  # Filtramos las proposiciones que tratan sobre el tema "comercio local de barrio"
  ?prop a bo:Proposicion ;
        bo:trataTemaAmplio br:t_comerciolocaldebarrio .

  # Aseguramos que la proposición tenga un resultado
  ?prop bo:tieneResultado ?resultado .

  # Filtramos solo las proposiciones con resultado "Aprobada"
  FILTER(?resultado = bo:Aprobada) .

  # Asociamos el grupo que presentó la proposición
  ?prop bo:presentadaPor ?grupo .

  # Obtenemos el nombre del grupo
  ?grupo rdfs:label ?nombreGrupo .
}
GROUP BY ?grupo ?nombreGrupo
ORDER BY DESC(?n)
LIMIT 20
```

filas: `[]`

> No se encontraron datos para responder a la pregunta sobre cuál es el grupo que ha impulsado más iniciativas a favor del comercio local de barrio. Según los datos proporcionados, no hay ninguna proposición aprobada relacionada con este tema en el grafo utilizado para esta consulta.
> 
> Para poder ofrecer una respuesta precisa y completa, se necesitarían datos adicionales sobre las iniciativas presentadas por cada grupo político en relación al comercio local de barrio.


---

## D1. ¿Cuántas proposiciones ha presentado cada grupo y qué tasa de aprobación tiene?

**Capturar:** ambos  
**Debe salir:** grafo: tabla 11 grupos con total+aprobadas. vector: no puede

### GraphRAG (45s, 10 filas)

```sparql
SELECT ?grupo ?nombreGrupo (COUNT(?prop) AS ?totalProposiciones)
       (COUNT(?aprobada) AS ?aprobadas)
WHERE {
  # Proposición presentada por un grupo
  ?prop a bo:Proposicion ;
        bo:presentadaPor ?grupo .

  # Nombre del grupo
  ?grupo rdfs:label ?nombreGrupo .

  # Filtrar grupos conocidos (excluimos el grupo desconocido)
  FILTER(?grupo != br:grupo_desconocido)

  # Opcionalmente, ver si la proposición fue aprobada
  OPTIONAL {
    ?prop bo:tieneResultado ?_rApr . FILTER(?_rApr IN (bo:Aprobada, bo:AprobadaConEnmienda)) .
    BIND(?prop AS ?aprobada)
  }
}
GROUP BY ?grupo ?nombreGrupo
ORDER BY DESC(?totalProposiciones) LIMIT 20
```

filas: `[{"grupo": "http://bilbao.tfg/resource/grupo_pp", "nombreGrupo": "PP", "totalProposiciones": "936", "aprobadas": "276"}, {"grupo": "http://bilbao.tfg/resource/grupo_eh_bildu", "nombreGrupo": "EH BILDU", "totalProposiciones": "638", "aprobadas": "200"}, {"grupo": "http://bilbao.tfg/resource/grupo_pse_ee", "nombreGrupo": "PSE-EE", "totalProposiciones": "271", "aprobadas": "88"}, {"grupo": "http://bilbao.tfg/resource/grupo_elkarrekin_bilbao", "nombreGrupo": "ELKARREKIN BILBAO", "totalProposiciones": "231", "aprobadas": "78"}, {"grupo": "http://bilbao.tfg/resource/grupo_udalberri", "nombreGrupo": "UDALBERRI", "totalProposiciones": "136", "aprobadas": "45"}, {"grupo": "http://bilbao.tfg/resource/grupo_goazen_bilbao", "nombreGrupo": "GOAZEN BILBAO", "totalProposiciones": "119", "aprobadas": "37"}, {"grupo": "http://bilbao.tfg/resource/grupo_equipo_de_gobierno", "nombreGrupo": "EQUIPO DE GOBIERNO", "totalProposiciones": "67", "aprobadas": "16"}, {"grupo": "http://bilbao.tfg/resource/grupo_eaj_pnv", "nombreGrupo": "EAJ-PNV", "totalProposiciones": "41", "aprobadas": "13"}]`

> Basándonos en los datos proporcionados, podemos observar que el grupo político con mayor número de proposiciones presentadas es PP (Partido Popular) con un total de 936 propuestas, seguido del grupo EH BILDU (Euskal Herria Bildu) con 638 propuestas. El tercer lugar lo ocupa PSE-EE (Partido Socialista Euskadi-Euskadiko EzkerBEREZ), con 271 propuestas.
> 
> En términos de tasa de aprobación, el grupo PP tiene una tasa del 29.5%, seguido de EH BILDU con un 31.3%. PSE-EE presenta la mayor tasa de aprobación entre los grupos analizados, alcanzando el 32.5% de sus propuestas presentadas.
> 
> El grupo ELKARREKIN BILBAO tiene una tasa de aprobación del 33.8%, seguido de UDALBERRI con un 33.1%. GOAZEN BILBAO presenta la menor tasa de aprobación entre los grupos analizados, con solo el 31.1% de sus propuestas siendo aprobadas.
> 
> El grupo EQUIPO DE GOBIERNO tiene una tasa de aprobación del 23.9%, mientras que EAJ-PNV (Euskal Alderdi Jeltzalea-Partido Nacionalista Vasco) presenta un 31.7%. EZKER BATUA-IU tiene la mayor tasa de aprobación entre los grupos analizados, con el 50% de sus propuestas presentadas siendo aprobadas.
> 
> Finalmente, CIUDADANOS (Ciudadanos Partido de la Ciudadanía) presenta la menor tasa de aprobación del grupo analizado, con solo un 14.3% de sus propuestas presentadas siendo aprobadas.
> 
> En resumen, el análisis muestra que EH BILDU es el grupo más activo en 

### RAG vectorial (56s, 16 plenos)

plenos: 18-06-2008, 19-05-2008, 24-09-2009, 24-11-2015, 25-03-2010, 27-05-2010, 27-09-2012, 27-11-2008, 28-01-2009, 28-01-2010, 28-05-2009, 28-06-2012, 28-11-2007, 29-10-2009, 30-10-2008, 31-05-2012

> **27-05-2010 — Grupo Municipal Popular**
> - Propuesta: El Pleno del Ayuntamiento de Bilbao pide a la BBK que presente públicamente el nuevo proyecto que justificó el cierre rápido e inmediato de la Residencia de la Paz y insta a la BBK a que, una vez subsanada la situación de ilegalidad en la que se encontraba el centro, el nuevo proyecto contenga en el tiempo una residencia para mayores.
> - Argumentos: El PNV ha tomado medidas desde su obra social contrarias al sentido común y a los intereses de las familias. Se han demostrado que las justificaciones del cierre eran falsas y que se actuó con otros intereses.
> - Resultado: Votos emitidos: 29 | a favor: 15, abstenciones: 14
> 
> **27-11-2008 — Grupo Municipal Popular**
> - Propuesta: El Ayuntamiento de Bilbao insta al Gobierno Vasco a que en el primer trimestre del año 2009 transfiera a la Diputación Foral de Vizcaya la cantidad de 1.229.648.988.- euros anuales, actualizables todos los años conforme al IPC.
> - Argumentos: Se trata de mejorar la financiación local para poder desarrollar políticas y actividades para la ciudadanía. El problema estructural de la financiación foral y local ha llevado a que sea necesario poner en marcha reformas inmediatas.
> - Resultado: Votos emitidos: 29 | a favor: 7, en contra: 22
> 
> **28-05-2009 — Grupo Municipal Popular**
> - Propuesta: El Pleno del Ayuntamiento de Bilbao insta a la Junta 


---

## D2. ¿Cuál fue el pleno concreto con más proposiciones rechazadas de toda la serie?

**Capturar:** ambos  
**Debe salir:** grafo: 22-03-2018 (23). vector: ve 2 plenos -> falso

### GraphRAG (66s, 1 filas)

```sparql
SELECT ?pleno ?nombrePleno (COUNT(?prop) AS ?nRechazadas)
WHERE {
  ?prop a bo:Proposicion ;
        bo:tieneResultado bo:Rechazada  .
  ?pleno rdfs:label ?nombrePleno .
}
GROUP BY ?pleno ?nombrePleno
ORDER BY DESC(?nRechazadas)
LIMIT 1
```

filas: `[{"pleno": "http://bilbao.tfg/ontology#", "nombrePleno": "Ontología de los Plenos del Ayuntamiento de Bilbao", "nRechazadas": "457"}]`

> Según los datos proporcionados, el pleno con más proposiciones rechazadas en toda la serie fue el "Ontología de los Plenos del Ayuntamiento de Bilbao", con un total de 457 proposiciones que resultaron ser rechazadas.
> 
> Esta cifra es significativa y destaca entre todos los plenos analizados, ya que no hay otras filas en los datos proporcionados. Por lo tanto, podemos afirmar que este pleno tuvo el mayor número de proposiciones rechazadas en comparación con cualquier otro pleno registrado en la serie.
> 
> Es importante notar que esta información se basa únicamente en los datos del grafo proporcionados y no incluye otros posibles plenos o sesiones del ayuntamiento.

### RAG vectorial (44s, 18 plenos)

plenos: 18-06-2008, 19-09-2008, 23-09-2010, 24-04-2008, 24-11-2015, 25-02-2010, 25-03-2010, 25-10-2018, 26-01-2012, 27-05-2010, 27-09-2012, 28-01-2010, 28-06-2012, 28-11-2007, 29-10-2009, 30-03-2017, 30-10-2008, 31-01-2013

> **24-04-2008 — Grupo Municipal Popular**
> - Propuesta: Instar a la Junta de Gobierno a depurar toda responsabilidad política en el atropello que costó la vida a dos menores en Basurto.
> - Argumentos: El fallo del juicio atribuye la confusión del conductor al mal posicionamiento del semáforo y señala la dejadez institucional. Se exige asunción de responsabilidad política para evitar futuros errores.
> - Resultado: Rechazada. Votos emitidos: 29 | a favor: 14, en contra: 15
> 
> **28-01-2010 — Grupo Municipal Popular**
> - Propuesta: Instar a la Junta de Gobierno Local a la puesta en práctica de medidas para la promoción del aparcamiento municipal de la calle Carmelo II de Santutxu.
> - Argumentos: Se critica que el Pleno se ha convertido en un espacio de confrontación y se propone llevar la discusión a una Comisión para debatir los pros y contras. Se defiende la importancia de las políticas de aparcamientos, aunque no se detallan más argumentos.
> - Resultado: [Sin resultado en acta]
> 
> **24-11-2015 — Grupo Municipal GOAZEN BILBAO**
> - Propuesta: Rechazo a la instalación de un almacén temporal de residuos nucleares en las instalaciones de la central nuclear de Santa María de Garoña.
> - Argumentos: Se critica que la Central Nuclear es obsoleta y peligrosa, y se solicita el cierre definitivo. Se propone un plan de cierre para otras centrales termonucleares del Estado.
> - Res


---

## D3. ¿Cuántas veces ha votado en contra la concejala Yolanda Díez?

**Capturar:** grafo  
**Debe salir:** 309 (tras el guard de tildes)

### GraphRAG (39s, 0 filas)

```sparql
SELECT ?n (COUNT(*) AS ?vecesEnContra)
WHERE {
  ?prop bo:concejalVotoEnContra ?c .
  ?c rdfs:label ?n .
  FILTER(CONTAINS(LCASE(STR(?n)), "yolanda de"))
}
GROUP BY ?c ?n
ORDER BY DESC(?vecesEnContra) LIMIT 20
```

filas: `[]`

> No se encontraron datos para la concejala Yolanda Díez que cumplan con los criterios especificados en la consulta. Según la consulta SPARQL ejecutada, buscamos las veces en contra que ha votado una concejala cuyo nombre contenga "Yolanda de". Sin embargo, no hay resultados en el grafo que correspondan a esta búsqueda.
> 
> Es importante señalar que esto significa que, según los datos disponibles, la concejala Yolanda Díez no ha votado en contra en ninguna proposición registrada en el Ayuntamiento de Bilbao.

