import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdflib.namespace import SKOS  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402

from comun.grupos import extrae_grupo, canon_grupo, prop_id  # noqa: E402
from comun.rutas import CHROMA_PATH, dato_grafo  # noqa: E402
from vectorial.recuperacion import strip_accents  # noqa: E402
from grafo.construccion.build_rdf import (  # noqa: E402
    ONTOLOGY, THEMES, Graph,
    resultado_cruzado, normaliza_resultado, parse_votos,
    find_canonical, canon_theme_map, norm_label,
)
from grafo.construccion.texto_actas import load_jsonl, iter_jsonl  # noqa: E402
from vectorial.pipeline import RAGPipeline  # noqa: E402

ENRICHED = dato_grafo("proposals_enriched.jsonl")
PROPOSALS = dato_grafo("proposals.jsonl")
PREFIX_LEN = 80
BATCH = 300


def join_key(date: str, topic: str) -> str:
    t = strip_accents((topic or "").strip().lower())[:PREFIX_LEN]
    return (date or "") + "||" + t


def build_derived_map():
    g = Graph()
    g.parse(ONTOLOGY, format="turtle")
    g.parse(THEMES, format="turtle")
    canon = canon_theme_map(g)

    def label_of(uri):
        if uri is None:
            return None
        lbl = g.value(uri, SKOS.prefLabel)
        return str(lbl) if lbl else None

    # slug del tema de NIVEL 1 al que sube un URI de tema (siguiendo skos:broader).
    # Sirve para escribir en cada chunk banderas booleanas tf_<slug> con TODOS
    # los temas de nivel 1 que le tocan (por tema_principal Y por los secundarios),
    # y que _thematic_search filtre por ahí en vez de solo por tema_principal —
    # así una proposición con tema_principal="movilidad" pero un tema secundario
    # "calidad del aire" sí aparece al preguntar por medio ambiente.
    def top_slug(uri):
        seen = set()
        while uri is not None and uri not in seen:
            seen.add(uri)
            parent = g.value(uri, SKOS.broader)
            if parent is None:
                local = str(uri).rsplit("/", 1)[-1].split("#")[-1]
                return local[2:] if local.startswith("t_") else local  # "t_medioambiente" -> "medioambiente"
            uri = parent
        return None

    TOP_SLUGS = sorted({
        top_slug(c) for c in g.subjects(SKOS.topConceptOf, None)
    } - {None})

    textmap = {}
    if os.path.exists(PROPOSALS):
        for p in iter_jsonl(PROPOSALS):
            textmap[prop_id(p)] = p.get("text", "")

    recs = load_jsonl(ENRICHED)
    derived_by_key = {}
    for r in recs:
        meta = {"prop_id": r["id"]}

        res = normaliza_resultado(r.get("resultado", ""))
        res = resultado_cruzado(r.get("vote_result", ""), res)
        meta["resultado"] = res

        # El campo vote_result ORIGINAL de cada chunk (calculado por
        # vectorial/indexado.py::_process_single_pdf en la indexación de ChromaDB)
        # se queda desactualizado si `_process_single_pdf` se corrige más
        # tarde (Ronda 42, 2026-09-17: fix de "decae") sin re-indexar de cero
        # (operación de horas, no la hace este script). `build_sources_data`/
        # `replace_votos_line` (vectorial/generacion.py)
        # leen justo esta clave como texto "de verdad" para sustituir la
        # línea de Votos del LLM -- si se queda obsoleta, el fix de "decae"
        # nunca llega a la respuesta real del RAG vectorial aunque el campo
        # derivado `resultado` de aquí abajo sí esté al día. Se sobrescribe
        # con el valor ya corregido de proposals_enriched.jsonl: dentro de UN
        # mismo segmento todos los chunks comparten idénticamente el mismo
        # vote_result (se asigna una vez por segmento, no por chunk), así que
        # no hay pérdida de precisión frente al original -- mismo riesgo
        # residual ya aceptado para `resultado` en los ~25/3422 casos de
        # desalineación segmento/chunk documentados y sin arreglar.
        meta["vote_result"] = r.get("vote_result") or None

        vf, vc = parse_votos(r.get("vote_result", ""))
        if vf is not None:
            meta["votos_favor"] = vf
        if vc is not None:
            meta["votos_contra"] = vc

        grupo = extrae_grupo(r.get("topic", ""), textmap.get(r["id"], ""))
        grupo = canon_grupo(grupo)
        if grupo == "Desconocido" and r.get("grupo") and r["grupo"] != "Desconocido":
            grupo = canon_grupo(r["grupo"])
        meta["grupo_proponente"] = grupo

        tp = norm_label(r.get("tema_principal") or "otros")
        tp_uri = find_canonical(tp, canon)
        tema_principal_label = label_of(tp_uri) if tp_uri else tp
        meta["tema_principal"] = tema_principal_label

        tops = {top_slug(tp_uri)} if tp_uri else set()
        temas_labels = []
        for t in (r.get("temas") or [])[:4]:
            if not isinstance(t, str) or not t.strip():
                continue
            t_uri = find_canonical(norm_label(t), canon)
            if t_uri:
                tops.add(top_slug(t_uri))
            lbl = label_of(t_uri) if t_uri else None
            if lbl and lbl != tema_principal_label and lbl not in temas_labels:
                temas_labels.append(lbl)
        if temas_labels:
            meta["temas"] = ", ".join(temas_labels)

        tops.discard(None)
        for slug in TOP_SLUGS:
            meta[f"tf_{slug}"] = True if slug in tops else None  # None = borra la clave en el merge

        derived_by_key[join_key(r["date"], r["topic"])] = meta

    print(f"[*] {len(recs)} proposiciones enriquecidas -> {len(derived_by_key)} claves de union")
    return derived_by_key


def patch_chroma(derived_by_key: dict, dry_run: bool):
    rag = RAGPipeline()
    rag.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=rag.embeddings)
    col = rag.vector_store._collection
    total = col.count()
    print(f"[*] {total} chunks en ChromaDB")

    hit = miss = general = 0
    offset = 0
    while offset < total:
        res = col.get(limit=BATCH, offset=offset, include=["metadatas"])
        ids_batch, metas_batch = [], []
        for cid, meta in zip(res["ids"], res["metadatas"]):
            date = meta.get("date")
            topic = meta.get("topic", "")
            if not date or not topic:
                miss += 1
                continue
            if topic.strip().lower().startswith("general"):
                general += 1
                continue
            derived = derived_by_key.get(join_key(date, topic))
            if derived is None:
                miss += 1
                continue
            hit += 1
            patch = dict(derived)
            # NO usar como fallback el vote_result propio del chunk cuando el
            # registro enriquecido no trae votos: un segmento largo (una misma
            # proposición) puede contener VARIAS votaciones intermedias (p.ej.
            # sobre una enmienda) antes de la votación final, y cada chunk se
            # queda con la más cercana en su propio texto — verificado con un
            # caso real (28-03-2012, punto 13): un chunk traía "6 a favor / 23
            # en contra" de una votación sobre consenso previo, mientras que el
            # resultado final real de la proposición fue "27 a favor / 0 en
            # contra". Adjuntar ese número al prop_id de la proposición entera
            # habría sido incorrecto. Se fuerza explícitamente a None (borra la
            # clave si ya existiera de una ejecución anterior) en vez de dejarlo
            # sin definir, para no dejar valores parciales de una pasada previa.
            patch.setdefault("votos_favor", None)
            patch.setdefault("votos_contra", None)
            ids_batch.append(cid)
            metas_batch.append(patch)
        if ids_batch and not dry_run:
            col.update(ids=ids_batch, metadatas=metas_batch)
        offset += BATCH
        print(f"\r[*] progreso: {min(offset, total)}/{total}", end="", flush=True)

    print()
    print(f"[+] Hits: {hit} | Miss: {miss} | General/Introduccion (sin match esperado): {general}")
    print(f"[+] Cobertura sobre proposiciones reales: {100*hit/(hit+miss):.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Calcula y muestra cobertura sin escribir en ChromaDB")
    args = parser.parse_args()

    derived_by_key = build_derived_map()
    patch_chroma(derived_by_key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
