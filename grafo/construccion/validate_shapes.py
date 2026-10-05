"""
Valida el grafo (bilbao_reasoned.ttl) contra las formas SHACL de shapes.ttl.

    python -m grafo.construccion.validate_shapes
"""
import sys

from pyshacl import validate

from comun.rutas import FORMAS_SHACL, GRAFO_TTL

DATA = GRAFO_TTL
SHAPES = FORMAS_SHACL


def main():
    # inference=None: el grafo YA viene razonado por owlrl (build_rdf.py). Pedirle
    # a pyshacl que razone otra vez sería lento y redundante -- las shapes se
    # validan tal cual está el grafo materializado, que es lo que de verdad
    # consulta el GraphRAG (grafo/consulta).
    conforms, results_graph, results_text = validate(
        DATA,
        shacl_graph=SHAPES,
        data_graph_format="turtle",
        shacl_graph_format="turtle",
        inference=None,
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )
    print(results_text)
    n_violations = results_text.count("Constraint Violation in")
    n_warnings = results_text.count("Validation Result in")
    print(f"[*] {n_violations} violacion(es) real(es) -- {n_warnings} aviso(s) "
          f"(huecos de datos conocidos, ver comentarios en shapes.ttl)")
    if conforms:
        print("[+] El grafo cumple todas las shapes de shapes.ttl (sh:Violation).")
    else:
        print(f"[!] El grafo NO cumple todas las shapes -- {n_violations} violacion(es) listadas arriba.")
    return 0 if conforms else 1


if __name__ == "__main__":
    sys.exit(main())
