import json
from typing import Iterator, List


# carga un JSONL completo en una lista de dicts
def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# itera un JSONL línea a línea sin cargarlo entero en memoria
def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
