import io
from pathlib import Path

import avro.schema
from avro.io import BinaryEncoder, DatumWriter


def _resolve_schema_path(schema_path: str) -> Path:
    path = Path(schema_path)
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    parents = Path(__file__).resolve().parents
    if len(parents) > 4:
        repo_root = parents[4]
        repo_candidate = repo_root / path
        if repo_candidate.exists():
            return repo_candidate

    return cwd_candidate

def load_avro_schema(schema_path: str) -> avro.schema.Schema:
    resolved = _resolve_schema_path(schema_path)
    with resolved.open("r", encoding="utf-8") as f:
        return avro.schema.parse(f.read())

def serialize_avro(schema: avro.schema.Schema, datum: dict) -> bytes:
    writer = DatumWriter(schema)
    buf = io.BytesIO()
    encoder = BinaryEncoder(buf)
    writer.write(datum, encoder)
    return buf.getvalue()
