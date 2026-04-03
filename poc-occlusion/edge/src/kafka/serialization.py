import io
import json

import avro.schema
from avro.io import BinaryEncoder, DatumWriter


def load_avro_schema(schema_path: str) -> avro.schema.Schema:
    with open(schema_path, "r") as f:
        return avro.schema.parse(f.read())


def serialize_avro(schema: avro.schema.Schema, datum: dict) -> bytes:
    writer = DatumWriter(schema)
    buf = io.BytesIO()
    encoder = BinaryEncoder(buf)
    writer.write(datum, encoder)
    return buf.getvalue()
