import argparse
import hashlib
import json
import re
import sqlite3
import struct
from contextlib import closing
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from fractions import Fraction
from mmap import ACCESS_READ, mmap
from os import fstat
from pathlib import Path

from arithmetic import compile_value


POOL_MAGIC = b"VLTNUM01"
POOL_HEADER = struct.Struct("<8sQQQI")
DEFAULT_PROGRAM = Path(__file__).resolve().parents[1] / "identity.vault"


@dataclass(frozen=True)
class PoolLayout:
    float32: int
    uint16: int
    decimal: int
    decimal_width: int

    @property
    def byte_size(self):
        return POOL_HEADER.size + self.float32 * 4 + self.uint16 * 2 + self.decimal * self.decimal_width


def decode_layout(header, size):
    if len(header) != POOL_HEADER.size:
        raise ValueError("The scalar pool header is incomplete.")
    magic, *fields = POOL_HEADER.unpack(header)
    layout = PoolLayout(*fields)
    if magic != POOL_MAGIC or not layout.decimal_width or size != layout.byte_size:
        raise ValueError("The scalar pool does not match its layout.")
    return layout


def pool_layout(path):
    with path.open("rb") as stream:
        return decode_layout(stream.read(POOL_HEADER.size), fstat(stream.fileno()).st_size)


class ScalarPool:
    def __init__(self, stream, mapping, layout):
        self.stream = stream
        self.mapping = mapping
        self.layout = layout

    def read(self, offset, size):
        if self.mapping is not None:
            raw = self.mapping[offset:offset + size]
        else:
            self.stream.seek(offset)
            raw = self.stream.read(size)
        if len(raw) != size:
            raise ValueError("The scalar pool changed during a read.")
        return raw

    def scalar(self, index):
        layout = self.layout
        if not 0 <= index < layout.float32 + layout.uint16 + layout.decimal:
            raise ValueError("Scalar address is out of range.")
        if index < layout.float32:
            raw = self.read(POOL_HEADER.size + index * 4, 4)
            return {
                "address": index,
                "type": "float32",
                "bits": f"{int.from_bytes(raw, 'little'):08x}",
                "value": str(struct.unpack("<f", raw)[0]),
            }, raw
        relative = index - layout.float32
        offset = POOL_HEADER.size + layout.float32 * 4
        if relative < layout.uint16:
            raw = self.read(offset + relative * 2, 2)
            return {"address": index, "type": "uint16", "value": int.from_bytes(raw, "little")}, raw
        relative -= layout.uint16
        offset += layout.uint16 * 2
        raw = self.read(offset + relative * layout.decimal_width, layout.decimal_width)
        literal = raw.rstrip(b"\0").decode("ascii")
        return {"address": index, "type": "decimal", "literal": literal}, raw


@contextmanager
def open_pool(path, *, mapped=False):
    with path.open("rb") as stream:
        layout = decode_layout(stream.read(POOL_HEADER.size), fstat(stream.fileno()).st_size)
        with (mmap(stream.fileno(), 0, access=ACCESS_READ) if mapped else nullcontext()) as mapping:
            yield ScalarPool(stream, mapping, layout)


def read_scalar(path, index):
    with open_pool(path) as pool:
        value, _ = pool.scalar(index)
        return value


def asset_path(directory, name):
    path = (directory / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(directory):
        raise ValueError("Assets must be inside the vault and referenced by relative paths.")
    return path


def read_program(path):
    path = Path(path).resolve()
    program = json.loads(path.read_text(encoding="utf-8"))
    if program["format"] != "vault-identity/1":
        raise ValueError("Unsupported identity data format.")
    return program, asset_path(path.parent, program["values"]), asset_path(path.parent, program["database"])


def sql_name(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError(f"Invalid SQLite identifier {name!r}")
    return f'"{name}"'


def measure(db, columns):
    table, *fields = map(sql_name, columns)
    if len(fields) > 1:
        raise ValueError("A measurement accepts a table and at most one summed column.")
    expression = f"COALESCE(SUM({fields[0]}), 0)" if fields else "COUNT(*)"
    return db.execute(f"SELECT {expression} FROM {table}").fetchone()[0]


def identify(path):
    program, values, database = read_program(path)
    layout = pool_layout(values)
    facts = {
        "float32_values": layout.float32,
        "uint16_values": layout.uint16,
        "decimal_values": layout.decimal,
    }
    identity_rule = program["identity"]
    if identity_rule["operation"] != "sha256" or not identity_rule["inputs"]:
        raise ValueError("An identity rule needs SHA-256 and one or more input assets.")
    assets = {"values": values, "database": database}
    directory = Path(path).resolve().parent
    for name, relative in program.get("assets", {}).items():
        if name in assets:
            raise ValueError(f"Asset {name!r} is already defined.")
        assets[name] = asset_path(directory, relative)
    recipe = json.dumps(program, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(recipe)

    uri = database.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute("BEGIN")
        for key, columns in program["measure"].items():
            facts[key] = measure(db, columns)
        for input_name in identity_rule["inputs"]:
            with assets[input_name].open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())

    bindings = {name: Fraction(value) for name, value in facts.items()}
    for name, expression in program["compute"].items():
        rule = compile_value(expression)
        value = rule.evaluate(bindings)
        bindings[name] = value
        facts[name] = value.numerator if value.denominator == 1 else str(value)
    return {"identity": digest.hexdigest(), "facts": facts}


def main():
    parser = argparse.ArgumentParser(description="Execute the vault's identity data file.")
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--scalar", type=int, help="read a zero-based address in the numeric pool")
    args = parser.parse_args()
    try:
        if args.scalar is None:
            result = identify(args.program)
        else:
            _, values, _ = read_program(args.program)
            result = read_scalar(values, args.scalar)
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
