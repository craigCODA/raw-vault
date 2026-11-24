import argparse
import importlib
import json
import re
import sqlite3
from contextlib import closing
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from functools import cache
from pathlib import Path
from uuid import uuid4

from response_links import decimal_position, format_response
from vault_execution import holds, substitute
from vault_identity import POOL_HEADER, asset_path, pool_layout, read_scalar
from vocabulary_links import DEFAULT_DATABASE, ROOT, resolve


DEFAULT_MANIFEST = ROOT / "operations.vault"
MODULES = ("vault_language", "vault_world", "vault_execution", "vault_state")
CREATE = (
    """CREATE TABLE operations (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
        module TEXT NOT NULL, operator TEXT NOT NULL, parameters TEXT NOT NULL
    )""",
    """CREATE TABLE operation_cases (
        operation_id INTEGER NOT NULL REFERENCES operations(id), position INTEGER NOT NULL,
        name TEXT NOT NULL, input TEXT NOT NULL, expected TEXT NOT NULL,
        PRIMARY KEY (operation_id, position), UNIQUE (operation_id, name)
    ) WITHOUT ROWID""",
    """CREATE TABLE operation_anchors (
        operation_id INTEGER NOT NULL REFERENCES operations(id), node_id INTEGER NOT NULL REFERENCES nodes(id),
        PRIMARY KEY (operation_id, node_id)
    ) WITHOUT ROWID""",
    """CREATE TABLE operation_dependencies (
        operation_id INTEGER NOT NULL REFERENCES operations(id), dependency_id INTEGER NOT NULL REFERENCES operations(id),
        PRIMARY KEY (operation_id, dependency_id)
    ) WITHOUT ROWID""",
    "CREATE INDEX operation_anchor_reverse ON operation_anchors (node_id, operation_id)",
)


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@cache
def implementations(module):
    if module not in MODULES:
        raise ValueError(f"Unknown operation module {module!r}.")
    return importlib.import_module(module).OPERATORS


def read_declarations(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format"] != "vault-operations/1":
        raise ValueError("Unknown operation manifest format.")
    declarations = []
    for entry in manifest["files"]:
        path = asset_path(manifest_path.parent, entry["path"])
        rows = json.loads(path.read_text(encoding="utf-8"))
        declarations.extend(row | {"module": entry["module"]} for row in rows)
    if sorted(row["id"] for row in declarations) != list(range(1, manifest["operation_count"] + 1)):
        raise ValueError("The declarations must include every numbered operation exactly once.")
    return declarations


def matches_expected(output, expected):
    if isinstance(expected, dict):
        return isinstance(output, dict) and all(
            key in output and matches_expected(output[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(output, list) and len(output) == len(expected) and all(
            matches_expected(item, value) for item, value in zip(output, expected, strict=True)
        )
    return type(output) is type(expected) and output == expected


class Vault:
    def __init__(self, db, root):
        self.db = db
        self.root = Path(root).resolve()
        self.trace = []
        self.operations_run = 0
        self.bytes_read = 0
        self._active = set()

    def catalog(self):
        names = self.db.execute("SELECT id, name, title FROM operations ORDER BY id")
        anchors = {}
        for operation_id, word in self.db.execute("""
            SELECT a.operation_id, n.key FROM operation_anchors a JOIN nodes n ON n.id = a.node_id
            ORDER BY a.operation_id, n.key
        """):
            anchors.setdefault(operation_id, []).append(word)
        return [{"id": i, "name": name, "title": title, "anchors": anchors.get(i, [])}
                for i, name, title in names]

    def lookup(self, operation):
        if type(operation) is int or isinstance(operation, str) and operation.isdecimal():
            row = self.db.execute("SELECT * FROM operations WHERE id = ?", (int(operation),)).fetchone()
        else:
            row = self.db.execute("SELECT * FROM operations WHERE name = ?", (operation,)).fetchone()
        if row is None:
            raise ValueError(f"No compiled operation {operation!r}.")
        return row

    def run(self, operation, arguments):
        index, name, _, module, operator_name, parameters = self.lookup(operation)
        if index in self._active:
            raise ValueError(f"Recursive operation call at {name!r}.")
        execute = implementations(module)[operator_name]
        self._active.add(index)
        self.operations_run += 1
        self.bytes_read += len(parameters.encode("utf-8"))
        inputs = deepcopy(arguments)
        try:
            result = execute(arguments, json.loads(parameters), self)
        finally:
            self._active.remove(index)
        self.trace.append({"operation": index, "name": name, "input": inputs, "result": deepcopy(result)})
        return result

    def run_address(self, address, instructions):
        offset = decimal_position(str(address["position"]))
        page = address_page(self.db, address.get("links", 1), address.get("width", 1), offset, 1)["page"]
        if not page:
            raise ValueError("No chain at that position.")
        chain = page[0]
        if len(chain["hops"]) != len(instructions):
            raise ValueError("An addressed execution needs one input specification per link.")
        steps = []
        for hop, instruction in zip(chain["hops"], instructions, strict=True):
            scope = hop | {"results": [step["result"] for step in steps]}
            if "when" in instruction and not holds(instruction["when"], scope):
                return {"status": "waiting", "address": chain["address"], "steps": steps}
            arguments = substitute(instruction["arguments"], scope)
            steps.append(hop | {"result": self.run(hop["operation"], arguments)})
        return {"status": "completed", "address": chain["address"], "steps": steps}

    def read_asset(self, relative):
        payload = asset_path(self.root, relative).read_bytes()
        self.bytes_read += len(payload)
        return payload

    def resolve(self, selector):
        return resolve(self.db, selector)

    def for_key(self, selector):
        node_id, key = self.resolve(selector)
        rows = self.db.execute("""
            SELECT o.id, o.name, o.title FROM operation_anchors a
            JOIN operations o ON o.id = a.operation_id WHERE a.node_id = ? ORDER BY o.id
        """, (node_id,))
        return {"key": key, "operations": [{"id": i, "name": name, "title": title} for i, name, title in rows]}

    def operand(self, spec):
        if "value" in spec:
            return spec["value"]
        if "scalar" in spec:
            path = self.root / "raw" / "values.bin"
            value = read_scalar(path, spec["scalar"])
            width = 4 if value["type"] == "float32" else 2
            if value["type"] == "decimal":
                width = pool_layout(path).decimal_width
            self.bytes_read += POOL_HEADER.size + width
            if value["type"] == "float32":
                return str(Fraction(float(value["value"])))
            return value["literal"] if "literal" in value else value["value"]
        key = packed(spec["record"])
        encoding, position = spec["encoding"], spec.get("index", 0)
        if encoding != "decimal" or type(position) is not int or position < 0:
            raise ValueError("A decimal record operand needs a nonnegative index.")
        records = self.db.execute("""
            SELECT value FROM data_records WHERE key = ? AND encoding = ? LIMIT 2
        """, (key, encoding)).fetchall()
        if len(records) != 1:
            raise ValueError(f"Expected one decimal record for {key}.")
        value = records[0][0]
        self.bytes_read += len(value)
        fields = value.split(b"\0")
        if position >= len(fields):
            raise ValueError(f"Decimal record index {position} is out of range.")
        return fields[position].decode("ascii")

    def check(self, operation=None):
        selected = self.lookup(operation)[0] if operation is not None else None
        rows = self.db.execute("""
            SELECT operation_id, name, input, expected FROM operation_cases
            WHERE ? IS NULL OR operation_id = ? ORDER BY operation_id, position
        """, (selected, selected)).fetchall()
        results = []
        for index, name, arguments, expected in rows:
            self.trace.clear()
            actual = self.run(index, json.loads(arguments))
            expect = json.loads(expected)
            if not matches_expected(actual, expect):
                raise ValueError(f"Operation {index} case {name!r} expected {packed(expect)}, got {packed(actual)}")
            results.append({"operation": index, "case": name, "passed": True})
        return {"passed": len(results), "operations": len({row["operation"] for row in results}),
                "cases": results}


def install(database, root, declarations):
    database = Path(database).resolve()
    declarations = list(declarations)
    with closing(sqlite3.connect(database.as_uri() + "?mode=rw", uri=True, timeout=30)) as db, db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("BEGIN IMMEDIATE")
        for table in ("operation_dependencies", "operation_anchors", "operation_cases", "operations"):
            db.execute(f"DROP TABLE IF EXISTS {table}")
        for definition in CREATE:
            db.execute(definition)
        for spec in declarations:
            name, index = spec["name"], spec["id"]
            if type(index) is not int or index < 1 or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
                raise ValueError("Each operation needs a positive ID and an address composed of words.")
            module = spec.get("module", "vault_execution")
            operator_name = spec["operator"]
            if operator_name not in implementations(module) or not spec["cases"] or not spec["anchors"]:
                raise ValueError(f"Operation {index} needs an implementation, vocabulary anchors, and cases.")
            db.execute("INSERT INTO operations VALUES (?, ?, ?, ?, ?, ?)",
                       (index, name, spec["title"], module, operator_name, packed(spec.get("parameters", {}))))
            db.executemany("INSERT INTO operation_cases VALUES (?, ?, ?, ?, ?)", [
                (index, position, case["name"], packed(case["input"]), packed(case["expect"]))
                for position, case in enumerate(spec["cases"])
            ])
            db.executemany("INSERT INTO operation_anchors VALUES (?, ?)", [
                (index, node) for node in sorted({resolve(db, anchor)[0] for anchor in spec["anchors"]})
            ])
        db.executemany("INSERT INTO operation_dependencies VALUES (?, ?)", [
            (row["id"], dependency) for row in declarations for dependency in row.get("requires", [])
        ])
        vault = Vault(db, root)
        checked = vault.check()
        return {"operations": checked["operations"], "passed": checked["passed"],
                "anchors": db.execute("SELECT COUNT(*) FROM operation_anchors").fetchone()[0]}


def address_page(db, links, width, offset, limit):
    if min(links, width, limit) < 1 or offset < 0:
        raise ValueError("Choose positive counts and a nonnegative offset.")
    count, first, last = db.execute("SELECT COUNT(*), MIN(id), MAX(id) FROM nodes").fetchone()
    relations = [{"name": name, "roles": json.loads(definition)["roles"]}
                 for name, definition in db.execute("SELECT name, definition FROM reasoning_relations ORDER BY name")]
    operations = [index for index, in db.execute("SELECT id FROM operations ORDER BY id")]
    if not count or count != last - first + 1 or not relations or not operations:
        raise ValueError("Compile the vocabulary, relations, and operations before paging.")
    bases = [count] * width + ([len(relations), len(operations)] + [count] * width) * links
    combinations = count ** (width * (links + 1)) * (len(relations) * len(operations))**links
    if offset > combinations:
        raise ValueError("Offset exceeds the number of combinations.")

    @cache
    def key_at(digit):
        return db.execute("SELECT key FROM nodes WHERE id = ?", (first + digit,)).fetchone()[0]

    page = []
    for position in range(offset, min(offset + limit, combinations)):
        digits, ordinal = [], position
        for base in reversed(bases):
            ordinal, digit = divmod(ordinal, base)
            digits.append(digit)
        digits.reverse()
        left = [key_at(digit) for digit in digits[:width]]
        parts = [format_response(left)[2:-1]]
        hops = []
        for start in range(width, len(digits), width + 2):
            relation = relations[digits[start]]
            index = operations[digits[start + 1]]
            right = [key_at(digit) for digit in digits[start + 2:start + 2 + width]]
            parts.extend(["{" + relation["name"] + "}", "{operation:" + str(index) + "}", format_response(right)[2:-1]])
            hops.append({"left": left, "right": right, "relation": relation, "operation": index})
            left = right
        text = str(Decimal(position))
        page.append({"position": text, "link": " ".join(parts),
                     "address": {"position": text, "links": links, "width": width}, "hops": hops})
    return {"combinations": str(Decimal(combinations)), "keys": count, "relations": len(relations),
            "operations": len(operations), "links": links, "width": width, "page": page}


def save_result(path, result):
    path = Path(path).resolve()
    draft = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        draft.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        draft.replace(path)
    finally:
        draft.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Compile and execute the Vault's numbered operations.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="compile the declarations and verify their cases")
    commands.add_parser("list", help="list the compiled operations and vocabulary anchors")
    anchored = commands.add_parser("for-key", help="find compiled operations linked to a vocabulary key")
    anchored.add_argument("key")
    check = commands.add_parser("check", help="replay the compiled cases")
    check.add_argument("--operation")
    run = commands.add_parser("run", help="execute an operation named in a data file")
    run.add_argument("datafile", type=Path)
    run.add_argument("--save", type=Path, help="save the result as a data file")
    address = commands.add_parser("address", help="page through combinations of links and operations")
    address.add_argument("--links", type=int, default=1)
    address.add_argument("--width", type=int, default=1)
    address.add_argument("--offset", type=decimal_position, default=0)
    address.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    root = args.manifest.resolve().parent

    try:
        if args.command == "build":
            result = install(args.database, root, read_declarations(args.manifest))
        else:
            uri = args.database.resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as db:
                vault = Vault(db, root)
                if args.command == "run":
                    data = json.loads(args.datafile.read_text(encoding="utf-8"))
                    if "address" in data:
                        result = vault.run_address(data["address"], data["input"])
                    else:
                        result = vault.run(data["operation"], data["input"])
                    if args.save:
                        save_result(args.save, result)
                elif args.command == "check":
                    result = vault.check(args.operation)
                elif args.command == "address":
                    result = address_page(db, args.links, args.width, args.offset, args.limit)
                elif args.command == "for-key":
                    result = vault.for_key(args.key)
                else:
                    result = vault.catalog()
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, ZeroDivisionError,
            argparse.ArgumentTypeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
