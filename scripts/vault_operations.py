import argparse
import base64
import hashlib
import importlib
import json
import re
import sqlite3
from contextlib import ExitStack, closing, contextmanager, nullcontext
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from functools import cache
from pathlib import Path
from time import perf_counter_ns
from statistics import median
from uuid import uuid4

from response_links import decimal_position, format_response
from vault_identity import POOL_HEADER, asset_path, open_pool
from vault_machine import Machine
from vault_state_access import pointer_component
from vault_trace import TraceStore, canonical, capture, content_id, file_digest, ordered_json, replay
from vocabulary_links import DEFAULT_DATABASE, ROOT, resolve


DEFAULT_MANIFEST = ROOT / "operations.vault"
DEFAULT_TRACE_STORE = ROOT / ".vault-data" / "traces.sqlite"
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
    def __init__(self, db, root, *, storage="file", publication=None):
        if storage not in ("file", "mmap"):
            raise ValueError("Storage must be 'file' or 'mmap'.")
        self.db = db
        self.root = Path(root).resolve()
        self.machine = Machine(self.root)
        self.storage = storage
        self.publication = publication
        mapped = self.db.execute("PRAGMA mmap_size=1073741824" if storage == "mmap" else "PRAGMA mmap_size").fetchone()
        self.sqlite_mapped_bytes = mapped[0] if mapped else 0
        self.trace = []
        self.operations_run = 0
        self.bytes_read = 0
        self.recorder = None
        self._active = set()
        self._storage_stack = None
        self._scalar_pool = None
        self._read_parent = None

    @contextmanager
    def storage_session(self):
        if self._storage_stack is not None:
            yield
            return
        with ExitStack() as stack:
            self._storage_stack = stack
            try:
                yield
            finally:
                self._storage_stack = None
                self._scalar_pool = None

    def runtime_identity(self):
        scripts = Path(__file__).resolve().parent
        used = ("vault_operations", "vault_trace", "vault_context")
        host = {name: file_digest(scripts / (name + ".py")) for name in used
                if (scripts / (name + ".py")).is_file()}
        return content_id({"host": host, "machine": self.machine.fingerprint()})

    def graph_event(self, event):
        if self.recorder is not None:
            self.recorder.event(event)

    def observe(self, name, value):
        return self.recorder.observe(name, value) if self.recorder is not None else value

    def primitive(self, function, *args, **kwargs):
        if self.recorder is None:
            return function(*args, **kwargs)
        started = perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            self.recorder.primitive_ns += perf_counter_ns() - started

    @contextmanager
    def reading_after(self, sequence):
        prior = self._read_parent
        self._read_parent = sequence
        try:
            yield
        finally:
            self._read_parent = prior

    def previous_read(self):
        return self.recorder.previous_read if self.recorder is not None else None

    def read_event(self, address, byte_count, **details):
        if self.recorder is not None:
            return self.recorder.read(address, byte_count, depends_on=self._read_parent, **details)
        return None

    def read_rows(self, address, query, arguments=()):
        rows = self.primitive(lambda: self.db.execute(query, arguments).fetchall())
        if self.recorder is not None:
            self.read_event(address, len(canonical(rows).encode("utf-8")), rows=rows)
        return rows

    def bind(self, value, scope):
        return self.machine.call_helper("vault_execution", "substitute", value, scope, context=self)

    def condition(self, value, scope):
        return self.machine.call_helper("vault_execution", "holds", value, scope, context=self)

    def run_program(self, program):
        with self.storage_session():
            if "turn" in program:
                from vault_context import execute_turn
                return execute_turn(self, program["turn"], program["context"])
            if "address" in program:
                return self.run_address(program["address"], program["input"])
            return self.run(program["operation"], program["input"])

    def catalog(self):
        names = self.read_rows("index/operation-catalog", "SELECT id, name, title FROM operations ORDER BY id")
        anchors = {}
        for operation_id, word in self.read_rows("index/operation-anchors", """
            SELECT a.operation_id, n.key FROM operation_anchors a JOIN nodes n ON n.id = a.node_id
            ORDER BY a.operation_id, n.key
        """):
            anchors.setdefault(operation_id, []).append(word)
        return [{"id": i, "name": name, "title": title, "anchors": anchors.get(i, [])}
                for i, name, title in names]

    def lookup(self, operation):
        if type(operation) is int or isinstance(operation, str) and operation.isdecimal():
            query, value = "SELECT * FROM operations WHERE id = ?", int(operation)
        else:
            query, value = "SELECT * FROM operations WHERE name = ?", operation
        row = self.primitive(lambda: self.db.execute(query, (value,)).fetchone())
        if row is None:
            raise ValueError(f"No compiled operation {operation!r}.")
        if self.recorder is not None:
            self.read_event(f"operation/{row[0]}", len(canonical(row).encode("utf-8")), row=row)
        return row

    def run(self, operation, arguments):
        index, name, _, module, operator_name, parameters = self.lookup(operation)
        definition_read = self.previous_read()
        if index in self._active:
            raise ValueError(f"Recursive operation call at {name!r}.")
        body = self.machine.body_address(module, operator_name)
        self._active.add(index)
        self.operations_run += 1
        self.bytes_read += len(parameters.encode("utf-8"))
        inputs = deepcopy(arguments)
        parameters = json.loads(parameters)
        address = f"operation/{index}"
        self.graph_event({"kind": "call", "address": address, "target": body,
                          "input": inputs, "parameters": parameters})
        started = perf_counter_ns() if self.recorder is not None else None
        try:
            with self.reading_after(definition_read):
                result = self.machine.execute(module, operator_name, arguments, parameters, self)
            if self.recorder is not None:
                self.record_state_changes(inputs, result)
                self.graph_event({"kind": "return", "address": address, "result": result})
        finally:
            self._active.remove(index)
            if started is not None:
                self.recorder.operation_time(address, perf_counter_ns() - started)
        self.trace.append({"operation": index, "name": name, "input": inputs, "result": deepcopy(result)})
        return result

    def record_state_changes(self, inputs, result):
        if not isinstance(result, dict):
            return
        after = result.get("state")
        before = inputs.get("state", {}) if isinstance(inputs, dict) else {}
        if not isinstance(after, dict) or not isinstance(before, dict):
            return
        for key in sorted(before.keys() - after.keys()):
            self.graph_event({"kind": "state.delete", "address": "state/" + pointer_component(key),
                              "value": before[key]})
        for key, value in sorted(after.items()):
            if key not in before or ordered_json(value) != ordered_json(before[key]):
                self.graph_event({"kind": "state.write", "address": "state/" + pointer_component(key),
                                  "value": value})

    def run_address(self, address, instructions):
        offset = decimal_position(str(address["position"]))
        page = address_page(self.db, address.get("links", 1), address.get("width", 1), offset, 1, vault=self)["page"]
        if not page:
            raise ValueError("No chain at that position.")
        chain = page[0]
        if len(chain["hops"]) != len(instructions):
            raise ValueError("An addressed execution needs one input specification per link.")
        steps = []
        for hop, instruction in zip(chain["hops"], instructions, strict=True):
            scope = hop | {"results": [step["result"] for step in steps]}
            self.graph_event({"kind": "link.follow", "address": f"address/{offset}/{len(steps)}",
                              "target": f"operation/{hop['operation']}", "hop": hop})
            if "when" in instruction and not self.condition(instruction["when"], scope):
                return {"status": "waiting", "address": chain["address"], "steps": steps}
            arguments = self.bind(instruction["arguments"], scope)
            with self.reading_after(self.previous_read()):
                result = self.run(hop["operation"], arguments)
            steps.append(hop | {"result": result})
        return {"status": "completed", "address": chain["address"], "steps": steps}

    def read_asset(self, relative):
        path = asset_path(self.root, relative)
        address = "asset/" + path.relative_to(self.root).as_posix()
        if self.recorder is not None and self.recorder.restore_assets:
            payload = base64.b64decode(self.recorder.observe(address, None, asset=True), validate=True)
        else:
            payload = self.primitive(path.read_bytes)
            if self.recorder is not None:
                self.recorder.observe(address, base64.b64encode(payload).decode("ascii"), asset=True)
        self.read_event(address, len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.bytes_read += len(payload)
        return payload

    def resolve(self, selector):
        node_id, key = self.primitive(resolve, self.db, selector)
        self.read_event(f"node/{node_id}", len(key.encode("utf-8")), key=key)
        return node_id, key

    def for_key(self, selector):
        node_id, key = self.resolve(selector)
        with self.reading_after(self.previous_read()):
            rows = self.read_rows(f"anchors/{node_id}", """
                SELECT o.id, o.name, o.title FROM operation_anchors a
                JOIN operations o ON o.id = a.operation_id WHERE a.node_id = ? ORDER BY o.id
            """, (node_id,))
        return {"key": key, "operations": [{"id": i, "name": name, "title": title} for i, name, title in rows]}

    def scalar_operand(self, index):
        with self.storage_session():
            if self._scalar_pool is None:
                self._scalar_pool = self.primitive(
                    self._storage_stack.enter_context,
                    open_pool(self.root / "raw" / "values.bin", mapped=self.storage == "mmap"),
                )
            value, raw = self.primitive(self._scalar_pool.scalar, index)
        self.bytes_read += POOL_HEADER.size + len(raw)
        self.read_event(f"scalar/{value['address']}", len(raw), raw_hex=raw.hex(), value=value)
        if value["type"] == "float32":
            return str(Fraction(float(value["value"])))
        return value["literal"] if "literal" in value else value["value"]

    def operand(self, spec):
        if "value" in spec:
            return spec["value"]
        if "scalar" in spec:
            return self.scalar_operand(spec["scalar"])
        key = packed(spec["record"])
        encoding, position = spec["encoding"], spec.get("index", 0)
        if encoding != "decimal" or type(position) is not int or position < 0:
            raise ValueError("A decimal record operand needs a nonnegative index.")
        records = self.primitive(lambda: self.db.execute("""
            SELECT value FROM data_records WHERE key = ? AND encoding = ? LIMIT 2
        """, (key, encoding)).fetchall())
        if len(records) != 1:
            raise ValueError(f"Expected one decimal record for {key}.")
        value = records[0][0]
        self.bytes_read += len(value)
        if self.recorder is not None:
            self.read_event("record/" + key, len(value), encoding=encoding, raw_hex=value.hex())
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
        vault = Vault(db, root)
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
            if not vault.machine.has(module, operator_name) or not spec["cases"] or not spec["anchors"]:
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
        checked = vault.check()
        return {"operations": checked["operations"], "passed": checked["passed"],
                "anchors": db.execute("SELECT COUNT(*) FROM operation_anchors").fetchone()[0]}


def address_page(db, links, width, offset, limit, *, vault=None):
    if min(links, width, limit) < 1 or offset < 0:
        raise ValueError("Choose positive counts and a nonnegative offset.")

    def read(address, query, arguments=()):
        if vault is not None:
            return vault.read_rows(address, query, arguments)
        return db.execute(query, arguments).fetchall()

    (count, first, last), = read("index/vocabulary-range", "SELECT COUNT(*), MIN(id), MAX(id) FROM nodes")
    relations = [{"name": name, "roles": json.loads(definition)["roles"]}
                 for name, definition in read("index/relation-order",
                                              "SELECT name, definition FROM reasoning_relations ORDER BY name")]
    operations = [index for index, in read("index/operation-order", "SELECT id FROM operations ORDER BY id")]
    if not count or count != last - first + 1 or not relations or not operations:
        raise ValueError("Compile the vocabulary, relations, and operations before paging.")
    bases = [count] * width + ([len(relations), len(operations)] + [count] * width) * links
    combinations = count ** (width * (links + 1)) * (len(relations) * len(operations))**links
    if offset > combinations:
        raise ValueError("Offset exceeds the number of combinations.")
    index_read = vault.previous_read() if vault is not None else None

    @cache
    def key_at(digit):
        node = first + digit
        with vault.reading_after(index_read) if vault is not None else nullcontext():
            (key,), = read(f"node/{node}", "SELECT key FROM nodes WHERE id = ?", (node,))
        return key

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


def compare_storage(database, root, program, iterations):
    if iterations < 1:
        raise ValueError("Choose a positive profiling iteration count.")
    uri = database.resolve().as_uri() + "?mode=ro"
    reference = None
    profiles = {}
    for storage in ("file", "mmap"):
        samples = []
        with closing(sqlite3.connect(uri, uri=True)) as db:
            for _ in range(iterations):
                vault = Vault(db, root, storage=storage)
                if reference is None:
                    reference = capture(vault, program)
                    if "error" in reference:
                        raise ValueError(reference["error"]["message"])
                    samples.append(reference["profile"])
                else:
                    samples.append(replay(vault, reference, live_assets=True)["profile"])
        profiles[storage] = {
            "samples": samples,
            "median_elapsed_ns": median(sample["elapsed_ns"] for sample in samples),
        }
    counts = Counter(event["address"] for event in reference["execution"] if event["kind"] == "read")
    return {"matched": True, "revision": reference["revision"], "result": reference["result"],
            "hot_reads": [{"address": address, "reads": count} for address, count in counts.most_common(20)],
            "profiles": profiles}


def main():
    parser = argparse.ArgumentParser(description="Compile and execute the Vault's numbered operations.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--training", type=Path, help="read a published vault from a training directory")
    parser.add_argument("--revision", help="select a published training revision")
    parser.add_argument("--storage", choices=("file", "mmap"), default="file")
    parser.add_argument("--trace-store", type=Path)
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
    tracer = commands.add_parser("trace", help="record an execution and its backlinks")
    tracer.add_argument("datafile", type=Path)
    playback = commands.add_parser("replay", help="verify a recorded execution")
    playback.add_argument("trace_id")
    backlinks = commands.add_parser("backlinks", help="find traces referencing an address")
    backlinks.add_argument("target")
    graph = commands.add_parser("node", help="inspect an addressed operation body or statement")
    graph.add_argument("address")
    turn = commands.add_parser("turn", help="send a message to a saved vault session")
    turn.add_argument("message")
    turn.add_argument("--session", default="default")
    profile = commands.add_parser("profile", help="compare file and mapped storage on the same execution")
    profile.add_argument("datafile", type=Path)
    profile.add_argument("--iterations", type=int, default=3)
    address = commands.add_parser("address", help="page through combinations of links and operations")
    address.add_argument("--links", type=int, default=1)
    address.add_argument("--width", type=int, default=1)
    address.add_argument("--offset", type=decimal_position, default=0)
    address.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    trace_store = args.trace_store or (
        args.training / "traces.sqlite" if args.training is not None else DEFAULT_TRACE_STORE
    )
    store = TraceStore(trace_store)
    status = 0

    try:
        record = store.load(args.trace_id) if args.command == "replay" else None
        publication = None
        if args.training is not None and args.command != "backlinks":
            from vault_train import published_revision

            if args.database != DEFAULT_DATABASE or args.manifest != DEFAULT_MANIFEST:
                raise ValueError("Choose a training revision or an explicit database and manifest.")
            if args.command == "build":
                raise ValueError("Build training revisions through vault_train.py build.")
            selected = args.revision or (record.get("publication") if record is not None else None)
            snapshot = published_revision(args.training, selected)
            if snapshot is None:
                raise ValueError("Initialize the training directory before selecting a revision.")
            args.database, args.manifest = Path(snapshot["database"]), Path(snapshot["manifest"])
            publication = snapshot["identity"]
        elif args.revision is not None:
            raise ValueError("A published revision requires --training.")
        root = args.manifest.resolve().parent
        if args.command == "build":
            result = install(args.database, root, read_declarations(args.manifest))
        elif args.command == "backlinks":
            result = {"target": args.target, "traces": store.backlinks(args.target)}
        elif args.command == "profile":
            data = json.loads(args.datafile.read_text(encoding="utf-8"))
            result = compare_storage(args.database, root, data, args.iterations)
        else:
            uri = args.database.resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as db:
                vault = Vault(db, root, storage=args.storage, publication=publication)
                if args.command in ("run", "trace"):
                    data = json.loads(args.datafile.read_text(encoding="utf-8"))
                    if args.command == "trace":
                        record = capture(vault, data)
                        result = {"trace_id": store.save(record), **record}
                        status = int("error" in record)
                    else:
                        result = vault.run_program(data)
                        if args.save:
                            save_result(args.save, result)
                elif args.command == "replay":
                    result = replay(vault, record)
                elif args.command == "turn":
                    from vault_context import Sessions
                    result = Sessions(trace_store).send(vault, args.session, args.message)
                    status = int("error" in result)
                elif args.command == "check":
                    result = vault.check(args.operation)
                elif args.command == "node":
                    result = {"address": args.address, "node": vault.machine.node(args.address)}
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
    return status


if __name__ == "__main__":
    raise SystemExit(main())
