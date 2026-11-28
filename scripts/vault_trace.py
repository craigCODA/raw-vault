import hashlib
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from time import perf_counter_ns


FORMAT = "vault-trace/2"
TRACE_FORMATS = ("vault-trace/1", FORMAT)
SCHEMA = """
    CREATE TABLE IF NOT EXISTS trace_runs (
        id TEXT PRIMARY KEY,
        revision TEXT NOT NULL,
        record TEXT NOT NULL
    ) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS trace_links (
        target TEXT NOT NULL,
        trace_id TEXT NOT NULL REFERENCES trace_runs(id),
        PRIMARY KEY (target, trace_id)
    ) WITHOUT ROWID;
"""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def ordered_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def payload_id(payload):
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_id(value):
    return payload_id(canonical(value))


def record_id(record):
    return payload_id(ordered_json(record)) if record["format"] == FORMAT else content_id(record)


def event_address(value):
    return "event/" + payload_id(ordered_json(value))


@lru_cache(maxsize=32)
def digest_file(path, signature, header_size):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if file_signature(path, header_size) != signature:
        raise ValueError(f"The file changed while its revision was measured: {path.name}")
    return digest


def file_signature(path, header_size=0):
    stat = path.stat()
    header = b""
    if header_size:
        with path.open("rb") as stream:
            header = stream.read(header_size)
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, header


def file_digest(path, *, header_size=0, live=False):
    signature = file_signature(path, header_size)
    if live or signature[0] < 1024 * 1024:
        return digest_file.__wrapped__(path, signature, header_size)
    return digest_file(path, signature, header_size)


def revision(vault):
    database = Path(vault.db.execute("PRAGMA database_list").fetchone()[2])
    if database.is_file():
        # SQLite's file header changes on commits outside WAL mode. Windows
        # can report identical timestamps for two rapid, same-sized writes.
        database_id = file_digest(database, header_size=100)
        wal = database.with_name(database.name + "-wal")
        if wal.is_file():
            database_id = content_id([database_id, file_digest(wal, live=True)])
    else:
        database_id = content_id(list(vault.db.iterdump()))
    files = [vault.root / name for name in (
        "raw/values.bin", "reasoning.vault", "commonsense.vault", "activation.vault",
    )]
    artifacts = {name.relative_to(vault.root).as_posix(): file_digest(name) for name in files if name.is_file()}
    return content_id({"database": database_id, "artifacts": artifacts, "runtime": vault.runtime_identity()})


class Recorder:
    def __init__(self, version, observations=None, *, live_assets=False):
        self.revision = version
        self.execution = []
        self.observations = []
        self.saved_observations = iter(observations) if observations is not None else None
        self.restore_assets = observations is not None and not live_assets
        self.primitive_ns = 0
        self.logical_reads = 0
        self.dependent_reads = 0
        self.payload_bytes = 0
        self.previous_read = None
        self.operation_times = {}

    def event(self, entry):
        row = deepcopy(entry)
        row["sequence"] = len(self.execution)
        self.execution.append(row)
        return row["sequence"]

    def read(self, address, byte_count, *, depends_on=None, **details):
        sequence = self.event({
            "kind": "read", "address": address, "bytes": byte_count,
            "previous_read": self.previous_read, "depends_on": depends_on, **details,
        })
        self.previous_read = sequence
        self.logical_reads += 1
        self.dependent_reads += int(depends_on is not None)
        self.payload_bytes += byte_count
        return sequence

    def operation_time(self, address, elapsed_ns):
        calls, total = self.operation_times.get(address, (0, 0))
        self.operation_times[address] = calls + 1, total + elapsed_ns

    def observe(self, name, value, *, asset=False):
        if self.saved_observations is not None:
            saved = next(self.saved_observations, None)
            if saved is None or saved["name"] != name:
                raise ValueError(f"Replay observation differs at {name!r}.")
            if not asset or self.restore_assets:
                value = saved["value"]
        self.observations.append({"name": name, "value": deepcopy(value)})
        return deepcopy(value)

    def finish(self):
        if self.saved_observations is not None and next(self.saved_observations, None) is not None:
            raise ValueError("Replay left unused observations.")

    def profile(self, elapsed_ns, vault):
        return {
            "elapsed_ns": elapsed_ns,
            "primitive_ns": self.primitive_ns,
            "interpretation_ns": max(0, elapsed_ns - self.primitive_ns),
            "logical_reads": self.logical_reads,
            "dependent_reads": self.dependent_reads,
            "payload_bytes": self.payload_bytes,
            "storage": vault.storage,
            "sqlite_mapped_bytes": vault.sqlite_mapped_bytes,
            "operations": [
                {"address": address, "calls": calls, "inclusive_ns": elapsed}
                for address, (calls, elapsed) in sorted(
                    self.operation_times.items(), key=lambda item: (-item[1][1], item[0])
                )
            ],
        }


def capture(vault, invocation, *, observations=None, live_assets=False):
    version = revision(vault)
    recorder = Recorder(version, observations, live_assets=live_assets)
    original = deepcopy(invocation)
    if vault.recorder is not None:
        raise ValueError("A vault execution already has a trace recorder.")
    vault.recorder = recorder
    started = perf_counter_ns()
    try:
        with vault.storage_session():
            result = vault.run_program(deepcopy(invocation))
        outcome = {"result": result}
    except Exception as error:
        outcome = {"error": {"type": type(error).__name__, "message": str(error)}}
    finally:
        elapsed_ns = perf_counter_ns() - started
        vault.recorder = None
    recorder.finish()
    if revision(vault) != version:
        raise ValueError("The vault revision changed during execution.")
    event = invocation.get("turn", invocation.get("input"))
    event = event.get("event", event) if isinstance(event, dict) else event
    return {
        "format": FORMAT,
        "revision": version,
        **({"publication": vault.publication} if vault.publication is not None else {}),
        "event": {"address": event_address(event), "value": deepcopy(event)},
        "invocation": original,
        **outcome,
        "execution": recorder.execution,
        "observations": recorder.observations,
        "profile": recorder.profile(elapsed_ns, vault),
    }


def replay(vault, record, *, live_assets=False):
    if record.get("format") != FORMAT or record["revision"] != revision(vault):
        raise ValueError("The trace requires the original executable vault revision.")
    if record.get("publication") != vault.publication:
        raise ValueError("The trace was captured from a different publication.")
    actual = capture(vault, record["invocation"], observations=record["observations"], live_assets=live_assets)
    for key in ("event", "result", "error", "execution", "observations"):
        if ordered_json(actual.get(key)) != ordered_json(record.get(key)):
            raise ValueError(f"Replay differs in {key}.")
    return {"matched": True, "revision": actual["revision"], "profile": actual["profile"]}


class TraceStore:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def writer(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA foreign_keys = ON")
            db.executescript(SCHEMA)
        except BaseException:
            db.close()
            raise
        return db

    @staticmethod
    def insert(db, record):
        if record.get("format") != FORMAT:
            raise ValueError("Unsupported trace format.")
        payload = ordered_json(record)
        trace_id = payload_id(payload)
        targets = {record["event"]["address"], "revision/" + record["revision"]}
        if record.get("publication") is not None:
            targets.add("publication/" + record["publication"])
        for event in record["execution"]:
            targets.update(event[key] for key in ("address", "target") if isinstance(event.get(key), str))
        db.execute("INSERT OR IGNORE INTO trace_runs VALUES (?, ?, ?)",
                   (trace_id, record["revision"], payload))
        db.executemany("INSERT OR IGNORE INTO trace_links VALUES (?, ?)",
                       [(target, trace_id) for target in sorted(targets)])
        return trace_id

    def save(self, record):
        with closing(self.writer()) as db, db:
            return self.insert(db, record)

    def load(self, trace_id):
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute("SELECT record FROM trace_runs WHERE id = ?", (trace_id,)).fetchone()
        if row is None:
            raise ValueError(f"Trace {trace_id!r} was not found.")
        if payload_id(row[0]) != trace_id:
            raise ValueError(f"Trace {trace_id!r} failed its content check.")
        return json.loads(row[0])

    def backlinks(self, target):
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
            return [trace_id for trace_id, in db.execute(
                "SELECT trace_id FROM trace_links WHERE target = ? ORDER BY trace_id", (target,)
            )]
