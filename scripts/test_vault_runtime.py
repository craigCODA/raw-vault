import json
import os
import sqlite3
import subprocess
import sys
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vault_identity import POOL_HEADER, POOL_MAGIC
from vault_operations import Vault, install
from vocabulary_links import build_index


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.database = self.root / "language.sqlite"
        records = [(word, word, "word") for word in ("state", "value", "name", "count")]
        build_index(self.database, records)
        raw = self.root / "raw"
        raw.mkdir()
        (raw / "values.bin").write_bytes(
            POOL_HEADER.pack(POOL_MAGIC, 1, 1, 1, 5)
            + bytes.fromhex("00000080")
            + b"\x07\x00"
            + b"2.50\0"
        )
        declarations = [
            {
                "id": 55, "name": "execute", "title": "Execute",
                "operator": "executable_links", "parameters": {"max_steps": 128},
                "anchors": ["state"],
                "cases": [{"name": "empty", "input": {}, "expect": {"status": "stopped"}}],
            },
            {
                "id": 56, "name": "calculate", "title": "Calculate",
                "operator": "arithmetic_dependencies", "anchors": ["value"],
                "cases": [{
                    "name": "read an exact decimal",
                    "input": {"expression": "amount * 2", "operands": {"amount": {"scalar": 2}}},
                    "expect": {"value": 5},
                }],
            },
            {
                "id": 59, "name": "increase", "title": "Increase",
                "operator": "parameterized_transitions",
                "parameters": {"expression": "x + step", "target": "x"},
                "anchors": ["count"],
                "cases": [{
                    "name": "add",
                    "input": {"state": {"x": 1}, "values": {"step": 2}},
                    "expect": {"value": 3},
                }],
            },
            {
                "id": 80, "name": "measure", "title": "Measure",
                "module": "vault_state", "operator": "measure_the_work", "anchors": ["value"],
                "cases": [{
                    "name": "measure a calculation",
                    "input": {"operation": 56, "input": {
                        "expression": "amount * 2", "operands": {"amount": {"scalar": 2}},
                    }},
                    "expect": {"result": {"value": 5}},
                }],
            },
        ]
        install(self.database, self.root, declarations)

    def test_trace_persists_exact_values_and_replays_from_backlinks(self):
        from vault_trace import TraceStore, capture, replay

        program = {"operation": 80, "input": {"operation": 56, "input": {
            "expression": "amount * 2", "operands": {"amount": {"scalar": 2}},
        }}}
        store_path = self.root / "traces.sqlite"
        with closing(sqlite3.connect(self.database)) as db:
            record = capture(Vault(db, self.root), program)
            self.assertEqual(record["result"]["result"]["value"], 5)
            trace_id = TraceStore(store_path).save(record)
            reopened = TraceStore(store_path)
            self.assertEqual(reopened.backlinks("operation/56"), [trace_id])
            self.assertEqual(reopened.backlinks("scalar/2"), [trace_id])
            persisted = reopened.load(trace_id)
            scalar = next(event for event in persisted["execution"] if event["address"] == "scalar/2")
            self.assertEqual(scalar["raw_hex"], "322e353000")
            self.assertEqual(persisted["result"], record["result"])
            self.assertTrue(replay(Vault(db, self.root), persisted)["matched"])

    def test_trace_links_cursor_reads_and_state_writes(self):
        from vault_trace import TraceStore, capture, replay

        program = {
            "operation": 55,
            "input": {
                "event": {"kind": "message", "text": "increase"},
                "state": {"x": 5, "step": 2},
                "cursor": "count",
                "links": {"count": {
                    "operation": 59,
                    "input": {"values": {"step": {"$": "state.step"}}, "state": {"x": {"$": "state.x"}}},
                    "write": {"x": {"$": "result.value"}},
                }},
            },
        }
        with closing(sqlite3.connect(self.database)) as db:
            trace = capture(Vault(db, self.root), program)
        store = TraceStore(self.root / "traces.sqlite")
        trace_id = store.save(trace)
        self.assertEqual(trace["result"]["state"]["x"], 7)
        self.assertEqual(store.backlinks("link/count"), [trace_id])
        self.assertEqual(store.backlinks("state/x"), [trace_id])
        self.assertTrue(any(event["kind"] == "state.write" and event["value"] == 7
                            for event in trace["execution"]))
        persisted = store.load(trace_id)
        self.assertEqual(list(persisted["invocation"]["input"]["links"]["count"]["input"]), ["values", "state"])
        with closing(sqlite3.connect(self.database)) as db:
            self.assertTrue(replay(Vault(db, self.root), persisted)["matched"])

    def test_trace_hash_covers_input_key_order(self):
        from vault_trace import TraceStore, canonical, capture

        with closing(sqlite3.connect(self.database)) as db:
            record = capture(Vault(db, self.root), {"operation": 59, "input": {
                "values": {"step": 2}, "state": {"x": 5},
            }})
        store = TraceStore(self.root / "traces.sqlite")
        trace_id = store.save(record)
        with closing(sqlite3.connect(store.path)) as db, db:
            db.execute("UPDATE trace_runs SET record = ? WHERE id = ?", (canonical(record), trace_id))
        with self.assertRaisesRegex(ValueError, "content check"):
            store.load(trace_id)

    def test_replay_detects_different_executable_catalog(self):
        from vault_trace import capture, replay

        program = {"operation": 59, "input": {"state": {"x": 5}, "values": {"step": 2}}}
        with closing(sqlite3.connect(self.database)) as db:
            record = capture(Vault(db, self.root), program)
            stat = self.database.stat()
            db.execute("UPDATE operations SET parameters = ? WHERE id = 59",
                       (json.dumps({"expression": "x * step", "target": "x"}),))
            db.commit()
            os.utime(self.database, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(self.database.stat().st_size, stat.st_size)
            with self.assertRaisesRegex(ValueError, "revision"):
                replay(Vault(db, self.root), record)

    def test_mapped_storage_preserves_result_and_exposes_read_work(self):
        from vault_trace import capture, replay

        program = {"operation": 56, "input": {"expression": "x + y", "operands": {
            "x": {"scalar": 0}, "y": {"scalar": 1},
        }}}
        with closing(sqlite3.connect(self.database)) as db:
            file_run = capture(Vault(db, self.root), program)
            mapped_run = capture(Vault(db, self.root, storage="mmap"), program)
            check = replay(Vault(db, self.root, storage="mmap"), file_run)
        self.assertEqual(file_run["result"], mapped_run["result"])
        self.assertEqual(mapped_run["result"]["value"], 7)
        self.assertEqual(mapped_run["profile"]["storage"], "mmap")
        self.assertGreater(mapped_run["profile"]["sqlite_mapped_bytes"], 0)
        self.assertGreaterEqual(mapped_run["profile"]["logical_reads"], 3)
        self.assertEqual(mapped_run["profile"]["dependent_reads"], 2)
        self.assertGreater(mapped_run["profile"]["primitive_ns"], 0)
        self.assertTrue(check["matched"])

    def test_replay_with_different_allocation_tracer_ownership(self):
        program = {"operation": 80, "input": {"operation": 56, "input": {
            "expression": "amount * 2", "operands": {"amount": {"scalar": 2}},
        }}}
        check = """
import json
import sqlite3
import sys
import tracemalloc
from contextlib import closing

from vault_operations import Vault
from vault_trace import capture, replay

database, root, invocation = sys.argv[1:]
program = json.loads(invocation)
with closing(sqlite3.connect(database)) as db:
    own_trace = capture(Vault(db, root), program)
    tracemalloc.start()
    try:
        external_trace = capture(Vault(db, root), program)
        assert replay(Vault(db, root), own_trace)["matched"]
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()
    assert replay(Vault(db, root), external_trace)["matched"]
    assert not tracemalloc.is_tracing()
"""
        result = subprocess.run([sys.executable, "-B", "-c", check, str(self.database), str(self.root),
                                 json.dumps(program)], cwd=Path(__file__).parent,
                                env=os.environ | {"PYTHONTRACEMALLOC": ""}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_addressed_profile_measures_catalog_and_selected_key_reads(self):
        from vault_trace import capture

        program = {"address": {"position": "9"}, "input": [
            {"arguments": {"state": {"x": 5}, "values": {"step": 2}}},
        ]}
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("CREATE TABLE reasoning_relations (name TEXT PRIMARY KEY, definition TEXT)")
            db.execute("INSERT INTO reasoning_relations VALUES (?, ?)", ("test", '{"roles":["left","right"]}'))
            db.commit()
            record = capture(Vault(db, self.root), program)
        self.assertEqual(record["result"]["steps"][0]["result"]["value"], 7)
        addresses = [event["address"] for event in record["execution"] if event["kind"] == "read"]
        self.assertEqual(addresses[:3], ["index/vocabulary-range", "index/relation-order", "index/operation-order"])
        self.assertEqual(len({address for address in addresses if address.startswith("node/")}), 2)
        self.assertEqual(record["profile"]["logical_reads"], 6)
        self.assertEqual(record["profile"]["dependent_reads"], 3)

    def test_each_storage_profile_sample_reads_input_assets(self):
        from vault_operations import compare_storage

        asset = (self.root / "activation.vault").resolve()
        asset.write_text(json.dumps({
            "format": "vault-activation/1",
            "activation": {"max_terms": 8, "phrase_width": 2, "memo_entries": 4},
            "fallback": "No matching operation.",
            "routes": [],
        }), encoding="utf-8")
        original_read = Path.read_bytes
        reads = []

        def read_bytes(path):
            if path == asset:
                reads.append(path)
            return original_read(path)

        with patch.object(Path, "read_bytes", read_bytes):
            report = compare_storage(self.database, self.root, {
                "turn": {"kind": "message", "text": "unknown"}, "context": {},
            }, 2)
        self.assertTrue(report["matched"])
        self.assertEqual(len(reads), 4)

    def test_next_turn_activates_persisted_state_and_can_be_replayed(self):
        from vault_context import Sessions
        from vault_trace import TraceStore, replay

        (self.root / "activation.vault").write_text(json.dumps({
            "format": "vault-activation/1",
            "activation": {"max_terms": 16, "phrase_width": 2, "memo_entries": 4},
            "fallback": "I need a matching operation.",
            "routes": [
                {
                    "id": "begin",
                    "pattern": r"start count (?P<number>\d+)",
                    "program": {"operation": 59, "input": {
                        "state": {"x": 0}, "values": {"step": {"$": "match.number"}},
                    }},
                    "write": {"x": {"$": "result.value"}},
                    "reply": {"$": "result.value"},
                    "memoize": True,
                },
                {
                    "id": "step",
                    "pattern": r"add (?P<number>\d+) to count",
                    "program": {"operation": 59, "input": {
                        "state": {"x": {"$": "state.x"}},
                        "values": {"step": {"$": "match.number"}},
                    }},
                    "write": {"x": {"$": "result.value"}},
                    "reply": {"$": "result.value"},
                },
            ],
        }), encoding="utf-8")
        store_path = self.root / "turns.sqlite"
        with closing(sqlite3.connect(self.database)) as db:
            first = Sessions(store_path).send(Vault(db, self.root), "session-a", "start count 5")
            second = Sessions(store_path).send(Vault(db, self.root), "session-a", "add 4 to count")
            independent = Sessions(store_path).send(Vault(db, self.root), "session-b", "start count 1")
            cached = Sessions(store_path).send(Vault(db, self.root), "session-a", "start count 5")
            self.assertEqual(first["reply"], 5)
            self.assertEqual(second["reply"], 9)
            self.assertEqual(independent["state"], {"x": 1})
            self.assertEqual(second["state"], {"x": 9})
            self.assertIn("count", [record["key"] for record in second["active"]["keys"]])
            self.assertGreater(second["profile"]["logical_reads"], 0)
            store = TraceStore(store_path)
            self.assertTrue(any(event["kind"] == "memory.hit"
                                for event in store.load(cached["trace_id"])["execution"]))
            self.assertTrue(replay(Vault(db, self.root), store.load(second["trace_id"]))["matched"])
            self.assertIn(second["trace_id"], store.backlinks("activation/step"))

    def test_replay_uses_original_training_revision_after_publication(self):
        from test_vault_training import proposal, source_vault, teacher
        from vault_train import TrainingStore

        source = self.root / "source"
        work = self.root / "work"
        source_vault(source)
        activation = source / "activation.vault"
        definition = json.loads(activation.read_text(encoding="utf-8"))
        definition["activation"]["max_links"] = 8
        activation.write_text(json.dumps(definition), encoding="utf-8")
        training = TrainingStore(work, source / "operations.vault", source / "identity.vault")
        first = training.initialize()
        invocation = self.root / "double.vault"
        invocation.write_text(json.dumps({"operation": "double", "input": {"state": {"x": 3}}}),
                              encoding="utf-8")
        command = [sys.executable, "-B", str(Path(__file__).with_name("vault_operations.py")),
                   "--training", str(work)]
        traced = subprocess.run([*command, "trace", str(invocation)], capture_output=True, text=True)
        self.assertEqual(traced.returncode, 0, traced.stderr)
        record = json.loads(traced.stdout)
        self.assertEqual(record["publication"], first["identity"])
        training.submit(proposal(), teacher("later-conversation"))
        self.assertEqual(training.build()[0]["status"], "accepted")
        self.assertNotEqual(training.current()["identity"], first["identity"])
        turn = subprocess.run([*command, "turn", "rain measure"], capture_output=True, text=True)
        self.assertEqual(turn.returncode, 0, turn.stderr)
        links = json.loads(turn.stdout)["active"]["links"]
        self.assertEqual([(link["left"], link["relation"], link["right"]) for link in links],
                         [("rain measure", "logic:implies", "water")])
        replayed = subprocess.run([*command, "replay", record["trace_id"]], capture_output=True, text=True)
        self.assertEqual(replayed.returncode, 0, replayed.stderr)
        self.assertTrue(json.loads(replayed.stdout)["matched"])
        archived = Path(first["root"]) / "scripts" / "vault_operations.py"
        portable = subprocess.run([sys.executable, "-I", "-B", "-c",
                                   "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
                                   "sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')",
                                   str(archived.parent), str(archived), "--training", str(work),
                                   "replay", record["trace_id"]],
                                  cwd=first["root"], capture_output=True, text=True)
        self.assertEqual(portable.returncode, 0, portable.stderr)
        self.assertTrue(json.loads(portable.stdout)["matched"])


if __name__ == "__main__":
    unittest.main()
