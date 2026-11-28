import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from vault_context import Sessions
from vault_operations import Vault, install
from vault_trace import capture, replay
from vocabulary_links import build_index


ROOT = Path(__file__).resolve().parents[1]


class SessionProvenanceTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.database = self.root / "catalog.sqlite"
        words = ("recall", "symbol", "meaning", "the panel", "panel", "what", "remember")
        build_index(self.database, iter((word, word, "word") for word in words))
        settings = json.loads((ROOT / "activation.vault").read_text(encoding="utf-8"))
        settings["activation"]["max_links"] = 0
        (self.root / "activation.vault").write_text(json.dumps(settings), encoding="utf-8")
        definitions = json.loads((ROOT / "operations/language.vault").read_text(encoding="utf-8"))
        base = next(row for row in definitions if row["id"] == 33) | {"module": "vault_language"}
        learned = base | {"id": 81, "name": "learned-recall", "anchors": ["recall"],
                          "parameters": base["parameters"] | {"activation": {
                              "phrases": ["recall {subject:text}"], "input": {
                                  "save": {"symbols": {}}, "query": "x", "encounters": [],
                              }, "reply": "new",
                          }}}
        install(self.database, self.root, [base, learned])
        self.sessions = Sessions(self.root / "sessions.sqlite")

    def test_exact_learned_fields_survive_binding_and_copy_in_recall(self):
        with closing(sqlite3.connect(self.database)) as db:
            self.sessions.send(Vault(db, self.root), "demo", "remember the panel is blue")
            self.sessions.send(Vault(db, self.root), "demo", "remember the wheel is red")
            result = self.sessions.send(Vault(db, self.root), "demo", "what is the panel")
            self.assertEqual(result["reply"], "blue")
            recorded = self.sessions.store.load(result["trace_id"])
            fields = [event for event in recorded["execution"]
                      if event["kind"] == "state.read" and event["address"].startswith("state/save/symbols/")]
            self.assertEqual([(field["address"], field["value"]) for field in fields],
                             [("state/save/symbols/the panel", "blue")])
            self.assertTrue(fields[0]["source"].startswith("body/vault_language/encounter/"))
            self.assertIn(result["trace_id"], self.sessions.store.backlinks(fields[0]["address"]))
            self.assertTrue(replay(Vault(db, self.root), recorded)["matched"])

    def test_each_selected_vocabulary_node_links_back_to_the_turn(self):
        with closing(sqlite3.connect(self.database)) as db:
            outcome = self.sessions.send(Vault(db, self.root), "demo", "what is the panel")
        active = outcome["active"]["keys"]
        self.assertTrue(active)
        record = self.sessions.store.load(outcome["trace_id"])
        self.assertEqual([event["address"] for event in record["execution"] if event["kind"] == "activation.node"],
                         [node["address"] for node in active])
        for node in active:
            self.assertIn(outcome["trace_id"], self.sessions.store.backlinks(node["address"]))

    def test_matching_builtin_route_keeps_precedence_over_learned_phrase(self):
        with closing(sqlite3.connect(self.database)) as db:
            self.sessions.send(Vault(db, self.root), "demo", "remember the panel is blue")
            result = self.sessions.send(Vault(db, self.root), "demo", "recall the panel")
        self.assertEqual((result["activation"], result["operation"], result["reply"]), ("recall", 33, "blue"))

    def test_persisted_symbol_order_follows_the_encounters(self):
        with closing(sqlite3.connect(self.database)) as db:
            self.sessions.send(Vault(db, self.root), "demo", "remember zeta is blue")
            self.sessions.send(Vault(db, self.root), "demo", "remember alpha is red")
            third = self.sessions.send(Vault(db, self.root), "demo", "recall zeta")
        self.assertEqual(list(third["state"]["save"]["symbols"]), ["zeta", "alpha"])

    def test_memo_differentiates_ordered_inputs_for_an_ordered_output(self):
        path = self.root / "activation.vault"
        program = json.loads(path.read_text(encoding="utf-8"))
        program["routes"].append({
            "id": "show", "pattern": "show", "memoize": True,
            "program": {"operation": 33, "input": {"save": {"$": "save"}, "encounters": [], "query": "z"}},
            "reply": {"$": "result.save.symbols"},
        })
        path.write_text(json.dumps(program), encoding="utf-8")
        event = {"turn": {"kind": "message", "text": "show"}}
        with closing(sqlite3.connect(self.database)) as db:
            vault = Vault(db, self.root)
            first = capture(vault, event | {"context": {"state": {"save": {"symbols": {"z": 0, "a": 1}}}}})
            ctx = first["result"]["context"] | {"state": {"save": {"symbols": {"a": 1, "z": 0}}}}
            second = capture(vault, event | {"context": ctx})
        self.assertEqual(list(second["result"]["reply"]), ["a", "z"])
        self.assertFalse(any(e["kind"] == "memory.hit" for e in second["execution"]))

    def test_a_declared_condition_and_operation_bind_retrieved_relation_endpoints(self):
        path = self.root / "activation.vault"
        program = json.loads(path.read_text(encoding="utf-8"))
        program["activation"]["max_links"] = 1
        program["routes"].append({
            "id": "neighbor", "pattern": "neighbor of .+",
            "when": {"path": "active.links.0.left", "value": "panel"},
            "program": {"operation": 33, "input": {"save": {"symbols": {"switch": "input control"}},
                                                   "encounters": [], "query": {"$": "active.links.0.right"}}},
            "reply": {"endpoint": {"$": "active.links.0.right"}, "meaning": {"$": "result.meaning"},
                      "selected_key": {"$": "active.keys.0.key"}},
        })
        path.write_text(json.dumps(program), encoding="utf-8")
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("""CREATE TABLE reasoning_evidence (
                relation TEXT, left_key TEXT, right_key TEXT, sequence_id INTEGER,
                cue TEXT, position INTEGER, occurrences INTEGER)""")
            db.execute("INSERT INTO reasoning_evidence VALUES ('uses', 'panel', 'switch', 1, 'uses', 2, 3)")
            db.commit()
            selected = self.sessions.send(Vault(db, self.root), "demo", "neighbor of panel")
            absent = self.sessions.send(Vault(db, self.root), "demo", "neighbor of moonlight")
        self.assertEqual(selected["reply"], {"endpoint": "switch", "meaning": "input control", "selected_key": "panel"})
        self.assertIsNone(absent["activation"])


if __name__ == "__main__":
    unittest.main()
