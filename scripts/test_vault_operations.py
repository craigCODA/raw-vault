import sqlite3
import unittest
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

from vault_operations import Vault, install
from vault_execution import holds
from vault_language import reply
from vault_world import numeric_feedback
from vocabulary_links import build_index


class CompiledVaultTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent / f".operation-test-{uuid4().hex}"
        root.mkdir()
        self.addCleanup(root.rmdir)
        self.root = root
        self.path = root / "language.sqlite"
        self.addCleanup(self.path.unlink, missing_ok=True)
        build_index(self.path, iter([(word, word, "word") for word in ("2", "3", "state")]))
        self.spec = {
            "id": 59, "name": "parameterized-transitions", "title": "Parameterized transitions",
            "operator": "parameterized_transitions", "anchors": ["state"],
            "parameters": {"expression": "x * 2", "target": "x"},
            "cases": [{"name": "double", "input": {"state": {"x": 3}}, "expect": {"state": {"x": 6}}}],
        }

    def test_compiled_operation_executes_with_the_stored_parameters(self):
        self.assertEqual(install(self.path, self.root, [self.spec])["passed"], 1)
        with closing(sqlite3.connect(self.path)) as db:
            vault = Vault(db, self.root)
            self.assertEqual(vault.run(59, {"state": {"x": 5}})["state"]["x"], 10)
            self.assertEqual(vault.catalog()[0]["anchors"], ["state"])

    def test_a_failed_case_rolls_back_the_compiled_program(self):
        install(self.path, self.root, [self.spec])
        broken = self.spec | {"parameters": {"expression": "x * 3", "target": "x"}}
        with self.assertRaisesRegex(ValueError, "double"):
            install(self.path, self.root, [broken])
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(Vault(db, self.root).run(59, {"state": {"x": 5}})["state"]["x"], 10)

    def test_exact_fraction_survives_json_goal_boundaries(self):
        self.assertTrue(holds({"path": "value", "op": "gt", "value": 0}, {"value": "1/2"}))
        self.assertTrue(holds({"path": "value", "value": "1/2"}, {"value": Fraction(1, 2)}))
        self.assertFalse(holds({"path": "value", "value": "01"}, {"value": "1"}))
        feedback = numeric_feedback(
            {"start": 0, "steps": 4, "expression": "value + 1/2",
             "goal": {"path": "value", "value": "1/2"}},
            {"max_steps": 4}, None,
        )
        self.assertEqual(feedback, {"history": [0, "1/2"], "value": "1/2", "reached": True})

    def test_reply_preserves_a_zero_or_false_conclusion(self):
        formats = {"formats": {"answer": {"fields": ["conclusion"], "template": "{conclusion}"}}}
        for conclusion in (0, False):
            answer = reply({"conclusion": conclusion, "format": "answer", "evidence": {"conclusion": 7}},
                           formats, None)
            self.assertEqual(answer["text"], str(conclusion))

    def test_decoded_address_binds_its_words_and_relation_into_execution(self):
        install(self.path, self.root, [self.spec])
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE reasoning_relations (name TEXT PRIMARY KEY, definition TEXT NOT NULL)")
            db.execute("INSERT INTO reasoning_relations VALUES (?, ?)",
                       ("logic:implies", '{"roles":["condition","result"]}'))
            vault = Vault(db, self.root)
            result = vault.run_address({"position": "1", "links": 1, "width": 1}, [{
                "when": {"path": "relation.roles.0", "value": "condition"},
                "arguments": {"state": {"x": {"$": "left.0"}}},
            }])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["steps"][0]["left"], ["2"])
            self.assertEqual(result["steps"][0]["right"], ["3"])
            self.assertEqual(result["steps"][0]["result"]["value"], 4)


if __name__ == "__main__":
    unittest.main()
