import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4

from reasoning_links import (
    DEFAULT_RULES, build_reasoning_index, chain_page, evaluate_truth, link_entries, load_rules, prove,
)
from vocabulary_links import build_index


class ReasoningLinkTests(unittest.TestCase):
    def setUp(self):
        folder = Path(__file__).resolve().parent / f".reasoning-test-{uuid4().hex}"
        folder.mkdir()
        self.addCleanup(folder.rmdir)
        path = folder / "language.sqlite"
        self.addCleanup(path.unlink, missing_ok=True)
        records = [
            ("BUY OR SELL", "BUY OR SELL", "sequence"),
            ("FINISH BECAUSE READY", "FINISH BECAUSE READY", "sequence"),
            ("FINISH BECAUSE READY", "FINISH BECAUSE READY", "sequence"),
            ("TRIED ALTHOUGH TIRED", "TRIED ALTHOUGH TIRED", "sequence"),
            ("rain", "rain", "word"),
            ("wet", "wet", "word"),
        ]
        build_index(path, iter(records))
        self.rules = load_rules(DEFAULT_RULES)
        build_reasoning_index(path, self.rules)
        self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.addCleanup(self.db.close)

    def test_support_link_reverses_the_because_phrase_into_premise_then_claim(self):
        result = link_entries(self.db, self.rules, "argument:supports", ["[[ready]]"], ["[[finish]]"])
        self.assertEqual(result["link"], "[[ready]] {argument:supports} [[finish]]")
        self.assertEqual(result["evidence"], [{"cue": "because", "example": "FINISH BECAUSE READY", "count": 2}])
        opposite = link_entries(self.db, self.rules, "argument:supports", ["[[finish]]"], ["[[ready]]"])
        self.assertEqual(opposite["evidence"], [])

    def test_address_product_accepts_repeats_and_jumps_to_its_last_position(self):
        keys = [key for key, in self.db.execute("SELECT key FROM nodes ORDER BY id")]
        relations = sorted(self.rules["relations"])
        count = len(keys)**6 * len(relations)**2
        page = chain_page(self.db, self.rules, links=2, width=2, offset=count - 1, limit=3, family="all")
        self.assertEqual(page["combinations"], str(count))
        self.assertEqual(page["chains"][0]["position"], str(count - 1))
        self.assertEqual(page["chains"][0]["statements"], [[keys[-1]] * 2] * 3)
        self.assertEqual(page["chains"][0]["relations"], [relations[-1]] * 2)

    def test_existing_chain_positions_use_the_original_relations_by_default(self):
        original = self.rules | {
            "relations": {name: definition for name, definition in self.rules["relations"].items()
                          if not name.startswith("commonsense:")},
        }
        options = {"links": 2, "width": 1, "offset": 25, "limit": 2}
        self.assertEqual(chain_page(self.db, self.rules, **options), chain_page(self.db, original, **options))

    def test_logic_truth_function_comes_from_the_rule_file(self):
        results = [evaluate_truth(self.rules, "logic:implies", a, b)["result"]
                   for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]]
        self.assertEqual(results, [1, 1, 0, 1])

    def test_proof_derives_c_from_declared_premises(self):
        result = prove(self.rules, {
            "premises": [["logic:implies", "[[rain]]", "[[wet]]"], "[[rain]]"],
            "steps": [{"rule": "modus_ponens", "from": [0, 1]}],
            "c": "[[wet]]",
        })
        self.assertTrue(result["reached_c"])
        self.assertEqual(result["steps"][0]["conclusion"], "[[wet]]")

    def test_support_is_not_a_substitute_for_implication_in_a_proof(self):
        with self.assertRaisesRegex(ValueError, "premises do not match"):
            prove(self.rules, {
                "premises": [["argument:supports", "[[rain]]", "[[wet]]"], "[[rain]]"],
                "steps": [{"rule": "modus_ponens", "from": [0, 1]}],
                "c": "[[wet]]",
            })


if __name__ == "__main__":
    unittest.main()
