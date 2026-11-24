import unittest

from commonsense import evaluate_situation
from reasoning_links import DEFAULT_RULES, load_rules


class CommonSenseTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules(DEFAULT_RULES)
        self.rain = [
            ["raining_at", "[[park]]"],
            ["located_at", "[[path]]", "[[park]]"],
            ["outdoor_surface", "[[path]]"],
            ["walking_surface", "[[path]]"],
        ]

    def test_expectations_can_support_a_second_step(self):
        result = evaluate_situation(self.rules, {"observed": self.rain, "c": ["slippery", "[[path]]"]})
        self.assertEqual(result["status"], "expected")
        self.assertEqual([step["fact"] for step in result["expected"]],
                         [["wet", "[[path]]"], ["slippery", "[[path]]"]])

    def test_cover_blocks_wetting_and_its_downstream_effect(self):
        cover = ["covered", "[[path]]"]
        result = evaluate_situation(self.rules, {
            "observed": [*self.rain, cover],
            "c": ["slippery", "[[path]]"],
        })
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["expected"], [])
        self.assertEqual(result["blocked"][0]["by"], cover)

    def test_observed_negation_overrides_default(self):
        goal = ["wet", "[[path]]"]
        result = evaluate_situation(self.rules, {"observed": [*self.rain, ["logic:not", goal]], "c": goal})
        self.assertEqual(result["status"], "contradicted")
        self.assertEqual(result["expected"], [])

    def test_conflicting_observations_do_not_supply_a_premise(self):
        wet = ["wet", "[[path]]"]
        result = evaluate_situation(self.rules, {
            "observed": [wet, ["logic:not", wet], ["walking_surface", "[[path]]"]],
            "c": ["slippery", "[[path]]"],
        })
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["expected"], [])

    def test_overflow_uses_a_numeric_comparison_with_matching_units(self):
        observations = [
            ["pouring", "[[water]]", "[[cup]]"],
            ["amount", "[[water]]", "300", "ml"],
            ["remaining_capacity", "[[cup]]", "250", "ml"],
        ]
        goal = ["overflows", "[[cup]]"]
        result = evaluate_situation(self.rules, {"observed": observations, "c": goal})
        self.assertEqual(result["status"], "expected")
        self.assertEqual(result["expected"][0]["checks"][0],
                         {"left": "300", "comparison": ">", "right": "250", "holds": True})
        observations[-1][-1] = "oz"
        self.assertEqual(evaluate_situation(self.rules, {"observed": observations, "c": goal})["status"],
                         "unknown")


if __name__ == "__main__":
    unittest.main()
