import json
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from vault_health import Probe, compare, run_probe


SCRIPT = Path(__file__).with_name("vault_health.py")
ENVIRONMENT = {"implementation": "CPython", "python": "3.12.10", "system": "Windows",
               "machine": "AMD64", "sqlite": "3.49.1"}


def snapshot():
    return {
        "format": "vault-health/1",
        "environment": ENVIRONMENT.copy(),
        "probes": {
            "identity": {"command": ["vault_identity.py"], "output": {"identity": "saved", "facts": {"keys": 1}},
                         "elapsed_ns": 12_000_000, "peak_python_bytes": 20_000},
            "operation-results": {"command": ["<compiled cases>"], "output": {
                "1:old": {"value": 3}, "80:measure": {"result": {"value": 8}}},
                "elapsed_ns": 80_000_000, "peak_python_bytes": 30_000,
                "observations": {"80:measure": {"elapsed_ns": 40_000, "peak_python_bytes": 8_000}}},
            "unit-tests": {"command": ["<unittest discover>"], "output": {"passed": ["Example.test_known"]},
                           "elapsed_ns": 250_000_000, "peak_python_bytes": 99_000},
        },
    }


class HealthComparisonTests(unittest.TestCase):
    def test_reports_exact_value_and_type_drift(self):
        baseline = snapshot()
        current = deepcopy(baseline)
        current["probes"]["identity"]["output"]["facts"]["keys"] = True
        current["probes"]["operation-results"]["output"]["1:old"]["value"] = 4
        result = compare(baseline, current)
        self.assertFalse(result["passed"])
        self.assertEqual(
            [(delta["path"], delta["expected"], delta["observed"]) for delta in result["differences"]],
            [("/probes/identity/output/facts/keys", 1, True),
             ("/probes/operation-results/output/1:old/value", 3, 4)],
        )

    def test_signed_zero_is_preserved_in_exact_outputs(self):
        baseline = snapshot()
        baseline["probes"]["operation-results"]["output"]["1:old"]["value"] = -0.0
        current = deepcopy(baseline)
        current["probes"]["operation-results"]["output"]["1:old"]["value"] = 0.0
        result = compare(baseline, current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["differences"][0]["path"], "/probes/operation-results/output/1:old/value")

    def test_highlights_missing_required_unit_tests(self):
        baseline = snapshot()
        current = deepcopy(baseline)
        current["probes"]["unit-tests"]["output"]["passed"] = ["Example.test_addition"]
        result = compare(baseline, current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["coverage"]["missing_unit_tests"], ["Example.test_known"])

    def test_new_unit_and_compiled_cases_preserve_required_coverage(self):
        baseline = snapshot()
        current = deepcopy(baseline)
        current["probes"]["unit-tests"]["output"]["passed"].append("Example.test_addition")
        current["probes"]["unit-tests"]["elapsed_ns"] = 20_000_000_000
        current["probes"]["operation-results"]["output"]["1:extra"] = {"value": 5}
        result = compare(baseline, current)
        self.assertTrue(result["passed"])
        self.assertEqual(result["coverage"]["unit_tests"], {"baseline": 1, "passed": 2})
        self.assertEqual(result["coverage"]["compiled_cases"], {"baseline": 2, "passed": 3})
        self.assertEqual(result["performance"]["unit-tests"]["elapsed_ns"]["status"], "coverage differs")

    def test_checks_probe_budget_and_reports_both_metric_deltas(self):
        baseline = snapshot()
        current = deepcopy(baseline)
        probe = current["probes"]["identity"]
        probe["elapsed_ns"] = 6_000_000_000
        probe["peak_python_bytes"] = 21_000
        result = compare(baseline, current)
        self.assertFalse(result["passed"])
        self.assertIn("/probes/identity/elapsed_ns", {row["path"] for row in result["differences"]})
        self.assertEqual(result["performance"]["identity"]["peak_python_bytes"]["delta"], 1_000)
        self.assertEqual(result["performance"]["identity"]["peak_python_bytes"]["status"], "passed")

    def test_operations_startup_uses_its_own_time_budget(self):
        baseline = snapshot()
        current = deepcopy(baseline)
        current["probes"]["operation-results"]["elapsed_ns"] = 1_000_000_000
        result = compare(baseline, current)
        self.assertTrue(result["passed"])
        self.assertEqual(result["performance"]["operation-results"]["elapsed_ns"]["budget"], 1_330_000_000)

    def test_performance_on_a_different_environment_is_reported_without_a_gate(self):
        baseline = snapshot()
        current = deepcopy(baseline)
        current["environment"]["system"] = "Linux"
        current["probes"]["identity"]["elapsed_ns"] = 6_000_000_000
        result = compare(baseline, current)
        self.assertTrue(result["passed"])
        self.assertFalse(result["performance_comparable"])
        self.assertEqual(result["performance"]["identity"]["elapsed_ns"]["status"], "environment differs")

    def test_case_observations_are_reported_separately_from_exact_outputs(self):
        baseline = snapshot()
        current = deepcopy(baseline)
        current["probes"]["operation-results"]["observations"]["80:measure"]["elapsed_ns"] = 41_000
        result = compare(baseline, current)
        self.assertTrue(result["passed"])
        metric = result["performance"]["operation-results/80:measure"]["elapsed_ns"]
        self.assertEqual(metric["delta"], 1_000)


class HealthCommandTests(unittest.TestCase):
    def test_missing_baseline_has_a_failing_json_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.vault"
            result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--baseline", str(path), "--json"],
                                    capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(result.stdout, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["passed"])
        self.assertIn("missing.vault", report["error"])

    def test_writes_and_captures_artifacts_in_the_scratch_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / "raw").mkdir()
            database = root / "raw" / "language.sqlite"
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("CREATE TABLE demo (entry TEXT)")
            (root / "scripts" / "writer.py").write_text(
                "import json, sqlite3, sys\n"
                "from contextlib import closing\n"
                "from pathlib import Path\n"
                "with closing(sqlite3.connect(sys.argv[1])) as db, db:\n"
                "    db.execute(\"INSERT INTO demo VALUES ('scratch')\")\n"
                "Path(sys.argv[2]).write_text('sentinel\\n', encoding='utf-8')\n"
                "print(json.dumps({'database': str(Path(sys.argv[1]).resolve()), 'note': sys.argv[2]}))\n",
                encoding="utf-8",
            )
            with tempfile.TemporaryDirectory() as scratch:
                work = Path(scratch) / "working"
                work.mkdir()
                result = run_probe(root, work / "..",
                                   Probe("scratch", ("writer.py", "@database", "@artifact"), "demo.md"))
            with closing(sqlite3.connect(database)) as db:
                self.assertEqual(db.execute("SELECT entry FROM demo").fetchall(), [])
        self.assertEqual(result["output"], {"database": "@database", "note": "@artifact"})
        self.assertEqual(result["artifact"], "sentinel\n")
        self.assertGreater(result["elapsed_ns"], 0)
        self.assertGreater(result["peak_python_bytes"], 0)

    def test_reports_passing_test_ids_when_the_suite_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / "scripts" / "test_fixture.py").write_text(
                "import unittest\n"
                "print('discovered')\n"
                "class Fixture(unittest.TestCase):\n"
                "    def test_passes(self):\n"
                "        print('debug')\n"
                "        self.assertEqual(1, 1)\n"
                "    def test_fails(self):\n"
                "        self.assertEqual(1, 2)\n", encoding="utf-8",
            )
            result = run_probe(root, root, Probe("unit-tests", ("<unittest discover>",)))
        self.assertEqual(result["error"]["returncode"], 1)
        self.assertIn("test_fails", result["error"]["stderr"])
        self.assertEqual(result.get("output"), {"passed": ["test_fixture.Fixture.test_passes"], "tests_run": 2})


if __name__ == "__main__":
    unittest.main()
