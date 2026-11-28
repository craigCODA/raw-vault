"""Run the shipped Vault examples, commands, and tests against a reference capture."""

import argparse
from contextlib import closing, redirect_stdout
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import runpy
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
import tracemalloc
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASELINE_NAME = "vault-health-baseline.vault"
FORMAT = "vault-health/1"
METRIC_FLOORS = {"elapsed_ns": 500_000_000, "peak_python_bytes": 2 * 1024**2}
OPERATION_METRIC_FLOORS = METRIC_FLOORS | {"elapsed_ns": 1_250_000_000}
CASE_METRIC_FLOORS = {"elapsed_ns": 50_000_000, "peak_python_bytes": 128 * 1024}
METRIC_FACTOR = 3


@dataclass(frozen=True)
class Probe:
    name: str
    command: tuple[str, ...]
    artifact: str | None = None


PROBES = (
    Probe("identity", ("vault_identity.py",)),
    Probe("scalar", ("vault_identity.py", "--scalar", "0")),
    Probe("vocabulary-links", ("vocabulary_links.py", "links", "[[hello]]")),
    Probe("vocabulary-combine", ("vocabulary_links.py", "combine", "[[hello]]", "[[world]]")),
    Probe("vocabulary-markdown", ("vocabulary_links.py", "links", "[[hello]]", "--markdown", "@artifact"),
          "vocabulary.md"),
    Probe("response", ("response_links.py", "respond", "[[hello]]")),
    Probe("response-page", ("response_links.py", "respond", "[[hello]]", "--length", "2", "--limit", "2")),
    Probe("response-reply", ("response_links.py", "respond", "[[hello]]", "--reply", "{@[[world]]}")),
    Probe("reasoning-link", ("reasoning_links.py", "link", "rhetoric:contrast", "[[slow]]", "--right", "[[steady]]")),
    Probe("reasoning-examples", ("reasoning_links.py", "examples", "argument:supports", "--limit", "5")),
    Probe("reasoning-truth", ("reasoning_links.py", "truth", "logic:implies", "1", "0")),
    Probe("example-argument", ("reasoning_links.py", "prove", "example-argument.vault")),
    Probe("reasoning-chain", ("reasoning_links.py", "chain", "--links", "3", "--width", "1",
                              "--offset", "123456789", "--limit", "2")),
    Probe("commonsense-chain", ("reasoning_links.py", "chain", "--family", "commonsense", "--links", "3", "--limit", "2")),
    Probe("all-reasoning-chain", ("reasoning_links.py", "chain", "--family", "all", "--links", "2", "--limit", "2")),
    Probe("example-commonsense", ("commonsense.py", "example-commonsense.vault")),
    Probe("operation-list", ("vault_operations.py", "list")),
    Probe("operation-for-key", ("vault_operations.py", "for-key", "[[state]]")),
    Probe("operation-check", ("vault_operations.py", "check")),
    Probe("operation-check-selected", ("vault_operations.py", "check", "--operation", "76")),
    Probe("operation-address", ("vault_operations.py", "address", "--links", "2", "--width", "1", "--limit", "2")),
    Probe("example-execution", ("vault_operations.py", "run", "example-execution.vault", "--save", "@artifact"),
          "execution.vault"),
    Probe("example-address", ("vault_operations.py", "run", "example-address.vault")),
    Probe("operation-results", ("<compiled cases>",)),
    Probe("unit-tests", ("<unittest discover>",)),
    Probe("vocabulary-reindex", ("vocabulary_links.py", "--database", "@database", "reindex")),
    Probe("response-build", ("response_links.py", "--database", "@database", "build")),
    Probe("reasoning-build", ("reasoning_links.py", "--database", "@database", "build")),
    Probe("operation-build", ("vault_operations.py", "--database", "@database", "build")),
    Probe("operation-check-rebuilt", ("vault_operations.py", "--database", "@database", "check")),
)


def _operation_results(root):
    from vault_operations import Vault

    database = (root / "raw" / "language.sqlite").as_uri() + "?mode=ro"
    outputs = {}
    observations = {}
    with closing(sqlite3.connect(database, uri=True)) as db:
        vault = Vault(db, root)
        for operation, name, arguments in db.execute(
            "SELECT operation_id, name, input FROM operation_cases ORDER BY operation_id, position"
        ).fetchall():
            vault.trace.clear()
            result = vault.run(operation, json.loads(arguments))
            address = f"{operation}:{name}"
            if operation == 80:
                observations[address] = {metric: result.pop(metric) for metric in METRIC_FLOORS}
            outputs[address] = result
    return {"output": outputs, "observations": observations}


class _TestResult(unittest.TextTestResult):
    def __init__(self, *args):
        super().__init__(*args)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def _unit_tests(root):
    with redirect_stdout(sys.stderr):
        suite = unittest.defaultTestLoader.discover(str(root / "scripts"))
        result = unittest.TextTestRunner(resultclass=_TestResult).run(suite)
    print(json.dumps({"passed": sorted(result.passed), "tests_run": result.testsRun}))
    if not result.wasSuccessful():
        raise SystemExit(1)


def _measure(path, script, *arguments):
    root = Path.cwd()
    sys.path.insert(0, str(root / "scripts"))
    tracemalloc.start()
    start = perf_counter_ns()
    try:
        if script == "<compiled cases>":
            print(json.dumps(_operation_results(root), allow_nan=False))
        elif script == "<unittest discover>":
            _unit_tests(root)
        else:
            script_path = str(root / "scripts" / script)
            sys.argv = [script_path, *arguments]
            runpy.run_path(script_path, run_name="__main__")
    finally:
        elapsed = perf_counter_ns() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        Path(path).write_text(json.dumps({"elapsed_ns": elapsed, "peak_python_bytes": peak}), encoding="utf-8")


def run_probe(root, scratch, probe):
    scratch = scratch.resolve()
    paths = {}
    if "@database" in probe.command:
        database = scratch / "language.sqlite"
        if not database.exists():
            source = (root / "raw" / "language.sqlite").as_uri() + "?mode=ro"
            with closing(sqlite3.connect(source, uri=True)) as original, closing(sqlite3.connect(database)) as copy:
                original.backup(copy)
        paths["@database"] = str(database)
    if probe.artifact:
        paths["@artifact"] = str(scratch / probe.artifact)

    measure_path = scratch / f"{probe.name}.metrics"
    result = subprocess.run(
        [sys.executable, "-B", "-c", "import runpy, sys; runpy.run_path(sys.argv[1])['_measure'](*sys.argv[2:])",
         str(Path(__file__).resolve()), str(measure_path), *(paths.get(arg, arg) for arg in probe.command)],
        cwd=root, env=os.environ | {"PYTHONHASHSEED": "0", "PYTHONUTF8": "1", "PYTHONTRACEMALLOC": ""},
        capture_output=True, encoding="utf-8", check=False,
    )
    capture = {"command": list(probe.command)}
    unit_suite = probe.command[0] == "<unittest discover>"
    if measure_path.is_file():
        capture.update(json.loads(measure_path.read_text(encoding="utf-8")))
    if result.returncode:
        capture["error"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        if not unit_suite:
            return capture
    try:
        output = json.loads(result.stdout)
        if paths and isinstance(output, dict):
            tokens = {path: token for token, path in paths.items()}
            output = {key: tokens.get(value, value) if isinstance(value, str) else value
                      for key, value in output.items()}
        if probe.command[0] == "<compiled cases>":
            capture.update(output)
        else:
            capture["output"] = output
        if probe.artifact:
            capture["artifact"] = Path(paths["@artifact"]).read_text(encoding="utf-8")
        if result.stderr and not unit_suite:
            capture["stderr"] = result.stderr
    except (ValueError, OSError) as error:
        capture.setdefault("error", {"message": str(error), "stdout": result.stdout, "stderr": result.stderr})
    return capture


def capture(root):
    """Capture current stable results, test coverage, and probe observations."""
    root = Path(root).resolve()
    for filename in ("raw/language.sqlite", "raw/values.bin"):
        if not (root / filename).is_file():
            raise ValueError(f"Missing asset {root / filename}")
    with TemporaryDirectory(prefix="vault-health-") as folder:
        probes = {probe.name: run_probe(root, Path(folder), probe) for probe in PROBES}
    return {"format": FORMAT, "environment": {
        "implementation": platform.python_implementation(), "python": platform.python_version(),
        "system": platform.system(), "machine": platform.machine(), "sqlite": sqlite3.sqlite_version,
    }, "probes": probes}


def _path(parent, item):
    return f"{parent}/{str(item).replace('~', '~0').replace('/', '~1')}"


def _difference(path, expected, observed, differences, reason="value differs"):
    differences.append({"path": path, "reason": reason, "expected": expected, "observed": observed})


def _compare_values(expected, observed, path, differences):
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(expected.keys() | observed.keys()):
            field = _path(path, key)
            if key not in expected:
                _difference(field, None, observed[key], differences, "unexpected field")
            elif key not in observed:
                _difference(field, expected[key], None, differences, "missing field")
            else:
                _compare_values(expected[key], observed[key], field, differences)
    elif isinstance(expected, list) and isinstance(observed, list) and len(expected) == len(observed):
        for index, (before, after) in enumerate(zip(expected, observed, strict=True)):
            _compare_values(before, after, _path(path, index), differences)
    elif (type(expected) is not type(observed) or expected != observed
          or isinstance(expected, float) and expected.hex() != observed.hex()):
        _difference(path, expected, observed, differences)


def _performance(expected, observed, floors, path, skip_reason, differences):
    deltas = {}
    for metric, floor in floors.items():
        initial, current = expected.get(metric), observed.get(metric)
        if type(initial) is not int or type(current) is not int or min(initial, current) < 0:
            _difference(_path(path, metric), initial, current, differences, "missing or invalid measurement")
            continue
        budget = max(initial * METRIC_FACTOR, initial + floor)
        if skip_reason:
            status = skip_reason
        elif current > budget:
            status = "outside budget"
            _difference(_path(path, metric), budget, current, differences, status)
        else:
            status = "passed"
        deltas[metric] = {"baseline": initial, "observed": current, "delta": current - initial,
                          "budget": budget, "status": status}
    return deltas


def compare(baseline, current):
    """Compare one run with the shipped stable outputs and resource budgets."""
    if baseline.get("format") != FORMAT or current.get("format") != FORMAT:
        raise ValueError("Unsupported vault health baseline format")
    before, after = baseline["probes"], current["probes"]
    comparable = baseline["environment"] == current["environment"]
    environment_difference = None if comparable else "environment differs"
    original_tests = set(before.get("unit-tests", {}).get("output", {}).get("passed", []))
    passed_tests = set(after.get("unit-tests", {}).get("output", {}).get("passed", []))
    original_cases = before.get("operation-results", {}).get("output", {})
    passed_cases = after.get("operation-results", {}).get("output", {})
    missing_tests = sorted(original_tests - passed_tests)
    differences = []
    performance = {}
    for name in sorted(before.keys() | after.keys()):
        path = _path("/probes", name)
        if name not in before or name not in after:
            _difference(path, "present" if name in before else None, "present" if name in after else None,
                        differences, "probe coverage differs")
            continue
        expected, observed = before[name], after[name]
        if "error" in expected or "error" in observed:
            _difference(_path(path, "error"), expected.get("error"), observed.get("error"), differences, "probe failed")
            continue
        for field in ("command", "stderr", "artifact"):
            _compare_values(expected.get(field), observed.get(field), _path(path, field), differences)
        output_path = _path(path, "output")
        if name == "unit-tests":
            for test in missing_tests:
                _difference(_path(output_path, test), "passed", None, differences, "required test did not pass")
        elif name == "operation-results":
            _compare_values(original_cases, {key: passed_cases[key] for key in original_cases if key in passed_cases},
                            output_path, differences)
        else:
            _compare_values(expected["output"], observed["output"], output_path, differences)
        coverage_changed = (
            name == "unit-tests" and original_tests != passed_tests
            or name == "operation-results" and original_cases.keys() != passed_cases.keys()
        )
        skip_reason = environment_difference or ("coverage differs" if coverage_changed else None)
        command = expected["command"][0]
        floors = OPERATION_METRIC_FLOORS if command in ("vault_operations.py", "<compiled cases>") else METRIC_FLOORS
        performance[name] = _performance(expected, observed, floors, path, skip_reason, differences)
        for case, metrics in expected.get("observations", {}).items():
            metric_path = _path(_path(path, "observations"), case)
            performance[f"{name}/{case}"] = _performance(
                metrics, observed.get("observations", {}).get(case, {}), CASE_METRIC_FLOORS, metric_path,
                environment_difference, differences,
            )
    return {"passed": not differences, "differences": differences, "performance_comparable": comparable,
            "environments": {"baseline": baseline["environment"], "observed": current["environment"]},
            "performance": performance, "coverage": {
                "probes": {"baseline": len(before), "completed": sum("error" not in probe for probe in after.values())},
                "examples": {"baseline": sum(name.startswith("example-") for name in before),
                             "completed": sum(name.startswith("example-") and "error" not in probe
                                              for name, probe in after.items())},
                "compiled_cases": {"baseline": len(original_cases), "passed": len(passed_cases)},
                "unit_tests": {"baseline": len(original_tests), "passed": len(passed_tests)},
                "missing_unit_tests": missing_tests,
            }}


def _print_report(report):
    print(f"Vault health {'PASS' if report['passed'] else 'FAIL'}")
    if "error" in report:
        print(report["error"])
        return
    if "path" in report:
        print(f"Recorded {report['path']}")
        return
    for name, counts in report["coverage"].items():
        if isinstance(counts, dict):
            actual = counts.get("passed", counts.get("completed"))
            print(f"  {name.replace('_', ' ')}  {actual} run, {counts['baseline']} in baseline")
    if not report["performance_comparable"]:
        print("Performance budgets skipped because the interpreter, SQLite, or platform differs from the baseline.")
    for name, metrics in report["performance"].items():
        if metrics:
            deltas = ", ".join(f"{metric} {data['delta']:+,}" for metric, data in metrics.items())
            print(f"  {name}  {deltas}")
    for row in report["differences"]:
        print(f"{row['path']}  {row['reason']}")
        print(f"  expected {json.dumps(row['expected'])}")
        print(f"  observed {json.dumps(row['observed'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="vault checkout to exercise")
    parser.add_argument("--baseline", type=Path, help="reference capture, relative to the current working directory")
    parser.add_argument("--record", action="store_true", help="write a new capture to a file that does not exist")
    parser.add_argument("--json", action="store_true", help="print a machine-readable result")
    args = parser.parse_args()
    path = args.baseline or args.root / BASELINE_NAME
    try:
        if args.record:
            if path.exists():
                raise FileExistsError(f"Baseline already exists at {path}")
            current = capture(args.root)
            report = compare(current, current)
            if report["passed"]:
                with path.open("x", encoding="utf-8", newline="\n") as output:
                    json.dump(current, output, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
                    output.write("\n")
                report = {"passed": True, "path": str(path), "coverage": report["coverage"]}
        else:
            baseline = json.loads(path.read_text(encoding="utf-8"))
            if baseline.get("format") != FORMAT or set(baseline["probes"]) != {probe.name for probe in PROBES}:
                raise ValueError(f'Baseline {path} has an unsupported format or incomplete probe coverage')
            report = compare(baseline, capture(args.root))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, sqlite3.Error) as error:
        report = {"passed": False, "error": str(error)}
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        _print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
