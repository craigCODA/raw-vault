import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = "vault_execution"


class MachineTests(unittest.TestCase):
    def machine(self, root=ROOT, **options):
        self.assertIsNotNone(importlib.util.find_spec("vault_machine"),
                             "The graph machine must be available")
        from vault_machine import Machine
        return Machine(root, **options)

    def test_shipped_catalog_resolves_every_body(self):
        machine = self.machine()
        manifest = json.loads((ROOT / "operations.vault").read_text())
        operations = [ (entry["module"], row["operator"])
                       for entry in manifest["files"]
                       for row in json.loads((ROOT / entry["path"]).read_text()) ]
        self.assertEqual(len(operations), 80)
        for module, operator in operations:
            self.assertTrue(machine.has(module, operator))
            self.assertTrue(machine.node(machine.body_address(module, operator))["entry"])
        self.assertFalse(machine.has(MODULE, "get_value"))

    def test_fixture_fallback_traverses_stored_dispatch_and_body(self):
        with tempfile.TemporaryDirectory() as root:
            events = []
            machine = self.machine(root, trace=events.append)
            output = machine.execute(MODULE, "goal_conditions", {
                "state": {"level": 3}, "c": {"path": "state.level", "value": 3},
            }, {}, None)
        self.assertEqual(output, {"reached": True})
        calls = [event["address"] for event in events if event["kind"] == "graph.call"]
        for name in ("resolve_body", "bind_call", "complete_call"):
            self.assertIn("body/vault_machine/" + name, calls)
        self.assertIn("body/vault_execution/holds", calls)
        reads = [event for event in events if event["kind"] == "state.read"]
        self.assertEqual([(event["path"], event["value"]) for event in reads], [("state.level", 3)])
        self.assertEqual(reads[0]["address"], "state/level")
        self.assertIn("op", machine.node(reads[0]["source"]))
        for event in events:
            if event["kind"] == "graph.step":
                self.assertIn("op", machine.node(event["address"]))

    def test_cursor_and_each_written_field_have_backlink_addresses(self):
        class Context:
            def run(self, operation, arguments):
                return {"value": arguments["n"] + 1}

        events = []
        self.machine(trace=events.append).execute(MODULE, "executable_links", {
            "state": {"x": 1}, "entry": "first",
            "links": {"first": {"operation": 59, "input": {"n": {"$": "state.x"}},
                                "write": {"a/b": {"$": "result.value"}}}},
        }, {"max_steps": 8}, Context())
        self.assertEqual([event["address"] for event in events if event["kind"] == "link.select"],
                         ["link/first"])
        self.assertEqual([(event["address"], event["value"]) for event in events if event["kind"] == "state.write"],
                         [("state/a~1b", 2)])

    def test_local_body_edit_changes_execution_and_revision(self):
        machine = self.machine()
        args = {"state": "A", "event": "E", "table": {}}
        first = machine.execute(MODULE, "direct_transitions", args, {}, None)
        self.assertEqual(first, {"found": False, "state": "A"})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / "runtime", root / "runtime")
            path = root / "runtime/vault_execution.vault"
            graph = json.loads(path.read_text())
            body = graph["bodies"]["direct_transitions"]
            body["nodes"][body["entry"][0]] = {"op": "Return", "value": {"const": "edited"}}
            path.write_text(json.dumps(graph))
            edited = self.machine(root)
            self.assertNotEqual(machine.fingerprint(), edited.fingerprint())
            self.assertEqual(edited.execute(MODULE, "direct_transitions", args, {}, None), "edited")
            body["nodes"][body["entry"][0]] = {"op": "Import", "names": []}
            path.write_text(json.dumps(graph))
            with self.assertRaises(ValueError):
                self.machine(root)

    def test_reconnecting_a_reflective_edge_changes_real_dispatch(self):
        self.machine()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / "runtime", root / "runtime")
            path = root / "runtime/vault_machine.vault"
            graph = json.loads(path.read_text())
            body = graph["bodies"]["complete_call"]
            body["entry"] = ["alternate"]
            body["nodes"]["alternate"] = {"op": "Return", "value": {"const": 17}}
            path.write_text(json.dumps(graph))
            machine = self.machine(root)
            self.assertEqual(machine.execute(MODULE, "goal_conditions", {
                "state": {}, "c": {"all": []},
            }, {}, None), 17)

    def test_only_bound_host_calls_and_graph_edges_can_execute(self):
        self.machine()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / "runtime", root / "runtime")
            path = root / "runtime/vault_machine.vault"
            graph = json.loads(path.read_text())
            body = graph["bodies"]["complete_call"]
            for target in ({"name": "open"}, {"name": "result"}):
                with self.subTest(target=target):
                    body["nodes"]["s001"] = {"op": "Return", "value": {"op": "Call", "func": target}}
                    path.write_text(json.dumps(graph))
                    with self.assertRaisesRegex(ValueError, "call"):
                        self.machine(root)

    def test_exact_arithmetic_calls_can_be_accounted_by_the_context(self):
        calls = []

        class Context:
            def primitive(self, function, *args, **kwargs):
                calls.append(function.__name__)
                return function(*args, **kwargs)

        result = self.machine().execute(MODULE, "parameterized_transitions", {"state": {"x": "1/3"}},
                                        {"expression": "x * 3", "target": "x"}, Context())
        self.assertEqual(result["value"], 1)
        self.assertIn("compile_value", calls)
        self.assertIn("evaluate", calls)

    def test_address_binding_enters_its_own_machine_trace(self):
        events = []
        machine = self.machine()
        self.assertTrue(hasattr(machine, "call_helper"), "Address binding needs its active graph context")
        class Context:
            graph_event = staticmethod(events.append)
        result = machine.call_helper(MODULE, "substitute", {"$": "left.0"}, {"left": ["oak"]}, context=Context())
        self.assertEqual(result, "oak")
        self.assertIn("scope/left/0", [event["address"] for event in events])

    def test_transition_traces_only_selected_state_inputs_and_its_write(self):
        events = []
        machine = self.machine(trace=events.append)
        state = {"x": 2, "unselected": 7, "step": 11}
        result = machine.execute(MODULE, "parameterized_transitions", {"state": state, "values": {"step": 3}},
                                 {"expression": "x + step", "target": "x"}, None)
        self.assertEqual(result["state"], {"x": 5, "unselected": 7, "step": 11})
        accesses = [(e["kind"], e["address"], e["value"]) for e in events if e["kind"].startswith("state.")]
        self.assertEqual(accesses, [("state.read", "state/x", 2), ("state.write", "state/x", 5)])

    def test_unselected_effect_branch_never_reads_the_alternative_operand(self):
        events = []
        machine = self.machine(trace=events.append)
        machine.execute(MODULE, "ordered_effects", {
            "effects": [{"target": "count", "value": 0}, {"target": "next", "expression": "count + 1"}],
            "state": {"count": 4, "other": "unused"},
        }, {}, None)
        accesses = [(e["kind"], e["address"], e["value"]) for e in events if e["kind"].startswith("state.")]
        self.assertEqual(accesses, [("state.write", "state/count", 0), ("state.read", "state/count", 0),
                                    ("state.read", "state/count", 0), ("state.write", "state/next", 1),
                                    ("state.read", "state/next", 1)])

    def test_state_update_reports_only_events_before_the_time_cutoff(self):
        events = []
        self.machine(trace=events.append).execute("vault_world", "fact_persistence", {
            "state": {"x": 0}, "through": 2, "events": [{"time": 1, "writes": {"x": 2}},
                                                        {"time": 4, "writes": {"later": 3}}],
        }, {}, None)
        writes = [(e["address"], e["value"]) for e in events if e["kind"] == "state.write"]
        self.assertEqual(writes, [("state/x", 2)])

    def test_direct_transition_reads_and_replaces_the_scalar_state(self):
        events = []
        self.machine(trace=events.append).execute(MODULE, "direct_transitions", {
            "state": "waiting", "event": "press", "table": {'["waiting","press"]': {"next": "ready"}},
        }, {}, None)
        states = [(e["kind"], e["address"], e["value"]) for e in events if e["kind"].startswith("state.")]
        self.assertEqual(states, [("state.read", "state", "waiting"), ("state.write", "state", "ready")])


if __name__ == "__main__":
    unittest.main()
