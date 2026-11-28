import json
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from reasoning_links import build_reasoning_index, link_entries, load_rules
from vault_identity import POOL_HEADER, POOL_MAGIC, identify
from vault_operations import Vault, install, read_declarations
from vocabulary_links import build_index, resolve


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def source_vault(root):
    raw = root / "raw"
    raw.mkdir(parents=True)
    database = raw / "language.sqlite"
    build_index(database, [(key, key, "word") for key in ("state", "water", "seed")])
    (raw / "values.bin").write_bytes(POOL_HEADER.pack(POOL_MAGIC, 0, 1, 0, 1) + b"\x07\x00")
    write(root / "reasoning.vault", {
        "format": "vault-reasoning/1",
        "relations": {"logic:implies": {"roles": ["premise", "conclusion"], "cues": {}}},
    })
    write(root / "activation.vault", {
        "format": "vault-activation/1", "activation": {"max_terms": 32, "phrase_width": 4, "memo_entries": 8},
        "routes": [], "fallback": "Unresolved",
    })
    build_reasoning_index(database, load_rules(root / "reasoning.vault"))
    write(root / "identity.vault", {
        "format": "vault-identity/1", "values": "raw/values.bin", "database": "raw/language.sqlite",
        "identity": {"operation": "sha256", "inputs": ["values", "database"]},
        "measure": {"addressable_keys": ["nodes"]}, "compute": {},
    })
    write(root / "operations.vault", {"format": "vault-operations/1", "operation_count": 1,
                                    "files": [{"path": "baseline.vault", "module": "vault_execution"}]})
    write(root / "baseline.vault", [
        {"id": 1, "name": "double", "title": "Double a number", "operator": "parameterized_transitions",
         "parameters": {"expression": "x * 2", "target": "x"}, "anchors": ["state"],
         "cases": [{"name": "baseline", "input": {"state": {"x": 4}}, "expect": {"value": 8}}]},
    ])
    install(database, root, read_declarations(root / "operations.vault"))


def proposal(name="triple", key="rain measure"):
    return {
        "summary": "Check a typed association and triple a measured amount.",
        "keys": [key],
        "links": [{"relation": "logic:implies", "left": key, "right": "water"}],
        "operation": {
            "name": name, "title": "Triple a measurement", "anchors": ["state", key],
            "module": "vault_execution", "operator": "parameterized_transitions",
            "parameters": {"expression": "x * 3", "target": "x"},
        },
        "examples": [
            {"kind": "positive", "check": "link", "input": {
                "relation": "logic:implies", "left": key, "right": "water",
            }, "expect": True},
            {"kind": "counterexample", "check": "link", "input": {
                "relation": "logic:implies", "left": "water", "right": key,
            }, "expect": False},
            {"kind": "positive", "check": "operation", "input": {"state": {"x": 2}}, "expect": {"value": 6}},
            {"kind": "counterexample", "check": "operation", "input": {"state": {"x": 0}}, "expect": {"value": 0}},
        ],
    }


def teacher(session):
    return {"role": "teacher", "provider": "test", "session": session, "model": "test-model",
            "prompt": "A synthetic lesson", "response": "A synthetic proposal"}


def turn_proposal(reply):
    lesson = proposal()
    lesson["operation"]["parameters"]["activation"] = {
        "phrases": ["rain measure {number:int}"],
        "input": {"state": {"x": {"$": "match.number"}, "labels": ["one label"]}},
        "reply": reply,
    }
    lesson["examples"] += [
        {"kind": "positive", "check": "turn", "input": {"text": "rain measure 5"}, "expect": {"reply": 15}},
        {"kind": "counterexample", "check": "turn", "input": {"text": "rain measure wrong"},
         "expect": {"uses_learned": False}},
    ]
    return lesson


class TrainingTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "source"
        self.work = Path(directory.name) / "work"
        source_vault(self.root)

    def store(self):
        from vault_train import TrainingStore

        return TrainingStore(self.work, self.root / "operations.vault", self.root / "identity.vault")

    def test_publish_appends_addresses_and_keeps_old_snapshot_readable(self):
        from vault_train import published_revision

        store = self.store()
        baseline = store.initialize()
        ticket = store.submit(proposal(), teacher("session-a"), baseline["identity"])
        with closing(sqlite3.connect(baseline["database"])) as old:
            old.execute("BEGIN")
            original_address = resolve(old, "water")
            results = store.build()
            self.assertEqual(results[0]["status"], "accepted")
            self.assertEqual(results[0]["assessment"]["appended"], {"keys": 1, "links": 1, "operation": 2})
            revision = published_revision(self.work)
            self.assertNotEqual(revision["identity"], baseline["identity"])
            self.assertEqual(published_revision(self.work, baseline["identity"]), baseline)
            with self.assertRaisesRegex(ValueError, "published revision"):
                published_revision(self.work, "missing")
            self.assertEqual(identify(Path(revision["root"]) / "identity.vault")["identity"], revision["identity"])
            self.assertEqual(results[0]["proposals"], [ticket])
            self.assertEqual(Vault(old, baseline["root"]).run(1, {"state": {"x": 3}})["value"], 6)
            with self.assertRaises(ValueError):
                resolve(old, "rain measure")
            with closing(sqlite3.connect(revision["database"])) as db:
                self.assertEqual(resolve(db, "water"), original_address)
                self.assertEqual(resolve(db, "rain measure"), (4, "rain measure"))
                vault = Vault(db, revision["root"])
                self.assertEqual(vault.run(2, {"state": {"x": 3}})["value"], 9)
                rules = load_rules(Path(revision["root"]) / "reasoning.vault")
                self.assertTrue(link_entries(db, rules, "logic:implies", ["rain measure"], ["water"])["evidence"])
        self.assertEqual(store.evidence(ticket)["conversation"]["session"], "session-a")
        self.assertEqual(store.evidence(ticket)["revision"], revision["identity"])
        self.assertFalse((self.root / "trained.vault").exists())

    def test_failing_critic_preserves_counterexample_and_published_identity(self):
        store = self.store()
        baseline = store.initialize()
        ticket = store.submit(proposal(), teacher("session-a"), baseline["identity"])

        def critic(_context):
            return {"summary": "Try a nonzero boundary.", "counterexamples": [
                {"kind": "counterexample", "check": "operation", "input": {"state": {"x": -1}},
                 "expect": {"value": 0}},
            ]}, teacher("session-b") | {"role": "critic"}

        results = store.build(critic=critic)
        self.assertEqual(results[0]["status"], "rejected")
        self.assertEqual(store.current(), baseline)
        record = store.evidence(ticket)
        self.assertEqual(record["critic"]["session"], "session-b")
        self.assertEqual(record["assessment"]["example"]["input"], {"state": {"x": -1}})
        self.assertEqual(record["assessment"]["actual"]["value"], -3)
        self.assertEqual(record["proposal"]["operation"]["name"], "triple")

    def test_concurrent_teachers_factor_duplicate_mutations(self):
        store = self.store()
        baseline = store.initialize()
        with ThreadPoolExecutor(max_workers=3) as threads:
            tickets = list(threads.map(
                lambda session: self.store().submit(proposal(), teacher(session), baseline["identity"]),
                ["a", "b", "c"],
            ))
        self.assertEqual(len(set(tickets)), 3)
        results = store.build()
        self.assertEqual(len(results), 1)
        self.assertEqual(set(results[0]["proposals"]), set(tickets))
        self.assertEqual({store.evidence(ticket)["conversation"]["session"] for ticket in tickets}, {"a", "b", "c"})
        with closing(sqlite3.connect(store.current()["database"])) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM operations").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM operation_cases").fetchone()[0], 3)

    def test_stale_evidence_and_untrusted_operator_never_publish(self):
        store = self.store()
        baseline = store.initialize()
        first = store.submit(proposal(), teacher("a"), baseline["identity"])
        self.assertEqual(store.build()[0]["status"], "accepted")
        current = store.current()
        stale = store.submit(proposal("second", "another key"), teacher("b"), baseline["identity"])
        invalid = proposal("third", "new key")
        invalid["operation"]["operator"] = "executable_links"
        disallowed = store.submit(invalid, teacher("c"), current["identity"])
        store.build()
        self.assertEqual(store.evidence(first)["status"], "accepted")
        self.assertEqual(store.evidence(stale)["status"], "rejected")
        self.assertIn("stale", store.evidence(stale)["assessment"]["reason"])
        self.assertEqual(store.evidence(disallowed)["status"], "rejected")
        self.assertIn("allowlisted", store.evidence(disallowed)["assessment"]["reason"])
        self.assertEqual(store.current(), current)

    def test_fixture_round_runs_teachers_and_a_separate_critic(self):
        from vault_train import run_training
        from vault_training_providers import FixtureProvider

        store = self.store()
        fixture = self.work.parent / "fixture.json"
        write(fixture, {"teacher": [proposal(), proposal()],
                        "critic": [{"summary": "The zero boundary holds.", "counterexamples": []}]})
        provider = FixtureProvider(fixture)
        result = run_training(store, provider, provider, teachers=2, rounds=1)
        tickets = result["rounds"][0]["build"][0]["proposals"]
        self.assertEqual(len(tickets), 2)
        evidence = [store.evidence(ticket) for ticket in tickets]
        self.assertEqual({item["status"] for item in evidence}, {"accepted"})
        self.assertEqual(len({item["conversation"]["session"] for item in evidence}), 2)
        self.assertNotIn(evidence[0]["critic"]["session"], {item["conversation"]["session"] for item in evidence})

    def test_transport_failure_is_audited_without_losing_other_proposals(self):
        from vault_train import run_training
        from vault_training_providers import ProviderFailure

        store = self.store()

        def provider(role, index, _context):
            if role == "critic":
                return {"summary": "No additional cases.", "counterexamples": []}, teacher("critique") | {"role": role}
            if index == 0:
                raise ProviderFailure("Provider timeout.", teacher("timed-out"))
            return proposal(), teacher("completed")

        results = run_training(store, provider, provider, teachers=2, rounds=1)
        first, second = results["rounds"][0]["proposals"]
        self.assertEqual(store.evidence(first)["status"], "rejected")
        self.assertIn("Provider timeout.", store.evidence(first)["assessment"]["reason"])
        self.assertEqual(store.evidence(second)["status"], "accepted")

    def test_oversized_model_submission_keeps_provenance_and_completes_round(self):
        from vault_train import run_training

        store = self.store()

        def provider(_role, index, _context):
            if index == 0:
                oversized = "Too large " + "é" * 160_000
                return proposal() | {"summary": oversized}, teacher("model-overflow") | {"response": oversized}
            return proposal(), teacher("model-ok")

        report = run_training(store, provider, None, teachers=2, rounds=1)
        rejected, accepted = (store.evidence(ticket) for ticket in report["rounds"][0]["proposals"])
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("size limit", rejected["assessment"]["reason"])
        self.assertEqual(rejected["conversation"]["session"], "model-overflow")
        self.assertTrue(rejected["conversation"]["response"].startswith("Too large "))
        self.assertGreater(rejected["proposal"]["bytes"], 64 * 1024)
        self.assertEqual(len(rejected["proposal"]["sha256"]), 64)
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(store.status()["rounds"], {"completed": 1})

    def test_future_revision_must_preserve_an_accepted_critic_counterexample(self):
        store = self.store()
        baseline = store.initialize()
        first = store.submit(proposal(), teacher("a"), baseline["identity"])

        def critic(_context):
            return {"summary": "The inference from water to a seed is unproven.", "counterexamples": [
                {"kind": "counterexample", "check": "link", "input": {
                    "relation": "logic:implies", "left": "water", "right": "seed",
                }, "expect": False},
            ]}, teacher("critique") | {"role": "critic"}

        store.build(critic=critic)
        self.assertEqual(store.evidence(first)["status"], "accepted")
        current = store.current()
        other = {"relation": "logic:implies", "left": "water", "right": "seed"}
        second = store.submit({
            "summary": "Add a link challenged by the earlier review.",
            "keys": [], "links": [other], "operation": None,
            "examples": [
                {"kind": "positive", "check": "link", "input": other, "expect": True},
                {"kind": "counterexample", "check": "link", "input": other | {"left": "seed", "right": "water"},
                 "expect": False},
            ],
        }, teacher("b"), current["identity"])
        store.build()
        self.assertEqual(store.evidence(second)["status"], "rejected")
        self.assertEqual(store.evidence(second)["assessment"]["example"]["input"], other)
        self.assertEqual(store.current(), current)

    def test_learned_route_requires_counterexample_through_the_turn_entrypoint(self):
        from vault_training_data import prepare

        lesson = proposal()
        lesson["operation"]["parameters"]["activation"] = {
            "phrases": ["rain measure {number:int}"],
            "input": {"state": {"x": {"$": "match.number"}}},
            "reply": {"$": "result.value"},
        }
        lesson["examples"].append({
            "kind": "positive", "check": "turn", "input": {"text": "rain measure 5"}, "expect": {"reply": 15},
        })
        with self.assertRaisesRegex(ValueError, "positive and counterexample turn checks"):
            prepare(lesson)

    def test_invalid_template_is_rejected_before_executing_a_candidate(self):
        from vault_training_data import prepare

        lesson = turn_proposal({"answer": [{"$": 42}]})
        with self.assertRaisesRegex(ValueError, "binding path"):
            prepare(lesson)

    def test_index_error_in_a_model_turn_settles_and_preserves_its_example(self):
        from vault_train import run_training

        store = self.store()

        def provider(_role, index, _context):
            if index == 0:
                return turn_proposal({"$": "result.state.labels.9"}), teacher("bad-index")
            return proposal("triple-second", "another measure"), teacher("model-ok")

        report = run_training(store, provider, None, teachers=2, rounds=1)
        rejected, accepted = (store.evidence(ticket) for ticket in report["rounds"][0]["proposals"])
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["assessment"]["example"]["input"], {"text": "rain measure 5"})
        self.assertEqual(rejected["assessment"]["actual"]["error"]["type"], "IndexError")
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(store.status()["rounds"], {"completed": 1})

    def test_run_does_not_send_manually_queued_lessons_to_model_critic(self):
        from vault_train import run_training

        store = self.store()
        baseline = store.initialize()
        private = store.submit(proposal("private-lesson", "private key"), teacher("private"), baseline["identity"])
        received = []

        def provider(role, _index, context):
            if role == "critic":
                received.append(context["mutation"]["operation"]["name"])
                return {"summary": "Checked.", "counterexamples": []}, teacher("critic") | {"role": role}
            return proposal(), teacher("public")

        result = run_training(store, provider, provider, teachers=1, rounds=1)
        self.assertEqual(received, ["triple"])
        self.assertEqual(store.evidence(private)["status"], "queued")
        self.assertEqual(result["rounds"][0]["build"][0]["status"], "accepted")

    def test_critic_must_have_a_separate_session(self):
        store = self.store()
        baseline = store.initialize()
        ticket = store.submit(proposal(), teacher("same"), baseline["identity"])

        def critic(_context):
            return {"summary": "Same session.", "counterexamples": []}, teacher("same") | {"role": "critic"}

        store.build(critic=critic)
        self.assertEqual(store.evidence(ticket)["status"], "rejected")
        self.assertEqual(store.current(), baseline)

    def test_candidate_runs_learned_turn_against_pinned_activation_file(self):
        lesson = proposal()
        lesson["operation"]["parameters"]["activation"] = {
            "phrases": ["rain measure {number:int}"],
            "input": {"state": {"x": {"$": "match.number"}}},
            "reply": {"$": "result.value"},
        }
        lesson["examples"] += [
            {"kind": "positive", "check": "turn", "input": {"text": "rain measure 5"}, "expect": {"reply": 15}},
            {"kind": "counterexample", "check": "turn", "input": {"text": "rain measure wrong"},
             "expect": {"uses_learned": False}},
        ]
        store = self.store()
        store.initialize()
        ticket = store.submit(lesson, teacher("route"))
        store.build()
        self.assertEqual(store.evidence(ticket)["status"], "accepted")
        current = store.current()
        with closing(sqlite3.connect(current["database"])) as db:
            actual = Vault(db, current["root"]).run_program(
                {"turn": {"kind": "message", "text": "rain measure 9"}, "context": {}})
        self.assertEqual(actual["activation"], "learned/2")
        self.assertEqual(actual["reply"], 27)

    def test_publication_preserves_matching_built_in_turn(self):
        path = self.root / "activation.vault"
        activation = json.loads(path.read_text(encoding="utf-8"))
        activation["routes"] = [{
            "id": "special-measure", "pattern": "rain measure 1",
            "program": {"operation": 1, "input": {"state": {"x": 7}}},
            "reply": {"$": "result.value"},
        }]
        write(path, activation)
        store = self.store()
        baseline = store.initialize()
        ticket = store.submit(turn_proposal({"$": "result.value"}), teacher("learned-range"))
        store.build()
        self.assertEqual(store.evidence(ticket)["status"], "accepted")
        for revision in (baseline, store.current()):
            with closing(sqlite3.connect(revision["database"])) as db:
                result = Vault(db, revision["root"]).run_program(
                    {"turn": {"kind": "message", "text": "rain measure 1"}, "context": {}})
            self.assertEqual((result["activation"], result["operation"], result["reply"]), ("special-measure", 1, 14))

    def test_duration_stops_starting_rounds_and_streams_progress(self):
        from vault_train import run_training

        now, events, sleeps = [0.0], [], []

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        def provider(role, index, _context):
            if role == "critic":
                return {"summary": "Checked.", "counterexamples": []}, teacher(f"critic-{index}") | {"role": role}
            return proposal(f"triple-{index}", f"rain measure {index}"), teacher(f"teacher-{index}")

        report = run_training(self.store(), provider, provider, teachers=1, rounds=10,
                              duration=5, interval=3, clock=lambda: now[0], sleep=sleep, emit=events.append)
        self.assertEqual([item["round"] for item in report["rounds"]], [1, 2])
        self.assertEqual(sleeps, [3, 2])
        self.assertEqual(report["stop"], "duration")
        self.assertEqual([event["event"] for event in events],
                         ["round_started", "round_finished", "waiting", "round_started", "round_finished", "waiting"])

    def test_new_round_teaches_from_latest_publication_after_interval(self):
        from vault_train import run_training

        store = self.store()
        observed = []
        manual = []

        def provider(_role, index, context):
            observed.append(context["base"])
            return proposal(f"triple-{index}", f"rain measure {index}"), teacher(f"teacher-{index}")

        def publish_during_interval(_delay):
            ticket = store.submit(proposal("manual-triple", "manual measure"), teacher("manual"))
            store.build()
            self.assertEqual(store.evidence(ticket)["status"], "accepted")
            manual.append(store.current()["identity"])

        report = run_training(store, provider, None, teachers=1, rounds=2,
                              interval=1, sleep=publish_during_interval)
        self.assertEqual(observed[1], manual[0])
        self.assertEqual(report["rounds"][1]["build"][0]["status"], "accepted")

    def test_interrupted_round_resumes_only_missing_teachers(self):
        from vault_train import run_training

        store = self.store()

        def interrupted(role, index, _context):
            if index == 0:
                raise RuntimeError("Simulated interruption.")
            return proposal(), teacher("finished-worker")

        with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
            run_training(store, interrupted, None, teachers=2, rounds=1)
        self.assertEqual(store.status()["counts"], {"queued": 1})
        seen = []

        def resume(role, index, _context):
            seen.append(index)
            return proposal(), teacher("resumed-worker")

        report = run_training(store, resume, None, teachers=2, rounds=1)
        self.assertEqual(seen, [0])
        self.assertEqual([item["round"] for item in report["rounds"]], [1])
        self.assertEqual(store.status()["counts"], {"accepted": 2})

    def test_round_resumes_after_publishing_one_of_its_candidates(self):
        from vault_train import run_training

        store = self.store()

        def interrupted(role, index, context):
            if role == "teacher":
                return proposal(f"triple-{index}", f"rain measure {index}"), teacher(f"teacher-{index}")
            if context["mutation"]["operation"]["name"] == "triple-1":
                raise RuntimeError("Second review interrupted.")
            return {"summary": "Checked.", "counterexamples": []}, teacher("critic-a") | {"role": role}

        with self.assertRaisesRegex(RuntimeError, "Second review interrupted"):
            run_training(store, interrupted, interrupted, teachers=2, rounds=1)
        self.assertEqual(store.status()["counts"], {"accepted": 1, "queued": 1})

        def resumed(role, _index, context):
            self.assertEqual(role, "critic")
            self.assertEqual(context["mutation"]["operation"]["name"], "triple-1")
            return {"summary": "Checked.", "counterexamples": []}, teacher("critic-b") | {"role": role}

        report = run_training(store, resumed, resumed, teachers=2, rounds=1)
        self.assertEqual(report["rounds"][0]["round"], 1)
        self.assertEqual(store.status()["counts"], {"accepted": 2})
        self.assertEqual(store.status()["revisions"], 3)

    def test_encounters_and_failed_model_checks_are_available_to_later_teachers(self):
        from vault_train import load_encounters, run_training

        path = self.work.parent / "encounters.json"
        write(path, [{"format": "vault-trace/1",
                      "invocation": {"turn": {"kind": "message", "text": "rain measure 12"},
                                     "context": {"private": "omit from prompt"}},
                      "result": {"activation": None, "reply": "Unresolved"}}])
        encounters = load_encounters(path)
        seen = []

        def provider(role, index, context):
            if role == "critic":
                return {"summary": "Incorrect edge case.", "counterexamples": [
                    {"kind": "counterexample", "check": "operation", "input": {"state": {"x": -1}},
                     "expect": {"value": 0}},
                ]}, teacher(f"critic-{len(seen)}") | {"role": role}
            seen.append(context)
            return proposal(), teacher(f"teacher-{index}")

        run_training(self.store(), provider, provider, teachers=1, rounds=2, encounters=encounters)
        self.assertEqual(seen[0]["encounters"][0]["text"], "rain measure 12")
        self.assertNotIn("omit from prompt", json.dumps(seen))
        self.assertEqual(seen[1]["candidate_counterexamples"][0]["actual"]["value"], -3)

    def test_trace_exports_of_both_versions_retain_verified_ids(self):
        from vault_trace import record_id
        from vault_train import load_encounters

        path = self.work.parent / "traces.json"
        records = [
            {
                "format": version,
                "invocation": {"turn": {"kind": "message", "text": f"message {version}"}},
                "result": {"reply": "Unresolved", "activation": None},
            }
            for version in ("vault-trace/1", "vault-trace/2")
        ]
        trace_ids = [record_id(record) for record in records]
        write(path, [record | {"trace_id": identity} for record, identity in zip(records, trace_ids, strict=True)])
        self.assertEqual([entry["trace_id"] for entry in load_encounters(path)], trace_ids)

    def test_round_reports_exact_encounter_progress_and_preserves_solved_turn(self):
        from vault_train import run_training

        store = self.store()

        def teacher_for_first_round(_role, _index, _context):
            return turn_proposal({"$": "result.value"}), teacher("first-round")

        encounters = [{"text": "rain measure 9", "expected_reply": 27},
                      {"text": "a new synthetic question", "expected_reply": "Still to learn"}]
        report = run_training(store, teacher_for_first_round, None, teachers=1, rounds=1, encounters=encounters)
        accepted = report["rounds"][0]["build"][0]
        self.assertEqual(accepted["status"], "accepted")
        progress = report["rounds"][0]["encounters"]
        self.assertEqual(progress, {"total": 2, "solved_before": 0, "solved_after": 1, "remaining": 1})
        original = store.current()

        def overlapping_teacher(_role, _index, _context):
            lesson = proposal("overlapping-triple", "special measure")
            operation = lesson["operation"]
            operation["anchors"].append("rain measure")
            operation["parameters"]["activation"] = {
                "phrases": ["special measure", "rain measure 9"],
                "input": {"state": {"x": 4}},
                "reply": {"$": "result.value"},
            }
            lesson["examples"] += [
                {"kind": "positive", "check": "turn", "input": {"text": "special measure"}, "expect": {"reply": 12}},
                {"kind": "counterexample", "check": "turn", "input": {"text": "special measure word"},
                 "expect": {"uses_learned": False}},
            ]
            return lesson, teacher("overlap")

        next_report = run_training(store, overlapping_teacher, None, teachers=1, rounds=1)
        rejected = next_report["rounds"][0]["build"][0]
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["assessment"]["tested_against"], original["identity"])
        self.assertEqual(rejected["assessment"]["example"]["input"], {"text": "rain measure 9"})
        self.assertEqual(rejected["assessment"]["actual"]["reply"], "Unresolved")
        self.assertEqual(store.current(), original)


if __name__ == "__main__":
    unittest.main()
