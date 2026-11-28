"""Queue model lessons and publish only verified Vault revisions."""

import argparse
import json
import os
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep as sleep_for

from commonsense import evaluate_situation
from reasoning_links import index_reasoning_endpoints, load_rules, prove
from vault_identity import asset_path, identify, read_program
from vault_operations import DEFAULT_MANIFEST, Vault, install, read_declarations
from vault_training_data import (
    SIZE_LIMIT, CheckFailure, append_graph, canonical, check_examples, distinct, encounter_progress,
    encounter_targets, evaluate_encounters, fingerprint, learned_cases, load_encounters, prepare, prepare_criticism,
)
from vault_training_providers import ProviderFailure


REGRESSIONS = "training-regressions.vault"
QUEUE = """
    CREATE TABLE IF NOT EXISTS revisions (
        identity TEXT PRIMARY KEY, parent TEXT REFERENCES revisions(identity), metadata TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS publication (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1), identity TEXT NOT NULL REFERENCES revisions(identity)
    );
    CREATE TABLE IF NOT EXISTS proposals (
        id INTEGER PRIMARY KEY, base TEXT NOT NULL REFERENCES revisions(identity),
        status TEXT NOT NULL DEFAULT 'queued', payload TEXT NOT NULL, conversation TEXT NOT NULL,
        assessment TEXT, critic TEXT, revision TEXT REFERENCES revisions(identity)
    );
    CREATE INDEX IF NOT EXISTS pending_proposals ON proposals (status, id);
    CREATE TABLE IF NOT EXISTS training_rounds (
        id INTEGER PRIMARY KEY, base TEXT NOT NULL REFERENCES revisions(identity), workers INTEGER NOT NULL,
        offset INTEGER NOT NULL, context TEXT NOT NULL, report TEXT
    );
    CREATE TABLE IF NOT EXISTS round_participants (
        round_id INTEGER NOT NULL REFERENCES training_rounds(id),
        position INTEGER NOT NULL, proposal INTEGER NOT NULL UNIQUE REFERENCES proposals(id),
        PRIMARY KEY (round_id, position)
    ) WITHOUT ROWID;
"""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def connect(path, *, readonly=False, named=False):
    uri = path.resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rwc")
    db = sqlite3.connect(uri, uri=True, timeout=30)
    if named:
        db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def revision_data(work, identity, metadata):
    root = work / "revisions" / identity
    result = {"identity": identity, "root": str(root)}
    result.update({name: str(asset_path(root, relative)) for name, relative in metadata.items()})
    return result


def published_revision(work, identity=None):
    """Read the current or an explicitly identified published snapshot."""
    work = Path(work).resolve()
    queue = work / "training.sqlite"
    if queue.is_file():
        with closing(connect(queue, readonly=True, named=True)) as db:
            if identity is None:
                return current_revision(work, db)
            if db.execute("SELECT 1 FROM sqlite_schema WHERE name = 'revisions'").fetchone():
                row = db.execute("SELECT metadata FROM revisions WHERE identity = ?", (identity,)).fetchone()
                if row:
                    return revision_data(work, identity, json.loads(row["metadata"]))
    if identity is not None:
        raise ValueError(f"No published revision {identity!r}.")
    return None


def current_revision(work, db):
    if not db.execute("SELECT 1 FROM sqlite_schema WHERE name = 'publication'").fetchone():
        return None
    row = db.execute("""
        SELECT r.identity, r.metadata FROM publication AS p JOIN revisions AS r USING (identity)
    """).fetchone()
    return revision_data(work, row["identity"], json.loads(row["metadata"])) if row else None


def check_snapshot_encounters(revision, targets):
    if not targets:
        return []
    with closing(connect(Path(revision["database"]), readonly=True)) as db:
        return evaluate_encounters(db, Path(revision["root"]), targets)


def require_provenance(conversation, role):
    if not isinstance(conversation, dict) or not all(
        isinstance(conversation.get(name), str) and conversation[name]
        for name in ("role", "provider", "session", "model", "prompt", "response")
    ):
        raise ValueError(f"Supply the {role}'s role, provider, session, model, prompt, and response.")
    if conversation["role"] != role:
        raise ValueError(f"Expected provenance from a {role} conversation.")


class InvalidSubmission(ValueError):
    """A teacher response could not be stored as a normal submission."""


def submission_evidence(payload):
    return {"bytes": len(payload), "sha256": sha256(payload).hexdigest(),
            "excerpt": payload[:4096].decode("utf-8", errors="ignore")}


def retained_teacher_response(proposal, conversation, reason):
    try:
        detail = submission_evidence(canonical(proposal).encode("utf-8", errors="backslashreplace"))
    except (ValueError, TypeError, RecursionError):
        detail = {"type": type(proposal).__name__}
    conversation = conversation if isinstance(conversation, dict) else {}
    retained = {"role": "teacher", "failure": str(reason)}
    omitted = {}
    for field in ("provider", "session", "model", "prompt", "response"):
        value = conversation.get(field)
        if isinstance(value, str) and value:
            data = value.encode("utf-8", errors="backslashreplace")
            preview = submission_evidence(data)
            retained[field] = preview.pop("excerpt")
            if len(data) > 4096:
                omitted[field] = preview
        else:
            retained[field] = "Unavailable"
    if omitted:
        retained["truncated"] = omitted
    return detail, retained


def copy_fixed_asset(source, destination, *, pin):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not pin:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copyfile(source, destination)


def stage(source, destination, *, pin):
    program, values, database = read_program(Path(source["program"]))
    source_root = Path(source["root"])
    program_path = Path(source["program"])
    manifest = Path(source["manifest"])
    metadata = {
        "program": program_path.relative_to(source_root).as_posix(),
        "manifest": manifest.relative_to(source_root).as_posix(),
        "database": database.relative_to(source_root).as_posix(),
    }
    scripts = Path(__file__).resolve().parent
    bootstrap = {path.name: path for path in scripts.glob("*.py")
                 if not path.name.startswith("test_") and path.name != "vault_health.py"}
    fixed = {
        values,
        *(asset_path(source_root, name) for name in program.get("assets", {}).values()),
        *source_root.glob("example-*.vault"),
        *(path for path in (source_root / "reasoning.vault", source_root / "commonsense.vault",
                            source_root / "activation.vault") if path.is_file()),
    }
    for entry in read_json(manifest)["files"]:
        fixed.add(asset_path(source_root, entry["path"]))
    for path in sorted(fixed - {program_path, manifest, database, source_root / REGRESSIONS}):
        relative = path.relative_to(source_root)
        if relative.parts[0] == "runtime" and path.suffix == ".vault":
            continue
        if relative.parent == Path("scripts") and path.name in bootstrap:
            continue
        copy_fixed_asset(path, destination / relative, pin=pin)
    graph = source_root / "runtime"
    if not graph.is_dir():
        graph = scripts.parent / "runtime"
    for path in graph.rglob("*.vault"):
        copy_fixed_asset(path, destination / "runtime" / path.relative_to(graph),
                         pin=pin or graph.parent != source_root)
    # Archive independent copies of the trusted host that evaluates this
    # candidate so later installation edits leave it intact.
    for name, path in bootstrap.items():
        copy_fixed_asset(path, destination / "scripts" / name, pin=True)
    target_database = destination / metadata["database"]
    target_database.parent.mkdir(parents=True, exist_ok=True)
    if pin:
        with closing(connect(database, readonly=True)) as old, closing(connect(target_database)) as new:
            old.backup(new)
    else:
        shutil.copyfile(database, target_database)
    shutil.copyfile(manifest, destination / metadata["manifest"])
    write_json(destination / metadata["program"], program)
    regressions = source_root / REGRESSIONS
    write_json(destination / REGRESSIONS, read_json(regressions) if regressions.is_file() else [])
    return metadata


def revision_identity(root, metadata):
    program_path = root / metadata["program"]
    program = read_json(program_path)
    assets = program.setdefault("assets", {})
    inputs = program["identity"]["inputs"]
    manifest = root / metadata["manifest"]
    paths = {
        metadata["manifest"], REGRESSIONS,
        *(entry["path"] for entry in read_json(manifest)["files"]),
        *(file.relative_to(root).as_posix() for file in (root / "runtime").rglob("*.vault")),
        *(file.relative_to(root).as_posix() for file in (root / "scripts").glob("*.py")),
        *(file.relative_to(root).as_posix() for file in root.glob("example-*.vault")),
    }
    if (root / "activation.vault").is_file():
        paths.add("activation.vault")
    existing = {relative: name for name, relative in assets.items()}
    for relative in sorted(paths):
        name = existing.get(relative, "training:" + relative)
        assets[name] = relative
        if name not in inputs:
            inputs.append(name)
    write_json(program_path, program)
    return identify(program_path)["identity"]


def run_legacy_examples(db, root, *, address_db=None):
    """Replay historical sample inputs, rebasing mixed-radix links by component."""
    vault = Vault(db, root)
    results = {}
    for filename in ("example-execution.vault", "example-address.vault"):
        path = root / filename
        if not path.is_file():
            continue
        data = read_json(path)
        if "address" in data:
            address = data["address"]
            if address_db is not None:
                address = rebase_address(address_db, db, address)
            value = vault.run_address(address, data["input"])
            value.pop("address")
        else:
            value = vault.run(data["operation"], data["input"])
        results[filename] = value
    rules_path = root / "reasoning.vault"
    if rules_path.is_file():
        rules = load_rules(rules_path)
        for filename, evaluate in (("example-argument.vault", prove),
                                   ("example-commonsense.vault", evaluate_situation)):
            path = root / filename
            if path.is_file():
                results[filename] = evaluate(rules, read_json(path))
    return results


def rebase_address(before, after, address):
    from vault_operations import address_page

    hops = address_page(
        before, address.get("links", 1), address.get("width", 1), int(address["position"]), 1
    )["page"][0]["hops"]
    minimum, maximum = after.execute("SELECT MIN(id), MAX(id) FROM nodes").fetchone()
    count = maximum - minimum + 1
    relations = [row[0] for row in after.execute("SELECT name FROM reasoning_relations ORDER BY name")]
    operations = [row[0] for row in after.execute("SELECT id FROM operations ORDER BY id")]
    position = 0

    def append_keys(position, keys):
        for key in keys:
            node_id = after.execute("SELECT id FROM nodes WHERE key = ?", (key,)).fetchone()[0]
            position = position * count + node_id - minimum
        return position

    position = append_keys(position, hops[0]["left"])
    for hop in hops:
        position = position * len(relations) + relations.index(hop["relation"]["name"])
        position = position * len(operations) + operations.index(hop["operation"])
        position = append_keys(position, hop["right"])
    return address | {"position": str(position)}


def confirm_preserved(database, previous):
    with closing(connect(database, readonly=True)) as db:
        db.execute("ATTACH DATABASE ? AS prior", (Path(previous).as_uri() + "?mode=ro",))
        if db.execute("""
            SELECT 1 FROM prior.nodes AS p LEFT JOIN main.nodes AS n USING (id)
            WHERE p.key IS NOT n.key LIMIT 1
        """).fetchone():
            raise ValueError("A vocabulary address changed.")
        for table in ("operations", "operation_cases", "operation_anchors", "operation_dependencies"):
            if db.execute(f"""
                SELECT 1 FROM (SELECT * FROM prior.{table} EXCEPT SELECT * FROM main.{table}) LIMIT 1
            """).fetchone():
                raise ValueError(f"An existing {table} row changed.")


class TrainingStore:
    def __init__(self, work, manifest=DEFAULT_MANIFEST, program=None):
        self.work = Path(work).resolve()
        manifest = Path(manifest).resolve()
        program = Path(program).resolve() if program else manifest.parent / "identity.vault"
        if program.parent != manifest.parent:
            raise ValueError("Read the initial manifest and identity from one Vault root.")
        if any(self.work.is_relative_to(manifest.parent / name) for name in ("raw", "operations", "runtime")):
            raise ValueError("Choose a separate work directory outside the shipped assets.")
        self.source = {"root": str(manifest.parent), "manifest": str(manifest), "program": str(program)}
        self.queue = self.work / "training.sqlite"
        self.staging = self.work / ".training-staging"

    @contextmanager
    def coordinator(self):
        self.work.mkdir(parents=True, exist_ok=True)
        with closing(connect(self.work / "coordinator.sqlite")) as lock, lock:
            lock.execute("BEGIN IMMEDIATE")
            yield

    @contextmanager
    def builder(self):
        self.work.mkdir(parents=True, exist_ok=True)
        revisions = self.work / "revisions"
        revisions.mkdir(exist_ok=True)
        self.staging.mkdir(exist_ok=True)
        if revisions.resolve() != revisions or self.staging.resolve() != self.staging:
            raise ValueError("Training revisions and staging must be located inside the work directory.")
        with closing(connect(self.work / "builder.sqlite")) as lock, lock:
            lock.execute("BEGIN IMMEDIATE")
            with closing(connect(self.queue)) as db, db:
                db.execute("PRAGMA journal_mode = WAL")
                db.executescript(QUEUE)
            # A concurrent builder cannot touch staging while this lock is held.
            for path in self.staging.glob("candidate-*"):
                if path.is_dir() and path.resolve().is_relative_to(self.staging) and path.resolve() == path:
                    shutil.rmtree(path)
            yield

    def initialize(self):
        with self.builder():
            current = self.current()
            if current:
                return current
            with TemporaryDirectory(prefix="candidate-", dir=self.staging) as folder:
                root = Path(folder) / "candidate"
                root.mkdir()
                metadata = stage(self.source, root, pin=True)
                database = root / metadata["database"]
                install(database, root, read_declarations(root / metadata["manifest"]))
                with closing(connect(database)) as db, db:
                    if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'reasoning_evidence'").fetchone():
                        index_reasoning_endpoints(db)
                with closing(connect(database, readonly=True)) as db:
                    run_legacy_examples(db, root)
                identity = revision_identity(root, metadata)
                self._publish(root, identity, metadata, parent=None)
            return self.current()

    def current(self):
        return published_revision(self.work)

    def submit(self, proposal, conversation, base=None, *, participant=None):
        try:
            require_provenance(conversation, "teacher")
            payload = canonical(proposal)
            provenance = canonical(conversation)
            if len(payload.encode("utf-8")) > SIZE_LIMIT or len(provenance.encode("utf-8")) > SIZE_LIMIT * 4:
                raise ValueError("A proposal or conversation exceeds the submission size limit.")
        except (ValueError, TypeError, RecursionError) as error:
            raise InvalidSubmission(str(error)) from error
        if not self.queue.is_file():
            raise ValueError("Initialize the training Vault first.")
        with closing(connect(self.queue)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if base is None:
                base = db.execute("SELECT identity FROM publication").fetchone()[0]
            cursor = db.execute("INSERT INTO proposals (base, payload, conversation) VALUES (?, ?, ?)",
                                (base, payload, provenance))
            ticket = cursor.lastrowid
            if participant is not None:
                db.execute("INSERT INTO round_participants VALUES (?, ?, ?)", (*participant, ticket))
            return ticket

    def open_round(self):
        with closing(connect(self.queue, readonly=True, named=True)) as db:
            row = db.execute("SELECT * FROM training_rounds WHERE report IS NULL ORDER BY id LIMIT 1").fetchone()
        if row is None:
            return None
        return dict(row) | {"context": json.loads(row["context"])}

    def start_round(self, base, workers, offset, context):
        with closing(connect(self.queue)) as db, db:
            inserted = db.execute("INSERT INTO training_rounds (base, workers, offset, context) VALUES (?, ?, ?, ?)",
                                  (base, workers, offset, canonical(context)))
            return inserted.lastrowid

    def participants(self, round_id):
        with closing(connect(self.queue, readonly=True)) as db:
            return dict(db.execute("SELECT position, proposal FROM round_participants WHERE round_id = ?", (round_id,)))

    def finish_round(self, round_id, report):
        with closing(connect(self.queue)) as db, db:
            db.execute("UPDATE training_rounds SET report = ? WHERE id = ?", (canonical(report), round_id))

    def candidate_feedback(self):
        """Reuse only model-round evidence previously authorized for model prompts."""
        with closing(connect(self.queue, readonly=True, named=True)) as db:
            rows = db.execute("""
                SELECT p.id, p.assessment, p.payload
                FROM proposals AS p JOIN round_participants AS participant ON participant.proposal = p.id
                WHERE p.status = 'rejected'
                ORDER BY p.id DESC LIMIT 24
            """).fetchall()
        feedback, seen = [], set()
        for row in rows:
            assessment = json.loads(row["assessment"])
            if "example" not in assessment:
                continue
            details = assessment["actual"]
            if isinstance(details, dict) and assessment["example"]["check"] == "turn":
                details = {field: details[field] for field in ("reply", "uses_learned", "error") if field in details}
            signature = fingerprint([assessment["example"], details])
            if signature not in seen:
                payload = json.loads(row["payload"])
                lesson = {"example": assessment["example"], "actual": details, "summary": payload["summary"]}
                if len(canonical(lesson).encode("utf-8")) <= 2800:
                    feedback.append(lesson | {"proposal": row["id"]})
                    seen.add(signature)
            if len(feedback) == 6:
                break
        return feedback

    def evidence(self, proposal):
        with closing(connect(self.queue, readonly=True, named=True)) as db:
            record = db.execute("SELECT * FROM proposals WHERE id = ?", (proposal,)).fetchone()
        if record is None:
            raise ValueError(f"No proposal {proposal}.")
        value = dict(record)
        value["proposal"] = json.loads(value.pop("payload"))
        for name in ("conversation", "assessment", "critic"):
            if value[name] is not None:
                value[name] = json.loads(value[name])
        return value

    def status(self):
        if not self.queue.is_file():
            return {"current": None, "counts": {}, "revisions": 0}
        with closing(connect(self.queue, readonly=True, named=True)) as db:
            db.execute("BEGIN")
            current = current_revision(self.work, db)
            if current is None:
                return {"current": None, "counts": {}, "revisions": 0}
            counts = dict(db.execute("SELECT status, COUNT(*) FROM proposals GROUP BY status"))
            revisions = db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
            rounds = {}
            if db.execute("SELECT 1 FROM sqlite_schema WHERE name = 'training_rounds'").fetchone():
                rounds = dict(db.execute("""
                    SELECT CASE WHEN report IS NULL THEN 'in_progress' ELSE 'completed' END, COUNT(*)
                    FROM training_rounds GROUP BY 1
                """))
        return {"current": current, "counts": counts, "revisions": revisions, "rounds": rounds}

    @staticmethod
    def _settle(db, tickets, status, assessment, critic=None, revision=None):
        db.executemany("""
            UPDATE proposals SET status = ?, assessment = ?, critic = ?, revision = ? WHERE id = ?
        """, [(status, canonical(assessment), canonical(critic) if critic else None, revision, ticket)
              for ticket in tickets])

    def _publish(self, root, identity, metadata, parent, tickets=(), assessment=None, critic=None):
        destination = self.work / "revisions" / identity
        if not destination.exists():
            root.replace(destination)
        elif identify(destination / metadata["program"])["identity"] != identity:
            raise ValueError("A directory for this revision exists with an invalid fingerprint.")
        with closing(connect(self.queue)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            stored = db.execute("SELECT identity FROM publication").fetchone()
            if (stored[0] if stored else None) != parent:
                raise ValueError("The published revision changed during candidate verification.")
            db.execute("INSERT OR IGNORE INTO revisions VALUES (?, ?, ?)", (identity, parent, canonical(metadata)))
            db.execute("INSERT INTO publication VALUES (1, ?) ON CONFLICT(singleton) DO UPDATE SET identity=excluded.identity",
                       (identity,))
            self._settle(db, tickets, "accepted", assessment, critic, identity)

    def _reject(self, tickets, reason, critic=None, **facts):
        assessment = {"reason": reason, **facts}
        with closing(connect(self.queue)) as db, db:
            self._settle(db, tickets, "rejected", assessment, critic)
        return {"status": "rejected", "proposals": tickets, "assessment": assessment}

    def _test_candidate(self, current, group, critic, targets):
        mutation = group["mutation"]
        tickets = group["tickets"]
        examples = distinct(group["examples"])
        critique = None
        evidence = {"tested_against": current["identity"]}
        try:
            with TemporaryDirectory(prefix="candidate-", dir=self.staging) as folder:
                root = Path(folder) / "candidate"
                root.mkdir()
                metadata = stage(current, root, pin=False)
                database = root / metadata["database"]
                manifest_path = root / metadata["manifest"]
                with closing(connect(database)) as db, db:
                    appended = append_graph(db, mutation)
                operation = mutation["operation"]
                index = None
                if operation:
                    manifest = read_json(manifest_path)
                    index = manifest["operation_count"] + 1
                    manifest["operation_count"] = index
                    relative = f"training-operations/{index}.vault"
                    declaration = {name: value for name, value in operation.items() if name != "module"}
                    (root / "training-operations").mkdir(exist_ok=True)
                    write_json(root / relative, [declaration | {"id": index, "cases": learned_cases(examples)}])
                    manifest["files"].append({"path": relative, "module": operation["module"]})
                    write_json(manifest_path, manifest)
                elif not any(appended.values()):
                    raise ValueError("The proposed vocabulary and links are already present.")
                checked = install(database, root, read_declarations(manifest_path))
                confirm_preserved(database, current["database"])
                regressions = read_json(root / REGRESSIONS)
                with (
                    closing(connect(database, readonly=True)) as db,
                    closing(connect(Path(current["database"]), readonly=True)) as old,
                    closing(connect(Path(self.baseline()["database"]), readonly=True)) as baseline,
                ):
                    # The ranking of a mixed-radix position changes when a base grows.
                    # Recheck its same vocabulary and operation components in the new layout.
                    original = run_legacy_examples(old, Path(current["root"]), address_db=baseline)
                    actual = run_legacy_examples(db, root, address_db=baseline)
                    if canonical(original) != canonical(actual):
                        raise ValueError("An original Vault example changed its result.")
                    existing = [case for entry in regressions for case in entry.get("encounters", ())]
                    before = evaluate_encounters(old, Path(current["root"]), targets)
                    combined = distinct([*targets, *existing])
                    after = evaluate_encounters(db, root, combined)
                    selected = after[:len(targets)]
                    evidence["encounters"] = encounter_progress(before, selected)
                    must_preserve = {canonical(item) for item in existing}
                    must_preserve.update(canonical(item["target"]) for item in before if item["solved"])
                    for outcome in after:
                        target = outcome["target"]
                        if canonical(target) in must_preserve and not outcome["solved"]:
                            example = {"kind": "counterexample", "check": "turn", "input": {"text": target["text"]},
                                       "expect": {"reply": target["expected_reply"]}}
                            raise CheckFailure(example, outcome["actual"], "A previously solved encounter changed its reply.")
                    prior_examples = sum(check_examples(db, root, entry["mutation"], entry["examples"])
                                         for entry in regressions)
                    teacher_checks = check_examples(db, root, mutation, examples)
                    counterexamples = []
                    if critic is not None:
                        review, critique = critic({"base": current["identity"], "mutation": mutation,
                                                   "examples": examples,
                                                   "teacher_summaries": list(dict.fromkeys(group["summaries"]))})
                        require_provenance(critique, "critic")
                        if critique["session"] in group["sessions"]:
                            raise ValueError("The critic must use a session separate from the teachers.")
                        critique = critique | {"review": review}
                        counterexamples = prepare_criticism(review, mutation)
                        check_examples(db, root, mutation, counterexamples)
                regression = {"mutation": mutation, "examples": distinct([*examples, *counterexamples])}
                solved = [item["target"] for item in selected if item["solved"] and item["target"] not in existing]
                if solved:
                    regression["encounters"] = solved
                regressions.append(regression)
                write_json(root / REGRESSIONS, regressions)
                identity = revision_identity(root, metadata)
                evidence["candidate_identity"] = identity
                assessment = {"compiled": checked, "legacy_examples": len(original), "prior_examples": prior_examples,
                              "teacher_examples": teacher_checks, "critic_examples": len(counterexamples),
                              "appended": appended | {"operation": index}, "change": fingerprint(mutation), **evidence}
                self._publish(root, identity, metadata, current["identity"], tickets, assessment, critique)
                return {"status": "accepted", "proposals": tickets, "revision": identity, "assessment": assessment}
        except CheckFailure as error:
            return self._reject(tickets, str(error), critique, example=error.example, actual=error.actual, **evidence)
        except ProviderFailure as error:
            return self._reject(tickets, str(error), error.conversation, **evidence)
        except (ValueError, KeyError, TypeError, ArithmeticError, sqlite3.IntegrityError) as error:
            return self._reject(tickets, str(error), critique, **evidence)

    def baseline(self):
        with closing(connect(self.queue, readonly=True, named=True)) as db:
            row = db.execute("SELECT identity, metadata FROM revisions WHERE parent IS NULL").fetchone()
        return revision_data(self.work, row["identity"], json.loads(row["metadata"]))

    @staticmethod
    def _round_base(db, round_id, current):
        if round_id is None:
            return current
        base = db.execute("SELECT base FROM training_rounds WHERE id = ?", (round_id,)).fetchone()[0]
        allowed = {row[0] for row in db.execute("""
            SELECT proposal.revision FROM round_participants AS participant
            JOIN proposals AS proposal ON participant.proposal = proposal.id
            WHERE participant.round_id = ? AND proposal.status = 'accepted'
        """, (round_id,))}
        revision = current
        while revision != base and revision in allowed:
            revision = db.execute("SELECT parent FROM revisions WHERE identity = ?", (revision,)).fetchone()[0]
        # Resume work atop a descendant from this same round. An unrelated
        # publication still makes its original submissions stale.
        return base if revision == base else current

    def build(self, *, critic=None, proposals=None, round_id=None, limit=32, targets=()):
        with self.builder():
            current = self.current()
            if current is None:
                raise ValueError("Initialize the training Vault first.")
            with closing(connect(self.queue, readonly=True, named=True)) as db:
                proposal_base = self._round_base(db, round_id, current["identity"])
                query = "SELECT id, base, payload, conversation FROM proposals WHERE status = 'queued'"
                selection = tuple(proposals) if proposals is not None else ()
                if proposals is not None:
                    query += " AND id IN (" + ",".join("?" for _ in selection) + ")"
                else:
                    query += " AND NOT EXISTS (SELECT 1 FROM round_participants WHERE proposal = proposals.id)"
                pending = db.execute(query + " ORDER BY id LIMIT ?", (*selection, limit)).fetchall()
                if proposals is not None:
                    positions = {ticket: index for index, ticket in enumerate(selection)}
                    pending.sort(key=lambda row: positions[row["id"]])
            if not pending:
                return []
            if identify(Path(current["program"]))["identity"] != current["identity"]:
                raise ValueError("The published Vault failed its fingerprint check.")
            results, groups = [], {}
            for row in pending:
                ticket = row["id"]
                if row["base"] != proposal_base:
                    results.append(self._reject([ticket], "The proposal targets a stale revision."))
                    continue
                conversation = json.loads(row["conversation"])
                if "failure" in conversation:
                    results.append(self._reject([ticket], conversation["failure"]))
                    continue
                try:
                    proposal = json.loads(row["payload"])
                    mutation, examples = prepare(proposal)
                except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                    results.append(self._reject([ticket], str(error)))
                    continue
                group = groups.setdefault(fingerprint(mutation), {
                    "mutation": mutation, "tickets": [], "examples": [], "summaries": [], "sessions": set(),
                })
                group["tickets"].append(ticket)
                group["examples"].extend(examples)
                group["summaries"].append(proposal["summary"])
                group["sessions"].add(conversation["session"])
            for group in groups.values():
                results.append(self._test_candidate(self.current(), group, critic, targets))
            return results


def conduct_round(store, round_state, teacher, critic):
    round_id = round_state["id"]
    base = round_state["base"]
    context = round_state["context"] | {"round": round_id}
    count = round_state["workers"]
    participants = store.participants(round_id)
    targets = encounter_targets(context.get("encounters", ()))
    before = check_snapshot_encounters(published_revision(store.work, base), targets) if targets else []

    def teach(position):
        index = round_state["offset"] + position
        try:
            value, conversation = teacher("teacher", index, context)
        except ProviderFailure as error:
            value = None
            conversation = error.conversation | {"failure": str(error)}
        try:
            return store.submit(value, conversation, base, participant=(round_id, position))
        except InvalidSubmission as error:
            evidence, retained = retained_teacher_response(value, conversation, error)
            return store.submit(evidence, retained, base, participant=(round_id, position))

    with ThreadPoolExecutor(max_workers=count) as workers:
        futures = {position: workers.submit(teach, position) for position in range(count)
                   if position not in participants}
        participants.update({position: future.result() for position, future in futures.items()})
    tickets = [participants[position] for position in range(count)]
    build = store.build(critic=(lambda data: critic("critic", 0, data)) if critic else None,
                        proposals=tickets, round_id=round_id, targets=targets)
    reported = {ticket for assessment in build for ticket in assessment["proposals"]}
    for ticket in tickets:
        if ticket not in reported:
            record = store.evidence(ticket)
            build.append({"proposals": [ticket], "status": record["status"], "assessment": record["assessment"],
                          "revision": record["revision"]})
    after = check_snapshot_encounters(store.current(), targets)
    report = {"round": round_id, "base": base, "proposals": tickets, "build": build,
              "encounters": encounter_progress(before, after)}
    store.finish_round(round_id, report)
    return report


def run_training(store, teacher, critic, *, teachers, rounds, brief="", encounters=(),
                 duration=None, interval=0, clock=monotonic, sleep=sleep_for, emit=None):
    if not 1 <= teachers <= 12 or rounds < 1:
        raise ValueError("Choose 1–12 teachers and a positive round cap.")
    if (duration is not None and (not isfinite(duration) or duration <= 0) or
            not isfinite(interval) or interval < 0):
        raise ValueError("Choose a positive finite duration and a nonnegative finite interval.")
    started = clock()
    deadline = started + duration if duration is not None else float("inf")
    results, recent = [], []
    offset = 0
    with store.coordinator():
        current = store.initialize()
        while len(results) < rounds and clock() < deadline:
            round_state = store.open_round()
            resumed = round_state is not None
            if round_state is None:
                current = store.current()
                base = current["identity"]
                reasoning = Path(current["root"], "reasoning.vault")
                relations = load_rules(reasoning)["relations"] if reasoning.is_file() else {}
                context = {"base": base, "brief": brief, "recent_changes": recent[-6:],
                           "relations": {name: definition["roles"] for name, definition in relations.items()},
                           "encounters": list(encounters), "candidate_counterexamples": store.candidate_feedback()}
                if clock() >= deadline:
                    break
                round_id = store.start_round(base, teachers, offset, context)
                round_state = {"id": round_id, "base": base, "workers": teachers, "offset": offset, "context": context}
            if emit:
                emit({"event": "round_started", "round": round_state["id"], "base": round_state["base"],
                      "resumed": resumed, "workers": round_state["workers"], "completed_rounds": len(results),
                      "round_cap": rounds, "elapsed_seconds": clock() - started})
            report = conduct_round(store, round_state, teacher, critic)
            for assessment in report["build"]:
                if assessment["status"] == "accepted":
                    proposal = store.evidence(assessment["proposals"][0])["proposal"]
                    recent.append(prepare(proposal)[0])
            current = store.current()
            results.append(report)
            offset = round_state["offset"] + round_state["workers"]
            if emit:
                emit({"event": "round_finished", **report, "current": current["identity"],
                      "completed_rounds": len(results), "elapsed_seconds": clock() - started})
            if len(results) < rounds:
                delay = min(interval, max(0, deadline - clock()))
                if delay:
                    if emit:
                        emit({"event": "waiting", "seconds": delay, "completed_rounds": len(results)})
                    sleep(delay)
    return {"current": current, "rounds": results, "stop": "round_cap" if len(results) == rounds else "duration",
            "elapsed_seconds": clock() - started}


def main():
    from vault_training_providers import CodexProvider, CommandProvider, FixtureProvider

    parser = argparse.ArgumentParser(description="Train a writable copy of the Vault with model proposals.")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--program", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="pin the initial Vault revision")
    commands.add_parser("status", help="show the active fingerprint and training queue counts")
    submit = commands.add_parser("submit", help="enqueue a teacher's proposal and conversation JSON")
    submit.add_argument("proposal", type=Path)
    submit.add_argument("conversation", type=Path)
    submit.add_argument("--base")
    evidence = commands.add_parser("evidence", help="show the evidence behind a proposal")
    evidence.add_argument("proposal", type=int)
    commands.add_parser("build", help="test pending proposals locally and publish passing changes")
    run = commands.add_parser("run", help="start concurrent teachers, a critic, and a publisher")
    run.add_argument("--provider", choices=("codex", "fixture", "command"), default="codex")
    run.add_argument("--fixture", type=Path)
    run.add_argument("--command-argv", help="JSON array naming a trusted program and its arguments")
    run.add_argument("--model")
    run.add_argument("--critic-model")
    run.add_argument("--timeout", type=int, default=180)
    run.add_argument("--teachers", type=int, default=2)
    run.add_argument("--rounds", type=int, help="cap the number of rounds in this invocation")
    run.add_argument("--duration-hours", type=float, help="stop starting rounds after this many elapsed hours")
    run.add_argument("--interval-seconds", type=float, default=0, help="wait between completed rounds")
    run.add_argument("--jsonl", action="store_true", help="stream JSON Lines progress events to standard output")
    run.add_argument("--encounters", type=Path, help="explicitly selected public or synthetic message encounters")
    run.add_argument("--brief", default="", help="a public or synthetic task description passed to teachers")
    args = parser.parse_args()
    stream = args.command == "run" and (args.jsonl or args.duration_hours is not None)
    try:
        store = TrainingStore(args.work, args.manifest, args.program)
        if args.command == "init":
            value = store.initialize()
        elif args.command == "status":
            value = store.status()
        elif args.command == "submit":
            value = {"proposal": store.submit(read_json(args.proposal), read_json(args.conversation), args.base)}
        elif args.command == "evidence":
            value = store.evidence(args.proposal)
        elif args.command == "build":
            value = {"current": store.initialize(), "build": store.build()}
            value["current"] = store.current()
        else:
            if args.timeout < 1:
                raise ValueError("Choose a positive provider timeout.")
            if len(args.brief) > 4000:
                raise ValueError("The learning brief may contain up to 4000 characters.")
            if args.provider == "codex":
                teacher = CodexProvider(args.model, args.timeout)
                critic = CodexProvider(args.critic_model or args.model, args.timeout)
            elif args.provider == "fixture":
                if args.fixture is None:
                    raise ValueError("Provide a fixture path for the fixture provider.")
                teacher = critic = FixtureProvider(args.fixture)
            else:
                if args.command_argv is None:
                    raise ValueError("Provide --command-argv for the command provider.")
                teacher = CommandProvider(json.loads(args.command_argv), args.model, args.timeout)
                critic = CommandProvider(json.loads(args.command_argv), args.critic_model or args.model, args.timeout)
            duration = args.duration_hours * 3600 if args.duration_hours is not None else None
            rounds = args.rounds if args.rounds is not None else 1000 if duration is not None else 1
            encounters = load_encounters(args.encounters) if args.encounters else ()
            value = run_training(
                store, teacher, critic, teachers=args.teachers, rounds=rounds, brief=args.brief,
                encounters=encounters, duration=duration, interval=args.interval_seconds,
                emit=(lambda event: print(json.dumps(event, ensure_ascii=True), flush=True)) if stream else None,
            )
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, ArithmeticError) as error:
        parser.exit(1, f"{error}\n")
    if stream:
        print(json.dumps({"event": "training_finished", "current": value["current"],
                          "completed_rounds": len(value["rounds"]), "stop": value["stop"],
                          "elapsed_seconds": value["elapsed_seconds"]}, ensure_ascii=True), flush=True)
    else:
        print(json.dumps(value, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
