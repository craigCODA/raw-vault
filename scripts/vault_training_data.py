"""Validate training declarations and evaluate them through the Vault."""

import hashlib
import json
import re
from pathlib import Path
from string import Formatter

from arithmetic import SYMBOL, TOKEN, compile_value
from reasoning_links import link_entries, load_rules
from vault_operations import Vault, matches_expected
from vault_trace import TRACE_FORMATS, TraceStore, record_id
from vocabulary_links import link_key


SIZE_LIMIT = 65_536
EXAMPLE_KINDS = {"positive", "counterexample"}
TURN_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError, ArithmeticError, FileNotFoundError)
OPERATORS = {
    ("vault_execution", "parameterized_transitions"),
    ("vault_language", "sense"),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def fields(value, labels, description):
    if not isinstance(value, dict) or set(value) != set(labels.split()):
        raise ValueError(f"{description} needs exactly these fields: {labels}.")
    return value


def nonempty_string(value, limit, label):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} needs nonempty text of at most {limit} characters.")
    return value


def key(value):
    value = nonempty_string(value, 128, "A vocabulary key")
    result = link_key(value)
    if not result or len(result) > 128 or any(ord(char) < 32 for char in result) or result.startswith("[["):
        raise ValueError("Use a plain decoded vocabulary key.")
    return result


def array(value, limit, label):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{label} needs an array of at most {limit} items.")
    return value


def keys(value):
    return sorted({key(item) for item in array(value, 12, "Vocabulary keys")})


def typed_link(value):
    fields(value, "relation left right", "A typed link")
    relation = nonempty_string(value["relation"], 64, "A relation")
    return {"relation": relation, "left": key(value["left"]), "right": key(value["right"])}


def distinct(values):
    return list({canonical(value): value for value in values}.values())


def encounter_from_trace(trace):
    invocation = trace.get("invocation")
    turn = invocation.get("turn") if isinstance(invocation, dict) else None
    if not isinstance(turn, dict) or turn.get("kind") != "message":
        raise ValueError("Training encounters require conversational message traces.")
    if "error" not in trace and (not isinstance(trace.get("result"), dict) or
                                 trace["result"].get("activation") is not None):
        raise ValueError("Select a failed or unresolved turn trace for training.")
    encounter = {"text": turn.get("text"), "reason": "A Vault turn failed." if "error" in trace else
                 "A Vault turn had no matching activation."}
    if "trace_id" in trace:
        record = {key: value for key, value in trace.items() if key != "trace_id"}
        if trace["trace_id"] != record_id(record):
            raise ValueError("The selected trace export failed its content check.")
        encounter["trace_id"] = trace["trace_id"]
    return encounter


def load_encounters(path):
    """Read only the explicitly selected public or synthetic message examples."""
    path = Path(path).resolve()
    if path.stat().st_size > 2_000_000:
        raise ValueError("The encounter file exceeds two megabytes.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("format") == "vault-training-encounters/1":
        data = data["encounters"]
    elif isinstance(data, dict) and data.get("format") in TRACE_FORMATS:
        data = [data]
    result = []
    for entry in array(data, 24, "Training encounters"):
        if not isinstance(entry, dict):
            raise ValueError("Each encounter must be an object.")
        if entry.get("format") in TRACE_FORMATS:
            entry = encounter_from_trace(entry)
        elif "trace_id" in entry:
            if not isinstance(entry["trace_id"], str) or not re.fullmatch("[0-9a-f]{64}", entry["trace_id"]):
                raise ValueError("A selected trace needs its original 64-digit identifier.")
            trace_path = path.parent / nonempty_string(entry["trace_store"], 500, "A trace store path")
            trace = TraceStore(trace_path).load(entry["trace_id"])
            entry = encounter_from_trace(trace) | {"trace_id": entry["trace_id"]} | {
                key: entry[key] for key in ("expected_reply",) if key in entry
            }
        elif set(entry) - {"text", "reason", "expected_reply"}:
            raise ValueError("A message encounter needs text and may include a reason or expected reply.")
        nonempty_string(entry.get("text"), 512, "An encounter message")
        if "reason" in entry:
            nonempty_string(entry["reason"], 512, "An encounter reason")
        if len(canonical(entry).encode("utf-8")) > 2400:
            raise ValueError("An encounter exceeds the maximum size.")
        result.append(entry)
    return distinct(result)


def encounter_targets(encounters):
    return distinct({"text": entry["text"], "expected_reply": entry["expected_reply"]}
                    for entry in encounters if "expected_reply" in entry)


def run_turn(vault, text):
    return vault.run_program({"turn": {"kind": "message", "text": text}, "context": {}})


def turn_error(error):
    return {"error": {"type": type(error).__name__, "message": str(error)}}


def evaluate_encounters(db, root, targets):
    if not targets:
        return []
    vault = Vault(db, root)
    outcomes = []
    for target in targets:
        try:
            reply = run_turn(vault, target["text"])["reply"]
        except TURN_ERRORS as error:
            actual, solved = turn_error(error), False
        else:
            actual = {"reply": reply}
            solved = canonical(reply) == canonical(target["expected_reply"])
        outcomes.append({"target": target, "solved": solved, "actual": actual})
    return outcomes


def encounter_progress(before, after):
    solved_after = sum(item["solved"] for item in after)
    return {"total": len(after), "solved_before": sum(item["solved"] for item in before),
            "solved_after": solved_after, "remaining": len(after) - solved_after}


def operation(value):
    fields(value, "name title anchors module operator parameters", "An operation")
    name = nonempty_string(value["name"], 72, "An operation name")
    if not re.fullmatch("[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name):
        raise ValueError("Operation names use lowercase words separated by hyphens.")
    title = nonempty_string(value["title"], 160, "An operation title")
    anchors = keys(value["anchors"])
    if not anchors:
        raise ValueError("A learned operation needs at least one vocabulary anchor.")
    module = nonempty_string(value["module"], 64, "An operation module")
    operator = nonempty_string(value["operator"], 64, "An operator")
    if (module, operator) not in OPERATORS:
        raise ValueError("Choose an allowlisted declarative operator.")
    parameters = value["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("An operation needs a parameter object.")
    activation = parameters.get("activation")
    if "activation" in parameters and not isinstance(activation, dict):
        raise ValueError("A learned activation needs a declaration object.")
    parameters = {name: item for name, item in parameters.items() if name != "activation"}
    if operator == "parameterized_transitions":
        fields(parameters, "expression target", "A transition rule")
        expression = nonempty_string(parameters["expression"], 96, "A transition expression")
        target = nonempty_string(parameters["target"], 32, "A transition target")
        tokens = [token for token in TOKEN.findall(expression) if not token.isspace()]
        if len(tokens) > 18 or not SYMBOL.fullmatch(target) or len(compile_value(expression).inputs) > 4:
            raise ValueError("A transition needs a simple expression and a valid target symbol.")
        parameters = {"expression": " ".join(tokens), "target": target}
    else:
        fields(parameters, "words", "A word-sense rule")
        words = parameters["words"]
        if not isinstance(words, dict) or not 1 <= len(words) <= 6:
            raise ValueError("A sense rule needs between one and six words.")
        normalized = {}
        for word, senses in words.items():
            if word != key(word) or not isinstance(senses, dict) or not 2 <= len(senses) <= 8:
                raise ValueError("A word needs a normalized key and two to eight named senses.")
            normalized[word] = {}
            for sense, cues in senses.items():
                nonempty_string(sense, 64, "A sense")
                cues = keys(cues)
                if not cues or any(not re.fullmatch(r"\w+", cue) for cue in cues):
                    raise ValueError("Each word sense needs one or more single-token cues.")
                normalized[word][sense] = cues
        parameters = {"words": normalized}
    if activation is not None:
        parameters["activation"] = activation
    return {"name": name, "title": title, "anchors": anchors, "module": module,
            "operator": operator, "parameters": parameters}


def example(value, declaration):
    fields(value, "kind check input expect", "An example")
    kind, check, arguments, expected = (value[name] for name in ("kind", "check", "input", "expect"))
    operator = declaration["operator"] if declaration else None
    if kind not in ("positive", "counterexample"):
        raise ValueError("Examples are positive or counterexample.")
    if check == "key":
        fields(arguments, "key", "A vocabulary example")
        arguments = {"key": key(arguments["key"])}
    elif check == "link":
        arguments = typed_link(arguments)
    elif check == "operation" and operator == "parameterized_transitions":
        if not isinstance(arguments, dict) or arguments.keys() not in ({"state"}, {"state", "values"}):
            raise ValueError("A transition example needs state and may supply values.")
        for bindings in arguments.values():
            if not isinstance(bindings, dict) or len(bindings) > 16:
                raise ValueError("A transition needs small numeric bindings.")
            for name, number in bindings.items():
                if not SYMBOL.fullmatch(name) or type(number) not in (int, float, str) or len(str(number)) > 48:
                    raise ValueError("A transition needs bounded numeric bindings.")
    elif check == "operation" and operator == "sense":
        fields(arguments, "word context", "A word-sense example")
        nonempty_string(arguments["word"], 128, "A word")
        nonempty_string(arguments["context"], 2048, "An example context")
    elif check == "turn" and declaration and "activation" in declaration["parameters"]:
        fields(arguments, "text", "A turn example")
        nonempty_string(arguments["text"], 512, "A turn utterance")
        if (not isinstance(expected, dict) or
                ("reply" not in expected and not (kind == "counterexample" and expected.get("uses_learned") is False)) or
                "uses_learned" in expected and type(expected["uses_learned"]) is not bool):
            raise ValueError("A turn example needs an expected reply or a counterexample asserting uses_learned is false.")
    else:
        raise ValueError("Choose key, link, operation, or a learned turn as the example check.")
    if check in ("key", "link") and expected is not (kind == "positive"):
        raise ValueError("A vocabulary or link example expects presence for a positive and absence for a counterexample.")
    if check == "operation" and (not isinstance(expected, dict) or not expected):
        raise ValueError("An operation example needs one or more expected output fields.")
    if check == "operation" and operator == "parameterized_transitions" and "value" not in expected:
        raise ValueError("A transition example must check the resulting value.")
    if check == "operation" and operator == "sense" and "sense" not in expected:
        raise ValueError("A word-sense example must check the selected sense.")
    return {"kind": kind, "check": check, "input": arguments, "expect": expected}


def prepare(proposal):
    fields(proposal, "summary keys links operation examples", "A proposal")
    nonempty_string(proposal["summary"], 800, "A proposal summary")
    proposed_operation = operation(proposal["operation"]) if proposal["operation"] is not None else None
    mutation = {
        "keys": keys(proposal["keys"]),
        "links": sorted(distinct(map(typed_link, array(proposal["links"], 12, "Typed links"))), key=canonical),
        "operation": proposed_operation,
    }
    if not any(mutation.values()):
        raise ValueError("A proposal needs a vocabulary key, link, or operation.")
    operator = proposed_operation["operator"] if proposed_operation else None
    examples = distinct(example(item, proposed_operation) for item in array(proposal["examples"], 12, "Examples"))
    if {item["kind"] for item in examples} != EXAMPLE_KINDS:
        raise ValueError("A proposal needs a positive example and a counterexample.")
    operation_examples = [item for item in examples if item["check"] == "operation"]
    if operator and ({item["kind"] for item in operation_examples} != EXAMPLE_KINDS
                     or len({canonical(item["input"]) for item in operation_examples}) < 2):
        raise ValueError("A proposed operation needs a positive and counterexample with distinct inputs.")
    if proposed_operation and "activation" in proposed_operation["parameters"]:
        turn_examples = [item for item in examples if item["check"] == "turn"]
        if ({item["kind"] for item in turn_examples} != EXAMPLE_KINDS or
                len({canonical(item["input"]) for item in turn_examples}) < 2):
            raise ValueError("A learned activation needs positive and counterexample turn checks with distinct inputs.")
        activation = proposed_operation["parameters"]["activation"]
        for phrase in array(activation.get("phrases"), 6, "Learned phrases"):
            parts = list(Formatter().parse(nonempty_string(phrase, 160, "A learned phrase")))
            slots = [literal for literal, name, *_ in parts if name is not None]
            if len(slots) > 4 or any(not any(char.isspace() for char in literal) for literal in slots[1:]):
                raise ValueError("Use at most four learned slots with a literal space between slots.")
        from vault_context import validate_learned_activation

        validate_learned_activation(activation)
        for template in ("input", "reply", "write"):
            validate_bindings(activation.get(template))
    for word in mutation["keys"]:
        if not any(item["kind"] == "positive" and
                   (word == item["input"].get("key") or
                    item["check"] == "link" and word in (item["input"]["left"], item["input"]["right"]))
                   for item in examples):
            raise ValueError(f"New key {word!r} needs a positive vocabulary or link example.")
    for link in mutation["links"]:
        if not any(item["check"] == "link" and item["kind"] == "positive" and item["input"] == link
                   for item in examples):
            raise ValueError("Every proposed link needs a positive link example.")
    return mutation, examples


def prepare_criticism(criticism, mutation):
    fields(criticism, "summary counterexamples", "A critic review")
    nonempty_string(criticism["summary"], 800, "A critic summary")
    operation = mutation["operation"]
    examples = [example(item, operation) for item in array(criticism["counterexamples"], 8, "Critic examples")]
    if any(item["kind"] != "counterexample" for item in examples):
        raise ValueError("A critic review may only supply counterexamples.")
    return distinct(examples)


def learned_cases(examples):
    return [{"name": f"training-{fingerprint(item)[:16]}", "input": item["input"], "expect": item["expect"]}
            for item in examples if item["check"] == "operation"]


def validate_bindings(template):
    pending = [template]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            if value.keys() == {"$"}:
                path = value["$"]
                if not isinstance(path, str) or any(not part.strip() for part in path.split(".")):
                    raise ValueError("A learned binding path needs nonempty dot-separated text components.")
            else:
                pending.extend(value.values())


def append_graph(db, mutation):
    count = {"keys": 0, "links": 0}
    for word in mutation["keys"]:
        changed = db.execute("INSERT INTO nodes (key) VALUES (?) ON CONFLICT(key) DO NOTHING", (word,))
        if changed.rowcount:
            db.execute("INSERT INTO raw_entries VALUES (?, 'training', ?, 1)", (changed.lastrowid, word))
            count["keys"] += 1
    for link in mutation["links"]:
        relation, left, right = (link[name] for name in ("relation", "left", "right"))
        if db.execute("SELECT 1 FROM reasoning_relations WHERE name = ?", (relation,)).fetchone() is None:
            raise ValueError(f"Unknown relation {relation!r}.")
        endpoints = dict(db.execute("SELECT key, id FROM nodes WHERE key IN (?, ?)", (left, right)))
        if left not in endpoints or right not in endpoints:
            raise ValueError(f"Both endpoints of {link!r} must be addressable vocabulary keys.")
        # Training evidence lives in the existing typed-link index. It has its own cue
        # so a learned claim can be distinguished from a corpus occurrence by readers.
        changed = db.execute("""
            INSERT INTO reasoning_evidence (relation, left_key, right_key, sequence_id, cue, position, occurrences)
            SELECT :relation, :left, :right, :sequence, '[vault training]', -1, 1
            WHERE NOT EXISTS (
                SELECT 1 FROM reasoning_evidence WHERE relation = :relation AND left_key = :left AND right_key = :right
            )
        """, link | {"sequence": endpoints[left]})
        count["links"] += changed.rowcount
    return count


class CheckFailure(ValueError):
    def __init__(self, example, actual, reason="An example did not match its expected outcome."):
        super().__init__(reason)
        self.example = example
        self.actual = actual


def check_examples(db, root, mutation, examples):
    vault = Vault(db, root)
    rules = load_rules(root / "reasoning.vault") if any(item["check"] == "link" for item in examples) else None
    for item in examples:
        arguments = item["input"]
        if item["check"] == "operation":
            actual = vault.run(mutation["operation"]["name"], arguments)
        elif item["check"] == "turn":
            index = vault.lookup(mutation["operation"]["name"])[0]
            try:
                actual = run_turn(vault, arguments["text"])
            except TURN_ERRORS as error:
                raise CheckFailure(item, turn_error(error), "An example could not execute.") from error
            actual["uses_learned"] = actual.get("activation") == f"learned/{index}" and actual.get("operation") == index
            if item["kind"] == "positive" and not actual["uses_learned"]:
                raise CheckFailure(item, actual)
        elif item["check"] == "key":
            actual = db.execute("SELECT 1 FROM nodes WHERE key = ?", (arguments["key"],)).fetchone() is not None
        else:
            relation, left, right = (arguments[name] for name in ("relation", "left", "right"))
            if relation not in rules["relations"]:
                raise ValueError(f"Unknown relation {relation!r}.")
            actual = False
            if all(db.execute("SELECT 1 FROM nodes WHERE key = ?", (word,)).fetchone() for word in (left, right)):
                actual = bool(link_entries(db, rules, relation, [left], [right], limit=1)["evidence"])
        if not matches_expected(actual, item["expect"]):
            raise CheckFailure(item, actual)
    return len(examples)
