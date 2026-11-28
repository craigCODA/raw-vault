import json
import re
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from string import Formatter

from vault_state_access import pointer_component, turn_state
from vault_trace import TraceStore, canonical, capture, content_id, event_address, ordered_json, payload_id
from vocabulary_links import link_key


SLOTS = {"int": r"[+-]?\d+", "number": r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)",
         "word": r"\S+", "text": r".+"}
SESSION_SCHEMA = """
    CREATE TABLE IF NOT EXISTS context_sessions (
        name TEXT PRIMARY KEY,
        context TEXT NOT NULL
    ) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS context_turns (
        session TEXT NOT NULL,
        position INTEGER NOT NULL,
        trace_id TEXT NOT NULL REFERENCES trace_runs(id),
        PRIMARY KEY (session, position)
    ) WITHOUT ROWID;
"""


@lru_cache(maxsize=256)
def learned_pattern(phrase):
    parts = list(Formatter().parse(phrase))
    names = set()
    pattern = []
    for index, (literal, name, style, conversion) in enumerate(parts):
        pattern.append(re.escape(literal))
        if name is None:
            continue
        if (conversion or style not in SLOTS or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", name)
                or name in names or style == "text" and index != len(parts) - 1):
            raise ValueError(f"Invalid typed slot in {phrase!r}.")
        names.add(name)
        pattern.append(f"(?P<{name}>{SLOTS[style]})")
    return re.compile("".join(pattern), re.IGNORECASE)


def validate_learned_activation(spec):
    if (not isinstance(spec, dict) or set(spec) - {"phrases", "input", "reply", "write"}
            or not isinstance(spec.get("phrases"), list) or not spec["phrases"]
            or not isinstance(spec.get("input"), dict) or "reply" not in spec
            or not isinstance(spec.get("write", {}), dict)):
        raise ValueError("A learned activation needs phrases, input, and a reply template.")
    for phrase in spec["phrases"]:
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("Each learned phrase must be nonempty text.")
        learned_pattern(phrase)


def candidate_terms(text, width):
    words = link_key(text.strip().rstrip(".!?")).split()
    return list(dict.fromkeys(
        " ".join(words[start:start + length])
        for length in range(min(width, len(words)), 0, -1)
        for start in range(len(words) - length + 1)
    ))


def activate_keys(vault, text, settings):
    terms = candidate_terms(text, settings["phrase_width"])
    nodes = {}
    for start in range(0, len(terms), 500):
        keys = terms[start:start + 500]
        marks = ",".join("?" for _ in keys)
        rows = vault.primitive(lambda: vault.db.execute(
            f"SELECT id, key FROM nodes WHERE key IN ({marks})", keys,
        ).fetchall())
        nodes.update({key: node_id for node_id, key in rows})
        vault.read_event("index/keys", len(canonical(rows).encode("utf-8")), keys=keys, rows=rows)
        if len(nodes) >= settings["max_terms"]:
            break
    selected = [{"key": key, "address": f"node/{nodes[key]}"}
                for key in terms if key in nodes][:settings["max_terms"]]
    for node in selected:
        vault.graph_event({"kind": "activation.node", **node})
    return selected


def learned_routes(vault, keys, text):
    if not keys:
        return []
    ids = [key["address"].split("/")[1] for key in keys]
    marks = ",".join("?" for _ in ids)
    rows = vault.primitive(lambda: vault.db.execute(f"""
        SELECT DISTINCT o.id, o.parameters
        FROM operation_anchors a JOIN operations o ON o.id = a.operation_id
        WHERE a.node_id IN ({marks}) ORDER BY o.id
    """, ids).fetchall())
    vault.read_event("index/activation-anchors", len(canonical(rows).encode("utf-8")),
                     inputs=[key["address"] for key in keys], rows=rows)
    matches = []
    for index, raw in rows:
        spec = json.loads(raw).get("activation")
        if spec is None:
            continue
        validate_learned_activation(spec)
        for phrase in spec["phrases"]:
            match = learned_pattern(phrase).fullmatch(text)
            if match:
                matches.append(({
                    "id": f"learned/{index}",
                    "program": {"operation": index, "input": spec["input"]},
                    "write": spec.get("write", {}),
                    "reply": spec["reply"],
                }, match.groupdict()))
                break
    return matches


def activate_links(vault, keys, limit):
    if not keys or limit == 0:
        return []
    terms = [key["key"] for key in keys]
    marks = ",".join("?" for _ in terms)
    rows = vault.primitive(lambda: vault.db.execute(f"""
        SELECT relation, left_key, right_key, sequence_id, cue, position, occurrences
        FROM reasoning_evidence
        WHERE left_key IN ({marks}) OR right_key IN ({marks})
        ORDER BY occurrences DESC, relation, left_key, right_key, sequence_id, position
        LIMIT ?
    """, [*terms, *terms, limit]).fetchall())
    vault.read_event("index/active-relations", len(canonical(rows).encode("utf-8")), keys=terms, rows=rows)
    active = []
    for relation, left, right, source, cue, position, occurrences in rows:
        address = "relation/" + content_id([relation, left, right, source, position])
        link = {"address": address, "relation": relation, "left": left, "right": right,
                "source": f"node/{source}", "cue": cue, "occurrences": occurrences}
        active.append(link)
        vault.graph_event({"kind": "activation.link", "address": address, "target": link["source"],
                           "relation": relation, "left": left, "right": right})
    return active


def choose_route(vault, definitions, keys, text, scope):
    for rule in definitions:
        match = re.fullmatch(rule["pattern"], text, flags=re.IGNORECASE)
        if match and ("when" not in rule or vault.condition(rule["when"], scope)):
            return rule, match.groupdict()
    learned = learned_routes(vault, keys, text)
    if len(learned) == 1:
        return learned[0]
    if len(learned) > 1:
        vault.graph_event({"kind": "activation.ambiguous", "address": "activation/ambiguous",
                           "candidates": [route["id"] for route, _ in learned]})
        return None, None
    return None, None


def execute_turn(vault, event, context):
    state = context.get("state", {})
    with turn_state(state):
        return _execute_turn(vault, event, context, state)


def _execute_turn(vault, event, context, state):
    if event.get("kind") != "message" or not isinstance(event.get("text"), str):
        raise ValueError("A conversational turn needs a message event with text.")
    program = json.loads(vault.read_asset("activation.vault"))
    if program.get("format") != "vault-activation/1":
        raise ValueError("Unsupported activation format.")
    settings = program["activation"]
    max_links = settings.get("max_links", 0)
    if any(type(settings[key]) is not int or settings[key] < 1
           for key in ("max_terms", "phrase_width", "memo_entries")):
        raise ValueError("Activation limits must be positive integers.")
    if type(max_links) is not int or max_links < 0:
        raise ValueError("The active relation limit must be a nonnegative integer.")
    text = " ".join(event["text"].strip().rstrip(".!?").split())
    keys = activate_keys(vault, text, settings)
    version = vault.recorder.revision if vault.recorder else None
    memo = dict(context.get("memo", {})) if version and context.get("revision") == version else {}
    memo_order = list(dict.fromkeys([
        *(key for key in context.get("memo_order", []) if key in memo), *sorted(memo),
    ]))
    with vault.reading_after(vault.previous_read()):
        links = activate_links(vault, keys, max_links)
    active = {"keys": keys, "links": links, "event": event_address(event), "rule": None}
    scope = {"event": event, "state": state, "save": state.get("save", {"symbols": {}}), "active": active}
    with vault.reading_after(vault.previous_read()):
        rule, match = choose_route(vault, program["routes"], keys, text, scope)
    result = {"reply": program["fallback"], "operation": None, "activation": None}
    if rule:
        address = "activation/" + rule["id"]
        active["rule"] = address
        scope |= {"match": match, "keys": {key: link_key(value) if isinstance(value, str) else value
                                             for key, value in match.items()}}
        invocation = vault.bind(rule["program"], scope)
        vault.graph_event({"kind": "activation", "address": address,
                           "target": "operation/" + str(invocation["operation"]), "bindings": match})
        memo_address = payload_id(ordered_json(invocation)) if rule.get("memoize") else None
        if memo_address is not None and memo_address in memo:
            output = memo[memo_address]
            vault.graph_event({"kind": "memory.hit", "address": "memory/" + memo_address})
        else:
            with vault.reading_after(vault.previous_read()):
                output = vault.run_program(invocation)
            if memo_address is not None:
                memo[memo_address] = output
                vault.graph_event({"kind": "memory.store", "address": "memory/" + memo_address,
                                   "target": "operation/" + str(invocation["operation"])})
        if memo_address is not None:
            if memo_address in memo_order:
                memo_order.remove(memo_address)
            memo_order.append(memo_address)
        scope["result"] = output
        writes = vault.bind(rule.get("write", {}), scope)
        for key, value in sorted(writes.items()):
            vault.graph_event({"kind": "state.write", "address": "state/" + pointer_component(key), "value": value})
        state = state | writes
        reply = vault.bind(rule["reply"], scope)
        result = {"reply": rule.get("missing", program["fallback"]) if reply is None else reply,
                  "operation": invocation["operation"], "activation": rule["id"]}
    while len(memo_order) > settings["memo_entries"]:
        expired = memo_order.pop(0)
        del memo[expired]
        vault.graph_event({"kind": "memory.evict", "address": "memory/" + expired})
    saved = {"state": state, "active": active, "memo": memo, "memo_order": memo_order, "revision": version}
    return {**result, "state": state, "active": active,
            "active_size": len(keys) + len(links) + 1 + int(rule is not None), "context": saved}


class Sessions:
    def __init__(self, path):
        self.store = TraceStore(Path(path))

    def send(self, vault, session, message):
        if not session:
            raise ValueError("A session needs a name.")
        with closing(self.store.writer()) as db:
            db.executescript(SESSION_SCHEMA)
            with db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT context FROM context_sessions WHERE name = ?", (session,)).fetchone()
                context = json.loads(row[0]) if row else {}
                invocation = {"turn": {"kind": "message", "text": message}, "context": context}
                record = capture(vault, invocation)
                trace_id = self.store.insert(db, record)
                position = db.execute("""
                    SELECT COALESCE(MAX(position) + 1, 0) FROM context_turns WHERE session = ?
                """, (session,)).fetchone()[0]
                db.execute("INSERT INTO context_turns VALUES (?, ?, ?)", (session, position, trace_id))
                if "result" in record:
                    next_context = record["result"]["context"]
                    db.execute("""
                        INSERT INTO context_sessions VALUES (?, ?)
                        ON CONFLICT(name) DO UPDATE SET context = excluded.context
                    """, (session, ordered_json(next_context)))
                    outcome = {key: value for key, value in record["result"].items() if key != "context"}
                else:
                    outcome = {"error": record["error"]}
        return {"session": session, "position": position, "trace_id": trace_id,
                **outcome, "profile": record["profile"]}
