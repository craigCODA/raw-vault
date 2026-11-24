import hashlib
import json
import tracemalloc
from copy import deepcopy
from time import perf_counter_ns

from commonsense import evaluate_situation
from reasoning_links import load_rules
from vault_execution import substitute, transition_key
from vault_identity import asset_path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_save(args, context):
    if "save_file" in args and "save" in args:
        raise ValueError("Supply one saved state.")
    save = args.get("save_file", args.get("save", args))
    if isinstance(save, str):
        save = json.loads(context.read_asset(save))
    return {"state": deepcopy(save["state"]), "cursor": save["cursor"]}


def resume_addresses(args, parameters, context):
    options = {key: args[key] for key in ("links", "event", "max_steps", "c") if key in args}
    return context.run(parameters["execution"], load_save(args, context) | options)


def state_differences(args, _parameters, _context):
    before, after = args["before"], args["after"]
    changed = {
        key: value for key, value in after.items()
        if key not in before or canonical(before[key]) != canonical(value)
    }
    deleted = sorted(before.keys() - after.keys())
    restored = before | changed
    for key in deleted:
        del restored[key]
    return {"difference": {"changed": changed, "deleted": deleted},
            "state": restored, "matches_after": canonical(restored) == canonical(after)}


def branch_arguments(args, save, event):
    options = {key: args[key] for key in ("links", "max_steps", "c") if key in args}
    return options | {"save": deepcopy(save), "event": deepcopy(event)}


def branching_saves(args, parameters, context):
    save = load_save(args, context)
    branches = {
        name: context.run(parameters["resume"], branch_arguments(args, save, event))
        for name, event in args["branches"].items()
    }
    return {"save": save, "branches": branches}


def shared_states(args, _parameters, _context):
    events, states = args["events"], args["states"]
    for state, transitions in states.items():
        for event, route in transitions.items():
            if event not in events or route["next"] not in states:
                raise ValueError(f"Undeclared event or successor at state {state!r}.")
    representatives = dict.fromkeys(states)
    while True:
        classes = {}
        refined = {}
        for state, transitions in sorted(states.items()):
            signature = tuple(
                (canonical(transitions[event]["output"]), representatives[transitions[event]["next"]])
                if event in transitions else None
                for event in events
            )
            refined[state] = classes.setdefault(signature, state)
        if refined == representatives:
            return {"representatives": refined, "shared_count": len(classes)}
        representatives = refined


def context_lifetimes(args, parameters, _context):
    kept, expired = [], []
    for fact in args["facts"]:
        scope = fact.get("scope", {})
        fields = parameters["scopes"][fact["lifetime"]]
        if all(key in scope and key in args["scope"] and scope[key] == args["scope"][key] for key in fields):
            kept.append(fact)
        else:
            expired.append(fact["id"])
    return {"facts": kept, "expired": expired}


def computed_capabilities(args, _parameters, context):
    catalog = context.catalog()
    compiled = {entry[key] for entry in catalog for key in ("id", "name")}
    capabilities = {}
    for name, requirements in args["capabilities"].items():
        missing = [
            requirement for requirement in requirements
            if (int(requirement) if isinstance(requirement, str) and requirement.isdecimal() else requirement)
            not in compiled
        ]
        capabilities[name] = {"available": not missing, "missing": missing}
    return {"capabilities": capabilities}


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def identity_through_state_changes(args, _parameters, _context):
    core = fingerprint(args["core"])
    before, after = args["before"], args["after"]
    before_core, after_core = fingerprint(before["core"]), fingerprint(after["core"])
    before_save, after_save = fingerprint(before), fingerprint(after)
    return {"core_fingerprint": core, "before_core": before_core, "after_core": after_core,
            "before_save": before_save, "after_save": after_save,
            "same_core": core == before_core == after_core, "distinct_saves": before_save != after_save}


def compact_decision_tables(args, parameters, context):
    addresses, routes, route_ids = {}, [], {}
    for row in args["rows"]:
        route = {"next": row["next"], "reply": row.get("reply")}
        signature = canonical(route)
        if signature not in route_ids:
            route_ids[signature] = len(routes)
            routes.append(route)
        key = transition_key(row["state"], row["event"])
        route_id = route_ids[signature]
        if addresses.setdefault(key, route_id) != route_id:
            raise ValueError(f"Conflicting decision table entries for {key}.")
    state, event = args["state"], args["event"]
    key = transition_key(state, event)
    # Resolve just the selected route for the direct-transition operator.
    table = {key: routes[addresses[key]]} if key in addresses else {}
    result = context.run(parameters["transition"], {"state": state, "event": event, "table": table})
    return {"addresses": addresses, "routes": routes, "entry_count": len(addresses),
            "route_count": len(routes), "result": result}


def layout(records, requested, block_size):
    offset, offsets, blocks = 0, {}, set()
    for name, size in records:
        offsets[name] = offset
        if name in requested and size:
            blocks.update(range(offset // block_size, (offset + size - 1) // block_size + 1))
        offset += size
    return {"offsets": offsets, "blocks": sorted(blocks), "block_count": len(blocks), "bytes": offset}


def disk_locality(args, _parameters, _context):
    block_size = args["block_size"]
    if type(block_size) is not int or block_size < 1:
        raise ValueError("Block size must be a positive number of bytes.")
    records, groups = {}, {}
    for record in args["records"]:
        name = record["id"]
        if name in records:
            raise ValueError(f"Duplicate record {name!r}.")
        records[name] = len(record["payload"].encode("utf-8"))
        groups.setdefault(record["group"], []).append(name)
    requested = set(args["read"])
    if requested - records.keys():
        raise ValueError("Requested records must be declared.")
    before = layout(records.items(), requested, block_size)
    grouped = ((name, records[name]) for names in groups.values() for name in names)
    after = layout(grouped, requested, block_size)
    return {"kind": "layout simulation", "before": before, "after": after,
            "bytes_saved": before["bytes"] - after["bytes"]}


def physical_triggers(args, parameters, context):
    event = args["event"]
    kind = event["kind"]
    if kind not in parameters["kinds"]:
        raise ValueError(f"Unsupported trigger kind {kind!r}.")
    spec = args["triggers"].get(kind, {}).get(event["name"])
    result = {"kind": "local simulation", "dispatched": spec is not None}
    if spec is not None:
        scope = {"event": event, "state": args.get("state", {})}
        result["result"] = context.run(spec["operation"], substitute(spec.get("input", {}), scope))
    return result


def distinct_values(values):
    return len({canonical(value) for value in values}) > 1


def same_words_different_save(args, parameters, context):
    event = args["event"]
    results = {
        label: context.run(parameters["resume"], branch_arguments(args, save, event))
        for label, save in args["saves"].items()
    }
    answers = {label: value["state"][parameters["answer"]] for label, value in results.items()}
    return {"event": event, "answers": answers, "different": distinct_values(answers.values())}


def new_word_existing_operation(args, parameters, context):
    operation = parameters["encounter"]
    symbol = args["symbol"]
    lookup = {"save": args["save"], "encounters": [], "query": symbol}
    before = context.run(operation, lookup)["meaning"]
    association = {"symbol": symbol, "meaning": args["meaning"]}
    learned = context.run(operation, lookup | {"encounters": [association]})
    later = context.run(operation, lookup | {"save": learned["save"]})
    return {"before": before, "save": later["save"], "later": later["meaning"]}


def one_changed_variable(args, parameters, context):
    rules = load_rules(asset_path(context.root, parameters["rules"]))
    situation = args["situation"]
    if isinstance(situation, str):
        situation = json.loads(context.read_asset(situation))
    added = args["added_observation"]
    changed = situation | {"observed": [*situation["observed"], added]}
    return {"added_observation": added, "before": evaluate_situation(rules, situation),
            "after": evaluate_situation(rules, changed)}


def competing_viewpoints(args, parameters, context):
    perspectives = {
        name: context.run(parameters["perspective"], {"participant": name, "records": args["records"]})
        for name in args["participants"]
    }
    answers = {name: view["knowledge"].get(args["fact"]) for name, view in perspectives.items()}
    return {"answers": answers, "perspectives": perspectives,
            "different": distinct_values(answer["value"] if answer is not None else None for answer in answers.values())}


def restore_and_replay(args, parameters, context):
    save = load_save(args, context)
    serialized = canonical(save)
    operation = parameters["resume"]
    event = args.get("event", {})
    original = context.run(operation, branch_arguments(args, save, event))
    restored = context.run(operation, branch_arguments(args, json.loads(serialized), event))
    return {"equal": canonical(original) == canonical(restored), "original": original,
            "restored": restored, "serialized_bytes": len(serialized.encode("utf-8"))}


def measure_the_work(args, _parameters, context):
    own_trace = not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    try:
        initial_memory, initial_peak = tracemalloc.get_traced_memory()
        initial_operations, initial_bytes = context.operations_run, context.bytes_read
        start = perf_counter_ns()
        result = context.run(args["operation"], args["input"])
        elapsed_ns = perf_counter_ns() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if own_trace:
            tracemalloc.stop()
    return {"result": result, "elapsed_ns": elapsed_ns, "operations": context.operations_run - initial_operations,
            "logical_bytes_read": context.bytes_read - initial_bytes,
            "peak_python_bytes": peak - initial_memory if own_trace else peak - initial_peak,
            "allocation_scope": "operation window" if own_trace else "increase over existing tracer peak"}


OPERATORS = {
    "resume_addresses": resume_addresses,
    "state_differences": state_differences,
    "branching_saves": branching_saves,
    "shared_states": shared_states,
    "context_lifetimes": context_lifetimes,
    "computed_capabilities": computed_capabilities,
    "identity_through_state_changes": identity_through_state_changes,
    "compact_decision_tables": compact_decision_tables,
    "disk_locality": disk_locality,
    "physical_triggers": physical_triggers,
    "same_words_different_save": same_words_different_save,
    "new_word_existing_operation": new_word_existing_operation,
    "one_changed_variable": one_changed_variable,
    "competing_viewpoints": competing_viewpoints,
    "restore_and_replay": restore_and_replay,
    "measure_the_work": measure_the_work,
}
