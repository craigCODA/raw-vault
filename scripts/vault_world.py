"""Evaluate declared world and logic rules over supplied state and finite domains."""

from collections import deque
from fractions import Fraction
from itertools import product
from math import prod

from arithmetic import compile_equation, compile_value
from reasoning_links import instantiate, matches_term
from vault_execution import holds, number, substitute


def exact(value):
    return Fraction(str(value))


def interval(args):
    start, end = exact(args["start"]), exact(args["end"])
    if start > end:
        raise ValueError("An interval must start at or before its end.")
    return start, end


def temporal_position(args, parameters, context):
    start, end = interval(args)
    time = exact(args["time"])
    relation = "before" if time < start else "after" if time > end else "during"
    return {"relation": relation, "offset_from_start": number(time - start)}


def interval_duration(args, parameters, context):
    start, end = interval(args)
    return {"duration": number(end - start), "instantaneous": start == end}


def fact_persistence(args, parameters, context):
    state = dict(args["state"])
    time = exact(args["through"])
    applied = []
    for index, event in sorted(enumerate(args["events"]), key=lambda row: exact(row[1]["time"])):
        if exact(event["time"]) > time:
            break
        state.update(event["writes"])
        applied.append(index)
    return {"state": state, "applied": applied}


def containment_closure(args, parameters, context):
    start = args["item"]
    parents = {}
    for item, container in args["links"]:
        parents.setdefault(item, []).append(container)

    seen = {start}
    pending = deque([start])
    containers = []
    while pending:
        for parent in parents.get(pending.popleft(), []):
            if parent in seen:
                continue
            seen.add(parent)
            containers.append(parent)
            pending.append(parent)
    return {"containers": containers, "contained": "container" in args and args["container"] in containers}


def part_requirements(args, parameters, context):
    parts = args["parts"]
    missing = []
    unsuitable = []
    for requirement in args["requires"]:
        name = requirement["part"]
        if name not in parts:
            missing.append(name)
        elif not holds(requirement.get("when", {"all": []}), parts[name]):
            unsuitable.append(name)
    return {"available": not (missing or unsuitable), "missing": missing, "unsuitable": unsuitable}


def planar_orientation(args, parameters, context):
    turns = args["quarter_turns"]
    if type(turns) is not int:
        raise ValueError("A planar rotation requires a whole number of quarter turns.")
    x, y = map(exact, args["point"])
    origin_x, origin_y = map(exact, args.get("origin", [0, 0]))
    x, y = x - origin_x, y - origin_y
    for _ in range(turns % 4):
        x, y = -y, x
    return {"point": [number(x + origin_x), number(y + origin_y)]}


def unit_capacity(args, parameters, context):
    units = parameters["units"]
    amount, capacity = args["amount"], args["capacity"]
    source, destination = units[amount["unit"]], units[capacity["unit"]]
    if source["dimension"] != destination["dimension"]:
        return {"comparable": False}
    converted = exact(amount["value"]) * exact(source["factor"]) / exact(destination["factor"])
    remaining = exact(capacity["value"]) - converted
    return {"comparable": True, "converted": number(converted), "remaining": number(remaining),
            "unit": capacity["unit"], "fits": remaining >= 0}


def missing_preconditions(action, state):
    return [item["name"] for item in action.get("requires", []) if not holds(item["when"], state)]


def action_preconditions(args, parameters, context):
    missing = missing_preconditions(args["action"], args["state"])
    return {"applicable": not missing, "missing": missing}


def action_effects(args, parameters, context):
    state, action = args["state"], args["action"]
    missing = missing_preconditions(action, state)
    writes = {} if missing else substitute(action["writes"], {"state": state, "inputs": args.get("inputs", {})})
    return {"applied": not missing, "missing": missing, "state": state | writes, "writes": writes}


def available_actions(args, parameters, context):
    available = [name for name, action in args["actions"].items()
                 if not missing_preconditions(action, args["state"])]
    return {"actions": available}


def resource_budget(args, parameters, context):
    budget = {name: exact(value) for name, value in args["budget"].items()}
    applied = []
    blocked = []
    for request in args["requests"]:
        cost = {name: exact(value) for name, value in request["cost"].items()}
        if any(value < 0 for value in cost.values()):
            raise ValueError("Resource costs must be nonnegative.")
        remaining = {name: budget.get(name, Fraction(0)) - value for name, value in cost.items()}
        shortfalls = {name: number(-value) for name, value in remaining.items() if value < 0}
        if shortfalls:
            blocked.append({"request": request["name"], "shortfalls": shortfalls})
        else:
            budget.update(remaining)
            applied.append(request["name"])
    return {"remaining": {name: number(value) for name, value in budget.items()},
            "applied": applied, "blocked": blocked}


def numeric_feedback(args, parameters, context):
    limit = args["steps"]
    if type(limit) is not int or not 0 <= limit <= parameters["max_steps"]:
        raise ValueError("Feedback needs a bounded, nonnegative step count.")
    state = exact(args["start"])
    inputs = {name: exact(value) for name, value in args.get("inputs", {}).items()}
    rule = compile_value(args["expression"])
    goal = args.get("goal")
    history = [number(state)]
    for _ in range(limit):
        if goal is not None and holds(goal, inputs | {"value": state}):
            break
        state = rule.evaluate(inputs | {"value": state})
        history.append(number(state))
    return {"history": history, "value": number(state),
            "reached": goal is not None and holds(goal, inputs | {"value": state})}


def finite_quantifiers(args, parameters, context):
    items = args["items"]
    matches = [item for item in items if holds(args["test"], {"item": item})]
    return {"some": bool(matches), "all": len(matches) == len(items), "none": not matches,
            "exactly_one": len(matches) == 1, "matches": matches}


def bind_relation(args, parameters, context):
    bindings = {}
    matched = matches_term(args["pattern"], args["term"], bindings)
    if not matched:
        return {"matched": False, "bindings": {}}
    return {"matched": True, "bindings": bindings, "result": instantiate(args["result"], bindings)}


def combine_conditions(args, parameters, context):
    return {"satisfied": holds(args["premises"], args["facts"])}


def reverse_equation(args, parameters, context):
    target = args.get("target", "c")
    rule, = compile_equation(args["equation"], (target,))
    known = {name: exact(args["known"][name]) for name in rule.inputs}
    return {"target": target, "value": number(rule.evaluate(known))}


def assignments(domains, limit):
    values = list(domains.values())
    if prod(map(len, values)) > limit:
        raise ValueError("The supplied domains exceed the assignment limit.")
    for choice in product(*values):
        yield dict(zip(domains, choice, strict=True))


def finite_solutions(args, parameters, context):
    solutions = [assignment for assignment in assignments(args["domains"], parameters["max_assignments"])
                 if holds(args["constraint"], assignment)]
    return {"solutions": solutions, "count": len(solutions)}


def counterexample_search(args, parameters, context):
    for assignment in assignments(args["domains"], parameters["max_assignments"]):
        if holds(args["premise"], assignment) and not holds(args["conclusion"], assignment):
            return {"holds_on_domain": False, "witness": assignment}
    return {"holds_on_domain": True, "witness": None}


def compatible_explanations(explanations, observed):
    return {name: predictions for name, predictions in explanations.items()
            if all(key not in predictions or predictions[key] == value for key, value in observed.items())}


def alternative_explanations(args, parameters, context):
    effect = args["effect"]
    observed = args["observed"]
    if effect not in observed:
        raise ValueError("The effect must be included in the supplied observations.")
    possible = compatible_explanations(args["explanations"], observed)
    return {"possible": [name for name, prediction in possible.items() if effect in prediction],
            "unresolved": [name for name, prediction in possible.items() if effect not in prediction]}


def distinguish_observations(args, parameters, context):
    possible = compatible_explanations(args["explanations"], args.get("observed", {}))
    distinguishing = {}
    for key in args["candidates"]:
        predictions = {name: prediction[key] for name, prediction in possible.items() if key in prediction}
        values = list(predictions.values())
        if any(value != values[0] for value in values[1:]):
            distinguishing[key] = predictions
    return {"distinguishing": distinguishing}


def compare_counterfactual(args, parameters, context):
    state, consequence = args["state"], args["consequence"]
    change = args["change"]
    actual = holds(consequence, state)
    hypothetical = holds(consequence, state | {change["key"]: change["value"]})
    return {"actual": actual, "hypothetical": hypothetical, "changed": actual != hypothetical}


def supporting_warrants(warrants, facts, claim):
    return [name for name, warrant in warrants.items()
            if warrant["claim"] == claim and holds(warrant["when"], facts)]


def argument_support(args, parameters, context):
    supported_by = supporting_warrants(args["warrants"], args["facts"], args["claim"])
    return {"supported": bool(supported_by), "supported_by": supported_by}


def objection_repair(args, parameters, context):
    facts = args["facts"]
    rejected = set(args["objection"]["retract"])
    remaining = {name: value for name, value in facts.items() if name not in rejected}
    initial = supporting_warrants(args["warrants"], facts, args["claim"])
    challenged = supporting_warrants(args["warrants"], remaining, args["claim"])
    repaired = supporting_warrants(args["warrants"], remaining | args["repair"], args["claim"])
    return {"initial": initial, "challenged": challenged, "repaired": repaired, "supported": bool(repaired)}


def structural_analogy(args, parameters, context):
    relation, *roles = args["source"]
    relations, correspondences = args["relations"], args["correspondence"]
    missing = list(dict.fromkeys(role for role in roles if role not in correspondences))
    if relation not in relations or missing:
        return {"mapped": False, "missing_roles": missing, "missing_relation": relation not in relations}
    return {"mapped": True, "target": [relations[relation], *(correspondences[role] for role in roles)]}


OPERATORS = {
    operation.__name__: operation
    for operation in (
        temporal_position, interval_duration, fact_persistence, containment_closure,
        part_requirements, planar_orientation, unit_capacity, action_preconditions,
        action_effects, available_actions, resource_budget, numeric_feedback,
        finite_quantifiers, bind_relation, combine_conditions, reverse_equation,
        finite_solutions, counterexample_search, alternative_explanations, distinguish_observations,
        compare_counterfactual, argument_support, objection_repair, structural_analogy,
    )
}
