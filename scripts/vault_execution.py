import json
import operator
from collections import deque
from fractions import Fraction

from arithmetic import compile_value


COMPARISONS = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
    "le": operator.le, "gt": operator.gt, "ge": operator.ge,
}
NUMBER_TYPES = {int, float, Fraction}


def get_value(scope, path):
    value = scope
    for component in path.split("."):
        if isinstance(value, dict):
            value = value[component]
        elif isinstance(value, list) and component.isdecimal():
            value = value[int(component)]
        else:
            raise KeyError(path)
    return value


def substitute(value, scope):
    if isinstance(value, dict):
        if value.keys() == {"$"}:
            return get_value(scope, value["$"])
        return {key: substitute(item, scope) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, scope) for item in value]
    return value


def holds(condition, scope):
    if "all" in condition:
        return all(holds(item, scope) for item in condition["all"])
    if "any" in condition:
        return any(holds(item, scope) for item in condition["any"])
    if "not" in condition:
        return not holds(condition["not"], scope)
    try:
        value = get_value(scope, condition["path"])
        expected = substitute(condition["value"], scope)
    except (KeyError, IndexError):
        return False
    numeric = condition.get("numeric", False)
    mixed = ((type(value) in NUMBER_TYPES and isinstance(expected, str))
             or (type(expected) in NUMBER_TYPES and isinstance(value, str)))
    if numeric or mixed:
        try:
            value, expected = Fraction(str(value)), Fraction(str(expected))
        except (ValueError, ZeroDivisionError):
            if numeric:
                raise ValueError("A numeric condition needs two finite numbers.") from None
    return COMPARISONS[condition.get("op", "eq")](value, expected)


def number(value):
    return value.numerator if value.denominator == 1 else str(value)


def arithmetic_value(expression, inputs):
    rule = compile_value(expression)
    return number(rule.evaluate({name: Fraction(str(inputs[name])) for name in rule.inputs}))


def executable_links(args, parameters, context):
    state = dict(args.get("state", {}))
    cursor = args.get("cursor", args.get("entry"))
    goal = args.get("c")
    event = args.get("event", {})
    max_steps = args.get("max_steps", parameters["max_steps"])
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("An execution needs a positive step allowance.")
    trace = []
    visited = set()
    while True:
        scope = {"state": state, "event": event}
        if goal is not None and holds(goal, scope):
            status = "reached"
            break
        if cursor is None:
            status = "stopped"
            break
        if len(trace) == max_steps:
            status = "paused"
            break
        point = (cursor, json.dumps(state, sort_keys=True))
        if point in visited:
            status = "cycle"
            break
        visited.add(point)
        instruction = args["links"][cursor]
        if "when" in instruction and not holds(instruction["when"], scope):
            status = "waiting"
            break
        values = substitute(instruction.get("input", {}), scope)
        result = context.run(instruction["operation"], values)
        writes = substitute(instruction.get("write", {}), scope | {"result": result})
        next_cursor = instruction.get("next")
        trace.append({"address": cursor, "operation": instruction["operation"], "input": values,
                      "result": result, "writes": writes, "next": next_cursor})
        state = state | writes
        cursor = next_cursor
    return {"status": status, "cursor": cursor, "state": state, "steps": trace}


def arithmetic_dependencies(args, parameters, context):
    rule = compile_value(args["expression"])
    operands = {name: context.operand(args["operands"][name]) for name in rule.inputs}
    values = {name: Fraction(str(value)) for name, value in operands.items()}
    return {"value": number(rule.evaluate(values)), "operands": operands, "read": list(operands)}


def goal_conditions(args, parameters, context):
    return {"reached": holds(args["c"], {"state": args["state"], "event": args.get("event", {})})}


def transition_key(state, event):
    return json.dumps([state, event], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def direct_transitions(args, parameters, context):
    key = transition_key(args["state"], args["event"])
    entry = args["table"].get(key)
    if entry is None:
        return {"found": False, "state": args["state"]}
    return {"found": True, "state": entry["next"], "reply": entry.get("reply")}


def parameterized_transitions(args, parameters, context):
    state = args["state"]
    values = state | args.get("values", {})
    value = arithmetic_value(parameters["expression"], values)
    return {"state": state | {parameters["target"]: value}, "value": value}


def effect_value(effect, state):
    if "expression" in effect:
        return arithmetic_value(effect["expression"], state)
    return effect["value"]


def ordered_effects(args, parameters, context):
    state = dict(args["state"])
    writes = []
    for effect in args["effects"]:
        name = effect["target"]
        state[name] = effect_value(effect, state)
        writes.append({"target": name, "value": state[name]})
    return {"state": state, "writes": writes}


def parallel_effects(args, parameters, context):
    state = args["state"]
    writes = {}
    conflicts = set()
    for effect in args["effects"]:
        name = effect["target"]
        value = effect_value(effect, state)
        if name in writes and writes[name] != value:
            conflicts.add(name)
        writes[name] = value
    return {"state": state | {name: value for name, value in writes.items() if name not in conflicts},
            "conflicts": sorted(conflicts)}


def rule_composition(args, parameters, context):
    outputs = {}
    scope = {"input": args["input"], "outputs": outputs}
    for step in args["steps"]:
        name = step["name"]
        if name in outputs:
            raise ValueError(f"A composed step has duplicate name {name!r}.")
        outputs[name] = context.run(step["operation"], substitute(step["input"], scope))
    return {"value": substitute(args["result"], scope), "outputs": outputs}


def traceable_execution(args, parameters, context):
    start = len(context.trace)
    output = context.run(args["operation"], args["input"])
    return {"result": output, "trace": context.trace[start:]}


def cycle_detection(args, parameters, context):
    graph = args["links"]
    start, goal = args["start"], args["c"]
    previous = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in previous:
                previous[neighbor] = node
                queue.append(neighbor)

    route = []
    if goal in previous:
        node = goal
        while node is not None:
            route.append(node)
            node = previous[node]
        route.reverse()

    active, visited, edges = {start}, {start}, []
    stack = [(start, iter(graph.get(start, [])))]
    while stack:
        node, neighbors = stack[-1]
        neighbor = next(neighbors, None)
        if neighbor is None:
            active.remove(node)
            stack.pop()
        elif neighbor in active:
            edges.append([node, neighbor])
        elif neighbor not in visited:
            visited.add(neighbor)
            active.add(neighbor)
            stack.append((neighbor, iter(graph.get(neighbor, []))))
    return {"reached": bool(route), "route": route, "cycle_edges": edges}


OPERATORS = {
    "executable_links": executable_links,
    "arithmetic_dependencies": arithmetic_dependencies,
    "goal_conditions": goal_conditions,
    "direct_transitions": direct_transitions,
    "parameterized_transitions": parameterized_transitions,
    "ordered_effects": ordered_effects,
    "parallel_effects": parallel_effects,
    "rule_composition": rule_composition,
    "traceable_execution": traceable_execution,
    "cycle_detection": cycle_detection,
}
