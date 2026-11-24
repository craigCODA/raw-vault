import argparse
import json
import operator
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from arithmetic import compile_value
from reasoning_links import DEFAULT_RULES, instantiate, load_rules, matches_term


COMPARISONS = {"<": operator.lt, "<=": operator.le, "=": operator.eq,
               "!=": operator.ne, ">=": operator.ge, ">": operator.gt}
NEGATION = "logic:not"


def fact_key(fact):
    return json.dumps(fact, ensure_ascii=False, separators=(",", ":"))


def opposite(fact):
    if fact[0] == NEGATION:
        return fact[1]
    return [NEGATION, fact]


def fact_variables(fact, allow_negation=True):
    if not isinstance(fact, list) or not fact or not isinstance(fact[0], str) or not fact[0] or fact[0].startswith("$"):
        raise ValueError("A fact needs a fixed predicate followed by its arguments.")
    if fact[0] == NEGATION:
        if not allow_negation or len(fact) != 2:
            raise ValueError("Explicit negation takes a single positive fact.")
        return fact_variables(fact[1], allow_negation=False)
    if any(type(value) not in (str, int, float) for value in fact[1:]):
        raise ValueError("Fact arguments must be strings or numbers.")
    return {value for value in fact[1:] if isinstance(value, str) and value.startswith("$")}


def prepare_defaults(rules):
    defaults = []
    for name, rule in rules.get("defaults", {}).items():
        if rule["relation"] not in rules["relations"] or not rule["when"]:
            raise ValueError(f"Default {name!r} needs a relation and at least one premise.")
        bound = set().union(*(fact_variables(premise) for premise in rule["when"]))
        output = fact_variables(rule["expect"], allow_negation=False)
        exceptions = set().union(*(fact_variables(fact) for fact in rule.get("unless", [])))
        checks = []
        for left, comparison, right in rule.get("checks", []):
            if comparison not in COMPARISONS:
                raise ValueError(f"Unknown comparison {comparison!r}")
            a, b = compile_value(left), compile_value(right)
            output.update("$" + name for name in (*a.inputs, *b.inputs))
            checks.append((a, comparison, b))
        if (output | exceptions) - bound:
            raise ValueError(f"Default {name!r} refers to variables outside its premises.")
        defaults.append((name, rule, checks))
    return defaults


def match_premises(patterns, available, bindings, premises):
    if not patterns:
        yield bindings, premises
        return
    pattern, *remaining = patterns
    for fact in available[pattern[0]]:
        matched = bindings.copy()
        if matches_term(pattern, fact, matched):
            yield from match_premises(remaining, available, matched, [*premises, fact])


def run_check(check, bindings):
    a, comparison, b = check
    names = set(a.inputs) | set(b.inputs)
    values = {name: Fraction(str(bindings["$" + name])) for name in names}
    left, right = a.evaluate(values), b.evaluate(values)
    return {"left": str(left), "comparison": comparison, "right": str(right),
            "holds": COMPARISONS[comparison](left, right)}


def evaluate_situation(rules, situation):
    goal = situation["c"]
    observations = situation["observed"]
    for fact in [goal, *observations]:
        if fact_variables(fact):
            raise ValueError("Observed facts and c must have concrete arguments.")
    observed = {fact_key(fact): fact for fact in observations}
    known = {key: fact for key, fact in observed.items() if fact_key(opposite(fact)) not in observed}
    expected = {}
    blocked = {}
    defaults = prepare_defaults(rules)
    while True:
        available = defaultdict(list)
        for fact in known.values():
            available[fact[0]].append(fact)
        pending = {}
        for name, rule, checks in defaults:
            for bindings, premises in match_premises(rule["when"], available, {}, []):
                fact = instantiate(rule["expect"], bindings)
                key = fact_key(fact)
                if key in known or key in pending:
                    continue
                decisions = [run_check(check, bindings) for check in checks]
                if not all(check["holds"] for check in decisions):
                    continue
                blockers = [opposite(fact), *(instantiate(p, bindings) for p in rule.get("unless", []))]
                blocker = next((item for item in blockers if fact_key(item) in observed), None)
                step = {"fact": fact, "rule": name, "relation": rule["relation"],
                        "premises": premises, "checks": decisions}
                if blocker is not None:
                    blocked[name, key, fact_key(blocker)] = step | {"by": blocker}
                    continue
                pending[key] = fact
                expected[key] = step
        if not pending:
            break
        known.update(pending)

    goal_key, contrary = fact_key(goal), fact_key(opposite(goal))
    if goal_key in observed and contrary in observed:
        status = "conflict"
    elif goal_key in observed:
        status = "observed"
    elif contrary in observed:
        status = "contradicted"
    elif goal_key in expected:
        status = "expected"
    else:
        status = "unknown"
    return {"c": goal, "status": status, "expected": list(expected.values()), "blocked": list(blocked.values())}


def main():
    parser = argparse.ArgumentParser(description="Apply the Vault's stored defaults to observed facts.")
    parser.add_argument("situation", type=Path)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    args = parser.parse_args()
    try:
        rules = load_rules(args.rules)
        situation = json.loads(args.situation.read_text(encoding="utf-8"))
        result = evaluate_situation(rules, situation)
    except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
