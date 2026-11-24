"""Rule primitives for bounded language and discourse operations."""

from copy import deepcopy
from datetime import date, timedelta
import re


def words(text):
    return set(re.findall(r"\w+", text.casefold()))


def unique_best(scores):
    best = max(scores.values(), default=0)
    winners = [key for key, score in scores.items() if score == best]
    return winners[0] if best > 0 and len(winners) == 1 else None


def sense(a, p, _context):
    observed = words(a["context"])
    candidates = p["words"].get(a["word"].casefold(), {})
    scores = {name: len(observed.intersection(cues)) for name, cues in candidates.items()}
    return {"sense": unique_best(scores), "scores": scores}


def reference(a, p, _context):
    attributes = {**p["references"].get(a["reference"].casefold(), {}), **a.get("attributes", {})}
    candidates = [
        item for item in a["mentions"]
        if all(item.get(field) == value for field, value in attributes.items())
    ]
    scores = {item["id"]: item["position"] + 1 for item in candidates}
    return {"antecedent": unique_best(scores)}


def deixis(a, p, _context):
    resolved = {}
    for token in a["tokens"]:
        source = p["sources"].get(token)
        if source in a["setting"]:
            resolved[token] = a["setting"][source]
        elif token in p["days"]:
            anchor = date.fromisoformat(a["setting"]["date"])
            resolved[token] = (anchor + timedelta(days=p["days"][token])).isoformat()
        else:
            resolved[token] = None
    return resolved


def event(a, p, _context):
    for template in p["templates"]:
        match = re.fullmatch(template["pattern"], a["sentence"].strip().rstrip(".!?"), re.I)
        if match:
            return {
                "event": template["event"],
                "roles": {key: value.casefold() for key, value in match.groupdict().items()},
            }
    return {"event": None, "roles": {}}


def compound(a, p, _context):
    head, modifier = a["parts"][-1], " ".join(a["parts"][:-1])
    relation = p["relations"].get(a["relation"])
    return {
        "head": head, "modifier": modifier, "relation": a["relation"],
        "gloss": relation.format(head=head, modifier=modifier) if relation else None,
    }


def idiom(a, p, _context):
    context = words(a["context"])
    readings = p["idioms"].get(a["phrase"].casefold(), {})
    scores = {kind: len(context.intersection(spec["cues"])) for kind, spec in readings.items()}
    reading = unique_best(scores)
    return {"reading": reading, "meaning": readings[reading]["meaning"] if reading else None}


def negation(a, p, _context):
    result = deepcopy(a["statement"])
    target = result
    for step in a["scope"]:
        target = target[step]
    polarity = p["polarity_field"]
    target[polarity] = not target.get(polarity, True)
    return result


def presuppositions(a, p, _context):
    facts = []
    for trigger in p["triggers"]:
        match = re.fullmatch(trigger["pattern"], a["sentence"].strip().rstrip("?."), re.I)
        if match:
            facts.append({
                "predicate": trigger["predicate"],
                "roles": {key: value.casefold() for key, value in match.groupdict().items()},
            })
    return {"presupposed": facts}


def translate(a, p, _context):
    variants = p["variants"].get(a["language"], {})
    text = a["sentence"].strip().rstrip(".!?").casefold()
    fact = variants.get(text)
    return {"fact": p["facts"].get(fact)}


def classify(a, p):
    for rule in p["patterns"]:
        match = re.fullmatch(rule["pattern"], a["utterance"].strip().rstrip(".!?"), re.I)
        if match:
            return rule["route"], match.group("subject")
    return None, None


def speech_act(a, p, _context):
    route, subject = classify(a, p)
    return {"act": route, "subject": subject}


def intent(a, p, _context):
    route, subject = classify(a, p)
    return {"intent": route, "goal": subject, "fields": p["fields"].get(route, [])}


def constrain(a, p, _context):
    rule = p["constraints"][a["constraint"]]
    rows = a["results"][:rule.get("limit")]
    return [{key: item[key] for key in rule["fields"] if key in item} for item in rows]


def missing(a, p, _context):
    return {"missing": [
        field for field in p["required"][a["operation"]]
        if a["values"].get(field) is None or a["values"][field] == ""
    ]}


def clarify(a, p, _context):
    options = sorted(
        p["questions"], key=lambda item: item["priority"],
    )
    for option in options:
        field = option["field"]
        outcomes = {item["outcome"] for item in a["alternatives"] if field in item}
        possible_values = {item[field] for item in a["alternatives"] if field in item}
        if (a["known"].get(field) is None and len(outcomes) > 1
                and len(possible_values) > 1
                and all(len({item["outcome"] for item in a["alternatives"]
                             if item.get(field) == value}) == 1 for value in possible_values)):
            return {"field": field, "question": option["question"]}
    return {"field": None, "question": None}


def indirect(a, p, _context):
    cues = words(a["utterance"])
    actions = {
        item["action"]: len(item["cues"])
        for item in p["rules"]
        if item["setting"] == a["setting"] and item["intent"] == a["intent"]
        and set(item["cues"]).issubset(cues)
    }
    action = unique_best(actions)
    return {"act": p["request_label"] if action else p["default_label"], "action": action}


def commitment(a, p, _context):
    state = p["initial"]
    history = []
    for item in a["events"]:
        destination = p["transitions"].get(state, {}).get(item["event"])
        if destination is None:
            raise ValueError(f"Invalid commitment event {item['event']!r} from {state!r}")
        state = destination
        history.append(state)
    return {"state": state, "history": history}


def correction(a, p, _context):
    records = deepcopy(a["facts"])
    change = a["correction"]
    history = []
    for record in records:
        if all(record.get(key) == change[key] for key in p["identity"]):
            field = change["field"]
            if field in p["editable"]:
                history.append({**{key: record[key] for key in p["identity"]},
                                "field": field, "before": record.get(field), "after": change["value"]})
                record[field] = change["value"]
    return {"facts": records, "history": history}


def topic(a, p, _context):
    active = a["initial"]
    bookmarks = deepcopy(a.get("bookmarks", {}))
    for item in a["moves"]:
        kind = item["type"]
        if kind == p["leave"]:
            bookmarks[active] = item["checkpoint"]
        elif kind == p["resume"] and item["topic"] not in bookmarks:
            raise ValueError(f"Cannot resume unbookmarked topic {item['topic']!r}")
        elif kind not in (p["leave"], p["resume"]):
            raise ValueError(f"Unknown topic move {kind!r}")
        active = item["topic"]
    return {"topic": active, "checkpoint": bookmarks.get(active), "bookmarks": bookmarks}


def reply(a, p, _context):
    components = {**a.get("evidence", {}), "conclusion": a["conclusion"]}
    rule = p["formats"][a["format"]]
    missing_fields = [field for field in rule["fields"] if components.get(field) in (None, "")]
    return {"format": a["format"], "missing": missing_fields,
            "text": None if missing_fields else rule["template"].format_map(components)}

def encounter(a, p, _context):
    save = deepcopy(a["save"])
    bindings = save.setdefault(p["bindings"], {})
    for association in a["encounters"]:
        bindings[association["symbol"]] = association["meaning"]
    return {"save": save, "meaning": bindings.get(a["query"])}


def observation(a, p, _context):
    claim = a["claim"]
    by_status = {name: [] for name in p["statuses"].values()}
    for record in a["observations"]:
        if record["subject"] != claim["subject"] or record["property"] != claim["property"]:
            continue
        same_value = record["value"] == claim["value"]
        label = p["statuses"]["agree" if same_value else "disagree"]
        by_status[label].append(record["source"])
    return {"claim": claim, **by_status}


def conflict(a, p, _context):
    accounts = {}
    for record in a["accounts"]:
        if all(record.get(field) == value for field, value in a["topic"].items()):
            accounts.setdefault(record["value"], []).append(record["source"])
    pairs = [{"value": value, p["support_field"]: sources} for value, sources in accounts.items()]
    return {"conflict": len(pairs) > 1, "accounts": pairs}


def unresolved(a, p, _context):
    records = [item for item in a["evidence"]
               if item["question"] == a["question"] and item["kind"] in p["accepted_evidence"]]
    values = [item["value"] for item in records]
    if values and all(value == values[0] for value in values):
        return {"status": p["resolved"], "value": values[0]}
    return {"status": p["unknown"], "value": None, "revisit": a["question"]}


def participant_knowledge(records, participant, p):
    knowledge = {}
    for entry in records:
        if entry["participant"] == participant and entry["kind"] in p["access"]:
            rank = p["access"][entry["kind"]]
            earlier = knowledge.get(entry["fact"])
            if earlier is None or rank >= earlier["rank"]:
                knowledge[entry["fact"]] = {
                    "value": entry["value"], "basis": entry["kind"], "rank": rank,
                }
    return {fact: {"value": record["value"], "basis": record["basis"]}
            for fact, record in knowledge.items()}


def perspective(a, p, _context):
    return {"participant": a["participant"],
            "knowledge": participant_knowledge(a["records"], a["participant"], p)}


def beliefs(a, p, _context):
    views = a["perspectives"]
    for participant in a["path"]:
        view = views.get(participant)
        if view is None:
            return {"status": p["unknown"], "value": None}
        views = view.get("perspectives", {})
    facts = view.get("facts", {}) if a["path"] else a.get("facts", {})
    proposition = a["proposition"]
    if proposition["fact"] not in facts:
        return {"status": p["unknown"], "value": None}
    return {"status": p["known"], "value": facts[proposition["fact"]] == proposition["value"]}


def preferences(a, p, _context):
    result = {}
    for participant, weights in a["preferences"].items():
        scores = {
            option["id"]: sum(option["attributes"].get(attribute, p["missing_value"]) * weight
                              for attribute, weight in weights.items())
            for option in a["options"]
        }
        best = max(scores.values(), default=None)
        result[participant] = {
            "preferred": [option for option, score in scores.items() if score == best],
            "scores": scores,
        }
    return result


def norms(a, p, _context):
    rule = p["rules"].get(a["action"])
    if rule is None:
        return {"expectation": None}
    for exception in rule.get("exceptions", []):
        if all(a["situation"].get(key) == value for key, value in exception["when"].items()):
            return {"expectation": exception["expectation"]}
    return {"expectation": rule["expectation"]}


def relevance(a, p, _context):
    pending = [a["goal"]]
    relevant = set()
    dependencies = {**p.get("dependencies", {}), **a.get("dependencies", {})}
    while pending:
        item = pending.pop()
        if item not in relevant:
            relevant.add(item)
            pending.extend(dependencies.get(item, []))
    return {"facts": [entry for entry in a["facts"] if entry["id"] in relevant]}


def hypothetical(a, p, _context):
    scenario = deepcopy(a["world"])
    scenario.update(deepcopy(a["assumptions"]))
    fired = set()
    while True:
        match = next(
            (index for index, rule in enumerate(p["rules"])
             if index not in fired and all(scenario.get(key) == value and key in scenario
                                           for key, value in rule["when"].items())),
            None,
        )
        if match is None:
            break
        fired.add(match)
        scenario.update(deepcopy(p["rules"][match]["then"]))
    return {"world": deepcopy(a["world"]), "scenario": scenario,
            "consequences": {
                key: value for key, value in scenario.items()
                if key not in a["assumptions"] and (key not in a["world"] or a["world"][key] != value)
            }}


OPERATORS = {
    "sense": sense,
    "reference": reference,
    "deixis": deixis,
    "event": event,
    "compound": compound,
    "idiom": idiom,
    "negation": negation,
    "presuppositions": presuppositions,
    "translate": translate,
    "speech_act": speech_act,
    "intent": intent,
    "constrain": constrain,
    "missing": missing,
    "clarify": clarify,
    "indirect": indirect,
    "commitment": commitment,
    "correction": correction,
    "topic": topic,
    "reply": reply,
    "encounter": encounter,
    "observation": observation,
    "conflict": conflict,
    "unresolved": unresolved,
    "perspective": perspective,
    "beliefs": beliefs,
    "preferences": preferences,
    "norms": norms,
    "relevance": relevance,
    "hypothetical": hypothetical,
}
