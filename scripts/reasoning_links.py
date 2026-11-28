import argparse
import json
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from arithmetic import compile_value
from response_links import decimal_position, format_response
from vault_identity import asset_path
from vocabulary_links import DEFAULT_DATABASE, link_key, resolve


DEFAULT_RULES = Path(__file__).resolve().parents[1] / "reasoning.vault"
BOUNDARIES = {"<s>", "</s>", "<unk>"}
BASE_FAMILIES = ("rhetoric", "logic", "argument")
FAMILIES = (*BASE_FAMILIES, "commonsense")


def load_rules(path, *, read_bytes=Path.read_bytes):
    path = Path(path).resolve()
    rules = json.loads(read_bytes(path).decode("utf-8"))
    if rules["format"] != "vault-reasoning/1" or not rules["relations"]:
        raise ValueError("Expected a reasoning data file with relation definitions.")
    if "commonsense" in rules:
        common_path = asset_path(path.parent, rules["commonsense"])
        common = json.loads(read_bytes(common_path).decode("utf-8"))
        if common["format"] != "vault-commonsense/1" or rules["relations"].keys() & common["relations"].keys():
            raise ValueError("Expected a common-sense data file with distinct relation names.")
        rules["relations"].update(common["relations"])
        rules["defaults"] = common["defaults"]
    for name, definition in rules["relations"].items():
        family, _, label = name.partition(":")
        if family not in FAMILIES or re.fullmatch(r"[a-z][a-z_]*", label) is None:
            raise ValueError(f"Invalid relation name {name!r}")
        if len(definition["roles"]) != 2:
            raise ValueError(f"Relation {name!r} needs two roles.")
        for cue, direction in definition["cues"].items():
            if not cue.split() or link_key(cue) != cue or direction not in {"forward", "reverse"}:
                raise ValueError(f"Invalid cue {cue!r} for {name!r}")
    return rules


def definition_text(definition):
    return json.dumps(definition, sort_keys=True, separators=(",", ":"))


def cue_index(rules):
    queries = defaultdict(list)
    for relation, definition in rules["relations"].items():
        for cue, direction in definition["cues"].items():
            words = tuple(cue.split())
            queries[words[0]].append((relation, cue, words, direction))
    return queries


def matched_cues(text, queries):
    words = tuple(text.split())
    if not BOUNDARIES.isdisjoint(words):
        return
    start = 1
    while start < len(words) - 1:
        matches = []
        for relation, cue, parts, direction in queries.get(words[start], ()):
            stop = start + len(parts)
            if stop < len(words) and words[start:stop] == parts:
                matches.append((stop, relation, cue, direction))
        if not matches:
            start += 1
            continue
        end = max(stop for stop, _, _, _ in matches)
        before = " ".join(words[:start])
        after = " ".join(words[end:])
        for stop, relation, cue, direction in matches:
            if stop == end:
                left, right = (before, after) if direction == "forward" else (after, before)
                yield relation, left, right, cue, start
        start = end


def index_reasoning_endpoints(db):
    db.execute("CREATE INDEX IF NOT EXISTS reasoning_by_left ON reasoning_evidence (left_key, occurrences DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS reasoning_by_right ON reasoning_evidence (right_key, occurrences DESC)")


def build_reasoning_index(path, rules):
    queries = cue_index(rules)
    uri = Path(path).resolve().as_uri() + "?mode=rw"
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as db, db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("BEGIN IMMEDIATE")
        db.execute("DROP TABLE IF EXISTS reasoning_evidence")
        db.execute("DROP TABLE IF EXISTS reasoning_relations")
        db.execute("""
            CREATE TABLE reasoning_relations (
                name TEXT PRIMARY KEY,
                definition TEXT NOT NULL
            ) WITHOUT ROWID
        """)
        db.execute("""
            CREATE TABLE reasoning_evidence (
                relation TEXT NOT NULL REFERENCES reasoning_relations(name),
                left_key TEXT NOT NULL,
                right_key TEXT NOT NULL,
                sequence_id INTEGER NOT NULL REFERENCES nodes(id),
                cue TEXT NOT NULL,
                position INTEGER NOT NULL,
                occurrences INTEGER NOT NULL,
                PRIMARY KEY (relation, left_key, right_key, sequence_id, position)
            ) WITHOUT ROWID
        """)
        db.executemany(
            "INSERT INTO reasoning_relations VALUES (?, ?)",
            [(name, definition_text(definition)) for name, definition in sorted(rules["relations"].items())],
        )
        sequences = db.execute("""
            SELECT n.id, n.key, SUM(r.occurrences)
            FROM raw_entries AS r JOIN nodes AS n ON n.id = r.node_id
            WHERE r.kind = 'sequence' GROUP BY n.id
        """)
        db.executemany(
            "INSERT INTO reasoning_evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                (relation, left, right, sequence_id, cue, index, count)
                for sequence_id, text, count in sequences
                for relation, left, right, cue, index in matched_cues(text, queries)
            ),
        )
        index_reasoning_endpoints(db)
        counts = dict(db.execute("SELECT relation, COUNT(*) FROM reasoning_evidence GROUP BY relation"))
    return {"relation_types": len(rules["relations"]), "cue_links": sum(counts.values()), "counts": counts}


def cue_rows(db, rules, relation, limit, left=None, right=None):
    definition = rules["relations"][relation]
    stored = db.execute("SELECT definition FROM reasoning_relations WHERE name = ?", (relation,)).fetchone()
    if stored != (definition_text(definition),):
        raise ValueError("Rebuild the reasoning index to use the current relation definition.")
    return db.execute("""
        SELECT e.left_key, e.right_key, e.cue, (
            SELECT r.raw FROM raw_entries AS r
            WHERE r.node_id = e.sequence_id AND r.kind = 'sequence'
            ORDER BY r.raw LIMIT 1
        ), e.occurrences
        FROM reasoning_evidence AS e
        WHERE e.relation = :relation
          AND (:left IS NULL OR e.left_key = :left)
          AND (:right IS NULL OR e.right_key = :right)
        ORDER BY e.occurrences DESC, e.left_key, e.right_key, e.sequence_id, e.position
        LIMIT :limit
    """, {"relation": relation, "left": left, "right": right, "limit": limit}).fetchall()


def render_chain(statements, relations):
    parts = [format_response(statements[0])[2:-1]]
    for relation, statement in zip(relations, statements[1:], strict=True):
        parts.extend(["{" + relation + "}", format_response(statement)[2:-1]])
    return " ".join(parts)


def link_entries(db, rules, relation, left, right, limit=20):
    fields = [[resolve(db, item)[1] for item in side] for side in (left, right)]
    if not all(fields):
        raise ValueError("Supply one or more vocabulary links on each side.")
    a, b = (" ".join(side) for side in fields)
    rows = cue_rows(db, rules, relation, limit, link_key(a), link_key(b))
    return {
        "link": render_chain(fields, [relation]),
        "roles": rules["relations"][relation]["roles"],
        "evidence": [{"cue": cue, "example": example, "count": count} for _, _, cue, example, count in rows],
    }


def chain_page(db, rules, *, links, width, offset, limit, family=None):
    if min(links, width, limit) < 1 or offset < 0:
        raise ValueError("Link count, statement width, and limit must be positive. Offset starts at zero.")
    if family is None:
        families = BASE_FAMILIES
    elif family == "all":
        families = FAMILIES
    else:
        families = (family,)
    relation_ids = sorted(name for name in rules["relations"] if name.partition(":")[0] in families)
    if not relation_ids:
        raise ValueError("Choose a family with relation definitions.")
    size, first, last = db.execute("SELECT COUNT(*), MIN(id), MAX(id) FROM nodes").fetchone()
    if not size or size != last - first + 1:
        raise ValueError("The vocabulary needs a contiguous range of link IDs.")
    relation_count = len(relation_ids)
    combinations = size ** (width * (links + 1)) * relation_count**links
    if offset > combinations:
        raise ValueError("The offset exceeds the chain count.")
    bases = [size] * width + ([relation_count] + [size] * width) * links

    @lru_cache(maxsize=4096)
    def key_at(digit):
        return db.execute("SELECT key FROM nodes WHERE id = ?", (first + digit,)).fetchone()[0]

    chains = []
    for position in range(offset, min(offset + limit, combinations)):
        ordinal = position
        digits = []
        for base in reversed(bases):
            ordinal, digit = divmod(ordinal, base)
            digits.append(digit)
        digits.reverse()
        statements = [[key_at(digit) for digit in digits[:width]]]
        relations = []
        for start in range(width, len(digits), width + 1):
            relations.append(relation_ids[digits[start]])
            statements.append([key_at(digit) for digit in digits[start + 1:start + 1 + width]])
        chains.append({
            "position": str(Decimal(position)),
            "statements": statements,
            "relations": relations,
            "link": render_chain(statements, relations),
        })
    return {
        "vocabulary_keys": size,
        "relation_types": relation_count,
        "link_count": links,
        "statement_width": width,
        "combinations": str(Decimal(combinations)),
        "chains": chains,
    }


def evaluate_truth(rules, relation, left, right):
    expression = rules["relations"][relation].get("truth")
    if expression is None or left not in (0, 1) or right not in (0, 1):
        raise ValueError("Choose a relation with a truth expression and provide two values in {0, 1}.")
    result = compile_value(expression).evaluate({"a": Fraction(left), "b": Fraction(right)})
    if result not in (0, 1):
        raise ValueError("The declared truth expression must return 0 or 1.")
    return {"relation": relation, "a": left, "b": right, "expression": expression, "result": int(result)}


def matches_term(pattern, value, bindings):
    if isinstance(pattern, str) and pattern.startswith("$"):
        return bindings.setdefault(pattern, value) == value
    if isinstance(pattern, list):
        return (
            isinstance(value, list) and len(pattern) == len(value)
            and all(matches_term(p, v, bindings) for p, v in zip(pattern, value, strict=True))
        )
    return pattern == value


def instantiate(pattern, bindings):
    if isinstance(pattern, str) and pattern.startswith("$"):
        return bindings[pattern]
    if isinstance(pattern, list):
        return [instantiate(value, bindings) for value in pattern]
    return pattern


def prove(rules, argument):
    terms = list(argument["premises"])
    trace = []
    for step in argument["steps"]:
        rule = rules["inference"][step["rule"]]
        indices = step["from"]
        if any(type(index) is not int or not 0 <= index < len(terms) for index in indices):
            raise ValueError("Each step must refer to a previous term by its zero-based index.")
        bindings = {}
        patterns = rule["premises"]
        if len(patterns) != len(indices) or not all(
            matches_term(pattern, terms[index], bindings) for pattern, index in zip(patterns, indices, strict=True)
        ):
            raise ValueError(f"The selected premises do not match {step['rule']!r}.")
        conclusion = instantiate(rule["conclusion"], bindings)
        trace.append({"index": len(terms), "rule": step["rule"], "from": indices, "conclusion": conclusion})
        terms.append(conclusion)
    return {"c": argument["c"], "reached_c": argument["c"] in terms, "steps": trace}


def main():
    parser = argparse.ArgumentParser(description="Follow typed links and apply the vault's reasoning data.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="rebuild stored cue links")
    examples = commands.add_parser("examples", help="read literal cue matches for a relation")
    examples.add_argument("relation")
    examples.add_argument("--limit", type=int, default=10)
    link = commands.add_parser("link", help="connect two statements")
    link.add_argument("relation")
    link.add_argument("left", nargs="+")
    link.add_argument("--right", nargs="+", required=True)
    link.add_argument("--limit", type=int, default=20)
    chain = commands.add_parser("chain", help="address any ordered chain of typed links")
    chain.add_argument("--links", type=int, default=1)
    chain.add_argument("--width", type=int, default=1, help="vocabulary keys per statement")
    chain.add_argument("--family", choices=(*FAMILIES, "all"), help="select a family or combine all families")
    chain.add_argument("--offset", type=decimal_position, default=0)
    chain.add_argument("--limit", type=int, default=5)
    truth = commands.add_parser("truth", help="evaluate the selected truth operation")
    truth.add_argument("relation")
    truth.add_argument("a", type=int, choices=(0, 1))
    truth.add_argument("b", type=int, choices=(0, 1))
    proof = commands.add_parser("prove", help="apply the steps in an argument data file")
    proof.add_argument("argument", type=Path)
    args = parser.parse_args()

    try:
        rules = load_rules(args.rules)
        if args.command == "build":
            result = build_reasoning_index(args.database, rules)
        elif args.command == "truth":
            result = evaluate_truth(rules, args.relation, args.a, args.b)
        elif args.command == "prove":
            result = prove(rules, json.loads(args.argument.read_text(encoding="utf-8")))
        else:
            uri = args.database.resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as db:
                if args.command == "chain":
                    result = chain_page(db, rules, links=args.links, width=args.width,
                                        offset=args.offset, limit=args.limit, family=args.family)
                elif args.command == "link":
                    result = link_entries(db, rules, args.relation, args.left, args.right, args.limit)
                else:
                    rows = cue_rows(db, rules, args.relation, args.limit)
                    result = {
                        "relation": args.relation,
                        "roles": rules["relations"][args.relation]["roles"],
                        "examples": [
                            {"left": left, "right": right, "cue": cue, "example": example, "count": count}
                            for left, right, cue, example, count in rows
                        ],
                    }
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        if str(error) == "no such table: reasoning_relations":
            parser.exit(1, "Run reasoning_links.py build to index the stored cues.\n")
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
