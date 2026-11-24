import argparse
import json
import re
import sqlite3
from contextlib import closing
from decimal import Decimal
from functools import lru_cache
from itertools import product
from pathlib import Path

from vocabulary_links import DEFAULT_DATABASE, link_key, resolve


RESPONSE_PART = re.compile(r"\[\[([^\[\]]*)\]\]")
RESPONSE_SCHEMA = """
    CREATE TABLE response_edges (
        context TEXT NOT NULL,
        reply TEXT NOT NULL,
        sequence_id INTEGER NOT NULL REFERENCES nodes(id),
        occurrences INTEGER NOT NULL CHECK (occurrences > 0),
        PRIMARY KEY (context, reply, sequence_id)
    ) WITHOUT ROWID
"""


def adjacent_pairs(phrase):
    words = phrase.split()
    pairs = set()
    for split in range(1, len(words)):
        contexts = [" ".join(words[start:split]) for start in range(split)]
        replies = [" ".join(words[split:end]) for end in range(split + 1, len(words) + 1)]
        pairs.update(product(contexts, replies))
    return pairs


def build_response_index(path):
    uri = Path(path).resolve().as_uri() + "?mode=rw"
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as db:
        db.execute("PRAGMA cache_size = -65536")
        db.execute("PRAGMA temp_store = FILE")
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DROP TABLE IF EXISTS response_edges")
            db.execute(RESPONSE_SCHEMA)
            sequences = db.execute("""
                SELECT r.node_id, n.key, SUM(r.occurrences)
                FROM raw_entries AS r
                JOIN nodes AS n ON n.id = r.node_id
                WHERE r.kind = 'sequence'
                GROUP BY r.node_id
            """)
            db.executemany(
                "INSERT INTO response_edges VALUES (?, ?, ?, ?)",
                (
                    (context, reply, node_id, occurrences)
                    for node_id, sequence, occurrences in sequences
                    for context, reply in adjacent_pairs(sequence)
                ),
            )
            edges = db.execute("SELECT COUNT(*) FROM response_edges").fetchone()[0]
            pairs = db.execute("""
                SELECT COUNT(*) FROM (SELECT context, reply FROM response_edges GROUP BY context, reply)
            """).fetchone()[0]
            keys = db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    return {
        "vocabulary_keys": keys,
        "ordered_one_key_pairs": str(Decimal(keys**2)),
        "continuation_evidence_links": edges,
        "evidenced_pairs": pairs,
    }


def format_response(keys):
    encoded = []
    for key in keys:
        value = json.dumps(key, ensure_ascii=False)[1:-1]
        value = value.replace("[", r"\u005b").replace("]", r"\u005d")
        encoded.append("[[" + value + "]]")
    return "{@" + " ".join(encoded) + "}"


def parse_response(link):
    message = "Use a response link such as {@[[hello]]} or {@[[to]] [[go]]}."
    if not link.startswith("{@") or not link.endswith("}"):
        raise ValueError(message)
    body = link[2:-1]
    keys = []
    position = 0
    for match in RESPONSE_PART.finditer(body):
        if body[position:match.start()].strip():
            raise ValueError(message)
        keys.append(json.loads('"' + match[1] + '"'))
        position = match.end()
    if not keys or body[position:].strip():
        raise ValueError(message)
    return keys


def stored_continuations(db, prompt, limit, reply=None):
    words = link_key(prompt).split()
    for start in range(len(words)):
        context = " ".join(words[start:])
        rows = db.execute("""
            WITH matches AS (
                SELECT reply, SUM(occurrences) AS count, MIN(sequence_id) AS example
                FROM response_edges
                WHERE context = :context AND (:reply IS NULL OR reply = :reply)
                GROUP BY reply
                ORDER BY count DESC, reply
                LIMIT :limit
            )
            SELECT m.reply, m.count, (
                SELECT r.raw FROM raw_entries AS r
                WHERE r.node_id = m.example AND r.kind = 'sequence'
                ORDER BY r.raw LIMIT 1
            ), n.id
            FROM matches AS m
            LEFT JOIN nodes AS n ON n.key = m.reply
            ORDER BY m.count DESC, m.reply
        """, {"context": context, "reply": reply, "limit": limit}).fetchall()
        if rows:
            return context, [
                {
                    "link": format_response(text.split() if node_id is None else [text]),
                    "reply": text,
                    "evidence_count": count,
                    "example": example,
                }
                for text, count, example, node_id in rows
            ]
    return None, []


def response_page(db, input_length, length, offset, limit):
    size, first, last = db.execute("SELECT COUNT(*), MIN(id), MAX(id) FROM nodes").fetchone()
    if size == 0 or last - first + 1 != size:
        raise ValueError("The raw vocabulary needs a contiguous range of link IDs.")
    combinations = size**length
    if offset > combinations:
        raise ValueError("The response offset exceeds the number of combinations.")

    @lru_cache(maxsize=4096)
    def key_at(digit):
        return db.execute("SELECT key FROM nodes WHERE id = ?", (first + digit,)).fetchone()[0]

    responses = []
    for position in range(offset, min(offset + limit, combinations)):
        ordinal = position
        keys = []
        for _ in range(length):
            ordinal, digit = divmod(ordinal, size)
            keys.append(key_at(digit))
        responses.append({
            "position": str(Decimal(position)),
            "link": format_response(reversed(keys)),
        })
    return {
        "mode": "combinations",
        "vocabulary_keys": size,
        "input_length": input_length,
        "response_length": length,
        "responses_per_input": str(Decimal(combinations)),
        "ordered_pairs": str(Decimal(size**input_length * combinations)),
        "responses": responses,
    }


def respond(db, selectors, *, length=None, offset=0, limit=20, reply=None, separator=" "):
    keys = [resolve(db, selector)[1] for selector in selectors]
    if not keys:
        raise ValueError("Supply at least one input link.")
    if limit == 0 or limit < -1:
        raise ValueError("Use a positive limit, or -1 for all stored continuations.")
    if offset < 0 or (offset and length is None):
        raise ValueError("An offset must be nonnegative and used with --length.")
    if length is not None and (length < 1 or limit < 1 or reply is not None):
        raise ValueError("A combination page needs a positive length and limit.")

    prompt = separator.join(keys)
    result = {"prompt": prompt, "input_links": keys}
    if length is not None:
        return result | response_page(db, len(keys), length, offset, limit)
    if reply is not None:
        reply_keys = [resolve(db, "[[" + key + "]]")[1] for key in parse_response(reply)]
        reply_text = " ".join(reply_keys)
        context, matches = stored_continuations(db, prompt, 1, link_key(reply_text))
        return result | {
            "mode": "pair",
            "link": format_response(reply_keys),
            "reply": reply_text,
            "matched_context": context,
            "evidence_count": matches[0]["evidence_count"] if matches else 0,
            "example": matches[0]["example"] if matches else None,
        }

    context, matches = stored_continuations(db, prompt, limit)
    return result | {"mode": "continuations", "matched_context": context, "responses": matches}


def decimal_position(value):
    if not value.isdecimal():
        raise argparse.ArgumentTypeError("Use a nonnegative integer position.")
    return int(Decimal(value))


def main():
    parser = argparse.ArgumentParser(description="Index continuations and address any response combination.")
    parser.add_argument("--database", default=DEFAULT_DATABASE, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="index the continuations in the stored phrases")
    query = commands.add_parser("respond", help="look up responses to a combination of links")
    query.add_argument("entries", nargs="+")
    query.add_argument("--limit", type=int, default=20)
    query.add_argument("--join", default=" ", help="text inserted between input links")
    query.add_argument("--offset", type=decimal_position, default=0, help="zero-based position in a combination page")
    mode = query.add_mutually_exclusive_group()
    mode.add_argument("--length", type=int, help="page through every possible response of this link count")
    mode.add_argument("--reply", help="inspect a specific {@[[response]]} link")
    args = parser.parse_args()

    try:
        if args.command == "build":
            result = build_response_index(args.database)
        else:
            uri = args.database.resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as db:
                result = respond(
                    db, args.entries, length=args.length, offset=args.offset,
                    limit=args.limit, reply=args.reply, separator=args.join,
                )
    except (OSError, ValueError, sqlite3.Error) as error:
        if str(error) == "no such table: response_edges":
            parser.exit(1, "Run response_links.py build to index the stored phrase continuations.\n")
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
