import argparse
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from html import escape
from itertools import batched
from pathlib import Path
from unicodedata import normalize
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "raw" / "language.sqlite"
WORD = re.compile(r"\S+")
HAN = re.compile(r"[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]+")

BUILD_PRAGMAS = """
    PRAGMA journal_mode = OFF;
    PRAGMA synchronous = OFF;
    PRAGMA temp_store = FILE;
    PRAGMA cache_size = -65536;
"""

SCHEMA = """
    CREATE TABLE nodes (
        id INTEGER PRIMARY KEY,
        key TEXT NOT NULL UNIQUE
    );
    CREATE TABLE raw_entries (
        node_id INTEGER NOT NULL REFERENCES nodes(id),
        kind TEXT NOT NULL,
        raw TEXT NOT NULL,
        occurrences INTEGER NOT NULL CHECK (occurrences > 0),
        PRIMARY KEY (node_id, kind, raw)
    ) WITHOUT ROWID;
    CREATE TABLE links (
        whole_id INTEGER NOT NULL REFERENCES nodes(id),
        part_id INTEGER NOT NULL REFERENCES nodes(id),
        start_offset INTEGER NOT NULL,
        end_offset INTEGER NOT NULL,
        CHECK (whole_id <> part_id AND start_offset >= 0 AND end_offset > start_offset),
        PRIMARY KEY (whole_id, start_offset, end_offset, part_id)
    ) WITHOUT ROWID;
"""


def link_key(value):
    if value and not value.isspace():
        value = " ".join(value.split()).casefold()
    return normalize("NFC", value)


def literal_parts(key):
    if len(key) <= 1:
        return
    if HAN.fullmatch(key):
        spans = [(offset, offset + 1) for offset in range(len(key))]
    else:
        spans = [match.span() for match in WORD.finditer(key)]
    if len(spans) <= 1:
        return

    for index, (start, stop) in enumerate(spans):
        yield key, key[start:stop], start, stop, 1
        for tail in range(index + 1, len(spans)):
            if index == 0 and tail == len(spans) - 1:
                continue
            end = spans[tail][1]
            yield key, key[start:end], start, end, 0


def stage_language(path, records, extra_links):
    with closing(sqlite3.connect(path)) as db:
        db.executescript(BUILD_PRAGMAS + """
            CREATE TABLE incoming (key TEXT, kind TEXT, raw TEXT);
            CREATE TABLE parts (whole_key TEXT, part_key TEXT, start_offset INTEGER,
                                end_offset INTEGER, is_literal_member INTEGER);
        """)
        with db:
            for batch in batched(records, 20_000):
                entries = [(link_key(value), kind, raw) for raw, value, kind in batch]
                db.executemany("INSERT INTO incoming VALUES (?, ?, ?)", entries)
                db.executemany(
                    "INSERT INTO parts VALUES (?, ?, ?, ?, ?)",
                    (part for key, _, _ in entries for part in literal_parts(key)),
                )
            db.executemany("INSERT INTO parts VALUES (?, ?, ?, ?, ?)", extra_links)


def index_staged_language(path, staged, expected_entries):
    with closing(sqlite3.connect(path)) as db:
        db.executescript(BUILD_PRAGMAS + SCHEMA)
        db.execute("ATTACH DATABASE ? AS staged", (str(staged),))
        db.execute("PRAGMA staged.cache_size = -65536")
        with db:
            db.execute("""
                INSERT INTO nodes (key)
                SELECT key FROM staged.incoming
                UNION
                SELECT part_key AS key FROM staged.parts WHERE is_literal_member
                ORDER BY key
            """)
            db.execute("""
                INSERT INTO raw_entries (node_id, kind, raw, occurrences)
                SELECT n.id, i.kind, i.raw, COUNT(*)
                FROM staged.incoming AS i
                JOIN nodes AS n ON n.key = i.key
                GROUP BY n.id, i.kind, i.raw
                ORDER BY n.id, i.kind, i.raw
            """)
            db.execute("""
                INSERT INTO links (whole_id, part_id, start_offset, end_offset)
                SELECT DISTINCT w.id, p.id, r.start_offset, r.end_offset
                FROM staged.parts AS r
                JOIN nodes AS w ON w.key = r.whole_key
                JOIN nodes AS p ON p.key = r.part_key
                ORDER BY w.id, r.start_offset, r.end_offset, p.id
            """)
            db.execute("CREATE INDEX links_reverse ON links (part_id, whole_id)")
            db.execute("CREATE INDEX raw_spelling ON raw_entries (raw, node_id)")
            db.execute("PRAGMA user_version = 1")

        raw_entries, variants, stored_nodes = db.execute("""
            SELECT COALESCE(SUM(occurrences), 0), COUNT(*), COUNT(DISTINCT node_id)
            FROM raw_entries
        """).fetchone()
        if expected_entries is not None and raw_entries != expected_entries:
            raise ValueError(f"Expected {expected_entries:,} raw entries, found {raw_entries:,}")
        if db.execute("PRAGMA main.quick_check").fetchone() != ("ok",):
            raise ValueError("The vocabulary database failed its integrity check.")
        return {
            "raw_entries": raw_entries,
            "raw_variants": variants,
            "stored_nodes": stored_nodes,
            "nodes": db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
            "links": db.execute("SELECT COUNT(*) FROM links").fetchone()[0],
        }


def build_index(path, records, *, extra_links=(), expected_entries=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / f".language-index-{uuid4().hex}"
    staged = scratch.with_suffix(".staged")
    draft = scratch.with_suffix(".sqlite")
    try:
        stage_language(staged, records, extra_links)
        stats = index_staged_language(draft, staged, expected_entries)
        draft.replace(path)
        return stats
    finally:
        staged.unlink(missing_ok=True)
        draft.unlink(missing_ok=True)


def resolve(db, selector):
    value = selector[2:-2] if selector.startswith("[[") and selector.endswith("]]") else selector
    selected = db.execute("SELECT id, key FROM nodes WHERE key = ?", (link_key(value),)).fetchone()
    if selected is not None:
        return selected
    matches = db.execute("""
        SELECT DISTINCT n.id, n.key FROM raw_entries AS r
        JOIN nodes AS n ON n.id = r.node_id
        WHERE r.raw = ? LIMIT 2
    """, (value,)).fetchall()
    if len(matches) > 1:
        raise ValueError(f"{selector!r} has multiple decoded spellings. Use a decoded link key.")
    if not matches:
        raise ValueError(f"No vocabulary link for {selector!r}")
    return matches[0]


def lookup(db, selector, limit=20):
    node_id, key = resolve(db, selector)
    stored = db.execute("""
        SELECT raw, kind, occurrences FROM raw_entries WHERE node_id = ? ORDER BY kind, raw
    """, (node_id,)).fetchall()
    members = db.execute("""
        SELECT n.key FROM (
            SELECT part_id, MIN(start_offset) AS first_offset
            FROM links WHERE whole_id = ? GROUP BY part_id
            ORDER BY first_offset, part_id LIMIT ?
        ) AS related JOIN nodes AS n ON n.id = related.part_id
        ORDER BY related.first_offset, n.id
    """, (node_id, limit)).fetchall()
    containers = db.execute("""
        SELECT n.key FROM (
            SELECT DISTINCT whole_id FROM links WHERE part_id = ? ORDER BY whole_id LIMIT ?
        ) AS related JOIN nodes AS n ON n.id = related.whole_id
        ORDER BY n.id
    """, (node_id, limit)).fetchall()
    return {
        "key": key,
        "occurrences": sum(count for _, _, count in stored),
        "variants": stored,
        "members": [part[0] for part in members],
        "containers": [whole[0] for whole in containers],
    }


def combine(db, selectors, separator=" ", limit=20):
    nodes = [resolve(db, selector) for selector in selectors]
    if not nodes:
        raise ValueError("Supply at least one vocabulary link.")
    keys = [key for _, key in nodes]
    sequence = separator.join(keys)
    occurrences = db.execute("""
        SELECT COALESCE(SUM(r.occurrences), 0)
        FROM raw_entries AS r JOIN nodes AS n ON n.id = r.node_id
        WHERE n.key = ?
    """, (link_key(sequence),)).fetchone()[0]

    db.execute("CREATE TEMP TABLE IF NOT EXISTS requested (id INTEGER PRIMARY KEY, needed INTEGER)")
    db.execute("DELETE FROM requested")
    counts = Counter(node_id for node_id, _ in nodes)
    db.executemany("INSERT INTO requested VALUES (?, ?)", counts.items())
    shared = db.execute("""
        WITH matches AS (
            SELECT l.whole_id, r.id
            FROM requested AS r
            CROSS JOIN links AS l INDEXED BY links_reverse ON l.part_id = r.id
            GROUP BY l.whole_id, r.id
            HAVING COUNT(*) >= r.needed
        )
        SELECT n.key FROM (
            SELECT whole_id FROM matches
            GROUP BY whole_id HAVING COUNT(*) = ?
            ORDER BY whole_id LIMIT ?
        ) AS related JOIN nodes AS n ON n.id = related.whole_id
        ORDER BY n.id
    """, (len(counts), limit)).fetchall()
    return {
        "sequence": sequence,
        "links": keys,
        "stored_occurrences": occurrences,
        "shared_sequences": [key for key, in shared],
    }


def markdown_note(db, selector, limit):
    result = lookup(db, selector, limit)
    members = result["members"]
    containers = result["containers"]
    keys = list(dict.fromkeys([*members, *containers]))
    node_ids = {
        key: db.execute("SELECT id FROM nodes WHERE key = ?", (key,)).fetchone()[0]
        for key in keys
    }

    def label(value):
        value = json.dumps(value, ensure_ascii=False)[1:-1]
        return escape(value).replace("[", "&#91;").replace("]", "&#93;").replace("|", "&#124;")

    lines = ["# Vocabulary links", "", f"<code>{label(result['key'])}</code>", ""]
    if result["variants"]:
        lines.extend(["## Stored spellings", ""])
        lines.extend(
            f"- <code>{label(raw)}</code> · {kind} × {count:,}"
            for raw, kind, count in result["variants"]
        )
        lines.append("")
    for heading, related in [("Parts", members), ("Appears in", containers)]:
        if related:
            lines.extend([f"## {heading}", ""])
            lines.extend(f"- [[#^entry-{node_ids[key]}|{label(key)}]]" for key in related)
            lines.append("")
    if keys:
        lines.extend(["## Linked entries", ""])
        for key in keys:
            lines.extend([f"<code>{label(key)}</code> ^entry-{node_ids[key]}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Look up and combine the shared raw vocabulary.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("reindex", help="rebuild the SQLite indexes from the vault data")
    links = commands.add_parser("links")
    links.add_argument("entry")
    links.add_argument("--limit", type=int, default=20, help="results per direction, -1 for all")
    links.add_argument("--markdown", type=Path, metavar="FILE", help="write an Obsidian note")
    combinations = commands.add_parser("combine")
    combinations.add_argument("entries", nargs="+")
    combinations.add_argument("--join", default=" ", help="text inserted between link keys")
    combinations.add_argument("--limit", type=int, default=20, help="shared sequences, -1 for all")
    args = parser.parse_args()

    try:
        database_uri = args.database.resolve().as_uri()
        mode = "rw" if args.command == "reindex" else "ro"
        with closing(sqlite3.connect(f"{database_uri}?mode={mode}", uri=True, timeout=30)) as db:
            if args.command == "reindex":
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("REINDEX")
                result = {"database": str(args.database.resolve()), "status": "reindexed"}
            elif args.command == "combine":
                result = combine(db, args.entries, args.join, args.limit)
            elif args.markdown:
                args.markdown.write_text(markdown_note(db, args.entry, args.limit), encoding="utf-8")
                result = {"note": str(args.markdown), "entry": args.entry}
            else:
                result = lookup(db, args.entry, args.limit)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
