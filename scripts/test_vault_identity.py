import json
import shutil
import sqlite3
import struct
import unittest
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from vault_identity import identify, read_scalar


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / f".identity-test-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.root.rmdir)
        for name in ("values.bin", "data.sqlite", "identity.vault"):
            self.addCleanup((self.root / name).unlink, missing_ok=True)
        self.pool = self.root / "values.bin"
        self.pool.write_bytes(
            struct.pack("<8sQQQI", b"VLTNUM01", 1, 1, 2, 5)
            + struct.pack("<IH", 0x80000000, 16) + b"-2.00" + b"3\0\0\0\0"
        )
        with closing(sqlite3.connect(self.root / "data.sqlite")) as db, db:
            db.execute("CREATE TABLE entries (occurrences INTEGER)")
            db.executemany("INSERT INTO entries VALUES (?)", [(2,), (3,)])
        self.program = self.root / "identity.vault"
        self.program.write_text(json.dumps({
            "format": "vault-identity/1",
            "values": "values.bin",
            "database": "data.sqlite",
            "identity": {"operation": "sha256", "inputs": ["values", "database"]},
            "measure": {"stored_entries": ["entries", "occurrences"]},
            "compute": {"total": "float32_values + uint16_values + decimal_values + stored_entries"},
        }), encoding="utf-8")

    def test_identity_executes_the_data_declarations(self):
        result = identify(self.program)
        self.assertEqual(result["facts"]["stored_entries"], 5)
        self.assertEqual(result["facts"]["total"], 9)
        self.assertEqual(len(result["identity"]), 64)

    def test_identity_follows_bytes_and_survives_relocation(self):
        copy = self.root / "copy"
        copy.mkdir()
        self.addCleanup(copy.rmdir)
        for filename in ("identity.vault", "values.bin", "data.sqlite"):
            shutil.copyfile(self.root / filename, copy / filename)
            self.addCleanup((copy / filename).unlink)
        self.assertEqual(identify(self.program), identify(copy / "identity.vault"))
        with self.pool.open("r+b") as pool:
            pool.seek(-5, 2)
            pool.write(b"9")
        self.assertNotEqual(identify(self.program)["identity"], identify(copy / "identity.vault")["identity"])

    def test_scalar_addresses_preserve_exact_representation(self):
        self.assertEqual(read_scalar(self.pool, 0)["bits"], "80000000")
        self.assertEqual(read_scalar(self.pool, 1)["value"], 16)
        self.assertEqual(read_scalar(self.pool, 2)["literal"], "-2.00")
        self.assertEqual(read_scalar(self.pool, 3)["literal"], "3")

    def test_identity_data_file_selects_the_fingerprint_inputs(self):
        program = json.loads(self.program.read_text(encoding="utf-8"))
        program["identity"]["inputs"] = ["database"]
        self.program.write_text(json.dumps(program), encoding="utf-8")
        original = identify(self.program)
        with self.pool.open("r+b") as pool:
            pool.seek(-5, 2)
            pool.write(b"9")
        self.assertEqual(identify(self.program), original)

    def test_dependency_data_file_can_be_part_of_the_identity(self):
        rules = self.root / "rules.vault"
        rules.write_text("first", encoding="utf-8")
        self.addCleanup(rules.unlink)
        program = json.loads(self.program.read_text(encoding="utf-8"))
        program["assets"] = {"rules": rules.name}
        program["identity"]["inputs"].append("rules")
        self.program.write_text(json.dumps(program), encoding="utf-8")
        original = identify(self.program)
        rules.write_text("second", encoding="utf-8")
        self.assertNotEqual(identify(self.program)["identity"], original["identity"])

    def test_data_file_cannot_select_an_asset_outside_the_vault(self):
        program = json.loads(self.program.read_text(encoding="utf-8"))
        program["database"] = "../external.sqlite"
        self.program.write_text(json.dumps(program), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inside the vault"):
            identify(self.program)


if __name__ == "__main__":
    unittest.main()
