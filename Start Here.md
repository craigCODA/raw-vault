# Raw Vault

`identity.vault` declares what the Vault reads and computes to identify itself. It measures its own files, runs the declared arithmetic, and returns a content fingerprint and measured facts. The fingerprint covers both payload files, the rule files, the operation catalog, and the identity declarations.

The [reasoning links](Reasoning.md) connect statements by rhetoric, logical operations, and argument roles.

The [common-sense rules](<Common Sense.md>) apply everyday expectations to observations in a data file.

The [compiled operations](Operations.md) cover the 80 language, reasoning, execution, save-state, and experiment entries.

Use the [health check](<docs/Vault Health.md>) to verify the shipped payloads and cases. The [runtime guide](docs/Runtime.md) covers graph addresses, traces, replay, saved conversations, and storage profiles. [Training](Training.md) covers concurrent teachers, critics, and published revisions.

Run from the vault folder with Python 3.12 or newer.

```powershell
py -3 -B scripts/vault_identity.py
py -3 -B scripts/vault_identity.py --scalar 0
py -3 -B scripts/vocabulary_links.py links '[[hello]]'
py -3 -B scripts/response_links.py respond '[[hello]]'
```

## Raw data

`raw/values.bin` stores the pooled scalar values with their exact binary bits or decimal spelling. Its 36-byte header uses little-endian `<8sQQQI` and holds a format tag, three record counts, and the fixed byte width of each decimal literal. The sections then store binary32 words, unsigned 16-bit integers, and null-padded ASCII decimal literals. Each section is sorted to give a deterministic address order and retains duplicates.

`raw/language.sqlite` holds the addressable vocabulary, phrase links, and continuations. The `data_records` table holds literal keys as JSON arrays and the associated raw values. The `decimal` encoding uses null-separated ASCII literals. The `float32-le` encoding stores four little-endian bytes per value. Other encodings retain keyed text rows, observed pairs, and pixel bytes. An occurrence count preserves repeated identical rows.
