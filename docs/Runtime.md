# Execute and inspect the vault

Run from the vault directory with Python 3.12 or newer and the hydrated Git LFS payloads.

```powershell
git lfs pull
py -3.12 -B scripts/vault_health.py
py -3.12 -B scripts/vault_operations.py check
```

The health probe covers the original CLIs and all four original examples. It compiles and reindexes a temporary copy for the write probes. See [Vault Health](<Vault Health.md>) for exact comparisons, observed timing and allocation measurements, and performance budgets.

## Runtime graph

`operations.vault` and its declarations bind numbered operations to a module, body, parameters, vocabulary anchors, and cases. The execution bodies live in `runtime/*.vault`. The operation catalog continues to read historic SQLite payloads. The graph compiler assembles each trusted declaration into executable instructions and links references to other bodies. The bootstrap supplies primitives for file and database access, exact arithmetic, and compiled instruction execution.

The runtime graph includes the routine that resolves an operation, binds its frame, invokes its body, and completes the call. Its address and each linked statement can be inspected using the same node command as the operation bodies. For example, open the dispatch node, its first statement, or the transition body.

```powershell
py -3.12 -B scripts/vault_operations.py node body/vault_machine/dispatch
py -3.12 -B scripts/vault_operations.py node body/vault_machine/dispatch/s001
py -3.12 -B scripts/vault_operations.py node body/vault_execution/parameterized_transitions
```

The original 80 IDs resolve to graph bodies, including the two IDs that share the event parser. Direct Python imports of the original operation modules use adapters for existing callers. The compiler and the fixed host primitives form the bootstrap. Learning proposals supply data for the allowed operators through the [training pipeline](../Training.md).

## Trace and replay

`trace` runs an ordinary data file and preserves the complete trace in SQLite. `turn` records every conversation turn in the same store by default. Each record has a content address and links to its input event, execution revision, operation IDs, body and statement addresses, selected links, and state reads and writes. Scalar reads include their original bytes in hexadecimal. File observations retain the bytes used by the execution.

```powershell
$trace = py -3.12 -B scripts/vault_operations.py trace example-execution.vault | ConvertFrom-Json
py -3.12 -B scripts/vault_operations.py replay $trace.trace_id
py -3.12 -B scripts/vault_operations.py backlinks operation/33
py -3.12 -B scripts/vault_operations.py backlinks state/reply
```

Replay checks the executable revision and compares the exact serialized result, observed inputs, and ordered execution events. Records preserve the order of object fields because some operations traverse them. Physical timing and allocation values returned by the measurement operation are recorded as observations and fed into replay. The replay report measures its own fresh execution time.

Use `--trace-store path.sqlite` before the subcommand to choose where traces and session state are stored. The default local store is `.vault-data/traces.sqlite`. Use the ordinary `run` command for the original one-shot result format.

## Saved conversations

`activation.vault` declares the phrase width, active set sizes, message patterns, dispatch targets, templates, and state changes for a turn. Declared routes can read the selected keys and typed relations through `active.keys` and `active.links`. Each selected key and relation has an address. Rebuilt indexes and new training snapshots group relation entries by endpoint for activation. Each turn persists the current context and links to its full trace. The activation limits keep a bounded working set. Exact arithmetic results used by a memoized route are stored with their operation links. Hits and evictions appear in the trace.

```powershell
py -3.12 -B scripts/vault_operations.py turn 'Remember the panel is blue' --session workshop
py -3.12 -B scripts/vault_operations.py turn 'Tell me the panel' --session workshop
py -3.12 -B scripts/vault_operations.py turn 'What did I say it was?' --session workshop
py -3.12 -B scripts/vault_operations.py turn 'Calculate 5 / 4' --session workshop
```

Use `--training` to talk to the published revision of a training directory. Each invocation selects one immutable revision. Conversations retain their own state in the work directory. When a trace belongs to an older publication, replay resolves that publication automatically. `--revision` selects a particular published snapshot for direct inspection.

```powershell
py -3.12 -B scripts/vault_operations.py --training .vault-data/training turn 'calculate 7 / 4' --session experiment
py -3.12 -B scripts/vault_operations.py --training .vault-data/training replay $traceId
```

Republishing vocabulary or operation counts changes the radix of new numeric ordinals. An older numeric ordinal can be interpreted on its original revision. The logical node and operation IDs retain their existing assignments.

New training snapshots carry the Python bootstrap that evaluated them. Run the archived `scripts/vault_operations.py` from a published snapshot using Python 3.12 to execute that copy of the vault. For replay after a bootstrap update, invoke that archived script with the original `--training` work directory and trace ID. Use the snapshot's own `--manifest` and `--database` paths when a source used custom filenames.

## Storage profile

```powershell
py -3.12 -B scripts/vault_operations.py profile example-storage.vault --iterations 5
py -3.12 -B scripts/vault_operations.py --storage mmap trace example-storage.vault
```

The comparison verifies the same result and ordered trace with both file reads and read-only memory mapping. Each sample reads its inputs from storage, while the comparison controls for nondeterministic return values such as the internal timing of operation 80. The mapped path shares a mapping of the scalar pool within an execution and enables SQLite mapping for the connection. Both storage paths keep the same logical scalar, node, and operation addresses.

Each profile shows elapsed time, time inside measured primitives, and the remaining interpretation and trace overhead. It also shows logical read counts, dependent reads with a preceding address in the trace, and the number of serialized payload bytes delivered to the runtime. Per-operation timings include work performed by nested operations. The SQLite mapping setting is a size limit. It does not measure resident memory or physical device traffic.

## Continuous growth

The [training command](../Training.md) runs concurrent teaching sessions, tests their proposals against staged revisions, and records a separate critic's feedback. It retains rejections and commits successful revisions with their identity measurements and original conversations. Use the optional encounter batch to direct it toward previously unresolved turns.
