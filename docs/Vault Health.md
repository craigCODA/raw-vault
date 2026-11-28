# Vault health

Run the full health check from the repository with Python 3.12.

```powershell
py -3.12 -B scripts/vault_health.py
```

The command reports PASS or FAIL, suite coverage, and resource deltas. It returns exit status 1 on failure. The reference run took about two minutes, mostly in SQLite rebuilds. Add `--json` for structured results, including each affected field and its expected and observed values.

The reference capture in `vault-health-baseline.vault` comes from the hydrated Raw Vault snapshot at `535c5708d99e338137b77537280a7060ad55433a`. It contains the vault identity and facts, all four example outputs, the existing CLI commands, the full results of all compiled cases, and the original passing unittest IDs. Build and reindex work against one temporary SQLite backup. The temporary directory also holds exported notes and saved execution results.

Stable results compare field values, JSON types, and array order. Exported files compare as text with normalized line endings. New unittest IDs can extend the suite while the original IDs still have to pass.

Elapsed time uses a monotonic clock. Python allocation peaks come from `tracemalloc` and cover Python allocations. Each probe starts in a fresh process with a fixed Python hash seed. The measurement starts just before the command or suite runs. The upper bound is the greater of three times the reference value or the reference value plus 500 ms and 2 MiB, respectively. Probes that load the operations runtime use a 1.25 s time floor to allow for interpreter startup under allocation tracing. The measurement operation inside the compiled suite uses floors of 50 ms and 128 KiB for its own observations. A different Python version, interpreter, SQLite version, operating system, or architecture makes the measurements informational. Changes to suite membership make that suite's aggregate measurements informational.

Use `--root` to exercise a specific hydrated checkout and `--baseline` to select the reference capture. To capture an intentional new reference, pass `--record --baseline` with a new file path.
