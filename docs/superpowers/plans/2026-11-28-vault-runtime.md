# Raw Vault Runtime Implementation Plan

> Use the Superpowers test-driven development and verification workflows for each phase.

**Goal** Build and verify the seven phases of the executable vault described in the project discussion.

**Starting point** The public Raw Vault snapshot at `535c570`. Use its hydrated payloads and compiled cases as the behavioral baseline.

Use Python 3.12 and its standard library. Keep existing CLI, database, and scalar address formats readable. Put generated runtime activity in a separate writable directory. Work against local candidate versions. Keep model-generated changes declarative and evaluate them against the actual vault before adoption.

## Implementation tasks

- [x] Phase 0. Add `scripts/vault_health.py`, regression fixtures, and a shipped baseline. Cover each existing CLI and example. Compare stable outputs and identity exactly, with explicit performance budgets. Verify rebuild commands on scratch data.
- [x] Phases 1 and 2. Move the 80 operation bodies into addressable graph declarations. Expose `Machine(root).execute(module, operator, arguments, parameters, context)` and `Machine.has(module, operator)`. Keep built-in I/O and arithmetic as host primitives. Define a graph execution path for resolving operations, binding arguments, executing bodies, and completing calls.
- [x] Phase 1. Add `vault trace` and `vault replay` commands to the existing operations CLI. Store events and links to operations, state paths, raw values, and observed effects. Replay the same input under the same executable revision, including recorded timing observations. Compare ordered serialized outputs and execution evidence exactly.
- [x] Phases 3 and 6. Add a durable training queue and candidate revisions. Use separate model sessions for teaching and criticism. Capture proposal provenance, run both original and new cases, check identity, and publish only passing revisions. Keep rejection evidence. Run the integrated pipeline with concurrent participants.
- [x] Phase 4. Store explicit session context and activation rules. Run an input event through the activation graph and preserve the referenced facts, exact reads, state changes, and active set size.
- [x] Phase 5. Execute the same cases using a memory-mapped storage path. Keep addresses stable, observe bytes and read counts, and report the measured difference between interpretation and primitive execution.
- [x] Run end-to-end verification and review of the implementation.
- [x] Launch a 48-hour experiment and verify the first live round and offline checkpoint.
- [ ] Collect the final offline holdout report after the 48-hour run.

## Review focus

- Historic SQLite payloads start with the original catalog schema. The new runtime must resolve that catalog through bundled declaration bodies.
- Elapsed times and allocation peaks vary. Capture the observations and compare performance against explicit, reproducible tolerances.
- Query revisions carry their own program identity and ordered observations. A replay must detect a changed definition, input, output, or trace.
- A failed or stale training proposal must leave the published revision usable and preserve its evidence.
- A stored context may outlive an individual invocation. A new event must resolve against one coherent revision and the chosen session state.
- Normal and mapped reads must use exactly the same scalar addresses and bytes.
