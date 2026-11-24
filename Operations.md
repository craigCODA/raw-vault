# Compiled Vault operations

`operations.vault` indexes the 80 numbered operations. The four declaration files in `operations/` hold the settings, 185 vocabulary anchors, and 126 executable cases. Compilation stores the operations, dependencies, and reverse vocabulary links in `raw/language.sqlite`. It runs each case before committing the installation.

## Run a data file

From the vault folder, run the example that resumes with Craig's saved name. Its event passes through intent, recall, and reply operations. The result includes the instruction addresses and the state changes that reach `c`.

```powershell
py -3 -B scripts/vault_operations.py run example-execution.vault
```

A file names its operation by number or name and supplies a structured `input` object. The executable-links operation accepts a saved `cursor`, a state, an event, and the instructions at its addresses. An instruction can check a condition, run another operation, write results back to state, and choose the next address. A reference like `{"$": "state.answer"}` supplies a value from that state.

## Inspect the compilation

```powershell
py -3 -B scripts/vault_operations.py list
py -3 -B scripts/vault_operations.py for-key '[[state]]'
py -3 -B scripts/vault_operations.py check
py -3 -B scripts/vault_operations.py check --operation 76
```

The language and conversation operations use the declared patterns and supplied context. World and logic operations work on explicit facts, equations, and finite domains. State and storage operations use serialized saves, declared events, and a byte-layout simulation. The measurement experiment reports elapsed time, executed operations, logical data bytes, and Python allocation peaks.

## Address combinations

Each hop has a statement, one of the 33 reasoning relations, a compiled operation, and a next statement. With `s` hops and `w` vocabulary keys per statement, the address count is `2,298,793^(w × (s + 1)) × (33 × 80)^s`. That gives 13,950,946,038,081,360 possible addresses for one hop with one key per side. The address command retrieves a position directly. The runtime executes an addressed rule when its inputs and conditions are supplied.

```powershell
py -3 -B scripts/vault_operations.py address --links 2 --width 1 --limit 2
py -3 -B scripts/vault_operations.py run example-address.vault
```

An addressed data file supplies the returned `address` and one input object per hop. Each object can pass `left` and `right` keys to the chosen operation, check its relation and roles, and use `results` from earlier hops. The example takes the stored `2` and `3` keys at the chosen address and computes `5`.

Refresh the compilation with `py -3 -B scripts/vault_operations.py build` after changing a declaration.
