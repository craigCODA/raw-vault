# Training the Raw Vault

The trainer pins the shipped Vault in a writable work directory. Teachers submit declarative lessons with examples. The builder groups equivalent changes, assigns new IDs, compiles a candidate, and checks its original and new cases. An independent critic can add counterexamples. Publication records the candidate fingerprint and teacher and critic conversations together. Rejected proposals retain their evidence in the queue.

## Run a lesson

Use Python 3.12 or newer. The default model provider uses the authenticated Codex CLI and starts a new session for each teacher and critic.

```powershell
py -3.12 scripts/vault_train.py --work .vault-data/training run --provider codex --teachers 2 --rounds 1 --brief "Teach a new exact arithmetic turn using a fictional rain gauge."
py -3.12 scripts/vault_train.py --work .vault-data/training status
py -3.12 scripts/vault_train.py --work .vault-data/training evidence 1
```

Use `--model` and `--critic-model` to select provider-supported model names and `--timeout` to bound each model call in seconds. The Codex provider uses the local sign-in and sends the brief, selected encounters, lesson declarations, and bounded training feedback. Pass public or synthetic examples when using a remote model.

Run the bundled example with fixture responses to check the full workflow locally. Two teachers propose the same gauge operation and add separate cases. The critic finds a negative input that breaks a third teacher's rule.

```powershell
py -3.12 scripts/vault_train.py --work .vault-data/training-example run --provider fixture --fixture example-training.vault --teachers 3 --rounds 1
```

## Schedule multiple rounds

```powershell
py -3.12 scripts/vault_train.py --work .vault-data/training run --duration-hours 48 --interval-seconds 1800 --rounds 96 --teachers 2 --encounters encounters.json --jsonl
```

The command stops starting rounds when the elapsed time or round cap is reached. A round that already started can finish. The interval begins after the previous round finishes. Duration-based runs stream JSON Lines progress with the durable round number, proposal IDs, outcomes, encounter counts, and current fingerprint. Without a duration or an explicit cap, the command starts one round.

Rerun with the same work directory after an interruption. The trainer resumes an unfinished round using its original inputs, skips teachers whose submissions are already stored, and tests its remaining proposals. An invocation's time budget and round cap start when the command starts.

Each accepted revision keeps its own SQLite database. The scalar pool and fixed assets are shared between revisions when hard links are available. Readers can keep using older published snapshots while the current fingerprint changes. A numeric mixed-radix link position uses the vocabulary and operation counts of its revision.

## Select encounters and inspect evidence

An encounter file accepts up to 24 selected items. A message can supply an expected reply or a short explanation. Stored trace references use the trace store path relative to the encounter file and the ID returned by the Vault. The trainer also accepts `vault-trace/1` and `vault-trace/2` exports. It extracts the message only from failed or unresolved turns and reuses recent counterexamples saved from earlier model rounds.

When you supply expected replies, the trainer runs those turns before and after training and reports how many matched exactly. Lessons can improve other behavior while some encounters remain open. Once a published revision preserves or solves an encounter, future candidates must preserve its expected reply.

```json
[
  {
    "text": "blue-lid rain gauge 3",
    "expected_reply": "3/2",
    "reason": "The fictional gauge divides its reading by two."
  }
]
```

`status` gives the active fingerprint, snapshot paths, and counts. `evidence ID` gives the proposal, original conversation, critic review, and result. Manual submissions use `submit proposal.json conversation.json` and `build`. The example fixture shows the proposal format. A manual conversation JSON object includes `role`, `provider`, `session`, `model`, `prompt`, and `response`, each as a string. Set `role` to `teacher`.

An alternate provider can run a trusted executable with `--provider command --command-argv` and a JSON array of the command and arguments. The executable reads one JSON request with `role`, `index`, `model`, and `prompt` from standard input. It emits one JSON response with `session`, optional `model`, and a `submission` object.
