# Reasoning links

`[[slow]] {rhetoric:contrast} [[steady]]` links the words on either side of the cue in the stored phrase `SLOW BUT STEADY`. Rhetoric, logic, and argument account for 28,592 literal cue matches with their original wording. A cue marks a possible reading of the fragment.

`reasoning.vault` defines 19 ordered relations, including reason and concession, conjunction and implication, and support and rebuttal. It also contains truth expressions and rules for deriving conclusions from declared premises.

The [common-sense rules](<Common Sense.md>) add 14 more relations. Use `--family all` to address chains across all 33 types. The default chain catalog uses the 19 rhetoric, logic, and argument types.

## Follow a link or run a proof

Run these from the vault folder.

```powershell
py -3 -B scripts/reasoning_links.py link rhetoric:contrast '[[slow]]' --right '[[steady]]'
py -3 -B scripts/reasoning_links.py examples argument:supports --limit 5
py -3 -B scripts/reasoning_links.py truth logic:implies 1 0
py -3 -B scripts/reasoning_links.py prove example-argument.vault
```

The proof file uses three declared premises and two inference steps to reach `c`. Each step refers to earlier terms by zero-based position. The supplied premises are the conditions for the proof.

## Address a chain

Each position selects its statement keys and its ordered relations. Repeated keys are allowed. Set `--links` to the number of relations and `--width` to the number of vocabulary keys in each statement.

```powershell
py -3 -B scripts/reasoning_links.py chain --links 3 --width 1 --offset 123456789 --limit 2
```

With all 19 relation types, there are **100,404,535,880,131** one-link combinations with one key on each side and **191,540,345,680,785,753,893,210,966,059** three-link chains with one key per statement. A chain with `s` links and `w` keys per statement has `19^s × 2,298,793^(w × (s + 1))` possible addresses. The script decodes the requested positions directly.

Run `py -3 -B scripts/reasoning_links.py build` after editing the cue definitions to refresh their stored links.
