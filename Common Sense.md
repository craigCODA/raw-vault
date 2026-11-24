# Common sense

`commonsense.vault` defines 11 rules for everyday expectations. A rule takes observations, checks its premises, and can apply exact arithmetic before it adds an expectation. Observed exceptions and explicit contradictions can block a rule. Subsequent rules can use the expectations, and the result records each step and what it used.

## Run the example

Run from the vault folder.

```powershell
py -3 -B scripts/commonsense.py example-commonsense.vault
```

The example sets `c` to a slippery path. Rain makes the exposed path likely to get wet, and that wet surface supports the next expectation. The bridge's cover blocks the wetting rule for the bridge. The result shows both the chain to `c` and the observation that blocked the other application.

Facts are lists with a predicate followed by concrete arguments. For example, `["covered", "[[bridge]]"]` supplies an exception, and `["logic:not", ["wet", "[[path]]"]]` explicitly contradicts a wetting expectation. The result marks `c` as observed, expected, unknown, contradicted, or conflicting.

## Address common-sense links

The common-sense catalog has 14 relations and 1,879 matches to literal wording. With 2,298,793 keys, there are **73,982,289,595,886** possible one-link combinations using common-sense relations and **174,386,825,476,017** across all 33 relation types. Stored wording provides a cue. Applying a rule to a situation establishes the expectation for that situation.

```powershell
py -3 -B scripts/reasoning_links.py chain --family commonsense --links 3 --limit 2
py -3 -B scripts/reasoning_links.py chain --family all --links 2 --limit 2
```
