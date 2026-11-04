# Pilot log and protocol changes

No final-test inference or feedback was used in pilot development.

- `pilot`: MiniMax provider-default reasoning, 32K output limit. Construction
  calls exhausted the limit on reasoning with no final code. Stopped after the
  first in-flight mutation; failures retained and billed in the ledger.
- `pilot_v2`: explicit medium reasoning. MNIST returned a design description
  without executable code; Fashion-MNIST exhausted its output limit. Stopped
  these runs after their current mutation.
- `pilot_v3`: append an explicit final-code submission contract. MNIST construction
  returned valid code. A mutation exposed a wrapper issue: native NumPy dispatch
  needed the import builtin for its own helpers. The namespace and AST constraints
  still prevent generated-code imports and private file access. After that wrapper
  fix, replaying the unchanged mutation exposed a genuine generated-code shape
  error. No model code was manually repaired. Fashion construction still truncated.
- `pilot_deepseek`: user-authorized alternate model, exact official endpoint
  `accounts/fireworks/models/deepseek-v4p1-flash`. Construction included a bounded
  padding loop rejected by an overly strict while-loop check; inspection also
  found a generated-code variable-shadowing bug. The blanket while ban was removed:
  bounded construction loops are allowed and CPU/memory limits handle runaway code.
  No DeepSeek result is pooled with MiniMax's main results.
- `pilot_v4`: same explicit output contract, low reasoning effort. Both tasks
  returned executable construction code under the 32K limit. This completes the
  minimum output-interface criterion for selecting the main reflection setting.
  Choose this configuration for valid completion, not for final-test accuracy.

The MNIST v3 pilot is allowed to finish five total proposals to exercise resume
and history handling. v4 has one mutation per task. These are pilot-only runs.

Before main: add deterministic double execution of parameter construction,
include the actual base-weight hash in checkpoint identities, save run-level
configuration and source hashes, and enforce a global service/budget stop.

Main reflection configuration: MiniMax M3.1 Flash, low reasoning, temperature
0.6, 32,768 total output-token limit including reasoning. Same construction
settings for the one-shot baseline and GEPA seed. Forty main mutation proposals
per task/seed; invalid proposals count. No free corrective model calls.

The pilot source was iterated while jobs were running; the main source is hashed
and frozen separately. Treat pilot outputs as interface/debug evidence only.
