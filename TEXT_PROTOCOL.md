# Third architecture: strict weight edits on SST-2

This predeclared follow-up uses an 83,266-parameter transformer, random initialization,
no pretrained tokenizer, embedding, target weights, or language-model target.
Two bidirectional pre-LayerNorm blocks, width32, two heads, FFN64, ReLU,
vocabulary2000, maximum64 tokens, learned positions and masked mean pooling.
Final LayerNorm and a binary linear head complete the fixed forward pass.

Seeds0,1,2; each has 4000 balanced feedback texts and1000 disjoint selection
texts from SST-2 training. Official labeled validation (872 examples) is the final
held-out split; official unlabeled test is unused. Vocabulary is fit only from
feedback text frequency. The tokenizer is a fixed lowercase regex. Normalized
raw-text cross-split duplicates are removed; encoded-input collisions with
feedback/final inputs are excluded from selection before any model evaluation.
The final inputs are used only for a hash audit at preparation and for inference
after all three model selections are frozen. Labels never enter writer context.

One-shot sees16 balanced examples. Reflection sees at most12 rendered feedback
texts per128-example paired batch. The entire vocabulary/ID mapping is provided,
without label-conditional vocabulary counts. Additional examples are part of the
GEPA condition; this is not an equal-information comparison.

Same strict JSON interpreter, model, 8192-token thinking budget, 65536-token output
cap, temperature0.6, Pareto selection, fixed joint/layer schedule and40 mutation
API calls per seed as the image suite. A joint transformer proposal updates all
six modules; local proposals cycle embed,pos,block0,block1,norm,head. No fitting,
network execution, gradients, example matrix algebra or optimizer runs occur
inside the generated edits. Partial operation rejection is reported without
repair. Invalid submissions count. Adam uses the same model/feedback pool and
initial weights, batch128, learning rates1e-4/1e-3/1e-2, at most50 epochs each,
patience10, and the same selection score. Count all tuning.

The text backend runs in separate processes and a separate artifact root
`text_runs/`, with its own4000-index provenance audit. The frozen image sources
remain unchanged. Record text-specific source hashes alongside the shared freeze.
No claim that attention learned sentiment follows merely from classification
accuracy: report which parameter groups changed and how sparse the resulting
network is. A lexicon encoded in embeddings with zeroed attention remains a
transformer architecture but is a weaker mechanism claim.

This phase is within the original650-call/$20 whole-project plan. All image,
pilot and text spending remains in the same concurrent ledger. Final reporting
must retain malformed generations, random fallbacks and negative outcomes.
