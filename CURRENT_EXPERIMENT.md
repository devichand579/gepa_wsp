# Completed expanded experiment

All **45 runs** and final held-out evaluations are complete. The study covers three seeds, three LLMs, four target architectures, 162 nested budget checkpoints and 12 Adam baselines. Whole-project conservative API accounting is **$45.8210 / $50**, across 1,759 requests, with no pending reservations.

Authoritative outputs:

- [Final blog](BLOG_EXPANDED.md)
- [All results](expanded_results/results.json) and [aggregate CSV](expanded_results/summary.csv)
- [Completion marker](runtime/EXPANDED_RESULTS_READY.json)
- [Selection freeze](runtime/EXPANSION_SELECTIONS_FROZEN.json)
- [Selected-source inspection receipt](runtime/SELECTED_SOURCE_INSPECTION_COMPLETE.json)
- [Provenance/program audit](expanded_results/audit.json)

The final tests opened after every selection was frozen and all115 selected sources had recorded Codex source inspections. Reflection and generated code received permitted training arrays, current weights and forward diagnostics; no final-test arrays were used for fitting, reflection or model selection. Statistics and closed-form fitting were permitted; parameter-gradient/Newton updates were excluded from approved selected sources.

The DeepSeek budget study uses0/5/10/30/60 mutations. Fresh matched compact-prompt model comparisons use0/5/10; diffusion uses0/5/10/20. Seeds are0,1,2. See [EXPANSION_PLAN.md](EXPANSION_PLAN.md) and [RICH_PROTOCOL.md](RICH_PROTOCOL.md) for exact protocols, model IDs, budget accounting and pretest amendments.

The eight method-violating original trajectories and Nemotron seed0's first replacement are preserved and excluded. The authoritative replacement mapping is [runtime/EXPERIMENT_REPAIRS.json](runtime/EXPERIMENT_REPAIRS.json). All historical spending remains included. The original primary-only monitor/finalizer and strict-edit runs are superseded; expanded_results is authoritative.

The two-GPU queues finished all scheduled jobs. The review-gate attention records are preserved under runtime/resolved/. No additional API experiments are required. Remaining API budget was not spent merely to exhaust the cap.

The final blog's presentation edits and extra log-scale plot read frozen results only. Original generated text and the original linear diffusion figure remain preserved; [presentation notes](runtime/BLOG_PRESENTATION_NOTES.json) describe this separation. No checkpoints, selections, evaluation metrics or frozen experimental source hashes were changed after final-test inference.
