# Publication checks

Checked on 4 November 2026. These checks use frozen artifacts and make no paid model calls.

- `python3 scripts/verify_snapshot.py`: passed all 1,695 exported historical file hashes, 162 checkpoint hashes, 54 recomputed aggregates, 115 approved selected-source reviews, frozen source/data-manifest hashes, reliability totals, accounting consistency, blog data equality and local article links.
- Six preserved harness checks passed: parameter-gradient screening, diffusion GEPA adapter, job-queue/prefix integrity, finalization gates, evaluation flow, and report generation.
- `scripts/check_blog_browser.py`: passed in Chromium at desktop 1440px and mobile 390px/360px widths, served locally and opened directly as a file. Exercised all task/budget controls, all nine model/seed diffusion-grid choices, model comparisons, table filters and empty-result state. No JavaScript errors, horizontal page overflow or remote resource loads.
- Exported files were scanned against the private study API key and credential/token patterns before commit; no credential content or credential files were found. Raw request/response/prompt payloads are omitted.
- Original experiment files remain unchanged in the local study directory. Publication work did not launch experiments, alter selection, or spend the remaining experiment budget.

The original full-archive completion audit is preserved at `runtime/COMPLETION_AUDIT.json`. Publication integrity checks are distinct from that earlier prepublication research audit.
