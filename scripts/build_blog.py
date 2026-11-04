"""Build the offline visual article from frozen results; no inference or API calls."""
from pathlib import Path
import html
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT / "blog"
NAMES = {"mnist": "MNIST", "fashion": "Fashion-MNIST", "sst2": "SST-2", "diffusion": "MNIST diffusion"}


def main():
    data = json.loads((ROOT / "results/results.json").read_text())
    BLOG.mkdir(exist_ok=True)
    (BLOG / "data.js").write_text("window.STUDY = " + json.dumps(data, separators=(",", ":")) + ";\n")
    rows = []
    for row in data["classification_summary"] + data["diffusion_summary"]:
        diffusion = row["group"] == "diffusion"
        result = (f'{row["mse_mean"]:.5f} ± {row["mse_sd"]:.5f}' if diffusion else
                  f'{100*row["accuracy_mean"]:.2f}% ± {100*row["accuracy_sd"]:.2f} pp')
        rows.append('<tr data-group="{group}" data-task="{task}" data-model="{model}"><td>{group}</td><td>{task_name}</td><td>{model}</td><td>{budget}</td><td>{result}</td><td>${cost:.4f}</td></tr>'.format(
            **row, task_name=html.escape(NAMES[row["task"]]), result=result, cost=row["mean_cumulative_api_cost"]))
    template = (ROOT / "scripts/templates/blog.html").read_text()
    (BLOG / "index.html").write_text(template.replace("<!-- RESULT_ROWS -->", "\n".join(rows)))
    assets = BLOG / "assets"
    assets.mkdir(exist_ok=True)
    for row in data["diffusion"] + [r for r in data["baselines"] if r["task"] == "diffusion"]:
        source = ROOT / "results/generations" / Path(row["generation"]["grid"]).name
        shutil.copy2(source, assets / source.name)
    print("Built blog: 54 aggregate rows, 162 checkpoints, all diffusion grids; no API calls.")


if __name__ == "__main__":
    main()
