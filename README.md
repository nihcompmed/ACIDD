# Survey Semantics

Reusable implementation of **Periwal, *"Auditable cross-instrument detection of
unusual multivariate response configurations using a semantically aligned
covariance subspace."*** It embeds questionnaire **item wording** into a shared
semantic space, projects responses into it, and flags respondents whose
cross-instrument response pattern is multivariate-unusual — auditably, by tracing
each flag back to the original items.

> Validated in the paper on the **HRS** and **Xinxiang** cohorts. This package
> generalizes the method to any subjects×items survey; the worked example applies
> it to **NHIS 2021/2024** and adds optional survey weights.

**Full docs:** [docs/pipeline_overview.md](docs/pipeline_overview.md) — how it
works, file formats, every option. · [examples/nhis](examples/nhis/README.md) —
end-to-end worked example.

## Install

```bash
conda env create -f environment-bge-m3.yml
conda activate survey-semantics-bge-m3
pip install -e . --no-deps     # --no-deps keeps the env's modern stack
python -m pytest -q            # verify
```

## Embedding model (required)

The only backend is a local `sentence-transformers` model (e.g. bge-m3) — **no
fallback**, a missing model raises. It is offline-only, so fetch it once:

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='BAAI/bge-m3', local_dir='models/bge-m3', \
ignore_patterns=['onnx/**','openvino/**','*.onnx'])"
```

## Run

Single table:

```bash
survey-semantics analyze-file data.csv \
  --prompt-file prompts.csv \
  --embedding sentence-transformers --model models/bge-m3 \
  --d-selection variance --max-components 0 \
  --outdir outputs/run
```

Multi-instrument study (one data + one prompt file per instrument, in two dirs):

```bash
survey-semantics run-study \
  --embedding-model models/bge-m3 \
  --data-dir study/data --prompt-dir study/prompts \
  --d-selection variance --max-components 0 \
  --outdir outputs/study
```

## Inputs

| Flag | What it does |
|---|---|
| `--prompt-file` / `--prompt-dir` | Item wording (`item,prompt`) — builds the semantic space. |
| `--model` | Local bge-m3 path (required). |
| `--scale-file` *(opt)* | Per-item `min,max,sentinels,reverse` (+ optional `ceiling`): declares the analyzed items, per-item missing codes, ranges, reverse coding. |
| `--weights-file` *(opt)* | One survey weight per row → weighted residualization + Mahalanobis. |
| `--pan-mild` *(opt)* | Flag below-ceiling outliers (adds `At_Ceiling`, `Is_Pan_Mild_Emp<pct>`). |
| `--d-selection` | `variance` (default), `eigengap`, `parallel`, `stability`, `max`. |

Optional inputs are backward compatible — omit them and behavior is unchanged.
File formats and the full option list are in
[docs/pipeline_overview.md](docs/pipeline_overview.md).

## Outputs (`--outdir`)

- `*_scores.csv` — per-subject Mahalanobis distance, semantic PCs, outlier flags (the ranking).
- `*_prompt_loadings.csv` — how items load on each semantic dimension.
- `*_drivers.csv` — which items/dimensions drive each top outlier.
- `*_stability.csv`, `*_dimension_selection.csv` — how `D` was chosen and how stable it is.
- `case_studies/`, `summary.csv` — anonymized per-outlier reports and a run summary.

Add `--skip-umap` to skip UMAP files. Render plots with
`survey-semantics-plot outputs/run`.

## Layout

```text
src/survey_semantics/   package (io, prompts, scales, embedding, pipeline, combined, cli, plotting)
tests/                  test suite
examples/nhis/          NHIS worked example (converter + run script + README)
docs/                   pipeline_overview.md, env_setup.md
```
