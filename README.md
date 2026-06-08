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

You give the tool two things: a table of **responses** (one row per person) and
the **wording of each question**. It returns a ranked list of people whose answers
form an unusually different overall pattern across all the questions.

**General case.** Put every item — from one questionnaire or several — as columns
in one responses table, alongside the wording of each item. The tool builds one
shared semantic space across all of them and ranks unusual respondents:

```bash
survey-semantics analyze-file responses.csv \
  --prompt-file prompts.csv \
  --embedding sentence-transformers --model models/bge-m3 \
  --outdir outputs/run
```

If your questionnaires are in separate files instead of one table, use
`run-study --data-dir <dir> --prompt-dir <dir>` to merge them automatically.

**Advanced — longitudinal / multiple waves.** When you compare waves whose item
sets differ (e.g. NHIS 2021 vs 2024), restrict to the items common to both so they
share the same semantic space, then run each wave and compare. See the
[NHIS example](examples/nhis/README.md).

The results go to `--outdir`. The ranking is in `*_scores.csv` — the larger the
`Mahalanobis_Dist`, the more unusual the respondent. Add tuning flags
(`--scale-file`, `--weights-file`, `--pan-mild`, `--d-selection`) as needed.

## Input files

A few example rows of each, in the order the method uses them: it **embeds the
prompts first** to build the semantic space, then **projects the responses** into
it. The `item` keys tie the files together (`sad`, `sleep`, `worry`).

**1. `prompts.csv`** — the wording of each item; this builds the semantic space:

```csv
item,prompt
sad,How often do you feel sad?
sleep,How often do you have trouble sleeping?
worry,How often do you feel worried?
```

**2. `responses.csv`** — one row per person: an id, optional `age`/`sex`, then one
column per item (items from all your questionnaires go side by side):

```csv
id,age,sex,sad,sleep,worry
P001,42,F,1,3,2
P002,67,M,5,1,4
```

**3. `scales.csv`** *(optional)* — valid range, missing-value codes, reverse, ceiling:

```csv
item,min,max,sentinels,reverse,ceiling
sad,1,5,7;8;9,false,true
sleep,1,5,7;8;9,false,true
worry,1,5,7;8;9,false,true
```

**4. `weights.csv`** *(optional)* — one survey weight per row, same order as `responses.csv`:

```csv
weight
1842.6
903.1
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
