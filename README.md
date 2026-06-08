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
form an unusually different overall pattern.

**One questionnaire.** Point it at the responses CSV and the question-wording CSV:

```bash
survey-semantics analyze-file data.csv \
  --prompt-file prompts.csv \
  --embedding sentence-transformers --model models/bge-m3 \
  --outdir outputs/run
```

**Several questionnaires together.** Put the response files in one folder and
their wording files in another; the tool merges them into one shared space:

```bash
survey-semantics run-study \
  --embedding-model models/bge-m3 \
  --data-dir study/data --prompt-dir study/prompts \
  --outdir outputs/study
```

The results go to `--outdir`. The ranking is in `*_scores.csv` — the larger the
`Mahalanobis_Dist`, the more unusual the respondent. Add tuning flags (`--scale-file`,
`--weights-file`, `--pan-mild`, `--d-selection`) from the tables below as needed.

## Input files

A few example rows of each. The item names in `data.csv` (`sad`, `sleep`,
`worry`) match the `item` keys in the other files.

`data.csv` — one row per person: an id, optional `age`/`sex`, then one column per item:

```csv
id,age,sex,sad,sleep,worry
P001,42,F,1,3,2
P002,67,M,5,1,4
```

`prompts.csv` — the wording of each item (this is what builds the semantic space):

```csv
item,prompt
sad,How often do you feel sad?
sleep,How often do you have trouble sleeping?
worry,How often do you feel worried?
```

`scales.csv` *(optional)* — valid range, missing-value codes, reverse, ceiling:

```csv
item,min,max,sentinels,reverse,ceiling
sad,1,5,7;8;9,false,true
sleep,1,5,7;8;9,false,true
worry,1,5,7;8;9,false,true
```

`weights.csv` *(optional)* — one survey weight per row, same order as `data.csv`:

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
