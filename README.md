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

## Run — a worked example

You give the tool the **wording of each question** and a table of **responses**;
it returns a ranked list of people whose answers form an unusually different
overall pattern. The method embeds the prompts first to build a semantic space,
then projects the responses into it — so we set up the files in that order.

**1. `prompts.csv`** — the wording of each item. This is what builds the space:

```csv
item,prompt
sad,How often do you feel sad?
sleep,How often do you have trouble sleeping?
worry,How often do you feel worried?
```

**2. `responses.csv`** — one row per person: an id, optional `age`/`sex`, then one
column per item. Items from all your questionnaires go side by side, and the
column names match the `item` keys above:

```csv
id,age,sex,sad,sleep,worry
P001,42,F,1,3,2
P002,67,M,5,1,4
```

**3. Run it** (needs a local bge-m3 model — see above):

```bash
survey-semantics analyze-file responses.csv \
  --prompt-file prompts.csv \
  --embedding sentence-transformers --model models/bge-m3 \
  --outdir outputs/run
```

**4. Read the result.** Everything lands in `outputs/run/`; the ranking is
`*_scores.csv` — the larger a person's `Mahalanobis_Dist`, the more unusual they are.

### Optional refinements

Add a flag + file to refine the run; omit them and behavior is unchanged.

- **`--scale-file scales.csv`** — per-item valid range, missing-value codes, reverse, and (optional) ceiling. Declares the analyzed items and cleans each item's own missing codes:
  ```csv
  item,min,max,sentinels,reverse,ceiling
  sad,1,5,7;8;9,false,true
  ```
- **`--weights-file weights.csv`** — one survey weight per row (same order as `responses.csv`) → weighted residualization + Mahalanobis:
  ```csv
  weight
  1842.6
  ```
- **`--pan-mild`** — also flag below-ceiling outliers. **`--d-selection`** — choose the dimension rule (`variance` default, `eigengap`, `parallel`, `stability`, `max`).

Full file formats and options: [docs/pipeline_overview.md](docs/pipeline_overview.md).

### Variants

- **Questionnaires in separate files** (not one merged table): `run-study --data-dir <dir> --prompt-dir <dir>` merges them for you.
- **Longitudinal / multiple waves** with differing item sets (e.g. NHIS 2021 vs 2024): restrict to common items so the waves share one space, run each, and compare — see the [NHIS example](examples/nhis/README.md).

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
