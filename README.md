# Survey Semantics

This repository is the reusable implementation of the method in **Periwal,
*"Auditable cross-instrument detection of unusual multivariate response
configurations using a semantically aligned covariance subspace."*** The method
embeds questionnaire item *wording* into a shared semantic space, builds a
low-dimensional covariance subspace from the item prompts alone (no respondent
data), projects responses into it, and flags respondents whose **cross-instrument**
response configuration is multivariate-unusual — in an **auditable** way, by
backtracking every flag to original item loadings.

> **Provenance & scope.** The published method was developed and validated on two
> cohorts — the **HRS** (older adults) and **Xinxiang** (younger adults) samples.
> This package generalizes that method to any subjects×items survey. The included
> worked example applies it to **NHIS 2021/2024**, which *extends* the paper by
> adding **survey weights** (weighted residualization + weighted Mahalanobis) for
> a nationally representative design — a feature beyond the original two-cohort
> study. The paper's **"pan-mild" / ceiling-exclusion** audit (isolating
> below-threshold cases — outliers whose item profile is nowhere at the Likert
> ceiling) is available as an **opt-in flag** (`--pan-mild`), off by default.

The package keeps the method's main methodological spine:

- embed survey item wording,
- center and reduce item embeddings with PCA,
- normalize respondent item responses (optionally via a per-item **scale file**: declared ranges, per-item missing-value codes, reverse coding),
- project responses into the semantic item manifold,
- optionally residualize covariates such as age or sex (optionally **survey-weighted** via a weights file),
- score multivariate deviation with Ledoit-Wolf Mahalanobis distance (weighted when weights are supplied),
- flag outliers at an empirical percentile, and optionally **pan-mild** outliers (`--pan-mild`: those nowhere at the Likert ceiling),
- export ranked semantic outliers, item/component loadings, stability diagnostics, and case drivers.

New reusable code lives in `src/survey_semantics`.

## Documentation & worked example

- **[docs/pipeline_overview.md](docs/pipeline_overview.md)** — the general pipeline end to end: the 13 stages and *why*, what each module does, every input file format, and how to run it. **Start here.**
- **[examples/nhis/README.md](examples/nhis/README.md)** — a complete worked example: run the **NHIS 2021 & 2024** cohorts (prompts + scales + survey weights + shared-basis cross-year comparison). Includes a runnable [`run_nhis.sh`](examples/nhis/run_nhis.sh).

## Setup with conda

The repo ships two dependency tracks:

- **Modern (recommended)** — `environment-bge-m3.yml`: Python 3.11 + modern sci stack + CPU torch + sentence-transformers. Runs **both** the real `bge-m3` embeddings and the offline `tfidf` smoke test.
- **Legacy** — `requirements.txt` / the `pyproject.toml` pins: Python 3.7, old NumPy/pandas, **tfidf only** (cannot run bge-m3).

For a fresh clone, use the modern track:

```bash
# 1. Create the env from the shipped spec (named env: survey-semantics-bge-m3)
conda env create -f environment-bge-m3.yml
conda activate survey-semantics-bge-m3

# 2. Install this package WITHOUT deps, to keep the modern stack from the yml
pip install -e . --no-deps

# 3. Verify (no model download needed)
python -m pytest -q          # expect: 21 passed
```

> **Why `--no-deps`?** `pyproject.toml` pins legacy Python-3.7-era versions
> (`numpy<1.22`, `pandas<1.4`, …). A plain `pip install -e .` would try to
> *downgrade* the modern env and break it. `--no-deps` installs only the
> `survey-semantics` package and its console scripts, leaving the yml stack intact.

Offline smoke test (lexical tfidf, no model required):

```bash
survey-semantics analyze-file <data.csv> --prompt-file <prompts.csv> \
  --embedding tfidf --skip-umap --outdir outputs/test
```

### Running real `bge-m3` embeddings

The package is **offline-only by design** (it blocks network egress during
analysis), so the model must be downloaded once, ahead of time:

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='BAAI/bge-m3', local_dir='models/bge-m3', \
ignore_patterns=['onnx/**','openvino/**','*.onnx'])"
```

Then pass it by path: `--embedding sentence-transformers --model models/bge-m3`.

For the project-disk **prefix** env (everything under one data disk, plus the
libstdc++ activate-hook fix), see [docs/env_setup.md](docs/env_setup.md).

### Legacy / pip-only

```bash
python -m pip install -r requirements.txt   # Python 3.7, tfidf only
pip install -e . --no-deps
```

## Quick Start

Run the notebook-equivalent common-space workflow:

```bash
survey-semantics run-study \
  --embedding-model BAAI/bge-m3 \
  --prompt-dir /path/to/study/prompts \
  --data-dir /path/to/study/data \
  --d-selection variance \
  --outdir outputs/my-study \
  --max-components 0
```

The four core inputs are:

- `--embedding-model`: use `tfidf` for the local lexical fallback, or a locally cached/local-path sentence-transformers model such as `BAAI/bge-m3`.
- `--prompt-dir` or `--prompt-file`: a directory of per-instrument prompt files, or one combined prompt registry.
- `--data-dir`: directory containing the survey data tables.
- `--d-selection`: semantic PC count rule used for scoring; choose `variance`, `eigengap`, `parallel`, `stability`, or `max`/`all`.

Two optional inputs make the analysis fully spec-driven (both backward compatible — omit them and behavior is unchanged):

- `--scale-file` / `--scale-dir`: a per-item scale spec — `item,min,max,sentinels,reverse`. When supplied, the **declared items define the analyzed set** (inference is skipped), each item is cleaned with **its own** missing-value codes, normalized to its declared range, and reverse-scored if flagged. See [examples/nhis](examples/nhis/README.md) for why this matters (NHIS `7/8/9` refusal codes).
- `--weights-file`: one survey weight per subject, **row-aligned** to the response file. Enables weighted (WLS) covariate residualization and a weighted Mahalanobis covariance. Length must equal the response-row count (hard error otherwise).
- `--pan-mild`: add the manuscript's ceiling audit. Adds an `At_Ceiling` column and, per empirical percentile, an `Is_Pan_Mild_Emp<pct>` column = outlier **and** nowhere at the Likert ceiling. Which items count toward the ceiling is controlled by an optional **`ceiling` column in the scale file** (explicit allowlist — e.g. exclude a "time since last doctor visit" item where maxing out isn't a symptom); absent that, it falls back to all polytomous items (tune the level cutoff with `--ceiling-min-levels`, default 3). Set the percentiles with `--empirical-percentiles` (default `95 99`).

The preferred study layout is one data file and one prompt file per instrument:

```text
study/
  data/
    instrument_a.csv
    instrument_b.tsv
    instrument_c.csv
  prompts/
    instrument_a_prompts.csv
    instrument_b_prompts.csv
    instrument_c_prompts.csv
```

The data filename stem is the instrument name. For example, column `q1` in `instrument_a.csv` becomes feature `instrument_a__q1` in combined mode. Prompt files in `--prompt-dir` use the same convention: bare `item` keys inside `instrument_a_prompts.csv` are automatically namespaced as `instrument_a__item`, so repeated item names across instruments do not collide.

The command writes CSV outputs, anonymized case-study `.txt` files, and PDF plots in one pass.

Prompt files can be CSV/TSV files with `prompt` plus either `item`, `feature`, or `table,item` columns:

```csv
item,prompt
item_1,Prompt text for item 1
item_2,Prompt text for item 2
```

They can also be copied from notebook code as a Python literal dictionary:

```python
{
  "instrument_a__item_1": "Prompt text for item 1",
  "item_2": "Prompt text for item 2"
}
```

For a single combined `--prompt-file`, use fully qualified `table,item` or `table__item` keys when possible:

```csv
table,item,prompt
instrument_a,item_1,Prompt text for item 1
```

If both `--prompt-dir` and `--prompt-file` are provided, the directory is loaded first and the single prompt file is applied second as an override layer.

Combined mode embeds prompts from every viable questionnaire in one common semantic space, fits one PCA basis on that common prompt space, aggregates each subject's available responses across all questionnaires, and scores transdiagnostic outliers from the combined response vector.

The lower-level commands remain available when useful:

- `survey-semantics analyze-file path/to/table.tsv --outdir outputs/table --prompt-file prompts.txt`
- `survey-semantics analyze-package /path/to/data --outdir outputs/by-table --prompt-dir prompts/`
- `survey-semantics analyze-package-combined /path/to/data --outdir outputs/combined --prompt-dir prompts/`

Reverse-polarity handling has two layers. First, the combined builder checks each instrument for strong within-instrument anticorrelation. When two item responses are strongly anticorrelated for enough of the same subjects, the tool treats that as an opposite-polarity warning, explicitly negates one polarity group, and writes both `combined_auto_reverse_warnings.txt` and `combined_auto_reverse_warnings.csv`. Because anticorrelation identifies opposite polarity but not absolute clinical direction, this is an auditable heuristic: manual config wins when present; otherwise the smaller polarity group is negated, with one-vs-one ties keeping the earlier item as the anchor.

Use these options to tune or disable that pass:

```bash
--auto-reverse-corr-threshold 0.70
--auto-reverse-min-pairwise-subjects 10
--auto-reverse-min-pairwise-fraction 0.50
--no-auto-reverse
```

Second, provide a manual CSV for known reverse-scored items:

```csv
table,item,reverse
srs02,parentreport_3,true
```

or use the fully qualified feature name:

```csv
feature,reverse
srs02__parentreport_3,true
```

Then run:

```bash
survey-semantics analyze-package-combined /path/to/data \
  --outdir outputs/combined \
  --embedding sentence-transformers \
  --model BAAI/bge-m3 \
  --prompt-dir prompts/ \
  --reverse-config reverse_items.csv
```

The combined output writes `combined_prompt_inventory.csv`, which is the file to review/edit when deciding which items require manual reversal. It includes `reverse_scored` and `reverse_source`, so automatically inferred items are distinguishable from manual items.

Create or regenerate PDF plots from an output folder:

```bash
survey-semantics-plot outputs/my-study --outdir outputs/my-study/plots
```

The sentence-transformers backend is local-only by design. The code sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, disables Hugging Face telemetry, installs a process-wide outbound socket blocker during analysis, and loads models with `local_files_only=True`. If the model is not already local, the run fails instead of downloading or calling Hugging Face. You can pass `--model /absolute/path/to/local/model` to avoid relying on a cache location. For institutional compliance, run this in an OS/container environment with network egress disabled too; the package guard prevents accidental Python-level calls, while the OS policy is the final perimeter.

The `auto` embedding mode tries local sentence-transformers first and falls back to TF-IDF when the model is unavailable.
Every output filename includes the actual embedding backend/model slug, for example `rbs_r02__tfidf__word-1-2gram-max1024_scores.csv` or `rbs_r02__sentence-transformers__BAAI_bge-m3_scores.csv`. The summary also records both the requested embedding choice and the actual embedding backend/model used.

`tfidf` is a fully local scikit-learn fallback, not a neural semantic model. It uses `TfidfVectorizer` on prompt text with English stop-word removal, word unigrams/bigrams, and up to 1024 text features, then L2-normalizes those prompt vectors. This is useful for smoke tests and offline operation without a local transformer model, but it captures lexical overlap rather than deep semantic similarity.

PCA component signs are arbitrary: a valid PCA solution can flip any PC vector without changing explained variance, Mahalanobis distances, or outlier rankings. Treat PC magnitudes, such as `|z|`, as the stable severity diagnostic. Signed PC scores and prompt loadings are still useful within a fixed run because they preserve the geometric pattern of subjects and prompts in the fitted PC orientation; they should not be read as fixed clinical direction unless a separate anchoring convention is added.

By default, the selected semantic dimension `D` is the first PC count whose cumulative prompt-embedding variance reaches `--variance-threshold`. If the threshold is not reached within `--max-components`, the variance rule uses `D=--max-components` and records `variance_threshold_reached=False` in the summary. Use `--max-components 0` to evaluate all possible prompt PCs.

Each run also reports several D-choice diagnostics in parallel, and `--d-selection` chooses which one is used for the actual Mahalanobis scores:

- cumulative variance threshold with max-component fallback,
- largest adjacent eigengap ratio `lambda_D / lambda_{D+1}`,
- Horn-style parallel analysis against independently permuted embedding dimensions,
- neighboring-D outlier-set stability via Jaccard overlap.
- max/all evaluated prompt PCs.

Use `--d-null-permutations`, `--d-null-percentile`, `--stability-jaccard-threshold`, and `--stability-consecutive` to tune these diagnostics.

## Output Files

For every analyzed table, the CLI writes:

- `<table>__<embedding>_scores.csv`: row-level Mahalanobis scores, semantic PCs, ranks, and outlier flags (`Is_Outlier_Theo`, `Is_Outlier_Emp<pct>`; with `--pan-mild`, also `At_Ceiling` and `Is_Pan_Mild_Emp<pct>`).
- `<table>__<embedding>_prompt_loadings.csv`: prompt/item wording loadings on the variance-selected semantic PC components.
- `<table>__<embedding>_item_weights.csv`: compatibility alias for the prompt loading table.
- `<table>__<embedding>_drivers.csv`: top semantic components/items driving the most extreme cases.
- `<table>__<embedding>_stability.csv`: outlier count and Jaccard retention across component counts.
- `<table>__<embedding>_dimension_selection.csv`: per-D variance, eigengap, parallel-analysis null, and outlier-stability evidence.
- `<table>__<embedding>_dimension_methods.csv`: side-by-side D choices from the supported stopping rules.
- `<table>__<embedding>_raw_response_umap.csv`: UMAP coordinates fit directly on normalized raw item responses, without semantic embedding.
- `<table>__<embedding>_semantic_pc_umap.csv`: UMAP coordinates fit on semantic PC response scores using the variance-selected subspace.
- `case_studies/<table>__<embedding>_rankNNN_outlier_NNN.txt`: one standalone anonymized case-study report per top outlier.
- `<table>__<embedding>_case_study_label_map.csv`: private mapping from case-study `Outlier #N` labels back to subject identifiers.
- `combined_prompt_inventory.csv`: combined-mode prompt inventory, including reverse-scoring source.
- `combined_auto_reverse_warnings.txt` and `combined_auto_reverse_warnings.csv`: combined-mode audit trail for automatic opposite-polarity interpretations.
- `summary.csv`: one package-level summary row per scanned table.
- `<embedding>_summary.csv`: embedding-specific copy of the summary, so multiple model runs can share an output folder.

The plotting script writes:

- `<table>__<embedding>_variance_jaccard.pdf`: semantic variance, neighboring-D Jaccard, and outlier count across D.
- `<table>__<embedding>_raw_response_umap.pdf`: UMAP from normalized raw item responses.
- `<table>__<embedding>_semantic_pc_umap.pdf`: UMAP from variance-selected semantic PC response scores.
- `<table>__<embedding>_pc_z_heatmap.pdf`: outlier subjects by signed semantic PC z-scores. Use `--absolute-heatmap` for a magnitude-only view.
- `<table>__<embedding>_outlier_label_map.csv`: private mapping from anonymized plot labels such as `Outlier #1` back to subject identifiers. PDFs use only anonymized outlier labels.

Use `--skip-umap` to run only the semantic PCA/Mahalanobis outputs when `umap-learn` is not installed.
In this Python 3.7 Anaconda stack, bare `import umap` can trip a numba cache-locator error. The package imports UMAP through an internal shim that disables numba function caching while keeping JIT enabled, so run UMAP through `survey_semantics` rather than importing `umap` directly in notebooks.

## Repository Layout

```text
src/survey_semantics/   reusable package (io, prompts, scales, embedding, pipeline, combined, cli, plotting)
tests/                  unit tests for IO and core analysis
scripts/                convenience scripts
docs/                   pipeline_overview.md (general pipeline), env_setup.md, integration notes
examples/               prompt-file templates; examples/nhis/ — full NHIS worked example
```
