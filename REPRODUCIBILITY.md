# Reproducibility guide

## 1. Choose an experiment

You can reproduce either experiment or both. The Wiki experiment compares encoders and language models on WikiNER. The domain-shift experiment evaluates encoder transfer between WikiNER and Turkish Twitter NER. Twitter data is required only for domain shift. For Wiki only, skip sections 4 and 7 and use the Wiki-only training commands in section 5. For domain shift only, obtain both datasets, train both domains, and skip section 6.

Dataset records and trained weights are not included in the package. Obtain the source datasets separately; the workflow writes prepared datasets and trained models under `.artifacts/`, which Git ignores.

## 2. Set up the environment

The original Wiki run used Python 3.10.12 on Linux with an NVIDIA A100-SXM4-40GB. `requirements-lock.txt` records the application environment, and `requirements-vllm-lock.txt` fixes the recorded vLLM version at 0.29.0. The compatibility files `requirements.txt` and `requirements-vllm.txt` are available when the exact lock cannot be installed on a different CUDA runtime. Set `APP_REQUIREMENTS_FILE` or `VLLM_REQUIREMENTS_FILE` to select a compatibility file, and record that deviation.

For Google Colab, mount Drive if you want private outputs to persist, then place this repository in the mounted location. Runtime environments and caches should stay on local scratch storage:

```bash
bash ensure_python310.sh
bash install_colab_requirements.sh
```

After installation, select the application environment for the remaining commands:

```bash
export LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/local-scratch/thesis-artifacts-wiki}"
export DEFAULT_VENV_ROOT="${DEFAULT_VENV_ROOT:-$LOCAL_SCRATCH_ROOT/.venvs}"
export APP_VENV="${APP_VENV:-$DEFAULT_VENV_ROOT/app}"
export VLLM_VENV="${VLLM_VENV:-$DEFAULT_VENV_ROOT/vllm}"
source "$APP_VENV/bin/activate"
export PYTHON_BIN="$APP_VENV/bin/python"
```

If you used custom environment paths during installation, use those same paths here. Run subsequent commands from the repository root in this shell. In Colab, activation and exports in one shell cell do not persist into another; repeat this block at the start of each `%%bash` cell that runs the workflow.

On another Linux system, create separate Python 3.10 environments for the application and vLLM. Install `requirements-lock.txt` in the application environment and `requirements-vllm-lock.txt` in the vLLM environment. Use the compatibility files only if the recorded dependencies cannot be installed, and document the differences. Set `APP_VENV` and `VLLM_VENV` to the environment paths and `PYTHON_BIN` to the application environment's Python executable. Set `LOCAL_SCRATCH_ROOT` if the default scratch location does not fit your machine.

Activate the application environment with `source "$APP_VENV/bin/activate"` so that the direct `python3` commands below use the same environment as `reproduce.sh`.

## 3. Obtain and validate WikiNER

The public WikiNER source is:

```text
https://github.com/turkish-nlp-suite/Turkish-Wiki-NER-Dataset
```

Clone or download it outside this repository. `WIKI_SOURCE_DIR` may point at the repository root or a narrower directory that contains `train.conll`, `dev.conll`, and `test.conll`.

```bash
export WIKI_SOURCE_DIR=/path/to/Turkish-Wiki-NER-Dataset
python3 scripts/validate_source_data.py --wiki-source "$WIKI_SOURCE_DIR"
bash reproduce.sh prepare-wiki
```

The validator compares the three files with the thesis snapshot hashes in `configs/data_sources.json`. The expected materialized counts are 17,967 training examples, 1,000 development examples, and 1,000 test examples. If a public source revision has changed, `--allow-different-snapshot` permits a documented replication, but it is not an exact reproduction of the archived snapshot.

The workflow preserves the official WikiNER splits and generates local sentence IDs from each sentence's position within its split. You do not need a separate WikiNER ID file.

## 4. Obtain Twitter data for the domain-shift experiment

The source annotations are described at:

```text
https://github.com/SU-NLP/SUNLP-Twitter-NER-Dataset
```

Obtain the source annotations and permitted post text, then prepare `train_with_tweet_text.tsv`, `val_with_tweet_text.tsv`, and `test_with_tweet_text.tsv`. Each file must contain `tweet_id`, `start_pos`, `end_pos`, `named_entity_type`, and `tweet_text`. Set `TWITTER_SOURCE_DIR` to their directory. Post text is not included in this package.

```bash
export TWITTER_SOURCE_DIR=/path/to/researcher-obtained-twitter-source
bash reproduce.sh validate-domain-ids
python3 scripts/validate_source_data.py --twitter-source "$TWITTER_SOURCE_DIR"
```

The public domain-shift ID files select 1,347 training posts, 292 validation posts, and 286 test posts. Deleted posts, changed text, or unavailable post text can prevent exact reconstruction. Dataset preparation stops if a required post ID is missing.

## 5. Train the encoders locally

The two encoder revisions and all training settings are fixed in `configs/reproduction.json`. Seed 42 supplies the encoder checkpoints used by the primary Wiki matrix. The domain-shift experiment uses seeds 42, 43, 44, 45, and 46 for both model families and both training domains.

To prepare and train only the primary Wiki checkpoints:

```bash
bash reproduce.sh prepare-wiki
bash reproduce.sh train-wiki
```

To prepare both domains and train the complete five-seed matrix:

```bash
bash reproduce.sh prepare-domain
bash reproduce.sh train-domain
```

Training both domains produces 20 encoder runs under `.artifacts/models/bert/`: two model families, two training domains, and five seeds. The Wiki-only commands train two encoders with seed 42.

## 6. Run the Wiki experiment

The fixed executable matrix is `comparison-wrapper/wiki_http_exact_matrix.json`. It contains eleven LLMs, two transformer encoders, two transformer device profiles, and seven prompting configurations per LLM. The count is:

```text
11 LLMs x (1 zero-shot + 3 one-shot + 3 three-shot) = 77 rows
2 encoders x (CPU + A100 GPU)                         =  4 rows
Executable total                                     = 81 rows
Current reported cohort                              = 81 rows
```

After preparing WikiNER and training the seed-42 Wiki encoders, check that the inputs, checkpoints, and experiment configuration satisfy the Wiki protocol:

```bash
bash reproduce.sh wiki-preflight
```

Preflight verifies the 1,000-example test count, materialized split hashes, train-only demonstration selection, one-shot and three-shot nesting, exact scoring identifier, latency identifier, float32 CPU and GPU encoder profiles, A100 requirement, and the 81-row executable definition.

First, test the workflow on 20 examples with one language model and one encoder. This checks execution; its results are not comparable to the full 1,000-example reference results:

```bash
bash run_wiki_model_matrix.sh \
  --models qwen3-0.6b \
  --transformers dbmdz-distilbert-base-turkish-cased-wiki-tuned \
  --strategies zero-shot \
  --max-examples 20 \
  --results-dir .artifacts/results/wiki_smoke20 \
  --allow-failures
```

Run the complete matrix and analysis:

```bash
bash reproduce.sh wiki-matrix
bash reproduce.sh wiki-analysis
```

Every comparable request uses localhost HTTP, batch size 1, concurrency 1, stored test order, 20 unrecorded warm-up requests, and no automatic retry. The scorer requires exact token boundaries and exact Wiki labels. It performs no substring recovery, boundary repair, case folding, label aliasing, or overlap projection.

The latency outputs include client end-to-end percentiles and component summaries. Encoders record tokenization and transfer, synchronized model execution, postprocessing, serialization, and server total. LLM runs record time to first token, generation after the first token, end-to-end latency, token use, finish reason, and server metrics when vLLM returns them. Startup and model loading are outside the steady-state measurement.

## 7. Run the domain-shift experiment

After training the five-seed encoder matrix:

```bash
bash reproduce.sh domain-shift
```

The primary conservative mapping evaluates four cells for each family and seed: Wiki to Wiki, Wiki to Twitter, Twitter to Twitter, and Twitter to Wiki. This gives 40 primary evaluations. Wiki to Twitter is then rescored under `extended_safe`, `extended_risky`, and `strict_direct_overlap`, adding 30 mapping-sensitivity evaluations. `analysis/summarize_domain_shift.py` requires all 70 results unless `--allow-partial` is explicitly used.

Individual domain-shift evaluations are written under `.artifacts/results/cross-domain/`. Each evaluation records exact entity-level precision, recall, and F1 after label mapping, plus its bootstrap interval. The combined summary is written to `.artifacts/results/domain_shift_summary.json` and reports arithmetic means and sample standard deviations across seeds, as well as paired per-seed transfer gaps.

## 8. Compare against the public references

First, check the completeness of the bundled reference results and the consistency of the Wiki aggregates. This command checks the supplied reference files; it does not compare them with your run:

```bash
bash reproduce.sh verify-reference
```

The Wiki files include the full 81-row metric table, prompt-strategy aggregates, detailed latency-component aggregates, entity-type aggregates, and CPU/GPU speedups. The domain-shift files include 70 seed-level aggregate rows and the derived cell and paired-gap summaries. Neither directory contains a dataset example or per-example prediction.

Compare your Wiki `matrix_results.csv` with `reference-results/wiki/matrix_results_81.csv`, matching model, device, prompting strategy, and seed. Compare `.artifacts/results/domain_shift_summary.json` with `reference-results/domain-shift/domain_shift_summary.json`, matching model family, training and test domains, mapping profile, and seed for individual evaluations. Compare the `cell_summaries` and `paired_gap_summaries` sections for results aggregated across seeds.

Exact metric equality is expected only when the source snapshots, base-model revisions, software stack, prompts, random seeds, and relevant hardware behavior match. Record differences in source data, model revisions, software, and hardware alongside your results. Report metric differences explicitly. If you use an acceptance tolerance, state it and explain your choice; this guide does not specify a numerical tolerance.

## 9. Record your environment

With `PYTHON_BIN` set to the application environment's Python executable, run:

```bash
bash reproduce.sh capture-env
```

This saves a software and hardware record to `.artifacts/environment.json`. Keep it with your results and document any changes to the data, configuration, or dependencies.
