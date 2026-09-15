# WikiNER and domain-shift reproducibility package

This repository provides the code, configurations, and reference results needed to reproduce two thesis experiments:

1. The primary Wiki experiment compares two Wiki-fine-tuned Turkish encoders with eleven prompt-based language models on the fixed 1,000-example WikiNER test split. Every model is called through localhost HTTP. The executable and reported matrices both contain all 81 completed configurations.
2. The domain-shift experiment trains BERTurk and DistilBERTurk on WikiNER and Turkish Twitter NER data across seeds 42 through 46. It evaluates four combinations of training and test domains, including within-domain baselines, and three additional label-mapping variants, for 70 evaluations in total.

## What is included and what you need to obtain

The repository contains source code, fixed configuration, domain-shift Twitter post IDs, tests, and aggregate reference results. It does not contain WikiNER records, post text from the domain-shift source, prompts containing dataset records, per-example predictions, model responses, trained weights, checkpoints, tokenizer exports, or credentials.

WikiNER has official train, development, and test files. Download the official WikiNER splits from the upstream project and validate them against the recorded checksums. To run the domain-shift experiment, also obtain the Twitter source described in [DATA_SOURCES.md](DATA_SOURCES.md). The ID files under `datasets/domain_shift/twitter_ner/split_ids/` select and order the domain-shift subset without redistributing post text.

## Quick checks

Python 3.10 is the recorded runtime. The repository-level smoke test has no dataset or model requirement:

```bash
bash reproduce.sh smoke
```

This command checks the bundled reference results, Twitter split IDs, configuration files, and shell syntax, and runs the unit tests. It does not run either experiment or verify results from a new run.

`SHA256SUMS` lists checksums for the distributed files.

## Reproduce the Wiki and domain-shift experiments

Follow the environment setup and source-validation instructions in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) first. The commands below run both experiments. To reproduce only the Wiki experiment, use `prepare-wiki` and `train-wiki` instead of `prepare-domain` and `train-domain`, and omit the Twitter source setting and final `domain-shift` command.

```bash
export WIKI_SOURCE_DIR=/path/to/Turkish-Wiki-NER-Dataset
# Required only by the domain-shift workflow:
export TWITTER_SOURCE_DIR=/path/to/researcher-obtained-twitter-source

bash reproduce.sh prepare-domain
bash reproduce.sh train-domain
bash reproduce.sh wiki-preflight
bash reproduce.sh wiki-matrix
bash reproduce.sh wiki-analysis
bash reproduce.sh domain-shift
```

Your reconstructed datasets, trained weights, responses, predictions, and results are stored locally under directories excluded from version control. Downloaded base models are stored in the configured cache.

## Reference results

`reference-results/wiki/` contains the original results for all 81 configurations, including the seven Trendyol configurations, with aggregate quality, entity-type, and latency measurements. These configurations match the executable matrix.

`reference-results/domain-shift/` contains the 70 aggregate seed-level evaluations, cell summaries, paired-gap summaries, and aggregate training records. These files contain no examples, predictions, or weights.
