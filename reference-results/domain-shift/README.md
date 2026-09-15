# Domain-shift reference results

Use these aggregate results from the original five-seed encoder experiment to compare with your domain-shift run. Dataset records and trained weights are not included.

- `domain_shift_summary.json`: 70 seed-level aggregate evaluations, cell summaries, and paired transfer gaps
- `cell_summaries.csv`: arithmetic means and sample standard deviations by family, cell, and mapping profile
- `paired_gap_summaries.csv`: paired per-seed transfer, adaptation, and reverse-transfer summaries
- `training_runs_aggregate.json`: status, duration, and test metric for the 20 local training runs, with output paths removed
- `domain_shift_config.json`: fixed cells, seeds, model revisions, training settings, and mapping profiles
- `environment_summary.json`: runtime device and PyTorch version recorded during the original evaluation

The domain-shift Twitter post text is not present. The public post IDs are under `datasets/domain_shift/twitter_ner/split_ids/`.
