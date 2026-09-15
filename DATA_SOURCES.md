# Data sources and identifiers

No dataset records are included in this repository.

## Turkish Wiki NER

Download the dataset from <https://github.com/turkish-nlp-suite/Turkish-Wiki-NER-Dataset>. The workflow reads the official `train.conll`, `dev.conll`, and `test.conll` files. Their expected SHA-256 values and counts are stored in `configs/data_sources.json`.

Use the official upstream files to determine which examples belong to each split. During preparation, the workflow generates local IDs from the split name and sentence order.

## Turkish Twitter NER for domain shift

The domain-shift annotation source is <https://github.com/SU-NLP/SUNLP-Twitter-NER-Dataset>. Post text is not redistributed. To reproduce the domain-shift experiment, obtain the post text through a lawful route and prepare the three augmented TSV files named in `configs/data_sources.json`. Each TSV must contain `tweet_id`, `start_pos`, `end_pos`, `named_entity_type`, and `tweet_text`.

Only post IDs are published under `datasets/domain_shift/twitter_ner/split_ids/`. They define the exact subset and order used by the domain-shift experiment.

Within the domain-shift workflow, Twitter content can disappear or change. If any required ID cannot be recovered with the recorded text snapshot, the result is a replication on a different sample rather than an exact reproduction.
