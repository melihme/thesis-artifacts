from __future__ import annotations

import inspect
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from .data import load_manifest, load_materialized_dataset
from .utils import detect_device


DEFAULT_MAX_LENGTH = 512


def build_label_mappings(labels: Sequence[str]) -> tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: index for index, label in enumerate(sorted(labels, key=lambda item: (item != "O", item)))}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def build_dataset_dict(examples_by_split: Dict[str, Sequence[Dict]], label2id: Dict[str, int]) -> DatasetDict:
    dataset_dict = {}
    for split_name, examples in examples_by_split.items():
        dataset_dict[split_name] = Dataset.from_dict(
            {
                "tokens": [example["tokens"] for example in examples],
                "ner_tags": [
                    [label2id[label] for label in example["labels"]]
                    for example in examples
                ],
            }
        )
    return DatasetDict(dataset_dict)


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def tokenize_batch(examples: Dict[str, List[List[str]]]) -> Dict[str, List[List[int]]]:
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )

        aligned_labels: List[List[int]] = []
        for batch_index, word_level_labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_id = None
            label_ids: List[int] = []

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(word_level_labels[word_id])
                else:
                    label_ids.append(-100)
                previous_word_id = word_id

            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    return dataset.map(tokenize_batch, batched=True, desc="Tokenizing")


def build_compute_metrics(id2label: Dict[int, str]):
    def compute_metrics(eval_prediction) -> Dict[str, float]:
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)

        decoded_predictions = []
        decoded_labels = []

        for prediction_sequence, label_sequence in zip(predictions, labels):
            current_predictions = []
            current_labels = []

            for prediction, label in zip(prediction_sequence, label_sequence):
                if label == -100:
                    continue
                current_predictions.append(id2label[int(prediction)])
                current_labels.append(id2label[int(label)])

            decoded_predictions.append(current_predictions)
            decoded_labels.append(current_labels)

        return {
            "precision": precision_score(decoded_labels, decoded_predictions),
            "recall": recall_score(decoded_labels, decoded_predictions),
            "f1": f1_score(decoded_labels, decoded_predictions),
            "accuracy": accuracy_score(decoded_labels, decoded_predictions),
        }

    return compute_metrics


def build_training_arguments(config: Dict, output_dir: Path) -> TrainingArguments:
    signature_parameters = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "per_device_train_batch_size": config["batch_size"],
        "per_device_eval_batch_size": config["batch_size"],
        "save_strategy": config["save_strategy"],
        "logging_strategy": config["logging_strategy"],
        "load_best_model_at_end": config["eval_strategy"] == config["save_strategy"] and config["eval_strategy"] != "no",
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": config["seed"],
        "dataloader_num_workers": config["dataloader_workers"],
        "dataloader_pin_memory": False,
        "save_total_limit": config["save_total_limit"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "lr_scheduler_type": config["lr_scheduler_type"],
        "max_grad_norm": config["max_grad_norm"],
        "max_steps": config["max_steps"],
    }

    if config["warmup_steps"] > 0:
        training_kwargs["warmup_steps"] = config["warmup_steps"]
    elif config["warmup_ratio"] > 0:
        training_kwargs["warmup_ratio"] = config["warmup_ratio"]

    if config["eval_strategy"] == "steps":
        training_kwargs["eval_steps"] = config["eval_steps"]
    if config["save_strategy"] == "steps":
        training_kwargs["save_steps"] = config["save_steps"]
    if config["logging_strategy"] == "steps":
        training_kwargs["logging_steps"] = config["logging_steps"]

    if "eval_strategy" in signature_parameters:
        training_kwargs["eval_strategy"] = config["eval_strategy"]
    elif "evaluation_strategy" in signature_parameters:
        training_kwargs["evaluation_strategy"] = config["eval_strategy"]

    if "save_safetensors" in signature_parameters:
        training_kwargs["save_safetensors"] = True
    if "use_mps_device" in signature_parameters:
        import torch

        training_kwargs["use_mps_device"] = torch.backends.mps.is_available()

    return TrainingArguments(**training_kwargs)


def build_trainer(model, training_args, tokenized_dataset, tokenizer, id2label):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["eval"],
        "data_collator": DataCollatorForTokenClassification(tokenizer),
        "compute_metrics": build_compute_metrics(id2label),
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**trainer_kwargs)


def resolve_resume_checkpoint(checkpoint_dir: Path, resume_mode: str) -> Optional[str]:
    if resume_mode == "never" or not checkpoint_dir.exists():
        return None
    return get_last_checkpoint(str(checkpoint_dir))


def default_train_config() -> Dict:
    return {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.0,
        "warmup_steps": 0,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "max_length": DEFAULT_MAX_LENGTH,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 100,
        "save_total_limit": 2,
        "seed": 42,
        "dataloader_workers": 0,
        "resume": "auto",
        "delete_checkpoints_after_training": True,
    }


def train_token_classifier(
    source_model: str,
    source_model_revision: str,
    dataset_root: Path,
    model_output_dir: Path,
    config: Dict,
) -> Dict:
    config = {**default_train_config(), **config}
    examples_by_split = load_materialized_dataset(dataset_root)
    manifest = load_manifest(dataset_root)
    label_list = manifest["labels"]
    label2id, id2label = build_label_mappings(label_list)
    dataset = build_dataset_dict(examples_by_split, label2id)

    checkpoint_dir = model_output_dir / "checkpoints"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source_model, revision=source_model_revision)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=config["max_length"])

    training_args = build_training_arguments(config=config, output_dir=checkpoint_dir)
    resume_checkpoint = resolve_resume_checkpoint(checkpoint_dir=checkpoint_dir, resume_mode=config["resume"])

    model = AutoModelForTokenClassification.from_pretrained(
        source_model,
        revision=source_model_revision,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        id2label=id2label,
    )

    started_at = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_seconds = time.time() - started_at

    trainer.save_state()
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    eval_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["eval"], metric_key_prefix="eval")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"], metric_key_prefix="test")

    checkpoint_cleanup_deleted = False
    if config.get("delete_checkpoints_after_training", True) and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        checkpoint_cleanup_deleted = True

    return {
        "source_model": source_model,
        "source_model_revision": source_model_revision,
        "device": detect_device(),
        "dataset_root": str(dataset_root),
        "model_output_dir": str(model_output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "resume_checkpoint": resume_checkpoint,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "config": config,
        "label_count": len(label_list),
        "labels": label_list,
        "counts": {split_name: len(split_examples) for split_name, split_examples in examples_by_split.items()},
        "train_seconds": train_seconds,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
        "checkpoint_cleanup": {
            "requested": bool(config.get("delete_checkpoints_after_training", True)),
            "deleted": checkpoint_cleanup_deleted,
            "checkpoint_dir_exists_after_cleanup": checkpoint_dir.exists(),
        },
    }


def evaluate_model(
    model_dir: Path,
    dataset_root: Path,
    split_name: str,
    batch_size: int = 16,
    dataloader_workers: int = 0,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict:
    examples_by_split = load_materialized_dataset(dataset_root)
    manifest = load_manifest(dataset_root)
    label_list = manifest["labels"]
    label2id, id2label = build_label_mappings(label_list)

    if split_name == "all":
        eval_examples = []
        for source_split_name in ("train", "eval", "test"):
            eval_examples.extend(examples_by_split.get(source_split_name, []))
    elif split_name in examples_by_split:
        eval_examples = list(examples_by_split[split_name])
    else:
        raise ValueError(f"Unknown split '{split_name}'. Expected one of {', '.join(examples_by_split)}.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)

    dataset = build_dataset_dict({split_name: eval_examples}, label2id)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=max_length)

    with tempfile.TemporaryDirectory(prefix="bert-eval-", dir=str(model_dir.parent)) as temporary_dir:
        training_args = TrainingArguments(
            output_dir=temporary_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            dataloader_num_workers=dataloader_workers,
            dataloader_pin_memory=False,
        )
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "eval_dataset": tokenized_dataset[split_name],
            "data_collator": DataCollatorForTokenClassification(tokenizer),
            "compute_metrics": build_compute_metrics(id2label),
        }
        trainer_signature = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_signature:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_signature:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_kwargs)
        started_at = time.perf_counter()
        prediction_output = trainer.predict(tokenized_dataset[split_name], metric_key_prefix=split_name)
        wall_time = time.perf_counter() - started_at

    metrics = {
        key[len(f"{split_name}_") :]: value
        if key.startswith(f"{split_name}_")
        else value
        for key, value in prediction_output.metrics.items()
    }

    return {
        "model_dir": str(model_dir),
        "dataset_root": str(dataset_root),
        "split": split_name,
        "source_split_counts": {name: len(examples) for name, examples in examples_by_split.items()},
        "device": detect_device(),
        "item_count": len(eval_examples),
        "wall_time_seconds": wall_time,
        "average_latency_seconds": wall_time / max(1, len(eval_examples)),
        "metrics": metrics,
    }
