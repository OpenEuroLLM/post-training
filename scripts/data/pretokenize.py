"""
Pre-tokenize a dataset using SFTTrainer's built-in tokenization.
No GPU required - model loads on CPU.

Usage:
    python scripts/tokenize.py configs/tokenize/capybara.yaml --num-proc 8

Config file format (YAML):
    model_name: Qwen/Qwen2.5-0.5B-Instruct
    dataset_name: trl-lib/Capybara
    output_path: ./tokenized_capybara
    max_length: 2048            # optional, default: 2048
    train_split: train          # optional, default: train
    eval_split: test            # optional, default: None
    chat_template_path: ...     # optional, model/path to load chat template from (for base models)
"""

import argparse
import os

import yaml
from datasets import get_dataset_split_names, load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize a dataset for SFTTrainer")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--num-proc", type=int, default=4, help="Number of processes for tokenization (default: 4)")
    args = parser.parse_args()

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    output_path = config["output_path"]
    max_length = config.get("max_length", 2048)
    train_split = config.get("train_split", "train")
    eval_split = config.get("eval_split")
    chat_template_path = config.get("chat_template_path")

    # Print configuration
    print("=" * 50)
    print("Configuration:")
    print(f"  Model:        {model_name}")
    print(f"  Dataset:      {dataset_name}")
    print(f"  Output path:  {output_path}")
    print(f"  Max length:   {max_length}")
    print(f"  Train split:  {train_split}")
    print(f"  Eval split:   {eval_split or 'None'}")
    print(f"  Chat template: {chat_template_path or 'None (using model default)'}")
    print(f"  Num proc:     {args.num_proc}")
    print("=" * 50)

    # Load datasets
    print(f"Loading dataset {dataset_name} (train split: {train_split})")
    train_dataset = load_dataset(dataset_name, split=train_split)

    eval_dataset = None
    if eval_split:
        # Check if the eval split exists in the dataset
        available_splits = get_dataset_split_names(dataset_name)
        if eval_split in available_splits:
            print(f"Loading dataset {dataset_name} (eval split: {eval_split})")
            eval_dataset = load_dataset(dataset_name, split=eval_split)
        else:
            print(f"Warning: eval split '{eval_split}' not found. Available splits: {available_splits}")

    # Load tokenizer and optionally set chat template from another model
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if chat_template_path:
        print(f"Loading chat template from {chat_template_path}")
        template_tokenizer = AutoTokenizer.from_pretrained(chat_template_path)
        tokenizer.chat_template = template_tokenizer.chat_template

    # Create trainer with CPU-only model loading
    print(f"Loading model {model_name} on CPU")
    trainer = SFTTrainer(
        model=model_name,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir="tmp_tokenize",
            max_length=max_length,
            dataset_num_proc=args.num_proc,
            model_init_kwargs={
                "device_map": "cpu",
                "torch_dtype": "auto",
                "low_cpu_mem_usage": True,
            },
            use_cpu=True,
        ),
    )

    # Save the tokenized datasets
    os.makedirs(output_path, exist_ok=True)

    train_path = os.path.join(output_path, "train")
    print(f"Saving tokenized train dataset to {train_path}")
    train_dataset = trainer.train_dataset.remove_columns("messages")
    train_dataset.save_to_disk(train_path)

    if trainer.eval_dataset is not None:
        eval_path = os.path.join(output_path, "eval")
        print(f"Saving tokenized eval dataset to {eval_path}")
        eval_dataset = trainer.eval_dataset.remove_columns("messages")
        eval_dataset.save_to_disk(eval_path)

    print("Done!")


if __name__ == "__main__":
    main()
