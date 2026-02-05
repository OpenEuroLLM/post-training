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


def tokenize_example(example, tokenizer, max_length):
    """Apply chat template and tokenize a single example."""
    messages = example["messages"]

    # Apply chat template to convert messages to text
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize the text
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # Add labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


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

    # Load tokenizer (NOT the model - saves ~16GB of memory for 8B models)
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not set (common for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.pad_token}")

    # Optionally load chat template from another model
    if chat_template_path:
        print(f"Loading chat template from {chat_template_path}")
        template_tokenizer = AutoTokenizer.from_pretrained(chat_template_path)
        tokenizer.chat_template = template_tokenizer.chat_template

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

    # Tokenize datasets
    print(f"Tokenizing train dataset with {args.num_proc} processes...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_example(x, tokenizer, max_length),
        num_proc=args.num_proc,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )

    tokenized_eval = None
    if eval_dataset is not None:
        print(f"Tokenizing eval dataset with {args.num_proc} processes...")
        tokenized_eval = eval_dataset.map(
            lambda x: tokenize_example(x, tokenizer, max_length),
            num_proc=args.num_proc,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval",
        )

    # Save the tokenized datasets
    os.makedirs(output_path, exist_ok=True)

    train_path = os.path.join(output_path, "train")
    print(f"Saving tokenized train dataset to {train_path}")
    tokenized_train.save_to_disk(train_path)

    if tokenized_eval is not None:
        eval_path = os.path.join(output_path, "eval")
        print(f"Saving tokenized eval dataset to {eval_path}")
        tokenized_eval.save_to_disk(eval_path)

    # Print summary
    print("=" * 50)
    print("Done!")
    print(f"  Train examples: {len(tokenized_train)}")
    if tokenized_eval:
        print(f"  Eval examples:  {len(tokenized_eval)}")
    print(f"  Output path:    {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
