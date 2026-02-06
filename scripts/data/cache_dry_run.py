"""Sanity check: load model, tokenizer, and dataset from cache only (no internet).

Verifies that all HuggingFace resources in a config are properly cached and loadable
before submitting a training job. Fails loudly if anything is missing.

Usage:
    python scripts/data/sanity_check.py configs/sft/tulu3.yaml
    python scripts/data/sanity_check.py configs/dpo/tulu3.yaml
"""

import os

# Must be set before importing any HuggingFace libraries
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse

import yaml
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Sanity check: load all HF resources from cache (offline)")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--skip-model", action="store_true", help="Skip loading model weights (faster, less memory)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name_or_path")
    dataset_name = config.get("dataset_name")
    chat_template_path = config.get("chat_template_path")

    ok = True

    # Model
    if model_name:
        print(f"Model: {model_name}")
        try:
            model_config = AutoConfig.from_pretrained(model_name)
            print(f"  Architecture: {model_config.architectures}")
            if args.skip_model:
                print("  Weights: skipped (--skip-model)")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
                params = sum(p.numel() for p in model.parameters())
                print(f"  Parameters: {params:,}")
                print(f"  Dtype: {next(model.parameters()).dtype}")
                del model
        except OSError as e:
            print(f"  FAILED: {e}")
            ok = False

    # Tokenizer (from model)
    if model_name:
        print(f"\nTokenizer: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"  Vocab size: {tokenizer.vocab_size:,}")
        except OSError as e:
            print(f"  FAILED: {e}")
            ok = False

    # Chat template tokenizer
    if chat_template_path and chat_template_path != model_name:
        print(f"\nChat template: {chat_template_path}")
        try:
            chat_tokenizer = AutoTokenizer.from_pretrained(chat_template_path)
            print(f"  Vocab size: {chat_tokenizer.vocab_size:,}")
            print(f"  Has chat template: {chat_tokenizer.chat_template is not None}")
        except OSError as e:
            print(f"  FAILED: {e}")
            ok = False

    # Dataset
    if dataset_name:
        dataset_config = config.get("dataset_config")
        print(f"\nDataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, name=dataset_config)
            for split, ds in dataset.items():
                print(f"  {split}: {len(ds):,} examples")
            first_split = next(iter(dataset))
            print(f"  Columns: {dataset[first_split].column_names}")
        except OSError as e:
            print(f"  FAILED: {e}")
            ok = False

    if ok:
        print("\nAll resources loaded successfully from cache.")
    else:
        print("\nSome resources failed to load. Run `python scripts/data/cache.py` first.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
