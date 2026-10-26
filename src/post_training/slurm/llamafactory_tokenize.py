"""Tokenize the datasets of a LlamaFactory config into its tokenized_path, without training.

The submission copies this file into the run directory, and a --tokenize-only job runs it
under torchrun inside the LlamaFactory container. It is not imported by post_training:
it needs the container's llamafactory package.

It runs the first steps of the LlamaFactory 0.9.5 workflows (load_tokenizer,
get_template_and_fix_tokenizer, get_dataset) and exits before load_model.
"""

import os
import shutil
import sys

import yaml
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer

# The stage that each workflow passes to get_dataset(); DPO reads pairwise data like RM.
_DATASET_STAGE = {"pt": "pt", "sft": "sft", "rm": "rm", "dpo": "rm", "kto": "kto", "ppo": "ppo"}


def main(config_path: str) -> None:
    with open(config_path) as f:
        args = yaml.safe_load(f)

    # DeepSpeed does not change the tokenized data, and the job has one GPU.
    args.pop("deepspeed", None)

    tokenized_path = args["tokenized_path"]
    if os.path.isdir(tokenized_path) and os.listdir(tokenized_path):
        sys.exit(f"tokenized_path '{tokenized_path}' already holds data. Nothing to tokenize.")

    # get_dataset() saves into a partial directory that is swapped into place, so an
    # interrupted job never leaves data that a training job would load.
    partial_path = tokenized_path.rstrip("/") + ".partial"
    shutil.rmtree(partial_path, ignore_errors=True)
    args["tokenized_path"] = partial_path

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage=_DATASET_STAGE[finetuning_args.stage],
        **tokenizer_module,
    )

    # An empty tokenized_path directory is replaced; the check above refuses one with data.
    if os.path.isdir(tokenized_path):
        os.rmdir(tokenized_path)
    os.replace(partial_path, tokenized_path)
    print(f"Tokenized dataset saved at {tokenized_path}. Train with this tokenized_path.")


if __name__ == "__main__":
    main(sys.argv[1])
