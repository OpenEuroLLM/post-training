"""Tokenize the datasets of a LlamaFactory config into its tokenized_path, without training.

The submission copies this file into the run directory, and a --tokenize-only job runs it
under torchrun inside the LlamaFactory container. post_training does not import it: the
tokenize step needs the container's llamafactory package, so that import is deferred to
tokenize().

It runs the first steps of the LlamaFactory 0.9.5 workflows (load_tokenizer,
get_template_and_fix_tokenizer, get_dataset) and exits before load_model.

Two jobs can target the same tokenized_path, e.g. after a double submit, so each job
writes to its own partial directory and publishes it with one rename. The rename of a
directory fails when the target is a non-empty directory, so the first job to publish
wins and the other discards its data; neither can expose a half-written directory.
"""

import errno
import os
import shutil
import sys

import yaml

# The stage that each workflow passes to get_dataset(); DPO reads pairwise data like RM.
_DATASET_STAGE = {"pt": "pt", "sft": "sft", "rm": "rm", "dpo": "rm", "kto": "kto", "ppo": "ppo"}


def has_data(path: str) -> bool:
    """Same test as LlamaFactory's has_tokenized_data(): a directory that is not empty."""
    return os.path.isdir(path) and bool(os.listdir(path))


def partial_path(tokenized_path: str) -> str:
    """The directory that only this job writes to, named after its SLURM job ID.

    A requeued job keeps its job ID, so it finds and removes its own interrupted save.
    Outside SLURM the process ID keeps two local runs apart.
    """
    job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    return f"{tokenized_path.rstrip('/')}.partial-{job_id}"


def publish(partial: str, tokenized_path: str) -> bool:
    """Rename *partial* to *tokenized_path*; False when another job published first.

    rename() replaces a missing or empty target directory in one step and fails on a
    non-empty one, so two jobs cannot both publish. The loser removes its partial
    directory, so nothing is left next to the data that the training job loads.
    """
    try:
        os.rename(partial, tokenized_path)
    except OSError as exc:
        if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
            raise
        shutil.rmtree(partial)
        return False
    return True


def tokenize(args: dict) -> None:
    """Run the LlamaFactory data pipeline for *args* and save it to args["tokenized_path"]."""
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

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


def main(config_path: str) -> None:
    with open(config_path) as f:
        args = yaml.safe_load(f)

    # DeepSpeed does not change the tokenized data, and the job has one GPU.
    args.pop("deepspeed", None)

    tokenized_path = args["tokenized_path"]
    # Exit code 0: the job script would otherwise resubmit a job that has nothing to do.
    if has_data(tokenized_path):
        print(f"tokenized_path '{tokenized_path}' already holds data. Nothing to tokenize.")
        return

    partial = partial_path(tokenized_path)
    # Remove any interrupted save from a previous attempt so this run starts cleanly.
    shutil.rmtree(partial, ignore_errors=True)
    args["tokenized_path"] = partial
    tokenize(args)

    if publish(partial, tokenized_path):
        print(f"Tokenized dataset saved at {tokenized_path}. Train with this tokenized_path.")
    else:
        print(
            f"Another job saved tokenized data at {tokenized_path} while this job ran. "
            "The data of this job was discarded; train with that tokenized_path."
        )


if __name__ == "__main__":
    main(sys.argv[1])
