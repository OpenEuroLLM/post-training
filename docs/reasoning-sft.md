# Reasoning SFT on LUMI

Reasoning continuation is supervised fine-tuning: the trainer does not need a
separate `reasoning` method. It does need stricter data preparation and launch
discipline than a general instruction run. The reference profile is
[`configs/trl/reasoning-sft-lumi.yaml`](../configs/trl/reasoning-sft-lumi.yaml).

## What the framework already enforces

The TRL path is suitable for a production reasoning run because it:

- applies the Qwen3 chat template with generation markers and refuses to start
  if assistant-only loss cannot be constructed;
- drops rows whose assistant answer contributes no supervised tokens;
- can drop a row when the 16K boundary cuts through its supervised span, which
  prevents training on an unterminated `<think>` block or final answer;
- packs complete conversational records and derives the step count from a
  training-token budget;
- aligns generation EOS with the chat template's `<|im_end|>` turn terminator;
- pins the parent model, tokenizer, and Hub datasets to revisions;
- freezes the run config, source, Accelerate profile, and generated SLURM job;
- supports a one-GPU `--tokenize-only` gate before the full allocation; and
- resumes full checkpoints and writes lightweight inference checkpoints.

The reference profile adds the multi-node FSDP configuration used by the 9B,
16K LUMI run. The generated job still supplies the live node count, process
count, machine rank and rendezvous address, so the same profile works for the
one-GPU gate and the eight-node run.

## Data contract

The training input is one immutable, materialized Parquet dataset with a
`messages` column. Each value is a list of `{role, content}` messages and the
last turn is the assistant answer. Reasoning examples contain a non-empty
`<think>...</think>` block followed by a non-empty final answer. Replay examples
may contain a normal answer without a think block.

Prepare the artifact before this trainer sees it. A production builder should:

1. pin every source to a commit revision;
2. normalize records to the `messages` contract without flattening prompt and
   answer into one string;
3. keep only complete reasoning traces and final answers;
4. reject overlength rows rather than truncate them;
5. reject exact lexical loops (the current run uses a repeated 30-gram at least
   20 times as the strict threshold);
6. deduplicate normalized user prompts across sources, in a deterministic
   priority order;
7. allocate mixture shares by rendered tokens, not rows; and
8. write a manifest with source revisions, filter counts, selected row/token
   counts, tokenizer/template identity, seed, and output SHA-256.

The framework's ordinary `data.datasets[].weight` is deliberately a row-count
multiplier. It is useful for similarly distributed instruction datasets, but
it does **not** reproduce a token-balanced reasoning mixture whose traces range
from hundreds to tens of thousands of tokens. The reference config therefore
consumes the already-materialized mixture as one dataset.

For the current conservative continuation recipe, the materialized artifact is
524,288,000 rendered tokens: a 65% reasoning / 35% instruction-replay target,
plus a fixed multilingual reasoning coverage floor consumed once before the
weighted allocation. The exact source recipe and its manifest belong with the
data artifact; the training repository records the immutable artifact used by
the run.

## Configure the run

Copy the reference YAML and replace these site-specific values:

- `container.image` and `container.bind_mounts`;
- `container.env_file` with the shared Hugging Face cache variables;
- `data.datasets[0].path` with the immutable Parquet artifact;
- `slurm.account`; and
- `paths.output_base` if the default is not on a persistent filesystem.

Do not change the pinned model revision, Qwen3 template, 16K sequence length,
assistant-only loss behavior, or Liger fused-loss settings without a new smoke
test. The parent checkpoint contains its tokenizer, so the tokenizer follows
the same pinned revision.

## Gate, train, verify

First render and tokenize through the same code path as production:

```bash
python scripts/submit.py \
  --config configs/trl/reasoning-sft-lumi.yaml \
  --tokenize-only
```

Read the generated `slurm-<job-id>.out` and `.err`. The gate passes only when:

- the decoded preview uses Qwen3 ChatML and has a supervised assistant answer;
- no row is cut part-way through its supervised span;
- the resolved step count is 500;
- the parent and dataset paths resolve from the shared cache/filesystem; and
- trainer construction completes without an FSDP, FlashAttention or Liger
  error.

Then submit the unchanged configuration:

```bash
python scripts/submit.py --config configs/trl/reasoning-sft-lumi.yaml
```

After training starts, confirm that the full run reuses the tokenization cache,
that loss is finite and decreasing, that every rank advances together, and
that a resumable checkpoint is written at step 100. A release candidate should
be compared with the parent on the same multilingual instruction, math/code
reasoning, long-context, and repetition-loop probes. Publish the parent commit,
data-manifest checksum, frozen config, environment, SLURM job IDs, selected
checkpoint, and evaluation results in the model card.

## Scope still outside this repository

This trainer does not currently build token-balanced, cross-source-deduplicated
mixtures, perform benchmark decontamination, or run the evaluation suite. Those
steps should stay fail-closed in the data/evaluation pipelines and hand this
repository immutable artifacts plus manifests. Keeping that boundary explicit
prevents a convenient row-weighted training config from being mistaken for the
audited production data recipe.
