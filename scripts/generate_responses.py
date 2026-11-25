"""Generate a think and a no_think response for every prompt in a JSON prompt file.

The prompt file has the layout ``{"prompts": [{"lang": ..., "prompt": ..., ...}, ...]}``.
The output file has the same entries, each with an added field::

    "generated_response": {"think": "...", "no_think": "..."}

Usage:
    python scripts/generate_responses.py \
        --model runs/sft-.../inference_checkpoints/step_1000 \
        --prompts data/eval_prompts/multilingual_prompts.json \
        --output results/multilingual_responses.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from post_training.chat_templates.registry import get_chat_template, infer_end_token_from_render

# Values of the `thinking_instruction` template variable, as used in the SFT data.
MODES = ("think", "no_think")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local checkpoint directory")
    p.add_argument("--prompts", required=True, type=Path, help="input JSON prompt file")
    p.add_argument("--output", required=True, type=Path, help="output JSON file")
    p.add_argument("--chat-template", default="qwen3", help="name in the chat template registry")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-p", type=float, default=0.95, help="used only when temperature > 0")
    p.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    p.add_argument(
        "--preview-chars",
        type=int,
        default=2048,
        help="characters of each response to print when an entry is done; 0 = none, -1 = all",
    )
    return p.parse_args()


def stop_token_ids(tokenizer, model) -> list[int]:
    """The template's turn terminator plus every eos the tokenizer and the model know.

    An eos id from the model is kept only when it is an added token of this tokenizer.
    A ``generation_config`` can carry an id from a different vocabulary (for example
    Qwen's 151645), which is an ordinary word piece here and would cut responses short.
    """
    added_ids = set(tokenizer.get_added_vocab().values())
    ids = []
    probe = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    rendered = tokenizer.apply_chat_template(probe, tokenize=False)
    end_token = infer_end_token_from_render(rendered, tokenizer.get_added_vocab())
    if end_token is not None:
        ids.append(tokenizer.convert_tokens_to_ids(end_token))
    ids.append(tokenizer.eos_token_id)
    # `config.json` usually still has the pretraining eos, also when training
    # changed the tokenizer's eos_token and the generation_config.
    model_eos = []
    for value in (model.generation_config.eos_token_id, model.config.eos_token_id):
        model_eos.extend(value if isinstance(value, list) else [value])
    ignored = [i for i in model_eos if i is not None and i not in added_ids]
    if ignored:
        print(f"Ignoring eos ids that are ordinary tokens: {ignored}")
    ids.extend(i for i in model_eos if i in added_ids)
    return list(dict.fromkeys(i for i in ids if i is not None))


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: list[str], mode: str, stop_ids, args) -> list[str]:
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            thinking_instruction=mode,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    inputs = inputs.to(model.device)

    greedy = args.temperature == 0.0
    # An explicit `eos_token_id=None` removes the eos ids of generation_config, and
    # generation then never stops. Pass the argument only when there is a list.
    eos_kwargs = {} if stop_ids is None else {"eos_token_id": stop_ids}
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=not greedy,
        temperature=None if greedy else args.temperature,
        top_p=None if greedy else args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        **eos_kwargs,
    )

    responses = []
    for row in output[:, inputs["input_ids"].shape[1] :].tolist():
        if stop_ids is None:
            # Remove only the batch padding. The token that stopped the row stays visible.
            end = len(row)
            while end > 0 and row[end - 1] == tokenizer.pad_token_id:
                end -= 1
        else:
            end = next((i for i, token in enumerate(row) if token in stop_ids), len(row))
        # Keep special tokens, so the <think> block stays visible in the response.
        responses.append(tokenizer.decode(row[:end], skip_special_tokens=False))
    return responses


def preview(text: str, max_chars: int) -> str:
    if max_chars < 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]} ... [{len(text)} characters in total]"


def print_entry(number: int, total: int, entry: dict, max_chars: int) -> None:
    """Print one finished entry: the prompt and its response in each mode."""
    print(
        f"\n=== [{number}/{total}] {entry['lang']} #{entry.get('prompt_index', '?')} "
        f"{entry.get('category', '')}"
    )
    print(f"prompt: {entry['prompt']}")
    for mode, response in entry["generated_response"].items():
        print(f"--- {mode}\n{preview(response, max_chars)}")


def main() -> None:
    args = parse_args()

    data = json.loads(args.prompts.read_text(encoding="utf-8"))
    entries = data["prompts"][: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = get_chat_template(args.chat_template)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    # Disabled for now: run with the eos ids from the checkpoint's generation_config.json.
    # stop_ids = stop_token_ids(tokenizer, model)
    # print(f"Stop tokens: {tokenizer.convert_ids_to_tokens(stop_ids)}")
    stop_ids = None
    print(f"Stop ids from generation_config: {model.generation_config.eos_token_id}")

    result = {
        "model": args.model,
        "chat_template": args.chat_template,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "prompts": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    run_start = time.monotonic()
    for start in range(0, len(entries), args.batch_size):
        batch = entries[start : start + args.batch_size]
        prompts = [entry["prompt"] for entry in batch]
        by_mode = {
            mode: generate_batch(model, tokenizer, prompts, mode, stop_ids, args) for mode in MODES
        }
        for i, entry in enumerate(batch):
            response = {mode: by_mode[mode][i] for mode in MODES}
            result["prompts"].append({**entry, "generated_response": response})
            if args.preview_chars != 0:
                print_entry(
                    len(result["prompts"]), len(entries), result["prompts"][-1], args.preview_chars
                )

        # Written after every batch, so a stopped job keeps its finished entries.
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        done = len(result["prompts"])
        elapsed = time.monotonic() - run_start
        remaining = elapsed / done * (len(entries) - done)
        print(
            f"\n{done}/{len(entries)} prompts done, {elapsed / 60:.1f} min elapsed, "
            f"about {remaining / 60:.1f} min remaining",
            flush=True,
        )


if __name__ == "__main__":
    main()
