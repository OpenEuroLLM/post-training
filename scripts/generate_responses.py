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
    return p.parse_args()


def stop_token_ids(tokenizer, model) -> list[int]:
    """The template's turn terminator plus every eos the tokenizer and the model know."""
    ids = []
    probe = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    rendered = tokenizer.apply_chat_template(probe, tokenize=False)
    end_token = infer_end_token_from_render(rendered, tokenizer.get_added_vocab())
    if end_token is not None:
        ids.append(tokenizer.convert_tokens_to_ids(end_token))
    ids.append(tokenizer.eos_token_id)
    model_eos = model.generation_config.eos_token_id
    ids.extend(model_eos if isinstance(model_eos, list) else [model_eos])
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
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=not greedy,
        temperature=None if greedy else args.temperature,
        top_p=None if greedy else args.top_p,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    responses = []
    for row in output[:, inputs["input_ids"].shape[1] :].tolist():
        end = next((i for i, token in enumerate(row) if token in stop_ids), len(row))
        # Keep special tokens, so the <think> block stays visible in the response.
        responses.append(tokenizer.decode(row[:end], skip_special_tokens=False))
    return responses


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
    stop_ids = stop_token_ids(tokenizer, model)
    print(f"Stop tokens: {tokenizer.convert_ids_to_tokens(stop_ids)}")

    result = {
        "model": args.model,
        "chat_template": args.chat_template,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "prompts": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(entries), args.batch_size):
        batch = entries[start : start + args.batch_size]
        prompts = [entry["prompt"] for entry in batch]
        by_mode = {
            mode: generate_batch(model, tokenizer, prompts, mode, stop_ids, args) for mode in MODES
        }
        for i, entry in enumerate(batch):
            response = {mode: by_mode[mode][i] for mode in MODES}
            result["prompts"].append({**entry, "generated_response": response})

        # Written after every batch, so a stopped job keeps its finished entries.
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{len(result['prompts'])}/{len(entries)} prompts done")


if __name__ == "__main__":
    main()
