"""Dataset-specific transformation registry.

Transforms convert arbitrary dataset rows into the standard conversational
format expected by TRL's SFTTrainer::

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Register new transforms with the ``@register_transform`` decorator and
reference them by name in the ``data.datasets[].transform`` field of the
YAML config.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def register_transform(name: str) -> Callable:
    """Decorator that registers a transform function under *name*.

    Example
    -------
    >>> @register_transform("my_transform")
    ... def my_transform(example: dict) -> dict:
    ...     return {"messages": [...]}
    """

    def _decorator(fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable:
        if name in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform '{name}' is already registered.")
        TRANSFORM_REGISTRY[name] = fn
        return fn

    return _decorator


def get_transform(name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Look up a registered transform by *name*.

    Raises
    ------
    KeyError
        If no transform with the given name exists.
    """
    if name not in TRANSFORM_REGISTRY:
        available = ", ".join(sorted(TRANSFORM_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Transform '{name}' not found. Available transforms: {available}"
        )
    return TRANSFORM_REGISTRY[name]


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


@register_transform("rename_to_messages")
def rename_to_messages(example: dict[str, Any]) -> dict[str, Any]:
    """Rename common column variants to the ``messages`` key.

    Looks for ``conversation``, ``conversations``, or ``dialog`` and maps
    them to ``messages``.
    """
    for key in ("conversation", "conversations", "dialog"):
        if key in example:
            return {"messages": example[key]}
    raise KeyError(
        f"Could not find a conversation column in keys: {list(example.keys())}"
    )


@register_transform("instruction_response_to_messages")
def instruction_response_to_messages(example: dict[str, Any]) -> dict[str, Any]:
    """Convert ``instruction`` / ``response`` (or ``input`` / ``output``) pairs
    into the standard conversational format.
    """
    user_content = example.get("instruction") or example.get("input", "")
    assistant_content = example.get("response") or example.get("output", "")
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# Transforms ported from old_generate.py
# ---------------------------------------------------------------------------


@register_transform("smoltalk2")
def smoltalk2(example: dict[str, Any]) -> dict[str, Any]:
    """Transform HuggingFaceTB/smoltalk2 examples for SFT.

    - Merges consecutive turns from the same role.
    - Excludes examples that contain any tools in
      ``chat_template_kwargs['python_tools']`` or ``['xml_tools']``.
    """
    chat_template_kwargs = example.get("chat_template_kwargs", {})
    if (
        chat_template_kwargs.get("python_tools")
        and len(chat_template_kwargs["python_tools"]) > 0
    ) or (
        chat_template_kwargs.get("xml_tools")
        and len(chat_template_kwargs["xml_tools"]) > 0
    ):
        return {"messages": []}

    messages = [example["messages"][0]]
    for message in example["messages"][1:]:
        if message["role"] == messages[-1]["role"]:
            messages[-1]["content"] += "\n" + message["content"]
        else:
            messages.append(message)

    return {"messages": messages}


@register_transform("smoltalk2_preference")
def smoltalk2_preference(example: dict[str, Any]) -> dict[str, Any]:
    """Transform HuggingFaceTB/smoltalk2 Preference split.

    The ``chosen`` column is renamed to ``messages`` before this transform
    is called (handled by the loader's rename_columns), then the same
    filtering/merging as the SFT variant is applied.
    """
    # The 'chosen' column should already be mapped to 'messages' by rename,
    # but handle both for robustness.
    msgs = example.get("messages") or example.get("chosen", [])
    if not msgs:
        return {"messages": []}

    example_copy = dict(example)
    example_copy["messages"] = msgs
    return smoltalk2(example_copy)


@register_transform("perfectblend")
def perfectblend(example: dict[str, Any]) -> dict[str, Any]:
    """Transform mlabonne/open-perfectblend examples for SFT.

    - Renames roles: ``"human"`` -> ``"user"``, ``"gpt"`` -> ``"assistant"``.
    - Extracts message content under the ``"value"`` key.
    - Merges consecutive turns from the same role.
    """
    conversations = example["conversations"]
    if not conversations or not isinstance(conversations, list):
        return {"messages": []}

    role_map = {"human": "user", "gpt": "assistant"}
    messages: list[dict[str, str]] = []
    for turn in conversations:
        role = turn["from"]
        content = turn["value"]
        current_role = role_map[role]
        current_content = content.strip()
        if messages and messages[-1]["role"] == current_role:
            messages[-1]["content"] += "\n" + current_content
        else:
            messages.append({"role": current_role, "content": current_content})

    return {"messages": messages}


@register_transform("orca_agentinstruct_1m")
def orca_agentinstruct_1m(example: dict[str, Any]) -> dict[str, Any]:
    """Transform microsoft/orca-agentinstruct-1M-v1 examples for SFT.

    Converts the ``messages`` field from a JSON string to a list of dicts.
    """
    return {"messages": json.loads(example["messages"])}


@register_transform("anthropic_hh_rlhf")
def anthropic_hh_rlhf(example: dict[str, Any]) -> dict[str, Any]:
    """Transform Anthropic/hh-rlhf text blobs into messages.

    Parses ``Human:`` and ``Assistant:`` markers and merges consecutive
    turns from the same role.
    """
    text = example["chosen"]
    pattern = r"(Human|Assistant):([\s\S]+?)(?=(?:Human|Assistant):|$)"
    matches = re.findall(pattern, text)

    messages: list[dict[str, str]] = []
    role_map = {"Human": "user", "Assistant": "assistant"}

    for role, content in matches:
        clean_role = role_map.get(role, role)
        clean_content = content.strip()

        if messages and messages[-1]["role"] == clean_role:
            messages[-1]["content"] += "\n" + clean_content
        else:
            messages.append({"role": clean_role, "content": clean_content})

    return {"messages": messages}


@register_transform("stanfordnlp_shp")
def stanfordnlp_shp(example: dict[str, Any]) -> dict[str, Any]:
    """Transform stanfordnlp/SHP examples for SFT.

    Selects the preferred response based on the ``labels`` field
    (1 = A is better, 0 = B is better).
    """
    if example["labels"] == 1:
        chosen_response = example["human_ref_A"]
    else:
        chosen_response = example["human_ref_B"]

    return {
        "messages": [
            {"role": "user", "content": example["history"]},
            {"role": "assistant", "content": chosen_response},
        ]
    }


@register_transform("berkeley_nest_nectar")
def berkeley_nest_nectar(example: dict[str, Any]) -> dict[str, Any]:
    """Transform berkeley-nest/Nectar examples for SFT.

    1. Cleans the ``prompt`` field to remove ``Human:``/``Assistant:`` artifacts.
    2. Selects the answer with ``rank == 1`` from the ``answers`` list.
    """
    raw_prompt = example["prompt"]

    clean_prompt = re.sub(r"^\s*Human:\s*", "", raw_prompt, flags=re.IGNORECASE)
    clean_prompt = re.sub(r"\s*Assistant:\s*$", "", clean_prompt, flags=re.IGNORECASE)

    best_answer = ""
    if example["answers"]:
        best_answer = example["answers"][0]["answer"]

        for ans_obj in example["answers"]:
            if ans_obj["rank"] == 1:
                best_answer = ans_obj["answer"]
                break

    return {
        "messages": [
            {"role": "user", "content": clean_prompt},
            {"role": "assistant", "content": best_answer},
        ]
    }


@register_transform("arena_preference")
def arena_preference(example: dict[str, Any]) -> dict[str, Any]:
    """Transform lmarena-ai/arena-human-preference-55k examples.

    Selects the winning response (ties default to model A).
    """
    if example["winner_model_a"] or example["winner_tie"]:
        chosen_content = example["response_a"]
    else:
        chosen_content = example["response_b"]

    prompt_content = example["prompt"]
    if isinstance(prompt_content, list):
        prompt_content = "\n".join(prompt_content)

    return {
        "messages": [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": chosen_content},
        ]
    }


@register_transform("comparia_votes")
def comparia_votes(example: dict[str, Any]) -> dict[str, Any]:
    """Transform ministere-culture/comparia-votes examples.

    1. Identifies the winner (model A or B).
    2. Checks quality flags (incorrect/useful).
    3. Prepends the system prompt if available.
    4. Merges consecutive turns from the same role.
    """
    if example["both_equal"]:
        chosen_model = example["model_a_name"]
    elif example["chosen_model_name"] is not None:
        chosen_model = example["chosen_model_name"]
    else:
        return {"messages": []}

    model_a = example["model_a_name"]
    is_a_winner = chosen_model == model_a

    model = "a" if is_a_winner else "b"

    if example.get(f"conv_incorrect_{model}") is True:
        return {"messages": []}
    if example.get(f"conv_useful_{model}") is False:
        return {"messages": []}

    selected_conversation = example[f"conversation_{model}"]
    selected_system_prompt = example[f"system_prompt_{model}"]

    messages = list(selected_conversation)

    if selected_system_prompt and isinstance(selected_system_prompt, str):
        cleaned_prompt = selected_system_prompt.strip()
        if cleaned_prompt:
            messages.insert(0, {"role": "system", "content": cleaned_prompt})

    # Merge consecutive turns from the same role.
    merged_messages = [messages[0]]
    for message in messages[1:]:
        if message["role"] == merged_messages[-1]["role"]:
            merged_messages[-1]["content"] += "\n" + message["content"]
        else:
            merged_messages.append(message)

    # Keep only content and role keys.
    return {
        "messages": [
            {"content": m["content"], "role": m["role"]} for m in merged_messages
        ]
    }


@register_transform("ultrafeedback")
def ultrafeedback(example: dict[str, Any]) -> dict[str, Any]:
    """Transform argilla/ultrafeedback-binarized-preferences-cleaned.

    The ``chosen`` column already contains the preferred conversation in
    the standard messages format.
    """
    return {"messages": example["chosen"]}


@register_transform("aegis_safety")
def aegis_safety(example: dict[str, Any]) -> dict[str, Any]:
    """Transform nvidia/Aegis-AI-Content-Safety-Dataset-2.0.

    Keeps only safe, non-redacted responses.
    """
    if example["prompt"] == "REDACTED":
        return {"messages": []}

    if example["response_label"] == "unsafe":
        return {"messages": []}

    if example["response_label"] == "safe":
        return {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
        }

    return {"messages": []}


@register_transform("helpsteer2")
def helpsteer2(example: dict[str, Any]) -> dict[str, Any]:
    """Transform nvidia/HelpSteer2.

    Picks the response with non-negative preference strength.
    """
    if example["preference_strength"] >= 0:
        chosen_response = example["response_1"]
    else:
        chosen_response = example["response_2"]

    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": chosen_response},
        ]
    }


@register_transform("intel_orca_dpo")
def intel_orca_dpo(example: dict[str, Any]) -> dict[str, Any]:
    """Transform argilla/distilabel-intel-orca-dpo-pairs.

    Uses ``input`` as user prompt and ``chosen`` as assistant response.
    Includes ``system`` prompt if present.
    """
    messages: list[dict[str, str]] = []

    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})

    messages.append({"role": "user", "content": example["input"]})
    messages.append({"role": "assistant", "content": example["chosen"]})

    return {"messages": messages}


@register_transform("human_like_dpo")
def human_like_dpo(example: dict[str, Any]) -> dict[str, Any]:
    """Transform HumanLLMs/Human-Like-DPO-Dataset.

    Uses ``prompt`` and ``chosen`` columns.
    """
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]},
        ]
    }


@register_transform("capybara_dpo")
def capybara_dpo(example: dict[str, Any]) -> dict[str, Any]:
    """Transform argilla/distilabel-capybara-dpo-7k-binarized.

    The ``chosen`` column already contains the conversation in messages format.
    """
    return {"messages": example["chosen"]}


@register_transform("mt_bench_judgments")
def mt_bench_judgments(example: dict[str, Any]) -> dict[str, Any]:
    """Transform lmsys/mt_bench_human_judgments.

    Selects ``conversation_b`` when model B wins, otherwise returns empty.
    """
    winner = example["winner"]

    if winner == "model_b":
        return {"messages": example["conversation_b"]}
    else:
        return {"messages": []}


@register_transform("math_preference")
def math_preference(example: dict[str, Any]) -> dict[str, Any]:
    """Transform argilla/distilabel-math-preference-dpo.

    Uses ``instruction`` and ``chosen_response``.
    """
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["chosen_response"]},
        ]
    }


@register_transform("truthy_dpo")
def truthy_dpo(example: dict[str, Any]) -> dict[str, Any]:
    """Transform jondurbin/truthy-dpo-v0.1.

    Includes ``system`` prompt if available.
    """
    messages: list[dict[str, str]] = []

    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})

    messages.append({"role": "user", "content": example["prompt"]})
    messages.append({"role": "assistant", "content": example["chosen"]})

    return {"messages": messages}


@register_transform("lmsys_chat")
def lmsys_chat(example: dict[str, Any]) -> dict[str, Any]:
    """Transform lmsys/lmsys-chat-1m.

    The ``conversation`` column contains the messages list directly.
    """
    return {"messages": example["conversation"]}
