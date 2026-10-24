"""Dataset-specific transformation registry.

Transforms convert arbitrary dataset rows into the standard conversational
format expected by TRL's SFTTrainer::

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Register new transforms with the ``@register_transform`` decorator and
reference them by name in the ``data.datasets[].transform`` field of the
YAML config.
"""

from __future__ import annotations

import ast
import io
import json
import keyword
import re
import tokenize
import warnings
from collections.abc import Callable, Iterator
from typing import Any

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
        raise KeyError(f"Transform '{name}' not found. Available transforms: {available}")
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
    raise KeyError(f"Could not find a conversation column in keys: {list(example.keys())}")


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
        chat_template_kwargs.get("python_tools") and len(chat_template_kwargs["python_tools"]) > 0
    ) or (chat_template_kwargs.get("xml_tools") and len(chat_template_kwargs["xml_tools"]) > 0):
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
    return {"messages": [{"content": m["content"], "role": m["role"]} for m in merged_messages]}


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

    Keeps only unsafe, non-redacted responses.
    """
    if example["prompt"] == "REDACTED":
        return {"messages": []}

    if example["response_label"] != "unsafe":
        return {"messages": []}

    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


@register_transform("helpsteer2")
def helpsteer2(example: dict[str, Any]) -> dict[str, Any]:
    """Transform nvidia/HelpSteer2.

    Positive preference strength chooses response 2; otherwise response 1.
    """
    if example["preference_strength"] > 0:
        chosen_response = example["response_2"]
    else:
        chosen_response = example["response_1"]

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


@register_transform("helpsteer3_preference")
def helpsteer3_preference(example: dict[str, Any]) -> dict[str, Any]:
    """Transform nvidia/HelpSteer3 Preference split.

    Positive overall preference chooses response 2; otherwise response 1.
    """
    if example["overall_preference"] > 0:
        chosen_response = example["response2"]
    else:
        chosen_response = example["response1"]

    return {
        "messages": [
            *example["context"],
            {"role": "assistant", "content": chosen_response},
        ]
    }


@register_transform("helpsteer3_feedback")
def helpsteer3_feedback(example: dict[str, Any]) -> dict[str, Any]:
    """Transform nvidia/HelpSteer3 Feedback split for SFT.

    Picks the response with the higher averaged rubric score.
    """
    grades = {
        "not": 0,
        "slightly": 1,
        "partially": 2,
        "mostly": 3,
        "perfectly": 4,
    }

    def score(feedbacks: list[str]) -> float:
        scores = [
            next(grades[grade] for grade in grades if grade in feedback.split(".")[0])
            for feedback in feedbacks
        ]
        return sum(scores) / len(scores)

    response1_score = score(example["feedback1"])
    response2_score = score(example["feedback2"])
    chosen_response = (
        example["response1"] if response1_score >= response2_score else example["response2"]
    )

    return {
        "messages": [
            *example["context"],
            {"role": "assistant", "content": chosen_response},
        ]
    }


@register_transform("helpsteer3_edit")
def helpsteer3_edit(example: dict[str, Any]) -> dict[str, Any]:
    """Transform nvidia/HelpSteer3 Edit split for SFT."""
    return {
        "messages": [
            *example["context"],
            {"role": "assistant", "content": example["edited_response"]},
        ]
    }


@register_transform("helpsteer3_edit_quality")
def helpsteer3_edit_quality(example: dict[str, Any]) -> dict[str, Any]:
    """Transform nvidia/HelpSteer3 Edit Quality split for SFT."""
    return {
        "messages": [
            *example["context"],
            {"role": "assistant", "content": example["good_edited_response"]},
        ]
    }


@register_transform("helpsteer3_principle")
def helpsteer3_principle(example: dict[str, Any]) -> dict[str, Any]:
    """Transform nvidia/HelpSteer3 Principle split for SFT.

    Keeps only examples where the principle is fulfilled.
    """
    if example["fulfilment"] == "No":
        return {"messages": []}

    return {
        "messages": [
            *example["context"],
            {"role": "assistant", "content": example["response"]},
        ]
    }


# ---------------------------------------------------------------------------
# Dolci tool use
# ---------------------------------------------------------------------------

# Dolci tool rows open with this system prompt. It describes the OLMo 3
# <functions> / <function_calls> format, which the transform replaces.
_DOLCI_TOOL_PROMPT = (
    "You are a helpful function-calling AI assistant. You are provided with function signatures "
    "within <functions></functions> XML tags. You may call one or more functions to assist with "
    "the user query. Output any function calls within <function_calls></function_calls> XML tags. "
    "Don't make assumptions about what values to plug into functions."
)
_DOLCI_PYTHONIC_HINT = " Format calls like: tool_name(query=...)"

_JSON_NAMES = {"true": True, "false": False, "null": None}
_CALL_RE = re.compile(r"\s*([^()]+?)\((.*)\)\s*", re.DOTALL)
# Python keywords used as argument names (``from="EUR"``) are a syntax error;
# they are renamed before parsing and restored afterwards.
_KEYWORD_PREFIX = "_kw_"
# Some blocks lost a "\" before a call name: "\reverse(...)" was decoded as a
# carriage return + "everse(...)". The line break and the letter are restored.
_ESCAPE_LETTERS = {"\r": "r", "\t": "t", "\x0c": "f", "\x08": "b", "\x0b": "v", "\x07": "a"}
_LOST_ESCAPE_RE = re.compile(r"\)([\r\t\x0c\x08\x0b\x07])(?=[\w.]+\()")
# Nodes allowed in an argument value that is an expression over literals,
# e.g. ``12*60``, ``[0.03]*15`` or ``[c for c in "noon"]``. A bare name that is
# not bound by a comprehension is a placeholder and fails with NameError.
_SAFE_NODES = (
    ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.BinOp, ast.UnaryOp,
    ast.operator, ast.unaryop, ast.ListComp, ast.comprehension, ast.Name, ast.Call, ast.Subscript,
    ast.Slice, ast.Load, ast.Store, ast.Compare, ast.cmpop,
)  # fmt: skip
_SAFE_GLOBALS = {"__builtins__": {}, "range": range, "list": list}


class _DolciCallError(ValueError):
    """A Dolci call line cannot be converted to a JSON tool call."""


class _JsonNames(ast.NodeTransformer):
    """Replace JSON ``true`` / ``false`` / ``null`` names with Python constants."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in _JSON_NAMES:
            # compile() needs a source location on every node.
            return ast.copy_location(ast.Constant(_JSON_NAMES[node.id]), node)
        return node


def _dolci_value(node: ast.expr) -> Any:
    """Evaluate one argument value: a literal, or an expression over literals only."""
    node = _JsonNames().visit(node)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):  # TypeError: an unhashable key, e.g. {[]: 1}
        pass
    if not all(
        isinstance(n, _SAFE_NODES)
        and not isinstance(n, ast.Pow)
        and not (isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store))
        for n in ast.walk(node)
    ):
        raise _DolciCallError(f"unsupported value: {ast.unparse(node)}")
    try:
        return eval(compile(ast.Expression(node), "<dolci>", "eval"), _SAFE_GLOBALS)  # noqa: S307
    except Exception as exc:
        raise _DolciCallError(f"cannot evaluate value: {ast.unparse(node)}") from exc


def _tokens(text: str) -> list[tuple[tokenize.TokenInfo, int, int]] | None:
    """Python tokens of ``text`` with their start and end offsets, or None if it does not tokenize.

    The tokenizer knows strings and comments, so code built on it never
    mistakes a ``(``, ``#`` or ``from=`` inside a quoted value for syntax. Names
    with spaces (``Rates API``) still tokenize, unlike with ``ast.parse``.
    """
    if "\r" in text:
        return None  # the tokenizer counts "\r" as a line break; the offsets below do not
    # The last entry is for the NEWLINE / ENDMARKER tokens after the last line.
    line_starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"] + [len(text)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return None
    if any(tok.type == tokenize.ERRORTOKEN for tok in tokens):
        return None
    return [
        (
            tok,
            line_starts[tok.start[0] - 1] + tok.start[1],
            line_starts[tok.end[0] - 1] + tok.end[1],
        )
        for tok in tokens
    ]


def _rename_keyword_args(arg_text: str) -> str:
    """Prefix Python keywords used as argument names (``from="EUR"``), which are a syntax error."""
    tokens = _tokens(arg_text)
    if tokens is None:
        return arg_text  # the parse fails next and reports the error
    for (tok, start, _), (nxt, _, _) in reversed(list(zip(tokens, tokens[1:]))):
        if (
            tok.type == tokenize.NAME
            and keyword.iskeyword(tok.string)
            and tok.string not in ("True", "False", "None")
            and nxt.string == "="
        ):
            arg_text = f"{arg_text[:start]}{_KEYWORD_PREFIX}{arg_text[start:]}"
    return arg_text


def _dolci_call(text: str) -> dict[str, Any]:
    """Convert one Python-style call, ``name(k=v, ...)``, to an OpenAI-style tool call.

    The name may contain spaces, ``-``, ``.`` or ``/``, so it is split off with a
    regex. ``arguments`` is a JSON string, serialized like the HF ``tojson``
    filter and the vLLM ``hermes`` parser do, so a call the model writes and
    the client sends back renders to the same text as in training.
    """
    match = _CALL_RE.fullmatch(text)
    if not match:
        raise _DolciCallError(f"not a call: {text!r}")
    name = match[1].strip()
    if '"' in name or "\\" in name:
        raise _DolciCallError(f"name breaks the rendered JSON: {name!r}")
    arg_text = _rename_keyword_args(match[2])
    try:
        # Dolci strings hold regex escapes such as '\.'; they parse fine but warn.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            call = ast.parse(f"_f({arg_text})", mode="eval").body
    except SyntaxError as exc:
        raise _DolciCallError(f"syntax error: {text!r}") from exc
    if not isinstance(call, ast.Call) or call.args or any(kw.arg is None for kw in call.keywords):
        raise _DolciCallError(f"not a keyword-only call: {text!r}")

    args = {}
    for kw in call.keywords:
        unprefixed = kw.arg.removeprefix(_KEYWORD_PREFIX)
        args[unprefixed if keyword.iskeyword(unprefixed) else kw.arg] = _dolci_value(kw.value)
    try:
        arguments = json.dumps(args, ensure_ascii=False)
        round_trip_ok = json.loads(arguments) == args
    except (TypeError, ValueError):
        round_trip_ok = False
    if not round_trip_ok:
        # Tuples, sets, non-string keys or NaN would come back from the parser changed.
        raise _DolciCallError(f"arguments do not survive a JSON round trip: {text!r}")
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def _dolci_call_texts(block: str) -> Iterator[str]:
    """Split a Dolci ``function_calls`` block into one text per call.

    Skips ``null`` (Dolci's marker for "no call"), blank lines and ``#``
    comments, splits on a literal ``\\n`` between calls, and joins a call that
    spans several lines.

    The split follows Python's tokenizer: a call ends at a NEWLINE token (or a
    ``;``), which the tokenizer only emits outside brackets and strings. Thus
    ``(``, ``)`` and ``#`` inside strings are safe, and names with spaces
    (``Get Rounds(id=1)``) work. The literal ``\\n`` and ``\\r`` line breaks are
    only replaced if the block does not tokenize as it is, so a ``)\\n`` inside
    a string is kept.

    A block that does not tokenize at all, e.g. a name with a quote
    (``Get Player's Media(id=1)``), falls back to a line split that counts
    parentheses. A wrong join there gives a syntax error, so the row is
    dropped, not changed.
    """
    block = _LOST_ESCAPE_RE.sub(lambda m: f")\n{_ESCAPE_LETTERS[m[1]]}", block)
    fixed = re.sub(r"\)\\n(?=\S)", ")\n", block.replace("\r\n", "\n").replace("\r", "\n"))
    for text in (block, fixed):
        tokens = _tokens(text)
        if tokens is None:
            continue
        call: list[tuple[int, int]] = []
        for tok, start, end in tokens:
            if tok.type in (tokenize.NEWLINE, tokenize.ENDMARKER) or tok.string == ";":
                if call and text[call[0][0] : call[-1][1]] != "null":
                    yield text[call[0][0] : call[-1][1]]
                call = []
            elif tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                call.append((start, end))
        return

    pending = ""
    for line in fixed.splitlines():
        if line.strip() in ("", "null") or line.lstrip().startswith("#"):
            continue
        pending = f"{pending}\n{line}" if pending else line
        if pending.count("(") > pending.count(")"):
            continue
        yield pending
        pending = ""
    if pending:
        yield pending


@register_transform("dolci_tool_use")
def dolci_tool_use(example: dict[str, Any]) -> dict[str, Any]:
    """Transform allenai/Dolci-Instruct-SFT tool rows to the standard tool format.

    Dolci stores tools in the OLMo 3 layout: the tool list as a JSON string in
    ``messages[0]["functions"]``, each call as a Python-style line in an
    assistant ``function_calls`` field, and each result as role
    ``environment``. This transform produces what chat templates and the vLLM
    ``hermes`` parser expect:

    - ``messages[0]["functions"]`` -> the ``tools`` column (JSON string)
    - assistant ``function_calls`` -> ``tool_calls`` with JSON ``arguments``
    - role ``environment`` -> role ``tool``

    It also removes the Dolci system prompt text about ``<functions>`` tags
    (keeping any task instructions after it) and drops assistant turns with
    no content and no calls. Rows without a tool list pass through unchanged.

    The row is dropped (empty ``messages``) if any call cannot be converted:
    placeholders such as ``loan_amount=X``, calls nested in calls, positional
    arguments, or values that change in a JSON round trip.
    """
    messages = example["messages"]
    if not messages or not messages[0].get("functions"):
        return {
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "tools": None,
        }

    tool_names = {t.get("function", {}).get("name") for t in json.loads(messages[0]["functions"])}
    out: list[dict[str, Any]] = []
    system = (
        messages[0]["content"]
        .removeprefix(_DOLCI_TOOL_PROMPT)
        .replace(_DOLCI_PYTHONIC_HINT, "")
        .strip()
    )
    if system:
        out.append({"role": "system", "content": system})

    for message in messages[1:]:
        role, content = message["role"], message["content"] or ""
        if role == "environment":
            out.append({"role": "tool", "content": content})
        elif role == "assistant":
            try:
                tool_calls = [
                    _dolci_call(t) for t in _dolci_call_texts(message.get("function_calls") or "")
                ]
            except _DolciCallError:
                # Every row must return the same keys, or the datasets writer fails.
                return {"messages": [], "tools": None}
            for call in tool_calls:
                # A "\" before a name that starts with "n" became a plain line
                # break (see _LOST_ESCAPE_RE), so only the tool list can show it.
                name = call["function"]["name"]
                if name not in tool_names and f"n{name}" in tool_names:
                    call["function"]["name"] = f"n{name}"
            if tool_calls:
                out.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            elif content.strip():
                out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": role, "content": content})

    return {"messages": out, "tools": messages[0]["functions"]}
