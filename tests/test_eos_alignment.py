"""Tests for aligning the stop token with what the chat template teaches.

SFT trains the model to emit whatever the template writes at the end of a turn.
Under `qwen3` that is `<|im_end|>`; the model's own eos (`<|endoftext|>` on OLMo,
`<eos>` on Prelude) appears nowhere in the data. `generate()` stops on
`model.generation_config.eos_token_id`, NOT on `tokenizer.eos_token`, and that
value comes from the checkpoint — so without alignment the model emits the token
it was trained to emit and nothing is listening, running to `max_new_tokens` on
every prompt.

`test_generate_stops_...` is the one that matters: it drives a real
`model.generate()` on a tiny randomly-initialised model, forcing the terminator
at every step so that stopping is the only variable. No download, CPU, offline.
"""

from __future__ import annotations

import pytest

from post_training.chat_templates.registry import get_chat_template, terminator_from_render
from post_training.methods.common import align_generation_eos

# OLMo's ids, used throughout so the numbers mean something.
TERMINATOR_ID = 100265  # <|im_end|>
NATIVE_EOS_ID = 100257  # <|endoftext|>
ADDED = {"<|im_start|>": 100264, "<|im_end|>": TERMINATOR_ID, "<|endoftext|>": NATIVE_EOS_ID}


class _Tok:
    def __init__(self, eos_token="<|im_end|>", eos_token_id=TERMINATOR_ID):
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id


class _GenConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _Model:
    def __init__(self, eos_token_id):
        self.generation_config = _GenConfig(eos_token_id)


class _Trainer:
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.processing_class = tokenizer


# ── deriving the terminator from what the template rendered ────────────


def test_reads_the_terminator_off_the_render():
    assert terminator_from_render("<|im_start|>a<|im_end|>", ADDED) == "<|im_end|>"


def test_a_trailing_newline_does_not_hide_it():
    """qwen3 emits '<|im_end|>\\n' after a turn."""
    assert terminator_from_render("...<|im_end|>\n", ADDED) == "<|im_end|>"


def test_returns_none_when_the_render_ends_on_ordinary_text():
    """The conservative answer: the caller then changes nothing."""
    assert terminator_from_render("...just an answer", ADDED) is None
    assert terminator_from_render("", ADDED) is None


def test_longest_match_wins():
    added = {"<|im_end|>": 1, "end|>": 2}
    assert terminator_from_render("x<|im_end|>", added) == "<|im_end|>"


@pytest.mark.parametrize("name", ["qwen3", "chatml"])
def test_chatml_style_templates_terminate_on_im_end(name):
    """Rendered with the private helper here only because a test may; production
    renders through the tokenizer's public `apply_chat_template`."""
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    compiled = utils._compile_jinja_template(get_chat_template(name))
    rendered, _ = utils._render_with_assistant_indices(
        compiled,
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        None,
        None,
        False,
    )
    assert terminator_from_render(rendered, ADDED) == "<|im_end|>"


@pytest.mark.parametrize("name", ["olmo3", "olmo3-instruct-sft", "olmo3-think-sft", "tulu3"])
def test_olmo_style_templates_terminate_on_the_native_eos(name):
    """These end a final turn on `eos_token`, so alignment must be a NO-OP for
    them. Pinned because a change here would silently alter OLMo runs.
    """
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    compiled = utils._compile_jinja_template(get_chat_template(name))
    rendered, _ = utils._render_with_assistant_indices(
        compiled,
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        None,
        None,
        False,
        eos_token="<|endoftext|>",
    )
    assert terminator_from_render(rendered, ADDED) == "<|endoftext|>"


# ── writing it into the checkpoint ────────────────────────────────────


def test_terminator_goes_first_and_the_native_eos_is_kept():
    """The native eos stays as a secondary stop: it was the pretraining
    terminator, so the prior to emit it decays under SFT without vanishing."""
    trainer = _Trainer(_Model(NATIVE_EOS_ID), _Tok())

    align_generation_eos(trainer)

    assert trainer.model.generation_config.eos_token_id == [TERMINATOR_ID, NATIVE_EOS_ID]


def test_no_op_when_the_template_already_ends_on_the_models_eos():
    """The olmo3-* case. Left as a scalar, not rewritten to a one-element list,
    so an OLMo run is provably untouched."""
    trainer = _Trainer(_Model(NATIVE_EOS_ID), _Tok("<|endoftext|>", NATIVE_EOS_ID))

    align_generation_eos(trainer)

    assert trainer.model.generation_config.eos_token_id == NATIVE_EOS_ID


def test_an_existing_list_is_preserved_without_duplicating():
    trainer = _Trainer(_Model([NATIVE_EOS_ID, 999]), _Tok())

    align_generation_eos(trainer)

    assert trainer.model.generation_config.eos_token_id == [TERMINATOR_ID, NATIVE_EOS_ID, 999]


def test_already_aligned_is_left_alone():
    trainer = _Trainer(_Model([TERMINATOR_ID]), _Tok())

    align_generation_eos(trainer)

    assert trainer.model.generation_config.eos_token_id == [TERMINATOR_ID]


def test_a_none_eos_on_the_model_does_not_produce_a_none_stop_id():
    trainer = _Trainer(_Model(None), _Tok())

    align_generation_eos(trainer)

    assert trainer.model.generation_config.eos_token_id == [TERMINATOR_ID]


@pytest.mark.parametrize(
    "trainer", [_Trainer(None, _Tok()), _Trainer(_Model(1), None), _Trainer(None, None)]
)
def test_missing_pieces_are_tolerated(trainer):
    align_generation_eos(trainer)  # must not raise


# ── the proof: a real generate() call ─────────────────────────────────


def test_generate_stops_on_the_template_terminator_after_alignment():
    """End-to-end on a real model, because everything above only asserts that a
    field was set — this asserts that `generate()` obeys it.

    A tiny randomly-initialised model is forced to emit the terminator at every
    step, so stopping is the only variable. Without alignment it runs the full
    `max_new_tokens`; with it, it stops after one token.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.AutoConfig.for_model(
        "llama",
        vocab_size=100300,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = transformers.AutoModelForCausalLM.from_config(config).eval()

    class ForceTerminator(transformers.LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:] = -1e9
            scores[:, TERMINATOR_ID] = 0.0
            return scores

    prompt = torch.tensor([[1, 2, 3]])

    def generated_tokens() -> int:
        model.generation_config.pad_token_id = NATIVE_EOS_ID
        out = model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=20,
            do_sample=False,
            logits_processor=transformers.LogitsProcessorList([ForceTerminator()]),
        )
        return out.shape[1] - prompt.shape[1]

    # before: the model's own eos is the only stop id, and it is never emitted
    model.generation_config.eos_token_id = NATIVE_EOS_ID
    assert generated_tokens() == 20, "expected it to run to max_new_tokens unaligned"

    align_generation_eos(_Trainer(model, _Tok()))

    assert model.generation_config.eos_token_id == [TERMINATOR_ID, NATIVE_EOS_ID]
    assert generated_tokens() == 1, "expected generation to stop on the terminator"
