"""Tests for the method-specific dataset filters.

`_filter_sft_rows` drops rows that would contribute nothing to the SFT loss:
an empty `messages` list, or a conversation whose rendered `assistant_masks`
is all zero.  A row that survives to the trainer but produces no gradient
costs compute and quietly shrinks the effective dataset, so the filter is
worth pinning down.

`_filter_dpo_rows` drops preference pairs missing a `chosen` or a `rejected`
side.  It reads those two columns; a DPO dataset has no `messages` column at
all, so a filter that reaches for one raises `KeyError` on every run.

Which rows an all-zero mask catches depends on the template, so these tests
use two marker-bearing templates deliberately:

* `qwen3` puts only the assistant turns **after the last user query** in the
  loss.  It strips `<think>` from earlier turns, so those render as a bare
  answer and are excluded.  A conversation that ends on a user turn therefore
  has an all-zero mask under qwen3 and gets dropped.
* `olmo3-instruct-sft` puts **every** assistant turn in the loss, so the same
  conversation survives.

`tests/test_chat_template_masking.py` pins that span-level behaviour itself.
Here it is the input to the filter.

`max_length` splits the outcomes three ways, and the tests below pin each: a
span entirely inside the window is kept intact, a span entirely past it is
dropped, and a span that STRADDLES the window is kept but damaged — it trains
on an answer cut mid-sentence.  Only the third is invisible to a keep/drop
count, so it is the one that needs a warning rather than a filter.

No tokenizer is downloaded.  `_CharTokenizer` renders the real registered
templates through transformers' own generation tracker — the same code path
`return_assistant_tokens_mask=True` uses — and then treats one character as
one token.  Character offsets map one-to-one onto token offsets, so the masks
and the `max_length` cuts carry the same meaning as a real tokenizer's.
"""

from __future__ import annotations

import json
import logging

import pytest
from datasets import Dataset

from post_training.chat_templates.registry import get_chat_template
from post_training.methods.dpo import _filter_dpo_rows
from post_training.methods.sft import _classify_row, _filter_sft_rows

SFT_LOGGER = "post_training.methods.sft"

# Loss on the assistant turns after the last user query only.
MARKED_TEMPLATE = "qwen3"
# Loss on every assistant turn — the contrast case.
HISTORY_TEMPLATE = "olmo3-instruct-sft"
# No `{% generation %}` markers at all — every mask comes back all zero.
UNMARKED_TEMPLATE = "olmo3"

_COMPILED: dict[str, object] = {}


class _CharTokenizer:
    """Stand-in for a HF tokenizer, with one character per token.

    Holds the template *name* rather than the compiled Jinja object so that
    `datasets` can still fingerprint the `.map()` closure that captures it.
    """

    def __init__(self, template_name: str) -> None:
        self.template_name = template_name

    @property
    def chat_template(self) -> str:
        return get_chat_template(self.template_name)

    def _render(
        self, conversation: list[dict], tools: list[dict] | None = None
    ) -> tuple[str, list[tuple[int, int]]]:
        utils = pytest.importorskip("transformers.utils.chat_template_utils")
        if self.template_name not in _COMPILED:
            _COMPILED[self.template_name] = utils._compile_jinja_template(self.chat_template)
        return utils._render_with_assistant_indices(
            _COMPILED[self.template_name], conversation, tools, None, False
        )

    def render(self, conversation: list[dict], tools: list[dict] | None = None) -> str:
        rendered, _ = self._render(conversation, tools)
        return rendered

    def apply_chat_template(
        self,
        conversation: list[dict],
        *,
        tools: list[dict] | None = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict:
        rendered, indices = self._render(conversation, tools)

        input_ids = [ord(character) for character in rendered]
        assistant_masks = [0] * len(rendered)
        for start, end in indices:
            for position in range(start, end):
                assistant_masks[position] = 1

        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
            assistant_masks = assistant_masks[:max_length]

        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


@pytest.fixture
def tokenizer() -> _CharTokenizer:
    """A qwen3 tokenizer — the template our Qwen3 SFT config uses."""
    return _CharTokenizer(MARKED_TEMPLATE)


def _exchange(index: int = 0) -> list[dict]:
    return [
        {"role": "user", "content": f"question-{index}"},
        {"role": "assistant", "content": f"answer-{index}"},
    ]


def _sft_dataset(rows: list[list[dict]]) -> Dataset:
    return Dataset.from_dict({"messages": rows})


# ── SFT: which rows survive ────────────────────────────────────────────


def test_keeps_rows_with_an_assistant_turn(tokenizer) -> None:
    ds = _sft_dataset([_exchange(0), _exchange(1)])

    kept = _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)

    assert len(kept) == 2


def test_drops_rows_with_an_empty_messages_list(tokenizer) -> None:
    ds = _sft_dataset([_exchange(0), [], _exchange(1)])

    kept = _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)

    assert kept["messages"] == [_exchange(0), _exchange(1)]


def test_drops_a_conversation_with_no_assistant_turn(tokenizer) -> None:
    """A prompt-only row renders to an all-zero mask, so it trains on nothing."""
    ds = _sft_dataset([[{"role": "user", "content": "orphan question"}], _exchange(0)])

    kept = _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)

    assert kept["messages"] == [_exchange(0)]


def test_keeps_a_multi_turn_row_when_only_the_final_turn_is_in_the_loss(tokenizer) -> None:
    """qwen3 excludes the assistant turns at or before the last user query, so
    this row contributes one turn out of two.  One is enough — the filter asks
    whether *any* token enters the loss, not whether every turn does.
    """
    multi_turn = [*_exchange(0), *_exchange(1)]
    ds = _sft_dataset([multi_turn])

    kept = _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)

    assert kept["messages"] == [multi_turn]


def test_drops_a_conversation_ending_on_a_user_turn_under_qwen3(tokenizer) -> None:
    """Under qwen3 every assistant turn here sits at or before the last user
    query, so all of them render think-stripped and stay out of the loss.  The
    mask is all zero and the row trains on nothing.

    This is the case the filter exists for.  Such a row looks perfectly valid
    — it has an assistant turn and non-empty content — so nothing before the
    mask computation catches it.
    """
    trailing_user = [*_exchange(0), {"role": "user", "content": "follow-up"}]
    ds = _sft_dataset([trailing_user, _exchange(1)])

    kept = _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)

    assert kept["messages"] == [_exchange(1)]


def test_keeps_a_conversation_ending_on_a_user_turn_under_olmo() -> None:
    """Counterpoint to the qwen3 case above, pinned deliberately.  The OLMo
    templates keep every assistant turn in the loss, so the identical row
    survives.  Which rows the filter drops is a property of the template, not
    a house rule about trailing user turns.
    """
    trailing_user = [*_exchange(0), {"role": "user", "content": "follow-up"}]
    ds = _sft_dataset([trailing_user])

    kept = _filter_sft_rows(ds, num_proc=1, tokenizer=_CharTokenizer(HISTORY_TEMPLATE))

    assert kept["messages"] == [trailing_user]


def test_preserves_row_order_and_content(tokenizer) -> None:
    rows = [_exchange(0), [], _exchange(1), [], _exchange(2)]
    ds = _sft_dataset(rows)

    kept = _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)

    assert kept["messages"] == [_exchange(0), _exchange(1), _exchange(2)]


# ── SFT: truncation parity with the trainer ────────────────────────────


def test_drops_a_row_whose_assistant_turn_truncation_removes(tokenizer) -> None:
    """TRL cuts every row at `max_length` — plain truncation without packing,
    and the first-fragment-only rule of the default `bfd` packing strategy.
    A row whose assistant turn sits past that cut trains on nothing, so the
    filter must measure the same window.
    """
    long_prompt = [
        {"role": "user", "content": "x" * 500},
        {"role": "assistant", "content": "answer"},
    ]
    ds = _sft_dataset([long_prompt])

    assert len(_filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)) == 1

    with pytest.raises(ValueError, match="zero loss"):
        _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer, max_length=32)


def test_keeps_a_row_whose_assistant_turn_fits_inside_max_length(tokenizer) -> None:
    ds = _sft_dataset([_exchange(0)])
    width = len(tokenizer.render(_exchange(0)))

    kept = _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer, max_length=width)

    assert len(kept) == 1


def _span_bounds(tokenizer, conversation: list[dict]) -> tuple[int, int]:
    """(first, last) index of the supervised span. One char == one token here."""
    mask = tokenizer.apply_chat_template(
        conversation, return_dict=True, return_assistant_tokens_mask=True
    )["assistant_masks"]
    return mask.index(1), len(mask) - 1 - mask[::-1].index(1)


def test_keeps_a_row_whose_supervised_span_max_length_straddles(tokenizer, caplog) -> None:
    """The case a keep/drop count cannot show.

    The span starts inside the window and continues past it, so the row DOES
    produce gradient and is kept — while teaching an answer that stops
    mid-sentence with no end-of-turn token.  Under final-turn-only supervision
    the assistant content sits at the end of the row, so this is what a cap
    slightly below the row length produces: not a dropped row, a damaged one.
    """
    row = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "y" * 200}]
    start, end = _span_bounds(tokenizer, row)
    straddling = start + 10
    assert start < straddling < end

    with caplog.at_level(logging.INFO, logger=SFT_LOGGER):
        kept = _filter_sft_rows(
            _sft_dataset([row]), num_proc=1, tokenizer=tokenizer, max_length=straddling
        )

    assert len(kept) == 1
    cut = [record for record in caplog.records if "PART-WAY" in record.getMessage()]
    assert [record.levelno for record in cut] == [logging.WARNING]


def test_does_not_warn_when_every_supervised_span_fits(tokenizer, caplog) -> None:
    row = _exchange(0)
    _, end = _span_bounds(tokenizer, row)

    with caplog.at_level(logging.INFO, logger=SFT_LOGGER):
        _filter_sft_rows(_sft_dataset([row]), num_proc=1, tokenizer=tokenizer, max_length=end + 1)

    assert not [record for record in caplog.records if "PART-WAY" in record.getMessage()]


def test_does_not_warn_about_truncation_when_no_max_length_is_set(tokenizer, caplog) -> None:
    """With no cap there is nothing to cut, however long the row is."""
    row = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "y" * 5000}]

    with caplog.at_level(logging.INFO, logger=SFT_LOGGER):
        kept = _filter_sft_rows(_sft_dataset([row]), num_proc=1, tokenizer=tokenizer)

    assert len(kept) == 1
    assert not [record for record in caplog.records if "PART-WAY" in record.getMessage()]


def test_separates_the_two_reasons_a_row_was_dropped(tokenizer, caplog) -> None:
    """The two causes call for opposite fixes — change the data, or raise the
    cap — so a single drop count sends the reader to the wrong one half the time.
    """
    prompt_only = [{"role": "user", "content": "orphan"}]
    beyond_cap = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "a"}]

    with caplog.at_level(logging.INFO, logger=SFT_LOGGER):
        _filter_sft_rows(
            _sft_dataset([_exchange(0), prompt_only, beyond_cap]),
            num_proc=1,
            tokenizer=tokenizer,
            max_length=64,
        )

    causes = next(record for record in caplog.records if "by cause" in record.getMessage())
    assert "1 with no assistant tokens at all" in causes.getMessage()
    assert "1 with every assistant token beyond max_length=64" in causes.getMessage()


def _straddling_row(tokenizer) -> tuple[list[dict], int]:
    """A row plus a max_length that cuts its supervised span part-way.

    The prompt is long on purpose, so the cut lands deep enough into the row that
    an ordinary short row still fits inside the same cap and can serve as the
    control. With a short prompt the cap is ~50 tokens, every row exceeds it, and
    the filter empties the dataset instead of dropping one row.
    """
    row = [{"role": "user", "content": "x" * 300}, {"role": "assistant", "content": "y" * 200}]
    start, end = _span_bounds(tokenizer, row)
    cut = start + 50
    assert start < cut < end
    assert cut > len(tokenizer.render(_exchange(0))), "control row must fit under the cap"
    return row, cut


def test_a_straddling_row_is_kept_by_default(tokenizer) -> None:
    """Default is warn: the row stays, because the fix is normally to raise the
    cap rather than to throw the data away."""
    row, cut = _straddling_row(tokenizer)

    kept = _filter_sft_rows(_sft_dataset([row]), num_proc=1, tokenizer=tokenizer, max_length=cut)

    assert len(kept) == 1


def test_a_straddling_row_is_dropped_when_asked(tokenizer) -> None:
    """`drop` is for a sequence length fixed by memory, where shortening the
    data is the only lever left."""
    row, cut = _straddling_row(tokenizer)

    kept = _filter_sft_rows(
        _sft_dataset([row, _exchange(0)]),
        num_proc=1,
        tokenizer=tokenizer,
        max_length=cut,
        truncated_span_action="drop",
    )

    assert kept["messages"] == [_exchange(0)]


def test_dropping_still_warns(tokenizer, caplog) -> None:
    """Drop is not the quiet option. `data.datasets[].weight` multiplies the rows
    that survive filtering, so an uneven drop changes the realised mixture — that
    has to be visible in the log, not inferred from a row count.
    """
    row, cut = _straddling_row(tokenizer)

    with caplog.at_level(logging.INFO, logger=SFT_LOGGER):
        _filter_sft_rows(
            _sft_dataset([row, _exchange(0)]),
            num_proc=1,
            tokenizer=tokenizer,
            max_length=cut,
            truncated_span_action="drop",
        )

    warned = [record for record in caplog.records if "PART-WAY" in record.getMessage()]
    assert [record.levelno for record in warned] == [logging.WARNING]
    assert "DROPPED" in warned[0].getMessage()
    assert "mixture" in warned[0].getMessage()


def test_rejects_an_unknown_truncated_span_action(tokenizer) -> None:
    """Caught before the dataset is tokenized, not after — a typo must not cost
    a full pass over the pool before it surfaces.
    """
    with pytest.raises(ValueError, match="truncated_span_action"):
        _filter_sft_rows(
            _sft_dataset([_exchange(0)]),
            num_proc=1,
            tokenizer=tokenizer,
            max_length=4096,
            truncated_span_action="Drop",
        )


# ── SFT: the failure modes that must be loud ───────────────────────────


def test_raises_when_every_row_has_an_empty_messages_list(tokenizer) -> None:
    ds = _sft_dataset([[], []])

    with pytest.raises(ValueError, match="empty 'messages'"):
        _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)


def test_raises_when_the_template_has_no_generation_markers() -> None:
    """Without `{% generation %}` markers every mask is all zero.  The filter
    would otherwise return an empty dataset and the failure would surface far
    downstream as a confusing weight or resampling error.
    """
    ds = _sft_dataset([_exchange(0), _exchange(1)])

    with pytest.raises(ValueError, match="generation"):
        _filter_sft_rows(ds, num_proc=1, tokenizer=_CharTokenizer(UNMARKED_TEMPLATE))


def test_raises_when_every_row_ends_on_a_user_turn_under_qwen3(tokenizer) -> None:
    """A whole mix of trailing-user conversations is a plausible mistake — a
    prompt-completion dataset transformed into `messages` without pairing the
    completion back in.  Under qwen3 it yields zero usable rows, and the run
    must stop with that stated rather than fail later on an empty mix.
    """
    ds = _sft_dataset([[*_exchange(i), {"role": "user", "content": "next"}] for i in range(3)])

    with pytest.raises(ValueError, match="zero loss"):
        _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)


def test_total_drop_error_names_max_length_only_when_it_is_set(tokenizer) -> None:
    """Truncation and missing markers both empty the dataset.  The message
    must name the cause that applies, so the reader is not sent to the wrong
    one.
    """
    ds = _sft_dataset([[{"role": "user", "content": "no assistant turn"}]])

    with pytest.raises(ValueError) as without_length:
        _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)
    assert "max_length" not in str(without_length.value)

    with pytest.raises(ValueError) as with_length:
        _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer, max_length=4096)
    assert "max_length of 4096" in str(with_length.value)


def test_total_drop_error_blames_the_cap_when_every_row_has_an_assistant_turn(
    tokenizer,
) -> None:
    """Both causes empty the dataset, and they need opposite fixes.  When every
    row does have an assistant turn, the template is exonerated and saying
    otherwise sends the reader on a long detour through their jinja.
    """
    rows = [
        [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": f"a{i}"}]
        for i in range(3)
    ]

    with pytest.raises(ValueError) as excinfo:
        _filter_sft_rows(_sft_dataset(rows), num_proc=1, tokenizer=tokenizer, max_length=32)

    message = str(excinfo.value)
    assert "max_seq_length problem" in message
    assert "not a template problem" in message


def test_requires_the_tokenizer_as_a_keyword_argument() -> None:
    """`build_sft_trainer` binds the tokenizer with `functools.partial`.  A
    missing tokenizer must fail at the call, not as an `AttributeError` inside
    the worker pool — and not silently under `python -O`, which strips asserts.
    """
    ds = _sft_dataset([_exchange(0)])

    with pytest.raises(TypeError, match="tokenizer"):
        _filter_sft_rows(ds, num_proc=1)


# ── SFT: reporting ─────────────────────────────────────────────────────


def test_warns_when_the_drop_rate_passes_the_threshold(tokenizer, caplog) -> None:
    prompt_only = [{"role": "user", "content": "orphan"}]
    ds = _sft_dataset([_exchange(i) for i in range(9)] + [prompt_only])

    with caplog.at_level(logging.INFO, logger=SFT_LOGGER):
        _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)

    levels = {record.levelno for record in caplog.records if "all-zero" in record.message}
    assert levels == {logging.WARNING}


def test_logs_at_info_when_the_drop_rate_stays_under_the_threshold(tokenizer, caplog) -> None:
    prompt_only = [{"role": "user", "content": "orphan"}]
    ds = _sft_dataset([_exchange(i) for i in range(199)] + [prompt_only])

    with caplog.at_level(logging.INFO, logger=SFT_LOGGER):
        _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)

    levels = {record.levelno for record in caplog.records if "all-zero" in record.message}
    assert levels == {logging.INFO}


def test_reports_each_distinct_role_sequence_once(tokenizer, caplog) -> None:
    """The dropped-row report exists to show *why* rows went away.  It lists
    the trailing role sequence of every dropped row, de-duplicated, so a
    million identical drops still print one line.

    Under qwen3 the `user -> assistant -> user` line is the one that matters:
    it names the think-stripping exclusion, which is the non-obvious reason a
    complete-looking conversation contributes nothing.
    """
    prompt_only = [{"role": "user", "content": "orphan"}]
    trailing_user = [*_exchange(0), {"role": "user", "content": "follow-up"}]
    ds = _sft_dataset([_exchange(1), prompt_only, prompt_only, prompt_only, trailing_user])

    with caplog.at_level(logging.INFO, logger=SFT_LOGGER):
        _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer)

    report = next(record for record in caplog.records if "Role sequences" in record.getMessage())
    body = report.getMessage().split("\n", 1)[1]

    assert "4 dropped rows (2 unique)" in report.getMessage()
    assert sorted(body.splitlines()) == ["user", "user -> assistant -> user"]


# ── SFT: tool declarations ─────────────────────────────────────────────

TOOL_SCHEMA = [{"type": "function", "function": {"name": "lookup", "description": "d"}}]


def test_a_tools_declaration_counts_toward_max_length(tokenizer) -> None:
    """Why carrying the column and rendering it have to land together.

    A tool declaration emits the template's whole "# Tools" preamble ahead of the
    conversation, which pushes the assistant span later. A filter that rendered
    without tools would measure a shorter row than the one max_length is applied
    to, and would keep rows whose supervised span the trainer truncates away.
    """
    row = _exchange(0)
    bare = len(tokenizer.render(row))
    with_tools = len(tokenizer.render(row, TOOL_SCHEMA))
    assert with_tools > bare, "the tools preamble must lengthen the render"

    # A cap that fits the bare row but not the same row once tools are declared.
    cap = bare
    assert (
        len(_filter_sft_rows(_sft_dataset([row]), num_proc=1, tokenizer=tokenizer, max_length=cap))
        == 1
    )

    ds = Dataset.from_dict({"messages": [row], "tools": [json.dumps(TOOL_SCHEMA)]})
    with pytest.raises(ValueError, match="zero loss"):
        _filter_sft_rows(ds, num_proc=1, tokenizer=tokenizer, max_length=cap)


def test_tools_are_parsed_from_a_json_string(tokenizer) -> None:
    """TRL reads the column with
    `json.loads(tools) if isinstance(tools, str) else tools`, so the filter must
    too. Left as a string, the template would iterate over its characters and
    render something else entirely.
    """
    row = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    as_string = _classify_row(row, tokenizer, None, tools=json.dumps(TOOL_SCHEMA))
    as_list = _classify_row(row, tokenizer, None, tools=TOOL_SCHEMA)

    assert as_string == as_list


def test_a_row_without_tools_is_unaffected(tokenizer) -> None:
    """The column is null for most rows, and null must render exactly as before."""
    row = _exchange(0)
    assert tokenizer.render(row, None) == tokenizer.render(row)
    assert _classify_row(row, tokenizer, None, tools=None) == _classify_row(row, tokenizer, None)


# ── DPO ────────────────────────────────────────────────────────────────


def _dpo_dataset(pairs: list[tuple[list, list]]) -> Dataset:
    return Dataset.from_dict(
        {
            "chosen": [chosen for chosen, _ in pairs],
            "rejected": [rejected for _, rejected in pairs],
        }
    )


def test_dpo_keeps_complete_preference_pairs() -> None:
    ds = _dpo_dataset([(_exchange(0), _exchange(1)), (_exchange(2), _exchange(3))])

    assert len(_filter_dpo_rows(ds, num_proc=1)) == 2


def test_dpo_drops_a_pair_with_an_empty_chosen_side() -> None:
    ds = _dpo_dataset([([], _exchange(0)), (_exchange(1), _exchange(2))])

    kept = _filter_dpo_rows(ds, num_proc=1)

    assert kept["chosen"] == [_exchange(1)]


def test_dpo_drops_a_pair_with_an_empty_rejected_side() -> None:
    ds = _dpo_dataset([(_exchange(0), []), (_exchange(1), _exchange(2))])

    kept = _filter_dpo_rows(ds, num_proc=1)

    assert kept["rejected"] == [_exchange(2)]


def test_dpo_reads_chosen_and_rejected_rather_than_messages() -> None:
    """Regression guard.  A DPO dataset carries `chosen` / `rejected` and no
    `messages` column (see `configs/trl/dpo.yaml`, which uses `transform: null`
    on an already-paired dataset).  A filter that reads `messages` raises
    `KeyError` on every DPO run.
    """
    ds = _dpo_dataset([(_exchange(0), _exchange(1))])
    assert "messages" not in ds.column_names

    assert len(_filter_dpo_rows(ds, num_proc=1)) == 1
