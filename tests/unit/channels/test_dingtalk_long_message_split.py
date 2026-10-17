# -*- coding: utf-8 -*-
"""
DingTalk long-message regression tests.

DingTalk caps a markdown message body, so the channel used to guard against
oversized replies by downgrading the *whole* message to ``msgtype: text``.
That silently destroyed every Markdown construct in long replies -- most
visibly GFM pipe tables, which degraded into rows of literal ``| a | b |``.

Every other channel that has to deal with a length limit already ships the body
through :func:`qwenpaw.app.channels.utils.split_text`, which splits at newline
boundaries and keeps code fences and GFM tables renderable across chunks. The
DingTalk channel never imported it.

These tests pin the chunking policy itself, so they assert on the payloads the
channel hands to its transport rather than on HTTP plumbing.

Run:
    pytest tests/unit/channels/test_dingtalk_long_message_split.py -v
"""

# pylint: disable=redefined-outer-name,protected-access
from __future__ import annotations

import json
from typing import Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from qwenpaw.app.channels.renderer import ChannelDisplayConfig

# The pre-existing conservative budget for a single DingTalk markdown body.
# Kept as a literal here so the test fails loudly if production changes it
# without updating the expectation.
MAX_LEN = 3500

# DingTalk rejects markdown bodies over this; plain text has a higher limit.
HARD_MAX_LEN = 5000


@pytest.fixture
def mock_process_handler() -> AsyncMock:
    """Mock process handler that yields simple events."""

    async def mock_process(*_args, **_kwargs):
        mock_event = MagicMock()
        mock_event.object = "message"
        mock_event.status = "completed"
        mock_event.type = "text"
        yield mock_event

    return AsyncMock(side_effect=mock_process)


@pytest.fixture
def dingtalk_channel(
    mock_process_handler,
    tmp_path,
) -> Generator:
    """A DingTalkChannel instance, mirroring the other dingtalk tests."""
    from qwenpaw.app.channels.dingtalk.channel import DingTalkChannel

    media_dir = tmp_path / ".copaw" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    channel = DingTalkChannel(
        process=mock_process_handler,
        enabled=True,
        client_id="test_client_id",
        client_secret="test_client_secret",
        bot_prefix="[TestBot] ",
        media_dir=str(media_dir),
        display_config=ChannelDisplayConfig(
            show_tool_calls=False,
            show_tool_results=False,
        ),
    )
    yield channel


def _make_big_table(rows: int) -> str:
    """Build a GFM pipe table whose rendered body exceeds ``MAX_LEN``."""
    lines = [
        "| metric | p50 | p95 | p99 |",
        "| --- | --- | --- | --- |",
    ]
    for i in range(rows):
        lines.append(
            f"| request_{i:04d} | {i}.{i}ms | {i * 2}.{i}ms | {i * 3}.{i}ms |",
        )
    return "\n".join(lines)


def _big_table() -> str:
    """A table long enough to trip the limit, with room to spare."""
    table = _make_big_table(120)
    assert (
        len(table) > MAX_LEN
    ), "fixture must exceed the single-message budget"
    return table


@pytest.mark.asyncio
class TestDingTalkLongMessageSplit:
    """Long replies must stay Markdown, split into renderable chunks."""

    async def test_long_table_stays_markdown_via_session_webhook(
        self,
        dingtalk_channel,
    ):
        """The reported bug: a long table must not be downgraded to text."""
        sent: List[Dict] = []

        async def _capture(_session_webhook, payload):
            sent.append(payload)
            return True

        dingtalk_channel._send_payload_via_session_webhook = _capture

        ok = await dingtalk_channel._send_via_session_webhook(
            session_webhook="https://oapi.dingtalk.com/robot/send?session=abc",
            body=_big_table(),
            bot_prefix="[Bot]",
        )

        assert ok is True
        # Baseline sends exactly one payload with msgtype "text" -> this fails.
        assert (
            len(sent) > 1
        ), "an oversized reply must be split, not sent whole"
        assert {p["msgtype"] for p in sent} == {"markdown"}

        for payload in sent:
            # Normalization runs after splitting and can grow a chunk, so the
            # guarantee is DingTalk's ceiling, not the 3500 split budget.
            text = payload["markdown"]["text"]
            assert len(text) <= HARD_MAX_LEN

    async def test_split_chunks_keep_table_renderable(
        self,
        dingtalk_channel,
    ):
        """Every chunk must still be a valid table, header repeated."""
        sent: List[Dict] = []

        async def _capture(_session_webhook, payload):
            sent.append(payload)
            return True

        dingtalk_channel._send_payload_via_session_webhook = _capture

        await dingtalk_channel._send_via_session_webhook(
            session_webhook="https://oapi.dingtalk.com/robot/send?session=abc",
            body=_big_table(),
            bot_prefix="",
        )

        assert len(sent) > 1
        for payload in sent:
            text = payload["markdown"]["text"]
            # Header row and separator repeated per chunk keeps GFM happy.
            assert "| metric | p50 | p95 | p99 |" in text
            assert "| --- | --- | --- | --- |" in text

    async def test_mention_attached_to_first_chunk_only(
        self,
        dingtalk_channel,
    ):
        """Repeated @mentions on every chunk would spam notifications."""
        sent: List[Dict] = []

        async def _capture(_session_webhook, payload):
            sent.append(payload)
            return True

        dingtalk_channel._send_payload_via_session_webhook = _capture

        await dingtalk_channel._send_via_session_webhook(
            session_webhook="https://oapi.dingtalk.com/robot/send?session=abc",
            body=_big_table(),
            bot_prefix="[Bot]",
            at_dingtalk_ids=["ding-1"],
        )

        assert len(sent) > 1
        assert "at" in sent[0]
        assert all("at" not in p for p in sent[1:])

    async def test_short_reply_still_single_markdown_message(
        self,
        dingtalk_channel,
    ):
        """Short replies must be untouched: one message, no splitting."""
        sent: List[Dict] = []

        async def _capture(_session_webhook, payload):
            sent.append(payload)
            return True

        dingtalk_channel._send_payload_via_session_webhook = _capture

        await dingtalk_channel._send_via_session_webhook(
            session_webhook="https://oapi.dingtalk.com/robot/send?session=abc",
            body="Hello from test",
            bot_prefix="[Bot]",
        )

        assert len(sent) == 1
        assert sent[0]["msgtype"] == "markdown"
        assert "Hello from test" in sent[0]["markdown"]["text"]

    async def test_long_table_stays_markdown_via_open_api(
        self,
        dingtalk_channel,
    ):
        """The Open API fallback had the same cliff; fix it too."""
        calls: List[Dict] = []

        async def _capture(*, msg_key, msg_param, **_kwargs):
            calls.append({"msg_key": msg_key, "msg_param": msg_param})
            return True

        dingtalk_channel._send_robot_message = _capture

        ok = await dingtalk_channel._send_via_open_api(
            body=_big_table(),
            conversation_id="cid-1",
            conversation_type="group",
            sender_staff_id="staff-1",
        )

        assert ok is True
        # Baseline sends one "sampleText" message -> this fails.
        assert len(calls) > 1
        assert {c["msg_key"] for c in calls} == {"sampleMarkdown"}

        for call in calls:
            text = json.loads(call["msg_param"])["text"]
            assert len(text) <= HARD_MAX_LEN

    async def test_short_reply_open_api_single_markdown(
        self,
        dingtalk_channel,
    ):
        """Short replies on the fallback path must stay a single message."""
        calls: List[Dict] = []

        async def _capture(*, msg_key, msg_param, **_kwargs):
            calls.append({"msg_key": msg_key, "msg_param": msg_param})
            return True

        dingtalk_channel._send_robot_message = _capture

        await dingtalk_channel._send_via_open_api(
            body="Hello group",
            conversation_id="cid-1",
            conversation_type="group",
            sender_staff_id="staff-1",
        )

        assert len(calls) == 1
        assert calls[0]["msg_key"] == "sampleMarkdown"

    async def test_whitespace_only_reply_is_still_delivered(
        self,
        dingtalk_channel,
    ):
        """split_text drops whitespace chunks; the reply must not vanish."""
        sent: List[Dict] = []

        async def _capture(_session_webhook, payload):
            sent.append(payload)
            return True

        dingtalk_channel._send_payload_via_session_webhook = _capture

        ok = await dingtalk_channel._send_via_session_webhook(
            session_webhook="https://oapi.dingtalk.com/robot/send?session=abc",
            body=" " * 6000,
            bot_prefix="",
        )

        # Reporting success while sending nothing loses the reply silently.
        assert ok is True
        assert len(sent) == 1, "whitespace-only reply must still be delivered"

    async def test_partial_failure_is_reported_so_caller_can_fall_back(
        self,
        dingtalk_channel,
    ):
        """A lost chunk must make the whole call report failure.

        send()/send_content_parts() fall back to the Open API when this
        returns False, which re-sends the whole body and duplicates what
        already arrived. That duplication is deliberate: it is visible and
        recoverable, whereas reporting success on a partial send would drop
        the remaining chunks silently and bypass the api_send strict-delivery
        contract. See the comment in _send_via_session_webhook.
        """
        calls: List[int] = []

        async def _capture(_session_webhook, payload):
            calls.append(len(payload["markdown"]["text"]))
            return len(calls) != 3  # a middle chunk fails, the rest land

        dingtalk_channel._send_payload_via_session_webhook = _capture

        ok = await dingtalk_channel._send_via_session_webhook(
            session_webhook="https://oapi.dingtalk.com/robot/send?session=abc",
            body=_make_big_table(400),
            bot_prefix="",
        )

        assert len(calls) >= 4, "fixture must produce several chunks"
        assert ok is False, "a lost chunk must not be reported as success"

    async def test_total_failure_still_reports_failure(
        self,
        dingtalk_channel,
    ):
        """If nothing at all was delivered the caller must fall back."""
        calls: List[int] = []

        async def _capture(_session_webhook, _payload):
            calls.append(1)
            return False

        dingtalk_channel._send_payload_via_session_webhook = _capture

        ok = await dingtalk_channel._send_via_session_webhook(
            session_webhook="https://oapi.dingtalk.com/robot/send?session=abc",
            body=_big_table(),
            bot_prefix="",
        )

        assert ok is False
        assert len(calls) > 1

    async def test_indivisible_chunk_falls_back_to_plain_text(
        self,
        dingtalk_channel,
    ):
        """One enormous table row cannot be split; it must not be rejected.

        DingTalk rejects markdown over its ceiling, but accepts much longer
        plain text, so this chunk degrades instead of being lost.
        """
        sent: List[Dict] = []

        async def _capture(_session_webhook, payload):
            sent.append(payload)
            return True

        dingtalk_channel._send_payload_via_session_webhook = _capture

        body = "| a | b |\n| --- | --- |\n| x | " + ("z" * 9000) + " |"
        await dingtalk_channel._send_via_session_webhook(
            session_webhook="https://oapi.dingtalk.com/robot/send?session=abc",
            body=body,
            bot_prefix="",
        )

        assert len(sent) == 1
        assert sent[0]["msgtype"] == "text"
        assert sent[0]["text"]["content"], "content must survive the downgrade"

    async def test_normalized_chunk_stays_under_dingtalk_ceiling(
        self,
        dingtalk_channel,
    ):
        """Normalization runs after splitting and can grow a chunk."""
        sent: List[Dict] = []

        async def _capture(_session_webhook, payload):
            sent.append(payload)
            return True

        dingtalk_channel._send_payload_via_session_webhook = _capture

        # ensure_list_spacing only inserts a blank line before a numbered item
        # when the previous line is not itself numbered, so a dense numbered
        # list would not grow at all and the assertion below would be vacuous.
        # A non-numbered line before each item is what actually expands it.
        body = "\n".join(f"note\n{i}. item {i}" for i in range(1, 400))
        await dingtalk_channel._send_via_session_webhook(
            session_webhook="https://oapi.dingtalk.com/robot/send?session=abc",
            body=body,
            bot_prefix="",
        )

        markdown_bodies = [
            p["markdown"]["text"] for p in sent if p["msgtype"] == "markdown"
        ]
        assert len(markdown_bodies) > 1
        longest = max(len(b) for b in markdown_bodies)
        # Guard against a vacuous fixture: normalization must actually push a
        # chunk past the raw split budget, otherwise this test would still pass
        # with the normalizer stubbed out.
        assert longest > MAX_LEN, "fixture must exercise real growth"
        assert longest <= HARD_MAX_LEN
