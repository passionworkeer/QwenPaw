# -*- coding: utf-8 -*-
"""Quoted controls travel from real inbound parsers to runtime cancellation."""
# pylint: disable=protected-access,redefined-outer-name,unused-argument
# pylint: disable=too-many-branches,too-many-statements
from __future__ import annotations

import asyncio
import importlib
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from qwenpaw.app.channels.command_registry import CommandRegistry
from qwenpaw.app.channels.manager import ChannelManager, _process_batch
from qwenpaw.app.channels.unified_queue_manager import UnifiedQueueManager
from qwenpaw.app.task_tracker import TaskTracker
from qwenpaw.runtime.builtin_commands import _make_control_adapter
from qwenpaw.runtime.builtin_commands import _make_daemon_compound_adapter
from qwenpaw.runtime.commands.control.stop_handler import StopCommandHandler
from qwenpaw.runtime.hooks import HookRegistry
from qwenpaw.runtime.runtime import Runtime
from qwenpaw.runtime.slash_command_registry import SlashCommandRegistry
from qwenpaw.schemas import ContentType, ImageContent, TextContent


CHANNELS = {
    "wechat": "WeChatChannel",
    "wecom": "WecomChannel",
    "feishu": "FeishuChannel",
    "qq": "QQChannel",
    "onebot": "OneBotChannel",
    "yuanbao": "YuanbaoChannel",
    "dingtalk": "DingTalkChannel",
}


def _make_channel(name, tmp_path, process):
    module = importlib.import_module(
        f"qwenpaw.app.channels.{name}.channel",
    )
    options = {"process": process, "enabled": True, "bot_prefix": ""}
    if name in {"wechat", "wecom", "feishu", "qq", "yuanbao"}:
        options["media_dir"] = str(tmp_path / "media")
    if name == "wechat":
        options["bot_token"] = "test-token"
    elif name == "wecom":
        options.update(bot_id="bot", secret="test-secret")
    elif name in {"feishu", "yuanbao"}:
        options.update(app_id="bot", app_secret="test-secret")
    elif name == "qq":
        options.update(app_id="bot", client_secret="test-secret")
    elif name == "dingtalk":
        options.update(client_id="bot", client_secret="test-secret")
    channel = getattr(module, CHANNELS[name])(**options)
    channel._before_consume_process = AsyncMock()
    channel.on_event_message_completed = AsyncMock()
    channel.send_content_parts = AsyncMock()
    channel._finish_response_cycle = AsyncMock()
    return channel


async def _inbound(channel, text, quoted, *, voice=False):
    """Only remote I/O and delivery ACKs are replaced, never text parsing."""
    captured = []
    channel._enqueue = captured.append
    message_id = uuid.uuid4().hex
    name = channel.channel
    media_quote = quoted == "__media__"
    if name == "wechat":
        item = {"type": 1, "text_item": {"text": text}}
        if voice:
            item = {"type": 3, "voice_item": {"text": text}}
        if quoted is not None:
            item["ref_msg"] = {
                "message_item": (
                    {"type": 2, "image_item": {"media": {}}}
                    if media_quote
                    else {"type": 1, "text_item": {"text": quoted}}
                ),
            }
        await channel._on_message(
            {
                "from_user_id": message_id,
                "message_type": 1,
                "msg_id": message_id,
                "item_list": [item],
            },
            client=object(),
        )
    elif name == "wecom":
        body = {
            "from": {"userid": "user"},
            "msgid": message_id,
            "chattype": "single",
            "msgtype": "text",
            "text": {"content": text},
        }
        if voice:
            body.update(msgtype="voice", voice={"content": text})
        if quoted is not None:
            body["quote"] = {
                "msgtype": "text",
                "text": {"content": quoted},
            }
            if media_quote:
                channel._download_media = AsyncMock(return_value=None)
                body["quote"] = {
                    "msgtype": "image",
                    "image": {"url": "test-image"},
                }
        await channel._on_message({"body": body})
    elif name == "feishu":
        channel._fetch_quoted_message_content = AsyncMock(
            return_value=(
                ("image", json.dumps({"image_key": "quoted"}))
                if media_quote
                else ("text", json.dumps({"text": quoted}))
            ),
        )
        channel._download_image_resource = AsyncMock(return_value=None)
        channel._add_reaction = AsyncMock()
        await channel._on_message(
            SimpleNamespace(
                event=SimpleNamespace(
                    sender=SimpleNamespace(
                        sender_id=SimpleNamespace(open_id="user"),
                        name="Alice",
                    ),
                    message=SimpleNamespace(
                        message_id=message_id,
                        message_type="text",
                        chat_type="p2p",
                        chat_id="chat",
                        parent_id="parent" if quoted is not None else "",
                        content=json.dumps({"text": text}),
                    ),
                ),
            ),
        )
    elif name == "qq":
        channel._schedule_ack = lambda *args, **kwargs: None
        body = {
            "id": message_id,
            "author": {"user_openid": "user"},
            "content": text,
        }
        if quoted is not None:
            body.update(
                msg_elements=[{"msg_idx": "1", "content": quoted}],
                message_scene={"ext": ["ref_msg_idx=1"]},
            )
            if media_quote:
                channel._parse_qq_attachments = lambda _attachments: []
                body["msg_elements"][0].update(
                    content="",
                    attachments=[{"filename": "image.png"}],
                )
        channel._handle_msg_event("C2C_MESSAGE_CREATE", body)
    elif name == "onebot":
        channel._get_quoted_message_segments = AsyncMock(
            return_value=[
                {"type": "image", "data": {"url": "test-image"}}
                if media_quote
                else {"type": "text", "data": {"text": quoted}},
            ],
        )
        if media_quote:

            async def resolve(parts, *_args):
                return parts

            channel._inbound_media.resolve = resolve
        segments = [{"type": "text", "data": {"text": text}}]
        if quoted is not None:
            segments.insert(0, {"type": "reply", "data": {"id": "1"}})
        await channel._handle_message_event(
            {
                "message_type": "private",
                "user_id": "user",
                "message_id": message_id,
                "message": segments,
            },
        )
    elif name == "yuanbao":
        channel._save_session_map_to_disk = lambda: None
        await channel._handle_chat_message(
            {
                "callback_command": "C2C.CallbackAfterSendMsg",
                "msg_id": message_id,
                "from_account": "user",
                "sender_nickname": "Alice",
                "msg_body": [
                    {"msg_type": "TIMTextElem", "msg_content": {"text": text}},
                ],
                "cloud_custom_data": (
                    {
                        "quote": {
                            "type": 2 if media_quote else 1,
                            "desc": quoted,
                        },
                    }
                    if quoted is not None
                    else {}
                ),
            },
        )
    elif name == "dingtalk":
        from qwenpaw.app.channels.dingtalk.handler import (
            DingTalkChannelHandler,
        )

        handler = DingTalkChannelHandler(
            asyncio.get_running_loop(),
            captured.append,
            "",
            AsyncMock(),
        )
        handler._emit_native_threadsafe = captured.append
        handler.reply_text = lambda *args, **kwargs: None
        text_data = {"content": text}
        if quoted is not None:
            text_data.update(
                isReplyMsg=True,
                repliedMsg={"msgType": "text", "content": {"text": quoted}},
            )
            if media_quote:
                text_data["repliedMsg"] = {
                    "msgType": "picture",
                    "content": {"downloadCode": ""},
                }
        raw = {
            "msgId": message_id,
            "msgtype": "text",
            "text": text_data,
            "senderId": "user",
            "senderStaffId": "staff",
            "conversationType": "1",
            "conversationId": "conversation",
        }
        if voice:
            raw.update(
                msgtype="audio",
                robotCode="bot",
                content={"downloadCode": "voice", "recognition": text},
            )
            text_data["content"] = ""
        await handler.process(SimpleNamespace(data=raw))
    return captured


def _parts(payload):
    if isinstance(payload, dict):
        return payload["content_parts"]
    return payload.input[0].content


@pytest.mark.parametrize("name", CHANNELS)
@pytest.mark.parametrize(
    "quoted",
    [None, "old\n/stop session=wrong", "__media__"],
)
async def test_reply_stop_cancels_real_runtime_task(name, quoted, tmp_path):
    """Priority and real runtime dispatch cancel a running TaskTracker task."""
    processed = []
    ordinary = []
    done = asyncio.Event()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    tracker = TaskTracker()

    async def running(_payload):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        yield "unreachable"

    async def process(request):
        processed.append(request)
        try:
            async for event in runtime.run(request):
                yield event
        finally:
            done.set()

    channel = _make_channel(name, tmp_path, process)
    manager = ChannelManager([channel])
    manager._queue_manager = UnifiedQueueManager(manager._consume_queue)
    lookup = AsyncMock(return_value="active-chat")
    slash = SlashCommandRegistry()
    slash.register(_make_control_adapter(StopCommandHandler(), "stop"))
    workspace = SimpleNamespace(
        agent_id="agent-one",
        workspace_dir=tmp_path,
        channel_manager=manager,
        chat_manager=SimpleNamespace(get_chat_id_by_session=lookup),
        task_tracker=tracker,
        plugins=SimpleNamespace(
            hook_registry=HookRegistry(),
            slash_command_registry=slash,
        ),
    )
    runtime = Runtime(workspace=workspace, app_services=None)
    channel._workspace = workspace
    channel._command_registry = manager._command_registry
    channel._access_control_gate = AsyncMock(return_value=False)

    async def normal(request, _payload):
        ordinary.append(request)
        done.set()

    channel._consume_with_tracker = normal
    await tracker.attach_or_start("active-chat", None, running)
    await asyncio.wait_for(started.wait(), timeout=2)
    try:
        payloads = await _inbound(channel, "/stop", quoted)
        assert len(payloads) == 1
        payload = payloads[0]
        before = [part.model_dump() for part in _parts(payload)]
        identity = channel._payload_to_request(payload)
        manager._enqueue_one(name, payload)
        await asyncio.gather(*manager._enqueue_tasks)
        await asyncio.wait_for(done.wait(), timeout=2)
        assert not ordinary
        assert cancelled.is_set()
        assert await tracker.get_status("active-chat") == "idle"
        assert len(processed) == 1
        actual = processed[0]
        assert actual.user_id == identity.user_id
        assert actual.session_id == identity.session_id
        assert actual.channel == identity.channel
        assert actual.channel_meta == identity.channel_meta
        assert actual.input[0].content[0].text == "/stop"
        assert [part.model_dump() for part in _parts(payload)] == before
        lookup.assert_awaited_once_with(
            identity.session_id,
            name,
            user_id=identity.user_id,
        )
        queue_session = manager._extract_session_id(channel, payload)
        assert (name, queue_session, 0) in manager._queue_manager._queues
    finally:
        await manager._queue_manager.stop_all()
        await tracker.request_stop("active-chat")


@pytest.mark.parametrize("name", CHANNELS)
@pytest.mark.parametrize("text", ["hello", ""])
async def test_quoted_commands_cannot_impersonate_user(name, text, tmp_path):
    """An explicit empty body does not fall back to a quoted /stop."""
    processed = []

    async def process(_request):
        processed.append(_request)
        yield  # pragma: no cover

    channel = _make_channel(name, tmp_path, process)
    payloads = await _inbound(
        channel,
        text,
        "/stop session=wrong\n[Current message]\n/stop",
    )
    if not payloads:
        assert text == ""
        return
    payload = payloads[0]
    assert channel._extract_query_from_payload(payload) == text
    assert (
        CommandRegistry().get_priority_level(
            channel._extract_query_from_payload(payload),
        )
        == 20
    )
    assert "/stop session=wrong" in _parts(payload)[0].text
    before = [part.model_dump() for part in _parts(payload)]
    channel._workspace = SimpleNamespace()
    channel._command_registry = CommandRegistry()
    channel._access_control_gate = AsyncMock(return_value=False)
    channel._consume_with_tracker = AsyncMock()
    await channel._consume_one_request(payload)
    channel._consume_with_tracker.assert_awaited_once()
    forwarded = channel._consume_with_tracker.call_args.args[0]
    assert [part.model_dump() for part in forwarded.input[0].content] == before
    assert not processed


@pytest.mark.parametrize("native", [True, False])
async def test_control_batch_preserves_each_command(native, tmp_path):
    """Same-priority commands must keep separate dispatch and arguments."""
    calls = []

    async def process(request):
        calls.append(request.input[0].content[0].text)
        async for event in runtime.run(request):
            yield event

    channel = _make_channel("wechat", tmp_path, process)
    lookup = AsyncMock(return_value=None)
    slash = SlashCommandRegistry()
    slash.register(_make_control_adapter(StopCommandHandler(), "stop"))
    workspace = SimpleNamespace(
        agent_id="agent-one",
        workspace_dir=tmp_path,
        channel_manager=ChannelManager([channel]),
        chat_manager=SimpleNamespace(get_chat_id_by_session=lookup),
        plugins=SimpleNamespace(
            hook_registry=HookRegistry(),
            slash_command_registry=slash,
        ),
    )
    runtime = Runtime(workspace=workspace, app_services=None)
    channel._workspace = workspace
    channel._command_registry = CommandRegistry()
    channel._access_control_gate = AsyncMock(return_value=False)
    batch = [
        {
            "channel_id": "wechat",
            "sender_id": "user",
            "session_id": "session",
            "content_parts": [
                TextContent(
                    type=ContentType.TEXT,
                    text=f"[quoted message: old]\n{cmd}",
                ),
            ],
            "meta": {"user_text": cmd},
        }
        for cmd in ("/stop session=one", "/stop session=two")
    ]
    if not native:
        batch = [channel._payload_to_request(payload) for payload in batch]
    channel._consume_with_tracker = AsyncMock()
    await _process_batch(channel, batch)
    assert calls == ["/stop session=one", "/stop session=two"]
    assert [args.args for args in lookup.await_args_list] == [
        ("one", "wechat"),
        ("two", "wechat"),
    ]
    assert all(
        args.kwargs == {"user_id": "user"} for args in lookup.await_args_list
    )
    channel._consume_with_tracker.assert_not_awaited()


async def test_control_skips_time_debounce_after_acl(tmp_path):
    calls = []

    async def process(request):
        calls.append(request.input[0].content[0].text)
        yield SimpleNamespace()

    channel = _make_channel("wechat", tmp_path, process)
    channel._workspace = SimpleNamespace()
    channel._command_registry = CommandRegistry()
    channel._access_control_gate = AsyncMock(return_value=False)
    channel._debounce_seconds = 60
    payload = {
        "sender_id": "user",
        "session_id": "session",
        "content_parts": [
            TextContent(
                type=ContentType.TEXT,
                text="[quoted message: x]\n/stop",
            ),
            ImageContent(type=ContentType.IMAGE, image_url="image.png"),
        ],
        "meta": {"user_text": "/stop"},
    }
    try:
        await channel.consume_one(payload)
        assert calls == ["/stop"]
        assert channel._debounce_timers == {}
        channel._access_control_gate.assert_awaited_once_with(payload)
        channel._access_control_gate.reset_mock()
        channel._access_control_gate.return_value = True
        await channel.consume_one(payload)
        assert calls == ["/stop"]
        channel._access_control_gate.assert_awaited_once_with(payload)
    finally:
        for timer in channel._debounce_timers.values():
            timer.cancel()
        await asyncio.gather(
            *channel._debounce_timers.values(),
            return_exceptions=True,
        )


@pytest.mark.parametrize("name", CHANNELS)
@pytest.mark.parametrize("requests", [False, True])
async def test_ordinary_batch_preserves_each_original_user_text(
    name,
    requests,
    tmp_path,
):
    """Last/first routing metadata must not discard another message's body."""

    async def process(_request):
        yield  # pragma: no cover

    channel = _make_channel(name, tmp_path, process)
    batch = [
        {
            "channel_id": name,
            "sender_id": "user",
            "session_id": "session",
            "content_parts": [
                TextContent(
                    type=ContentType.TEXT,
                    text=f"[quoted message: old]\n{body}",
                ),
            ],
            "meta": {"user_text": body, "message_id": body},
        }
        for body in ("hello", "world")
    ]
    if requests:
        inputs = [channel._payload_to_request(item) for item in batch]
        merged = channel.merge_requests(inputs)
    else:
        merged = channel.merge_native_items(batch)
    assert channel._extract_query_from_payload(merged) == "hello\nworld"
    assert [part.text for part in _parts(merged)] == [
        "[quoted message: old]\nhello",
        "[quoted message: old]\nworld",
    ]
    assert [item["meta"]["user_text"] for item in batch] == [
        "hello",
        "world",
    ]


@pytest.mark.parametrize("user_text", [None, 123, [], {}])
def test_legacy_and_malformed_user_text_keeps_existing_extraction(
    user_text,
    tmp_path,
):
    async def process(_request):
        yield  # pragma: no cover

    channel = _make_channel("wechat", tmp_path, process)
    payload = {
        "content_parts": [TextContent(type=ContentType.TEXT, text="hello")],
        "meta": {"user_text": user_text},
    }
    assert channel._extract_query_from_payload(payload) == "hello"


def test_duck_typed_request_batch_preserves_user_text(tmp_path):
    """Non-Pydantic request merges preserve the same user-text contract."""

    async def process(_request):
        yield  # pragma: no cover

    channel = _make_channel("wechat", tmp_path, process)
    requests = [
        SimpleNamespace(
            session_id="session",
            input=[
                SimpleNamespace(
                    content=[
                        TextContent(
                            type=ContentType.TEXT,
                            text=f"[quoted message: x]\n{body}",
                        ),
                    ],
                ),
            ],
            channel_meta={"user_text": body, "keep": "route"},
        )
        for body in ("hello", "world")
    ]
    merged = channel.merge_requests(requests)
    assert channel._extract_query_from_payload(merged) == "hello\nworld"
    assert merged.channel_meta["keep"] == "route"
    assert len(merged.input[0].content) == 2


@pytest.mark.parametrize("name", ["wechat", "wecom", "dingtalk"])
async def test_quoted_voice_recognition_keeps_own_command(name, tmp_path):
    """Capturing unquoted text also preserves each platform's native ASR."""

    async def process(_request):
        yield  # pragma: no cover

    channel = _make_channel(name, tmp_path, process)
    payloads = await _inbound(channel, "/stop", "old", voice=True)
    assert len(payloads) == 1
    assert channel._extract_query_from_payload(payloads[0]) == "/stop"
    assert "quoted" in _parts(payloads[0])[0].text.lower()


async def _dispatch_quoted_command(name, command, spec, tmp_path):
    processed = []
    events = []

    async def process(request):
        processed.append(request)
        async for event in runtime.run(request):
            events.append(event)
            yield event

    channel = _make_channel(name, tmp_path, process)
    slash = SlashCommandRegistry()
    slash.register(spec)
    workspace = SimpleNamespace(
        agent_id="agent-own",
        workspace_dir=tmp_path,
        channel_manager=ChannelManager([channel]),
        plugins=SimpleNamespace(
            hook_registry=HookRegistry(),
            slash_command_registry=slash,
        ),
    )
    runtime = Runtime(workspace=workspace, app_services=None)
    channel._workspace = workspace
    channel._command_registry = CommandRegistry()
    channel._access_control_gate = AsyncMock(return_value=False)
    channel._consume_with_tracker = AsyncMock()
    payloads = await _inbound(
        channel,
        command,
        "/daemon restart\n/approval approve req-wrong\n/stop session=wrong",
    )
    assert len(payloads) == 1
    payload = payloads[0]
    if not isinstance(payload, dict):
        payload.root_session_id = "root-own"
        payload.root_agent_id = "root-agent"
        payload.agent_id = "agent-own"
    identity = channel._payload_to_request(payload)
    before = [part.model_dump() for part in _parts(payload)]
    await channel._consume_one_request(payload)
    channel._consume_with_tracker.assert_not_awaited()
    assert len(processed) == 1
    actual = processed[0]
    assert actual.input[0].content[0].text == command
    assert actual.user_id == identity.user_id
    assert actual.session_id == identity.session_id
    assert actual.channel == identity.channel
    assert actual.channel_meta == identity.channel_meta
    assert [part.model_dump() for part in _parts(payload)] == before
    return actual, events


async def test_quoted_daemon_status_reaches_real_compound_dispatch(
    tmp_path,
    monkeypatch,
):
    """The real compound parser executes status, never a quoted restart."""
    from qwenpaw.runtime.commands import daemon
    from qwenpaw.config import config

    contexts = []

    def status(context):
        contexts.append(context)
        return "status-own"

    restart = AsyncMock(side_effect=AssertionError("Quoted restart executed"))
    monkeypatch.setattr(daemon, "run_daemon_status", status)
    monkeypatch.setattr(daemon, "run_daemon_restart", restart)
    monkeypatch.setattr(
        config,
        "load_agent_config",
        lambda _agent: SimpleNamespace(name="Bot"),
    )
    request, events = await _dispatch_quoted_command(
        "wechat",
        "/daemon status",
        _make_daemon_compound_adapter(),
        tmp_path,
    )
    assert len(contexts) == 1
    assert contexts[0].session_id == request.session_id
    assert contexts[0].agent_id == "agent-own"
    assert "status-own" in "".join(str(event) for event in events)
    restart.assert_not_awaited()


async def test_quoted_approval_preserves_real_parser_and_actor(
    tmp_path,
    monkeypatch,
):
    """Deny keeps the caller's request, reason and root identity."""
    from qwenpaw.runtime.commands.control import approval_handler
    from qwenpaw.security.tool_guard.approval import ApprovalDecision

    pending = SimpleNamespace(
        agent_id="agent-own",
        session_id="pending-session",
        root_session_id="pending-session",
        tool_name="tool-own",
    )
    service = SimpleNamespace(
        get_request=AsyncMock(return_value=pending),
        resolve_request=AsyncMock(return_value=pending),
    )
    monkeypatch.setattr(
        approval_handler,
        "get_approval_service",
        lambda: service,
    )
    request, events = await _dispatch_quoted_command(
        "qq",
        "/approval deny req-own reason-own",
        _make_control_adapter(
            approval_handler.ApprovalCommandHandler(),
            "approval",
        ),
        tmp_path,
    )
    service.get_request.assert_awaited_once_with("req-own")
    service.resolve_request.assert_awaited_once()
    call = service.resolve_request.await_args
    assert call.args == ("req-own", ApprovalDecision.DENIED)
    actor = call.kwargs["actor"]
    assert actor.session_id == request.session_id
    assert actor.root_session_id == "root-own"
    assert actor.user_id == request.user_id
    assert actor.channel == "qq"
    assert actor.agent_id == "agent-own"
    assert request.root_agent_id == "root-agent"
    assert "reason-own" in "".join(str(event) for event in events)
