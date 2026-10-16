# -*- coding: utf-8 -*-
# pylint: disable=protected-access,redefined-outer-name
"""Reranker enablement and credential diagnostics.

Covers the two facets of a credential-less reranker configuration:

  - ``load_reranker_config`` must not report a reranker as enabled when no
    credential is configured, matching the rule the embedding path already
    applies in ``_is_embedding_enabled``.
  - ``call_reranker_api`` must name a missing credential in its first log
    line, the way it already names a missing ``base_url``, instead of
    surfacing the resulting ``h11`` header-serialization failure as
    "[rerank] unexpected error".

The call-path tests run against a real loopback HTTP server so that "no
request was attempted" is observed on the server side rather than inferred
from a stubbed client.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import qwenpaw.agents.memory.reme_reranker as reranker
from qwenpaw.config.config import (
    AgentProfileConfig,
    AgentsRunningConfig,
    ReMeLightMemoryConfig,
    RerankerConfig,
)

pytestmark = pytest.mark.unit


def _agent_config(reranker_config: RerankerConfig) -> AgentProfileConfig:
    """Build a real agent profile carrying *reranker_config*."""
    return AgentProfileConfig(
        id="agent-1",
        name="Agent One",
        running=AgentsRunningConfig(
            reme_light_memory_config=ReMeLightMemoryConfig(
                reranker_config=reranker_config,
            ),
        ),
    )


def _reranker_config(**overrides) -> RerankerConfig:
    """A reranker that is otherwise fully configured and enabled."""
    fields = {
        "enabled": True,
        "api_key": "sk-configured",
        "base_url": "https://reranker.example/v1",
        "model_name": "BAAI/bge-reranker-v2-m3",
    }
    fields.update(overrides)
    return RerankerConfig(**fields)


async def _load(monkeypatch, config: RerankerConfig):
    """Run ``load_reranker_config`` against *config*."""
    agent_config = _agent_config(config)
    monkeypatch.setattr(
        reranker,
        "load_agent_config_async",
        _async_return(agent_config),
    )
    return await reranker.load_reranker_config("agent-1")


def _async_return(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


class _RecordingHandler(BaseHTTPRequestHandler):
    """Answers every POST and records the request it received."""

    protocol_version = "HTTP/1.1"
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler API)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        type(self).requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": json.loads(body) if body else None,
            },
        )
        payload = json.dumps(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ],
            },
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        """Keep the test output free of per-request noise."""


@pytest.fixture
def rerank_server():
    """A real HTTP server standing in for the reranker endpoint."""
    _RecordingHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _base_url(server) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


# --- enablement ---------------------------------------------------------


async def test_reranker_not_enabled_without_api_key(monkeypatch, caplog):
    """An empty key must not count as an enabled reranker, and say so."""
    with caplog.at_level(logging.WARNING, logger=reranker.__name__):
        assert await _load(monkeypatch, _reranker_config(api_key="")) is None

    # The decision is made here, so this is the only place the user can be
    # told; the guard in call_reranker_api() is unreachable once enablement
    # has already rejected the configuration.
    assert "[rerank] enabled but api_key not configured" in caplog.text
    assert "[rerank] unexpected error" not in caplog.text


async def test_reranker_not_enabled_with_whitespace_api_key(
    monkeypatch,
    caplog,
) -> None:
    """A whitespace-only key is not a credential either."""
    with caplog.at_level(logging.WARNING, logger=reranker.__name__):
        assert (
            await _load(monkeypatch, _reranker_config(api_key="   ")) is None
        )

    assert "[rerank] enabled but api_key not configured" in caplog.text


async def test_reranker_enabled_with_api_key(monkeypatch) -> None:
    """A configured credential still enables the reranker."""
    config = _reranker_config()

    loaded = await _load(monkeypatch, config)

    assert loaded is not None
    assert loaded.model_name == "BAAI/bge-reranker-v2-m3"


async def test_reranker_still_disabled_when_flag_is_off(monkeypatch, caplog):
    """Requiring a credential must not turn the flag on by itself."""
    config = _reranker_config(enabled=False)

    with caplog.at_level(logging.WARNING, logger=reranker.__name__):
        assert await _load(monkeypatch, config) is None

    # A reranker nobody enabled is not a misconfiguration; stay quiet.
    assert caplog.text == ""


# --- diagnostics --------------------------------------------------------


async def test_missing_api_key_is_named_and_no_request_is_sent(
    rerank_server,
    caplog,
) -> None:
    """A missing key gets its own log line and never hits the network."""
    config = _reranker_config(api_key="", base_url=_base_url(rerank_server))

    with caplog.at_level(logging.WARNING, logger=reranker.__name__):
        result = await reranker.call_reranker_api("q", ["a", "b"], config)

    assert result is None
    assert "[rerank] api_key not configured" in caplog.text
    assert "[rerank] unexpected error" not in caplog.text
    assert not _RecordingHandler.requests


async def test_whitespace_api_key_is_named(rerank_server, caplog) -> None:
    """A whitespace-only key is reported the same way."""
    config = _reranker_config(api_key="  ", base_url=_base_url(rerank_server))

    with caplog.at_level(logging.WARNING, logger=reranker.__name__):
        result = await reranker.call_reranker_api("q", ["a", "b"], config)

    assert result is None
    assert "[rerank] api_key not configured" in caplog.text
    assert not _RecordingHandler.requests


async def test_configured_api_key_reaches_the_endpoint(rerank_server) -> None:
    """The working path is unchanged: the token is sent and order returns."""
    config = _reranker_config(
        api_key="sk-configured",
        base_url=_base_url(rerank_server),
    )

    result = await reranker.call_reranker_api("q", ["a", "b"], config)

    assert result == [1, 0]
    assert len(_RecordingHandler.requests) == 1
    assert _RecordingHandler.requests[0]["authorization"] == (
        "Bearer sk-configured"
    )


async def test_padded_api_key_is_sent_without_surrounding_space(
    rerank_server,
) -> None:
    """A padded credential must not become an unserializable header."""
    config = _reranker_config(
        api_key="  sk-padded  ",
        base_url=_base_url(rerank_server),
    )

    result = await reranker.call_reranker_api("q", ["a", "b"], config)

    assert result == [1, 0]
    assert _RecordingHandler.requests[0]["authorization"] == (
        "Bearer sk-padded"
    )


async def test_api_key_not_logged_on_the_request_path(
    rerank_server,
    caplog,
) -> None:
    """A sent credential does not appear in this module's own log lines.

    Scope: covers the request path only, and only for a well-formed key.
    A key that is still unserializable after stripping (interior
    whitespace) reaches the log through the pre-existing broad handler's
    ``exc_info=True`` traceback, which carries httpx's message verbatim.
    That is a pre-existing gap, tracked as a follow-up, not asserted here.
    """
    config = _reranker_config(
        api_key="sk-secret-value",
        base_url=_base_url(rerank_server),
    )

    with caplog.at_level(logging.DEBUG, logger=reranker.__name__):
        await reranker.call_reranker_api("q", ["a", "b"], config)

    assert "sk-secret-value" not in caplog.text
