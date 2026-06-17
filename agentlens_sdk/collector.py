"""Local AgentLens collector and provider monkeypatches."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .pricing import compute_cost_usd


RUNS_DIR = Path(".agentlens") / "runs"
_current_run: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "agentlens_current_run", default=None
)
_config: dict[str, Any] = {"api_key": None, "patched": set(), "originals": {}}


class AmbiguousRunIdError(ValueError):
    """Raised when a run_id prefix matches more than one local run."""

    def __init__(self, prefix: str, matches: list[str]):
        super().__init__(f"Multiple runs match '{prefix}'")
        self.prefix = prefix
        self.matches = matches


class AgentLensClient:
    """Backward-compatible direct Anthropic wrapper from the first Phase 1 pass."""

    def __init__(self, api_key: str, client: Any | None = None):
        self._client = client if client is not None else self._build_anthropic_client(api_key)
        self._run = start_run(name="manual")

    def messages_create(self, **kwargs: Any) -> Any:
        # Delegate to the shared proxy so capture logic lives in exactly one place.
        proxy = _AnthropicMessagesProxy(self._client.messages)
        token = _current_run.set(self._run)
        try:
            return proxy.create(**kwargs)
        finally:
            _current_run.reset(token)

    def record_tool_result(
        self, tool_name: str, output: Any, input: Any | None = None, tool_use_id: str | None = None
    ) -> None:
        token = _current_run.set(self._run)
        try:
            record_tool_result(tool_name=tool_name, output=output, input=input, tool_use_id=tool_use_id)
        finally:
            _current_run.reset(token)

    def save_run(self, path: str = "agentlens_run.json") -> None:
        save_run(path=path, run=self._run)

    @property
    def spans(self) -> list[dict[str, Any]]:
        return self._run["spans"]

    @staticmethod
    def _build_anthropic_client(api_key: str) -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic package is required when no client is provided. "
                "Install it or pass a test client into AgentLensClient(..., client=...)."
            ) from exc

        return anthropic.Anthropic(api_key=api_key)


def init(
    api_key: str | None = None,
    parent_context: dict[str, str] | None = None,
) -> None:
    """Enable local capture for supported SDKs already available in this environment.

    Pass ``parent_context=agentlens.get_trace_context()`` from a parent process to
    stitch sub-agent runs into the parent trace.
    """
    _config["api_key"] = api_key
    if parent_context and parent_context.get("parent_run_id"):
        _config["parent_run_id"] = parent_context["parent_run_id"]
    _patch_anthropic()
    _patch_anthropic_async()
    _patch_openai()
    _patch_openai_async()


def run(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Group all captured spans inside the decorated function into one saved run."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                run_data = start_run(name=name, parent_run_id=_config.get("parent_run_id"))
                token = _current_run.set(run_data)
                try:
                    result = await func(*args, **kwargs)
                    run_data["status"] = "success"
                    return result
                except Exception as exc:
                    run_data["status"] = "error"
                    run_data["error"] = str(exc)
                    capture_error(exc, context={"function": func.__name__})
                    raise
                finally:
                    _finalize_run(run_data)
                    _current_run.reset(token)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            run_data = start_run(name=name, parent_run_id=_config.get("parent_run_id"))
            token = _current_run.set(run_data)
            try:
                result = func(*args, **kwargs)
                run_data["status"] = "success"
                return result
            except Exception as exc:
                run_data["status"] = "error"
                run_data["error"] = str(exc)
                capture_error(exc, context={"function": func.__name__})
                raise
            finally:
                _finalize_run(run_data)
                _current_run.reset(token)

        return sync_wrapper

    return decorator


def start_run(name: str = "default", parent_run_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "name": name,
        "started_at": _now_iso(),
        "ended_at": None,
        "status": "running",
        "spans": [],
    }
    if parent_run_id:
        data["parent_run_id"] = parent_run_id
    return data


def get_trace_context() -> dict[str, str] | None:
    """Return a propagation context dict for passing to sub-agents / child runs.

    Usage::

        ctx = agentlens.get_trace_context()
        # pass ctx to the child process / service
        agentlens.init(parent_context=ctx)
    """
    run_data = _current_run.get()
    if run_data is None:
        return None
    return {"parent_run_id": run_data["run_id"]}


def record_memory_snapshot(label: str, state: dict[str, Any]) -> None:
    """Capture a snapshot of the agent's memory / state at the current point in time.

    Usage::

        agentlens.record_memory_snapshot("after_lookup", {"customer": "alex", "status": "active"})
    """
    append_span(
        {
            "type": "memory_snapshot",
            "ts": _now_iso(),
            "label": label,
            "state": _to_jsonable(state),
        }
    )


def current_run() -> dict[str, Any]:
    run_data = _current_run.get()
    if run_data is None:
        run_data = start_run(name="default")
        _current_run.set(run_data)
    return run_data


def append_span(span: dict[str, Any]) -> dict[str, Any]:
    run_data = current_run()
    enriched = {
        "id": str(uuid.uuid4()),
        "run_id": run_data["run_id"],
        **span,
    }
    run_data["spans"].append(enriched)
    return enriched


def _find_open_tool_span(tool_use_id: str | None) -> dict[str, Any] | None:
    """Return the request span the monkeypatch already recorded for this tool call.

    The provider monkeypatches record a tool_call span with output=None when the
    model *requests* a tool. record_tool_result then carries the *result*. Without
    merging, one logical call produces two spans sharing a tool_use_id, which
    double-counts the (tool, input) signature and makes the classifier mistake a
    single wrong tool choice for a loop. Match on tool_use_id so a genuine loop
    (same tool+input, *different* ids) still produces distinct spans.
    """
    if not tool_use_id:
        return None
    for span in reversed(current_run().get("spans", [])):
        if (
            span.get("type") == "tool_call"
            and span.get("tool_use_id") == tool_use_id
            and span.get("output") is None
        ):
            return span
    return None


def record_tool_result(
    tool_name: str, output: Any, input: Any | None = None, tool_use_id: str | None = None
) -> None:
    output_json = _to_jsonable(output)
    existing = _find_open_tool_span(tool_use_id)
    if existing is not None:
        # Fill the result onto the request span the monkeypatch already recorded.
        existing["output"] = output_json
        if input is not None and existing.get("input") in (None, {}):
            existing["input"] = _to_jsonable(input)
        if not existing.get("tool_name"):
            existing["tool_name"] = tool_name
    else:
        append_span(
            {
                "type": "tool_call",
                "ts": _now_iso(),
                "tool_name": tool_name,
                "input": _to_jsonable(input),
                "output": output_json,
                "tool_use_id": tool_use_id,
            }
        )
    if _is_error_output(output_json):
        capture_error(
            error=_extract_error_message(output_json),
            context={
                "tool_name": tool_name,
                "input": _to_jsonable(input),
                "output": output_json,
                "tool_use_id": tool_use_id,
            },
        )


def save_run(path: str | None = None, run: dict[str, Any] | None = None) -> Path:
    run_data = run if run is not None else current_run()
    if run_data.get("ended_at") is None:
        run_data["ended_at"] = _now_iso()
    if run_data.get("status") == "running":
        run_data["status"] = "success"

    output_path = Path(path) if path else RUNS_DIR / f"{run_data['run_id']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(run_data, indent=2), encoding="utf-8")
    return output_path


def _finalize_run(run_data: dict[str, Any]) -> None:
    if run_data["status"] == "success" and _run_has_error_span(run_data):
        run_data["status"] = "error"
    run_data["ended_at"] = _now_iso()
    save_run(run=run_data)


def capture_error(error: Exception | str, context: Any, latency_ms: float | None = None) -> None:
    append_span(
        {
            "type": "error",
            "ts": _now_iso(),
            "step_index": len(current_run()["spans"]) + 1,
            "latency_ms": latency_ms,
            "error": str(error),
            "context": _to_jsonable(context),
        }
    )


def capture_tool_results_from_messages(messages: Any, provider: str) -> None:
    run_data = current_run()
    for message in _as_list(messages):
        for block in _as_list(_get_value(message, "content")):
            if _get_value(block, "type") != "tool_result":
                continue
            tool_use_id = _get_value(block, "tool_use_id")
            # Anthropic tool_result blocks carry no "name" field — resolve via matching llm_call span
            tool_name = (
                _get_value(block, "name")
                or _resolve_tool_name_from_run(tool_use_id, run_data)
                or provider
            )
            record_tool_result(
                tool_name=tool_name,
                input=None,
                output=_get_value(block, "content"),
                tool_use_id=tool_use_id,
            )


def _resolve_tool_name_from_run(tool_use_id: str | None, run_data: dict[str, Any]) -> str | None:
    """Look backwards through llm_call spans to find the tool name for a given tool_use_id."""
    if not tool_use_id:
        return None
    for span in reversed(run_data.get("spans", [])):
        if span.get("type") != "llm_call":
            continue
        for block in _as_list(span.get("response_content")):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                return block.get("name")
    return None


def capture_anthropic_tool_calls(response_content: Any) -> None:
    for block in _as_list(response_content):
        if _get_value(block, "type") != "tool_use":
            continue
        append_span(
            {
                "type": "tool_call",
                "ts": _now_iso(),
                "tool_name": _get_value(block, "name"),
                "input": _to_jsonable(_get_value(block, "input")),
                "output": None,
                "tool_use_id": _get_value(block, "id"),
            }
        )


def capture_openai_tool_calls(response: Any) -> None:
    response_json = _to_jsonable(response)

    for tool_call in _find_tool_calls(response_json):
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        append_span(
            {
                "type": "tool_call",
                "ts": _now_iso(),
                "tool_name": function.get("name") or tool_call.get("name"),
                "input": _parse_maybe_json(function.get("arguments") or tool_call.get("input")),
                "output": None,
                "tool_use_id": tool_call.get("id"),
            }
        )


def _patch_anthropic() -> None:
    try:
        import anthropic
    except ImportError:
        return

    if not hasattr(anthropic, "Anthropic"):
        return

    if "anthropic" in _config["patched"]:
        return

    original_anthropic = anthropic.Anthropic
    _config["originals"]["anthropic.Anthropic"] = original_anthropic

    class AgentLensAnthropic:
        def __init__(self, *args: Any, **kwargs: Any):
            self._agentlens_client = original_anthropic(*args, **kwargs)
            self.messages = _AnthropicMessagesProxy(self._agentlens_client.messages)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._agentlens_client, name)

    anthropic.Anthropic = AgentLensAnthropic
    _config["patched"].add("anthropic")


def _patch_anthropic_async() -> None:
    """Patch anthropic.AsyncAnthropic so async agents are captured."""
    try:
        import anthropic
    except ImportError:
        return

    if not hasattr(anthropic, "AsyncAnthropic"):
        return

    if "anthropic_async" in _config["patched"]:
        return

    original_async = anthropic.AsyncAnthropic
    _config["originals"]["anthropic.AsyncAnthropic"] = original_async

    class AgentLensAsyncAnthropic:
        def __init__(self, *args: Any, **kwargs: Any):
            self._agentlens_client = original_async(*args, **kwargs)
            self.messages = _AsyncAnthropicMessagesProxy(self._agentlens_client.messages)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._agentlens_client, name)

    anthropic.AsyncAnthropic = AgentLensAsyncAnthropic
    _config["patched"].add("anthropic_async")


class _AnthropicMessagesProxy:
    def __init__(self, messages: Any):
        self._messages = messages

    def create(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        capture_tool_results_from_messages(kwargs.get("messages", []), provider="anthropic")
        try:
            response = self._messages.create(**kwargs)
            response_content = _to_jsonable(getattr(response, "content", None))
            usage = _to_jsonable(getattr(response, "usage", None))
            model = kwargs.get("model")
            append_span(
                {
                    "type": "llm_call",
                    "provider": "anthropic",
                    "ts": _now_iso(),
                    "latency_ms": _elapsed_ms(started),
                    "input_messages": _to_jsonable(kwargs.get("messages", [])),
                    "tools": _to_jsonable(kwargs.get("tools", [])),
                    "model": model,
                    "response_content": response_content,
                    "stop_reason": getattr(response, "stop_reason", None),
                    "usage": usage,
                    "cost_usd": compute_cost_usd(model, usage if isinstance(usage, dict) else {}),
                }
            )
            capture_anthropic_tool_calls(response_content)
            return response
        except Exception as exc:
            capture_error(exc, context=_anthropic_context(kwargs), latency_ms=_elapsed_ms(started))
            raise

    def stream(self, **kwargs: Any) -> "_AnthropicStreamContext":
        """Wrap Anthropic streaming so the span is captured when the stream closes."""
        started = time.perf_counter()
        capture_tool_results_from_messages(kwargs.get("messages", []), provider="anthropic")
        try:
            ctx = self._messages.stream(**kwargs)
        except Exception as exc:
            capture_error(exc, context=_anthropic_context(kwargs), latency_ms=_elapsed_ms(started))
            raise
        return _AnthropicStreamContext(ctx, kwargs, started)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class _AnthropicStreamContext:
    """Context manager wrapper that saves a span when Anthropic streaming completes."""

    def __init__(self, ctx: Any, kwargs: dict[str, Any], started: float) -> None:
        self._ctx = ctx
        self._kwargs = kwargs
        self._started = started
        self._stream: Any = None

    def __enter__(self) -> Any:
        self._stream = self._ctx.__enter__()
        return self._stream

    def __exit__(self, *args: Any) -> Any:
        result = self._ctx.__exit__(*args)
        self._save_span()
        return result

    def _save_span(self) -> None:
        if self._stream is None:
            return
        try:
            message = getattr(self._stream, "get_final_message", lambda: None)()
            if message is None:
                return
            response_content = _to_jsonable(getattr(message, "content", None))
            usage = _to_jsonable(getattr(message, "usage", None))
            model = self._kwargs.get("model")
            append_span(
                {
                    "type": "llm_call",
                    "provider": "anthropic",
                    "ts": _now_iso(),
                    "latency_ms": _elapsed_ms(self._started),
                    "input_messages": _to_jsonable(self._kwargs.get("messages", [])),
                    "tools": _to_jsonable(self._kwargs.get("tools", [])),
                    "model": model,
                    "response_content": response_content,
                    "stop_reason": getattr(message, "stop_reason", None),
                    "usage": usage,
                    "cost_usd": compute_cost_usd(model, usage if isinstance(usage, dict) else {}),
                    "streaming": True,
                }
            )
            capture_anthropic_tool_calls(response_content)
        except Exception:
            pass


def _patch_openai() -> None:
    try:
        import openai
    except ImportError:
        return

    if not hasattr(openai, "OpenAI"):
        return

    if "openai" in _config["patched"]:
        return

    original_openai = openai.OpenAI
    _config["originals"]["openai.OpenAI"] = original_openai

    class AgentLensOpenAI:
        def __init__(self, *args: Any, **kwargs: Any):
            self._agentlens_client = original_openai(*args, **kwargs)
            self.chat = _OpenAIChatProxy(self._agentlens_client.chat)
            if hasattr(self._agentlens_client, "responses"):
                self.responses = _OpenAIResponsesProxy(self._agentlens_client.responses)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._agentlens_client, name)

    openai.OpenAI = AgentLensOpenAI
    _config["patched"].add("openai")


def _patch_openai_async() -> None:
    """Patch openai.AsyncOpenAI so async agents are captured."""
    try:
        import openai
    except ImportError:
        return

    if not hasattr(openai, "AsyncOpenAI"):
        return

    if "openai_async" in _config["patched"]:
        return

    original_async = openai.AsyncOpenAI
    _config["originals"]["openai.AsyncOpenAI"] = original_async

    class AgentLensAsyncOpenAI:
        def __init__(self, *args: Any, **kwargs: Any):
            self._agentlens_client = original_async(*args, **kwargs)
            self.chat = _AsyncOpenAIChatProxy(self._agentlens_client.chat)
            if hasattr(self._agentlens_client, "responses"):
                self.responses = _AsyncOpenAIResponsesProxy(self._agentlens_client.responses)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._agentlens_client, name)

    openai.AsyncOpenAI = AgentLensAsyncOpenAI
    _config["patched"].add("openai_async")


class _OpenAIChatProxy:
    def __init__(self, chat: Any):
        self._chat = chat
        self.completions = _OpenAICompletionsProxy(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _OpenAICompletionsProxy:
    def __init__(self, completions: Any):
        self._completions = completions

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return self._create_stream(**kwargs)
        started = time.perf_counter()
        try:
            response = self._completions.create(**kwargs)
            response_json = _to_jsonable(response)
            usage = response_json.get("usage") if isinstance(response_json, dict) else None
            model = kwargs.get("model")
            append_span(
                {
                    "type": "llm_call",
                    "provider": "openai",
                    "ts": _now_iso(),
                    "latency_ms": _elapsed_ms(started),
                    "input_messages": _to_jsonable(kwargs.get("messages", [])),
                    "tools": _to_jsonable(kwargs.get("tools", kwargs.get("functions", []))),
                    "model": model,
                    "response_content": response_json,
                    "stop_reason": _first_choice_stop_reason(response_json),
                    "usage": usage,
                    "cost_usd": compute_cost_usd(model, usage if isinstance(usage, dict) else {}),
                }
            )
            capture_openai_tool_calls(response_json)
            return response
        except Exception as exc:
            capture_error(exc, context=_openai_context(kwargs), latency_ms=_elapsed_ms(started))
            raise

    def _create_stream(self, **kwargs: Any) -> "_OpenAIStreamWrapper":
        started = time.perf_counter()
        try:
            raw_stream = self._completions.create(**kwargs)
        except Exception as exc:
            capture_error(exc, context=_openai_context(kwargs), latency_ms=_elapsed_ms(started))
            raise
        return _OpenAIStreamWrapper(raw_stream, kwargs, started)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


def _accumulate_openai_stream(
    chunks: list[Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """Rebuild a non-streaming-shaped response from streamed chunks.

    Streaming deltas carry text in ``delta.content`` and tool calls in
    ``delta.tool_calls`` (whose ``arguments`` arrive in fragments keyed by
    ``index``). Earlier this only kept text, so a streamed tool-call response
    produced an empty response_content and no captured tool request. We
    reassemble both into the same {choices:[{message:{...}}]} shape the
    non-streaming path emits so downstream preprocessing and tool capture are
    uniform across streaming and non-streaming.
    """
    text_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None

    for chunk in chunks:
        chunk_json = _to_jsonable(chunk)
        if not isinstance(chunk_json, dict):
            continue
        if chunk_json.get("usage"):
            usage = chunk_json["usage"]
        for choice in chunk_json.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                stop_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            if delta.get("content"):
                text_parts.append(str(delta["content"]))
            for tool_call in delta.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                index = tool_call.get("index", 0)
                slot = tool_calls_by_index.setdefault(
                    index,
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tool_call.get("id"):
                    slot["id"] = tool_call["id"]
                function = tool_call.get("function") or {}
                if function.get("name"):
                    slot["function"]["name"] = function["name"]
                if function.get("arguments"):
                    slot["function"]["arguments"] += function["arguments"]

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if tool_calls_by_index:
        message["tool_calls"] = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]

    response_json = {
        "choices": [{"message": message, "finish_reason": stop_reason}],
        "usage": usage,
    }
    return response_json, usage, stop_reason


class _OpenAIStreamWrapper:
    """Wraps an OpenAI streaming response and saves a span on completion."""

    def __init__(self, stream: Any, kwargs: dict[str, Any], started: float) -> None:
        self._stream = stream
        self._kwargs = kwargs
        self._started = started
        self._chunks: list[Any] = []
        self._finalized = False
        self._iter: Any = None

    def __iter__(self) -> Any:
        self._iter = iter(self._stream)
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iter)
            self._chunks.append(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def __enter__(self) -> Any:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        result = None
        if hasattr(self._stream, "__exit__"):
            result = self._stream.__exit__(*args)
        self._finalize()
        return result

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            response_json, usage, stop_reason = _accumulate_openai_stream(self._chunks)
            model = self._kwargs.get("model")
            append_span(
                {
                    "type": "llm_call",
                    "provider": "openai",
                    "ts": _now_iso(),
                    "latency_ms": _elapsed_ms(self._started),
                    "input_messages": _to_jsonable(self._kwargs.get("messages", [])),
                    "tools": _to_jsonable(self._kwargs.get("tools", self._kwargs.get("functions", []))),
                    "model": model,
                    "response_content": response_json,
                    "stop_reason": stop_reason,
                    "usage": _to_jsonable(usage),
                    "cost_usd": compute_cost_usd(model, usage if isinstance(usage, dict) else {}),
                    "streaming": True,
                    "chunk_count": len(self._chunks),
                }
            )
            # Capture tool requests the same way the non-streaming path does, so
            # streamed tool calls are recorded even when the agent does not call
            # record_tool_result explicitly, and the dedup merge has a request
            # span to fill.
            capture_openai_tool_calls(response_json)
        except Exception:
            pass


class _OpenAIResponsesProxy:
    def __init__(self, responses: Any):
        self._responses = responses

    def create(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            response = self._responses.create(**kwargs)
            response_json = _to_jsonable(response)
            usage = response_json.get("usage") if isinstance(response_json, dict) else None
            model = kwargs.get("model")
            append_span(
                {
                    "type": "llm_call",
                    "provider": "openai",
                    "ts": _now_iso(),
                    "latency_ms": _elapsed_ms(started),
                    "input_messages": _to_jsonable(kwargs.get("input")),
                    "tools": _to_jsonable(kwargs.get("tools", [])),
                    "model": model,
                    "response_content": response_json,
                    "stop_reason": response_json.get("status") if isinstance(response_json, dict) else None,
                    "usage": usage,
                    "cost_usd": compute_cost_usd(model, usage if isinstance(usage, dict) else {}),
                }
            )
            capture_openai_tool_calls(response_json)
            return response
        except Exception as exc:
            capture_error(exc, context=_openai_context(kwargs), latency_ms=_elapsed_ms(started))
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._responses, name)


class _AsyncAnthropicMessagesProxy:
    """Async proxy for anthropic.AsyncAnthropic().messages — mirrors _AnthropicMessagesProxy."""

    def __init__(self, messages: Any) -> None:
        self._messages = messages

    async def create(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        capture_tool_results_from_messages(kwargs.get("messages", []), provider="anthropic")
        try:
            response = await self._messages.create(**kwargs)
            response_content = _to_jsonable(getattr(response, "content", None))
            usage = _to_jsonable(getattr(response, "usage", None))
            model = kwargs.get("model")
            append_span(
                {
                    "type": "llm_call",
                    "provider": "anthropic",
                    "ts": _now_iso(),
                    "latency_ms": _elapsed_ms(started),
                    "input_messages": _to_jsonable(kwargs.get("messages", [])),
                    "tools": _to_jsonable(kwargs.get("tools", [])),
                    "model": model,
                    "response_content": response_content,
                    "stop_reason": getattr(response, "stop_reason", None),
                    "usage": usage,
                    "cost_usd": compute_cost_usd(model, usage if isinstance(usage, dict) else {}),
                    "async": True,
                }
            )
            capture_anthropic_tool_calls(response_content)
            return response
        except Exception as exc:
            capture_error(exc, context=_anthropic_context(kwargs), latency_ms=_elapsed_ms(started))
            raise

    def stream(self, **kwargs: Any) -> "_AsyncAnthropicStreamContext":
        """Return an async context manager for streaming — supports:
            async with client.messages.stream(...) as s:   (standard Anthropic pattern)
        """
        started = time.perf_counter()
        capture_tool_results_from_messages(kwargs.get("messages", []), provider="anthropic")
        try:
            ctx = self._messages.stream(**kwargs)
        except Exception as exc:
            capture_error(exc, context=_anthropic_context(kwargs), latency_ms=_elapsed_ms(started))
            raise
        return _AsyncAnthropicStreamContext(ctx, kwargs, started)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class _AsyncAnthropicStreamContext:
    """Async context manager wrapper that saves a span when async Anthropic streaming completes.

    Supports:
        async with client.messages.stream(...) as s:
            async for chunk in s: ...
    """

    def __init__(self, ctx: Any, kwargs: dict[str, Any], started: float) -> None:
        self._ctx = ctx
        self._kwargs = kwargs
        self._started = started
        self._stream: Any = None

    async def __aenter__(self) -> Any:
        self._stream = await self._ctx.__aenter__()
        return self._stream

    async def __aexit__(self, *args: Any) -> Any:
        result = await self._ctx.__aexit__(*args)
        await self._save_span()
        return result

    async def _save_span(self) -> None:
        if self._stream is None:
            return
        try:
            get_final = getattr(self._stream, "get_final_message", None)
            message = await get_final() if get_final and asyncio.iscoroutinefunction(get_final) else (get_final() if get_final else None)
            if message is None:
                return
            response_content = _to_jsonable(getattr(message, "content", None))
            usage = _to_jsonable(getattr(message, "usage", None))
            model = self._kwargs.get("model")
            append_span(
                {
                    "type": "llm_call",
                    "provider": "anthropic",
                    "ts": _now_iso(),
                    "latency_ms": _elapsed_ms(self._started),
                    "input_messages": _to_jsonable(self._kwargs.get("messages", [])),
                    "tools": _to_jsonable(self._kwargs.get("tools", [])),
                    "model": model,
                    "response_content": response_content,
                    "stop_reason": getattr(message, "stop_reason", None),
                    "usage": usage,
                    "cost_usd": compute_cost_usd(model, usage if isinstance(usage, dict) else {}),
                    "streaming": True,
                    "async": True,
                }
            )
            capture_anthropic_tool_calls(response_content)
        except Exception:
            pass


class _AsyncOpenAIChatProxy:
    def __init__(self, chat: Any) -> None:
        self._chat = chat
        self.completions = _AsyncOpenAICompletionsProxy(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _AsyncOpenAICompletionsProxy:
    """Async proxy for openai.AsyncOpenAI().chat.completions."""

    def __init__(self, completions: Any) -> None:
        self._completions = completions

    async def create(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            response = await self._completions.create(**kwargs)
            response_json = _to_jsonable(response)
            usage = response_json.get("usage") if isinstance(response_json, dict) else None
            model = kwargs.get("model")
            append_span(
                {
                    "type": "llm_call",
                    "provider": "openai",
                    "ts": _now_iso(),
                    "latency_ms": _elapsed_ms(started),
                    "input_messages": _to_jsonable(kwargs.get("messages", [])),
                    "tools": _to_jsonable(kwargs.get("tools", kwargs.get("functions", []))),
                    "model": model,
                    "response_content": response_json,
                    "stop_reason": _first_choice_stop_reason(response_json),
                    "usage": usage,
                    "cost_usd": compute_cost_usd(model, usage if isinstance(usage, dict) else {}),
                    "async": True,
                }
            )
            capture_openai_tool_calls(response_json)
            return response
        except Exception as exc:
            capture_error(exc, context=_openai_context(kwargs), latency_ms=_elapsed_ms(started))
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _AsyncOpenAIResponsesProxy:
    """Async proxy for openai.AsyncOpenAI().responses."""

    def __init__(self, responses: Any) -> None:
        self._responses = responses

    async def create(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            response = await self._responses.create(**kwargs)
            response_json = _to_jsonable(response)
            usage = response_json.get("usage") if isinstance(response_json, dict) else None
            model = kwargs.get("model")
            append_span(
                {
                    "type": "llm_call",
                    "provider": "openai",
                    "ts": _now_iso(),
                    "latency_ms": _elapsed_ms(started),
                    "input_messages": _to_jsonable(kwargs.get("input")),
                    "tools": _to_jsonable(kwargs.get("tools", [])),
                    "model": model,
                    "response_content": response_json,
                    "stop_reason": response_json.get("status") if isinstance(response_json, dict) else None,
                    "usage": usage,
                    "cost_usd": compute_cost_usd(model, usage if isinstance(usage, dict) else {}),
                    "async": True,
                }
            )
            capture_openai_tool_calls(response_json)
            return response
        except Exception as exc:
            capture_error(exc, context=_openai_context(kwargs), latency_ms=_elapsed_ms(started))
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._responses, name)


def load_runs() -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for path in sorted(RUNS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return runs


def load_run(run_id: str) -> dict[str, Any] | None:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        matches = [run for run in load_runs() if run.get("run_id", "").startswith(run_id)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            match_ids = [str(run.get("run_id", "")) for run in matches if run.get("run_id")]
            raise AmbiguousRunIdError(run_id, match_ids)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _anthropic_context(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": "anthropic",
        "input_messages": _to_jsonable(kwargs.get("messages", [])),
        "model": kwargs.get("model"),
        "tools": _to_jsonable(kwargs.get("tools", [])),
    }


def _openai_context(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": "openai",
        "input_messages": _to_jsonable(kwargs.get("messages", kwargs.get("input"))),
        "model": kwargs.get("model"),
        "tools": _to_jsonable(kwargs.get("tools", kwargs.get("functions", []))),
    }


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _is_error_output(output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    if output.get("status") == "error":
        return True
    # Only flag "error" key if its value is truthy — {"error": None} is not an error
    error_val = output.get("error")
    return error_val is not None and bool(error_val)


def _extract_error_message(output: Any) -> str:
    if isinstance(output, dict):
        return str(output.get("error") or output)
    return str(output)


def _parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return _to_jsonable(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _find_tool_calls(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    _collect_tool_calls(value, found)
    # Deduplicate by tool call id — nested OpenAI response structures surface
    # the same call at multiple levels (e.g. top-level and inside choices[].message)
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in found:
        call_id = item.get("id")
        if call_id:
            if call_id in seen_ids:
                continue
            seen_ids.add(call_id)
        deduped.append(item)
    return deduped


def _collect_tool_calls(value: Any, found: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        maybe_calls = value.get("tool_calls")
        if isinstance(maybe_calls, list):
            found.extend(item for item in maybe_calls if isinstance(item, dict))
        # Skip the tool_calls key itself to avoid recursing into calls we already collected
        for key, item in value.items():
            if key != "tool_calls":
                _collect_tool_calls(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_tool_calls(item, found)


def _first_choice_stop_reason(response_json: Any) -> Any:
    if not isinstance(response_json, dict):
        return None
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        return choices[0].get("finish_reason")
    return None


def _run_has_error_span(run_data: dict[str, Any]) -> bool:
    return any(span.get("type") == "error" for span in run_data.get("spans", []))
