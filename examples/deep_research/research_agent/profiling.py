"""Local callback recorder for Deep Research workflow profiling."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult


_METADATA_KEYS = (
    "langgraph_node",
    "langgraph_step",
    "langgraph_checkpoint_ns",
    "checkpoint_ns",
    "lc_agent_name",
    "ls_agent_type",
)


def _json_chars(value: object) -> int:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return len(encoded)


def _content_chars(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if isinstance(block, str):
            total += len(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                total += len(text)
    return total


def _tool_calls(message: BaseMessage) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for call in getattr(message, "tool_calls", []) or []:
        calls.append(
            {
                "id": str(call.get("id", "")),
                "name": str(call.get("name", "")),
                "args_chars": _json_chars(
                    {"name": call.get("name", ""), "args": call.get("args", {})}
                ),
            }
        )
    return calls


def _message_stats(messages: list[list[BaseMessage]]) -> dict[str, int]:
    flat = [message for batch in messages for message in batch]
    return {
        "input_message_count": len(flat),
        "input_content_chars": sum(_content_chars(message.content) for message in flat),
        "input_tool_call_chars": sum(
            call["args_chars"]
            for message in flat
            for call in _tool_calls(message)
            if isinstance(call["args_chars"], int)
        ),
    }


def _metadata_fields(metadata: dict[str, Any] | None) -> dict[str, object]:
    source = metadata or {}
    fields: dict[str, object] = {}
    for key in _METADATA_KEYS:
        value = source.get(key)
        if isinstance(value, str | int | float | bool) or value is None:
            fields[key] = value
    return {
        **fields,
        "node_name": source.get("langgraph_node"),
        "graph_step": source.get("langgraph_step"),
        "graph_namespace": source.get("langgraph_checkpoint_ns")
        or source.get("checkpoint_ns"),
        "agent_name": source.get("lc_agent_name"),
    }


def _run_fields(
    run_id: UUID,
    parent_run_id: UUID | None,
    metadata: dict[str, Any] | None,
) -> dict[str, object]:
    return {
        "run_id": str(run_id),
        "parent_run_id": str(parent_run_id) if parent_run_id else None,
        **_metadata_fields(metadata),
    }


def _error_summary(error: BaseException) -> str:
    message = str(error).replace("\n", " ")
    message = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", message, flags=re.I)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_-]?key\s*[=:]\s*)[^,;\s]+",
        r"\1[REDACTED]",
        message,
    )
    return message[:500]


def _usage_from_result(response: LLMResult) -> tuple[dict[str, int | None], str]:
    usage_messages = []
    for generations in response.generations:
        if generations:
            message = getattr(generations[0], "message", None)
            usage = getattr(message, "usage_metadata", None)
            if usage:
                usage_messages.append(usage)
    if usage_messages:
        return (
            {
                key: sum(int(usage.get(key, 0)) for usage in usage_messages)
                for key in ("input_tokens", "output_tokens", "total_tokens")
            },
            "usage_metadata",
        )
    token_usage = (response.llm_output or {}).get("token_usage") or {}
    if token_usage:
        return (
            {
                "input_tokens": token_usage.get(
                    "input_tokens", token_usage.get("prompt_tokens")
                ),
                "output_tokens": token_usage.get(
                    "output_tokens", token_usage.get("completion_tokens")
                ),
                "total_tokens": token_usage.get("total_tokens"),
            },
            "llm_output.token_usage",
        )
    return (
        {"input_tokens": None, "output_tokens": None, "total_tokens": None},
        "missing",
    )


def _output_fields(response: LLMResult) -> dict[str, object]:
    messages = [
        getattr(generations[0], "message", None)
        for generations in response.generations
        if generations
    ]
    messages = [message for message in messages if isinstance(message, BaseMessage)]
    calls = [call for message in messages for call in _tool_calls(message)]
    usage, source = _usage_from_result(response)
    message_ids = [str(message.id) for message in messages if message.id]
    return {
        **usage,
        "usage_source": source,
        "output_content_chars": sum(
            _content_chars(message.content) for message in messages
        ),
        "output_tool_call_chars": sum(
            int(call["args_chars"]) for call in calls
        ),
        "tool_calls": calls,
        "message_id": message_ids[0] if message_ids else None,
        "message_ids": message_ids,
    }


class ProfileRecorder(BaseCallbackHandler):
    """Collect compact callback events without serializing message bodies."""

    run_inline = True
    raise_error = True

    def __init__(self, workflow_id: str) -> None:
        self.workflow_id = workflow_id
        self.utc_anchor = datetime.now(timezone.utc).isoformat()
        self.mono_anchor_ns = time.perf_counter_ns()
        self._events: list[dict[str, object]] = []
        self._lock = Lock()

    @property
    def events(self) -> list[dict[str, object]]:
        """Return a stable copy of recorded events."""
        with self._lock:
            return [event.copy() for event in self._events]

    def record_workflow(self, event_type: str, **fields: object) -> None:
        """Record a workflow boundary outside the callback lifecycle."""
        self._append(event_type, **fields)

    def _append(
        self,
        event_type: str,
        *,
        ts_mono_ns: int | None = None,
        **fields: object,
    ) -> None:
        timestamp = ts_mono_ns if ts_mono_ns is not None else time.perf_counter_ns()
        with self._lock:
            self._events.append(
                {
                    "workflow_id": self.workflow_id,
                    "event_seq": len(self._events) + 1,
                    "event_type": event_type,
                    "ts_mono_ns": timestamp,
                    **fields,
                }
            )

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        timestamp = time.perf_counter_ns()
        self._append(
            "chain_start",
            ts_mono_ns=timestamp,
            **_run_fields(run_id, parent_run_id, metadata),
            span_name=(serialized or {}).get("name") or kwargs.get("name"),
            status="running",
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._append(
            "chain_end",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            status="success",
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._append(
            "chain_error",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            status="error",
            error_type=type(error).__name__,
            error_summary=_error_summary(error),
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        timestamp = time.perf_counter_ns()
        self._append(
            "model_start",
            ts_mono_ns=timestamp,
            **_run_fields(run_id, parent_run_id, metadata),
            model_call_id=str(run_id),
            model_name=serialized.get("name") or kwargs.get("name"),
            status="running",
            **_message_stats(messages),
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._append(
            "model_end",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            model_call_id=str(run_id),
            status="success",
            **_output_fields(response),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._append(
            "model_error",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            model_call_id=str(run_id),
            status="error",
            error_type=type(error).__name__,
            error_summary=_error_summary(error),
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        timestamp = time.perf_counter_ns()
        tool_name = (serialized or {}).get("name") or kwargs.get("name")
        task_fields: dict[str, object] = {}
        if tool_name == "task" and inputs:
            task_fields = {
                "subagent_type": inputs.get("subagent_type"),
                "task_description_chars": _content_chars(inputs.get("description")),
            }
        self._append(
            "tool_start",
            ts_mono_ns=timestamp,
            **_run_fields(run_id, parent_run_id, metadata),
            tool_name=tool_name,
            tool_call_id=kwargs.get("tool_call_id"),
            tool_input_chars=_json_chars(inputs) if inputs is not None else len(input_str),
            status="running",
            **task_fields,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        output_status = getattr(output, "status", None)
        status = "error" if output_status == "error" else "success"
        content = output.content if isinstance(output, ToolMessage) else output
        self._append(
            "tool_end",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            status=status,
            output_status=output_status,
            tool_output_chars=_content_chars(content),
            error_type="ToolMessageStatusError" if status == "error" else None,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._append(
            "tool_error",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            status="error",
            error_type=type(error).__name__,
            error_summary=_error_summary(error),
        )

    def write_jsonl(self, path: Path) -> dict[str, int | float]:
        """Write all compact events once and return write metrics."""
        path.parent.mkdir(parents=True, exist_ok=True)
        events = self.events
        payload = "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            for event in events
        )
        started = time.perf_counter_ns()
        path.write_text(payload, encoding="utf-8")
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return {
            "event_count": len(events),
            "events_file_bytes": path.stat().st_size,
            "events_write_ms": elapsed_ms,
        }
