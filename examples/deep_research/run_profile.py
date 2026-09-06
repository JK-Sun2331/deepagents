"""Run one profiled Deep Research workflow and save compact local events."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.messages import HumanMessage

from agent import (
    current_date,
    max_concurrent_research_units,
    max_researcher_iterations,
)
from agent import agent as research_agent
from research_agent.profiling import ProfileRecorder
from research_agent.tools import MAX_CONTENT_CHARS, MAX_SUCCESSFUL_SOURCES


DEFAULT_QUERY = "Compare the architectures of vLLM and SGLang, using primary sources."
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = (
    WORKSPACE / "runtime-data" / "motivation1-deep-research-profiling" / "runs"
)


def _query_id(query: str) -> str:
    words = ["".join(character for character in word.lower() if character.isalnum()) for word in query.split()]
    slug = "-".join(word for word in words if word)[:60].strip("-")
    return slug or "query"


def _default_output_dir(query: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUN_ROOT / f"{timestamp}_{_query_id(query)}"


def _git_snapshot(repository: Path) -> dict[str, object]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "path": str(repository),
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty_files": run("status", "--short").splitlines(),
    }


def _versions() -> dict[str, str]:
    packages = ("deepagents", "langchain", "langchain-core", "langgraph", "langchain-openai")
    return {package: importlib.metadata.version(package) for package in packages}


def _environment() -> dict[str, str | bool | None]:
    return {
        "LOCAL_LLM_BASE_URL": os.getenv("LOCAL_LLM_BASE_URL"),
        "LOCAL_LLM_MODEL": os.getenv("LOCAL_LLM_MODEL"),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        "SLACKKV_SERVICE_MAX_MODEL_LEN": os.getenv(
            "SLACKKV_SERVICE_MAX_MODEL_LEN"
        ),
        "SLACKKV_SERVICE_ROPE_SCALING": os.getenv(
            "SLACKKV_SERVICE_ROPE_SCALING"
        ),
        "SLACKKV_SERVICE_GPU_MEMORY_UTILIZATION": os.getenv(
            "SLACKKV_SERVICE_GPU_MEMORY_UTILIZATION"
        ),
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING", "").lower()
        in {"1", "true", "yes"},
    }


def _metadata(query: str, output_dir: Path, workflow_id: str) -> dict[str, object]:
    return {
        "workflow_id": workflow_id,
        "query": query,
        "query_id": _query_id(query),
        "output_dir": str(output_dir),
        "command": [sys.executable, *sys.argv],
        "python_version": sys.version.split()[0],
        "package_versions": _versions(),
        "repositories": {
            "deepagents": _git_snapshot(WORKSPACE / "deepagents"),
            "slackkv-control": _git_snapshot(WORKSPACE / "slackkv-control"),
        },
        "environment": _environment(),
        "agent_config": {
            "prompt_date": current_date,
            "max_concurrent_research_units": max_concurrent_research_units,
            "max_researcher_iterations": max_researcher_iterations,
            "max_content_chars": MAX_CONTENT_CHARS,
            "max_successful_sources": MAX_SUCCESSFUL_SOURCES,
            "model_temperature": 0.0,
            "model_max_tokens": 8192,
            "model_timeout_seconds": 180,
            "model_max_retries": 2,
        },
    }


def _report_from_result(result: dict[str, Any]) -> str:
    files = result.get("files") or {}
    report = files.get("/final_report.md") or files.get("final_report.md")
    if isinstance(report, str):
        return report
    if isinstance(report, dict):
        content = report.get("content")
        if isinstance(content, str):
            return content
    messages = result.get("messages") or []
    if messages:
        return str(messages[-1].content)
    return ""


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(query: str, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=False)
    workflow_id = str(uuid4())
    recorder = ProfileRecorder(workflow_id)
    metadata = _metadata(query, output_dir, workflow_id)
    metadata.update(
        {
            "started_at_utc": recorder.utc_anchor,
            "mono_anchor_ns": recorder.mono_anchor_ns,
            "status": "running",
        }
    )
    _write_metadata(output_dir / "metadata.json", metadata)
    started = time.perf_counter_ns()
    recorder.record_workflow("workflow_start", status="running")
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    try:
        result = research_agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "callbacks": [recorder],
                "metadata": {"workflow_id": workflow_id},
            },
        )
    except BaseException as exc:
        error = exc
    finally:
        status = "success" if error is None else "error"
        recorder.record_workflow(
            "workflow_end" if error is None else "workflow_error",
            status=status,
            error_type=type(error).__name__ if error else None,
        )
        duration_ms = (time.perf_counter_ns() - started) / 1_000_000
        write_metrics = recorder.write_jsonl(output_dir / "events.jsonl")
        metadata.update(
            {
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "workflow_duration_ms": duration_ms,
                "status": status,
                **write_metrics,
            }
        )
        _write_metadata(output_dir / "metadata.json", metadata)
        log_lines = [
            f"status={status}",
            f"workflow_id={workflow_id}",
            f"workflow_duration_ms={duration_ms:.3f}",
            f"event_count={write_metrics['event_count']}",
        ]
        if error:
            log_lines.append(f"error_type={type(error).__name__}")
        (output_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if error is not None:
        raise error
    report = _report_from_result(result or {})
    (output_dir / "final_report.md").write_text(report, encoding="utf-8")
    print(f"Profile saved to {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir or _default_output_dir(arguments.query)
    return run(arguments.query, output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
