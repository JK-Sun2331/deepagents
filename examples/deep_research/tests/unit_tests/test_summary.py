"""Exercise the explicit summary boundary through real agent graphs."""

import asyncio
import sys
import unittest
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research_agent"))
from prompts import SUMMARY_GENERATION_INSTRUCTIONS, SUMMARY_RESEARCH_INSTRUCTIONS
from summary import ResearchSummaryMiddleware

from deepagents import create_deep_agent


class ResearchModel(BaseChatModel):
    scenario: str = "normal"

    @property
    def _llm_type(self):
        return "summary-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        task = next(m.content for m in messages if isinstance(m, HumanMessage))
        if messages[0].content == SUMMARY_GENERATION_INSTRUCTIONS:
            evidence = [m.content for m in messages if isinstance(m, ToolMessage)]
            assert evidence == [f"Evidence for {task}"]
            text = f"Final findings for {task} [1]. Sources: https://example.test/{task}"
            if self.scenario == "empty":
                text = ""
            message = AIMessage(content=text, response_metadata={
                "finish_reason": "length" if self.scenario == "truncated" else "stop"})
        elif self.scenario == "direct":
            message = AIMessage(content="Bypassed summary")
        elif isinstance(messages[-1], HumanMessage):
            message = AIMessage(content="", tool_calls=[{
                "name": "search", "args": {"query": task}, "id": f"search-{task}"}])
        else:
            calls = [{"name": "summary", "args": {}, "id": f"summary-{task}"}]
            if self.scenario == "mixed":
                calls.append({"name": "search", "args": {"query": task}, "id": "extra"})
            message = AIMessage(content="Ready to finish", tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


class MainModel(ResearchModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            message = AIMessage(content="Main complete")
        else:
            message = AIMessage(content="", tool_calls=[{
                "name": "task", "args": {"description": task, "subagent_type": "research-agent"},
                "id": f"task-{task}"} for task in ("alpha", "beta")])
        return ChatResult(generations=[ChatGeneration(message=message)])


class Calls(BaseCallbackHandler):
    def __init__(self):
        self.phases = []
        self.tools = {}
        self.summary_parents = []

    def on_chat_model_start(self, serialized, messages, *, metadata=None, parent_run_id=None, **kwargs):
        phase = (metadata or {}).get("research_phase", "decision")
        self.phases.append(phase)
        if phase == "summary":
            self.summary_parents.append(parent_run_id)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.tools[run_id] = serialized["name"]


@tool
def search(query: str) -> str:
    """Return evidence for the current task."""
    return f"Evidence for {query}"


def middleware(model):
    return ResearchSummaryMiddleware(model, SUMMARY_RESEARCH_INSTRUCTIONS, SUMMARY_GENERATION_INSTRUCTIONS)


def branch(scenario="normal"):
    from langchain.agents import create_agent
    model = ResearchModel(scenario=scenario)
    return create_agent(model, tools=[search], middleware=[middleware(model)])


class SummaryTests(unittest.TestCase):
    def test_no_model_arguments(self):
        self.assertEqual(middleware(ResearchModel()).tools[0].tool_call_schema.model_json_schema()["properties"], {})

    def test_sync_summary_is_a_separate_final_call(self):
        recorder = Calls()
        result = branch().invoke({"messages": [HumanMessage(content="alpha")]},
                                 config={"callbacks": [recorder]})
        self.assertEqual(recorder.phases, ["decision", "decision", "summary"])
        self.assertEqual([recorder.tools[parent] for parent in recorder.summary_parents], ["summary"])
        self.assertTrue(result["messages"][-1].content.startswith("Final findings for alpha"))
        self.assertEqual(result["messages"][-2].name, "summary")

    def test_async_summary(self):
        recorder = Calls()
        result = asyncio.run(branch().ainvoke({"messages": [HumanMessage(content="beta")]},
                                             config={"callbacks": [recorder]}))
        self.assertTrue(result["messages"][-1].content.startswith("Final findings for beta"))
        self.assertEqual([recorder.tools[parent] for parent in recorder.summary_parents], ["summary"])

    def test_summary_keeps_effective_compacted_context(self):
        from langchain.agents import create_agent
        from langchain.agents.middleware import AgentMiddleware
        from deepagents.backends import StateBackend
        from deepagents.middleware.summarization import SummarizationMiddleware

        class SeedCompaction(AgentMiddleware):
            def before_agent(self, state, runtime):
                return {"_summarization_event": {"cutoff_index": 2,
                        "summary_message": HumanMessage(content="alpha"), "file_path": None}}

        model = ResearchModel()
        agent = create_agent(model, tools=[search], middleware=[
            SeedCompaction(),
            SummarizationMiddleware(model=model, backend=StateBackend(), trigger=("tokens", 100_000)),
            middleware(model)])
        result = agent.invoke({"messages": [HumanMessage(content="obsolete-history"),
                                            AIMessage(content="old answer")]})
        self.assertEqual(result["messages"][0].content, "obsolete-history")
        self.assertTrue(result["messages"][-1].content.startswith("Final findings for alpha"))

    def test_parallel_task_returns_and_context_isolation(self):
        model = ResearchModel()
        agent = create_deep_agent(model=MainModel(), subagents=[{
            "name": "research-agent", "description": "Research one task", "model": model,
            "tools": [search], "middleware": [middleware(model)], "system_prompt": "Research"}])
        for use_async in (False, True):
            with self.subTest(use_async=use_async):
                inputs = {"messages": [HumanMessage(content="compare")]}
                result = asyncio.run(agent.ainvoke(inputs)) if use_async else agent.invoke(inputs)
                outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual(len(outputs), 2)
                for name in ("alpha", "beta"):
                    output = next(m for m in outputs if m.tool_call_id == f"task-{name}")
                    self.assertTrue(output.content.startswith(f"Final findings for {name}"))
                    self.assertNotIn("Ready to finish", output.content)

    def test_invalid_completion_fails(self):
        for scenario in ("direct", "mixed", "empty", "truncated"):
            with self.subTest(scenario=scenario), self.assertRaises(RuntimeError):
                branch(scenario).invoke({"messages": [HumanMessage(content="alpha")]})


if __name__ == "__main__":
    unittest.main()
