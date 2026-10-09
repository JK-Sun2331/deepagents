"""Finish a research branch through an explicit, separately traced summary call."""

from collections.abc import Awaitable, Callable
from typing import Annotated, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse, PrivateStateAttr
from langchain.tools import ToolRuntime
from langchain_core.callbacks.manager import Callbacks
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs, patch_config
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command


class SummaryState(AgentState):
    """Keep the effective decision context private to its research branch."""

    summary_context: Annotated[NotRequired[list[BaseMessage]], PrivateStateAttr]


class ResearchSummaryMiddleware(AgentMiddleware):
    """Register summary and return its generated text as the subagent's answer.

    Args:
        model: The same unbound model used for research decisions.
        instructions: Research-phase instructions explaining the terminal tool.
        summary_prompt: Instructions for the separate tool-free model call.
    """

    state_schema = SummaryState

    def __init__(self, model: BaseChatModel, instructions: str, summary_prompt: str) -> None:
        self.model = model
        self.instructions = instructions
        self.summary_prompt = summary_prompt
        self.tools = [StructuredTool.from_function(
            func=self._summary,
            coroutine=self._asummary,
            name="summary",
            description=("Finish research and generate the final cited findings from this task's "
                         "existing evidence. Call with no arguments, alone, when ready to finish "
                         "or when the search budget is exhausted. This ends the subagent."),
            return_direct=True,
            response_format="content_and_artifact",
        )]

    def _messages(self, runtime: ToolRuntime) -> list[BaseMessage]:
        # Use the actual decision input after existing context management, not
        # raw state history that may already have been compacted. The pending
        # summary call is not in this snapshot, so every tool call is answered.
        return [SystemMessage(content=self.summary_prompt),
                *runtime.state["summary_context"],
                HumanMessage(content="Research is complete. Write the final findings now.")]

    def _config(self, config: RunnableConfig, callbacks: Callbacks) -> RunnableConfig:
        # StructuredTool injects the tool run's child callbacks separately from
        # RunnableConfig. Use them so the model is nested under summary itself.
        return merge_configs(patch_config(config, callbacks=callbacks), {"tags": ["research_summary"],
                                      "metadata": {"research_phase": "summary"}})

    def _result(self, response: AIMessage) -> tuple[str, dict[str, str]]:
        if response.tool_calls or not response.text.strip():
            msg = "Summary must return nonempty findings without tool calls"
            raise RuntimeError(msg)
        if response.response_metadata.get("finish_reason") == "length":
            msg = "Summary generation was truncated"
            raise RuntimeError(msg)
        # Preserve the complete answer even if FilesystemMiddleware evicts a
        # large ToolMessage body. Artifacts are not sent back to the model.
        return response.text, {"summary_text": response.text}

    def _summary(self, runtime: ToolRuntime, config: RunnableConfig, callbacks: Callbacks) -> tuple[str, dict[str, str]]:
        response = self.model.invoke(self._messages(runtime), config=self._config(config, callbacks))
        return self._result(response)

    async def _asummary(self, runtime: ToolRuntime, config: RunnableConfig, callbacks: Callbacks) -> tuple[str, dict[str, str]]:
        response = await self.model.ainvoke(self._messages(runtime), config=self._config(config, callbacks))
        return self._result(response)

    def _request(self, request: ModelRequest) -> ModelRequest:
        system = request.system_message
        blocks = list(system.content_blocks) if system else []
        blocks.append({"type": "text", "text": self.instructions})
        return request.override(system_message=SystemMessage(content=blocks), tool_choice="required")

    def _validate(self, response: ModelResponse, request: ModelRequest) -> ModelResponse | ExtendedModelResponse:
        message = response.result[-1]
        calls = message.tool_calls if isinstance(message, AIMessage) else []
        if not calls:
            msg = "Research must finish by calling summary, not by returning a direct answer"
            raise RuntimeError(msg)
        if any(call["name"] == "summary" for call in calls) and len(calls) != 1:
            msg = "summary must be called alone, without concurrent tools"
            raise RuntimeError(msg)
        if calls[0]["name"] == "summary":
            return ExtendedModelResponse(response, Command(update={"summary_context": list(request.messages)}))
        return response

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse | ExtendedModelResponse:
        """Require an explicit tool decision in the research phase."""
        return self._validate(handler(self._request(request)), request)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse | ExtendedModelResponse:
        """Require an explicit tool decision in asynchronous research."""
        return self._validate(await handler(self._request(request)), request)

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, list[BaseMessage]]:
        """Expose the tool-generated answer to the parent task return path."""
        message = state["messages"][-1]
        if not isinstance(message, ToolMessage) or message.name != "summary" or message.status != "success":
            msg = "Research ended without a successful summary tool result"
            raise RuntimeError(msg)
        return {"messages": [AIMessage(content=message.artifact["summary_text"])]}

    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> dict[str, list[BaseMessage]]:
        """Expose the final answer without another model invocation."""
        return self.after_agent(state, runtime)
