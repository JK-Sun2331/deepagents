"""Research Agent - Standalone script for LangGraph deployment.

This module creates a deep research agent with custom tools and prompts
for conducting web research with strategic thinking and context management.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from research_agent.tools import tavily_search, think_tool

from datetime import datetime

#from langchain.chat_models import init_chat_model
#from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import SummarizationMiddleware

from research_agent.prompts import (
    RESEARCHER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)

# Limits
max_concurrent_research_units = int(
    os.environ.get("MAX_CONCURRENT_RESEARCH_UNITS", "3")
)
max_researcher_iterations = int(os.environ.get("MAX_RESEARCHER_ITERATIONS", "3"))
summarization_trigger_tokens = 100_000

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

# Create research sub-agent
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search, think_tool],
}

# Model Gemini 3 
# model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.0)

# Model Claude 4.5
#model = init_chat_model(model="anthropic:claude-sonnet-4-5-20250929", temperature=0.0)

model = ChatOpenAI(
    model=os.environ.get("LOCAL_LLM_MODEL", "qwen3-30b-a3b"),
    base_url=os.environ.get(
        "LOCAL_LLM_BASE_URL",
        "http://127.0.0.1:18320/v1",
    ),
    api_key=os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"),
    temperature=0.0,
    max_tokens=4096,
    timeout=180,
    max_retries=2,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

def research_summarization():
    """Match the 128K local service; preserve the previous retention policy."""
    return SummarizationMiddleware(
        model=model,
        backend=StateBackend(),
        trigger=("tokens", summarization_trigger_tokens),
        keep=("messages", 6),
        trim_tokens_to_summarize=None,
        truncate_args_settings={
            "trigger": ("messages", 20),
            "keep": ("messages", 20),
        },
    )


research_sub_agent["middleware"] = [research_summarization()]

# Named middleware replaces the built-in summarizer for both agent levels.
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
    middleware=[research_summarization()],
)
