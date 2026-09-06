import sys

from langchain.messages import HumanMessage

from agent import agent


DEFAULT_QUERY = """
Compare vLLM, SGLang, and TensorRT-LLM in terms of serving architecture,
KV-cache management, and scheduling. Delegate the three systems to separate
research sub-agents when appropriate, and produce a concise cited report.
""".strip()


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    print("Research query:", query)
    print("Starting Deep Research. This may take several minutes.\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=query),
            ]
        }
    )

    messages = result.get("messages", [])
    if not messages:
        raise SystemExit("No messages returned by the agent")

    print("\n===== FINAL MESSAGE =====\n")
    print(messages[-1].content)

    files = result.get("files")
    if files:
        print("\nVirtual files in agent state:")
        for path in files:
            print(" -", path)


if __name__ == "__main__":
    main()