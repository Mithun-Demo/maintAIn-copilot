
from pathlib import Path

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.tools import analyze_telemetry_csv
from src.retrieval import search_historical_failures

maintenance_tools = [
    analyze_telemetry_csv,
    search_historical_failures,
]


# Locate the project directory and baseline prompt.
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPT_FILE = PROJECT_DIR / "prompts" / "baseline_diagnosis.txt"

if not PROMPT_FILE.is_file():
    raise FileNotFoundError(
        f"Baseline prompt not found: {PROMPT_FILE}"
    )

baseline_prompt = PROMPT_FILE.read_text(encoding="utf-8")

system_prompt = baseline_prompt + """


ADDITIONAL TOOL INSTRUCTIONS

You have access to two tools:
1. analyze_telemetry_csv
2. search_historical_failures

For every equipment diagnosis request, follow this workflow in order:

STEP 1: Call analyze_telemetry_csv for the requested telemetry file.
STEP 2: Read the actual tool results and identify the observed symptoms.
STEP 3: Call search_historical_failures with a short query based on
the observed symptoms, such as "high temperature high vibration".
STEP 4: Compare the telemetry evidence with the returned historical cases.
STEP 5: Provide the final diagnosis and recommendations.

Do not skip Steps 1, 2, 3, or 4.

Evidence rules:
- Telemetry readings are observations, not proof of a particular fault.
- Historical records are synthetic demonstration data, not verified
  industrial failure records.
- Similar symptoms do not prove the same cause.
- Never invent readings, historical records, or manual citations.
- State what information is missing when evidence is insufficient.
- The temperature and vibration thresholds are illustrative only.
- Work orders are drafts and require human approval.

"""

# Run Qwen locally through Ollama.
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

# Create the LangChain agent with your existing tool.
agent = create_agent(
    model=model,
    tools=maintenance_tools,
    system_prompt=system_prompt,
)


def main():
    question = (
        "Analyze sample_pump_01.csv for PUMP-01. "
    "First call analyze_telemetry_csv and inspect its results. "
    "Then call search_historical_failures using the observed "
    "symptoms of high temperature and high vibration. "
    "Use both tool outputs in your final response. "
    "Separate measured evidence from possible causes, and "
    "identify any missing information."
    )

    print("MaintAIn is analyzing the equipment...\n")

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    print("=== TOOL CALLS ===")

    tool_call_count = 0

    for message in result["messages"]:
        for call in getattr(message, "tool_calls", []):
            tool_call_count += 1
            print(f"Tool: {call['name']}")

    if tool_call_count == 0:
        print("No tool calls were recorded.")

    print("\n=== MAINTAIN RESPONSE ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
