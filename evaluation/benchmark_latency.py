
import json
import statistics
import time
from pathlib import Path

from src.agent import (
    PROJECT_DIR,
    FILE_NAME,
    SYMPTOM_QUERY,
    model,
    system_prompt,
    execute_tool,
    load_telemetry_rows,
    normalize_diagnosis,
)

from src.tools import analyze_telemetry_csv
from src.retrieval import search_historical_failures
from src.vector_tools import search_similar_historical_cases


# ==================================================
# 1. CONFIGURATION
# ==================================================

RUNS = 3

RESULT_FILE = (
    PROJECT_DIR
    / "evaluation"
    / "latency_benchmark_results.json"
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def contains_tool_error(result):
    """Identify a tool error returned by execute_tool."""

    try:
        data = json.loads(result)
        return isinstance(data, dict) and "tool_error" in data
    except (json.JSONDecodeError, TypeError):
        return False


def calculate_statistics(values):
    """Calculate timing statistics."""

    return {
        "average_seconds": round(statistics.mean(values), 4),
        "median_seconds": round(statistics.median(values), 4),
        "minimum_seconds": round(min(values), 4),
        "maximum_seconds": round(max(values), 4),
    }


# ==================================================
# 3. EXECUTE ONE COMPLETE DIAGNOSIS RUN
# ==================================================

def run_once(run_number):

    print(f"\n========== RUN {run_number} ==========")

    timings = {}
    total_start = time.perf_counter()

    try:
        # ------------------------------------------
        # Stage 1: Telemetry analysis
        # ------------------------------------------

        start = time.perf_counter()

        telemetry_result = execute_tool(
            "analyze_telemetry_csv",
            analyze_telemetry_csv,
            {"file_name": FILE_NAME},
        )

        timings["telemetry_analysis_seconds"] = (
            time.perf_counter() - start
        )

        if contains_tool_error(telemetry_result):
            raise RuntimeError("Telemetry analysis tool failed.")

        # ------------------------------------------
        # Stage 2: Keyword retrieval
        # ------------------------------------------

        start = time.perf_counter()

        keyword_result = execute_tool(
            "search_historical_failures",
            search_historical_failures,
            {"query": SYMPTOM_QUERY},
        )

        timings["keyword_retrieval_seconds"] = (
            time.perf_counter() - start
        )

        if contains_tool_error(keyword_result):
            raise RuntimeError("Keyword retrieval tool failed.")

        # ------------------------------------------
        # Stage 3: Vector retrieval
        # ------------------------------------------

        start = time.perf_counter()

        vector_result = execute_tool(
            "search_similar_historical_cases",
            search_similar_historical_cases,
            {"query": SYMPTOM_QUERY},
        )

        timings["vector_retrieval_seconds"] = (
            time.perf_counter() - start
        )

        if contains_tool_error(vector_result):
            raise RuntimeError("Vector retrieval tool failed.")

        # ------------------------------------------
        # Stage 4: Load authoritative CSV evidence
        # ------------------------------------------

        start = time.perf_counter()

        rows = load_telemetry_rows(FILE_NAME)

        timings["csv_evidence_loading_seconds"] = (
            time.perf_counter() - start
        )

        # ------------------------------------------
        # Stage 5: LLM diagnosis generation
        # ------------------------------------------

        final_input = f"""
USER REQUEST
Analyze {FILE_NAME} for equipment PUMP-01.

TELEMETRY TOOL RESULT
{telemetry_result}

KEYWORD RETRIEVAL TOOL RESULT
{keyword_result}

VECTOR RETRIEVAL TOOL RESULT
{vector_result}

SOURCE TELEMETRY ROWS
{json.dumps(rows, indent=2)}

Generate the final MaintAIn diagnosis.

Return valid JSON only.
Do not invent measurements or confirmed faults.
Identify historical cases as synthetic demonstration records.
"""

        start = time.perf_counter()

        response = model.invoke([
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": final_input,
            },
        ])

        timings["llm_generation_seconds"] = (
            time.perf_counter() - start
        )

        # ------------------------------------------
        # Stage 6: JSON parsing and normalization
        # ------------------------------------------

        start = time.perf_counter()

        generated = json.loads(response.content)

        if not isinstance(generated, dict):
            raise ValueError(
                "LLM response is not a JSON object."
            )

        diagnosis = normalize_diagnosis(
            generated=generated,
            rows=rows,
            keyword_result=keyword_result,
            vector_result=vector_result,
        )

        timings["normalization_seconds"] = (
            time.perf_counter() - start
        )

        # Basic completion check.
        required_fields = [
            "equipment_id",
            "observations",
            "historical_evidence",
            "possible_causes",
            "recommended_checks",
            "draft_work_order",
        ]

        if not all(field in diagnosis for field in required_fields):
            raise ValueError(
                "Diagnosis is missing required fields."
            )

        status = "SUCCESS"
        error = None

    except Exception as exc:
        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"

    total_seconds = time.perf_counter() - total_start

    timings["end_to_end_seconds"] = total_seconds

    print(f"\nRun status: {status}")

    for metric, duration in timings.items():
        print(f"{metric}: {duration:.4f} seconds")

    if error:
        print(f"Error: {error}")

    return {
        "run": run_number,
        "status": status,
        "error": error,
        "timings": {
            key: round(value, 4)
            for key, value in timings.items()
        },
    }


# ==================================================
# 4. RUN THE LATENCY BENCHMARK
# ==================================================

def main():

    print("==========================================")
    print(" MAINTAIN LATENCY BENCHMARK")
    print("==========================================")

    print(f"Model: qwen2.5:3b")
    print(f"Embedding model: nomic-embed-text")
    print(f"Benchmark runs: {RUNS}")

    all_results = []

    for run_number in range(1, RUNS + 1):
        result = run_once(run_number)
        all_results.append(result)

    successful_runs = [
        result
        for result in all_results
        if result["status"] == "SUCCESS"
    ]

    failed_runs = [
        result
        for result in all_results
        if result["status"] == "FAILED"
    ]

    if successful_runs:

        metric_names = [
            "telemetry_analysis_seconds",
            "keyword_retrieval_seconds",
            "vector_retrieval_seconds",
            "csv_evidence_loading_seconds",
            "llm_generation_seconds",
            "normalization_seconds",
            "end_to_end_seconds",
        ]

        summary = {}

        for metric in metric_names:
            values = [
                result["timings"][metric]
                for result in successful_runs
                if metric in result["timings"]
            ]

            if values:
                summary[metric] = calculate_statistics(values)

    else:
        summary = {}

    # ----------------------------------------------
    # Save benchmark report
    # ----------------------------------------------

    report = {
        "project": "MaintAIn",
        "model": "qwen2.5:3b",
        "embedding_model": "nomic-embed-text",
        "dataset": "synthetic_demo_records",
        "total_runs": RUNS,
        "successful_runs": len(successful_runs),
        "failed_runs": len(failed_runs),
        "summary": summary,
        "individual_runs": all_results,
        "limitations": [
            "Results depend on hardware and local Ollama performance.",
            "The first run may include model warm-up overhead.",
            "Measurements are from a small synthetic demonstration.",
            "These results are not production industrial benchmarks."
        ],
    }

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ----------------------------------------------
    # Print final summary
    # ----------------------------------------------

    print("\n==========================================")
    print(" LATENCY BENCHMARK SUMMARY")
    print("==========================================")

    print(f"Successful runs: {len(successful_runs)}/{RUNS}")
    print(f"Failed runs: {len(failed_runs)}")

    if summary:

        for metric, values in summary.items():

            print(f"\n{metric}")

            print(
                f"Average: {values['average_seconds']:.4f} seconds"
            )

            print(
                f"Minimum: {values['minimum_seconds']:.4f} seconds"
            )

            print(
                f"Maximum: {values['maximum_seconds']:.4f} seconds"
            )

    else:
        print("\nNo successful runs. Check the errors above.")

    print(f"\nDetailed results saved to:\n{RESULT_FILE}")


if __name__ == "__main__":
    main()
