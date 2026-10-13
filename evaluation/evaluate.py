
import json
from pathlib import Path

from src.anomaly_detection import analyze_telemetry

PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_CASES_FILE = PROJECT_DIR / "evaluation" / "test_cases.json"


def run_evaluation():
    with TEST_CASES_FILE.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    results = []

    for test_case in test_cases:
        test_id = test_case["test_id"]
        description = test_case["description"]
        file_name = test_case["file_name"]
        expected = test_case["expected"]

        print(f"\nRunning {test_id}: {description}")

        try:
            actual = analyze_telemetry(file_name)

            if expected["status"] == "error":
                passed = False
                details = "Expected an error, but analysis succeeded."
            else:
                checks = {
                    key: actual.get(key) == value
                    for key, value in expected.items()
                    if key != "status"
                }

                passed = all(checks.values())
                details = ", ".join(
                    f"{key}={'PASS' if ok else 'FAIL'}"
                    for key, ok in checks.items()
                )

        except Exception as error:
            if expected["status"] == "error":
                passed = type(error).__name__ == expected["error_type"]
                details = (
                    f"Expected {expected['error_type']}, "
                    f"received {type(error).__name__}"
                )
            else:
                passed = False
                details = (
                    f"Unexpected {type(error).__name__}: {error}"
                )

        result = {
            "test_id": test_id,
            "description": description,
            "passed": passed,
            "details": details
        }
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"{status}: {details}")

    passed_count = sum(result["passed"] for result in results)
    total_count = len(results)

    summary = {
        "passed": passed_count,
        "failed": total_count - passed_count,
        "total": total_count,
        "pass_rate_percent": round(
            passed_count / total_count * 100, 2
        ) if total_count else 0,
        "results": results
    }

    output_file = PROJECT_DIR / "evaluation" / "evaluation_results.json"
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Passed: {passed_count}/{total_count}")
    print(f"Pass rate: {summary['pass_rate_percent']}%")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
