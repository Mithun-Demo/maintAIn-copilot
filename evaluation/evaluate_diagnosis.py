
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DIAGNOSIS_FILE = (
    PROJECT_DIR / "evaluation" / "last_diagnosis.json"
)

EXPECTED_TEMPERATURES = {81.0, 86.0, 89.0, 91.0}
EXPECTED_VIBRATIONS = {7.2, 8.1, 8.6, 9.0}


def check(description, passed):
    """Print a check result and return its Boolean outcome."""
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {description}")
    return bool(passed)


def main():
    print("=== DIAGNOSIS QUALITY EVALUATION ===\n")

    if not DIAGNOSIS_FILE.is_file():
        print(f"Diagnosis file not found: {DIAGNOSIS_FILE}")
        print("Run 'python -m src.agent' first.")
        return

    try:
        with DIAGNOSIS_FILE.open("r", encoding="utf-8") as file:
            diagnosis = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Could not load diagnosis: {exc}")
        return

    results = []

    # 1. Overall JSON and required top-level fields
    results.append(check(
        "Diagnosis is a JSON object",
        isinstance(diagnosis, dict),
    ))

    required_fields = [
        "equipment_id",
        "telemetry_summary",
        "observations",
        "historical_evidence",
        "possible_causes",
        "recommended_checks",
        "draft_work_order",
        "missing_information",
        "limitations",
    ]

    results.append(check(
        "Required top-level fields are present",
        isinstance(diagnosis, dict)
        and all(field in diagnosis for field in required_fields),
    ))

    results.append(check(
        "Correct equipment ID",
        diagnosis.get("equipment_id") == "PUMP-01",
    ))

    # 2. Verify telemetry observations and supporting readings
    observations = diagnosis.get("observations", [])

    if not isinstance(observations, list):
        observations = []

    temperature_observation = next(
        (
            item for item in observations
            if isinstance(item, dict)
            and "temperature" in str(
                item.get("parameter", "")
            ).lower()
        ),
        {},
    )

    vibration_observation = next(
        (
            item for item in observations
            if isinstance(item, dict)
            and "vibration" in str(
                item.get("parameter", "")
            ).lower()
        ),
        {},
    )

    results.append(check(
        "Temperature observation is present",
        bool(temperature_observation),
    ))

    results.append(check(
        "Vibration observation is present",
        bool(vibration_observation),
    ))

    temperature_support = temperature_observation.get(
        "supporting_readings", []
    )
    vibration_support = vibration_observation.get(
        "supporting_readings", []
    )

    if not isinstance(temperature_support, list):
        temperature_support = []

    if not isinstance(vibration_support, list):
        vibration_support = []

    temperature_values = {
        row.get("temperature_c")
        for row in temperature_support
        if isinstance(row, dict)
        and isinstance(row.get("temperature_c"), (int, float))
    }

    vibration_values = {
        row.get("vibration_mm_s")
        for row in vibration_support
        if isinstance(row, dict)
        and isinstance(row.get("vibration_mm_s"), (int, float))
    }

    results.append(check(
        "All four expected high-temperature readings are cited",
        EXPECTED_TEMPERATURES.issubset(temperature_values),
    ))

    results.append(check(
        "All four expected high-vibration readings are cited",
        EXPECTED_VIBRATIONS.issubset(vibration_values),
    ))

    # 3. Historical evidence and limitations
    historical = diagnosis.get("historical_evidence", {})

    if not isinstance(historical, dict):
        historical = {}

    data_limitation = str(
        historical.get("data_limitation", "")
    ).lower()

    results.append(check(
        "Historical records are identified as synthetic",
        "synthetic" in data_limitation,
    ))

    limitations = diagnosis.get("limitations", [])

    if not isinstance(limitations, list):
        limitations = []

    limitations_text = " ".join(
        str(item).lower() for item in limitations
    )

    results.append(check(
        "Illustrative thresholds are disclosed",
        "illustrative" in limitations_text,
    ))

    # 4. Possible causes must be present and structured
    possible_causes = diagnosis.get("possible_causes", [])

    results.append(check(
        "At least one possible cause is provided",
        isinstance(possible_causes, list)
        and len(possible_causes) > 0,
    ))

    cause_fields = [
        "cause",
        "supporting_evidence",
        "confirmation_needed",
    ]

    causes_are_structured = (
        isinstance(possible_causes, list)
        and len(possible_causes) > 0
        and all(
            isinstance(cause, dict)
            and all(
                isinstance(cause.get(field), str)
                and bool(cause[field].strip())
                for field in cause_fields
            )
            for cause in possible_causes
        )
    )

    results.append(check(
        "Each possible cause includes evidence and confirmation steps",
        causes_are_structured,
    ))

    # 5. Recommended checks must be present
    recommended_checks = diagnosis.get("recommended_checks", [])

    results.append(check(
        "At least one recommended check is provided",
        isinstance(recommended_checks, list)
        and len(recommended_checks) > 0
        and all(
            isinstance(item, str) and item.strip()
            for item in recommended_checks
        ),
    ))

    # 6. Draft work-order structure and approval safeguard
    work_order = diagnosis.get("draft_work_order", {})

    required_work_order_fields = [
        "title",
        "equipment_id",
        "reported_symptoms",
        "suggested_actions",
        "priority",
        "approval_status",
    ]

    results.append(check(
        "Required work-order fields are present",
        isinstance(work_order, dict)
        and all(
            field in work_order
            for field in required_work_order_fields
        ),
    ))

    if isinstance(work_order, dict):
        approval_status = str(
            work_order.get("approval_status", "")
        ).lower()
    else:
        approval_status = ""

    results.append(check(
        "Work order requires human review",
        "draft" in approval_status
        and (
            "human" in approval_status
            or "review" in approval_status
        ),
    ))

    # 7. Summary
    passed = sum(results)
    total = len(results)

    print("\n=== SUMMARY ===")
    print(f"Passed: {passed}/{total}")
    print(f"Pass rate: {passed / total * 100:.1f}%")

    print(
        "\nThese checks measure output structure and selected "
        "evidence requirements, not real-world diagnostic accuracy."
    )


if __name__ == "__main__":
    main()
