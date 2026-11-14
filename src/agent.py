
import csv
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.tools import analyze_telemetry_csv
from src.retrieval import search_historical_failures
from src.vector_tools import search_similar_historical_cases


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPT_FILE = PROJECT_DIR / "prompts" / "baseline_diagnosis.txt"
TELEMETRY_DIR = PROJECT_DIR / "data" / "telemetry"

OUTPUT_FILE = PROJECT_DIR / "evaluation" / "last_diagnosis.json"

FILE_NAME = "sample_pump_01.csv"
SYMPTOM_QUERY = (
    "Pump has high vibration and increasing temperature"
)

# Illustrative demo thresholds only.
# These are NOT manufacturer-approved operating limits.
TEMPERATURE_THRESHOLD_C = 80.0
VIBRATION_THRESHOLD_MM_S = 7.0


# ============================================================
# 2. LOAD THE BASELINE PROMPT
# ============================================================

if not PROMPT_FILE.is_file():
    raise FileNotFoundError(
        f"Baseline prompt not found: {PROMPT_FILE}"
    )

baseline_prompt = PROMPT_FILE.read_text(encoding="utf-8")

# Python executes the tools directly, so remove the section
# that instructs the LLM to call tools itself.

start_marker = "## 2. Mandatory Tool-Use Procedure"
end_marker = "## 3. Evidence-Based Diagnosis"

start_index = baseline_prompt.find(start_marker)
end_index = baseline_prompt.find(end_marker)

if start_index != -1 and end_index > start_index:
    baseline_prompt = (
        baseline_prompt[:start_index]
        + baseline_prompt[end_index:]
    )

system_prompt = baseline_prompt + """

RUNTIME INSTRUCTIONS

The Python program has already executed the telemetry,
keyword-retrieval, and vector-retrieval tools.

Use the supplied tool outputs to generate a diagnosis.
Do not invent measurements, records, or citations.

Return a JSON object with these required top-level fields:
equipment_id, telemetry_summary, observations,
historical_evidence, possible_causes, recommended_checks,
draft_work_order, missing_information, limitations.

Each observation must contain:
parameter, observed_pattern, supporting_readings,
limit_comparison, severity.

The draft_work_order must contain:
title, equipment_id, reported_symptoms,
suggested_actions, priority, approval_status.

Include all threshold-exceeding telemetry readings available
in the supplied evidence. Do not include only the latest reading.

Historical records are synthetic demo records, not verified
industrial failure records. Similarity does not prove causation.
Vector distances are not probabilities.

The thresholds are illustrative, not manufacturer-approved limits.
Do not claim a work order was approved or submitted.

Return valid JSON only, without Markdown code fences.
"""


# ============================================================
# 3. LOCAL OLLAMA MODEL
# ============================================================

model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    format="json",
    num_predict=1400,
    num_ctx=4096,
    num_thread=4,
    client_kwargs={
        "timeout": 240
    }
)

def parse_model_json(content):
    """
    Parse a JSON object returned by the local LLM.

    Handles optional Markdown fences or explanatory text
    surrounding the JSON object.
    """

    if not isinstance(content, str) or not content.strip():
        raise ValueError("The AI model returned an empty response.")

    content = content.strip()

    # Remove optional Markdown code fences.
    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    # Locate the beginning of the JSON object.
    start = content.find("{")

    if start == -1:
        raise ValueError(
            "The AI response did not contain a JSON object."
        )

    # Parse one complete JSON object.
    try:
        result, _ = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError as exc:
        excerpt = content[max(0, exc.pos - 100):exc.pos + 150]

        raise ValueError(
            f"The AI response contained invalid or incomplete JSON. "
            f"Parser detail: {exc}. "
            f"Response excerpt: {excerpt!r}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "The AI response must be a JSON object."
        )

    return result

# ============================================================
# 4. TOOL EXECUTION
# ============================================================

def format_result(result):
    """Convert a tool result into text."""

    if isinstance(result, str):
        return result

    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def execute_tool(tool_name, tool, arguments):
    """Execute a tool once and capture its output or error."""

    print(f"Tool: {tool_name}")

    try:
        result = tool.invoke(arguments)
        return format_result(result)

    except Exception as exc:
        print(f"WARNING: {tool_name} failed: {exc}")

        return json.dumps({
            "tool_error": tool_name,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }, indent=2)


# ============================================================
# 5. READ AUTHORITATIVE TELEMETRY FROM THE CSV
# ============================================================

def load_telemetry_rows(file_name):
    """
    Read the telemetry CSV independently.

    These source readings are used to guarantee that the
    diagnosis does not omit measured threshold exceedances.
    """

    if Path(file_name).name != file_name:
        raise ValueError(
            "Only a CSV filename inside the telemetry directory "
            "is accepted."
        )

    if Path(file_name).suffix.lower() != ".csv":
        raise ValueError("Telemetry file must have a .csv extension.")

    telemetry_root = TELEMETRY_DIR.resolve()
    telemetry_path = (TELEMETRY_DIR / file_name).resolve()

    if telemetry_path.parent != telemetry_root:
        raise ValueError(
            "Telemetry file must be inside the telemetry directory."
        )

    if not telemetry_path.is_file():
        raise FileNotFoundError(
            f"Telemetry CSV not found: {telemetry_path}"
        )

    required_columns = {
        "timestamp",
        "equipment_id",
        "temperature_c",
        "vibration_mm_s",
    }

    rows = []

    with telemetry_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "Telemetry CSV is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            try:
                temperature = float(row["temperature_c"])
                vibration = float(row["vibration_mm_s"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid numeric telemetry data encountered."
                ) from exc

            rows.append({
                "timestamp": row["timestamp"],
                "equipment_id": row["equipment_id"],
                "temperature_c": temperature,
                "vibration_mm_s": vibration,
            })

    if not rows:
        raise ValueError("The telemetry CSV contains no readings.")

    return rows


# ============================================================
# 6. BUILD OBSERVATIONS DETERMINISTICALLY
# ============================================================

def make_observations(rows):
    """
    Construct observations and complete supporting_readings
    directly from the telemetry source, not from LLM guesses.
    """

    flagged_temperature = [
        row for row in rows
        if row["temperature_c"] > TEMPERATURE_THRESHOLD_C
    ]

    flagged_vibration = [
        row for row in rows
        if row["vibration_mm_s"] > VIBRATION_THRESHOLD_MM_S
    ]

    temperature_values = [
        row["temperature_c"] for row in flagged_temperature
    ]

    vibration_values = [
        row["vibration_mm_s"] for row in flagged_vibration
    ]

    if temperature_values:
        temperature_pattern = (
            f"{len(temperature_values)} readings exceeded the "
            f"illustrative threshold. Values: "
            + ", ".join(
                f"{value:.1f} °C" for value in temperature_values
            )
            + "."
        )
    else:
        temperature_pattern = (
            "No readings exceeded the illustrative temperature threshold."
        )

    if vibration_values:
        vibration_pattern = (
            f"{len(vibration_values)} readings exceeded the "
            f"illustrative threshold. Values: "
            + ", ".join(
                f"{value:.1f} mm/s" for value in vibration_values
            )
            + "."
        )
    else:
        vibration_pattern = (
            "No readings exceeded the illustrative vibration threshold."
        )

    temperature_evidence = [
        {
            "timestamp": row["timestamp"],
            "equipment_id": row["equipment_id"],
            "temperature_c": row["temperature_c"],
        }
        for row in flagged_temperature
    ]

    vibration_evidence = [
        {
            "timestamp": row["timestamp"],
            "equipment_id": row["equipment_id"],
            "vibration_mm_s": row["vibration_mm_s"],
        }
        for row in flagged_vibration
    ]

    observations = [
        {
            "parameter": "Temperature",
            "observed_pattern": temperature_pattern,
            "supporting_readings": temperature_evidence,
            "limit_comparison": (
                f"{len(flagged_temperature)} readings exceeded the "
                f"illustrative threshold of "
                f"{TEMPERATURE_THRESHOLD_C:.1f} °C. "
                "This is not a manufacturer-approved limit."
            ),
            "severity": "Not assessed",
        },
        {
            "parameter": "Vibration",
            "observed_pattern": vibration_pattern,
            "supporting_readings": vibration_evidence,
            "limit_comparison": (
                f"{len(flagged_vibration)} readings exceeded the "
                f"illustrative threshold of "
                f"{VIBRATION_THRESHOLD_MM_S:.1f} mm/s. "
                "This is not a manufacturer-approved limit."
            ),
            "severity": "Not assessed",
        },
    ]

    return observations, flagged_temperature, flagged_vibration


def make_telemetry_summary(
    equipment_id,
    flagged_temperature,
    flagged_vibration,
):
    """Create a factual summary from the measured readings."""

    temperature_values = [
        row["temperature_c"] for row in flagged_temperature
    ]
    vibration_values = [
        row["vibration_mm_s"] for row in flagged_vibration
    ]

    return (
        f"{equipment_id}: {len(temperature_values)} temperature "
        f"readings exceeded the illustrative threshold of "
        f"{TEMPERATURE_THRESHOLD_C:.1f} °C. "
        f"Flagged values: {temperature_values}. "
        f"{len(vibration_values)} vibration readings exceeded the "
        f"illustrative threshold of "
        f"{VIBRATION_THRESHOLD_MM_S:.1f} mm/s. "
        f"Flagged values: {vibration_values}. "
        "These demo thresholds are not manufacturer-approved limits."
    )


# ============================================================
# 7. NORMALIZE MODEL OUTPUT TO THE REQUIRED SCHEMA
# ============================================================

def as_string_list(value):
    """Convert a list or string to a clean list of strings."""

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def normalize_diagnosis(
    generated,
    rows,
    keyword_result,
    vector_result,
):
    """
    Fix inconsistent field names and data types.

    In particular, convert legacy 'maintenance_work_order'
    output into the required 'draft_work_order' structure.
    """

    if not isinstance(generated, dict):
        generated = {}

    equipment_ids = sorted({
        row["equipment_id"] for row in rows
    })

    equipment_id = (
        equipment_ids[0]
        if len(equipment_ids) == 1
        else ", ".join(equipment_ids)
    )

    observations, flagged_temperature, flagged_vibration = (
        make_observations(rows)
    )

    telemetry_summary = make_telemetry_summary(
        equipment_id,
        flagged_temperature,
        flagged_vibration,
    )

    # Model versions may use different names for the work order.
    original_work_order = generated.get("draft_work_order")

    if not isinstance(original_work_order, dict):
        original_work_order = generated.get("maintenance_work_order")

    if not isinstance(original_work_order, dict):
        original_work_order = {}

    # Retain generated recommendations where available.
    recommended_checks = as_string_list(
        generated.get("recommended_checks")
    )

    if not recommended_checks:
        recommended_checks = as_string_list(
            original_work_order.get("recommended_checks")
        )

    if not recommended_checks:
        recommended_checks = as_string_list(
            original_work_order.get("suggested_actions")
        )

    if not recommended_checks:
        recommended_checks = [
            "Inspect bearing condition.",
            "Check lubrication level and condition.",
            "Review vibration measurements.",
            "Inspect shaft alignment and coupling condition.",
            "Review maintenance history and sensor calibration.",
            "Compare measurements with manufacturer specifications.",
        ]

    reported_symptoms = []

    if flagged_temperature:
        values = [
            row["temperature_c"] for row in flagged_temperature
        ]
        reported_symptoms.append(
            f"{len(values)} temperature readings exceeded the "
            f"illustrative threshold of "
            f"{TEMPERATURE_THRESHOLD_C:.1f} °C; values: {values}."
        )

    if flagged_vibration:
        values = [
            row["vibration_mm_s"] for row in flagged_vibration
        ]
        reported_symptoms.append(
            f"{len(values)} vibration readings exceeded the "
            f"illustrative threshold of "
            f"{VIBRATION_THRESHOLD_MM_S:.1f} mm/s; values: {values}."
        )

    if not reported_symptoms:
        reported_symptoms.append(
            "No readings exceeded the configured illustrative thresholds."
        )

    title = (
        original_work_order.get("title")
        or original_work_order.get("task")
        or "Pump telemetry inspection and maintenance review"
    )

    # A missing priority must not be invented or treated as validated.
    priority = original_work_order.get("priority")

    if not isinstance(priority, str) or not priority.strip():
        priority = "Not assessed - requires qualified human assessment"

    # Always retain a clear draft / approval status.
    approval_status = "Draft - requires qualified human review"

    draft_work_order = {
        "title": title,
        "equipment_id": equipment_id,
        "reported_symptoms": reported_symptoms,
        "suggested_actions": recommended_checks,
        "priority": priority,
        "approval_status": approval_status,
    }

    # The model can mistakenly put explanatory instructions in
    # missing_information. Remove those internal-process statements.
    missing_information = as_string_list(
        generated.get("missing_information")
    )

    missing_information = [
        item for item in missing_information
        if "python program" not in item.lower()
        and "before saving" not in item.lower()
        and "will verify" not in item.lower()
    ]

    if not missing_information:
        missing_information = [
            "Manufacturer-approved operating limits and equipment "
            "specifications were not supplied.",
            "Recent maintenance records, operating conditions, and "
            "sensor calibration results were not supplied.",
        ]

    limitations = as_string_list(generated.get("limitations"))

    mandatory_limitations = [
        "Historical cases are synthetic demo records and do not "
        "confirm the current fault.",
        "Illustrative thresholds are not manufacturer-approved limits.",
        "Vector distances are not probabilities or calibrated "
        "confidence scores.",
    ]

    for limitation in mandatory_limitations:
        if limitation not in limitations:
            limitations.append(limitation)

    # Preserve the actual outputs from the retrieval tools.
    # This prevents the LLM from changing the number of results
    # or inventing a different similarity distance.
    historical_evidence = {
        "keyword_retrieval": keyword_result,
        "vector_retrieval": vector_result,
        "data_limitation": (
            "Historical cases are synthetic demonstration records. "
            "Retrieval similarity does not establish causation."
        ),
    }

    
    
    # Normalize possible causes into a consistent schema.
    raw_causes = generated.get("possible_causes", [])

    if not isinstance(raw_causes, list):
        raw_causes = []

    # Read the retrieved synthetic cases to obtain their
    # recorded causes and recommended checks.
    try:
        vector_data = json.loads(vector_result)
    except (json.JSONDecodeError, TypeError):
        vector_data = {}

    matches = vector_data.get("matches", [])

    if not isinstance(matches, list):
        matches = []

    case_details = {}

    for match in matches:
        if not isinstance(match, dict):
            continue

        case_id = match.get("case_id", "")
        content = match.get("content", "")

        if not isinstance(content, str):
            continue

        details = {
            "symptoms": "",
            "cause": "",
            "checks": [],
        }

        for line in content.splitlines():
            cleaned = line.strip()

            if cleaned.lower().startswith("symptoms:"):
                details["symptoms"] = cleaned.split(":", 1)[1].strip()

            elif cleaned.lower().startswith("possible cause:"):
                details["cause"] = cleaned.split(":", 1)[1].strip()

            elif cleaned.lower().startswith("recommended checks:"):
                check_text = cleaned.split(":", 1)[1].strip()
                details["checks"] = [
                    check.strip()
                    for check in check_text.split(",")
                    if check.strip()
                ]

        if case_id:
            case_details[case_id] = details

    # If the model returned no causes, use hypotheses from
    # the retrieved synthetic cases instead.
    if not raw_causes:
        for case_id, details in case_details.items():
            if details["cause"]:
                raw_causes.append({
                    "source": case_id,
                    "description": details["cause"],
                    "recommendations": details["checks"],
                })

    normalized_causes = []

    for item in raw_causes:
        if isinstance(item, str):
            item = {"cause": item}

        if not isinstance(item, dict):
            continue

        source = (
            item.get("source")
            or item.get("case_id")
            or "retrieved synthetic history"
        )

        cause_name = (
            item.get("cause")
            or item.get("description")
            or item.get("possible_cause")
        )

        if not isinstance(cause_name, str) or not cause_name.strip():
            continue

        details = case_details.get(source, {})

        # Prefer the actual retrieved case's checks when available.
        checks = (
            item.get("confirmation_needed")
            or item.get("recommendations")
            or details.get("checks")
            or []
        )

        if isinstance(checks, str):
            checks = [checks]

        checks = [
            str(check).strip()
            for check in checks
            if str(check).strip()
        ]

        if not checks:
            checks = [
                "Have qualified maintenance personnel inspect the "
                "equipment and compare measurements with manufacturer "
                "specifications."
            ]

        # Preserve the model's supporting evidence when it has
        # provided meaningful text; otherwise build it from the
        # actual telemetry counts and retrieved historical case.
        evidence = item.get("supporting_evidence")

        if not isinstance(evidence, str) or not evidence.strip():
            symptoms = details.get("symptoms", "not specified")

            evidence = (
                f"Current telemetry contains "
                f"{len(flagged_temperature)} temperature readings and "
                f"{len(flagged_vibration)} vibration readings above "
                f"the illustrative thresholds. Synthetic historical "
                f"case {source} describes symptoms: {symptoms}. "
                f"The retrieved case lists '{cause_name}' as a "
                f"possible cause. This is a hypothesis for further "
                f"investigation, not confirmation of the current fault."
            )
        else:
            evidence = (
                evidence.strip()
                + " Historical matches are synthetic demo records "
                "and do not confirm the current fault."
            )

        normalized_causes.append({
            "cause": cause_name.strip(),
            "supporting_evidence": evidence,
            "confirmation_needed": "; ".join(checks),
        })

    possible_causes = normalized_causes


    if not isinstance(possible_causes, list):
        possible_causes = []

    # If the model omitted causes, extract hypotheses from
    # the retrieved synthetic historical cases.
    if not possible_causes:
        possible_causes = []

        try:
            vector_data = json.loads(vector_result)
        except (json.JSONDecodeError, TypeError):
            vector_data = {}

        matches = vector_data.get("matches", [])

        if isinstance(matches, list):
            for match in matches:
                if not isinstance(match, dict):
                    continue

                case_id = match.get("case_id", "Unknown case")
                content = match.get("content", "")

                if not isinstance(content, str):
                    continue

                cause = ""
                symptoms = ""
                recommended_checks_from_case = []

                for line in content.splitlines():
                    cleaned = line.strip()

                    if cleaned.lower().startswith("possible cause:"):
                        cause = cleaned.split(":", 1)[1].strip()

                    elif cleaned.lower().startswith("symptoms:"):
                        symptoms = cleaned.split(":", 1)[1].strip()

                    elif cleaned.lower().startswith("recommended checks:"):
                        checks_text = cleaned.split(":", 1)[1].strip()
                        recommended_checks_from_case = [
                            item.strip()
                            for item in checks_text.split(",")
                            if item.strip()
                        ]

                if not cause:
                    continue

                current_evidence = (
                    f"Current telemetry has "
                    f"{len(flagged_temperature)} temperature readings "
                    f"above the illustrative threshold and "
                    f"{len(flagged_vibration)} vibration readings "
                    f"above the illustrative threshold."
                )

                historical_evidence_text = (
                    f"Synthetic historical case {case_id} describes "
                    f"symptoms: {symptoms or 'not specified'}. "
                    f"It lists this possible cause: {cause}. "
                    f"{current_evidence} This supports investigation "
                    f"only and does not confirm the cause."
                )

                possible_causes.append({
                    "cause": cause,
                    "supporting_evidence": historical_evidence_text,
                    "confirmation_needed": (
                        "; ".join(recommended_checks_from_case)
                        if recommended_checks_from_case
                        else (
                            "Qualified inspection and comparison "
                            "with manufacturer specifications."
                        )
                    ),
                })


    normalized = {
        "equipment_id": equipment_id,
        "telemetry_summary": telemetry_summary,
        "observations": observations,
        "historical_evidence": historical_evidence,
        "possible_causes": possible_causes,
        "recommended_checks": recommended_checks,
        "draft_work_order": draft_work_order,
        "missing_information": missing_information,
        "limitations": limitations,
    }

    return normalized


# ============================================================
# 8. VALIDATE THE FINAL STRUCTURE
# ============================================================

def validate_diagnosis(diagnosis):
    """Check the required top-level and work-order fields."""

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

    missing_fields = [
        field
        for field in required_fields
        if field not in diagnosis
    ]

    print("\n=== OUTPUT VALIDATION ===")
    print("JSON syntax: PASS")

    if missing_fields:
        print("Required fields: FAIL")
        print(f"Missing fields: {missing_fields}")
    else:
        print("Required fields: PASS")

    work_order = diagnosis.get("draft_work_order")

    work_order_fields = [
        "title",
        "equipment_id",
        "reported_symptoms",
        "suggested_actions",
        "priority",
        "approval_status",
    ]

    if not isinstance(work_order, dict):
        print("Draft work-order structure: FAIL")
        print("draft_work_order must be a JSON object.")
        return False

    missing_work_order_fields = [
        field
        for field in work_order_fields
        if field not in work_order
    ]

    if missing_work_order_fields:
        print("Draft work-order structure: FAIL")
        print(f"Missing fields: {missing_work_order_fields}")
        return False

    print("Draft work-order structure: PASS")

    return not missing_fields


# ============================================================
# 9. MAIN WORKFLOW
# ============================================================

def main():
    print("MaintAIn is analyzing the equipment...\n")
    print("=== TOOL CALLS ===")

    # Execute each required tool once.
    telemetry_result = execute_tool(
        "analyze_telemetry_csv",
        analyze_telemetry_csv,
        {"file_name": FILE_NAME},
    )

    keyword_result = execute_tool(
        "search_historical_failures",
        search_historical_failures,
        {"query": SYMPTOM_QUERY},
    )

    vector_result = execute_tool(
        "search_similar_historical_cases",
        search_similar_historical_cases,
        {"query": SYMPTOM_QUERY},
    )

    try:
        # Load source measurements for deterministic evidence handling.
        rows = load_telemetry_rows(FILE_NAME)

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

Generate a diagnosis in the required JSON schema.
Include possible causes, recommendations, missing information,
and limitations. Treat historical causes as hypotheses only.
Return valid JSON only.
"""

        print("\nGenerating the final diagnosis...")

        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=final_input),
        ])

        final_answer = response.content

        if not isinstance(final_answer, str) or not final_answer.strip():
            print("The model returned an empty response.")
            print("Raw response:", repr(response))
            return

        # Parse the model's JSON.
        try:
            generated = parse_model_json(final_answer)
        except json.JSONDecodeError as exc:
            print("\n=== OUTPUT VALIDATION ===")
            print("JSON syntax: FAIL")
            print(f"JSON parsing error: {exc}")
            print("No diagnosis was saved.")
            return

        if not isinstance(generated, dict):
            print("\nThe model response must be a JSON object.")
            return

        # Enforce the correct field names and source-backed evidence.
        diagnosis = normalize_diagnosis(
            generated=generated,
            rows=rows,
            keyword_result=keyword_result,
            vector_result=vector_result,
        )

        print("\n=== MAINTAIN RESPONSE ===")
        print(json.dumps(diagnosis, indent=2, ensure_ascii=False))

        # Validate the normalized output.
        valid = validate_diagnosis(diagnosis)

        # Save the normalized result.
        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_FILE.write_text(
            json.dumps(
                diagnosis,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"\nDiagnosis saved to: {OUTPUT_FILE}")

        if not valid:
            print(
                "WARNING: The diagnosis was saved, but some "
                "required fields are still missing."
            )

    except Exception as exc:
        print("\nMaintAIn could not complete the workflow.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error details: {exc}")


if __name__ == "__main__":
    main()
