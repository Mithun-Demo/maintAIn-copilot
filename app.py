
import json
import time
import uuid

import pandas as pd
import streamlit as st

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent import (
    PROJECT_DIR,
    TELEMETRY_DIR,
    TEMPERATURE_THRESHOLD_C,
    VIBRATION_THRESHOLD_MM_S,
    model,
    system_prompt,
    load_telemetry_rows,
    normalize_diagnosis,
    parse_model_json,
)

from src.tools import analyze_telemetry_csv
from src.retrieval import search_historical_failures
from src.vector_tools import search_similar_historical_cases


# ==================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="MaintAIn | Maintenance Copilot",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ MaintAIn")

st.subheader(
    "Evidence-Based Autonomous Maintenance Copilot"
)

st.caption(
    "Industrial equipment telemetry analysis, "
    "historical retrieval and AI-assisted diagnosis."
)

st.info(
    "Demo application: historical maintenance cases are synthetic. "
    "Temperature and vibration thresholds are illustrative, "
    "not manufacturer-approved operating limits."
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def tool_result_to_text(result):
    """Convert tool output into JSON or readable text."""

    if isinstance(result, str):
        return result

    return json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def check_tool_error(result):
    """Check whether a tool returned an error."""

    try:
        data = json.loads(result)

        return (
            isinstance(data, dict)
            and "tool_error" in data
        )

    except (json.JSONDecodeError, TypeError):
        return False


def parse_retrieval_result(value):
    """Parse a stored historical retrieval result."""

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)

        except json.JSONDecodeError:
            return {}

    return {}


# ==================================================
# 3. MAIN MAINTAIN ANALYSIS FUNCTION
# ==================================================

def run_maintain(file_name, symptom_query, progress=None):
    """
    Run telemetry analysis, historical retrieval,
    vector retrieval and AI diagnosis.

    The progress callback updates the Streamlit status panel.
    """

    def update_progress(message):

        if progress is not None:
            progress.update(
                label=message,
                state="running",
                expanded=True,
            )

    timings = {}

    total_start = time.perf_counter()

    # ----------------------------------------------
    # STAGE 1: TELEMETRY ANALYSIS
    # ----------------------------------------------

    update_progress(
        "Step 1/4: Analyzing equipment telemetry..."
    )

    start = time.perf_counter()

    telemetry_result = tool_result_to_text(
        analyze_telemetry_csv.invoke({
            "file_name": file_name
        })
    )

    timings["Telemetry analysis"] = (
        time.perf_counter() - start
    )

    if check_tool_error(telemetry_result):

        raise RuntimeError(
            f"Telemetry analysis failed: {telemetry_result}"
        )

    # ----------------------------------------------
    # STAGE 2: KEYWORD RETRIEVAL
    # ----------------------------------------------

    update_progress(
        "Step 2/4: Searching historical failures..."
    )

    start = time.perf_counter()

    keyword_result = tool_result_to_text(
        search_historical_failures.invoke({
            "query": symptom_query
        })
    )

    timings["Keyword retrieval"] = (
        time.perf_counter() - start
    )

    if check_tool_error(keyword_result):

        raise RuntimeError(
            f"Keyword retrieval failed: {keyword_result}"
        )

    # ----------------------------------------------
    # STAGE 3: VECTOR RETRIEVAL
    # ----------------------------------------------

    update_progress(
        "Step 3/4: Performing semantic vector retrieval..."
    )

    start = time.perf_counter()

    vector_result = tool_result_to_text(
        search_similar_historical_cases.invoke({
            "query": symptom_query
        })
    )

    timings["Vector retrieval"] = (
        time.perf_counter() - start
    )

    if check_tool_error(vector_result):

        raise RuntimeError(
            f"Vector retrieval failed: {vector_result}"
        )

    # ----------------------------------------------
    # READ SOURCE TELEMETRY
    # ----------------------------------------------

    update_progress(
        "Reading and validating source telemetry..."
    )

    start = time.perf_counter()

    rows = load_telemetry_rows(file_name)

    if not rows:
        raise ValueError(
            "The uploaded CSV contains no telemetry records."
        )

    timings["CSV evidence loading"] = (
        time.perf_counter() - start
    )

    # ----------------------------------------------
    # STAGE 4: AI DIAGNOSIS
    # ----------------------------------------------

    update_progress(
        "Step 4/4: Generating AI diagnosis. "
        "The local Ollama model may take around 1–2 minutes..."
    )

    final_input = f"""
USER REQUEST
Analyze telemetry file {file_name}.

SYMPTOM QUERY
{symptom_query}

TELEMETRY ANALYSIS TOOL RESULT
{telemetry_result}

KEYWORD RETRIEVAL TOOL RESULT
{keyword_result}

VECTOR RETRIEVAL TOOL RESULT
{vector_result}

SOURCE TELEMETRY ROWS
{json.dumps(rows, indent=2)}

Generate the final MaintAIn diagnosis.

Use only supported evidence.
Include all required JSON fields.
Historical cases are synthetic demonstration records.
Do not present possible causes as confirmed faults.
Return valid JSON only.
"""

    start = time.perf_counter()

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=final_input),
    ])

    timings["LLM generation"] = (
        time.perf_counter() - start
    )

    # ----------------------------------------------
    # NORMALIZE DIAGNOSIS
    # ----------------------------------------------

    update_progress(
        "Validating and formatting the diagnosis..."
    )

    start = time.perf_counter()

    generated = parse_model_json(response.content)

    if not isinstance(generated, dict):

        raise ValueError(
            "The LLM response is not a JSON object."
        )

    diagnosis = normalize_diagnosis(
        generated=generated,
        rows=rows,
        keyword_result=keyword_result,
        vector_result=vector_result,
    )

    timings["Normalization"] = (
        time.perf_counter() - start
    )

    timings["Total execution"] = (
        time.perf_counter() - total_start
    )

    # ----------------------------------------------
    # SAVE LATEST DIAGNOSIS
    # ----------------------------------------------

    output_path = (
        PROJECT_DIR
        / "evaluation"
        / "last_diagnosis.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            diagnosis,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return diagnosis, rows, timings


# ==================================================
# 4. USER INPUT
# ==================================================

st.header("Equipment Analysis")

uploaded_file = st.file_uploader(
    "Upload equipment telemetry CSV",
    type=["csv"],
    help=(
        "Expected columns: timestamp, equipment_id, "
        "temperature_c, vibration_mm_s, pressure_bar, rpm."
    ),
)

symptom_query = st.text_input(
    "Equipment symptoms",
    value=(
        "Pump has high vibration and increasing temperature"
    ),
)

analyze_button = st.button(
    "Run MaintAIn Analysis",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
)


# ==================================================
# 5. RUN THE DIAGNOSIS
# ==================================================

if analyze_button and uploaded_file is not None:

    uploaded_name = (
        f"streamlit_{uuid.uuid4().hex}.csv"
    )

    TELEMETRY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    uploaded_path = TELEMETRY_DIR / uploaded_name

    status = None

    # Clear previous result before a new run.
    st.session_state["analysis_success"] = False

    try:

        with st.status(
            "Preparing MaintAIn analysis...",
            expanded=True,
        ) as status:

            uploaded_path.write_bytes(
                uploaded_file.getvalue()
            )

            diagnosis, rows, timings = run_maintain(
                uploaded_name,
                symptom_query.strip()
                or "Analyze the uploaded equipment telemetry",
                progress=status,
            )

            # Save results in Streamlit session state.
            st.session_state["diagnosis"] = diagnosis

            st.session_state["rows"] = rows

            st.session_state["timings"] = timings

            st.session_state["analysis_success"] = True

            status.update(
                label=(
                    "MaintAIn analysis completed successfully!"
                ),
                state="complete",
                expanded=False,
            )

        st.success(
            "Your equipment analysis is ready!"
        )

    except Exception as exc:

        st.session_state["analysis_success"] = False

        if status is not None:

            status.update(
                label="MaintAIn analysis failed.",
                state="error",
                expanded=True,
            )

        st.error(
            f"Analysis failed: {type(exc).__name__}: {exc}"
        )

    finally:

        if uploaded_path.exists():
            uploaded_path.unlink()


# ==================================================
# 6. DISPLAY RESULTS
# ==================================================

if st.session_state.get("analysis_success", False):

    diagnosis = st.session_state["diagnosis"]

    rows = st.session_state["rows"]

    timings = st.session_state["timings"]

    dataframe = pd.DataFrame(rows)

    # ----------------------------------------------
    # VALIDATE REQUIRED TELEMETRY COLUMNS
    # ----------------------------------------------

    required_columns = [
        "timestamp",
        "temperature_c",
        "vibration_mm_s",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        st.error(
            f"Missing telemetry columns: {missing_columns}"
        )

    else:

        # ------------------------------------------
        # CALCULATE TELEMETRY FLAGS
        # ------------------------------------------

        temperature_flags = int(
            (
                dataframe["temperature_c"]
                > TEMPERATURE_THRESHOLD_C
            ).sum()
        )

        vibration_flags = int(
            (
                dataframe["vibration_mm_s"]
                > VIBRATION_THRESHOLD_MM_S
            ).sum()
        )

        total_flags = int(
            (
                (
                    dataframe["temperature_c"]
                    > TEMPERATURE_THRESHOLD_C
                )
                |
                (
                    dataframe["vibration_mm_s"]
                    > VIBRATION_THRESHOLD_MM_S
                )
            ).sum()
        )

        # ------------------------------------------
        # DASHBOARD HEADER
        # ------------------------------------------

        st.divider()

        st.header("Diagnosis Dashboard")

        # ------------------------------------------
        # OVERVIEW METRICS
        # ------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Equipment ID",
            diagnosis["equipment_id"],
        )

        col2.metric(
            "Telemetry readings",
            len(rows),
        )

        col3.metric(
            "Temperature exceedances",
            temperature_flags,
        )

        col4.metric(
            "Vibration exceedances",
            vibration_flags,
        )

        st.metric(
            "Readings with at least one exceedance",
            total_flags,
        )

        # ------------------------------------------
        # DASHBOARD TABS
        # ------------------------------------------

        (
            overview_tab,
            telemetry_tab,
            history_tab,
            diagnosis_tab,
            performance_tab,
        ) = st.tabs([
            "Overview",
            "Telemetry",
            "Historical Evidence",
            "Diagnosis & Work Order",
            "Performance",
        ])

        # ==========================================
        # TAB 1: OVERVIEW
        # ==========================================

        with overview_tab:

            st.subheader("Telemetry Summary")

            st.write(
                diagnosis["telemetry_summary"]
            )

            st.subheader("Key Observations")

            for observation in diagnosis["observations"]:

                with st.expander(
                    observation["parameter"],
                    expanded=True,
                ):

                    st.write(
                        observation["observed_pattern"]
                    )

                    st.write(
                        observation["limit_comparison"]
                    )

            st.warning(
                "Threshold exceedances indicate readings "
                "requiring investigation. They do not "
                "independently confirm equipment failure."
            )

        # ==========================================
        # TAB 2: TELEMETRY
        # ==========================================

        with telemetry_tab:

            st.subheader("Temperature Trend")

            temperature_chart = dataframe.set_index(
                "timestamp"
            )[["temperature_c"]]

            st.line_chart(
                temperature_chart
            )

            st.caption(
                f"Illustrative temperature threshold: "
                f"{TEMPERATURE_THRESHOLD_C} °C"
            )

            st.divider()

            st.subheader("Vibration Trend")

            vibration_chart = dataframe.set_index(
                "timestamp"
            )[["vibration_mm_s"]]

            st.line_chart(
                vibration_chart
            )

            st.caption(
                f"Illustrative vibration threshold: "
                f"{VIBRATION_THRESHOLD_MM_S} mm/s"
            )

            st.divider()

            st.subheader("Uploaded Telemetry Data")

            st.dataframe(
                dataframe,
                use_container_width=True,
            )

        # ==========================================
        # TAB 3: HISTORICAL EVIDENCE
        # ==========================================

        with history_tab:

            historical = diagnosis["historical_evidence"]

            st.subheader(
                "Keyword-Based Retrieval"
            )

            keyword_data = parse_retrieval_result(
                historical["keyword_retrieval"]
            )

            keyword_matches = keyword_data.get(
                "matches",
                [],
            )

            if keyword_matches:

                st.dataframe(
                    pd.DataFrame(keyword_matches),
                    use_container_width=True,
                )

            else:

                st.info(
                    "No keyword matches available."
                )

            st.divider()

            st.subheader(
                "Semantic Vector Retrieval"
            )

            vector_data = parse_retrieval_result(
                historical["vector_retrieval"]
            )

            vector_matches = vector_data.get(
                "matches",
                [],
            )

            if vector_matches:

                vector_table = []

                for match in vector_matches:

                    vector_table.append({
                        "Case ID": match.get("case_id"),
                        "Equipment": match.get("equipment_type"),
                        "Distance": match.get("distance"),
                        "Source": match.get("source_type"),
                    })

                st.dataframe(
                    pd.DataFrame(vector_table),
                    use_container_width=True,
                )

            else:

                st.info(
                    "No vector matches available."
                )

            st.caption(
                historical["data_limitation"]
            )

            st.warning(
                "Historical cases are synthetic demonstration "
                "records. Similarity does not prove the "
                "current equipment fault."
            )

        # ==========================================
        # TAB 4: DIAGNOSIS AND WORK ORDER
        # ==========================================

        with diagnosis_tab:

            st.subheader("Possible Causes")

            for index, cause in enumerate(
                diagnosis["possible_causes"],
                start=1,
            ):

                with st.expander(
                    f"Possible cause {index}: {cause['cause']}",
                    expanded=True,
                ):

                    st.markdown(
                        "**Supporting evidence**"
                    )

                    st.write(
                        cause["supporting_evidence"]
                    )

                    st.markdown(
                        "**Confirmation required**"
                    )

                    st.write(
                        cause["confirmation_needed"]
                    )

            st.divider()

            st.subheader("Recommended Checks")

            for check in diagnosis["recommended_checks"]:

                st.markdown(
                    f"- {check}"
                )

            st.divider()

            work_order = diagnosis["draft_work_order"]

            st.subheader(
                "Draft Maintenance Work Order"
            )

            st.markdown(
                f"**Title:** {work_order['title']}"
            )

            st.markdown(
                f"**Equipment:** {work_order['equipment_id']}"
            )

            st.markdown(
                f"**Priority:** {work_order['priority']}"
            )

            st.markdown(
                f"**Status:** {work_order['approval_status']}"
            )

            st.markdown(
                "**Reported Symptoms**"
            )

            for symptom in work_order["reported_symptoms"]:

                st.markdown(
                    f"- {symptom}"
                )

            st.markdown(
                "**Suggested Actions**"
            )

            for action in work_order["suggested_actions"]:

                st.markdown(
                    f"- {action}"
                )

            st.divider()

            st.subheader("Missing Information")

            for item in diagnosis["missing_information"]:

                st.markdown(
                    f"- {item}"
                )

            st.subheader("Limitations")

            for item in diagnosis["limitations"]:

                st.markdown(
                    f"- {item}"
                )

        # ==========================================
        # TAB 5: PERFORMANCE
        # ==========================================

        with performance_tab:

            st.subheader(
                "Execution Performance"
            )

            performance_table = pd.DataFrame(
                [
                    {
                        "Stage": stage,
                        "Time (seconds)": round(
                            duration,
                            4,
                        ),
                    }
                    for stage, duration in timings.items()
                ]
            )

            st.dataframe(
                performance_table,
                use_container_width=True,
            )

            st.metric(
                "Total execution time",
                f"{timings['Total execution']:.2f} seconds",
            )

            st.caption(
                "Runtime depends on local hardware, "
                "Ollama model generation and embedding operations."
            )

        # ==========================================
        # DOWNLOAD DIAGNOSIS JSON
        # ==========================================

        st.divider()

        st.subheader(
            "Export Diagnosis"
        )

        diagnosis_json = json.dumps(
            diagnosis,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            label="Download Diagnosis JSON",
            data=diagnosis_json,
            file_name="maintain_diagnosis.json",
            mime="application/json",
            use_container_width=True,
        )


# ==================================================
# 7. INITIAL / EMPTY DASHBOARD
# ==================================================

else:

    st.divider()

    st.markdown(
        """
        ### How to use MaintAIn

        1. Upload an industrial equipment telemetry CSV.
        2. Enter the observed equipment symptoms.
        3. Click **Run MaintAIn Analysis**.
        4. Follow the live analysis progress.
        5. Review telemetry anomalies and historical evidence.
        6. Examine possible causes and maintenance recommendations.
        7. Download the generated JSON diagnosis.

        **Note:** AI-generated diagnoses are decision-support
        outputs and require review by qualified maintenance personnel.
        """
    )
