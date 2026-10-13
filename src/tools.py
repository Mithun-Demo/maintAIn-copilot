
import json
from pathlib import Path

from langchain.tools import tool
from src.anomaly_detection import analyze_telemetry


@tool
def analyze_telemetry_csv(
    file_name: str = "sample_pump_01.csv",
) -> str:
    """Analyze an industrial equipment telemetry CSV.

    Use this tool to identify readings exceeding the configured
    illustrative temperature and vibration thresholds.

    The CSV must be located in the project's data/telemetry folder.
    Returns structured results including flagged timestamps.
    Threshold exceedances do not confirm a machine fault.
    """

    # Ensure the input is a filename, even if the model provides a path.
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("Please provide a valid telemetry filename.")

    safe_file_name = Path(file_name.strip()).name

    if (
        not safe_file_name
        or safe_file_name in {".", ".."}
        or not safe_file_name.lower().endswith(".csv")
    ):
        raise ValueError(
            "Please provide a valid CSV filename, "
            "such as sample_pump_01.csv."
        )

    # The analysis function reads only from data/telemetry.
    results = analyze_telemetry(safe_file_name)

    return json.dumps(results, indent=2)


# Tools available to the MaintAIn agent.
maintenance_tools = [
    analyze_telemetry_csv,
]


if __name__ == "__main__":
    result = analyze_telemetry_csv.invoke({
        "file_name": "sample_pump_01.csv"
    })

    print(result)
