
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TELEMETRY_DIR = PROJECT_DIR / "data" / "telemetry"

# Illustrative limits for synthetic test data only.
TEMPERATURE_LIMIT_C = 80.0
VIBRATION_LIMIT_MM_S = 7.0


def analyze_telemetry(
    file_name: str = "sample_pump_01.csv",
) -> dict:
    """Analyze telemetry and return structured anomaly results."""

    # Allow only filenames inside data/telemetry.
    if Path(file_name).name != file_name:
        raise ValueError("Provide a filename, not a file path.")

    data_file = (TELEMETRY_DIR / file_name).resolve()

    if data_file.parent != TELEMETRY_DIR.resolve():
        raise ValueError("Invalid telemetry file location.")

    if not data_file.is_file():
        raise FileNotFoundError(
            f"Telemetry file not found: {data_file}"
        )

    df = pd.read_csv(data_file, encoding="utf-8-sig")

    required = [
        "timestamp",
        "equipment_id",
        "temperature_c",
        "vibration_mm_s",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Telemetry CSV contains no readings.")

    for column in ["temperature_c", "vibration_mm_s"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if df[["temperature_c", "vibration_mm_s"]].isna().any().any():
        raise ValueError(
            "Telemetry contains missing or invalid sensor readings."
        )

    df["temperature_anomaly"] = (
        df["temperature_c"] > TEMPERATURE_LIMIT_C
    )
    df["vibration_anomaly"] = (
        df["vibration_mm_s"] > VIBRATION_LIMIT_MM_S
    )
    df["any_anomaly"] = (
        df["temperature_anomaly"] | df["vibration_anomaly"]
    )

    flagged = df.loc[df["any_anomaly"]]

    # Convert pandas values to ordinary Python values for JSON.
    flagged_readings = []

    for _, row in flagged.iterrows():
        flagged_readings.append({
            "timestamp": str(row["timestamp"]),
            "equipment_id": str(row["equipment_id"]),
            "temperature_c": float(row["temperature_c"]),
            "vibration_mm_s": float(row["vibration_mm_s"]),
            "temperature_anomaly": bool(
                row["temperature_anomaly"]
            ),
            "vibration_anomaly": bool(
                row["vibration_anomaly"]
            ),
        })

    return {
        "equipment_id": str(df["equipment_id"].iloc[0]),
        "total_readings": int(len(df)),
        "temperature_limit_c": TEMPERATURE_LIMIT_C,
        "vibration_limit_mm_s": VIBRATION_LIMIT_MM_S,
        "temperature_exceedances": int(
            df["temperature_anomaly"].sum()
        ),
        "vibration_exceedances": int(
            df["vibration_anomaly"].sum()
        ),
        "flagged_reading_count": int(len(flagged)),
        "first_flagged_timestamp": (
            str(flagged.iloc[0]["timestamp"])
            if not flagged.empty else None
        ),
        "flagged_readings": flagged_readings,
        "note": (
            "Threshold-based flags only. These illustrative limits "
            "are not manufacturer-approved safety limits and do not "
            "confirm an equipment fault."
        ),
    }


if __name__ == "__main__":
    import json

    result = analyze_telemetry()
    print(json.dumps(result, indent=2))
