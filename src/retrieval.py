
import json
from pathlib import Path

from langchain.tools import tool

PROJECT_DIR = Path(__file__).resolve().parents[1]
HISTORY_FILE = (
    PROJECT_DIR
    / "data"
    / "failure_history"
    / "historical_failures.json"
)


@tool
def search_historical_failures(query: str) -> str:
    """Search synthetic historical maintenance cases for symptoms
    similar to a reported equipment problem. Returns possible
    comparisons, not confirmed diagnoses."""

    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "matches": [],
            "message": "Provide symptoms or keywords to search."
        })

    if not HISTORY_FILE.is_file():
        raise FileNotFoundError(
            f"Historical dataset not found: {HISTORY_FILE}"
        )

    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    terms = set(query.lower().replace("-", " ").split())
    results = []

    for record in records:
        searchable_text = " ".join([
            record["equipment_type"],
            *record["symptoms"],
            record["possible_cause"],
            *record["recommended_checks"]
        ]).lower().replace("-", " ")

        searchable_terms = set(searchable_text.split())
        matched_terms = terms & searchable_terms

        # Avoid returning cases with no matching keywords.
        if matched_terms:
            results.append({
                "case_id": record["case_id"],
                "equipment_type": record["equipment_type"],
                "matched_terms": sorted(matched_terms),
                "symptoms": record["symptoms"],
                "possible_cause": record["possible_cause"],
                "recommended_checks": record["recommended_checks"],
                "source_type": record["source_type"]
            })

    results.sort(
        key=lambda item: len(item["matched_terms"]),
        reverse=True
    )

    return json.dumps({
        "query": query,
        "matching_method": "keyword_overlap",
        "matches": results[:3],
        "note": (
            "Records are synthetic demo examples. "
            "Keyword overlap does not establish causation."
        )
    }, indent=2)


if __name__ == "__main__":
    print(
        search_historical_failures.invoke({
            "query": "high vibration rising temperature"
        })
    )
