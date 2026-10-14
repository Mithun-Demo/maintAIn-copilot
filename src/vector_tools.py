
import json

from langchain.tools import tool
from src.vector_retrieval import search_similar_cases


@tool
def search_similar_historical_cases(query: str) -> str:
    """Search synthetic maintenance history using semantic vector
    similarity. Use this to find cases related in meaning to current
    equipment symptoms. Similarity does not prove a fault."""

    result = search_similar_cases(query, k=3)
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    print(
        search_similar_historical_cases.invoke({
            "query": "Pump has high vibration and increasing temperature"
        })
    )
