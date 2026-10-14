
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

PROJECT_DIR = Path(__file__).resolve().parents[1]

HISTORY_FILE = (
    PROJECT_DIR
    / "data"
    / "failure_history"
    / "historical_failures.json"
)

CHROMA_DIR = PROJECT_DIR / "data" / "chroma_db"

COLLECTION_NAME = "maintain_historical_failures"
EMBEDDING_MODEL = "nomic-embed-text"


def load_historical_documents():
    """Convert historical JSON records into searchable documents."""

    if not HISTORY_FILE.is_file():
        raise FileNotFoundError(
            f"Historical dataset not found: {HISTORY_FILE}"
        )

    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    documents = []

    for record in records:
        content = (
            f"Equipment type: {record['equipment_type']}\n"
            f"Symptoms: {', '.join(record['symptoms'])}\n"
            f"Possible cause: {record['possible_cause']}\n"
            f"Recommended checks: "
            f"{', '.join(record['recommended_checks'])}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "case_id": record["case_id"],
                    "source_type": record["source_type"],
                    "equipment_type": record["equipment_type"],
                },
            )
        )

    if not documents:
        raise ValueError("The historical dataset contains no records.")

    return documents


def build_vector_store():
    """Build or refresh the local ChromaDB collection."""

    documents = load_historical_documents()

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # This project currently uses a small, fixed demo dataset.
    # Clear the collection so changed records are not left stale.
    existing = vector_store.get()
    existing_ids = existing.get("ids", [])

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    vector_store.add_documents(documents)

    return vector_store


def search_similar_cases(query: str, k: int = 3):
    """Return historical cases most semantically similar to a query."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    vector_store = build_vector_store()

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )

    matches = []

    for document, distance in results:
        matches.append({
            "case_id": document.metadata["case_id"],
            "equipment_type": document.metadata["equipment_type"],
            "content": document.page_content,
            "source_type": document.metadata["source_type"],
            "distance": float(distance),
        })

    return {
        "query": query,
        "retrieval_method": "ChromaDB vector similarity",
        "matches": matches,
        "note": (
            "Historical cases are synthetic demo records. "
            "Similarity is not proof of the current fault. "
            "Distance is model-dependent and is not a probability."
        ),
    }


if __name__ == "__main__":
    result = search_similar_cases(
        "Pump has high vibration and increasing temperature",
        k=3,
    )
    print(json.dumps(result, indent=2))
