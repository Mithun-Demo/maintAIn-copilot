
import argparse
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


# ==================================================
# 1. CONFIGURATION
# ==================================================

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


# ==================================================
# 2. LOAD HISTORICAL DOCUMENTS
# ==================================================

def load_historical_documents():
    """Load historical failure records as LangChain documents."""

    if not HISTORY_FILE.is_file():
        raise FileNotFoundError(
            f"Historical dataset not found: {HISTORY_FILE}"
        )

    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError(
            "Historical dataset must contain a non-empty JSON list."
        )

    documents = []

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid historical record.")

        required_fields = [
            "case_id",
            "equipment_type",
            "symptoms",
            "possible_cause",
            "recommended_checks",
            "source_type",
        ]

        missing = [
            field for field in required_fields
            if field not in record
        ]

        if missing:
            raise ValueError(
                f"Historical record is missing fields: {missing}"
            )

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
                    "case_id": str(record["case_id"]),
                    "source_type": str(record["source_type"]),
                    "equipment_type": str(record["equipment_type"]),
                },
            )
        )

    return documents


# ==================================================
# 3. CREATE THE CHROMA DATABASE CONNECTION
# ==================================================

def create_vector_store():
    """Connect to the persistent ChromaDB collection."""

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return vector_store


# ==================================================
# 4. REBUILD THE VECTOR INDEX
# ==================================================

def build_vector_store():
    """
    Force a complete rebuild of the vector index.

    Use this after deliberately changing the historical dataset.
    """

    documents = load_historical_documents()

    vector_store = create_vector_store()

    existing = vector_store.get()
    existing_ids = existing.get("ids", [])

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    document_ids = [
        document.metadata["case_id"]
        for document in documents
    ]

    vector_store.add_documents(
        documents,
        ids=document_ids,
    )

    print(
        f"Vector index rebuilt successfully: "
        f"{len(documents)} historical records."
    )

    return vector_store


# ==================================================
# 5. REUSE OR SYNCHRONIZE THE PERSISTENT INDEX
# ==================================================

def get_vector_store():
    """
    Reuse the existing index when its documents match the dataset.

    Rebuild the index only when it is empty or its records change.
    """

    documents = load_historical_documents()

    vector_store = create_vector_store()

    existing = vector_store.get(
        include=["documents"]
    )

    existing_ids = existing.get("ids", [])
    existing_documents = existing.get("documents", [])

    existing_by_id = dict(
        zip(existing_ids, existing_documents)
    )

    expected_by_id = {
        document.metadata["case_id"]: document.page_content
        for document in documents
    }

    if existing_by_id == expected_by_id:
        return vector_store

    print("Historical data changed or index is empty.")
    print("Updating the ChromaDB index...")

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    document_ids = [
        document.metadata["case_id"]
        for document in documents
    ]

    vector_store.add_documents(
        documents,
        ids=document_ids,
    )

    print(
        f"Vector index synchronized: "
        f"{len(documents)} historical records."
    )

    return vector_store


# ==================================================
# 6. SEMANTIC VECTOR SEARCH
# ==================================================

def search_similar_cases(query: str, k: int = 3):
    """
    Retrieve historical maintenance cases using vector similarity.

    The persistent index is reused when it is already up to date.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Search query must not be empty."
        )

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(
            "k must be a positive integer."
        )

    vector_store = get_vector_store()

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


# ==================================================
# 7. COMMAND-LINE TEST
# ==================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MaintAIn historical vector retrieval"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the ChromaDB vector index.",
    )

    args = parser.parse_args()

    if args.rebuild:
        build_vector_store()

    test_query = (
        "Pump has high vibration and increasing temperature"
    )

    result = search_similar_cases(
        test_query,
        k=3,
    )

    print("\n=== VECTOR SEARCH RESULTS ===")
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
