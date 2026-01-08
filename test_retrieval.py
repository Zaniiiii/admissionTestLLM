from src.vector_store import VectorStore
from src.data_ingestion import load_and_slice_datasets, prepare_documents

def test_retrieval():
    print("=== TESTING RETRIEVAL SYSTEM ONLY ===")
    
    # 1. Initialize DB
    store = VectorStore(persist_dir="./data/chroma_db")
    
    # 2. Check content (Auto-Ingest if empty)
    if store.collection.count() == 0:
        print("DB Empty. Ingesting data for test...")
        personal, cve = load_and_slice_datasets()
        docs, metas, ids = prepare_documents(personal, cve)
        store.add_documents(docs, metas, ids)
    else:
        print(f"DB Content Count: {store.collection.count()} items.")

    # 3. Test Cases
    queries = [
        "What is CVE-2025-5331",  # Should retrieve CVE
        "Does anyone use password '123456'?", # Should retrieve Person (maybe)
        "Who is Alicia Gonzalez?", # Should retrieve Person
    ]
    
    for q in queries:
        print(f"\nQUERY: {q}")
        print("-" * 30)
        results = store.query(q, k=2)
        for i, doc in enumerate(results):
            print(f"Result {i+1}: {doc[:200]}...") # Print first 200 chars
        print("-" * 30)

if __name__ == "__main__":
    test_retrieval()
