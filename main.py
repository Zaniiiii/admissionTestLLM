import sys
from src.pipeline import RAGPipeline

def main():
    rag = RAGPipeline()
    
    # Check if we need to ingest data (simple check)
    # in real app, maybe use~ a flag or check DB
    rag.initialize_data()
    
    print("\n--- TEST SYSTEM READY (Research RAG) ---")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            q = input("\nEnter Query: ")
            if q.lower() in ['exit', 'quit']: break
            if not q.strip(): continue
            
            ans = rag.run_query(q)
            print(f"\nAI: {ans}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
