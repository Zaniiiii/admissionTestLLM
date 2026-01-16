import argparse
import sys
import time
from src.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Admission Test RAG System")
    parser.add_argument("--query", type=str, help="Single query to run")
    args = parser.parse_args()

    print("==================================================")
    print("   ADMISSION TEST RAG PIPELINE SYSTEM")
    print("==================================================")
    
    try:
        pipeline = RAGPipeline()
        pipeline.initialize_data()
        print("[System] System Ready.")
    except Exception as e:
        print(f"[Error] Failed to initialize: {e}")
        sys.exit(1)

    print("==================================================\n")

    if args.query:
        run_single_query(pipeline, args.query)
    else:
        run_interactive_mode(pipeline)

def run_single_query(pipeline, query):
    process_pipeline_request(pipeline, query)

def run_interactive_mode(pipeline):
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("USER >> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if not user_input:
                continue

            process_pipeline_request(pipeline, user_input)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Error] {e}")

def process_pipeline_request(pipeline, query):
    start_time = time.time()
    
    response = pipeline.run_query(query)
    
    duration = time.time() - start_time
    
    print("\n--- [Final Output] ---")
    print(response)
    print("----------------------")
    print(f"[Metrics] Latency: {duration:.2f}s")
    print("==================================================\n")

if __name__ == "__main__":
    main()
