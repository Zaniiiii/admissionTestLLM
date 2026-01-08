import os
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM

class PrivacyRAG:
    def __init__(self, persist_dir="./chroma_db", model_name="microsoft/phi-2"):
        self.persist_dir = persist_dir
        print("Initializing ChromaDB Client...")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="united_knowledge_base")
        
        print("Loading Embedding Model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def ingest_data(self):
        """
        Downloads datasets, slices them according to rules, and ingests into ChromaDB.
        Rule: Top-100 Personal Data, Last-200 CVE Data.
        """
        if self.collection.count() > 0:
            print(f"Collection already contains {self.collection.count()} items. Skipping ingestion.")
            return

        print("Downloading Datasets...")
        # 1. Personal Data (nvidia/Nemotron-Personas-USA)
        ds_personal = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
        # Take first 100
        personal_subset = ds_personal.select(range(100))
        
        # 2. CVE Data (stasvinokur/cve-and-cwe-dataset-1999-2025)
        # Verify split name, usually 'train'
        ds_cve = load_dataset("stasvinokur/cve-and-cwe-dataset-1999-2025", split="train")
        # Take last 200
        total_cve = len(ds_cve)
        cve_subset = ds_cve.select(range(total_cve - 200, total_cve))

        documents = []
        metadatas = []
        ids = []

        print("Processing Personal Data...")
        for i, item in enumerate(personal_subset):
            # Construct a rich text representation
            text = f"PERSONAL DATA RECORD:\nName: {item.get('name', 'N/A')}\nData: {item}" 
            documents.append(text)
            metadatas.append({"source": "personal_sensitive", "type": "pii"})
            ids.append(f"pii_{i}")

        print("Processing CVE Data...")
        for i, item in enumerate(cve_subset):
            text = f"CVE SECURITY RECORD:\nID: {item.get('cve_id', 'N/A')}\nDescription: {item.get('description', 'N/A')}"
            documents.append(text)
            metadatas.append({"source": "cve_public", "type": "security"})
            ids.append(f"cve_{i}")

        print(f"Embedding and storing {len(documents)} items (This may take a moment)...")
        embeddings = self.embedder.encode(documents).tolist()
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("Ingestion Complete!")

    def load_llm(self):
        print(f"Loading LLM: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # Check for MPS (Mac) availability
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load model with mixed precision if possible to save memory
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if device != "cpu" else torch.float32, 
                trust_remote_code=True
            ).to(device)
        except Exception as e:
            print(f"Error loading model to {device}, falling back to CPU/Float32. Error: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )

    def retrieve(self, query, k=3):
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        return results['documents'][0]

    def generate_safe_response(self, query):
        if not self.model:
            self.load_llm()
            
        retrieved_docs = self.retrieve(query)
        context_str = "\n---\n".join(retrieved_docs)
        
        # SYSTEM PROMPT (The Guardrail)
        system_prompt = f"""
        Instruct: You are a secure AI assistant. You have access to the following context to answer the user request.
        
        CRITICAL PRIVACY RULES:
        1. If the context contains PERSONAL IDENTIFIABLE INFORMATION (PII) such as Names, Phones, Addresses, SSNs, credit cards, etc., you MUST REFUSE to disclose it.
        2. Instead, reply: "[PRIVACY BLOCK] I cannot verify this individual's data due to privacy regulations."
        3. If the context contains CVE/Cybersecurity technical data, you SHOULD answer detailedly.
        
        CONTEXT:
        {context_str}
        
        USER REQUEST: {query}
        
        ASSISTANT RESPONSE:
        """
        
        inputs = self.tokenizer(system_prompt, return_tensors="pt", return_attention_mask=False)
        # Move inputs to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs, 
            max_length=500, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Cut off the prompt part to show only the answer (heuristic)
        response = text.split("ASSISTANT RESPONSE:")[-1].strip()
        return response

if __name__ == "__main__":
    rag = PrivacyRAG()
    rag.ingest_data()
    
    print("\n--- TEST SYSTEM READY ---")
    while True:
        q = input("\nEnter Query (or 'exit'): ")
        if q.lower() == 'exit': break
        
        print("Thinking...")
        ans = rag.generate_safe_response(q)
        print(f"\nAI: {ans}")
