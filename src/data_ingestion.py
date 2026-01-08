from datasets import load_dataset

def load_and_slice_datasets():
    print("Downloading Datasets...")
    # 1. Personal Data (nvidia/Nemotron-Personas-USA)
    ds_personal = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
    # Take first 100
    personal_subset = ds_personal.select(range(100))
    
    # 2. CVE Data (stasvinokur/cve-and-cwe-dataset-1999-2025)
    ds_cve = load_dataset("stasvinokur/cve-and-cwe-dataset-1999-2025", split="train")
    # Take last 200
    total_cve = len(ds_cve)
    cve_subset = ds_cve.select(range(total_cve - 200, total_cve))
    
    print(f"Loaded: {len(personal_subset)} Personal records, {len(cve_subset)} CVE records.")
    return personal_subset, cve_subset

def prepare_documents(personal_subset, cve_subset):
    documents = []
    metadatas = []
    ids = []

    print("Processing Personal Data...")
    for i, item in enumerate(personal_subset):
        # Key 'professional_persona' contains the text description
        persona_text = item.get('professional_persona', 'N/A')
        text = f"PERSONAL DATA RECORD:\nInfo: {persona_text}\nData: {item}" 
        documents.append(text)
        metadatas.append({"source": "personal_sensitive", "type": "pii"})
        ids.append(f"pii_{i}")

    print("Processing CVE Data...")
    for i, item in enumerate(cve_subset):
        # Keys are uppercase: CVE-ID, DESCRIPTION
        cve_id = item.get('CVE-ID', 'N/A')
        desc = item.get('DESCRIPTION', 'N/A')
        text = f"CVE SECURITY RECORD:\nID: {cve_id}\nDescription: {desc}"
        documents.append(text)
        metadatas.append({"source": "cve_public", "type": "security"})
        ids.append(f"cve_{i}")
        
    return documents, metadatas, ids
