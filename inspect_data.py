from datasets import load_dataset

def inspect():
    print("Inspecting Personal Data...")
    ds_personal = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
    print("Keys:", ds_personal[0].keys())
    print("Example:", ds_personal[0])

    print("\nInspecting CVE Data...")
    ds_cve = load_dataset("stasvinokur/cve-and-cwe-dataset-1999-2025", split="train")
    print("Keys:", ds_cve[0].keys())
    print("Example:", ds_cve[0])

if __name__ == "__main__":
    inspect()
