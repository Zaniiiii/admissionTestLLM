import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        print(f"Loading LLM: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # MPS (Mac) is causing driver crashes (NDArray > 2^32) even for small models.
        # Since Qwen-0.5B is tiny, CPU is fast enough and 100% stable.
        device = "cpu"
        print(f"Using device: {device} (Forced for Stability)")
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(device)
        except Exception as e:
            print(f"Error loading model to {device}, falling back to CPU. Error: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
    def generate(self, messages, max_new_tokens=200):
        # Use apply_chat_template for correct formatting (Qwen/Phi/Llama)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Check input length
        input_len = inputs["input_ids"].shape[1]
        print(f"Input tokens: {input_len}")
        
        if input_len > 4000:
            print(f"Warning: Input context too long ({input_len}), truncating...")
             # Basic truncation (unsafe for chat templates but prevents crash)
            inputs["input_ids"] = inputs["input_ids"][:, -4000:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -4000:]

        # Add no_grad to save memory for inference
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True, 
                temperature=0.7,
                repetition_penalty=1.1, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode ONLY the new tokens (exclude input prompt)
        generated_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text