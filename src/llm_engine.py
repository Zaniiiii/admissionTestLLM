import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

class LLMEngine:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.api_token = os.getenv("HF_TOKEN")
        
        if not self.api_token:
            raise ValueError("HF_TOKEN not found in .env file. Please add your Hugging Face token.")
            
        print(f"Initializing Hosted LLM: {self.model_name}...")
        self.client = InferenceClient(model=self.model_name, token=self.api_token)

    def generate(self, messages, max_new_tokens=512):
        """
        Generates response using Hugging Face Inference API.
        Args:
            messages (list): List of dicts [{"role": "user", "content": "..."}, ...]
            max_new_tokens (int): Max tokens to generate
        """
        try:
            # The InferenceClient.chat_completion handles templating automatically
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.7,
                seed=42
            )
            
            # Extract content from response object
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error connecting to Hugging Face API: {str(e)}"