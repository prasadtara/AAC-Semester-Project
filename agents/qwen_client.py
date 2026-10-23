import os
import requests

class QwenClient:
    def __init__(self, mode="colab", model_name="Qwen/Qwen3-1.7B"):
        self.mode = mode
        self.model_name = model_name
        
        if self.mode == "colab":
            print(f"Loading {self.model_name} in Colab using transformers...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif self.mode == "local":
            # Ollama local endpoint used by your partner
            self.url = "http://localhost:11434/api/generate"
            self.local_model = "qwen3:1.7b"

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        # Ensure thinking mode is turned off as specified in your weekly plan
        formatted_prompt = f"/no_think {prompt}"
        
        if self.mode == "colab":
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7
            )
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response_text[len(formatted_prompt):].strip()
            
        elif self.mode == "local":
            payload = {
                "model": self.local_model,
                "prompt": formatted_prompt,
                "stream": False
            }
            res = requests.post(self.url, json=payload)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
            else:
                raise RuntimeError(f"Ollama local error: {res.text}")
