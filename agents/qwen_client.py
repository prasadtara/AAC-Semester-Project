import os
import requests

class QwenClient:
    def __init__(self, mode="colab", model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.mode = mode
        self.model_name = model_name
        
        if self.mode == "colab":
            print(f"Loading {self.model_name} in Colab with 4-bit quantization...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Configure 4-bit quantization to prevent memory bottlenecks
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
        elif self.mode == "local":
            # Automatically map HuggingFace model name to Ollama tag format
            if "7b" in self.model_name.lower():
                self.local_model = "qwen2.5:7b"
            elif "14b" in self.model_name.lower():
                self.local_model = "qwen2.5:14b"
            elif "32b" in self.model_name.lower():
                self.local_model = "qwen2.5:32b"
            else:
                self.local_model = "qwen2.5:1.5b"
                
            self.url = "http://localhost:11434/api/generate"
            print(f"Using local Ollama model: {self.local_model}")

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        # Ensure thinking mode is turned off 
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
