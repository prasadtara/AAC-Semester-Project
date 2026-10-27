import os
import requests

class QwenClient:
    def __init__(self, mode="colab", model_name="Qwen/Qwen2.5-1.5B"):
        self.mode = mode
        self.model_name = model_name
        
        if self.mode == "colab":
            print(f"Loading {self.model_name} in Colab...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load in standard half precision
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto"
            )
            
        elif self.mode == "local":
            # Map HuggingFace model name to Ollama tag format
            model_lower = self.model_name.lower()
            if "32b" in model_lower:
                self.local_model = "qwen3:32b"
            elif "14b" in model_lower:
                self.local_model = "qwen3:14b"
            elif "8b" in model_lower:
                self.local_model = "qwen3:8b"
            elif "4b" in model_lower:
                self.local_model = "qwen3:4b"
            else:
                self.local_model = "qwen3:1.7b"
                
            self.url = "http://localhost:11434/api/generate"
            print(f"Using local Ollama model: {self.local_model}")

   def generate(self, prompt: str, max_tokens: int = 150, enable_thinking: bool = False) -> str:
        if self.mode == "colab":
            messages = [{"role": "user", "content": prompt}]
            
            # Format prompt with chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            
            # Tokenize prompt
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            prompt_length = inputs["input_ids"].shape[1]
            
            # Generate new tokens
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7
            )
            
            # Slice only the newly generated tokens (skipping the prompt tokens)
            generated_tokens = outputs[0][prompt_length:]
            
            # Decode generated output
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clean up thinking tags if present
            if "</think>" in response_text:
                response_text = response_text.split("</think>")[-1].strip()
                
            return response_text
            
        elif self.mode == "local":
            formatted_prompt = prompt if enable_thinking else f"/no_think {prompt}"
            
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
