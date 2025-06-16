"""Standalone inference script for the fine-tuned sentiment model."""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional


class SentimentInference:
    """Handles inference for the fine-tuned sentiment model."""
    
    def __init__(self, model_path: str, base_model: Optional[str] = None):
        self.model_path = model_path
        self.base_model = base_model or "Qwen/Qwen2.5-1.5B-Instruct"
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load PEFT model
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
    
    def predict_sentiment(self, text: str, max_new_tokens: int = 50, 
                         temperature: float = 0.7) -> str:
        """Predict sentiment for given text."""
        formatted_prompt = f"[INST] Analyze the sentiment of this text: {text} [/INST]"
        
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        response = response[len(formatted_prompt):].strip()
        return response
    
    def batch_predict(self, texts: list, max_new_tokens: int = 50, 
                     temperature: float = 0.7) -> list:
        """Predict sentiment for a batch of texts."""
        results = []
        for text in texts:
            try:
                result = self.predict_sentiment(text, max_new_tokens, temperature)
                results.append({"text": text, "sentiment": result})
            except Exception as e:
                results.append({"text": text, "error": str(e)})
        return results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Sentiment Analysis Inference")
    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct", 
                       help="Base model name")
    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--file", help="File containing texts to analyze (one per line)")
    parser.add_argument("--max_tokens", type=int, default=50, 
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = SentimentInference(args.model_path, args.base_model)
    
    if args.text:
        # Single text analysis
        result = inference.predict_sentiment(args.text, args.max_tokens, args.temperature)
        print(f"Text: {args.text}")
        print(f"Sentiment: {result}")
    
    elif args.file:
        # Batch analysis from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            results = inference.batch_predict(texts, args.max_tokens, args.temperature)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Text: {result['text']}")
                if 'sentiment' in result:
                    print(f"   Sentiment: {result['sentiment']}")
                else:
                    print(f"   Error: {result['error']}")
        
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
    
    else:
        # Interactive mode
        print("Interactive mode - Enter text to analyze (press Ctrl+C to exit)")
        try:
            while True:
                text = input("\nEnter text: ").strip()
                if text:
                    result = inference.predict_sentiment(text, args.max_tokens, args.temperature)
                    print(f"Sentiment: {result}")
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()


# Example usage:
# python inference.py --model_path ./qwen-sentiment-model --text "I love this product!"
# python inference.py --model_path ./qwen-sentiment-model --file texts.txt
# python inference.py --model_path ./qwen-sentiment-model  # Interactive mode