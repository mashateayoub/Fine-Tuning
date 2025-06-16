"""Data processing utilities for sentiment analysis fine-tuning."""

import json
from typing import Dict, Any, Tuple, Optional
from datasets import load_dataset, Dataset
from config import DataConfig


class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def load_data(self) -> Dataset:
        """Load dataset from HuggingFace or local file."""
        print("Loading dataset...")
        
        if self.config.dataset_path:
            # Load from local JSON file
            dataset = Dataset.from_json(self.config.dataset_path)
        else:
            # Load from HuggingFace
            dataset = load_dataset(self.config.dataset_name)
        
        return dataset
    
    def inspect_dataset(self, dataset: Dataset) -> None:
        """Inspect and print dataset structure."""
        print("Dataset columns:", 
              dataset["train"].column_names if "train" in dataset else list(dataset.keys()))
        
        print("Sample data structure:")
        sample = (dataset["train"][0] if "train" in dataset 
                 else dataset[list(dataset.keys())[0]][0])
        print(json.dumps(sample, indent=2)[:500] + "...")
    
    def format_sentiment_data(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format sentiment data for instruction tuning."""
        text = example["text"]
        
        # Handle different label formats
        if isinstance(example["label"], int):
            label = "positive" if example["label"] == self.config.positive_label else "negative"
        else:
            label = example["label"].lower()
        
        formatted_text = (
            f"[INST] Analyze the sentiment of this text: {text} [/INST]\n"
            f"The sentiment of this text is {label}."
        )
        return {"text": formatted_text}
    
    def prepare_datasets(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Process and split datasets for training and evaluation."""
        print("Processing datasets...")
        
        # Format training data
        if "train" in dataset:
            train_dataset = dataset["train"].map(
                self.format_sentiment_data, 
                remove_columns=dataset["train"].column_names
            )
        else:
            # If no train split, use the entire dataset and split later
            full_dataset = dataset[list(dataset.keys())[0]]
            train_dataset = full_dataset.map(
                self.format_sentiment_data,
                remove_columns=full_dataset.column_names
            )
        
        # Handle evaluation dataset
        eval_dataset = None
        if "test" in dataset:
            eval_dataset = dataset["test"].map(
                self.format_sentiment_data,
                remove_columns=dataset["test"].column_names
            )
        elif "validation" in dataset:
            eval_dataset = dataset["validation"].map(
                self.format_sentiment_data,
                remove_columns=dataset["validation"].column_names
            )
        else:
            # Split training data
            train_test_split = train_dataset.train_test_split(
                test_size=self.config.test_split_size, 
                seed=self.config.seed if hasattr(self.config, 'seed') else 42
            )
            train_dataset = train_test_split["train"]
            eval_dataset = train_test_split["test"]
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")
        print("\nSample formatted text:")
        print("-" * 50)
        print(train_dataset[0]["text"][:300] + "...")
        print("-" * 50)
        
        return train_dataset, eval_dataset
    
    def validate_tokenization(self, train_dataset: Dataset, tokenizer) -> None:
        """Validate tokenization process."""
        sample = train_dataset[0]["text"]
        tokens = tokenizer(sample, truncation=True, padding=True)
        print("Tokenized sample keys:", tokens.keys())
        print("Input length:", len(tokens["input_ids"]))
        print("Decoded sample:", tokenizer.decode(tokens["input_ids"])[:200] + "...")


def get_test_prompts() -> list:
    """Return a list of test prompts for model evaluation."""
    return [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "The weather is okay today.",
        "This movie was absolutely fantastic!",
        "I'm disappointed with the service.",
        "It's an average experience, nothing special."
    ]