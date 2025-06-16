"""Configuration module for sentiment analysis fine-tuning."""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./qwen-sentiment-model"
    trust_remote_code: bool = True
    use_cache: bool = False


@dataclass
class QuantizationConfig:
    """Quantization configuration for BitsAndBytesConfig."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration parameters."""

    lora_alpha: int = 16
    lora_dropout: float = 0.1
    r: int = 64
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = False
    optim: str = "paged_adamw_32bit"
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    fp16: bool = True
    bf16: bool = False
    max_grad_norm: float = 0.3
    max_steps: int = -1
    warmup_ratio: float = 0.03
    group_by_length: bool = True
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 100
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    test_split_size: float = 0.1
    seed: int = 42
    report_to: str = "wandb"


@dataclass
class DataConfig:
    """Data configuration parameters."""

    dataset_name: str = "mteb/tweet_sentiment_extraction"
    dataset_path: Optional[str] = None  # For local files
    positive_label: int = 1
    negative_label: int = 0


@dataclass
class GenerationConfig:
    """Text generation configuration."""

    max_new_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9


def setup_environment():
    """Set up environment variables and logging."""
    # os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases if not needed
    # Add other environment setup here if needed
