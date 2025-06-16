"""Model setup and training utilities."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from typing import Tuple

from config import (
    ModelConfig,
    QuantizationConfig,
    LoRAConfig,
    TrainingConfig,
    GenerationConfig,
)


class ModelManager:
    """Manages model loading, configuration, and training."""

    def __init__(
        self,
        model_config: ModelConfig,
        quant_config: QuantizationConfig,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
    ):
        self.model_config = model_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        self.training_config = training_config

    def check_gpu_availability(self) -> None:
        """Check and print GPU availability."""
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

    def create_quantization_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytesConfig for quantization."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        return BitsAndBytesConfig(
            load_in_4bit=self.quant_config.load_in_4bit,
            bnb_4bit_quant_type=self.quant_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype_map.get(
                self.quant_config.bnb_4bit_compute_dtype, torch.float16
            ),
            bnb_4bit_use_double_quant=self.quant_config.bnb_4bit_use_double_quant,
        )

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load and configure model and tokenizer."""
        print("Loading model and tokenizer...")

        bnb_config = self.create_quantization_config()

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.model_config.trust_remote_code,
            use_cache=self.model_config.use_cache,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer

    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        return LoraConfig(
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            r=self.lora_config.r,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type,
            target_modules=self.lora_config.target_modules,
        )

    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments."""
        return TrainingArguments(
            output_dir=self.model_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            optim=self.training_config.optim,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            max_grad_norm=self.training_config.max_grad_norm,
            max_steps=self.training_config.max_steps,
            warmup_ratio=self.training_config.warmup_ratio,
            group_by_length=self.training_config.group_by_length,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            logging_strategy="steps",
            logging_steps=self.training_config.logging_steps,
            save_strategy="steps",
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            eval_strategy="steps",
            eval_steps=self.training_config.eval_steps,
            dataloader_pin_memory=self.training_config.dataloader_pin_memory,
            remove_unused_columns=self.training_config.remove_unused_columns,
            report_to=self.training_config.report_to,
        )

    def create_peft_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """Create PEFT model with LoRA configuration."""
        print("Creating PEFT model...")
        peft_config = self.create_lora_config()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def create_trainer(
        self, model: AutoModelForCausalLM, train_dataset: Dataset, eval_dataset: Dataset
    ) -> SFTTrainer:
        """Create SFT trainer."""
        training_args = self.create_training_arguments()

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        return trainer

    def save_model(self, trainer: SFTTrainer, tokenizer: AutoTokenizer) -> None:
        """Save the trained model and tokenizer."""
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.model_config.output_dir)
        print(f"Model saved to: {self.model_config.output_dir}")


class ModelTester:
    """Handles model testing and inference."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        gen_config: GenerationConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config

    def test_model(self, prompt_text: str) -> str:
        """Test the fine-tuned model with a prompt."""
        formatted_prompt = (
            f"[INST] Analyze the sentiment of this text: {prompt_text} [/INST]"
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_config.max_new_tokens,
                temperature=self.gen_config.temperature,
                do_sample=self.gen_config.do_sample,
                top_p=self.gen_config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        response = response[len(formatted_prompt) :].strip()
        return response

    def run_test_suite(self, test_prompts: list) -> None:
        """Run a suite of tests on the model."""
        print("\n" + "=" * 60)
        print("Testing the fine-tuned model:")
        print("=" * 60)

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            try:
                response = self.test_model(prompt)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
            print("-" * 50)
