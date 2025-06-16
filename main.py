"""Main training script for sentiment analysis fine-tuning."""

import logging
from transformers import logging as hf_logging

from config import (
    ModelConfig,
    QuantizationConfig,
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    GenerationConfig,
    setup_environment,
)
from data_utils import DataProcessor, get_test_prompts
from model_utils import ModelManager, ModelTester


def main():
    """Main training function."""
    # Setup environment
    setup_environment()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    hf_logging.set_verbosity_info()

    # Initialize configurations
    model_config = ModelConfig()
    quant_config = QuantizationConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    gen_config = GenerationConfig()

    # Initialize managers
    model_manager = ModelManager(
        model_config, quant_config, lora_config, training_config
    )
    data_processor = DataProcessor(data_config)

    try:
        # Check GPU availability
        model_manager.check_gpu_availability()

        # Load and process data
        dataset = data_processor.load_data()
        data_processor.inspect_dataset(dataset)
        train_dataset, eval_dataset = data_processor.prepare_datasets(dataset)

        # Load model and tokenizer
        model, tokenizer = model_manager.load_model_and_tokenizer()

        # Validate tokenization
        data_processor.validate_tokenization(train_dataset, tokenizer)

        # Create PEFT model
        model = model_manager.create_peft_model(model)

        # Create trainer
        trainer = model_manager.create_trainer(model, train_dataset, eval_dataset)

        # Train model
        print("Starting training...")
        trainer.train()

        # Save model
        model_manager.save_model(trainer, tokenizer)

        # Test the model
        tester = ModelTester(model, tokenizer, gen_config)
        test_prompts = get_test_prompts()
        tester.run_test_suite(test_prompts)

        print("\nTraining completed successfully!")

    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
