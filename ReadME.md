# Sentiment Analysis Fine-tuning with Qwen2.5

This project fine-tunes the Qwen2.5-1.5B-Instruct model for sentiment analysis using LoRA (Low-Rank Adaptation) and 4-bit quantization for efficient training.

## Project Structure

```
├── config.py             # Configuration classes and parameters
├── data_utils.py         # Data loading and preprocessing utilities
├── model_utils.py        # Model setup and training utilities
├── main.py               # Main training script
├── inference.py          # Standalone inference script
├── requirements.txt      # Python dependencies
├── README.md             
└── environment.yml       # full conda env file
```

## Features

- **Modular Design**: Separated into logical components for maintainability
- **Configurable**: Easy-to-modify configuration classes
- **Memory Efficient**: Uses 4-bit quantization and LoRA for reduced memory usage
- **Flexible Data Loading**: Supports both HuggingFace datasets and local files
- **Comprehensive Testing**: Built-in model testing functionality
- **Standalone Inference**: Separate script for production inference

## Installation

1. Clone the repository or download the files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All configurations are centralized in `config.py`:

- **ModelConfig**: Base model, output directory settings
- **QuantizationConfig**: 4-bit quantization parameters
- **LoRAConfig**: LoRA adaptation parameters
- **TrainingConfig**: Training hyperparameters
- **DataConfig**: Dataset configuration
- **GenerationConfig**: Text generation parameters

## Usage

### Training

Run the main training script (using uv with my conda env ) :

```bash
 uv run main.py
 or 
 python main.py
```

The script will:

1. Load and preprocess the dataset
2. Set up the model with quantization and LoRA
3. Train the model
4. Save the fine-tuned model
5. Run test predictions

### Inference

Use the standalone inference script for predictions:

```bash
# Single text analysis
python inference.py --model_path ./qwen-sentiment-model --text "I love this product!"

# Batch analysis from file
python inference.py --model_path ./qwen-sentiment-model --file texts.txt

# Interactive mode
python inference.py --model_path ./qwen-sentiment-model
```



## Memory Requirements

The optimized version uses several techniques to reduce memory usage:

- **4-bit Quantization**: Reduces model size by ~75%
- **LoRA**: Only trains a small number of parameters
- **Gradient Accumulation**: Allows larger effective batch sizes with limited memory
- **Optional Gradient Checkpointing**: Further memory reduction if needed

**Estimated GPU Memory Usage:**

- Qwen2.5-1.5B with 4-bit quantization: ~1-2GB
- Training overhead: ~2-3GB
- **Total: ~3-5GB GPU memory**


## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable `gradient_checkpointing=True`

2. **Model Not Learning**
   - Check data formatting
   - Adjust learning rate
   - Increase training epochs

3. **Slow Training**
   - Increase batch size if memory allows
   - Use `group_by_length=True`
   - Consider using bf16 instead of fp16 on supported hardware

### Performance Monitoring

The project uses Weights & Biases (Wandb) for monitoring metrics and logs. An API to be required in the terminal.


## License

This project is provided as-is for educational and research purposes. Please ensure compliance with the original model licenses (Qwen2.5) when using in production.

## Contributing

Feel free to submit issues and pull requests to improve the codebase. Areas for contribution:

- Additional model architectures
- More evaluation metrics (Actual metrics like BLEU, Rouge , perplexity, ....)
- Better data preprocessing
- Performance optimizations
