# AES_LLM_LORA

LoRA fine-tuning experiments for Korean automated essay scoring.

## Main script

- `train_llm_token.py`: trains an LLM scoring model with PEFT/LoRA and optional quantization.

## Setup

```bash
pip install -r requirements.txt
```

## Training

Check dataset paths, model name, and output directories in `train_llm_token.py`, then run:

```bash
python train_llm_token.py
```

The script uses Hugging Face Transformers, PEFT, PyTorch, and Weights & Biases for experiment tracking.
