# LoRA Run Notes

Use this checklist when running `train_llm_token.py` for Korean AES LoRA fine-tuning.

## Before Training

- Confirm dataset paths and output directories in the script.
- Record the base model name or local path.
- Check LoRA rank, alpha, dropout, and target modules.
- Note quantization settings and available GPU memory.
- Confirm W&B project and run names before launching long jobs.

## After Training

Save the adapter path, training log, final metrics, and inference command together. If the run is resumed, record the original checkpoint and resumed command.

## Comparison

Compare LoRA runs on the same held-out split and prompt template. Include the base model revision whenever reporting scores.
