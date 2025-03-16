import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from kobert_transformers import get_tokenizer
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
import logging
from models.aes_kobert import KoBERTForSequenceRegression
import wandb
from datetime import datetime
from datasets import load_from_disk

def load_model():
    model = KoBERTForSequenceRegression()
    return model

def data_collator(features):
    input_ids = torch.tensor([f["input_ids"] for f in features])
    attention_mask = torch.tensor([f["attention_mask"] for f in features])
    
    # 🔹 labels을 리스트 → Tensor 변환
    labels = torch.tensor([f["label"] for f in features], dtype=torch.float)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def llm_train():
    wb_token = "8b738bb3f5650780015aa6c3d98a2c811b470916"
    wandb.login(key=wb_token)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(project="kobert_aes", name = formatted_datetime, reinit=True)

    dataset_path = './aes_dataset'

    model = load_model()
    tokenizer = get_tokenizer()

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    dataset = load_from_disk(dataset_path)
    tokenized_dataset = dataset.map(tokenize_function,batched=True)
    
    training_args = TrainingArguments(
        output_dir='./kobert_results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, 
                                        early_stopping_threshold=0.0)]
    )

    trainer.train()
    trainer.save_model("./essay_scorer_model")


if __name__ == "__main__":
    llm_train()
