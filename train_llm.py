from models.aes_llama import LlamaForSequenceRegression
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
import torch
import wandb
from datetime import datetime

class SaveBestPeftModelCallback(TrainerCallback):
    """
    평가 시점마다 eval_loss를 확인하고,
    이전까지의 최소 eval_loss보다 작으면 모델(LoRA+Regressor) 가중치 저장
    """
    def __init__(self, model):
        super().__init__()
        self.model = model                # 콜백에 모델을 저장
        self.best_eval_loss = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            print(f"Best eval_loss 갱신: {self.best_eval_loss:.4f} -> {eval_loss:.4f}")
            self.best_eval_loss = eval_loss

            # LoRA 가중치 저장
            self.model.model.save_pretrained("./model_weights/lora_weights")

            # Regressor 가중치 저장
            torch.save(
                self.model.regressor.state_dict(),
                "./model_weights/regressor_weights.pth"
            )

        return control

def load_model(model_name):
    model = LlamaForSequenceRegression(model_name=model_name)
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
    wandb.init(project="automated_essay_scoring", name = formatted_datetime, reinit=True)


    model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    dataset_path = './aes_dataset'

    model = load_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )

    
    dataset = load_from_disk(dataset_path)
    tokenized_dataset = dataset.map(tokenize_function,batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=1,
        optim="paged_adamw_8bit",
        evaluation_strategy="epoch",  # 🔹 Evaluation 수행
        save_strategy="epoch",
        save_safetensors=False,# 🔹 저장은 하되, 훈련 후 삭제 가능
        load_best_model_at_end=True,  # 🔹 Best Model 로드
        save_total_limit=1  # 🔹 가장 좋은 모델 하나만 유지
    )


    # 11. 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        data_collator=data_collator,
        callbacks=[SaveBestPeftModelCallback(model), EarlyStoppingCallback(early_stopping_patience=5, 
                                                                           early_stopping_threshold=0.0)]
    )

    # 12. 학습 실행
    trainer.train()

    model.model.save_pretrained("./model_weights/lora_weights")

    # Regressor 가중치 저장
    torch.save(model.regressor.state_dict(), "./model_weights/regressor_weights.pth")



if __name__ == "__main__":
    llm_train()