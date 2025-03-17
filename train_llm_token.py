from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback)
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from train_llm import SaveBestPeftModelCallback
import wandb
from datetime import datetime

def load_llama_for_generation_model():
    model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def load_dataset_for_generation():
    dataset = load_from_disk("aes_dataset_with_label")
    return dataset 


def DataCollator(examples):
    input_ids = torch.LongTensor([example['input_ids'] for example in examples])
    attention_mask = torch.LongTensor([example['attention_mask'] for example in examples])
    labels = torch.LongTensor([example['labels'] for example in examples])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def llm_token_train():
    wb_token = "8b738bb3f5650780015aa6c3d98a2c811b470916"
    
    wandb.login(key=wb_token)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(project="automated_essay_scoring", name = formatted_datetime, reinit=True)

    model, tokenizer = load_llama_for_generation_model()
    
    dataset = load_dataset_for_generation()
    
    def preprocessing_data(examples):
        input_ids = []
        attention_masks = []
        labels = []

        for instruction,response in zip(examples['instruction'],examples['output']):
            message = [
            {'role': 'user', 'content': [
                {'type':'text', 'text': instruction}
            ]},
            ]
            inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True)
            label = tokenizer(response+'<|eot_id|>',add_special_tokens=False)['input_ids']
            input_id = inputs+label+([tokenizer.pad_token_id]*4096)
            input_id = input_id[:4096]
            label_id = [-100]*len(inputs) + label + ([tokenizer.pad_token_id]*4096)
            label_id = label_id[:4096]
            attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in input_id]

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(label_id)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }
    
    train_dataset = dataset['train']
    valid_dataset = dataset['valid']

    train_dataset = train_dataset.map(
        preprocessing_data,
        num_proc=2,
        batched=True,
        remove_columns=['instruction','output','scores']
    )

    valid_dataset = valid_dataset.map(
        preprocessing_data,
        num_proc=2,
        batched=True,
        remove_columns=['instruction','output','scores']
    )
    print(train_dataset)
    print(train_dataset[0]["input_ids"][:10])
    tokenizer.decode(train_dataset[0]["input_ids"][:100])


    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        learning_rate=1e-5,
        fp16=True,
        remove_unused_columns=False,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        evaluation_strategy="epoch",  # 🔹 Evaluation 수행
        save_strategy="epoch",
        logging_steps=1,
        save_safetensors=False,# 
        load_best_model_at_end=True,  # 🔹 Best Model 로드
        save_total_limit=1,  # 🔹 가장 좋은 모델 하나만 유지
        label_names=['labels'],
        torch_compile=True,
    )
    
    target_modules = []
    for n,p in model.named_parameters():
        if 'language_model' in n and 'cross_attn' not in n and 'embed_tokens' not in n and 'lm_head' not in n and 'norm' not in n:
            target_modules.append(n.replace('.weight',''))
        # else:
        #     p.require_grad=False

    # Lora Tuning
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,  #<-- 요거 줄이면 GPU메모리 절약됩니다.
        lora_alpha=8, #<-- 요거 줄이면 GPU메모리 절약됩니다.
        target_modules=target_modules
    )

    lora_model = get_peft_model(model,peft_config)

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=valid_dataset["valid"],
        data_collator=DataCollator,
        callbacks=[SaveBestPeftModelCallback(model), EarlyStoppingCallback(early_stopping_patience=5, 
                                                                           early_stopping_threshold=0.0)]
    )
    trainer.train()
    lora_model.save_pretrained("./model_for_gen_weights/lora_weights")

if __name__ == "__main__":
    llm_token_train()
    