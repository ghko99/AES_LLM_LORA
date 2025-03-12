import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class LlamaForSequenceRegression(torch.nn.Module):
    def __init__(self,model_name,output_dim = 11):
        super().__init__()
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 3. 기본 모델 로드 (LoRA를 적용하기 전)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # LLaMA의 attention 모듈 대상
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.regressor = torch.nn.Linear(self.model.config.hidden_size, output_dim)  # 11개 출력
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # 마지막 토큰 사용
        logits = self.regressor(last_hidden)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            
        return (loss, logits) if loss is not None else logits