import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from models.aes_llama import LlamaForSequenceRegression

def load_fine_tuned_model(model_name):
    model = LlamaForSequenceRegression(model_name=model_name)
    model.model.load_adapter('./model_weights/lora_weights', adapter_name="default")
    model.regressor.load_state_dict(torch.load('./model_weights/regressor_weights.pth'))
    model.eval()
    return model

def llm_infer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = load_fine_tuned_model(model_name).to(device=device)

    def tokenize_function(examples):
        return tokenizer(
            examples,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
    dataset = load_from_disk("aes_dataset")  # 사용자 데이터셋으로 교체 필요
    test_dataset = dataset['test']
    tokenized_dataset = [tokenize_function(data) for data in tqdm(test_dataset['text'])]
    tokenized_dataset = [{key: value.to(device) for key, value in data.items()} for data in tqdm(tokenized_dataset)]

    test_dict = test_dataset.to_dict()
    test_dict['output'] = []

    for inputs in tqdm(tokenized_dataset):
        with torch.no_grad():
            outputs = model(**inputs)
        test_dict['output'].append(outputs[0].tolist())
    

    df = pd.DataFrame(test_dict)
    result = Dataset.from_pandas(df)
    
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S_llama_output")
    
    result.save_to_disk(formatted_datetime)

if __name__ == "__main__":
    llm_infer()