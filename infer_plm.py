import torch
from kobert_transformers import get_tokenizer
from safetensors.torch import load_file
from datasets import load_from_disk
from tqdm import tqdm
from models.aes_kobert import KoBERTForSequenceRegression
import numpy as np
from datasets import Dataset
import pandas as pd
from datetime import datetime

# Load the saved model
def load_fine_tuned_kobert(model_path):
    model = KoBERTForSequenceRegression()
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def plm_infer():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./kobert_results/final_model/model.safetensors"
    tokenizer = get_tokenizer()
    model = load_fine_tuned_kobert(model_path=model_path).to(device=device)

    def tokenize_function(examples):
        return tokenizer(
            examples,
            padding="max_length",
            truncation=True,
            max_length=512,
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
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        predicted_scores = outputs['logits'].squeeze().tolist()
        test_dict['output'].append(predicted_scores)
    

    df = pd.DataFrame(test_dict)
    result = Dataset.from_pandas(df)
    
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S_kobert_output")
    
    result.save_to_disk(formatted_datetime)

if __name__ == "__main__":
    plm_infer()