import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from train_gru import splitted_essays, get_embedded_essay, EssayDataset
from models.aes_gru import GRUScoreModule
import pandas as pd
from datetime import datetime

def infer_gru(
    model_path='./gru_model/best_model.pth',
    data_path='./aes_dataset',
    model_name='kobert',
    batch_size=512,
    maxlen=128,
    n_outputs=11,
    dropout=0.5,
    hidden_dim=128
):
    """
    1) dataset['test']의 에세이를 불러와서 문장 단위로 쪼개기
    2) 미리 생성해 둔 임베딩 csv를 불러와 test용 임베딩 벡터를 획득
    3) DataLoader를 통해 모델에 입력 -> 추론 결과 획득
    4) 추론 결과(predictions)를 리스트로 반환 ( shape: [num_samples, n_outputs] )
    """

    # 1) 테스트 데이터 로드 및 문장 분리
    dataset = load_from_disk(data_path)
    test_essays = splitted_essays(dataset['test']['text'])   # test 데이터에 대한 문장 분리
    test_labels = dataset['test']['label']                   # (옵션) 실제 라벨, 필요 시 사용

    # 2) Kobert 임베딩 벡터 불러오기
    test_embedded_essay = get_embedded_essay(test_essays, model_name, "test")

    # 3) Dataset / Dataloader 생성
    test_dataset = EssayDataset(test_embedded_essay, test_labels, maxlen=maxlen)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4) 모델 로드 및 추론
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GRUScoreModule(output_dim=n_outputs, hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_dict = dataset['test'].to_dict()
    test_dict['output'] = []
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # outputs.shape: (batch_size, n_outputs)
            test_dict['output'].extend(outputs.cpu().numpy())

    df = pd.DataFrame(test_dict)
    result = Dataset.from_pandas(df)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S_gru_output")
    result.save_to_disk('./outputs/{}'.format(formatted_datetime))

if __name__ == "__main__":
    infer_gru()