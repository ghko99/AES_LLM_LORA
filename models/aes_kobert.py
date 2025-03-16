import torch.nn as nn
from kobert_transformers import get_kobert_model

class KoBERTForSequenceRegression(nn.Module):
    def __init__(self, output_dim=11):
        super(KoBERTForSequenceRegression, self).__init__()
        self.bert = get_kobert_model()
        self.classifier = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 모델의 최종 출력
        logits = self.classifier(pooled_output)  # 예측값

        loss = None
        if labels is not None:
            criterion = nn.MSELoss()
            loss = criterion(logits, labels)  # MSE 손실 계산

        if loss is not None:
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}  # loss 없이 logits만 반환
