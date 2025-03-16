from datasets import load_from_disk
from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
from datetime import datetime

def load_output_data(path):
    dataset = load_from_disk(path)
    return dataset

def compute_metrics(y_sent_pred, y_test):
    accuracy_scores = np.mean(y_sent_pred == y_test, axis=0)
    kappa_scores = [cohen_kappa_score(y_sent_pred[:,i], y_test[:,i], weights='quadratic') for i in range(11)]

    return accuracy_scores, np.array(kappa_scores)

def save_performance_results(path):

    dataset = load_output_data(path)
    y_sent_pred = np.array(dataset['label'])*3
    y_real = np.array(dataset['output'])*3

    pred = np.rint(y_sent_pred).astype(int)
    real = np.rint(y_real).astype(int)

    accuracy, qwk = compute_metrics(y_sent_pred=pred,y_test=real)

    rubric = ['문법 정확도', '단어 선택의 적절성', '문장 표현의 적절성',
              '문단 내 구조의 적절성', '문단 간 구조의 적절성', '구조의 일관성',
              '분량의 적절성', '주제 명료성', '창의성', '프롬프트 독해력',
              '설명의 구체성', '평균']
    
    accuracy_avg = np.mean(accuracy)
    qwk_avg = np.mean(qwk)

    accuracy = accuracy.tolist()
    qwk = qwk.tolist()

    accuracy.append(accuracy_avg)
    qwk.append(qwk_avg)

    df = pd.DataFrame({"rubric":rubric, "accuracy":accuracy, "QWK": qwk})
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    df.to_csv('./performance_logs/{}.csv'.format(formatted_datetime),index=False)

if __name__ == "__main__":
    output_path = "2025_03_16_06_19_03_output"
    save_performance_results(output_path)