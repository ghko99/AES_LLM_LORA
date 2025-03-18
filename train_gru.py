import dask.dataframe as dd
import os
from tqdm import tqdm
from datasets import load_from_disk
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.aes_gru import GRUScoreModule
import time

def pad_sequences_np(sequences, maxlen, dtype='float32', padding='pre'):
    padded_sequences = np.zeros((len(sequences), maxlen, sequences[0].shape[1]), dtype=dtype)
    for i, seq in enumerate(sequences):
        if padding == 'pre':
            padded_sequences[i, -len(seq):] = seq
        elif padding == 'post':
            padded_sequences[i, :len(seq)] = seq
    return padded_sequences

class EssayDataset(Dataset):
    def __init__(self, embedded_essays, labels, maxlen = 128):
        self.embedded_essays = embedded_essays
        self.embedded_essays = torch.tensor(pad_sequences_np(self.embedded_essays, maxlen=maxlen, padding='pre', dtype='float32'), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embedded_essays[idx], self.labels[idx]

def splitted_essays(essays):
    splitted = []
    for essay in tqdm(essays):
        essay_sentences = essay.split('\n')
        essay_sentences = [sent.strip() for sent in essay_sentences if sent.strip() != '']
        splitted.append(essay_sentences)
    return splitted

def get_embedded_essay(essays, model_name, data_type):
    embedded_essay_raw = dd.read_csv(os.path.join('./emb/{}_emb_{}.csv'.format(model_name, data_type)), encoding='cp949').compute()
    print(embedded_essay_raw.shape)
    embedded_essay = []
    tmp_ix = 0
    for ix, essay_raw in enumerate(essays):
        tmp_len = len(essay_raw)
        essay = embedded_essay_raw[tmp_ix:tmp_ix + tmp_len]
        embedded_essay.append(essay)
        tmp_ix += tmp_len
    return embedded_essay


def get_train_valid_dataset(batch_size=512, maxlen=128):
    
    dataset = load_from_disk('./aes_dataset')
    train_essays, valid_essays = splitted_essays(dataset['train']['text']), splitted_essays(dataset['valid']['text'])
    train_labels, valid_labels = dataset['train']['label'], dataset['valid']['label']

    train_embedded_essay = get_embedded_essay(train_essays, "kobert", "train")
    valid_embedded_essay = get_embedded_essay(valid_essays, "kobert", "valid")

    train_dataset = EssayDataset(train_embedded_essay, train_labels, maxlen=maxlen)
    valid_dataset = EssayDataset(valid_embedded_essay, valid_labels, maxlen=maxlen)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, valid_dataloader

def gru_train(n_outputs=11, dropout=0.5, learning_rate=0.001, n_epochs=50, patience=5):
    train_loader, valid_loader = get_train_valid_dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUScoreModule(output_dim=n_outputs,hidden_dim=128, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    prev_time = time.time()
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for inputs,labels in train_loader:
            inputs ,labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)        
            loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_outputs = []
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device),labels.to(device)
                outputs = model(inputs)            
                loss = criterion(outputs, labels)
                all_outputs.extend(outputs.cpu().numpy())
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(valid_loader)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time Elapsed: {time.time() - prev_time:.4f}')
        prev_time = time.time()
        
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), './gru_model/best_model.pth')
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping")
                break

if __name__ == "__main__":
    gru_train()