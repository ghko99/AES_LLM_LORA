import torch.nn as nn

class GRUScoreModule(nn.Module):
    def __init__(self,output_dim,hidden_dim, dropout=0.5):
        super(GRUScoreModule, self).__init__()
        self.gru = nn.GRU(768,hidden_dim, num_layers=2,dropout=dropout, batch_first=True, bidirectional=True)        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x, _ = self.gru(x)
        
        x = x[:, -1, :]  # Use the output of the last time step
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x