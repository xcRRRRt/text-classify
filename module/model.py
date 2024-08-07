import torch
import torch.nn as nn
import torch.nn.functional as f


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.score = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mask = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        scores = self.mask(self.score(x))
        weights = f.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_dim, num_class):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_size - 1)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=0.1)
        self.attention = SelfAttention(hidden_dim * 2)
        # self.attention = nn.MultiheadAttention(hidden_dim * 2, 2, dropout=0.1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.embedding_dim).to(x.device)
        embedded = self.embedding(x)
        gru_output, hidden = self.gru(embedded, h_0)
        output = self.attention(gru_output)
        output = self.fc(output)
        return output
