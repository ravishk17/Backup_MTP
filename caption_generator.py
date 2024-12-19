import torch
import torch.nn as nn
# 4. Caption Generator
class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CaptionGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, captions, features):
        embedded = self.embedding(captions)
        lstm_input = torch.cat((features.unsqueeze(1).repeat(1, embedded.size(1), 1), embedded), dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        return self.fc(lstm_out)

