import torch.nn as nn
# Topic Predictor Module
class TopicPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_topics):
        super(TopicPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_topics)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        topic_scores = self.fc2(x)
        return topic_scores