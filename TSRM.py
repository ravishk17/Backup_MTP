import torch
import torch.nn as nn
import torch.optim as optim

# 3. Hierarchical Representation
# class TemporalSemanticRelationModule(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(TemporalSemanticRelationModule, self).__init__()
#         self.fc_temporal = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.Tanh(),
#             nn.Linear(output_dim, 1)
#         )
#         self.fc_semantic = nn.Linear(input_dim, output_dim)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, proposals, context):
#         temporal_scores = self.fc_temporal(proposals)
#         semantic_scores = torch.matmul(self.fc_semantic(proposals), self.fc_semantic(context).T)
#         final_scores = self.softmax(temporal_scores * semantic_scores)
#         return final_scores
    
# Temporal-Semantic Relation Module
class TemporalSemanticRelationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalSemanticRelationModule, self).__init__()
        self.temporal_fc = nn.Linear(input_dim, hidden_dim)
        self.semantic_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, proposals):
        temporal_relation = self.temporal_fc(proposals)
        semantic_relation = self.semantic_fc(proposals)
        fused_relation = temporal_relation * semantic_relation
        return self.output_fc(fused_relation)

# class TopicPredictor(nn.Module):
#     def __init__(self, input_dim, topic_dim):
#         super(TopicPredictor, self).__init__()
#         self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2)
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, topic_dim),
#             nn.ReLU(),
#             nn.Linear(topic_dim, topic_dim)
#         )

#     def forward(self, x):
#         x = self.conv(x.permute(0, 2, 1))  # (batch, features, frames)
#         x = torch.max(x, dim=-1)[0]  # Max-pooling
#         return self.fc(x)