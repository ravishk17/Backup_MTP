import torch.nn as nn

# 2. Temporal Event Proposal (TEP)
class TemporalEventProposal(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_proposals):
        super(TemporalEventProposal, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_proposals = nn.Linear(hidden_dim, num_proposals)  # For proposal scores
        self.fc_output = nn.Linear(num_proposals, hidden_dim)  # Match output dim with TSRM input
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs, _ = self.gru(x)
        proposal_scores = self.sigmoid(self.fc_proposals(outputs))
        transformed_output = self.fc_output(proposal_scores)  # Transform proposals to required dim
        return transformed_output