"""Small binary correspondence classifier, independent of Global/Local roles."""
from torch import nn


class GRMInlierHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.graph_hidden_dim
        self.classifier = nn.Sequential(nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, tokens):
        return self.classifier(tokens).squeeze(-1)
