"""Models over a window of ten transactions, each transaction a set of embedded fields, as in TabFormer's
"raw embedded features" baselines (MLP, LSTM), plus a small transformer encoder trained directly."""
import torch
from torch import nn

FIELDS = ["tok_amount", "tok_hour", "tok_weekday", "tok_chip", "tok_merchant", "tok_city", "tok_state", "tok_zip",
          "tok_mcc", "tok_errors", "tok_card"]


class RowEmbedding(nn.Module):
    """One embedding table per field, concatenated: a transaction becomes one vector."""

    def __init__(self, vocab_sizes: list[int], dim: int = 16):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(v + 1, dim) for v in vocab_sizes)
        self.out_dim = dim * len(vocab_sizes)

    def forward(self, tokens):  # batch x rows x fields
        return torch.cat([t(tokens[..., i]) for i, t in enumerate(self.tables)], dim=-1)


class WindowMLP(nn.Module):
    def __init__(self, vocab_sizes, rows=10, hidden=256):
        super().__init__()
        self.embed = RowEmbedding(vocab_sizes)
        self.net = nn.Sequential(nn.Linear(self.embed.out_dim * rows, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, 1))

    def forward(self, tokens):
        return self.net(self.embed(tokens).flatten(1)).squeeze(-1)


class WindowLSTM(nn.Module):
    def __init__(self, vocab_sizes, hidden=128):
        super().__init__()
        self.embed = RowEmbedding(vocab_sizes)
        self.lstm = nn.LSTM(self.embed.out_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, tokens):
        h, _ = self.lstm(self.embed(tokens))
        return self.head(h[:, -1]).squeeze(-1)


class WindowTransformer(nn.Module):
    """Every transaction attends to the others; a learned [CLS] vector summarizes the window."""

    def __init__(self, vocab_sizes, rows=10, d=128, layers=2, heads=4):
        super().__init__()
        self.embed = RowEmbedding(vocab_sizes)
        self.project = nn.Linear(self.embed.out_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.position = nn.Parameter(torch.zeros(1, rows + 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, tokens):
        x = self.project(self.embed(tokens))
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.position
        return self.head(self.encoder(x)[:, 0]).squeeze(-1)


MODELS = {"mlp": WindowMLP, "lstm": WindowLSTM, "transformer": WindowTransformer}
