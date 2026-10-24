"""Bi-LSTM baseline, a PyTorch re-implementation of an earlier Keras LSP recognizer::

    BiLSTM(64, seq) -> Dropout -> BiLSTM(32) -> Dropout -> Dense(64, ReLU) -> Dropout -> softmax

It receives the same ``(B, T, V, F)`` tensor as the graph models and flattens it to
``(B, T, V*F)``, so the comparison differs only in the architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden1: int = 64,
        hidden2: int = 32,
        dense: int = 64,
        dropout: float = 0.4,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(in_features, hidden1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(2 * hidden1, hidden2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * hidden2, dense)
        self.relu = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(head_dropout)
        self.fc2 = nn.Linear(dense, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B, T, -1)  # (B, T, V*F)
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])  # last time step
        x = self.relu(self.fc1(x))
        x = self.drop3(x)
        return self.fc2(x)
