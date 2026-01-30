import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F

class RankNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_blocks)
        ])

        self.output = nn.Linear(hidden_dim, 1)

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.activation(x)

        for block in self.blocks:
            x = x + block(x)   # 🔥 residual connection

        return self.output(x).squeeze(-1)


def _sanitize_X(X):
    if isinstance(X, pd.DataFrame):
        X = X.values

    if isinstance(X, np.ndarray):
        X = X.astype(np.float32)

    if not torch.is_tensor(X):
        X = torch.from_numpy(X)

    return X



def _sanitize_y(y):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = pd.to_numeric(y.squeeze(), errors="coerce")
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    if isinstance(y, np.ndarray):
        y = y.astype(np.float32)

    if not torch.is_tensor(y):
        y = torch.from_numpy(y)

    return y


def train_rank_model(model, X, y, epochs=20, lr=1e-3):
    """
    X: pandas DataFrame / numpy array / torch tensor
    y: pandas Series / numpy array / torch tensor
    """

    X = _sanitize_X(X)
    y = _sanitize_y(y)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model(X)
        loss = ranknet_loss(pred, y)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    return model


def ranknet_loss(pred, y, num_pairs=1024):
    """
    Stable stochastic RankNet loss
    - O(N) memory
    - Strong gradients
    - Works for time-series
    """

    device = pred.device
    n = pred.shape[0]

    if n < 2:
        return torch.tensor(0.0, device=device)

    # 🔹 Sample random pairs
    idx1 = torch.randint(0, n, (num_pairs,), device=device)
    idx2 = torch.randint(0, n, (num_pairs,), device=device)

    pred_diff = pred[idx1] - pred[idx2]
    true_diff = y[idx1] - y[idx2]

    # Ignore equal targets
    mask = true_diff != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    pred_diff = pred_diff[mask]
    true_diff = true_diff[mask]

    # RankNet logistic loss
    loss = F.softplus(-torch.sign(true_diff) * pred_diff).mean()

    return loss