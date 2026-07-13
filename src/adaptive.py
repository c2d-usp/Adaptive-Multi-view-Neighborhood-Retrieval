import numpy as np
import torch
import torch.nn as nn


class ViewWeighter(nn.Module):
    def __init__(self, n_features: int, n_views: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_views),
        )
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def soft_knn_loss(
    alpha: torch.Tensor,
    D_views_bat: list,
    y_train_t: torch.Tensor,
    y_batch_t: torch.Tensor,
    k: int,
    task: str,
    temperature: float,
    lambda_ent: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    D_stack = torch.stack(D_views_bat, dim=0)
    D_final = torch.einsum("bv,vbn->bn", alpha, D_stack)

    _, nn_idx = torch.topk(-D_final, k, dim=1)
    D_topk = D_final.gather(1, nn_idx)
    S = torch.exp(-D_topk / (temperature + eps))
    P = S / (S.sum(dim=1, keepdim=True) + eps)

    nn_labels = y_train_t[nn_idx]

    if task == "regression":
        y_pred = (P * nn_labels).sum(dim=1)
        loss_task = torch.mean((y_pred - y_batch_t) ** 2)
    else:
        n_classes = int(y_train_t.max().item()) + 1
        y_soft = torch.zeros(alpha.shape[0], n_classes, device=alpha.device, dtype=alpha.dtype)
        for c in range(n_classes):
            y_soft[:, c] = (P * (nn_labels == c).float()).sum(dim=1)
        y_soft = y_soft.clamp(min=eps)
        loss_task = torch.nn.functional.nll_loss(torch.log(y_soft), y_batch_t.long())

    H = -(alpha * torch.log(alpha + eps)).sum(dim=1).mean()
    loss_total = loss_task - lambda_ent * H
    return loss_total


def train_hypernetwork(
    weighter: ViewWeighter,
    X_val: np.ndarray,
    D_views_val: list,
    y_train: np.ndarray,
    y_val: np.ndarray,
    k: int,
    task: str,
    temperature: float,
    lambda_ent: float,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    eps: float = 1e-12,
) -> list:
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    D_views_val_t = [torch.tensor(D, dtype=torch.float32, device=device) for D in D_views_val]

    weighter.to(device)
    optimizer = torch.optim.Adam(weighter.parameters(), lr=lr)
    n_val = X_val_t.shape[0]
    loss_curve = []
    rng = np.random.default_rng(0)

    for epoch in range(n_epochs):
        weighter.train()
        idx = rng.permutation(n_val)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_val, batch_size):
            batch_idx = idx[start : start + batch_size]
            bidx_t = torch.tensor(batch_idx, device=device)

            X_batch = X_val_t[bidx_t]
            y_batch = y_val_t[bidx_t]
            D_batch = [D[bidx_t] for D in D_views_val_t]

            alpha = weighter(X_batch)
            loss = soft_knn_loss(
                alpha,
                D_batch,
                y_train_t,
                y_batch,
                k,
                task,
                temperature,
                lambda_ent,
                eps,
            )

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weighter.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        loss_curve.append(epoch_loss / max(n_batches, 1))

    return loss_curve
