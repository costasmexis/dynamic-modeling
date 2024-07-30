from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numpy_to_tensor(array):
    return (
        torch.tensor(array, requires_grad=True, dtype=torch.float32)
        .to(DEVICE)
        .reshape(-1, 1)
    )


class PINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        t_start: Union[np.float32, torch.Tensor],
        t_end: Union[np.float32, torch.Tensor],
    ):
        super().__init__()
        self.input = nn.Linear(input_dim, 128)
        self.hidden = nn.Linear(128, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.hidden3 = nn.Linear(512, 128)
        self.output = nn.Linear(128, output_dim)

        self.c1 = nn.Parameter(torch.tensor([0.1]))
        self.c2 = nn.Parameter(torch.tensor([0.1]))

        self.mu_max = torch.tensor([0.870], dtype=torch.float32)
        self.K_s = torch.tensor([0.214], dtype=torch.float32)
        self.Y_xs = torch.tensor([0.496], dtype=torch.float32)

        self.t_start = t_start
        self.t_end = t_end
        if isinstance(self.t_start, torch.Tensor):
            self.t_start = self.t_start.item()
        if isinstance(self.t_end, torch.Tensor):
            self.t_end = self.t_end.item()

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


def loss_ode(
    net: torch.nn.Module,
    F,
    t_start: Union[np.float32, torch.Tensor],
    t_end: Union[np.float32, torch.Tensor],
) -> torch.Tensor:
    if isinstance(t_start, torch.Tensor):
        t_start = t_start.item()
    if isinstance(t_end, torch.Tensor):
        t_end = t_end.item()

    t = torch.linspace(t_start, t_end, steps=50).view(-1, 1).requires_grad_(True)

    Sin = 1.43 * 200

    F = torch.tensor(
        [F for _ in range(t.shape[0])], requires_grad=True, dtype=torch.float32
    ).view(-1, 1)

    u_pred = net.forward(t)
    X_pred = u_pred[:, 0].view(-1, 1)
    S_pred = u_pred[:, 1].view(-1, 1)
    P_pred = u_pred[:, 2].view(-1, 1)
    V_pred = u_pred[:, 3].view(-1, 1)

    dXdt_pred = torch.autograd.grad(
        X_pred, t, grad_outputs=torch.ones_like(X_pred), create_graph=True
    )[0]
    dSdt_pred = torch.autograd.grad(
        S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True
    )[0]
    dPdt_pred = torch.autograd.grad(
        P_pred, t, grad_outputs=torch.ones_like(P_pred), create_graph=True
    )[0]

    mu = net.mu_max * S_pred / (net.K_s + S_pred)

    alpha = net.c1 * (1 - torch.exp(-net.c2 * t))

    error_dXdt = nn.MSELoss()(dXdt_pred, mu * X_pred + X_pred * F / V_pred)
    error_dSdt = nn.MSELoss()(
        dSdt_pred, - mu * X_pred / net.Y_xs + F / V_pred * (Sin - S_pred)
    )
    error_dPdt = nn.MSELoss()(dPdt_pred, alpha * mu * X_pred - F * P_pred / V_pred)
    error_dVdt = nn.MSELoss()(V_pred, F)

    error_ode = error_dXdt + error_dSdt + error_dPdt + error_dVdt
    return error_ode


def train(
    net: nn.Module,
    t_train: torch.Tensor,
    u_train: torch.Tensor,
    df: pd.DataFrame,
    F,
    num_epochs: int = 1000,
    verbose: bool = True,
) -> nn.Module:
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        u_pred = net.forward(t_train)

        # Data loss
        X_data_loss = nn.MSELoss()(u_pred[:, 0], u_train[:, 0])
        S_data_loss = nn.MSELoss()(u_pred[:, 1], u_train[:, 1])
        P_data_loss = nn.MSELoss()(u_pred[:, 2], u_train[:, 2])
        V_data_loss = nn.MSELoss()(u_pred[:, 3], u_train[:, 3])
        loss_data = X_data_loss + S_data_loss + P_data_loss + V_data_loss

        # Initial condition loss
        X_IC_loss = nn.MSELoss()(u_pred[0, 0], u_train[0, 0])
        S_IC_loss = nn.MSELoss()(u_pred[0, 1], u_train[0, 1])
        P_IC_loss = nn.MSELoss()(u_pred[0, 2], u_train[0, 2])
        V_IC_loss = nn.MSELoss()(u_pred[0, 3], u_train[0, 3])
        loss_ic = X_IC_loss + S_IC_loss + P_IC_loss + V_IC_loss

        # ODE loss
        loss_pde = loss_ode(net, F, df["RTime"].min(), df["RTime"].max())

        # Total loss
        total_loss = loss_data + loss_pde + loss_ic

        if verbose and epoch % 1000 == 0:
            print(
                f"Epoch {epoch} || Total Loss: {total_loss.item():.6f} | Data Loss: {loss_data.item():.6f} | IC Loss: {loss_ic.item():.6f} | ODE Loss: {loss_pde.item():.6f}"
            )

        total_loss.backward()
        optimizer.step()

    return net