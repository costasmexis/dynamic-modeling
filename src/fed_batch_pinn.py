from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from .utils import feeding_strategy

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
        self.input = nn.Linear(input_dim, 16)
        self.hidden = nn.Linear(16, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.hidden3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, output_dim)

        self.mu_max = nn.Parameter(torch.tensor([0.5]))
        self.K_s = nn.Parameter(torch.tensor([0.5]))
        self.Y_xs = nn.Parameter(torch.tensor([0.5]))

        self.t_start = t_start
        self.t_end = t_end
        if isinstance(self.t_start, torch.Tensor):
            self.t_start = self.t_start.item()
        if isinstance(self.t_end, torch.Tensor):
            self.t_end = self.t_end.item()

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = self.output(x)
        return x


def loss_ode(
    net: torch.nn.Module,
    feeds: pd.DataFrame,
    t_start: Union[np.float32, torch.Tensor],
    t_end: Union[np.float32, torch.Tensor],
):
    if isinstance(t_start, torch.Tensor):
        t_start = t_start.item()
    if isinstance(t_end, torch.Tensor):
        t_end = t_end.item()

    t = torch.linspace(t_start, t_end, steps=100).view(-1, 1).requires_grad_(True)

    Sin = 1.43 * 200

    F = torch.tensor([feeding_strategy(feeds, t) for t in t], dtype=torch.float32).view(
        -1, 1
    )

    u_pred = net.forward(t)
    X_pred = u_pred[:, 0].view(-1, 1)
    S_pred = u_pred[:, 1].view(-1, 1)
    V_pred = u_pred[:, 2].view(-1, 1)

    dXdt_pred = torch.autograd.grad(
        X_pred, t, grad_outputs=torch.ones_like(X_pred), create_graph=True
    )[0]
    # dSdt_pred = torch.autograd.grad(
    #     S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True
    # )[0]
    # dVdt_pred = torch.autograd.grad(
    #     V_pred, t, grad_outputs=torch.ones_like(V_pred), create_graph=True
    # )[0]

    mu = net.mu_max * S_pred / (net.K_s + S_pred)

    error_dXdt = nn.MSELoss()(dXdt_pred, mu * X_pred + X_pred * F / V_pred)
    # error_dSdt = nn.MSELoss()(
    #     dSdt_pred, -mu * X_pred / net.Y_xs + F / V_pred * (Sin - S_pred)
    # )
    # error_dVdt = nn.MSELoss()(dVdt_pred, F)

    error_ode = error_dXdt 
    return error_ode


def train(
    net: nn.Module,
    t_train: torch.Tensor,
    u_train: torch.Tensor,
    df: pd.DataFrame,
    feeds: pd.DataFrame,
    num_epochs: int = 1000,
    verbose: bool = True,
):
    TOTAL_LOSS = []
    LOSS_DATA = []
    LOSS_ODE = []
    optimizer = torch.optim.RMSprop(net.parameters(), lr=5e-4)

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        u_pred = net.forward(t_train)
        loss_data = nn.MSELoss()(u_pred, u_train)
        loss_ic = nn.MSELoss()(u_pred[0], u_train[0])
        loss_pde = loss_ode(net, feeds, df["RTime"].min(), df["RTime"].max())

        total_loss = loss_data + loss_pde + loss_ic
        total_loss.backward()
        optimizer.step()

        if verbose:
            if epoch % 100 == 0:
                print(f"Epoch {epoch} || Total Loss: {total_loss.item():.6f}")
                print(f"mu_max: {net.mu_max.item():.4f}, Ks: {net.K_s.item():.4f}, Yxs: {net.Y_xs.item():.4f}")

        TOTAL_LOSS.append(total_loss.item())
        LOSS_DATA.append(loss_data.item())
        # LOSS_IC.append(loss_ic.item())
        LOSS_ODE.append(loss_pde.item())

    return net, TOTAL_LOSS, LOSS_DATA, LOSS_ODE
