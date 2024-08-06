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
        self.input = nn.Linear(input_dim, 64)
        self.hidden = nn.Linear(64, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 64) 
        self.output = nn.Linear(64, output_dim)

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
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


def loss_ode(
    net: torch.nn.Module,
    feeds: pd.DataFrame,
    t_start: Union[np.float32, torch.Tensor],
    t_end: Union[np.float32, torch.Tensor],
) -> torch.Tensor:
    
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
    dSdt_pred = torch.autograd.grad(
        S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True
    )[0]
    dVdt_pred = torch.autograd.grad(
        V_pred, t, grad_outputs=torch.ones_like(V_pred), create_graph=True
    )[0]

    mu = net.mu_max * S_pred / (net.K_s + S_pred)

    error_dXdt = nn.MSELoss()(
        dXdt_pred, mu * X_pred - X_pred * F / V_pred
    )
    error_dSdt = nn.MSELoss()(
        dSdt_pred, - mu * X_pred / net.Y_xs + F / V_pred * (Sin - S_pred)
    )
    error_dVdt = nn.MSELoss()(
        dVdt_pred, torch.ones_like(dVdt_pred) * F
    )

    error_ode = error_dXdt + error_dSdt + error_dVdt
    return error_ode

def train(
    net: nn.Module,
    t_train: torch.Tensor,
    u_train: torch.Tensor,
    df: pd.DataFrame,
    feeds: pd.DataFrame,
    num_epochs: int = 1000,
    verbose: bool = True,
) -> nn.Module:
    
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u_pred = net.forward(t_train)
        
        # Data loss
        X_data_loss = nn.MSELoss()(u_pred[:, 0], u_train[:, 0])
        S_data_loss = nn.MSELoss()(u_pred[:, 1], u_train[:, 1])
        V_data_loss = nn.MSELoss()(u_pred[:, 2], u_train[:, 2])
        loss_data = X_data_loss + S_data_loss + V_data_loss
        loss_data = loss_data * 0.5

        # Initial condition loss
        X_IC_loss = nn.MSELoss()(u_pred[0, 0], u_train[0, 0])
        S_IC_loss = nn.MSELoss()(u_pred[0, 1], u_train[0, 1])
        V_IC_loss = nn.MSELoss()(u_pred[0, 2], u_train[0, 2])
        loss_ic = X_IC_loss + S_IC_loss + V_IC_loss
        
        # ODE loss
        loss_pde = loss_ode(net, feeds, df["RTime"].min(), df["RTime"].max()) 

        total_loss = loss_data + loss_pde + loss_ic
        total_loss.backward()
        optimizer.step()

        if verbose and epoch % 250 == 0:
            tqdm.write(f"Epoch {epoch} || Total Loss: {total_loss.item():.4f}, Loss Data: {loss_data.item():.4f}, Loss ODE: {loss_pde.item():.4f}, Loss IC: {loss_ic.item():.4f}")
            tqdm.write(
            f"mu_max: {net.mu_max.item():.4f}, Ks: {net.K_s.item():.4f}, Yxs: {net.Y_xs.item():.4f}"
            )

        # if epoch == 0:
        #     # Checking that the initialization of the ANN results in positive values for the state variables
        #     if (u_pred < 0).any():
        #         raise ValueError("u_pred has negative values")
        
        # Early stopping if total_loss <= 0.001 for 100 consecutive epochs
        if total_loss <= 0.005 and loss_data <= 0.005 and loss_pde <= 0.005 and epoch >= 5000:
            print(f"Early stopping at epoch {epoch}")
            print(f'mu_max: {net.mu_max.item():.4f}, Ks: {net.K_s.item():.4f}, Yxs: {net.Y_xs.item():.4f}')
            break
        


    return net
