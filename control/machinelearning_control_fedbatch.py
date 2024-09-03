import sys
sys.path.append("../")

from typing import Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy

from src.utils import feeding_strategy, get_volume

T_START = 0
T_END = 10.5
NUM_EPOCHS = 30000
LEARNING_RATE = 1e-4
NUM_POINTS = 10000
NUM_COLLOCATION = 100000
PATIENCE = 1000
THRESHOLD = 1e-3
EARLY_STOPPING_EPOCH = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def numpy_to_tensor(array):
    return (
        torch.tensor(array, requires_grad=True, dtype=torch.float32)
        .to(DEVICE)
        .reshape(-1, 1)
    )

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )

class PINN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

def loss_fn(
    net: nn.Module, 
    t_start: Union[np.float32, torch.Tensor],
    t_end: Union[np.float32, torch.Tensor],
    feeds: pd.DataFrame,
    Sin: float,
    V0: float,
    S0: float,
    mu_max: float,
    K_s: float,
    Y_xs: float,
) -> torch.Tensor:

    t_col = numpy_to_tensor(np.arange(t_start, t_end, 1)).to(DEVICE)
    X_col = numpy_to_tensor(np.random.uniform(2.5, 5, NUM_COLLOCATION)).to(DEVICE)
    S_col = numpy_to_tensor([S0 for _ in range(len(t_col))]).to(DEVICE)
    F_col = numpy_to_tensor(np.random.uniform(15, 35, NUM_COLLOCATION)).to(DEVICE)
    V_col = (
        torch.tensor(get_volume(feeds=feeds, V0=V0, t=t_col.cpu().detach().numpy().reshape(-1,)), requires_grad=True)
        .view(-1, 1)
        .to(DEVICE)
    )

    u_col = torch.cat((t_col, X_col, S_col, F_col), 1).to(DEVICE)

    preds = net.forward(u_col)

    X_pred = preds[:, 0].view(-1, 1)
    S_pred = preds[:, 1].view(-1, 1)

    dXdt_pred = grad(X_pred, t_col)[0]
    dSdt_pred = grad(S_pred, t_col)[0]

    mu = mu_max * S_pred / (K_s + S_pred)

    error_dXdt = dXdt_pred - mu * X_pred + X_pred * F_col / V_col
    error_dSdt = dSdt_pred + mu * X_pred / Y_xs - F_col / V_col * (Sin - S_pred)

    error_ode = torch.mean(error_dXdt**2 + error_dSdt**2)
    return error_ode
