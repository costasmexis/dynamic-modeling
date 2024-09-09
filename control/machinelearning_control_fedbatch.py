import sys

sys.path.append("../")

from typing import Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy

from system_ode_fedbatch import get_volume

NUM_EPOCHS = 30000
LEARNING_RATE = 1e-3
NUM_POINTS = 500
NUM_COLLOCATION = 500
PATIENCE = 100
THRESHOLD = 1e-3
EARLY_STOPPING_EPOCH = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_dataset(full_df: pd.DataFrame) -> Union[torch.Tensor, torch.Tensor]:
    df = pd.DataFrame(columns=["t", "Biomass", "Glucose"])
    df["Biomass"] = np.random.uniform(2.5, 5, NUM_POINTS)
    df["Glucose"] = full_df["Glucose"].iloc[0]
    df["t"] = 0.0
    df["F"] = np.random.uniform(0.015, 0.065, NUM_POINTS) 

    print(f"Dataset shape: {df.shape}")

    t_train = numpy_to_tensor(df["t"].values)
    X_train = numpy_to_tensor(df["Biomass"].values)
    S_train = numpy_to_tensor(df["Glucose"].values)
    F_train = numpy_to_tensor(df["F"].values)

    in_train = torch.cat([t_train, X_train, S_train, F_train], dim=1)
    out_train = torch.cat([X_train, S_train], dim=1)
    return in_train, out_train


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
        self.input = nn.Linear(input_dim, 64)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output(x)
        return x


def loss_fn(
    net: nn.Module,
    t_start: Union[np.float32, torch.Tensor],
    t_end: Union[np.float32, torch.Tensor],
    Sin: float,
    S0: float,
    mu_max: float,
    K_s: float,
    Y_xs: float,
) -> torch.Tensor:
    
    t_col = numpy_to_tensor(np.linspace(t_start, t_end, NUM_COLLOCATION)).to(DEVICE)
    X_col = numpy_to_tensor(np.random.uniform(3, 4, NUM_COLLOCATION)).to(DEVICE)
    S_col = numpy_to_tensor([S0 for _ in range(len(t_col))]).to(DEVICE)
    F_col = numpy_to_tensor(np.random.uniform(0.015, 0.065, NUM_COLLOCATION)).to(DEVICE)
    F_col = numpy_to_tensor([0.01 for _ in range(len(t_col))]).to(DEVICE)
    V_col = numpy_to_tensor([get_volume(t) for t in t_col]).to(DEVICE)

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


def main(
    in_train: torch.Tensor,
    out_train: torch.Tensor,
    t_start: Union[np.float32, torch.Tensor],
    t_end: Union[np.float32, torch.Tensor],
    S_in: float,
    S0: float,
    mu_max: float,
    Ks: float,
    Yxs: float,
    verbose: int = 100
):
    
    net = PINN(4, 2).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Loss weights
    w_data, w_ode, w_ic = 1, 1, 1

    # Initialize early stopping variables
    best_loss = float("inf")
    best_model_weights = None
    patience = PATIENCE
    threshold = THRESHOLD

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        preds = net.forward(in_train)

        loss_data = nn.MSELoss()(preds, out_train)

        loss_ode = loss_fn(net, t_start, t_end, S_in, S0, mu_max, Ks, Yxs)

        loss = w_data * loss_data + w_ode * loss_ode
        loss.backward()
        optimizer.step()

        if epoch % verbose == 0:
            print(
                f"Epoch {epoch}, Loss_data: {loss_data.item():.4f}, Loss_ode: {loss_ode.item():.4f}"
            )

        if epoch >= EARLY_STOPPING_EPOCH:
            if loss < best_loss - threshold:
                best_loss = loss
                best_model_weights = copy.deepcopy(net.state_dict())
                patience = 1000
            else:
                patience -= 1
                if patience == 0:
                    print(f"Early stopping at epoch {epoch}")
                    net.load_state_dict(best_model_weights)
                    break

    return net
