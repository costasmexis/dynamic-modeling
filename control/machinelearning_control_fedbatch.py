import sys

sys.path.append("../")

from typing import Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy

NUM_EPOCHS = 50000
LEARNING_RATE = 1e-4
NUM_COLLOCATION = 10000
PATIENCE = 100
THRESHOLD = 1e-3
EARLY_STOPPING_EPOCH = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_dataset(
    data: pd.DataFrame, num_points: int, F_min: float = 0.010, F_max: float = 0.070, default: bool = False) -> Union[torch.Tensor, torch.Tensor]:
    """Generate dataset of random multiple initial conditions and control actions"""
    df = pd.DataFrame(columns=["t", "Biomass", "Glucose"])
    
    if default:    
        df["Biomass"] = np.random.uniform(
            data["Biomass"].min(), data["Biomass"].max(), num_points
        )
        df["Glucose"] = np.random.uniform(
            data["Glucose"].min(), data["Glucose"].max(), num_points
        )
        df["V"] = np.random.uniform(
            data["V"].min(), data["V"].max(), num_points
        )
        df["t"] = 0.0
    else:    
        df['Biomass'] = np.random.uniform(4, 20, num_points)
        df['Glucose'] = np.random.uniform(0, 0.05, num_points)
        df['V'] = np.random.uniform(1.5, 2, num_points)
        df["F"] = np.random.uniform(F_min, F_max, num_points)
        df["t"] = 0.0

    print(f"Dataset shape: {df.shape}")

    t_train = numpy_to_tensor(df["t"].values)
    X_train = numpy_to_tensor(df["Biomass"].values)
    S_train = numpy_to_tensor(df["Glucose"].values)
    V_train = numpy_to_tensor(df["V"].values)
    F_train = numpy_to_tensor(df["F"].values)

    in_train = torch.cat([t_train, X_train, S_train, V_train, F_train], dim=1)
    out_train = torch.cat([X_train, S_train, V_train], dim=1)
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
        self.input = nn.Linear(input_dim, 128)
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.output = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.output(x)
        return x


def loss_fn(
    net: nn.Module,
    data: pd.DataFrame,
    t_start: Union[np.float32, torch.Tensor],
    t_end: Union[np.float32, torch.Tensor],
    Sin: float,
    mu_max: float,
    K_s: float,
    Y_xs: float,
) -> torch.Tensor:
    
    t_col = numpy_to_tensor(np.random.uniform(t_start, t_end, NUM_COLLOCATION))
    X0_col = numpy_to_tensor(np.random.uniform(data["Biomass"].min(), data["Biomass"].max(), NUM_COLLOCATION))
    S0_col = numpy_to_tensor(np.random.uniform(data["Glucose"].min(), data["Glucose"].max(), NUM_COLLOCATION))
    V0_col = numpy_to_tensor(np.random.uniform(data["V"].min(), data["V"].max(), NUM_COLLOCATION))
    F_col = numpy_to_tensor(np.random.uniform(0.015, 0.065, NUM_COLLOCATION))
    
    u_col = torch.cat([t_col, X0_col, S0_col, V0_col, F_col], dim=1)

    preds = net.forward(u_col)

    X_pred = preds[:, 0].view(-1, 1)
    S_pred = preds[:, 1].view(-1, 1)
    V_pred = preds[:, 2].view(-1, 1)

    dXdt_pred = grad(X_pred, t_col)[0]
    dSdt_pred = grad(S_pred, t_col)[0]
    dVdt_pred = grad(V_pred, t_col)[0]

    mu = mu_max * S_pred / (K_s + S_pred)

    error_dXdt = dXdt_pred - mu * X_pred + X_pred * F_col / V0_col
    error_dSdt = dSdt_pred + mu * X_pred / Y_xs - F_col / V0_col * (Sin - S_pred)
    error_dVdt = dVdt_pred - F_col
    
    error_ode = 1/3 * torch.mean(error_dXdt**2) + 1/3 * torch.mean(error_dSdt**2) + 1/3 * torch.mean(error_dVdt**2)

    return error_ode


def main(
    data: pd.DataFrame,
    in_train: torch.Tensor,
    out_train: torch.Tensor,
    t_start: Union[np.float32, torch.Tensor],
    t_end: Union[np.float32, torch.Tensor],
    S_in: float,
    mu_max: float,
    Ks: float,
    Yxs: float,
    verbose: int = 100,
):
    net = PINN(input_dim=5, output_dim=3).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.7)

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
        X_pred = preds[:, 0].view(-1, 1)
        S_pred = preds[:, 1].view(-1, 1)
        V_pred = preds[:, 2].view(-1, 1)
        loss_X = nn.MSELoss()(X_pred, out_train[:, 0].view(-1, 1))
        loss_S = nn.MSELoss()(S_pred, out_train[:, 1].view(-1, 1))
        loss_V = nn.MSELoss()(V_pred, out_train[:, 2].view(-1, 1))
        loss_data = 0.33 * (loss_X + loss_S + loss_V)

        loss_ode = loss_fn(net, data, t_start, t_end, S_in, mu_max, Ks, Yxs)

        loss = w_data * loss_data + w_ode * loss_ode
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % verbose == 0:
            print(
                f"Epoch {epoch}, Loss_data: {loss_data.item():.4f}, Loss_ode: {loss_ode.item():.4f}"
            )
            # Print the current learning rate of the optimizer
            for param_group in optimizer.param_groups:
                print("Current learning rate: ", param_group["lr"])

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
