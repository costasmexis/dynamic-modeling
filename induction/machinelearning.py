from typing import Union
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from system_ode import Fs, Volume, K_S, Y_XS, S_IN, T_START, T_END

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parameter values
LEARNING_RATE = 1e-3
NUM_COLLOCATION = 25
PATIENCE = 1000
THRESHOLD = 1e-3
EARLY_STOPPING_EPOCH = 1

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
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super(PINN, self).__init__()
        self.input = nn.Linear(input_dim, 64)
        self.fc1 = nn.Linear(64, 128)
        self.hidden1 = nn.Linear(128, 256)
        self._hidden = nn.Linear(256, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, output_dim)

        # Kinetic parameters
        self.mu_max = nn.Parameter(torch.tensor([0.5]))
        self.alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        x = nn.functional.gelu(self.input(x))
        x = nn.functional.gelu(self.fc1(x))
        x = nn.functional.gelu(self.hidden1(x))
        x = nn.functional.gelu(self._hidden(x))
        x = nn.functional.gelu(self.hidden2(x))
        x = nn.functional.gelu(self.fc2(x))
        x = self.output(x)
        return x

def loss_fn(
    net: torch.nn.Module,
    t_start: Union[np.float32, torch.Tensor] = T_START,
    t_end: Union[np.float32, torch.Tensor] = T_END,
) -> torch.Tensor:
    if isinstance(t_start, torch.Tensor):
        t_start = t_start.item()
    if isinstance(t_end, torch.Tensor):
        t_end = t_end.item()

    t = (
        torch.linspace(
            t_start,
            t_end,
            steps=NUM_COLLOCATION,
        )
        .view(-1, 1)
        .requires_grad_(True)
        .to(DEVICE)
    )
    F = torch.tensor([Fs(i) for i in t], dtype=torch.float32).view(-1, 1).to(DEVICE)
    V = torch.tensor([Volume(i) for i in t], dtype=torch.float32).view(-1, 1).to(DEVICE)

    u_pred = net.forward(t)
    X_pred = u_pred[:, 0].view(-1, 1)
    S_pred = u_pred[:, 1].view(-1, 1)
    P_pred = u_pred[:, 2].view(-1, 1)

    dXdt_pred = grad(X_pred, t)[0]
    dSdt_pred = grad(S_pred, t)[0]
    dPdt_pred = grad(P_pred, t)[0]

    mu = net.mu_max * S_pred / (K_S + S_pred)

    error_dXdt = nn.MSELoss()(dXdt_pred, mu * X_pred - X_pred * F / V)
    error_dSdt = nn.MSELoss()(dSdt_pred, -mu * X_pred / Y_XS + F / V * (S_IN - S_pred))
    error_dPdt = nn.MSELoss()(dPdt_pred, net.alpha * mu * X_pred - P_pred * F / V)

    error_ode = error_dXdt + error_dSdt + error_dPdt

    return error_ode


def main(train_df: pd.DataFrame, full_df: pd.DataFrame, num_epochs: int = 10000):
    t_train = numpy_to_tensor(train_df["RTime"].values)
    X_train = numpy_to_tensor(train_df["Biomass"].values)
    S_train = numpy_to_tensor(train_df["Glucose"].values)
    P_train = numpy_to_tensor(train_df["Protein"].values)

    u_train = torch.cat((X_train, S_train, P_train), dim=1).to(DEVICE)

    net = PINN(1, 3).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.75)

    # Loss weights
    w_data, w_ode, w_ic = 1, 1, 0

    # Initialize early stopping variables
    best_loss = float("inf")
    best_model_weights = None
    patience = PATIENCE
    threshold = THRESHOLD

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u_pred = net.forward(t_train)

        loss_data = nn.MSELoss()(u_pred, u_train) * w_data
        loss_ode = loss_fn(net, T_START, T_END) * w_ode
        loss_ic = nn.MSELoss()(u_pred[0, :], u_train[0, :]) * w_ic

        loss = loss_data + loss_ode + loss_ic
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Loss Data: {loss_data.item():.4f}, Loss ODE: {loss_ode.item():.4f}, Loss IC: {loss_ic.item():.4f}"
            )
            print(f"mu_max: {net.mu_max.item():.2f}, alpha: {net.alpha.item():.2f}")

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

    t_test = numpy_to_tensor(full_df["RTime"].values)
    u_pred = pd.DataFrame(net.forward(t_test).detach().cpu().numpy())
    u_pred.columns = ["Biomass", "Glucose", "Protein"]
    u_pred["RTime"] = t_test.detach().cpu().numpy()

    return net, u_pred


def plot_net_predictions(
    full_df: pd.DataFrame, train_df: pd.DataFrame, u_pred: pd.DataFrame
):
    _, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].scatter(
        full_df["RTime"],
        full_df["Biomass"],
        color="orange",
        label="_Biomass",
        alpha=0.3,
    )
    ax[0].scatter(
        full_df["RTime"], full_df["Glucose"], color="green", label="_Glucose", alpha=0.3
    )
    ax[0].scatter(
        train_df["RTime"],
        train_df["Biomass"],
        color="orange",
        label="Biomass",
        alpha=1.0,
    )
    ax[0].scatter(
        train_df["RTime"],
        train_df["Glucose"],
        color="green",
        label="Glucose",
        alpha=1.0,
    )
    ax[1].scatter(
        full_df["RTime"], full_df["Protein"], color="blue", label="_Protein", alpha=0.3
    )
    ax[1].scatter(
        train_df["RTime"], train_df["Protein"], color="blue", label="Protein", alpha=1.0
    )

    ax[0].plot(u_pred["RTime"], u_pred["Biomass"], color="orange", label="Biomass_A")
    ax[0].plot(u_pred["RTime"], u_pred["Glucose"], color="green", label="Glucose_A")
    ax[1].plot(u_pred["RTime"], u_pred["Protein"], color="blue", label="Protein_A")

    ax[0].legend()
    ax[1].legend()

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Concentration")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Concentration")

    plt.show()
