from typing import Union
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parameter values
LEARNING_RATE = 1e-4
NUM_COLLOCATION = 50
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
        super().__init__()
        self.input = nn.Linear(input_dim, 64)
        self.hidden = nn.Linear(64, 1024)
        self.hidden2 = nn.Linear(1024, 1024)
        self.hidden3 = nn.Linear(1024, 64)
        self.output = nn.Linear(64, output_dim)

        self.mu_max = nn.Parameter(torch.tensor([0.5]))
        self.K_s = nn.Parameter(torch.tensor([0.5]))
        self.Y_xs = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.tanh(self.hidden(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


def loss_fn(
    net: torch.nn.Module,
    t_start: Union[np.float32, torch.Tensor],
    t_end: Union[np.float32, torch.Tensor],
) -> torch.Tensor:
    
    if isinstance(t_start, torch.Tensor):
        t_start = t_start.item()
    if isinstance(t_end, torch.Tensor):
        t_end = t_end.item()

    t = (
        torch.linspace(t_start, t_end, steps=50)
        .view(-1, 1)
        .requires_grad_(True)
        .to(DEVICE)
    )

    u_pred = net.forward(t)
    X_pred = u_pred[:, 0].view(-1, 1)
    S_pred = u_pred[:, 1].view(-1, 1)

    dXdt_pred = grad(X_pred, t)[0]
    dSdt_pred = grad(S_pred, t)[0]

    mu = net.mu_max * S_pred / (net.K_s + S_pred)

    error_dXdt = nn.MSELoss()(dXdt_pred, mu * X_pred)
    error_dSdt = nn.MSELoss()(dSdt_pred, -mu * X_pred / net.Y_xs)

    error_ode = error_dXdt + error_dSdt

    return error_ode


def main(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    num_epochs: int = 10000,
    verbose: int = 100,
):
    t_train = numpy_to_tensor(train_df["RTime"].values)
    X_train = numpy_to_tensor(train_df["Biomass"].values)
    S_train = numpy_to_tensor(train_df["Glucose"].values)

    u_train = torch.cat([X_train, S_train], dim=1)

    net = PINN(input_dim=1, output_dim=2).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Loss weights
    w_data, w_ode, w_ic = 1, 1, 1

    # Initialize early stopping variables
    best_loss = float("inf")
    best_model_weights = None
    patience = PATIENCE
    threshold = THRESHOLD

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u_pred = net.forward(t_train)

        loss_data = nn.MSELoss()(u_pred, u_train) * w_data
        loss_ode = loss_fn(net, full_df["RTime"].min(), full_df["RTime"].max()) * w_ode
        loss_ic = nn.MSELoss()(u_pred[0], u_train[0]) * w_ic

        loss = loss_data + loss_ode + loss_ic
        loss.backward()
        optimizer.step()

        if epoch % verbose == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")
            print(
                f"mu_max: {net.mu_max.item():.4f}, K_s: {net.K_s.item():.4f}, Y_xs: {net.Y_xs.item():.4f}"
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

    t_test = numpy_to_tensor(full_df["RTime"].values)
    u_pred = pd.DataFrame(
        net.forward(t_test).detach().cpu().numpy(), columns=["Biomass", "Glucose"]
    )
    u_pred["RTime"] = t_test.detach().cpu().numpy()

    return net, u_pred, loss.item()


def plot_net_predictions(
    full_df: pd.DataFrame, train_df: pd.DataFrame, u_pred: pd.DataFrame, title: str
):
    plt.figure(figsize=(12, 3))
    plt.scatter(
        full_df["RTime"],
        full_df["Biomass"],
        color="orange",
        label="_Biomass",
        alpha=0.3,
    )
    plt.scatter(
        full_df["RTime"], full_df["Glucose"], color="green", label="_Glucose", alpha=0.3
    )
    plt.scatter(
        train_df["RTime"],
        train_df["Biomass"],
        color="orange",
        label="Biomass",
        alpha=1.0,
    )
    plt.scatter(
        train_df["RTime"],
        train_df["Glucose"],
        color="green",
        label="Glucose",
        alpha=1.0,
    )

    plt.plot(u_pred["RTime"], u_pred["Biomass"], color="orange", marker='x', label="_Biomass")
    plt.plot(u_pred["RTime"], u_pred["Glucose"], color="green", marker='x', label="_Glucose")

    plt.title(title)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Concentration")

    plt.show()

def validate_predictions(full_df: pd.DataFrame, u_pred: pd.DataFrame, i: int) -> None:
    """ Validate the prediction accuracy of the PINN

    :param full_df: Full training dataset
    :type full_df: pd.DataFrame
    :param u_pred: Predictions of the PINN 
    :type u_pred: pd.DataFrame
    :param i: Number of training data points used for training
    :type i: int
    """
    print('************************************************************************')
    print('************************************************************************')
    
    full_df['Biomass_pred'] = u_pred['Biomass'].values
    full_df['Glucose_pred'] = u_pred['Glucose'].values
    try:
        next_biomass = full_df['Biomass'].iloc[i]
        next_glucose = full_df['Glucose'].iloc[i]
        pred_biomass = full_df['Biomass_pred'].iloc[i]
        pred_glucose = full_df['Glucose_pred'].iloc[i]
        print(f'Biomass error: {abs(next_biomass - pred_biomass)}')
        print(f'Glucose error: {abs(next_glucose - pred_glucose)}')
        print(f'Real Biomass: {next_biomass} || Predicted Biomass: {pred_biomass}')
        print(f'Real Glucose: {next_glucose} || Predicted Glucose: {pred_glucose}')  
    except IndexError:
        pass
    biomass_mse = mean_squared_error(full_df['Biomass'], full_df['Biomass_pred'])
    glucose_mse = mean_squared_error(full_df['Glucose'], full_df['Glucose_pred'])
    print(f'Biomass MSE: {biomass_mse}')
    print(f'Glucose MSE: {glucose_mse}')
        
    
    
    