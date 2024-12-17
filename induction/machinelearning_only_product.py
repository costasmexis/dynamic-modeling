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
NUM_COLLOCATION = 50
PATIENCE = 100
THRESHOLD = 1e-3
EARLY_STOPPING_EPOCH = 1000


def numpy_to_tensor(array):
    return torch.tensor(array, requires_grad=True, dtype=torch.float32).to(DEVICE).reshape(-1, 1)

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

class PINN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(PINN, self).__init__()
        self.input = nn.Linear(input_dim, 64)
        self.fc1 = nn.Linear(64, 256)
        self.hidden1 = nn.Linear(256, 256)
        self._hidden = nn.Linear(256, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, output_dim)

        # Kinetic parameters
        self.mu_max = torch.tensor([0.75], device=DEVICE)
        
        # Kinetic parameters
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        x = nn.functional.relu(self.input(x))
        x = nn.functional.relu(self.fc1(x))
        # x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self._hidden(x))
        # x = nn.functional.relu(self.hidden2(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.output(x)
        return x


def loss_fn(net: torch.nn.Module, data: pd.DataFrame, t_start: Union[np.float32, torch.Tensor], t_end: Union[np.float32, torch.Tensor], model: str) -> torch.Tensor:

    if isinstance(t_start, torch.Tensor):
        t_start = t_start.item()
    if isinstance(t_end, torch.Tensor):
        t_end = t_end.item()

    data = data[(data['t'] >= t_start) & (data['t'] <= t_end)]
    data = data.iloc[::int(len(data) / NUM_COLLOCATION), :]
    t = torch.tensor(data["t"].values, dtype=torch.float32).view(-1, 1).view(-1, 1).requires_grad_(True).to(DEVICE)
    X = torch.tensor(data["X"].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
    S = torch.tensor(data["S"].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
    V = torch.tensor(data["V"].values, dtype=torch.float32).view(-1, 1).to(DEVICE)    
    F = torch.tensor([Fs(i) for i in t], dtype=torch.float32).view(-1, 1).to(DEVICE)
    # V = torch.tensor([Volume(i) for i in t], dtype=torch.float32).view(-1, 1).to(DEVICE)

    P_pred = net.forward(t)

    dPdt_pred = grad(P_pred, t)[0]

    mu = net.mu_max * S / (K_S + S)

    # Model selection for alpha: a=a(t)
    if model == "A":
        alpha = net.alpha * 1 / (1 + torch.exp(-t))
    elif model == "B":
        alpha = net.alpha * torch.exp(-t)
    elif model == "C":
        alpha = net.alpha - net.beta * t
    else:
        raise ValueError("Model not found!")

    error_ode = nn.MSELoss()(dPdt_pred, alpha * mu * X - P_pred * F / V)

    return error_ode


def main(train_df: pd.DataFrame,
         full_df: pd.DataFrame,
         data: pd.DataFrame,
         num_epochs: int = 10000,
         model: str = "A",
         t_start: float = T_START,
         t_end: float = T_END) -> tuple:
    
    t_train = numpy_to_tensor(train_df["RTime"].values)
    P_train = numpy_to_tensor(train_df["Protein"].values)

    u_train = torch.cat((P_train,), dim=1).to(DEVICE)

    net = PINN(1, 1).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.75)

    # Loss weights
    w_data, w_ode = 1, 1

    # Initialize early stopping variables
    best_loss = float("inf")
    best_model_weights = None
    patience = PATIENCE
    threshold = THRESHOLD

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u_pred = net.forward(t_train)
        
        loss_data = nn.MSELoss()(u_pred, u_train) * w_data

        loss_ode = loss_fn(net, data, t_start, t_end, model) * w_ode
        
        loss = 1/2 * (loss_data + loss_ode) 
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, Loss Data: {loss_data.item():.2f}, Loss ODE: {loss_ode.item():.2f}")
            print(f" *** mu_max: {net.mu_max.item():.2f}, alpha: {net.alpha.item():.2f}, beta: {net.beta.item():.2f}")   

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
    u_pred.columns = ["Protein"]
    u_pred["RTime"] = t_test.detach().cpu().numpy()

    return net, u_pred, loss_data.item(), loss_ode.item()


