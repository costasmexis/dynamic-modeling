from typing import Union

import numpy as np
import torch
import torch.nn as nn
import copy

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

def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )

def loss_ode(net: torch.nn.Module, t_start, t_end):
    if isinstance(t_start, torch.Tensor):
        t_start = t_start.item()
    if isinstance(t_end, torch.Tensor):
        t_end = t_end.item()

    t = torch.linspace(t_start, t_end, steps=50).view(-1, 1).requires_grad_(True).to(DEVICE)

    u_pred = net.forward(t)
    X_pred = u_pred[:, 0].view(-1, 1)
    S_pred = u_pred[:, 1].view(-1, 1)

    dXdt_pred = grad(X_pred, t)[0]
    dSdt_pred = grad(S_pred, t)[0]

    mu = net.mu_max * S_pred / (net.K_s + S_pred)
    
    error_dXdt = (dXdt_pred - mu * X_pred) 
    error_dSdt = (dSdt_pred + mu * X_pred / net.Y_xs) 

    error_ode = torch.mean(error_dXdt**2 + error_dSdt**2) 
    return error_ode


def train(net, t, X_S, df, num_epochs=1000, verbose=True):
    # Initialize early stopping variables
    best_loss = float('inf')
    best_model_weights = None
    patience = 1000

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u_pred = net.forward(t)
        
        loss_data = nn.MSELoss()(u_pred, X_S)
        loss_ic = nn.MSELoss()(u_pred[0], X_S[0]) 
        loss_pde = loss_ode(net, df["RTime"].min(), df["RTime"].max()) 
        
        total_loss = loss_data + loss_ic + loss_pde
        total_loss.backward()
        optimizer.step()

        if verbose:
            if epoch % 500 == 0:
                print(f"Epoch {epoch} || Total Loss: {total_loss.item():.4f}, Loss Data: {loss_data.item():.4f}, Loss IC: {loss_ic.item():.4f}, Loss ODE: {loss_pde.item():.4f}")
                print(f"mu_max: {net.mu_max.item():.4f}, K_s: {net.K_s.item():.4f}, Y_xs: {net.Y_xs.item():.4f}")   
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_model_weights = copy.deepcopy(net.state_dict())
            patience = 1000
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch}")
                net.load_state_dict(best_model_weights)
                break
        
    return net, total_loss.item()
