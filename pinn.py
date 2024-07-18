import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numpy_to_tensor(array):
    return torch.tensor(array, requires_grad=True, dtype=torch.float32).to(DEVICE).reshape(-1,1)

class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, t_start, t_end):
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

                
    def get_params(self):
        return self.mu_max.item(), self.K_s.item(), self.Y_xs.item()
    
    
def loss_fn(net: torch.nn.Module, t_start, t_end):
    t = torch.linspace(t_start, t_end, steps=2000).view(-1, 1).requires_grad_(True)
    
    u_pred = net.forward(t)
    X_pred = u_pred[:, 0].view(-1,1)
    S_pred = u_pred[:, 1].view(-1,1)

    dXdt_pred = torch.autograd.grad(X_pred, t, grad_outputs=torch.ones_like(X_pred), create_graph=True)[0]
    dSdt_pred = torch.autograd.grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]

    mu = net.mu_max * S_pred / (net.K_s + S_pred)

    error_dXdt = nn.MSELoss()(dXdt_pred, mu * X_pred)
    error_dSdt = nn.MSELoss()(dSdt_pred, -mu * X_pred / net.Y_xs)

    error_ode = error_dXdt + error_dSdt
    return error_ode

def train(net, t, X_S, df, num_epochs=1000, verbose=True):
    
    TOTAL_LOSS = []
    LOSS_DATA = []
    LOSS_IC = []
    LOSS_ODE = []
    optimizer = torch.optim.RMSprop(net.parameters(), lr=5e-4)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u_pred = net.forward(t)
        loss_data = nn.MSELoss()(u_pred, X_S)
        loss_ic = nn.MSELoss()(u_pred[0], X_S[0])
        loss_ode = loss_fn(net, df['RTime'].min(), df['RTime'].max())
        
        total_loss = loss_data + loss_ic + loss_ode
        total_loss.backward()
        optimizer.step()
        
        if verbose:
            if epoch % 100 == 0:
                print(f'Epoch {epoch} || Total Loss: {total_loss.item():.2f}')
        
        TOTAL_LOSS.append(total_loss.item())
        LOSS_DATA.append(loss_data.item())
        LOSS_IC.append(loss_ic.item())
        LOSS_ODE.append(loss_ode.item())
        
        # Early stopping
        # if total_loss.item() < 0.07:
            # break
            
    return net, TOTAL_LOSS, LOSS_DATA, LOSS_IC, LOSS_ODE