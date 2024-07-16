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
        self.input = nn.Linear(input_dim, 50)
        self.hidden = nn.Linear(50, 50)
        self.output = nn.Linear(50, output_dim)

        self.mu_max = nn.Parameter(torch.tensor(0.5))
        self.K_s = nn.Parameter(torch.tensor(0.5))
        self.Y_xs = nn.Parameter(torch.tensor(0.5))

        self.t_start = t_start
        self.t_end = t_end
        
        if isinstance(self.t_start, torch.Tensor):
            self.t_start = self.t_start.item()
        if isinstance(self.t_end, torch.Tensor):
            self.t_end = self.t_end.item()
             
    def forward(self, x):
        x = torch.relu(self.input(x))
        x = self.output(x)
        return x

    def loss(self, t, u_true):
        u_pred = self.forward(t)
        mse_u = nn.MSELoss()(u_true, u_pred)
        mse_f = self.loss_f()
        return mse_u + mse_f
    
    def loss_f(self):
        t = torch.linspace(self.t_start, self.t_end, 5).view(-1,1)
        t.requires_grad = True
        u = self.forward(t)
        u_X = u[:, 0].view(-1, 1)
        u_S = u[:, 1].view(-1, 1)
        
        u_X_t = torch.autograd.grad(u_X, t, grad_outputs=torch.ones_like(u_X), create_graph=True)[0]
        u_S_t = torch.autograd.grad(u_S, t, grad_outputs=torch.ones_like(u_S), create_graph=True)[0]
        
        mu = self.mu_max * u_S / (self.K_s + u_S)    
                
        error_X = u_X_t - mu * u_S / (self.K_s + u_S) * u_X
        error_S = u_S_t + mu * u_X / self.Y_xs 
        
        return torch.mean(error_X**2) + torch.mean(error_S**2)
    
    def fit(self, t, u_true, epochs=10000, lr=1e-4, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss(t, u_true)
            loss.backward()
            optimizer.step()
            
            if verbose:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} Loss: {loss.item()}")
                    print(f'mu_max: {self.mu_max.item():.4f} K_s: {self.K_s.item():.4f} Y_xs: {self.Y_xs.item():.4f}')
                
    def get_params(self):
        return self.mu_max.item(), self.K_s.item(), self.Y_xs.item()
    