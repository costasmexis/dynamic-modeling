import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

from src.fed_batch_pinn import PINN, numpy_to_tensor, train
from src.utils import feeding_strategy, get_data_and_feed

pd.options.mode.chained_assignment = None

np.set_printoptions(precision=4)

FILENAME = './data/data_processed.xlsx'
EXPERIMENT = 'BR07'

df, feeds = get_data_and_feed(FILENAME, EXPERIMENT)

# Get Fed-Batch start time and initial volume
t_fedbatch = df[df['Process'] == 'FB']['RTime'].min()
v_fedbatch = df[df['Process'] == 'FB']['V'].iloc[0]

def simulate(df, y0, t_span, mu_max, Ks, Yxs):
    mu_max = mu_max
    Ks = Ks
    Yxs = Yxs
    
    def system_ode(t, y):
        X, S, V = y
        
        Sin = 1.43 * 200
        if t < t_fedbatch:
            F = 0
        else:
            F = feeding_strategy(feeds, t)
        
        mu = mu_max * S / (Ks + S)
        dVdt = F
        dXdt = mu * X - X * F / V
        dSdt = - mu * X / Yxs + F / V * (Sin - S)
        return [dXdt, dSdt, dVdt]
    
    t_eval = np.linspace(df['RTime'].min(), df['RTime'].max(), 10000)
    sol = solve_ivp(system_ode, t_span=t_span, y0=y0, t_eval=t_eval)
    return sol

# Keep only Batch and Fed-Batch data
df = df[(df['Process'] == 'B') | (df['Process'] == 'FB')]

t_start, t_end = df['RTime'].min(), df['RTime'].max()
t_span = (t_start, t_end)
X0, S0, V0 = df['Biomass'].iloc[0], df['Glucose'].iloc[0], df['V'].iloc[0]
y0 = [X0, S0, V0]

# Simulate the system
sol = simulate(df, y0, t_span, 0.870, 0.214, 0.496)

# Plot the simulation
plt.figure(figsize=(12, 3))
plt.plot(sol.t, sol.y[0], label='X (sim)')
plt.plot(sol.t, sol.y[1], label='S (sim)')
plt.scatter(df['RTime'], df['Biomass'], c='g', label='X (data)', s=10, alpha=0.5)
plt.scatter(df['RTime'], df['Glucose'], c='r', label='S (data)', s=10, alpha=0.5)
plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.title('Fed-Batch Simulation using ODE')
plt.legend()
plt.savefig('plot.png')

# TODO: Implement PINN for Batch and Fed-Batch process
## PINN

# Keep only FED-BATCH data
df = df[df['Process'] == 'FB']

net = PINN(1, 3, t_start, t_end)

t_start, t_end = df['RTime'].min(), df['RTime'].max()

t = numpy_to_tensor(df['RTime'].values)
X = numpy_to_tensor(df['Biomass'].values)
S = numpy_to_tensor(df['Glucose'].values)
V = numpy_to_tensor(df['V'].values)
u_train = torch.cat((X, S, V), 1)

net, total_loss, loss_data, loss_ode = \
    train(net, t, u_train, df, feeds, num_epochs=2000, verbose=True)
    
# Store the results
net_df = pd.DataFrame(columns=['RTime', 'Biomass', 'Glucose'])
t_test = df['RTime'].values
net_df['RTime'] = t_test
t_test = numpy_to_tensor(t_test)
net_df['Biomass'] = net.forward(t_test).detach().cpu().numpy()[:, 0]
net_df['Glucose'] = net.forward(t_test).detach().cpu().numpy()[:, 1]
net_df['V'] = net.forward(t_test).detach().cpu().numpy()[:, 2]

mu_max = net.mu_max.item()
Ks = net.K_s.item()
Yxs = net.Y_xs.item()

print(f'mu_max: {mu_max:4f}, Ks: {Ks:4f}, Yxs: {Yxs:.4f}')

plt.figure(figsize=(12, 3))
plt.plot(net_df['RTime'], net_df['Biomass'], label='X (PINN)')
plt.plot(net_df['RTime'], net_df['Glucose'], label='S (PINN)')
plt.scatter(df['RTime'], df['Biomass'], c='g', label='X (data)', s=10, alpha=0.5)
plt.scatter(df['RTime'], df['Glucose'], c='r', label='S (data)', s=10, alpha=0.5)
plt.xlabel('Time (hours)')
plt.ylabel('Concentration')
plt.title('Fed-Batch Simulation using PINN')
plt.legend()
plt.savefig('plot.png')