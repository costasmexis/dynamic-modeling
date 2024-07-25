import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.process import BatchProcess, FedBatchProcess
from src.fed_batch_pinn import PINN, numpy_to_tensor, train
from src.utils import get_data_and_feed, feeding_strategy

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
plt.legend()
plt.savefig('plot.png')
