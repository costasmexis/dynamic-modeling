import sys

sys.path.append("../")

import numpy as np
from machinelearning_fedbatch import main, plot_net_predictions, validate_predictions
from system_ode_fedbatch import simulate, PlotSolution
from src.utils import get_data_and_feed, plot_experiment
import matplotlib.pyplot as plt

FILENAME = "../data/data_processed.xlsx"
EXPERIMENT = "BR01"
S_IN = 1.43 * 200

full_df, feeds = get_data_and_feed(
    file_name=FILENAME, experiment=EXPERIMENT, keep_only="FB"
)

# Manually fix some measurements
full_df['Biomass'].iloc[1] = 5.0

# add new line to full_df
full_df = full_df.append(
    {
        "Process": "FB",
        "RTime": 5.85,
        "V": 1.56,
        "Biomass": 5.8,
        "Glucose": 0.013,
        "Protein": 0.0,
        
    }, ignore_index=True)

full_df.sort_values(by="RTime", inplace=True)

# Get initial volume
V0 = full_df["V"].iloc[0]

print(f"Dataset shape: {full_df.shape}")

# inlet flowrate
def Fs(t):
    if t <= 4.73:
        return 0.017
    elif t <= 7.33:
        return 0.031
    elif t <= 9.17:
        return 0.060
    elif t <= 9.78:
        return 0.031
    else:
        return 0.017

T_START = full_df['RTime'].iloc[0]
T_END = full_df['RTime'].iloc[-1]

# Plot Fs(t) 
Fs_t = [Fs(i) for i in np.linspace(T_START,T_END,1000)]

t_start = full_df["RTime"].iloc[0]
t_end = full_df["RTime"].iloc[-1]
IC = [full_df["Biomass"].iloc[0], full_df["Glucose"].iloc[0], V0]
sol_df = simulate(
    feeds,
    0.724,
    0.160,
    0.660,
    S_IN,
    t_start,
    t_end,
    NUM_SAMPLES=1000,
    IC=IC,
    return_df=True
)

for i in range(2, len(full_df)+1):
    print(f"Training with {i} samples")
    train_df = full_df.iloc[:i]
    net, u_pred, loss = main(train_df=train_df, full_df=full_df, 
                             feeds=feeds, Sin=S_IN, V0=V0, num_epochs=30000, verbose=100)
    
    # Clip u_pred to be positive
    u_pred[u_pred < 0] = 0
    
    print(f"mu_max = {net.mu_max.item():.4f}")
    print(f"Ks = {net.K_s.item():.4f}")
    print(f"Yxs = {net.Y_xs.item():.4f}")

    title = f"mu_max = {net.mu_max.item():.4f}, Ks = {net.K_s.item():.4f}, Yxs = {net.Y_xs.item():.4f} | Loss = {loss:.4f}"
    
    # Write title to txt file
    with open(f"./temp/title_{i}.txt", "w") as f:
        f.write(title)
    
    # Save u_pred to file
    u_pred.to_csv(f"./temp/u_pred_{i}.csv", index=False)