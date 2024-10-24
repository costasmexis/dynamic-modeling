import sys

sys.path.append("../")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from machinelearning_control_fedbatch import generate_dataset, main
from system_ode_fedbatch import generate_data

from src.utils import get_data_and_feed, plot_experiment

FILENAME = '../data/data_processed.xlsx'
EXPERIMENT = 'BR01'
S_IN = 1.43 * 200

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

full_df, feeds = get_data_and_feed(
    file_name=FILENAME, experiment=EXPERIMENT, keep_only="FB"
)

full_df['Biomass'].iloc[1] = 5.0
# add new line to full_df
new_row = pd.DataFrame([{"Process": "FB", "RTime": 5.85, "V": 1.56, "Biomass": 5.8, "Glucose": 0.013, "Protein": 0.0}])
full_df = pd.concat([full_df, new_row], ignore_index=True)
full_df.sort_values(by="RTime", inplace=True)

T_FB = full_df["RTime"].iloc[0] # Time of fed-batch
T_START = 0
T_END = full_df["RTime"].iloc[-1] - T_FB  # End of experiment

# Normalize time
full_df["RTime"] = full_df["RTime"] - T_FB
feeds["Time"] = feeds["Time"] - T_FB

print(f"Dataset shape: {full_df.shape}")

# Get dataset (multiple initial conditions)
in_train, out_train = generate_dataset(data=full_df, num_points=1000)

# parameter values
mumax = 0.6710     # 1/hour
Ks = 0.3086          # g/liter
Yxs = 0.5624         # g/g
Sin = 1.43 * 200  # g/liter

t_start = full_df['RTime'].iloc[0]
t_end = full_df['RTime'].iloc[-1]
T_s = 0.5 

# initial conditions
V0 = full_df['V'].iloc[0]
S0 = full_df['Glucose'].iloc[0]
X0 = full_df['Biomass'].iloc[0]

print(f'T_start = {t_start}')
print(f'T_end = {t_end}')

# Train network
net = main(full_df, in_train, out_train, t_start, t_end, Sin, mumax, Ks, Yxs, verbose=10)
torch.save(net, "pinc_trained_exp.pth")