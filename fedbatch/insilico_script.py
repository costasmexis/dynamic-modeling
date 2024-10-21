import sys

sys.path.append("../")
import numpy as np
import pandas as pd
from machinelearning_fedbatch import main
from system_ode_fedbatch import simulate

from src.utils import get_data_and_feed


FILENAME = "../data/data_processed.xlsx"
EXPERIMENT = "BR01"
S_IN = 1.43 * 200

# Load and manual "fix" data
full_df, feeds = get_data_and_feed(
    file_name=FILENAME, experiment=EXPERIMENT, keep_only="FB"
)
# Fix bad Biomass value
full_df["Biomass"].iloc[1] = 5.0
# Add extra line
new_row = pd.DataFrame(
    [
        {
            "Process": "FB",
            "RTime": 5.85,
            "V": 1.56,
            "Biomass": 5.8,
            "Glucose": 0.013,
            "Protein": 0.0,
        }
    ]
)
full_df = pd.concat([full_df, new_row], ignore_index=True)
full_df.sort_values(by="RTime", inplace=True)

T_FB = full_df["RTime"].iloc[0]  # Time of fed-batch
T_START = 0
T_END = full_df["RTime"].iloc[-1] - T_FB  # End of experiment

# Get initial volume
V0 = full_df["V"].iloc[0]

# Normalize time
full_df["RTime"] = full_df["RTime"] - T_FB
feeds["Time"] = feeds["Time"] - T_FB

print(f"Dataset shape: {full_df.shape}")


# inlet flowrate
def Fs(t):
    if t <= 4.73 - T_FB:
        return 0.017
    elif t <= 7.33 - T_FB:
        return 0.031
    elif t <= 9.17 - T_FB:
        return 0.060
    elif t <= 9.78 - T_FB:
        return 0.031
    else:
        return 0.017


for i in range(2, len(full_df) + 1):
    print(f"Training with {i} samples")
    train_df = full_df.iloc[:i]
    net, u_pred, loss = main(
        train_df=train_df,
        full_df=full_df,
        feeds=feeds,
        Sin=S_IN,
        V0=V0,
        num_epochs=30000,
        verbose=100,
    )

    # Clip u_pred to be positive
    u_pred[u_pred < 0] = 0

    print(f"mu_max = {net.mu_max.item():.4f}")
    print(f"Ks = {net.K_s.item():.4f}")
    print(f"Yxs = {net.Y_xs.item():.4f}")

    title = f"mu_max = {net.mu_max.item():.4f}, Ks = {net.K_s.item():.4f}, Yxs = {net.Y_xs.item():.4f} | Loss = {loss:.4f}"

    # Write title to txt file
    with open(f"./temp/parameters_{i}.txt", "w") as f:
        f.write(title)

    # Save u_pred to file
    u_pred.to_csv(f"./temp/u_pred_{i}.csv", index=False)
