import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

np.random.seed(0)

# Define parameters
T_START, T_END = 0, 12
NUM_SAMPLES = 25

# Simulation time points
t_sim = np.linspace(T_START, T_END, NUM_SAMPLES)

# Kinetic parameters
MU_MAX = 0.75  # 1/hour
K_S = 0.20  # g/liter
Y_XS = 0.40  # g/g
S_IN = 1.43 * 200
ALPHA = 0.30

# Initial conditions
X0, S0, P0, V0 = 4.163095, 0.013, 0.0, 1.55
IC = [X0, S0, P0, V0]

# inlet flowrate
def Fs(t):
    if t <= 0:
        return 0.03
    elif t <= 4:
        return 0.05
    elif t <= 6:
        return 0.060
    elif t <= 8:
        return 0.030
    else:
        return 0.020

def Volume(t):
    return V0 + Fs(t) * t

def mu(S, mumax, Ks):
    return mumax * S / (Ks + S)

def Rg(X, S, mumax, Ks):
    return mu(S, mumax, Ks) * X

def a(t, alpha: list):
    '''
    - Sigmoidal Increase: a(t) = a_max * 1 / (1 + exp(-k(t-t0))
    - Exponentially Decaying: a(t) = a_0 exp(-k t)
    - Periodic fluctuation: a(t) = a_0 + a_1 sin(ωt + φ)
    - Linearly Decreasing: a(t) = a_0 - k t
    '''
    if t < 4:
        return alpha[0] * 1 / (1 + np.exp(-t))
    elif t < 10:
        return alpha[0] * np.exp(-t) 
    else:
        return alpha[1] - alpha[2] * t

def simulate(
    mumax: float = MU_MAX,
    Ks: float = K_S,
    Yxs: float = Y_XS,
    alpha: float = ALPHA,
    Sin: float = S_IN,
):

    # differential equations
    def SystemODE(x, t):
        X, S, P, V = x
        dX = -Fs(t) * X / V + Rg(X, S, mumax, Ks)
        dP = -Fs(t) * P / V + a(t, alpha) * Rg(X, S, mumax, Ks)
        dS = Fs(t) * (Sin - S) / V - Rg(X, S, mumax, Ks) / Yxs
        dV = Fs(t)
        return [dX, dS, dP, dV]

    sol = odeint(SystemODE, IC, t_sim)

    return sol.transpose()


# Get dataset
def GetDataset(
    mumax: float = MU_MAX,
    Ks: float = K_S,
    Yxs: float = Y_XS,
    alpha: float = ALPHA,
    Sin: float = S_IN,
    noise: bool = False,

) -> pd.DataFrame:
    X, S, P, V = simulate(mumax, Ks, Yxs, alpha, Sin)
    df = pd.DataFrame({"RTime": t_sim, "Biomass": X, "Glucose": S, "Protein": P, "V": V})
    if noise:
        # Make the noise reproducible
        np.random.seed(0)
        # Add noise to the dataset
        df["Biomass"] += np.random.normal(0, 0.2, NUM_SAMPLES)
        df["Glucose"] += np.random.normal(0, 0.01, NUM_SAMPLES)
        df["Protein"] += np.random.normal(0, 0.05, NUM_SAMPLES)
        # df["V"] += np.random.normal(0, 0.01, NUM_SAMPLES)

    df.loc[df["Biomass"] < 0, "Biomass"] = np.random.uniform(0, 0.05)
    df.loc[df["Glucose"] < 0, "Glucose"] = np.random.uniform(0, 0.05)
    df.loc[df["Protein"] < 0, "Protein"] = np.random.uniform(0, 0.05)
    df.loc[df["V"] < 0, "V"] = 0

    return df


# Plot solution
def PlotSolution(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    
    # Plot Biomass and Glucose in WWthe first subplot
    axs[0].scatter(df['RTime'], df['Biomass'], label="Biomass", s=10, alpha=1)
    axs[0].scatter(df['RTime'], df['Glucose'], label="Glucose", s=10, alpha=1)
    axs[0].plot(df['RTime'], df['Biomass'], label="_Biomass", alpha=0.2)
    axs[0].plot(df['RTime'], df['Glucose'], label="_Glucose", alpha=0.2)
    axs[0].set_ylabel("Concentration (g/lt)")
    axs[0].legend(loc="upper left")
    
    # Plot Protein in the second subplot
    axs[1].scatter(df['RTime'], df['Protein'], label="Protein", color='green', s=10, alpha=1)
    axs[1].plot(df['RTime'], df['Protein'], label="_Protein", color='green', alpha=0.2)
    axs[1].set_xlabel("Time (hours)")
    axs[1].set_ylabel("Concentration (g/lt)")
    axs[1].legend(loc="upper left")
    
    # Plot Volume in the third subplot
    axs[2].scatter(df['RTime'], df['V'], label="Volume", color='black', s=10, alpha=1)
    axs[2].plot(df['RTime'], df['V'], label="_Volume", color='black', alpha=0.2)
    axs[2].set_xlabel("Time (hours)")
    axs[2].set_ylabel("Volume")
    axs[2].legend(loc="upper left")
    
    plt.show()

# Plot predictions vs actual
def PlotPredictions(train_df: pd.DataFrame, df_pred: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 4))
    plt.scatter(train_df['RTime'], train_df['Biomass'], label="Biomass", s=50, alpha=0.3)
    plt.scatter(train_df['RTime'], train_df['Glucose'], label="Glucose", s=50, alpha=0.3)
    plt.scatter(train_df['RTime'], train_df['Protein'], label="Protein", s=50, alpha=0.3)
    plt.plot(df_pred['RTime'], df_pred['Biomass'], label="_Biomass", alpha=1)
    plt.plot(df_pred['RTime'], df_pred['Glucose'], label="_Glucose", alpha=1)
    plt.plot(df_pred['RTime'], df_pred['Protein'], label="_Protein", alpha=1)
    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration (g/lt)")
    plt.legend(loc="best")
    plt.show()