import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
F0 = 0.05  # Constant feed rate (liter/hour)
IC = [X0, S0, P0, V0]

# Function to calculate volume
def Volume(t):
    return V0 + F0*t

# def Fs(t):
#     return F0

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


def mu(S, mumax, Ks):
    return mumax * S / (Ks + S)

def Rg(X, S, mumax, Ks):
    return mu(S, mumax, Ks) * X

def a(t, alpha):
    return alpha * (1 - np.exp(-t**2))

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

) -> pd.DataFrame:
    X, S, P, V = simulate(mumax, Ks, Yxs, alpha, Sin)
    df = pd.DataFrame(
        {"RTime": t_sim, "Biomass": X, "Glucose": S, "Protein": P, "V": V}
    )
    return df


# Plot solution
def PlotSolution(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    
    # Plot Biomass and Glucose in WWthe first subplot
    axs[0].scatter(df['RTime'], df['Biomass'], label="Biomass", s=10, alpha=1)
    axs[0].scatter(df['RTime'], df['Glucose'], label="Glucose", s=10, alpha=1)
    axs[0].plot(df['RTime'], df['Biomass'], label="_Biomass", alpha=0.2)
    axs[0].plot(df['RTime'], df['Glucose'], label="_Glucose", alpha=0.2)
    axs[0].set_ylabel("Concentration (g/lt)")
    axs[0].legend(loc="best")
    
    # Plot Protein in the second subplot
    axs[1].scatter(df['RTime'], df['Protein'], label="Protein", s=10, alpha=1)
    axs[1].plot(df['RTime'], df['Protein'], label="_Protein", alpha=0.2)
    axs[1].set_xlabel("Time (hours)")
    axs[1].set_ylabel("Concentration (g/lt)")
    axs[1].legend(loc="best")
    
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