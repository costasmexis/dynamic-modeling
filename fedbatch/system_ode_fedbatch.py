import sys

sys.path.append("../")

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from src.utils import feeding_strategy


def simulate(
    feeds: pd.DataFrame,
    mumax: float,
    Ks: float,
    Yxs: float,
    Sin: float,
    T_START: float,
    T_END: float,
    IC: list,
    NUM_SAMPLES: int = 1000,
    return_df: bool = False
):
    """IC: initial conditions [X0, S0, V0]"""

    mumax = mumax
    Ks = Ks
    Yxs = Yxs

    # reaction rates
    def mu(S):
        return mumax * S / (Ks + S)

    def Rg(X, S):
        return mu(S) * X

    # differential equations
    def xdot(x, t):
        X, S, V = x
        dX = -feeding_strategy(feeds=feeds, time=t) * X / V + Rg(X, S)
        dS = feeding_strategy(feeds=feeds, time=t) * (Sin - S) / V - Rg(X, S) / Yxs
        dV = feeding_strategy(feeds=feeds, time=t)
        return [dX, dS, dV]

    t = np.linspace(T_START, T_END, NUM_SAMPLES)
    sol = odeint(xdot, IC, t)

    if return_df:
        df = pd.DataFrame(sol, columns=["Biomass", "Glucose", "Volume"])
        df["RTime"] = t
        return df

    return sol.transpose()

# Plot solution of ODE vs actual datapoints
def PlotSolution(full_df: pd.DataFrame, train_df: pd.DataFrame, df_pred: pd.DataFrame):
    plt.figure(figsize=(12, 4))
    plt.scatter(full_df['RTime'], full_df['Biomass'], color='green', label="_Biomass", s=50, alpha=0.3)
    plt.scatter(full_df['RTime'], full_df['Glucose'], color='orange', label="_Glucose", s=50, alpha=0.3)
    plt.scatter(train_df['RTime'], train_df['Biomass'], color='green', label="Biomass", s=50, alpha=1)
    plt.scatter(train_df['RTime'], train_df['Glucose'], color='orange', label="Glucose", s=50, alpha=1)
    plt.plot(df_pred['RTime'], df_pred['Biomass'], color='green', label="_Biomass", alpha=1)
    plt.plot(df_pred['RTime'], df_pred['Glucose'], color='orange', label="_Glucose", alpha=1)
    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration (g/lt)")
    plt.legend(loc="best")
    plt.show()