import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def simulate(df, mu_max, Ks, Yxs):
    mu_max = mu_max
    Ks = Ks
    Yxs = Yxs

    def system_ode(t, y):
        X, S = y
        mu = mu_max * S / (Ks + S)
        dXdt = mu * X
        dSdt = -mu * X / Yxs
        return [dXdt, dSdt]

    t_eval = np.linspace(df["RTime"].min(), df["RTime"].max(), 10000)
    sol = solve_ivp(
        system_ode,
        [df["RTime"].min(), df["RTime"].max()],
        [df["Biomass"].iloc[0], df["Glucose"].iloc[0]],
        t_eval=t_eval,
    )
    return sol