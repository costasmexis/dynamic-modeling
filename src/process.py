import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BatchProcess:
    def __init__(self, df: pd.DataFrame, mu_max: float, Ks: float, Yxs: float):
        self.df = df[df["Process"] == "B"]
        self.t_span = (self.df["RTime"].values[0], self.df["RTime"].values[-1])
        self.S0 = self.df["Glucose"].values[0]
        self.X0 = self.df["Biomass"].values[0]
        self.mu_max = mu_max
        self.Ks = Ks
        self.Yxs = Yxs
        self.sol = None

    def system_ode(self, t, y):
        X, S = y
        mu = self.mu_max * S / (self.Ks + S)
        dXdt = mu * X
        dSdt = -mu * X / self.Yxs
        return [dXdt, dSdt]

    def simulate(self, eval: bool = True) -> pd.DataFrame:
        if eval:
            self.sol = solve_ivp(
                self.system_ode,
                self.t_span,
                [self.X0, self.S0],
                t_eval=self.df["RTime"].values,
            )
        else:
            t_eval = np.linspace(self.t_span[0], self.t_span[1], 10000)
            self.sol = solve_ivp(
                self.system_ode, self.t_span, [self.X0, self.S0], t_eval=t_eval
            )

    def plot_simulation(self, title: str) -> None:
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(self.sol.t, self.sol.y[0], label="Biomass", color="green")
        plt.scatter(
            self.df["RTime"], self.df["Biomass"], color="green", label="Exp. Biomass"
        )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Biomass")
        plt.subplot(3, 1, 2)
        plt.plot(self.sol.t, self.sol.y[1], label="Substrate", color="blue")
        plt.scatter(
            self.df["RTime"], self.df["Glucose"], color="blue", label="Exp. Glucose"
        )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Substrate")
        plt.tight_layout()
        plt.suptitle(title, y=1.01)
        plt.show()


class FedBatchProcess:
    def __init__(
        self,
        df: pd.DataFrame,
        feeds: pd.DataFrame,
        mu_max: float,
        Ks: float,
        Yxs: float,
    ):
        self.df = df
        self.feeds = feeds

        self.t_span = (self.df["RTime"].values[0], self.df["RTime"].values[-1])
        self.S0 = self.df["Glucose"].values[0]
        self.X0 = self.df["Biomass"].values[0]
        self.V0 = self.df["V"].values[0]

        self.mu_max = mu_max
        self.Ks = Ks
        self.Yxs = Yxs

        self.sol = None

        self.Sin = 1.43 * 200  # Substrate concentration in feed (g/L)

    def feeding_strategy(self, time):
        for index, row in self.feeds.iterrows():
            start_time = row["Time"]
            end_time = row["Time"] + row["Duration"]
            if start_time <= time < end_time:
                return row["F"] / 1000
        return 0

    def system_ode(self, t, y):
        X, S, V = y
        mu = self.mu_max * S / (self.Ks + S)  # Monod equation for specific growth rate
        q_S = mu / self.Yxs  # Specific substrate uptake rate

        F = self.feeding_strategy(t)

        dVdt = F
        dXdt = mu * X - X * (F / V)
        dSdt = -q_S * X + (self.Sin - S) * (F / V)

        return [dXdt, dSdt, dVdt]

    def simulate(self, eval: bool = True) -> pd.DataFrame:
        y0 = [self.X0, self.S0, self.V0]

        if eval:
            self.sol = solve_ivp(
                self.system_ode,
                self.t_span,
                y0,
                t_eval=self.df["RTime"].values,
            )
        else:
            t_eval = np.linspace(self.t_span[0], self.t_span[1], 1000)
            self.sol = solve_ivp(self.system_ode, self.t_span, y0, t_eval=t_eval)

    def plot_simulation(self, title: str) -> None:
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(self.sol.t, self.sol.y[0], label="Biomass", color="green")
        plt.scatter(
            self.df["RTime"], self.df["Biomass"], color="green", label="Exp. Biomass"
        )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Biomass")
        plt.subplot(3, 1, 2)
        plt.plot(self.sol.t, self.sol.y[1], label="Substrate", color="blue")
        plt.scatter(
            self.df["RTime"], self.df["Glucose"], color="blue", label="Exp. Glucose"
        )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Substrate")
        plt.tight_layout()
        plt.suptitle(title, y=1.01)
        plt.show()
