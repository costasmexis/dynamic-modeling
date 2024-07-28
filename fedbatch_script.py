import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src.fed_batch_pinn import PINN, numpy_to_tensor, train
from src.utils import get_data_and_feed

pd.options.mode.chained_assignment = None
np.set_printoptions(precision=4)

FILENAME = "./data/data_processed.xlsx"
EXPERIMENT = "BR01"


def plot_results(
    train_df: pd.DataFrame, full_df: pd.DataFrame, net_df: pd.DataFrame, title: str
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(net_df["RTime"], net_df["Biomass"], label="X (PINN)")
    plt.plot(net_df["RTime"], net_df["Glucose"], label="S (PINN)")
    plt.scatter(
        full_df["RTime"], full_df["Biomass"], c="g", label="X (all)", s=10, alpha=0.2
    )
    plt.scatter(
        full_df["RTime"], full_df["Glucose"], c="r", label="S (all)", s=10, alpha=0.2
    )
    plt.scatter(
        train_df["RTime"],
        train_df["Biomass"],
        c="g",
        label="X (train)",
        s=10,
        alpha=0.5,
    )
    plt.scatter(
        train_df["RTime"],
        train_df["Glucose"],
        c="r",
        label="S (train)",
        s=10,
        alpha=0.5,
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration")
    plt.title(title)
    plt.legend()
    plt.savefig(f"./plots/plot_{len(train_df)}.png")


def generate_data(df: pd.DataFrame, num_points: int = 25):
    t_train = df["RTime"].values
    y_train = df[["Biomass", "Glucose", "V"]].values

    poly = PolynomialFeatures(degree=3)
    t_train = poly.fit_transform(t_train.reshape(-1, 1))
    lin_reg = LinearRegression()
    lin_reg.fit(t_train, y_train)

    t_sim = np.linspace(df["RTime"].min(), df["RTime"].max(), num_points)
    t_sim_poly = poly.fit_transform(t_sim.reshape(-1, 1))
    y_sim = lin_reg.predict(t_sim_poly)
    sim_df = pd.DataFrame(
        {
            "RTime": t_sim,
            "Biomass": y_sim[:, 0],
            "Glucose": y_sim[:, 1],
            "V": y_sim[:, 2],
        }
    )
    return sim_df


def main(
    filename: str = FILENAME, experiment: str = EXPERIMENT, num_epochs: int = 1000
):
    df, feeds = get_data_and_feed(filename, experiment)

    # Keep only FED-BATCH data
    df = df[df["Process"] == "FB"]

    # Generate data
    # df = generate_data(df, num_points=25)

    for i in range(2, len(df) + 1):
        print(f"Training using {i} data points")

        _df = df.iloc[:i]
        t_start, t_end = _df["RTime"].min(), _df["RTime"].max()

        t = numpy_to_tensor(_df["RTime"].values)
        X = numpy_to_tensor(_df["Biomass"].values)
        S = numpy_to_tensor(_df["Glucose"].values)
        V = numpy_to_tensor(_df["V"].values)
        u_train = torch.cat((X, S, V), 1)

        net = PINN(1, 3, t_start, t_end)
        net = train(net, t, u_train, df, feeds, num_epochs=num_epochs, verbose=False)

        # Store the resutls
        net_df = pd.DataFrame(columns=["RTime", "Biomass", "Glucose"])
        t_test = df["RTime"].values
        net_df["RTime"] = t_test
        t_test = numpy_to_tensor(t_test)
        net_df["Biomass"] = net.forward(t_test).detach().cpu().numpy()[:, 0]
        net_df["Glucose"] = net.forward(t_test).detach().cpu().numpy()[:, 1]
        net_df["V"] = net.forward(t_test).detach().cpu().numpy()[:, 2]

        title = f"mu_max: {net.mu_max.item():4f}, Ks: {net.K_s.item():4f}, Yxs: {net.Y_xs.item():.4f}"

        plot_results(train_df=_df, full_df=df, net_df=net_df, title=title)


if __name__ == "__main__":
    print("Training PINN for Fed-Batch Process")
    main(num_epochs=5000)
