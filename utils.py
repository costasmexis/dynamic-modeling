import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import savgol_filter

pd.options.mode.chained_assignment = None

def get_data(file_name: str = "./Data.xlsx"):
    # Read excel file and see avaialble sheets
    xls = pd.ExcelFile(file_name)
    (xls.sheet_names)
    data = xls.parse("Raw data")
    # Fill NaN in 'Process' column by last valid value
    data["Process"] = data["Process"].ffill()
    data["Process"] = data["Process"].replace(
        {"Batch": "B", "Fed Batch": "FB", "Fed Batch & Induction": "FBI"}
    )
    data.set_index("Label", inplace=True)
    data.index.name = None
    columns = [
        "Process",
        "RTime",
        # "PTime",
        "Glucose",
        "Biomass",
        "Protein",
        "Temperature",
        "Induction",
        "V"
    ]
    data.columns = columns
    return data


def plot_experiment(df: pd.DataFrame, title: str) -> None:
    induction = df["Induction"].sum()
    plt.figure(figsize=(12, 3))
    plt.scatter(df["RTime"], df["Glucose"], c="r", label="Glucose", s=10, alpha=0.5)
    plt.scatter(df["RTime"], df["Biomass"], c="g", label="Biomass", s=10, alpha=0.5)
    if induction > 0:
        plt.scatter(df["RTime"], df["Protein"], c="b", label="Protein", s=10, alpha=0.5)
    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration")
    plt.title(title)
    plt.legend()
    plt.show()


def filter_dataset(
    df: pd.DataFrame, window_length: int, polyorder: int
) -> pd.DataFrame:
    induction = df["Induction"].sum()
    df["Glucose_filter"] = savgol_filter(df["Glucose"], window_length, polyorder)
    df["Biomass_filter"] = savgol_filter(df["Biomass"], window_length, polyorder)
    if induction > 0:
        df["Protein_filter"] = savgol_filter(df["Protein"], window_length, polyorder)
    return df


def plot_filter_vs_raw(df: pd.DataFrame, title: str) -> None:
    induction = df["Induction"].sum()
    plt.figure(figsize=(12, 3))
    plt.scatter(df["RTime"], df["Glucose"], c="r", label="Glucose", s=10, alpha=0.5)
    plt.plot(
        df["RTime"], df["Glucose_filter"], c="r", label="Glucose filter", linewidth=0.8
    )
    plt.scatter(df["RTime"], df["Biomass"], c="g", label="Biomass", s=10, alpha=0.5)
    plt.plot(
        df["RTime"], df["Biomass_filter"], c="g", label="Biomass filter", linewidth=0.8
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration")
    plt.title(f"Glucose and Biomass {title}")
    plt.show()

    if induction > 0:
        plt.figure(figsize=(12, 3))
        plt.scatter(df["RTime"], df["Protein"], c="b", label="Protein", s=10, alpha=0.5)
        plt.plot(
            df["RTime"],
            df["Protein_filter"],
            c="b",
            label="Protein filter",
            linewidth=0.8,
        )
        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration")
        plt.title(f"Protein {title}")
        plt.show()
