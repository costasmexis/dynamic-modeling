import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import savgol_filter
from typing import Tuple

pd.options.mode.chained_assignment = None

def get_data_and_feed(file_name: str, experiment: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = get_data(file_name=file_name)
    df = data.loc[experiment]
    feeds = pd.read_excel(file_name, sheet_name='Feeds')
    feeds.drop(columns='index', inplace=True)
    feeds.columns = ['Time', 'Duration', 'F', 'Induction', 'Label']
    feeds.set_index('Label', inplace=True)
    feeds.index.name = None
    feeds = feeds.loc[experiment]
    
    return df, feeds

def get_data(file_name: str) -> pd.DataFrame:
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


def feeding_strategy(feeds: pd.DataFrame, time: float) -> float:
    for _, row in feeds.iterrows():
        start_time = row["Time"]
        end_time = row["Time"] + row["Duration"]
        if start_time <= time < end_time:
            return row["F"] / 1000
    return 0