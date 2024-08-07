import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fed_batch_pinn import PINN, numpy_to_tensor, train
from src.utils import get_data_and_feed
from typing import Optional, Union
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn

def plot_feed(feeds):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(feeds['Time'], feeds['F'], width=feeds['Duration'], align='edge')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Feed (mL/h)')
    ax.set_title('Feed vs Time')
    plt.show()
    
def plot_simulation(t=None, y=None, feeds: Optional[pd.DataFrame] = None, full_df: Optional[pd.DataFrame] = None, train_df: Optional[pd.DataFrame] = None, net_df: Optional[pd.DataFrame] = None, title: Optional[str] = None, save: Optional[bool] = False):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if t is not None and y is not None:
        ax1.plot(t, y[0], label='Biomass (ODE)', alpha=0.6)
        ax1.plot(t, y[1], label='Glucose (ODE)', alpha=0.6)
    
    if full_df is not None:
        ax1.scatter(full_df['RTime'], full_df['Glucose'], label='Glucose (EXP)', color='red', alpha=0.2)   
        ax1.scatter(full_df['RTime'], full_df['Biomass'], label='Biomass (EXP)', color='green', alpha=0.2)
    
    if train_df is not None:
        ax1.scatter(train_df['RTime'], train_df['Glucose'], label='_Glucose (Train)', color='red', alpha=1)   
        ax1.scatter(train_df['RTime'], train_df['Biomass'], label='_Biomass (Train)', color='green', alpha=1)
    
    if net_df is not None:

        # Check if len(net_df) == len(full_df); If yes->scatter, else->plot
        if len(net_df) == len(full_df):
            ax1.scatter(net_df['RTime'], net_df['Glucose'], label='Glucose (PINN)', marker='x', color='red', alpha=0.5)
            ax1.scatter(net_df['RTime'], net_df['Biomass'], label='Biomass (PINN)', marker='x', color='green', alpha=0.5)
        else:
            ax1.plot(net_df['RTime'], net_df['Glucose'], label='Glucose (PINN)', color='red', alpha=0.5)
            ax1.plot(net_df['RTime'], net_df['Biomass'], label='Biomass (PINN)', color='green', alpha=0.5)

    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration")
    plt.title(title)
    
    if feeds is not None:
        ax2 = ax1.twinx()
        ax2.bar(feeds['Time'], feeds['F'], width=feeds['Duration'], \
            align='edge', label='Feed', alpha=0.5, color=None, \
            edgecolor='black', linewidth=1, fill=False)
        ax2.set_ylabel('Feed Rate')

    
    handles1, labels1 = ax1.get_legend_handles_labels()
    if feeds is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        ax2.legend(handles, labels, loc='upper left')
    else:
        ax1.legend(handles1, labels1, loc='upper left')

    if save:
        plt.savefig(f'./plots/new_fed_batch_{len(train_df)}.png')
    plt.show()
 
def get_feed(feeds: pd.DataFrame, time: float) -> float:
    for _, row in feeds.iterrows():
        start_time = row["Time"]
        end_time = row["Time"] + row["Duration"]
        if start_time <= time < end_time:
            return row["F"] / 1000
    return 0

def simulate(df: pd.DataFrame, feeds: pd.DataFrame, mu_max, Ks, Yxs, plot: bool = True):
    mu_max = mu_max
    Ks = Ks
    Yxs = Yxs
    Sin = 1.43 * 200

    def system_ode(t, y):
        X, S, V = y
        mu = mu_max * S / (Ks + S)
        F = get_feed(feeds, t)
        dXdt = mu * X - F * X / V
        dSdt = - mu * X / Yxs + F * (Sin - S) / V
        dVdt = F
        return [dXdt, dSdt, dVdt]
    
    t_start, t_end = df['RTime'].min(), df['RTime'].max()
    t_span = (t_start, t_end)
    y0 = [df['Biomass'].iloc[0], df['Glucose'].iloc[0], df['V'].iloc[0]]

    t_eval = np.linspace(t_start, t_end, 10000)
    sol = solve_ivp(system_ode, t_span=t_span, \
        y0=y0, t_eval=t_eval)
    
    # Transform negative values to 0
    for i in range(sol.y.shape[0]):
        sol.y[i][sol.y[i] < 0] = 0

    if plot:
        plot_simulation(sol.t, sol.y, feeds=feeds, full_df=df)

    return sol

def get_predictions_df(net: nn.Module, df: pd.DataFrame, method: str = 'validation'):
    net_df = pd.DataFrame(columns=['RTime', 'Biomass', 'Glucose'])
    # If method == 'validation', we use the real time values
    if method == 'validation':
        t_test = df['RTime'].values
        net_df['RTime'] = t_test
    elif method == 'full':
        t_test = np.linspace(df['RTime'].min(), df['RTime'].max(), 1000)
        net_df["RTime"] = t_test

    t_test = numpy_to_tensor(t_test)
    net_df["Biomass"] = net.forward(t_test).detach().cpu().numpy()[:, 0]
    net_df["Glucose"] = net.forward(t_test).detach().cpu().numpy()[:, 1]
    net_df["V"] = net.forward(t_test).detach().cpu().numpy()[:, 2]
    net_df.loc[net_df['Glucose'] < 0, 'Glucose'] = 0
    return net_df

##############################################
FILENAME = './data/data_processed.xlsx'
EXPERIMENT = 'BR01'

_df, feeds = get_data_and_feed(FILENAME, EXPERIMENT)

# Only FED-BATCH data
_df = _df[_df['Process'] == 'FB']
feeds = feeds[feeds['Induction']==0]

print(f'Dataset shape: {_df.shape}')

# Fit polynomial to data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

STEP = 20

poly = PolynomialFeatures(degree=4)
t_train = poly.fit_transform(_df['RTime'].values.reshape(-1, 1))
y_train = _df[['Biomass', 'Glucose', 'V']].values
reg = LinearRegression().fit(t_train, y_train)

t_sim = poly.fit_transform(np.linspace(_df['RTime'].min(), _df['RTime'].max(), STEP).reshape(-1, 1))
y_sim = reg.predict(t_sim)

df_sim = pd.DataFrame(columns=['RTime', 'Biomass', 'Glucose', 'V'])
df_sim['RTime'] = np.linspace(_df['RTime'].min(), _df['RTime'].max(), STEP) 
df_sim['Biomass'] = y_sim[:, 0]
df_sim['Glucose'] = y_sim[:, 1]
df_sim['V'] = y_sim[:, 2]
df_sim.loc[df_sim['Glucose'] < 0, 'Glucose'] = 0
_df = pd.concat([_df, df_sim], axis=0)
_df = _df.sort_values(by='RTime', ascending=True).reset_index(drop=True)
_df['RTimeDiff'] = _df['RTime'].diff()
_df = _df[_df['RTimeDiff'] > 0.2]
_df.drop(columns=['RTimeDiff', 'Process', 'Protein', 'Temperature', 'Induction'], inplace=True)

def main(full_df: pd.DataFrame, i: int, num_epochs: int = 1000):
    print(f'Training with {i} data points')
    train_df = full_df.iloc[:i]
    print(f'Training shape: {train_df.shape}')
    t_start, t_end = train_df['RTime'].min(), train_df['RTime'].max()

    t_train = numpy_to_tensor(train_df['RTime'].values)
    Biomass_train = numpy_to_tensor(train_df['Biomass'].values)
    Glucose_train = numpy_to_tensor(train_df['Glucose'].values)
    V_train = numpy_to_tensor(train_df['V'].values)
    u_train = torch.cat((Biomass_train, Glucose_train, V_train), 1)

    net = PINN(input_dim=1, output_dim=3, t_start=t_start, t_end=t_end)

    repeat = True
    while repeat:
        try:
            net = train(net, t_train, u_train, full_df, feeds, num_epochs=num_epochs, verbose=True, lr=0.0001)
            repeat = False
        except ValueError:
            print('ValueError caught. Retrying...')

    net_df = get_predictions_df(net, full_df, method='full')    

    sol = simulate(full_df, feeds, net.mu_max.item(), net.K_s.item(), net.Y_xs.item(), plot=False)

    title = f"mu_max: {net.mu_max.item():4f}, Ks: {net.K_s.item():4f}, Yxs: {net.Y_xs.item():.4f}"
    plot_simulation(sol.t, sol.y, net_df=net_df, train_df=train_df, full_df=full_df, title=title, save=True) 
    return net, net_df

for i in range(12, len(_df)+1, 2):
    net, net_df = main(full_df=_df, i=i, num_epochs=5000)
