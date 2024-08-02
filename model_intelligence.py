import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.induction_pinn import PINN, numpy_to_tensor, train, init_weights
from src.utils import get_data_and_feed
from typing import Optional, Union
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
from tqdm import tqdm 

FILENAME = './data/data_processed.xlsx'
EXPERIMENT = 'BR02'
EPOCHS = 3000

def plot_feed(feeds):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(feeds['Time'], feeds['F'], width=feeds['Duration'], align='edge')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Feed (mL/h)')
    ax.set_title('Feed vs Time')
    plt.show()
    
def plot_simulation(t=None, y=None, feeds: Optional[pd.DataFrame] = None, full_df: Optional[pd.DataFrame] = None, train_df: Optional[pd.DataFrame] = None, net_df: Optional[pd.DataFrame] = None, title: Optional[str] = None):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    if t is not None and y is not None:
        ax1.plot(t, y[0], label='Biomass (ODE)', alpha=0.6)
        ax1.plot(t, y[1], label='Glucose (ODE)', alpha=0.6)
        ax2.plot(t, y[3], label='Protein (ODE)', alpha=0.6)
    
    if full_df is not None:
        ax1.scatter(full_df['RTime'], full_df['Glucose'], color='red', alpha=0.2)   
        ax1.scatter(full_df['RTime'], full_df['Biomass'], color='green', alpha=0.2)
        ax2.scatter(full_df['RTime'], full_df['Protein'], color='blue', alpha=0.2)
        
    if train_df is not None:
        ax1.scatter(train_df['RTime'], train_df['Glucose'], label='Glucose (Train)', color='red', alpha=1)   
        ax1.scatter(train_df['RTime'], train_df['Biomass'], label='Biomass (Train)', color='green', alpha=1)
        ax2.scatter(train_df['RTime'], train_df['Protein'], label='Protein (Train)', color='blue', alpha=1)
        
    if net_df is not None:
        ax1.scatter(net_df['RTime'], net_df['Glucose'], label='Glucose (Predicted)', marker='x', color='red', alpha=0.5)
        ax1.scatter(net_df['RTime'], net_df['Biomass'], label='Biomass (Predicted)', marker='x', color='green', alpha=0.5)
        ax2.scatter(net_df['RTime'], net_df['Protein'], label='Protein (Predicted)', marker='x', color='blue', alpha=0.5)
        
    ax1.legend()
    ax2.legend()
    
    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration")
    plt.title(title)
    plt.show()
 
def get_feed(feeds: pd.DataFrame, time: float) -> float:
    for _, row in feeds.iterrows():
        start_time = row["Time"]
        end_time = row["Time"] + row["Duration"]
        if start_time <= time < end_time:
            return row["F"] / 1000
    return 0

def simulate(net: nn.Module, scf: int, df: pd.DataFrame, feeds: pd.DataFrame, plot: bool = True):
    mu_max = net.mu_max.item()
    Ks = net.K_s.item()
    Yxs = net.Y_xs.item()
    c1 = net.c1.item()
    c2 = net.c2.item()
    c3 = net.c3.item()
    c4 = net.c4.item()
    
    Sin = 1.43 * 200
    
    def alpha(t):
        if scf == 1:
            return c1
        elif scf == 2:
            return c1 * (1 - np.exp(-c2 * t**2 + c3*t))
        elif scf == 3:
            return c1 * (1 - np.exp(-c2*t**2)) + c3 * (1 - np.exp(-c4*t**2)) 
        
    def system_ode(t, y):
        X, S, V, P = y
        mu = mu_max * S / (Ks + S)
        F = get_feed(feeds, t)
        dXdt = mu * X + F * X / V
        dSdt = - mu * X / Yxs + F * (Sin - S) / V
        dVdt = F
        dPdt = alpha(t) * mu * X - F * P / V
        return [dXdt, dSdt, dVdt, dPdt]
    
    t_start, t_end = df['RTime'].min(), df['RTime'].max()
    t_span = (t_start, t_end)
    y0 = [df['Biomass'].iloc[0], df['Glucose'].iloc[0], df['V'].iloc[0], df['Protein'].iloc[0]]

    t_eval = np.linspace(t_start, t_end, 10000)
    sol = solve_ivp(system_ode, t_span=t_span, \
        y0=y0, t_eval=t_eval)
    
    if plot:
        plot_simulation(sol.t, sol.y, feeds=feeds, full_df=df)

    for i in range(sol.y.shape[0]):
        sol.y[i][sol.y[i] < 0] = 0

    return sol

def get_predictions_df(net: nn.Module, df: pd.DataFrame):
    net_df = pd.DataFrame()
    t_test = df['RTime'].values
    t_test = numpy_to_tensor(t_test)
    net_df["RTime"] = df["RTime"].values
    net_df["Biomass"] = net.forward(t_test).detach().cpu().numpy()[:, 0]
    net_df["Glucose"] = net.forward(t_test).detach().cpu().numpy()[:, 1]
    net_df["V"] = net.forward(t_test).detach().cpu().numpy()[:, 2]
    net_df["Protein"] = net.forward(t_test).detach().cpu().numpy()[:, 3]
    net_df.loc[net_df['Glucose'] < 0, 'Glucose'] = 0
    return net_df

def plot_model_intelligence(sol_A: Union[solve_ivp, None] = None, 
                            sol_B: Union[solve_ivp, None] = None, 
                            sol_C: Union[solve_ivp, None] = None,
                            net_df_A: Union[pd.DataFrame, None] = None, 
                            net_df_B: Union[pd.DataFrame, None] = None, 
                            net_df_C: Union[pd.DataFrame, None] = None,
                            full_df: Union[pd.DataFrame, None] = None, 
                            train_df: Union[pd.DataFrame, None] = None): 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 12))
    
    # Plot for Model A
    # ax1.plot(sol_A.t, sol_A.y[0], label='Biomass (ODE) A', color='red', alpha=0.6)
    # ax1.plot(sol_A.t, sol_A.y[1], label='Glucose (ODE) A', color='red', linestyle='--', alpha=0.6)
    # ax2.plot(sol_A.t, sol_A.y[3], label='Protein (ODE) A', color='red', alpha=0.6)
    # ax1.plot(net_df_A['RTime'], net_df_A['Glucose'], label='Glucose (Predicted) A', marker='x', color='red', alpha=0.5)
    ax1.plot(net_df_A['RTime'], net_df_A['Biomass'], label='Biomass (Predicted) A', marker='^', color='red', alpha=0.5)
    ax2.plot(net_df_A['RTime'], net_df_A['Protein'], label='Protein (Predicted) A', marker='s', color='red', alpha=0.5)
    
    # Plot for Model B
    # ax1.plot(sol_B.t, sol_B.y[0], label='Biomass (ODE) B', color='green', alpha=0.6)
    # ax1.plot(sol_B.t, sol_B.y[1], label='Glucose (ODE) B', color='green', linestyle='--', alpha=0.6)
    # ax2.plot(sol_B.t, sol_B.y[3], label='Protein (ODE) B', color='green', alpha=0.6)
    # ax1.plot(net_df_B['RTime'], net_df_B['Glucose'], label='Glucose (Predicted) B', marker='x', color='green', alpha=0.5)
    ax1.plot(net_df_B['RTime'], net_df_B['Biomass'], label='Biomass (Predicted) B', marker='^', color='green', alpha=0.5)
    ax2.plot(net_df_B['RTime'], net_df_B['Protein'], label='Protein (Predicted) B', marker='s', color='green', alpha=0.5)
    
    # Plot for Model C
    # ax1.plot(sol_C.t, sol_C.y[0], label='Biomass (ODE) C', color='blue', alpha=0.6)
    # ax1.plot(sol_C.t, sol_C.y[1], label='Glucose (ODE) C', color='blue', linestyle='--', alpha=0.6)
    # ax2.plot(sol_C.t, sol_C.y[3], label='Protein (ODE) C', color='blue', alpha=0.6)
    # ax1.plot(net_df_C['RTime'], net_df_C['Glucose'], label='Glucose (Predicted) C', marker='x', color='blue', alpha=0.5)
    ax1.plot(net_df_C['RTime'], net_df_C['Biomass'], label='Biomass (Predicted) C', marker='^', color='blue', alpha=0.5)
    ax2.plot(net_df_C['RTime'], net_df_C['Protein'], label='Protein (Predicted) C', marker='s', color='blue', alpha=0.5)

    # Experimental Data
    # ax1.scatter(full_df['RTime'], full_df['Glucose'], label='Glucose (EXP)', color='purple', alpha=0.2)
    ax1.scatter(full_df['RTime'], full_df['Biomass'], label='Biomass (EXP)', color='brown', alpha=0.2)
    ax2.scatter(full_df['RTime'], full_df['Protein'], label='Protein (EXP)', color='black', alpha=0.2)
    
    # ax1.scatter(train_df['RTime'], train_df['Glucose'], label='Glucose (EXP) Train', color='purple', alpha=1)
    ax1.scatter(train_df['RTime'], train_df['Biomass'], label='Biomass (EXP) Train', color='brown', alpha=1)
    ax2.scatter(train_df['RTime'], train_df['Protein'], label='Protein (EXP) Train', color='black', alpha=1)
   
    ax1.legend(fontsize=8, markerscale=0.5)
    ax2.legend(fontsize=8, markerscale=0.5)
    ax1.title.set_text(f'Trained using {train_df.shape[0]} data points') 
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Concentration")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Concentration")
    plt.savefig(f'./plots/{train_df.shape[0]}_points.png')

def train_and_simulate(train_df: pd.DataFrame, full_df: pd.DataFrame, scf: int, epochs: int = 1000, plot: bool = True):
    t_start, t_end = train_df['RTime'].min(), train_df['RTime'].max()

    t_train = numpy_to_tensor(train_df['RTime'].values)
    Biomass_train = numpy_to_tensor(train_df['Biomass'].values)
    Glucose_train = numpy_to_tensor(train_df['Glucose'].values)
    V_train = numpy_to_tensor(train_df['V'].values)
    Protein_train = numpy_to_tensor(train_df['Protein'].values)
    u_train = torch.cat((Biomass_train, Glucose_train, V_train, Protein_train), 1)

    net = PINN(input_dim=1, output_dim=4, t_start=t_start, t_end=t_end)
    net.apply(init_weights)
    
    net = train(net, scf, t_train, u_train, train_df, feeds, num_epochs=epochs, \
        verbose=100)
    
    net_df = get_predictions_df(net, full_df)    

    sol = simulate(net, scf, train_df, feeds, plot=False)

    title = f"mu_max: {net.mu_max.item():4f}, Ks: {net.K_s.item():4f}, Yxs: {net.Y_xs.item():.4f}"
    
    if plot:
        plot_simulation(t=sol.t, y=sol.y, net_df=net_df, train_df=train_df, full_df=full_df, title=title)
    
    return net, net_df, sol

# Read data
full_df, feeds = get_data_and_feed(FILENAME, EXPERIMENT)

# Get FB and FBI data
full_df = full_df[full_df['Process'] == 'FBI']
feeds = feeds[feeds['Induction']==1]

print(f'Dataset shape: {full_df.shape}')

for i in range(len(full_df), 1, -1):
    print(f'Using {i} data points')
    train_df = full_df.iloc[:i].copy() 

    repeat = True
    while repeat:
        try:
            print('Model A')
            net_A, net_df_A, sol_A = train_and_simulate(train_df, full_df, scf=1, epochs=EPOCHS, plot=False)
            repeat = False
        except ValueError:
            pass
        
    repeat = True
    while repeat:
        try:
            print('Model B')
            net_B, net_df_B, sol_B = train_and_simulate(train_df, full_df, scf=2, epochs=EPOCHS, plot=False)
            repeat = False
        except ValueError:
            pass
        
    repeat = True
    while repeat:
        try:
            print('Model C')
            net_C, net_df_C, sol_C = train_and_simulate(train_df, full_df, scf=3, epochs=EPOCHS, plot=False)
            repeat = False
        except ValueError:
            pass

    with open('./plots/model_parameters.txt', 'a') as file:
        file.write(f' ********** Training using {train_df.shape[0]} data points ********** \n')
        file.write(f' *** Model A: {net_A.mu_max.item():.4f}, {net_A.K_s.item():.4f}, {net_A.Y_xs.item():.4f}\n')
        file.write(f' *** Model B: {net_B.mu_max.item():.4f}, {net_B.K_s.item():.4f}, {net_B.Y_xs.item():.4f}\n')
        file.write(f' *** Model C: {net_C.mu_max.item():.4f}, {net_C.K_s.item():.4f}, {net_C.Y_xs.item():.4f}\n')
        # Write predictions to file
        file.write('Model A\n')
        file.write(net_df_A.to_string())
        file.write('\n\n')
        file.write('Model B\n')
        file.write(net_df_B.to_string())
        file.write('\n\n')
        file.write('Model C\n')
        file.write(net_df_C.to_string())
        file.write('\n\n')

    plot_model_intelligence(sol_A=sol_A, sol_B=sol_B, sol_C=sol_C,
                            net_df_A=net_df_A, net_df_B=net_df_B, net_df_C=net_df_C,
                            full_df=full_df, train_df=train_df)
