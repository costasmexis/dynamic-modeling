import sys
sys.path.append('../src')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from batch_process import BatchProcess, FedBatchProcess
from pinn import PINN, numpy_to_tensor, train
from utils import get_data

pd.options.mode.chained_assignment = None

np.set_printoptions(precision=4)


def plot_simulation(t, y, full_df, train_df, net_df, i, mu_max, Ks, Yxs):
    plt.figure(figsize=(10, 5))
    plt.rc('font', size=10)
    plt.rcParams['legend.handlelength'] = 1
    plt.plot(t, y[0], label='Biomass (ODE)', alpha=0.3)
    plt.plot(t, y[1], label='Glucose (ODE)', alpha=0.3)
    plt.scatter(full_df['RTime'], full_df['Glucose'], s=10, label='Glucose (All)', color='red', alpha=0.2)   
    plt.scatter(full_df['RTime'], full_df['Biomass'], s=10, label='Biomass (All)', color='green', alpha=0.2)
    plt.scatter(train_df['RTime'], train_df['Glucose'], s=10, label='Glucose (Train)', color='red', alpha=1)   
    plt.scatter(train_df['RTime'], train_df['Biomass'], s=10, label='Biomass (Train)', color='green', alpha=1)
    plt.scatter(net_df['RTime'], net_df['Glucose'], marker='x', label='Glucose (Predicted)', color='red', s=10, alpha=0.5)
    plt.scatter(net_df['RTime'], net_df['Biomass'], marker='x', label='Biomass (Predicted)', color='green', s=10, alpha=0.5)
    plt.legend()
    plt.title(f'mu_max={mu_max:.4f}, Ks={Ks:.4f}, Yxs={Yxs:.4f}')
    plt.savefig(f'./plots/simulation_{i}.png')
    plt.close()

def simulate(df, mu_max, Ks, Yxs):
    mu_max = mu_max
    Ks = Ks
    Yxs = Yxs
    
    def system_ode(t, y):
        X, S = y
        mu = mu_max * S / (Ks + S)
        dXdt = mu * X
        dSdt = - mu * X / Yxs
        return [dXdt, dSdt]
    
    t_eval = np.linspace(df['RTime'].min(), df['RTime'].max(), 10000)
    sol = solve_ivp(system_ode, [df['RTime'].min(), df['RTime'].max()], \
        [df['Biomass'].iloc[0], df['Glucose'].iloc[0]], t_eval=t_eval)
    return sol


def fit_polynomial(df: pd.DataFrame, column: str, degree: int = 3, num_points: int = 50):
    t_train = df['RTime'].values
    y_train = df[column].values

    poly = PolynomialFeatures(degree=degree)
    t_train = poly.fit_transform(t_train.reshape(-1, 1))
    poly.fit(t_train, y_train)
    lin_reg = LinearRegression()
    lin_reg.fit(t_train, y_train)
    
    t_simul = np.linspace(df['RTime'].min(), df['RTime'].max(), num_points)
    t_simul_poly = poly.fit_transform(t_simul.reshape(-1, 1))
    y_simul = lin_reg.predict(t_simul_poly)
    return t_simul, y_simul


def main():
    data = get_data(file_name='./Data_processed.xlsx')

    df = data.loc['BR01']
    df = df[df['Process']=='B']
    print(f'Dataset shape: {df.shape}')

    t_simul, biomass_simul = fit_polynomial(df, 'Biomass', degree=3, num_points=25)
    _, glucose_simul = fit_polynomial(df, 'Glucose', degree=3, num_points=25)

    # # Add gaussian noise
    # biomass_simul += np.random.normal(0, 0.05, biomass_simul.shape)
    # glucose_simul += np.random.normal(0, 0.05, glucose_simul.shape)

    # Concat df and simulated data
    df_simul = pd.DataFrame({'RTime': t_simul, 'Biomass': biomass_simul, 'Glucose': glucose_simul})
    # df = pd.concat([df, df_simul])
    # df.sort_values('RTime', inplace=True)
    # df = df[['RTime', 'Glucose', 'Biomass']]
    # df = df[~df['RTime'].duplicated()]
    
    # Use ONLY simulated data
    df = df_simul.copy()

    for i in range(2, len(df)+1):
        print(f'Training using {i} data points')
        
        _df = df.iloc[:i]
        t_start, t_end = _df['RTime'].min(), _df['RTime'].max()
        
        t = numpy_to_tensor(_df['RTime'].values)
        X = numpy_to_tensor(_df['Biomass'].values)
        S = numpy_to_tensor(_df['Glucose'].values)
        X_S = torch.cat((X, S), 1)
        
        # Define and Train PINN 
        net = PINN(1, 2, t_start=t_start, t_end=t_end)     
        net, total_loss, loss_data, loss_ic, loss_ode = train(net, t, X_S, df, num_epochs=10000, verbose=False)

        # Store the results
        net_df = pd.DataFrame(columns=['RTime', 'Biomass', 'Glucose'])
        t_test = df['RTime'].values
        net_df['RTime'] = t_test
        t_test = numpy_to_tensor(t_test)
        net_df['Biomass'] = net.forward(t_test).detach().cpu().numpy()[:, 0]
        net_df['Glucose'] = net.forward(t_test).detach().cpu().numpy()[:, 1]
        
        mu_max = net.mu_max.item()
        Ks = net.K_s.item()
        Yxs = net.Y_xs.item()
        
        print(f'mu_max: {mu_max:4f}, Ks: {Ks:4f}, Yxs: {Yxs:.4f}')
        
        solution = simulate(df, mu_max, Ks, Yxs)
        plot_simulation(solution.t, solution.y, df, _df, net_df, i, mu_max, Ks, Yxs)


if __name__ == '__main__':
    print(' *********************************** ')
    main()