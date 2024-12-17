import numpy as np
import torch
import argparse
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from machinelearning_only_product import main
from system_ode import GetDataset

# set seed for reproducibility
np.random.seed(0)

# Define parameters
T_START, T_END = 0, 12
NUM_SAMPLES = 10000
NUM_EPOCHS = 50000

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
IC = [X0, S0, P0, V0]

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

def Volume(t):
    return V0 + Fs(t) * t

def mu(S, mumax, Ks):
    return mumax * S / (Ks + S)

def Rg(X, S, mumax, Ks):
    return mu(S, mumax, Ks) * X

def a(t, alpha: list):
    '''
    - Sigmoidal Increase: a(t) = a_max * 1 / (1 + exp(-k(t-t0))
    - Exponentially Decaying: a(t) = a_0 exp(-k t)
    - Periodic fluctuation: a(t) = a_0 + a_1 sin(ωt + φ)
    - Linearly Decreasing: a(t) = a_0 - k t
    '''
    if t < 4:
        return alpha[0] * 1 / (1 + np.exp(-t))
    elif t < 10:
        return alpha[0] * np.exp(-t) 
    else:
        return alpha[1] - alpha[2] * t

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

def run(first_point: int, last_point: int):

    X, S, P, V = simulate(alpha=[0.3, 0.5, 0.1])

    data = pd.DataFrame({'t': t_sim,'X': X,'S': S,'V': V})

    # Get dataset
    full_df = GetDataset(alpha=[0.3, 0.5, 0.1], noise=True)
    # Specify train dataset
    train_df = full_df[first_point:last_point].copy()
    print(f'Train dataset shape: {train_df.shape}')

    net_A, u_pred_A, error_P_A, error_dP_A = main(train_df, full_df, data, num_epochs=NUM_EPOCHS, model='A', t_start=train_df['RTime'].values[0], t_end=train_df['RTime'].values[-1])  
    net_B, u_pred_B, error_P_B, error_dP_B = main(train_df, full_df, data, num_epochs=NUM_EPOCHS, model='B', t_start=train_df['RTime'].values[0], t_end=train_df['RTime'].values[-1])  
    net_C, u_pred_C, error_P_C, error_dP_C = main(train_df, full_df, data, num_epochs=NUM_EPOCHS, model='C', t_start=train_df['RTime'].values[0], t_end=train_df['RTime'].values[-1])  

    # Save trained torch model
    torch.save(net_A, f'./output_only_P/net_A_{first_point}-{last_point}.pth')
    torch.save(net_B, f'./output_only_P/net_B_{first_point}-{last_point}.pth')
    torch.save(net_C, f'./output_only_P/net_C_{first_point}-{last_point}.pth')

    # Save predictions 
    u_pred_A.to_csv(f'./output_only_P/u_pred_A_{first_point}-{last_point}.csv', index=False)
    u_pred_B.to_csv(f'./output_only_P/u_pred_B_{first_point}-{last_point}.csv', index=False)
    u_pred_C.to_csv(f'./output_only_P/u_pred_C_{first_point}-{last_point}.csv', index=False)

    with open(f'./output_only_P/report_{first_point}-{last_point}.txt', 'w') as report_file:
        report_file.write('Model A\n')
        report_file.write(f' * MSE P = {error_P_A:.4f}\n')
        report_file.write(f' * MSE dP = {error_dP_A:.4f}\n')
        report_file.write(f' * mu_max = {net_A.mu_max.item():.4f}\n')
        report_file.write(f' * alpha = {net_A.alpha.item():.4f}\n')

        report_file.write('Model B\n')
        report_file.write(f' * MSE P = {error_P_B:.4f}\n')
        report_file.write(f' * MSE dP = {error_dP_B:.4f}\n')
        report_file.write(f' * mu_max = {net_B.mu_max.item():.4f}\n')
        report_file.write(f' * alpha = {net_B.alpha.item():.4f}\n')

        report_file.write('Model C\n')
        report_file.write(f' * MSE P = {error_P_C:.4f}\n')
        report_file.write(f' * MSE dP = {error_dP_C:.4f}\n')
        report_file.write(f' * mu_max = {net_C.mu_max.item():.4f}\n')
        report_file.write(f' * alpha = {net_C.alpha.item():.4f}\n')
        report_file.write(f' * beta = {net_C.beta.item():.4f}\n')
    
# Use args from the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('first_point', type=int, help='First point of the training dataset')
    parser.add_argument('last_point', type=int, help='Last point of the training dataset')
    args = parser.parse_args()
    run(args.first_point, args.last_point)