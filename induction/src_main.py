import os

import matplotlib.pyplot as plt
import numpy as np
from machinelearning import main, plot_net_predictions
from sklearn.metrics import mean_squared_error
from system_ode import GetDataset, PlotPredictions, PlotSolution
import torch
import argparse

# set seed for reproducibility
np.random.seed(0)

def run(num_samples: int):
    # Get dataset
    full_df = GetDataset(alpha=[0.3, 0.5, 0.1], noise=True)
    # Specify train dataset
    train_df = full_df[:num_samples].copy()
    print(f'Train dataset shape: {train_df.shape}')

    net_A, u_pred_A = main(train_df, full_df, num_epochs=50000, model='A')
    net_B, u_pred_B = main(train_df, full_df, num_epochs=50000, model='B')
    net_C, u_pred_C = main(train_df, full_df, num_epochs=50000, model='C')

    # Save trained torch model
    torch.save(net_A, f'./output/net_A_{num_samples}.pth')
    torch.save(net_B, f'./output/net_B_{num_samples}.pth')
    torch.save(net_C, f'./output/net_C_{num_samples}.pth')

    # Save predictions 
    u_pred_A.to_csv(f'./output/u_pred_A_{num_samples}.csv', index=False)
    u_pred_B.to_csv(f'./output/u_pred_B_{num_samples}.csv', index=False)
    u_pred_C.to_csv(f'./output/u_pred_C_{num_samples}.csv', index=False)

    if os.path.exists(f'./output/report_{num_samples}.txt'):
        os.remove(f'./output/report_{num_samples}.txt')
        
    with open(f'./output/report_{num_samples}.txt', 'w') as report_file:
        report_file.write('Model A\n')
        report_file.write(f' * mu_max = {net_A.mu_max.item():.4f}\n')
        report_file.write(f' * alpha = {net_A.alpha.item():.4f}\n')

        report_file.write('Model B\n')
        report_file.write(f' * mu_max = {net_B.mu_max.item():.4f}\n')
        report_file.write(f' * alpha = {net_B.alpha.item():.4f}\n')

        report_file.write('Model C\n')
        report_file.write(f' * mu_max = {net_C.mu_max.item():.4f}\n')
        report_file.write(f' * alpha = {net_C.alpha.item():.4f}\n')
        report_file.write(f' * beta = {net_C.beta.item():.4f}\n')
    
# Use args from the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_samples', type=int, help='Number of samples for training')
    args = parser.parse_args()
    run(args.num_samples)