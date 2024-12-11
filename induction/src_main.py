import os

import matplotlib.pyplot as plt
import numpy as np
from machinelearning import main, plot_net_predictions
from sklearn.metrics import mean_squared_error
from system_ode import GetDataset, PlotPredictions, PlotSolution

# set seed for reproducibility
np.random.seed(0)

# Get dataset
full_df = GetDataset(alpha=[0.3, 0.5, 0.1], noise=True)

# Specify train dataset
train_df = full_df[full_df['RTime'].between(10, 13)].copy()
print(f'Train dataset shape: {train_df.shape}')

net_A, u_pred_A = main(train_df, full_df, num_epochs=50000, model='A')
net_B, u_pred_B = main(train_df, full_df, num_epochs=50000, model='B')
net_C, u_pred_C = main(train_df, full_df, num_epochs=50000, model='C')

# Save predictions 
u_pred_A.to_csv('./output/u_pred_A.csv', index=False)
u_pred_B.to_csv('./output/u_pred_B.csv', index=False)
u_pred_C.to_csv('./output/u_pred_C.csv', index=False)

if os.path.exists('./output/report.txt'):
    os.remove('./output/report.txt')
    
with open('./output/report.txt', 'w') as report_file:
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
    