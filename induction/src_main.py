import numpy as np
from machinelearning import main
from system_ode import GetDataset
import torch
import argparse

# set seed for reproducibility
np.random.seed(0)

def run(first_point: int, last_point: int):
    last_point = last_point # Include last point
    # Get dataset
    full_df = GetDataset(alpha=[0.3, 0.5, 0.1], noise=True)
    # Specify train dataset
    train_df = full_df[first_point:last_point].copy()
    print(f'Train dataset shape: {train_df.shape}')

    net_A, u_pred_A = main(train_df, full_df, num_epochs=50000, model='A')
    net_B, u_pred_B = main(train_df, full_df, num_epochs=50000, model='B')
    net_C, u_pred_C = main(train_df, full_df, num_epochs=50000, model='C')

    # Save trained torch model
    torch.save(net_A, f'./output/net_A_{first_point}-{last_point}.pth')
    torch.save(net_B, f'./output/net_B_{first_point}-{last_point}.pth')
    torch.save(net_C, f'./output/net_C_{first_point}-{last_point}.pth')

    # Save predictions 
    u_pred_A.to_csv(f'./output/u_pred_A_{first_point}-{last_point}.csv', index=False)
    u_pred_B.to_csv(f'./output/u_pred_B_{first_point}-{last_point}.csv', index=False)
    u_pred_C.to_csv(f'./output/u_pred_C_{first_point}-{last_point}.csv', index=False)

    with open(f'./output/report_{first_point}-{last_point}.txt', 'w') as report_file:
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
    parser.add_argument('first_point', type=int, help='First point of the training dataset')
    parser.add_argument('last_point', type=int, help='Last point of the training dataset')
    args = parser.parse_args()
    run(args.first_point, args.last_point)