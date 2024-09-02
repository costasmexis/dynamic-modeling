import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import copy

T_START = 0
T_END = 10.5
NUM_EPOCHS = 30000
LEARNING_RATE = 1e-4
NUM_POINTS = 10000
NUM_COLLOCATION = 100000
PATIENCE = 1000
THRESHOLD = 1e-3
EARLY_STOPPING_EPOCH = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def numpy_to_tensor(array):
    return (
        torch.tensor(array, requires_grad=True, dtype=torch.float32)
        .to(DEVICE)
        .reshape(-1, 1)
    )

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )

# TODO: COMPLETE 

