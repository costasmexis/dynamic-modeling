import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from src.utils import feeding_strategy

def simulate(mumax, Ks, Yxs, Sin, T_START, T_END, NUM_SAMPLES, IC: list):
    """ IC: initial conditions [X0, S0, V0] """

    mumax = mumax
    Ks = Ks
    Yxs = Yxs
    
    # reaction rates
    def mu(S):
        return mumax*S/(Ks + S)

    def Rg(X,S):
        return mu(S)*X

    # differential equations
    def xdot(x,t):
        X,S,V = x
        dX = -feeding_strategy(t)*X/V + Rg(X,S)
        dS = feeding_strategy(t)*(Sin-S)/V - Rg(X,S)/Yxs
        dV = feeding_strategy(t)
        return [dX,dS,dV]

    t = np.linspace(T_START,T_END,NUM_SAMPLES)
    sol = odeint(xdot,IC,t)

    return sol
