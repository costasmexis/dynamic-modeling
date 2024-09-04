import numpy as np
import pandas as pd
from scipy.integrate import odeint

T_START = 0
T_END = 10.5
NUM_SAMPLES = 25
S0 = 0.01
X0 = 4.16
V0 = 1.55

# parameter values
mumax = 0.84     # 1/hour
Ks = 0.2          # g/liter
Yxs = 0.5         # g/g
Sin = 1.43 * 200  # g/liter

# inlet flowrate
def Fs(t):
    if t <= 4.73:
        return 0.017
    elif t <= 7.33:
        return 0.031
    elif t <= 9.17:
        return 0.060
    elif t <= 9.78:
        return 0.031
    else:
        return 0.017

def simulate(mumax, Ks, Yxs, Sin, T_START, T_END, NUM_SAMPLES):
    
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
        dX = -Fs(t)*X/V + Rg(X,S)
        dS = Fs(t)*(Sin-S)/V - Rg(X,S)/Yxs
        dV = Fs(t)
        return [dX,dS,dV]

    IC = [X0, S0, V0]

    t = np.linspace(T_START,T_END,NUM_SAMPLES)
    sol = odeint(xdot,IC,t)

    return sol

def generate_data() -> pd.DataFrame:
    X,S,V = simulate(mumax, Ks, Yxs, Sin, T_START, T_END, NUM_SAMPLES).transpose()
    # Generate dataset 
    t = np.linspace(T_START,T_END,NUM_SAMPLES)
    full_df = pd.DataFrame({'RTime': t, 'Biomass': X, 'Glucose': S, 'V': V})
    # Add smal noise to BIomass and Glucose
    full_df['Biomass'] = full_df['Biomass'] + np.random.normal(0,0.2,NUM_SAMPLES)
    full_df['Glucose'] = full_df['Glucose'] + np.random.uniform(0,0.1,NUM_SAMPLES)
    full_df['F'] = [Fs(t) for t in full_df['RTime']]
    return full_df

def integrate_Fs(a, b):
    integral = 0
    if b <= 4.73:
        integral = 0.017 * (b - a)
    elif a <= 4.73 < b <= 7.33:
        integral = 0.017 * (4.73 - a) + 0.031 * (b - 4.73)
    elif a <= 4.73 < b <= 9.17:
        integral = 0.017 * (4.73 - a) + 0.031 * (7.33 - 4.73) + 0.060 * (b - 7.33)
    elif a <= 4.73 < b <= 9.78:
        integral = 0.017 * (4.73 - a) + 0.031 * (7.33 - 4.73) + 0.060 * (9.17 - 7.33) + 0.031 * (b - 9.17)
    elif a <= 4.73 < b:
        integral = 0.017 * (4.73 - a) + 0.031 * (7.33 - 4.73) + 0.060 * (9.17 - 7.33) + 0.031 * (9.78 - 9.17) + 0.017 * (b - 9.78)
    elif 4.73 < a <= 7.33 < b <= 9.17:
        integral = 0.031 * (7.33 - a) + 0.060 * (b - 7.33)
    elif 4.73 < a <= 7.33 < b <= 9.78:
        integral = 0.031 * (7.33 - a) + 0.060 * (9.17 - 7.33) + 0.031 * (b - 9.17)
    elif 4.73 < a <= 7.33 < b:
        integral = 0.031 * (7.33 - a) + 0.060 * (9.17 - 7.33) + 0.031 * (9.78 - 9.17) + 0.017 * (b - 9.78)
    elif 7.33 < a <= 9.17 < b <= 9.78:
        integral = 0.060 * (9.17 - a) + 0.031 * (b - 9.17)
    elif 7.33 < a <= 9.17 < b:
        integral = 0.060 * (9.17 - a) + 0.031 * (9.78 - 9.17) + 0.017 * (b - 9.78)
    elif 9.17 < a <= 9.78 < b:
        integral = 0.031 * (9.78 - a) + 0.017 * (b - 9.78)
    elif 9.17 < a <= 9.78 and b <= 9.78:
        integral = 0.031 * (b - a)
    elif 9.78 < a and 9.78 < b:
        integral = 0.017 * (b - a)
    return integral

def get_volume(t):
    if t <= 4.73:
        return integrate_Fs(0, t) + V0
    elif t <= 7.33:
        return integrate_Fs(0, 4.73) + integrate_Fs(4.73, t) + V0
    elif t <= 9.17:
        return integrate_Fs(0, 4.73) + integrate_Fs(4.73, 7.33) + integrate_Fs(7.33, t) + V0
    elif t <= 9.78:
        return integrate_Fs(0, 4.73) + integrate_Fs(4.73, 7.33) + integrate_Fs(7.33, 9.17) + integrate_Fs(9.17, t) + V0
    else:
        return integrate_Fs(0, 4.73) + integrate_Fs(4.73, 7.33) + integrate_Fs(7.33, 9.17) + integrate_Fs(9.17, 9.78) + integrate_Fs(9.78, t) + V0
    
