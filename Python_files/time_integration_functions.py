import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math



# The first method is Runge-Kutta 4
def rhs_L96(u, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    
    L96 = (torch.roll(u, -1) - torch.roll(u, 2)) * torch.roll(u, 1) - u + F_L96
    return L96  

def rk4_L96(x_in, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    y = x_in[0]
    y0 = x_in[1]

    k1 = rhs_L96(y, model_parameters)
    k2 = rhs_L96(y + 0.5 * dt * k1, model_parameters)
    k3 = rhs_L96(y + 0.5 * dt * k2, model_parameters)
    k4 = rhs_L96(y + dt * k3, model_parameters)
    return [y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4), y]


# The next method is Crank-Nicolson
def CN_L96(x_in, model_parameters, iter_count=5, tol=1e-6):
    (N, dx, dt, alpha, beta, F_96) = model_parameters 
    x_current = x_in[0]
    x_past = x_in[1]
    
    def f(x):
        # Lorenz 96 dynamics
        return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + F_96

    # 1. Ensure x_in is 1D (shape: [N])
    x_current = x_current.flatten()
    
    explicit_part = x_current + (dt / 2.0) * f(x_current)
    x_next = x_current.clone().requires_grad_(True)
    
    for _ in range(iter_count):
        # f(x_next) and explicit_part are now both 1D
        residual = x_next - (dt / 2.0) * f(x_next) - explicit_part
        
        if torch.norm(residual) < tol:
            break
            
        # 2. Jacobian will now be (N, N) because x_next is (N,)
        jacobian = torch.autograd.functional.jacobian(
            lambda x: x - (dt / 2.0) * f(x) - explicit_part, 
            x_next
        )
        
        # 3. Solve (both jacobian and residual are now correctly shaped)
        # We use -residual.unsqueeze(-1) if we want a column back, 
        # but solve handles (N, N) and (N) perfectly.
        delta = torch.linalg.solve(jacobian, -residual)
        
        with torch.no_grad():
            x_next += delta
            
    # Return it in the original shape if necessary (e.g., [N, 1])
    return [x_next.detach().view(-1, 1), x_current.detach().view(-1, 1)]


# The final function is Leapfrog


def LF_L96(x_in, model_parameters):
    def robert_asselin_filter(x_prev, x_curr, x_next, gamma=0.1):
        return x_curr + gamma * (x_prev - 2 * x_curr + x_next)
    x_current = x_in[0]
    x_past = x_in[1]
    (N, dx, dt, alpha, beta, F_96) = model_parameters

    # def f(x):
    #     # Lorenz 96 dynamics
    #     return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + F_96
    
    def f(x):
        # Standard L96: (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
        return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + F_96
    
    trend = f(x_current)
    X_next = x_past + dt * trend *2
    X_current = robert_asselin_filter(x_past, x_current, X_next, gamma=0.1)
    return [X_next, X_current]


#  This is a function that allows us to run any of the models with out changing the outputs
def run_model(x_in, model, model_parameters):
    x_current = x_in[0]
    x_past = x_in[1]
    if model == LF_L96 and x_past is False:
        x_next = rk4_L96(x_in, model_parameters)
    else:
        x_next = model(x_in, model_parameters)
    return x_next



