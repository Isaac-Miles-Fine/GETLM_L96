# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.integrate import solve_ivp
# import torch
# from tqdm import tqdm
# import math



# # The first method is Runge-Kutta 4
# def rhs_L96(u, model_parameters):
#     (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    
#     L96 = (torch.roll(u, -1) - torch.roll(u, 2)) * torch.roll(u, 1) - u + F_L96
#     return L96  

# def rk4_L96(x_in, model_parameters):
#     (N, dx, dt, alpha, beta, F_L96) = model_parameters 
#     y = x_in[0]
#     y0 = x_in[1]

#     k1 = rhs_L96(y, model_parameters)
#     k2 = rhs_L96(y + 0.5 * dt * k1, model_parameters)
#     k3 = rhs_L96(y + 0.5 * dt * k2, model_parameters)
#     k4 = rhs_L96(y + dt * k3, model_parameters)
#     return [y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4), y]




# # The next method is Crank-Nicolson
# def CN_L96(x_in, model_parameters, iter_count=5, tol=1e-6):
#     (N, dx, dt, alpha, beta, F_96) = model_parameters 
#     x_current = x_in[0]
#     x_past = x_in[1]
    
#     def f(x):
#         # Lorenz 96 dynamics
#         return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + F_96

#     # 1. Ensure x_in is 1D (shape: [N])
#     x_current = x_current.flatten()
    
#     explicit_part = x_current + (dt / 2.0) * f(x_current)
#     x_next = x_current.clone().requires_grad_(True)
    
#     for _ in range(iter_count):
#         # f(x_next) and explicit_part are now both 1D
#         residual = x_next - (dt / 2.0) * f(x_next) - explicit_part
        
#         if torch.norm(residual) < tol:
#             break
            
#         # 2. Jacobian will now be (N, N) because x_next is (N,)
#         jacobian = torch.autograd.functional.jacobian(
#             lambda x: x - (dt / 2.0) * f(x) - explicit_part, 
#             x_next
#         )
        
#         # 3. Solve (both jacobian and residual are now correctly shaped)
#         # We use -residual.unsqueeze(-1) if we want a column back, 
#         # but solve handles (N, N) and (N) perfectly.
#         delta = torch.linalg.solve(jacobian, -residual)
        
#         with torch.no_grad():
#             x_next += delta
            
#     # Return it in the original shape if necessary (e.g., [N, 1])
#     return [x_next.detach().view(-1, 1), x_current.detach().view(-1, 1)]


# # The final function is Leapfrog


# def LF_L96(x_in, model_parameters):
#     def robert_asselin_filter(x_prev, x_curr, x_next, gamma=0.1):
#         return x_curr + gamma * (x_prev - 2 * x_curr + x_next)
#     x_current = x_in[0]
#     x_past = x_in[1]
#     (N, dx, dt, alpha, beta, F_96) = model_parameters

#     # def f(x):
#     #     # Lorenz 96 dynamics
#     #     return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + F_96
    
#     def f(x):
#         # Standard L96: (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
#         return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + F_96
    
#     trend = f(x_current)
#     X_next = x_past + dt * trend *2
#     X_current = robert_asselin_filter(x_past, x_current, X_next, gamma=0.1)
#     return [X_next, X_current]


# #  This is a function that allows us to run any of the models with out changing the outputs
# def run_model(x_in, model, model_parameters):
#     x_current = x_in[0]
#     x_past = x_in[1]
#     if model == LF_L96 and x_past is False:
#         x_next = rk4_L96(x_current, model_parameters)
#     else:
#         x_next = model(x_in, model_parameters)
#     return x_next


import torch

# -------------------------------------------------------------------------
# Unified Lorenz 96 Right-Hand Side (Always expects 1D or handles dimensions safely)
# -------------------------------------------------------------------------
def rhs_L96(u, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    # Ensure u is flat for the rolling shift operation to work correctly
    u_flat = u.view(-1)
    L96 = (torch.roll(u_flat, -1) - torch.roll(u_flat, 2)) * torch.roll(u_flat, 1) - u_flat + F_L96
    return L96.view_as(u) # Returns in the exact shape it received

# -------------------------------------------------------------------------
# Runge-Kutta 4 Solver (Returns 1D arrays)
# -------------------------------------------------------------------------
def rk4_L96(x_in, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    y = x_in[0].view(-1)  # Force 1D
    y0 = x_in[1].view(-1) # Force 1D

    k1 = rhs_L96(y, model_parameters)
    k2 = rhs_L96(y + 0.5 * dt * k1, model_parameters)
    k3 = rhs_L96(y + 0.5 * dt * k2, model_parameters)
    k4 = rhs_L96(y + dt * k3, model_parameters)
    
    return [y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4), y]

# -------------------------------------------------------------------------
# Crank-Nicolson Solver (Fixed to return 1D arrays matching RK4/LF)
# -------------------------------------------------------------------------
def CN_L96(x_in, model_parameters, iter_count=5, tol=1e-14):
    (N, dx, dt, alpha, beta, F_96) = model_parameters 
    x_current = x_in[0].flatten()
    x_past = x_in[1].flatten()
    
    explicit_part = x_current + (dt / 2.0) * rhs_L96(x_current, model_parameters)
    x_next = x_current.clone().requires_grad_(True)
    
    for _ in range(iter_count):
        residual = x_next - (dt / 2.0) * rhs_L96(x_next, model_parameters) - explicit_part
        
        if torch.norm(residual) < tol:
            break
            
        jacobian = torch.autograd.functional.jacobian(
            lambda x: x - (dt / 2.0) * rhs_L96(x, model_parameters) - explicit_part, 
            x_next
        )
        
        delta = torch.linalg.solve(jacobian, -residual)
        
        with torch.no_grad():
            x_next += delta
            
    # FIXED: Return as flat 1D vectors [N], NOT [N, 1]
    return [x_next.detach(), x_current.detach()]

# -------------------------------------------------------------------------
# Leapfrog Solver (Returns 1D arrays)
# -------------------------------------------------------------------------
def LF_L96(x_in, model_parameters):
    def robert_asselin_filter(x_prev, x_curr, x_next, gamma=0.1):
        return x_curr + gamma * (x_prev - 2 * x_curr + x_next)
        
    x_current = x_in[0].view(-1)
    x_past = x_in[1].view(-1)
    (N, dx, dt, alpha, beta, F_96) = model_parameters

    trend = rhs_L96(x_current, model_parameters)
    X_next = x_past + dt * trend * 2
    X_current = robert_asselin_filter(x_past, x_current, X_next, gamma=0.1)
    # X_current = x_in
    
    return [X_next, X_current[0]]

# -------------------------------------------------------------------------
# Master Run Model Wrapper
# -------------------------------------------------------------------------
def run_model(x_in, model, model_parameters):
    # Ensure inputs are always decoupled vectors
    x_in = [x_in[0].view(-1), x_in[1].view(-1)]
    
    if model == LF_L96 and x_in[1] is False:
        # Fallback to RK4 if no history exists for Leapfrog
        x_next = rk4_L96(x_in, model_parameters)
    else:
        x_next = model(x_in, model_parameters)
        
    # print(x_next)
    return x_next



# def run_model(x_in, model, model_parameters):
#     # 1. Recursive flattening: Extract the two most recent states 
#     # Regardless of how deeply nested the list is.
#     def flatten_state(state):
#         if isinstance(state, list):
#             # If it's a list, the first element is always the 'newest'
#             return flatten_state(state[0])
#         return torch.as_tensor(state).view(-1)

#     # 2. Get x_current and x_past
#     x_curr = flatten_state(x_in[0])
    
#     # Handle the 'False' initialization case
#     if isinstance(x_in[1], bool) and x_in[1] is False:
#         x_past = None 
#     else:
#         x_past = flatten_state(x_in[1])

#     # 3. Model Execution
#     if model == LF_L96 and x_past is None:
#         # Spin-up: RK4 requires [current, past_dummy]
#         # Your rk4_L96 returns [new, old]
#         return rk4_L96([x_curr, x_curr], model_parameters) 
    
#     elif model == LF_L96:
#         return LF_L96([x_curr, x_past], model_parameters)
        
#     else:
#         # Other models (CN/RK4) only need the current state
#         return model([x_curr, x_past], model_parameters)

# -------------------------------------------------------------------------
# Adams-Bashforth (2nd Order) Solver
# -------------------------------------------------------------------------
def AB2_L96(x_in, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    
    # Force 1D arrays matching your other solvers
    x_current = x_in[0].view(-1)
    x_past = x_in[1].view(-1)
    
    # Calculate the trends for current and previous time steps
    trend_current = rhs_L96(x_current, model_parameters)
    trend_past = rhs_L96(x_past, model_parameters)
    
    # 2nd-order Adams-Bashforth integration
    x_next = x_current + (dt / 2.0) * (3.0 * trend_current - trend_past)
    
    return [x_next, x_current]