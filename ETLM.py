import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from python_files import base_functions as bf
from python_files import TLM_tests as TT



def spin_up(x, model_parameters, model,spin_up_time):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    x_in = x

    for i in range(spin_up_time):
        x_out = model(x_in, model_parameters)
        # print(x_out.shape)
        x_in = x_out
    return x_out

# We will now see what happens if we set everything on not on the tridiagonal to be zero
def zeronator(N, matrix):
    matrix_copy = matrix.clone().detach()
    for i in tqdm(range(N)):
        for j in range(N):
            if abs(i-j) > 5:
               matrix_copy[i,j] = 0
    return matrix_copy



def ETLM_generator(ensemble_size, sd, x_out, model_parameters, model):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    
    X = torch.zeros(N, ensemble_size)
    Chi = torch.zeros(N, ensemble_size)

    for i in tqdm(range(ensemble_size)):
        x_pert = torch.randn(N,1)*sd
        # print(x_pert)
        x_in_pert = x_out + x_pert
        x_out_pert = model(x_in_pert,model_parameters)
        x_out_unpert = model(x_out, model_parameters)
        Chi_i = x_out_pert - x_out_unpert
        # print("x_out_pert shape:", x_out_pert.shape)
        # print("x_out_unpert shape:", x_out_unpert.shape)
        # print("Chi_i shape:", Chi_i.shape)
        X[:,i] = x_pert.squeeze()
        Chi[:,i] = Chi_i.squeeze()
    ETLM = Chi @ torch.transpose(X,0,1) @ torch.linalg.inv( torch.matmul(X,  torch.transpose(X,0,1)))

    
    return ETLM, Chi, X

        
        