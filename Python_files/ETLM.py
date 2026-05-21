import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from Python_files import time_integration_functions as TST



def spin_up(x, model_parameters, model,spin_up_time):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    x_in = x
    TST.run_model([x_in, False], model, model_parameters)
    for i in range(spin_up_time):
        x_out = TST.run_model(x_in, model, model_parameters)
        # print(x_out.shape)
        x_in = x_out
    return x_out



def ETLM_generator(ensemble_size, sd, x_out, model_parameters, model):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    
    X = torch.zeros(N, ensemble_size)
    Chi = torch.zeros(N, ensemble_size)

    for i in tqdm(range(ensemble_size)):
        x_pert = [torch.randn(N,1)*sd, torch.zeros(N,1)]
        # print(x_pert)
        x_in_pert = [x_out[0] + x_pert[0], x_out[1]+x_pert[1]]
        # print(len(x_in_pert))
        x_out_pert = TST.run_model(x_in_pert, model ,model_parameters)
        x_out_unpert = TST.run_model(x_out, model, model_parameters)
        Chi_i = x_out_pert[0] - x_out_unpert[0]
        # print("x_out_pert shape:", x_out_pert.shape)
        # print("x_out_unpert shape:", x_out_unpert.shape)
        # print("Chi_i shape:", Chi_i.shape)
        X[:,i] = x_pert[0].squeeze()
        Chi[:,i] = Chi_i.squeeze()
    ETLM = Chi @ torch.transpose(X,0,1) @ torch.linalg.inv( torch.matmul(X,  torch.transpose(X,0,1)))

    
    return ETLM, Chi, X

        
        