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




# We will now define a function that selects the 9 members of each row 
def member_past_selector(grid_index,X,N,ensemble_size):
    subset = torch.zeros(13,ensemble_size)
    for i in range(13):
        j = (grid_index +i - 8)%N
        # print(grid_index, j)
        subset[i,:] = X[j,:]
    return subset

def member_current_selector(grid_index,Xi,N,ensemble_size):
    subset = torch.zeros(1,ensemble_size)
    j = grid_index
    subset[0,:] = Xi[j,:]
    # print(subset.shape)
    return subset



def little_LETLM(current_subset, past_subset):
    etlm = current_subset @ past_subset.T@(torch.inverse(past_subset@ past_subset.T))
    return etlm

# def put_in_place(M_i, N, M, i):
#     M_i = M_i.squeeze()
#     # For debugging use this test tensor
#     # M_i = torch.tensor([1,2,3,4,5,6,7,8,9])

#     M_i_reversed = torch.flip(M_i, dims=[0])

#     for k in range(9):
#         j = k - 4 + i
#         M[j % N, i] = M_i_reversed[k]

#     return M

def put_in_place(M_i, N, M, i):
    # M_i shape is (1, 9)
    M_i = M_i.squeeze() 
    
    # We are calculating the dependencies for output i
    # So we fill ROW i, across the 9 columns that contributed to it
    for k in range(13):
        input_idx = (i + k - 8) % N
        M[i, input_idx] = M_i[k] # Note: row i, column input_idx
    return M

def LETLM_generator(ensemble_size, sd, x_out, model_parameters, model):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    M = torch.zeros(N,N)
    
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

    for grid_index in range(N):
        X_i = member_past_selector(grid_index,X,N,ensemble_size)
        Chi_i = member_current_selector(grid_index,Chi,N,ensemble_size)
        M_i = little_LETLM(Chi_i, X_i)
        # print(M_i.shape)
        # print(sum(abs(Chi_i - M_i @ X_i)))
        # print(sum(abs(Chi_i[:,0] - M_i @ X_i[:,0])))
        # x_pert_test = torch.randn(N,1)*sd
        # x_pert_test_t1 = model(x_pert_test,model_parameters)


        # print(sum(abs(Chi_i[:,0] - M_i @ X_i[:,0])))
        # print(sum(abs(x_pert_test_t1 - M_i @ x_pert_test)))

        M = put_in_place(M_i, N, M, grid_index)


    
    return M

        
        