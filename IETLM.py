import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from python_files import base_functions as bf
from python_files import TLM_tests as TT


torch.set_default_dtype(torch.float64)



def spin_up(x, model_parameters, model,spin_up_time):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    x_in = x

    for i in range(spin_up_time):
        x_out = model(x_in, model_parameters)
        # print(x_out.shape)
        x_in = x_out
    return x_out

# # We will now define a function that selects the 9 members of each row 
# def member_past_selector(grid_index,X,N,ensemble_size):
#     subset = torch.zeros(13,ensemble_size)
#     for i in range(13):
#         j = (grid_index +i - 8)%N
#         # print(grid_index, j)
#         subset[i,:] = X[j,:]
#     return subset

# def member_current_selector(grid_index,Xi,N,ensemble_size):
#     subset = torch.zeros(1,ensemble_size)
#     j = grid_index
#     subset[0,:] = Xi[j,:]
#     # print(subset.shape)
    return subset

def stencil_selector(grid_index, Ensemble, N, ensemble_size,stencil_members):

    i = grid_index
    # print(i)
    # print(stencil_members)

    subset = torch.zeros(len(stencil_members), ensemble_size)
    for ii in range(len(stencil_members)):
        j = grid_index + stencil_members[ii]
        # print(j)
        subset[(ii%N),:] = Ensemble[(j%N),:]
        

    return subset


def put_in_place(n_i, N, N_tilde, grid_index, stencil_members):
    # print(len(stencil_members))
    # print(n_i.shape)
    for i in range(len(stencil_members)):
        # print(i)
        local = grid_index + stencil_members[i]
        N_tilde[grid_index, local%N] = n_i[i]
    return N_tilde

def little_IETLM(Xi_i, X_i, stencil_members_large):
    # Construct matrix
    # (1)
    Pi_half = torch.cat((Xi_i, -X_i), dim=0)
    Pi = Pi_half @ Pi_half.T
    # print(Pi_half.shape)

    # (2)
    # Compute smallest eigenpair
    eigenvals, eigenvecs = torch.linalg.eig(Pi)
    print(eigenvals)

    k = 1  # Number of eigenvalues you want

    # 1. Get the magnitudes (or use .real if that's your preference)
    magnitudes = torch.abs(eigenvals)

    # 2. Get the indices of the k smallest values
    # largest=False gives us the smallest values
    values, indices = torch.topk(magnitudes, k, largest=False)

    # 3. Use those indices to select from the original complex tensors
    smallest_eigenvals = eigenvals[indices]
    smallest_eigenvecs = eigenvecs[:, indices]
    # print(smallest_eigenvecs.shape)
    # Smallest eigenvector (shape: [18])
    
    # eigenvals, eigenvecs = torch.lobpcg(Pi, k=1,largest=False)


    # (3)
    # Split into two 9-vectors
    split = len(stencil_members_large[0])
    n_i = smallest_eigenvecs[:split]
    l_i = smallest_eigenvecs[split:]

    return n_i, l_i

def IETLM_generator(ensemble_size, sd, x_out, model_parameters, model, stencil_members_large):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    N_tilde = torch.zeros(N,N)
    L_tilde = torch.zeros(N,N)

    future_members = stencil_members_large[0]
    current_members = stencil_members_large[1]

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


        X_i = stencil_selector(grid_index, X, N, ensemble_size,current_members)
        Chi_i = stencil_selector(grid_index, Chi, N, ensemble_size,future_members)

        n_i, l_i = little_IETLM(Chi_i, X_i, stencil_members_large)
        # print(n_i.shape)
        # print(l_i)

        # print(M_i.shape)
        # print(sum(abs(Chi_i - M_i @ X_i)))
        # print(sum(abs(Chi_i[:,0] - M_i @ X_i[:,0])))
        # x_pert_test = torch.randn(N,1)*sd
        # x_pert_test_t1 = model(x_pert_test,model_parameters)


        # print(sum(abs(Chi_i[:,0] - M_i @ X_i[:,0])))
        # print(sum(abs(x_pert_test_t1 - M_i @ x_pert_test)))

        # l_i = [1,2,3,4,5,6,7,8]


        N_tilde = put_in_place(n_i, N, N_tilde, grid_index, future_members)
        L_tilde = put_in_place(l_i, N, L_tilde, grid_index, current_members)


    
    return  N_tilde, L_tilde

