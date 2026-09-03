
# We start by importing all the required packages
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from Python_files import time_integration_functions as TST
# from python_files import TLM_tests as TT
from Python_files import ETLM as ETLMC
from Python_files import IETLM as IETLMC

# We also set the float to be higher as the perturbation sizes are smaller
torch.set_default_dtype(torch.float64)


# We also need to define all the small functions that are required to compute the LETLM


# We define a function that will allow us to select the correct members from the ensemble
def stencil_selector(grid_index, Ensemble, N, ensemble_size,stencil_members):

    i = grid_index
    # print(i)
    # print(stencil_members)

    subset = torch.zeros(len(stencil_members), ensemble_size)
    # print(subset.shape)
    for ii in range(len(stencil_members)):
        j = grid_index + stencil_members[ii]
        # print(j)
        subset[(ii%N),:] = Ensemble[(j%N),:]
        

    return subset

# We define a subject that calculates the LETLM for each row.
def little_LETLM(current_subset, past_subset):
    etlm = current_subset @ past_subset.T@(torch.inverse(past_subset@ past_subset.T))
    return etlm


# This function puts the smaller local LETLM into the larger global LETLM
def put_in_place_row(x_i, N, x_tilde, grid_index, stencil_members):
    # print("x_i shape:", x_i.shape)
    # print("x_i:", x_i)
    # print("stencil_members:", stencil_members)
    for i in range(len(stencil_members)):
        local = grid_index + stencil_members[i]
        # print(i, local)
        x_tilde[grid_index, local % N] = x_i[0,i]

    return x_tilde


# Now that we have defined all the needed functions we can define the global LETLM generator function
def LETLM_generator(Chi, Xi, stencil_members_large, model_parameters, ensemble_size):
    # We start by unpacking the required model parameters
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    
    # We then define an empty shell that will become the LETLM
    M_tilde = torch.zeros(N,N)

    # We now unpack the stencil members required
    future_members = stencil_members_large[0]
    current_members = stencil_members_large[1]

    # We now go through each gridpoint and calculate the LETLM at that point
    for grid_index in range(N):

        # we select the necessary members from the larger ensembles
        Xi_i = stencil_selector(grid_index, Xi, N, ensemble_size,current_members)
        Chi_i = stencil_selector(grid_index, Chi, N, ensemble_size,future_members)
        # print(Xi_i.shape)

        
        # We now calculate the little ensemble
        M_i = little_LETLM(Chi_i, Xi_i)
        # print(M_i.shape)
        M_tilde = put_in_place_row(M_i, N, M_tilde, grid_index, current_members)



        

    
    return  M_tilde