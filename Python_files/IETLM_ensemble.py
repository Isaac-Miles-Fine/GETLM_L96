# We start by importing the required packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from Python_files import time_integration_functions as TST
# from python_files import TLM_tests as TT
from Python_files import ETLM as ETLMC
from Python_files import LETLM as LETLMC
# from Python_files import IETLM as IETLMC

# We also update the precision so that we can use small perturbations.
torch.set_default_dtype(torch.float64)

# This function selects the correct members from the ensemble for the grid point specified 
def stencil_selector(grid_index, Ensemble, N, ensemble_size,stencil_members):
    i = grid_index
    # print(i)
    # print(stencil_members)
    subset = torch.zeros(len(stencil_members), ensemble_size)
    # print(subset.shape)
    for ii in range(len(stencil_members)):
        j = grid_index + stencil_members[ii]
        # print(j)
        subset[ii, :] = Ensemble[(j%N),:]
        
    return subset


# def little_IETLM(Chi_i, Xi_i):
#     # Construct matrix
#     Pi_half = torch.cat((Chi_i,-Xi_i), dim=0)
#     Pi = Pi_half @ Pi_half.T
#     # print(Pi)

#     # print(Pi_half.shape)

#     # We now need to find the zero-eigenvalue eigenvectors
#     zero_eigenvalue_eigenvecs, eigenvalues = eigen_finder_kth(Pi,1)
#     return zero_eigenvalue_eigenvecs, eigenvalues

def little_IETLM(Xi_i, X_i):
    # Construct matrix
    # (1)
    Pi_half = torch.cat((Xi_i, -X_i), dim=0)
    Pi = Pi_half @ Pi_half.T
    # print(Pi_half.shape)

    # (2)
    # Compute smallest eigenpair
    eigenvals, eigenvecs = torch.lobpcg(Pi, k=1, largest=False)

    # Smallest eigenvector (shape: [18])
    v = eigenvecs[:, 0]


    return v, eigenvals


# This function finds the kth smallest eigenvalue-eigenvectors
def eigen_finder_kth(Pi, k):
    eigenvals, eigenvecs = torch.linalg.eigh(Pi)
    magnitudes = torch.abs(eigenvals)
    values, indices = torch.topk(magnitudes, k, largest=False)
    smallest_eigenvals = eigenvals[indices]
    smallest_eigenvecs = eigenvecs[:, indices]
    return smallest_eigenvecs, eigenvals


def put_in_place_row(x_i, N, x_tilde, grid_index, stencil_members):
    # print("x_i shape:", x_i.shape)
    # print("x_i:", x_i)
    # print("stencil_members:", stencil_members)
    for i in range(len(stencil_members)):
        local = grid_index + stencil_members[i]
        # print(i, local)
        x_tilde[grid_index, local % N] = x_i[i]

    return x_tilde


def member_splitter(zero_eigenvalue_eigenvector, stencil_members_large):
    # print(zero_eigenvalue_eigenvector.shape)
    split_1 = len(stencil_members_large[0])
    split_2 = len(stencil_members_large[1])
    splits = [split_1, split_2]
    # print(splits)
    # print(zero_eigenvalue_eigenvector)
    # n_i = zero_eigenvalue_eigenvector[:split_1]
    # l_i = zero_eigenvalue_eigenvector[split_1: split_2]
    # k_i = zero_eigenvalue_eigenvector[split_2:]
    (n_i, l_i) = torch.split(zero_eigenvalue_eigenvector, splits, dim=0)

    return n_i, l_i



# Now that we have defined our inner functions we can define our full IETLM function
def IETLM_generator(Chi, Xi, stencil_members_large, model_parameters, ensemble_size):

    # We start by unpacking our model parameters
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    
    # We now form the empty shells that will form the structure of the IETLM
    N_tilde = torch.zeros(N,N)
    L_tilde = torch.zeros(N,N)

    # We also unpack the stencil members that we want to use for the IETLM
    future_members = stencil_members_large[0]
    current_members = stencil_members_large[1]
    
    
    # We now go through each of the grid points and calculate the IETLM for each point
    for grid_index in range(N):

        # The first step is to select the subset of the ensembles that we need for each grid point
        Xi_i = stencil_selector(grid_index, Xi, N, ensemble_size,current_members)
        Chi_i = stencil_selector(grid_index, Chi, N, ensemble_size,future_members)
        # print(Xi_i.shape)

        zeros, eigenvalues = little_IETLM(Chi_i, Xi_i)

        # print(zeros.shape)
        # zeros = torch.transpose(zeros, 0, 1)
        zero_eigenvalue_eigenvector = zeros#[0]
        # print(zero_eigenvalue_eigenvector)
        # We now need to split the eigenvectors in the components for n, l and k
        n_i, l_i = member_splitter(zero_eigenvalue_eigenvector, stencil_members_large)
        # print('n_i', n_i)
        # print('l_i', l_i)
        # print('k_i', k_i)

        # We now need to put n_i, l_i and k_i into place 
        N_tilde = put_in_place_row(n_i, N, N_tilde, grid_index, future_members)
        L_tilde = put_in_place_row(l_i, N, L_tilde, grid_index, current_members)

    return  N_tilde, L_tilde