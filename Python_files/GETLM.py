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
from Python_files import IETLM as IETLMC
torch.set_default_dtype(torch.float64)


def eigen_finder_kth(Pi, k):
    eigenvals, eigenvecs = torch.linalg.eigh(Pi)
    magnitudes = torch.abs(eigenvals)
    values, indices = torch.topk(magnitudes, k, largest=False)
    smallest_eigenvals = eigenvals[indices]
    smallest_eigenvecs = eigenvecs[:, indices]
    return smallest_eigenvecs, eigenvals


def eigen_finder_abs(Pi, threshold):
    eigenvals, eigenvecs = torch.linalg.eig(Pi)
    magnitudes = torch.abs(eigenvals)
    zero_eigenvals = []
    zero_eigenvecs = []
    for i in range(len(eigenvals)):
        if magnitudes[i] <= threshold:
            zero_eigenvals.append(eigenvals[i])
            zero_eigenvecs.append(eigenvecs[i])
    return zero_eigenvecs, zero_eigenvals


def eigen_finder_scaled(Pi, threshold):
    eigenvals, eigenvecs = torch.linalg.eig(Pi)
    magnitudes = torch.abs(eigenvals)
    total_mag = torch.sum(magnitudes)
    border = total_mag * threshold
    zero_eigenvals = []
    zero_eigenvecs = []
    for i in range(len(eigenvals)):
        if magnitudes[i] <= border:
            zero_eigenvals.append(eigenvals[i])
            zero_eigenvecs.append(eigenvecs[i])
    return zero_eigenvecs, zero_eigenvals



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


def put_in_place_row(x_i, N, x_tilde, grid_index, stencil_members):
    # print("x_i shape:", x_i.shape)
    # print("x_i:", x_i)
    # print("stencil_members:", stencil_members)

    sub_row = torch.zeros(1, N)

    for i in range(len(stencil_members)):
        local = grid_index + stencil_members[i]
        # print(i, local)

        sub_row[0, local % N] = x_i[i]

    x_tilde = torch.cat((x_tilde, sub_row), dim=0)

    return x_tilde


# We will define a function that generates the GETLM at each gridpoint
def little_GETLM(Chi_i, Xi_i, X_i, k):
    # Construct matrix
    Pi_half = torch.cat((Chi_i,-Xi_i, -X_i), dim=0)
    Pi = Pi_half @ Pi_half.T
    # print(Pi)

    # print(Pi_half.shape)

    # We now need to find the zero-eigenvalue eigenvectors
    zero_eigenvalue_eigenvecs, eigenvalues = eigen_finder_kth(Pi,k)
    return zero_eigenvalue_eigenvecs, eigenvalues



def member_splitter(zero_eigenvalue_eigenvector, stencil_members_large):
    # print(zero_eigenvalue_eigenvector.shape)
    split_1 = len(stencil_members_large[0])
    split_2 = len(stencil_members_large[1])
    split_3 = len(stencil_members_large[2])
    splits = [split_1, split_2, split_3]
    # print(splits)
    # print(zero_eigenvalue_eigenvector)
    # n_i = zero_eigenvalue_eigenvector[:split_1]
    # l_i = zero_eigenvalue_eigenvector[split_1: split_2]
    # k_i = zero_eigenvalue_eigenvector[split_2:]
    (n_i, l_i, k_i) = torch.split(zero_eigenvalue_eigenvector, splits, dim=0)

    return n_i, l_i, k_i

def GETLM_generator(Chi, Xi, X, stencil_members_large, model_parameters, ensemble_size, k, plot = True):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    N_tilde = torch.zeros(1,N)
    L_tilde = torch.zeros(1,N)
    K_tilde = torch.zeros(1,N)
    eigenvalues_4_plot = []

    future_members = stencil_members_large[0]
    current_members = stencil_members_large[1]
    past_members = stencil_members_large[2]
    for grid_index in range(N):


        X_i = stencil_selector(grid_index, X, N, ensemble_size,past_members)
        Xi_i = stencil_selector(grid_index, Xi, N, ensemble_size,current_members)
        Chi_i = stencil_selector(grid_index, Chi, N, ensemble_size,future_members)
        # print(Xi_i.shape)

        zeros, eigenvalues = little_GETLM(Chi_i, Xi_i, X_i, k)
        eigenvalues_4_plot.append(eigenvalues)

        # print(zeros.shape)
        zeros = torch.transpose(zeros, 0, 1)
        for i in range(len(zeros)):
            zero_eigenvalue_eigenvector = zeros[i]
            # print(zero_eigenvalue_eigenvector)
            # We now need to split the eigenvectors in the components for n, l and k
            n_i, l_i, k_i = member_splitter(zero_eigenvalue_eigenvector, stencil_members_large)
            # print('n_i', n_i)
            # print('l_i', l_i)
            # print('k_i', k_i)

            # We now need to put n_i, l_i and k_i into place 
            N_tilde = put_in_place_row(n_i, N, N_tilde, grid_index, future_members)
            L_tilde = put_in_place_row(l_i, N, L_tilde, grid_index, current_members)
            K_tilde = put_in_place_row(k_i, N, K_tilde, grid_index, past_members)

    N_tilde= N_tilde[1:]
    L_tilde= L_tilde[1:]
    K_tilde= K_tilde[1:]

    if plot == True:
        x_coords = []
        y_coords = []

        for grid_idx, eigenvalues in enumerate(eigenvalues_4_plot):
            # Repeat the current grid index for every eigenvalue in this cell
            x_coords.extend([grid_idx] * len(eigenvalues))
            # Add the actual eigenvalues to the y-list
            y_coords.extend(eigenvalues)

        # 3. Create the plot
        plt.figure(figsize=(12, 6))

        # Use a small marker size (s) and slight transparency (alpha) 
        # to see overlapping values or high density clearly
        plt.scatter(x_coords, y_coords, alpha=0.6, s=15, color='blue', edgecolors='none')

        # Formatting the plot
        plt.yscale('log')
        plt.title("Eigenvalue Spectrum per Grid Cell", fontsize=14, fontweight='bold')
        plt.xlabel("Grid Point Index (Columns)", fontsize=12)
        plt.ylabel("Eigenvalue Magnitude (Rows)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
        

    
    return  N_tilde, L_tilde, K_tilde



# This works for a single time step
# We are going to start by defining a function that will generate ensembles for the GETLM 
def GETLM_ensemble_generator(x_current, x_past, ensemble_size, standard_deviation,model,model_parameters, number_of_steps = 1):
    (N, dx, dt, alpha, beta, F_L96)  = model_parameters

    # The first step is to perturb the inputed run
    # We start by making 'ensemble size' copies of the input vector
    x_current_matrix = x_current.repeat(N, ensemble_size)
    x_past_matrix = x_past.repeat(N, ensemble_size)
    # print(x_current_matrix.shape)
    # print(x_past_matrix.shape)



    
    # We are now going to add random normal noise to this matrix to give us our original function 
    X = x_current_matrix + torch.randn(N, ensemble_size) * standard_deviation
    Xi = torch.zeros(N, ensemble_size)
    Chi = torch.zeros(N, ensemble_size)
    # We now iterate though each of the ensemble members to run the model
    for member in range(ensemble_size):
        # print('member:', member)
        state_pert = [X[:,member], x_past_matrix[:,member]]
        state = [x_current_matrix[:,member], x_past_matrix[:,member]]
        # print(state)
        # print(state[0])

        # state_pert = TST.run_model(state_pert, model, model_parameters)
        state_pert = TST.run_model(state_pert, model, model_parameters)

        # state = TST.run_model(state, model, model_parameters)
        
        state = TST.run_model(state, model, model_parameters)
        # print(state_pert[0].shape) 
        # print(state[0].shape)
        # print(state_pert[0].shape)
        
        # print(state_pert[0].shape)

        # print(state[0].shape)
        # print(Xi[:, member].shape)

        Xi[:, member] = (state_pert[0] - state[0]).squeeze()
        Chi[:, member] = (state_pert[1] - state[1]).squeeze()

    return Chi, Xi, X


    


# # Gemini Version
# def GETLM_ensemble_generator_mult(x_current, x_past, ensemble_size, standard_deviation, model, model_parameters, number_of_steps = 1):
#     (N, dx, dt, alpha, beta, F_L96) = model_parameters
#     Ensembles = []
    
#     # Ensure initial test perturbations are distinct 2D shapes [N, 1]
#     test = [torch.rand((N, 1)) * standard_deviation, torch.zeros((N, 1))]
#     test_values = [test]
    
#     # 1. FIX: Use .repeat(1, ensemble_size) instead of (N, ensemble_size)
#     # This correctly creates an [N, ensemble_size] matrix if x_current is [N, 1]
#     x_current_matrix = x_current.repeat(1, ensemble_size)
#     x_past_matrix = x_past.repeat(1, ensemble_size)

#     # Initialize perturbed states for the ensemble
#     X = x_current_matrix + torch.randn(N, ensemble_size) * standard_deviation
#     X_past = x_past_matrix.clone()
    
#     # 2. FIX: Track a single unperturbed control trajectory across time steps
#     control_state = [x_current.clone(), x_past.clone()]
#     for i in range(number_of_steps):
#         # Allocate fresh zero matrices per timestep to prevent reference overwriting
#         Xi = torch.zeros(N, ensemble_size)
#         Chi = torch.zeros(N, ensemble_size)
        
#         # --- PART A: EVOLVE CONTROL & TEST TRAJECTORIES (Exactly 1 Step) ---
#         # 1. Align test state with current control state
#         state_test = [control_state[0] + test_values[-1][0], control_state[1] + test_values[-1][1]]
        
#         # 2. Advance both control and test states by exactly ONE step
#         control_state = TST.run_model(control_state, model, model_parameters)
#         state_test = TST.run_model(state_test, model, model_parameters)
        
#         # 3. Compute the linear perturbation growth for this single step
#         test = [state_test[0] - control_state[0], state_test[1] - control_state[1]]
#         test_values.append(test)
        
#         # --- PART B: EVOLVE ENSEMBLE MEMBERS (Exactly 1 Step) ---
#         # Track the absolute states for the next iteration step
#         X_next = torch.zeros(N, ensemble_size)
        
#         for member in range(ensemble_size):
#             state_pert = [X[:, member].unsqueeze(1), X_past[:, member].unsqueeze(1)]

#             # Advance the absolute state of the ensemble member by ONE step
#             state_pert = TST.run_model(state_pert, model, model_parameters)

#             # Store the absolute state for the next loop iteration
#             X_next[:, member] = state_pert[0].squeeze()

#             # Measure deviation relative to the newly evolved control run
#             Xi[:, member] = state_pert[0].squeeze() - control_state[0].squeeze()
#             Chi[:, member] = state_pert[1].squeeze() - control_state[1].squeeze()
            
#         # Save explicit copies of the perturbation matrices for this timestep
#         Ensembles.append([Chi.clone(), Xi.clone(), X.clone()])
        
#         # Shift the time window forward using absolute physical states
#         X_past = X.clone()  # The old current state becomes the new past state
#         X = X_next.clone()  # The newly generated absolute state becomes the current state

        
#         for member in range(ensemble_size):
#             state_pert = [X[:, member], X_past[:, member]]

#             # Advance the absolute state
#             state_pert = TST.run_model(state_pert, model, model_parameters)
#             # state_pert = TST.run_model(state_pert, model, model_parameters)

#             # Store the absolute state for the next iteration
#             X_next[:, member] = state_pert[0]
            
#             # Measure deviation relative to control for saving
#             Xi[:, member] = state_pert[0].squeeze() - control_state[0].squeeze()
#             Chi[:, member] = state_pert[1].squeeze() - control_state[1].squeeze()
            
#         # Save explicit copies of the matrices for this timestep
#         Ensembles.append([Chi.clone(), Xi.clone(), X.clone()])
        
#         # FIX: Correctly advance the absolute states
#         # X_past = X_past_next
#         # X = X_next
#         # Ensembles.append([Chi.clone(), Xi.clone(), X.clone()])
        
#         # FIX: Correctly advance the absolute states using clones
#         X_past = X.clone()       # The old current absolute state becomes the past absolute state
#         X = X_next.clone()

#     return Ensembles, test_values