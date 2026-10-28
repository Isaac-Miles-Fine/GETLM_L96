# We start by importing all the necessary packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from Python_files import time_integration_functions as TST
from Python_files import TLM_Test as TT
from Python_files import ETLM as ETLMC
from Python_files import LETLM_ensemble as LETLMC
from Python_files import IETLM_ensemble as IETLMC
from Python_files import GETLM as GETLMC


#  We also need to set the precision to be higher as we are working with very small perturbations
torch.set_default_dtype(torch.float64)


def GETLM_experiment(model, model_parameters, perturbation_size, ensemble_size, num_steps, LETLM_stencil, IETLM_stencil, GETLM_stencil, plotted = False, spin_up_time = 1000, x_alt = False, pert_plots = False, testing_pert = False):
    # print("Inside function:", len(GETLM_stencil))
    # print("Object ID:", id(GETLM_stencil))
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    # print('Step: 3',len(GETLM_stencil))

    if x_alt == False:
        # we start by initialising the model with a vector of random normal noise
        x_in = torch.randn(N,1)
        # We then run the model so that the Lorenz 96 model has reached it's 'equilibrium'
        for i in range(spin_up_time-1):
            x_in = TST.run_model(x_in, model, model_parameters)

        x_out = TST.run_model(x_in, model, model_parameters)
    else:
        x_out = x_alt
        # print('test work!')


    # We also need to define our test perturbation
    if testing_pert == False:
        Test_pert_current = torch.randn(N) * perturbation_size
        Test_pert_past = torch.randn(N) * perturbation_size
        Test_pert = (Test_pert_current, Test_pert_past)
    else:
        Test_pert = testing_pert


    test_final_pert, Ensemble_final, Ensembles, test_guess, unpert_run=  GETLMC.GETLM_ensemble_generator_mult(x_out, ensemble_size, perturbation_size, model, model_parameters, Test_pert, number_of_steps = num_steps)

    print(len(Ensemble_final))
    # Now that we have our ensembles generated we need to generate the LETLM, IETLM and GETLM for each time step
    GETLMS_1= []
    GETLMS_2= []
    GETLMS_3= []
    GETLMS_4= []

    IETLMS = []
    LETLMS = []


    for i in range(len(Ensemble_final)-2):
        # We start by selecting the ensembles at each time level
        Chi = Ensemble_final[i+2]  # Future step matrix (N, ensemble_size)
        Xi  = Ensemble_final[i+1]  # Current step matrix (N, ensemble_size)
        X   = Ensemble_final[i]
        k1 = 1 # number of zero eigenvalue eigenvalues
        k2 = 2 # number of zero eigenvalue eigenvalues
        k3 = 3 # number of zero eigenvalue eigenvalues
        k4 = 4 # number of zero eigenvalue eigenvalues

        # print('Step 4',len(GETLM_stencil))
        # We now make the GETLM
        N_tilde_G, L_tilde_G, K_tilde_G = GETLMC.GETLM_generator(Chi, Xi, X, GETLM_stencil, model_parameters, ensemble_size, k1, plot = plotted)
        GETLMS_1.append([N_tilde_G, L_tilde_G, K_tilde_G])
        # print('hello')

        N_tilde_G, L_tilde_G, K_tilde_G = GETLMC.GETLM_generator(Chi, Xi, X, GETLM_stencil, model_parameters, ensemble_size, k2, plot = False)
        GETLMS_2.append([N_tilde_G, L_tilde_G, K_tilde_G])

        N_tilde_G, L_tilde_G, K_tilde_G = GETLMC.GETLM_generator(Chi, Xi, X, GETLM_stencil, model_parameters, ensemble_size, k3, plot = False)
        GETLMS_3.append([N_tilde_G, L_tilde_G, K_tilde_G])

        N_tilde_G, L_tilde_G, K_tilde_G = GETLMC.GETLM_generator(Chi, Xi, X, GETLM_stencil, model_parameters, ensemble_size, k4, plot = False)
        GETLMS_4.append([N_tilde_G, L_tilde_G, K_tilde_G])

        # We will now make the IETLM
        N_tilde_I, L_tilde_I = IETLMC.IETLM_generator(Chi, Xi, IETLM_stencil, model_parameters, ensemble_size)
        IETLM_temp = torch.pinverse(N_tilde_I) @ L_tilde_I
        IETLMS.append(IETLM_temp)

        M_temp = LETLMC.LETLM_generator(Chi, Xi, LETLM_stencil, model_parameters, ensemble_size)
        LETLMS.append(M_temp)

    return GETLMS_1[0], GETLMS_2[0], GETLMS_3[0], GETLMS_4[0], IETLMS[0], LETLMS[0]



# now that we have a neat experiment set up, we can do it again, but with multiple trials
def GETLM_experiment_trials(model, model_parameters, perturbation_size, ensemble_size, num_step, num_trials, LETLM_stencil, IETLM_stencil, GETLM_stencil, plotted = False, spin_up_time = 1000, x_alt = False, pert_plots = False, testing_pert = False):
    # print("Before function call:", len(GETLM_stencil))
    # print("Object ID:", id(GETLM_stencil))

    result = GENIO.GETLM_experiment(
        model,
        model_parameters,
        perturbation_size,
        ensemble_size,
        2,
        LETLM_stencil,
        IETLM_stencil,
        GETLM_stencil,
        spin_up_time=0,
        x_alt=x_out
    )


    print('2',len(GETLM_stencil))
    for i in range(num_trials):
        GETLMS_1, GETLMS_2, GETLMS_3, GETLMS_4, IETLMS, LETLMS = GETLM_experiment(model, model_parameters, perturbation_size, ensemble_size, num_step, LETLM_stencil, IETLM_stencil, GETLM_stencil, plotted = plotted, spin_up_time = spin_up_time, x_alt=x_alt, pert_plots = False, testing_pert=testing_pert)


    return  GETLMS_1, GETLMS_2, GETLMS_3, GETLMS_4, IETLMS, LETLMS