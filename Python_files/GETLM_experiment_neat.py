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


def GETLM_experiment(model, model_parameters, perturbation_size, ensemble_size, num_steps, LETLM_stencil, IETLM_stencil, GETLM_stencil, plotted = False, spin_up_time = 1000):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 

    # we start by initialising the model with a vector of random normal noise
    x_in = torch.randn(N,1)
    # We then run the model so that the Lorenz 96 model has reached it's 'equilibrium'
    for i in range(spin_up_time-1):
        x_in = TST.run_model(x_in, model, model_parameters)

    x_out = TST.run_model(x_in, model, model_parameters)


    # We also need to define our test perturbation
    Test_pert_current = torch.randn(N) * perturbation_size
    Test_pert_past = torch.randn(N) * perturbation_size
    Test_pert = (Test_pert_current, Test_pert_past)


    test_final_pert, Ensemble_final, Ensembles, test_guess, unpert_run=  GETLMC.GETLM_ensemble_generator_mult(x_out, ensemble_size, perturbation_size, model, model_parameters, Test_pert, number_of_steps = num_steps)


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

        # We now make the GETLM
        N_tilde_G, L_tilde_G, K_tilde_G = GETLMC.GETLM_generator(Chi, Xi, X, GETLM_stencil, model_parameters, ensemble_size, k1, plot = plotted)
        GETLMS_1.append([N_tilde_G, L_tilde_G, K_tilde_G])

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




        # We will now compare the TLMS to the actual data to see 

    # We first set a starting point for the TLMS
    starting_time = 1

    x_values = np.arange(1, N + 1)

    IETLM_norm = []
    GETLM_norm = []


    IETLM_TT_result = []
    LETLM_TT_result = []

    GETLM_1_TT_result = []
    GETLM_2_TT_result = []
    GETLM_3_TT_result = []
    GETLM_4_TT_result = []



    # The first step is to get the test perturbation and then copy it for the IETLM, LETlM and GETLM 
    test_0 = test_final_pert[1+starting_time]
    IETLM_test = test_0
    LETLM_test = test_0



    GETLM_test_future_1 = test_0
    GETLM_test_future_2 = test_0
    GETLM_test_future_3 = test_0
    GETLM_test_future_4 = test_0


    GETLM_test_current_1 = test_final_pert[1+starting_time]
    GETLM_test_current_2 = test_final_pert[1+starting_time]
    GETLM_test_current_3 = test_final_pert[1+starting_time]
    GETLM_test_current_4 = test_final_pert[1+starting_time]

    # getlm_norm = L2_norm(GETLM_test_future, test_0)






    index = 1+starting_time


    IETLM_TT = TT.TLM_test(test_guess[index], unpert_run[index], test_0 )
    LETLM_TT = TT.TLM_test(test_guess[index], unpert_run[index], test_0 )


    GETLM_1_TT = TT.TLM_test(test_guess[index], unpert_run[index], test_0 )
    GETLM_2_TT = TT.TLM_test(test_guess[index], unpert_run[index], test_0 )
    GETLM_3_TT = TT.TLM_test(test_guess[index], unpert_run[index], test_0 )
    GETLM_4_TT = TT.TLM_test(test_guess[index], unpert_run[index], test_0 )





    IETLM_TT_result.append(IETLM_TT)
    LETLM_TT_result.append(LETLM_TT)


    GETLM_1_TT_result.append(GETLM_1_TT)
    GETLM_2_TT_result.append(GETLM_2_TT)
    GETLM_3_TT_result.append(GETLM_3_TT)
    GETLM_4_TT_result.append(GETLM_4_TT)





    # plt.plot(x_values, test_0.detach().cpu().numpy(), label='Actual Data', color='red', linestyle='--')

    # plt.plot(x_values, IETLM_test.detach().cpu().numpy(), label='Predicted (test_out)', color='green', linewidth=2)
    # plt.plot(x_values, LETLM_test.detach().cpu().numpy(), label='Predicted (test_out)', color='green', linewidth=2)



    # plt.plot(x_values, GETLM_test_future_1.detach().cpu().numpy(), label='Predicted (test_out)', color='purple', linewidth=2)
    # plt.plot(x_values, GETLM_test_future_2.detach().cpu().numpy(), label='Predicted (test_out)', color='purple', linewidth=2)
    # plt.plot(x_values, GETLM_test_future_3.detach().cpu().numpy(), label='Predicted (test_out)', color='purple', linewidth=2)
    # plt.plot(x_values, GETLM_test_future_4.detach().cpu().numpy(), label='Predicted (test_out)', color='purple', linewidth=2)

    # plt.title(f"Ensemble at zero: Prediction vs Actual")
    # plt.xlabel("N")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show() # This creates a fresh window/plot for each step in the loop

    # We will now go through the steps to see how each of the TLMS goes 
    for i in range(len(test_final_pert)-3):
        index = i+2 + starting_time
        test = test_final_pert[i+2 + starting_time]


        # Ploting the IETLM
        ietlm = IETLMS[i+starting_time]
        IETLM_test = ietlm @ IETLM_test


        ietlm_tt = TT.TLM_test(test_guess[index], unpert_run[index], IETLM_test)
        IETLM_TT_result.append(ietlm_tt)


        # Ploting the LETLM
        letlm = LETLMS[i+starting_time]
        LETLM_test = letlm @ LETLM_test


        letlm_tt = TT.TLM_test(test_guess[index], unpert_run[index], LETLM_test)
        LETLM_TT_result.append(letlm_tt)


        # IETLM_TT = torch.cat((IETLM_TT, ietlm_tt), dim = 0)



        GETLM_test_past_1 = GETLM_test_current_1
        GETLM_test_current_1 = GETLM_test_future_1

        GETLM_test_past_2 = GETLM_test_current_2
        GETLM_test_current_2 = GETLM_test_future_2

        GETLM_test_past_3 = GETLM_test_current_3
        GETLM_test_current_3 = GETLM_test_future_3

        GETLM_test_past_4 = GETLM_test_current_4
        GETLM_test_current_4 = GETLM_test_future_4

        NL = torch.pinverse(GETLMS_1[i+starting_time][0]) @ GETLMS_1[i+starting_time][1]
        NK = torch.pinverse(GETLMS_1[i+starting_time][0]) @ GETLMS_1[i+starting_time][2]
        GETLM_test_future_1 = NL @ GETLM_test_current_1 + NK @ GETLM_test_past_1

        NL = torch.pinverse(GETLMS_2[i+starting_time][0]) @ GETLMS_2[i+starting_time][1]
        NK = torch.pinverse(GETLMS_2[i+starting_time][0]) @ GETLMS_2[i+starting_time][2]
        GETLM_test_future_2 = NL @ GETLM_test_current_2 + NK @ GETLM_test_past_2
        
        NL = torch.pinverse(GETLMS_3[i+starting_time][0]) @ GETLMS_3[i+starting_time][1]
        NK = torch.pinverse(GETLMS_3[i+starting_time][0]) @ GETLMS_3[i+starting_time][2]
        GETLM_test_future_3 = NL @ GETLM_test_current_3 + NK @ GETLM_test_past_3

        NL = torch.pinverse(GETLMS_4[i+starting_time][0]) @ GETLMS_4[i+starting_time][1]
        NK = torch.pinverse(GETLMS_4[i+starting_time][0]) @ GETLMS_4[i+starting_time][2]
        GETLM_test_future_4 = NL @ GETLM_test_current_4 + NK @ GETLM_test_past_4

        getlm_tt_1 = TT.TLM_test(test_guess[index], unpert_run[index], GETLM_test_future_1)
        getlm_tt_2 = TT.TLM_test(test_guess[index], unpert_run[index], GETLM_test_future_2)
        getlm_tt_3 = TT.TLM_test(test_guess[index], unpert_run[index], GETLM_test_future_3)
        getlm_tt_4 = TT.TLM_test(test_guess[index], unpert_run[index], GETLM_test_future_4)


        # GETLM_TT = torch.cat((GETLM_TT, getlm_tt), dim = 0)
        GETLM_1_TT_result.append(getlm_tt_1)
        GETLM_2_TT_result.append(getlm_tt_2)
        GETLM_3_TT_result.append(getlm_tt_3)
        GETLM_4_TT_result.append(getlm_tt_4)




        

        # # We now need to calculate the L2 norm

        # letlm_norm = L2_norm(LETLM_test, test)
        # ietlm_norm = L2_norm(IETLM_test, test)
        # getlm_norm = L2_norm(GETLM_test_future, test)
        # print('IETLM norm:', ietlm_norm)
        # print('LETLM norm:', letlm_norm)
        # print('GETLM norm:', getlm_norm)
        # LETLM_norm.append(letlm_norm)
        # IETLM_norm.append(ietlm_norm)
        # GETLM_norm.append(getlm_norm)

        # plt.figure(figsize=(12, 4))  # width=12 inches, height=4 inches
        # # plt.ylim(-0.0015, 0.0015)


        # plt.plot(x_values, test.detach().cpu().numpy(), label='Actual Data', color='red', linestyle='--')
        # plt.plot(x_values, IETLM_test.detach().cpu().numpy(), label='Predicted (IETLM)', color='black', linewidth=1)
        # plt.plot(x_values, LETLM_test.detach().cpu().numpy(), label='Predicted (LETLM)', color='blue', linewidth=1)

        # plt.plot(x_values, GETLM_test_future_1.detach().cpu().numpy(), label='Predicted (GETLM 1 eigenvalue)', color='purple', linewidth=1)
        # plt.plot(x_values, GETLM_test_future_2.detach().cpu().numpy(), label='Predicted (GETLM 2 eigenvalues)', color='aqua', linewidth=1)
        # plt.plot(x_values, GETLM_test_future_3.detach().cpu().numpy(), label='Predicted (GETLM 3 eigenvalues)', color='orange', linewidth=1)
        # plt.plot(x_values, GETLM_test_future_4.detach().cpu().numpy(), label='Predicted (GETLM 4 eigenvalues)', color='lime', linewidth=1)



        # plt.title(f"Adams-Bashforth Perturbations after {i+1} time steps: Prediction vs Actual")

        # plt.xlabel("N")
        # plt.ylabel("Value")
        # plt.legend()

        # plt.grid(True, alpha=0.3)
        # plt.show() # This creates a fresh window/plot for each step in the loop
        # plt.clf()
        
        # print('TLM Test GETLM (1):', getlm_tt_1)
        # print('TLM Test GETLM (2):', getlm_tt_2)
        # print('TLM Test GETLM (3):', getlm_tt_3)
        # print('TLM Test GETLM (4):', getlm_tt_4)


        # print('TLM Test IETLM:', ietlm_tt)

        
        # plt.plot(x_values, IETLM_test.detach().cpu().numpy() - test.detach().cpu().numpy(), label='Predicted (IETLM)', color='green', linewidth=1)
        # plt.show()
        # plt.clf()

    return LETLM_TT_result, IETLM_TT_result, GETLM_1_TT_result, GETLM_2_TT_result, GETLM_3_TT_result, GETLM_4_TT_result



# now that we have a neat experiment set up, we can do it again, but with multiple trials
def GETLM_experiment_trials(model, model_parameters, perturbation_size, ensemble_size, num_step, num_trials, LETLM_stencil, IETLM_stencil, GETLM_stencil, plotted = False, spin_up_time = 1000):
    LETLM_TT_result_large = torch.zeros((num_trials, num_step))
    # print(LETLM_TT_result_large.shape)
    IETLM_TT_result_large = torch.zeros((num_trials, num_step))
    GETLM_1_TT_result_large =  torch.zeros((num_trials, num_step))
    GETLM_2_TT_result_large =  torch.zeros((num_trials, num_step))
    GETLM_3_TT_result_large =  torch.zeros((num_trials, num_step))
    GETLM_4_TT_result_large =  torch.zeros((num_trials, num_step))

    for i in tqdm(range(num_trials)):
        LETLM_TT_result, IETLM_TT_result, GETLM_1_TT_result, GETLM_2_TT_result, GETLM_3_TT_result, GETLM_4_TT_result = GETLM_experiment(model, model_parameters, perturbation_size, ensemble_size, num_step, LETLM_stencil, IETLM_stencil, GETLM_stencil, plotted = plotted, spin_up_time = spin_up_time)
        LETLM_TT_result_large[i,:] = torch.tensor(LETLM_TT_result)
        IETLM_TT_result_large[i,:] = torch.tensor(IETLM_TT_result)
        GETLM_1_TT_result_large[i,:] = torch.tensor(GETLM_1_TT_result)
        GETLM_2_TT_result_large[i,:] = torch.tensor(GETLM_2_TT_result)
        GETLM_3_TT_result_large[i,:] = torch.tensor(GETLM_3_TT_result)
        GETLM_4_TT_result_large[i,:] = torch.tensor(GETLM_4_TT_result)
        # print(LETLM_TT_result_large.shape)


    # print(torch.tensor(LETLM_TT_result).shape)

    # We then take the mean to get the average TLM test error at each time step

    LETLM_mean = torch.mean(torch.tensor(LETLM_TT_result_large), dim = 0)
    IETLM_mean = torch.mean(torch.tensor(IETLM_TT_result_large), dim = 0)
    GETLM_1_mean = torch.mean(torch.tensor(GETLM_1_TT_result_large), dim = 0)
    GETLM_2_mean = torch.mean(torch.tensor(GETLM_2_TT_result_large), dim = 0)
    GETLM_3_mean = torch.mean(torch.tensor(GETLM_3_TT_result_large), dim = 0)
    GETLM_4_mean = torch.mean(torch.tensor(GETLM_4_TT_result_large), dim = 0)

    return LETLM_mean, IETLM_mean, GETLM_1_mean, GETLM_2_mean, GETLM_3_mean, GETLM_4_mean