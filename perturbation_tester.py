import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm
import math
from python_files import base_functions as bf
from python_files import TLM_tests as TT
from python_files import ETLM_code as ETLMC
from python_files import LETLM_code as LETLMC
from python_files import IETLM_code as IETLMC


torch.set_default_dtype(torch.float64)


# We will start by defining our error function
def norm_error(x_g,x_pert, model, M, model_parameters):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    x_in_pert = x_g + x_pert
    x_out_pert = model(x_in_pert, model_parameters)
    x_out_unpert = model(x_g, model_parameters)
    diff = x_out_pert - x_out_unpert - torch.matmul(M, x_pert)
    return diff


def normalisor(x_g, x_pert, model_parameters, M, model):
    x_in_pert = x_g + x_pert
    x_out_pert = model(x_in_pert, model_parameters)
    x_out_unpert = model(x_g, model_parameters)
    norm = x_out_pert - x_out_unpert
    return norm

def alt_TLM_test(x_g, x_pert, model_parameters, M, model):
    (N, dx, dt, alpha, beta, F_L96) = model_parameters 
    x_in_pert = x_g + x_pert
    x_out_pert = model(x_in_pert, model_parameters)
    x_out_unpert = model(x_g, model_parameters)
    diff = sum(abs(x_out_pert - x_out_unpert - torch.matmul(M, x_pert)))
    average = sum(abs(x_pert))
    return (diff/average)/N
    

# We will now define a very quick generic TLM test that will be used for the multistep TLM test 
def generic_TLM_test(diff, norm):
    error = math.sqrt(torch.matmul(diff.T, diff)/torch.matmul(norm.T, norm))
    return error


def perturbation_test():

    results_IETLM = []
    trials = 50
    spin_up_time = 1000
    x_0 = torch.randn(N,1)
    x = IETLMC.spin_up(x_0, model_parameters, model, spin_up_time)
    x_g = x.clone()
    for i in perturbations:
        sd = 10**(-i) #(-1*perturbations[i])
        # print('sd:',sd)
        ensemble_size = 30
        model = bf.rk4_L96

        # model = test_model

        # We first need to spin up the model
        
        N_tilde, L_tilde = IETLMC.IETLM_generator(ensemble_size, sd, x, model_parameters, model)
        IETLM = torch.inverse(N_tilde) @ L_tilde

        # plt.imshow(LETLM,aspect='auto', cmap='viridis')
        # plt.colorbar(label='Value')
        # plt.title("LETLM")
        # plt.show()
        # print(LETLM[i,:])
        # we will now run the TLM test
        error = 0
        for j in range(trials):
            torch.manual_seed(j**2)
            x_pert = torch.randn(N,1) * sd

            diff = norm_error(x_g, x_pert, model, IETLM, model_parameters)
            norm = normalisor(x_g, x_pert, model_parameters, IETLM, model)

            TLM_test_result = generic_TLM_test(diff, norm)
            error += alt_TLM_test(x_g, x_pert, model_parameters, IETLM, model)
            # print(error)
            # print(error.shape)
        results_IETLM.append(error/trials)
        plt.plot(perturbations, results,label = 'ETLM' )
    plt.plot(perturbations, results_local,label = 'LETLM' )
    plt.plot(perturbations, results_IETLM,label = 'IETLM' )
    plt.plot(perturbations, results_pytorch,label = 'Pytorch Jacobian' )



    plt.xlabel('perturbation size (10^(-x))')
    plt.ylabel('error')
    plt.title('Different TLM error at different perturbation sizes')
    plt.legend()

    plt.show()
    plt.yscale('log')
    plt.plot(perturbations, results,label = 'ETLM' )
    plt.plot(perturbations, results_local,label = 'LETLM' )
    plt.plot(perturbations, results_pytorch,label = 'Pytorch Jacobian' )
    plt.plot(perturbations, results_IETLM,label = 'IETLM' )
    plt.xlabel('perturbation size (10^(-x))')
    plt.ylabel('error')
    plt.title('Different TLM error at different perturbation sizes (log scale)')
    plt.legend()

    plt.show()