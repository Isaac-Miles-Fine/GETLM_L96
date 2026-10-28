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


def TLM_test(test_guess, unpert_run, TLM_pert):
    alpha = test_guess - unpert_run - TLM_pert
    beta = test_guess - unpert_run
    TLM_test_result = torch.sqrt((alpha.T @ alpha)/(beta.T @ beta))

    TLM_test_result = (alpha.T @ alpha)/(beta.T @ beta)
    alpha = test_guess - unpert_run - TLM_pert
    beta = test_guess - unpert_run

    # print("alpha norm squared:", alpha @ alpha)
    # print("beta norm squared:", beta @ beta)
    # print("alpha min/max:", alpha.min().item(), alpha.max().item())
    # print("beta min/max:", beta.min().item(), beta.max().item())

    # score = (alpha @ alpha) / (beta @ beta)
    # print("TLM score:", score.item())


    return TLM_test_result