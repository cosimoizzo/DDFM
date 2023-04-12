# This function simulate the DGP, estimates the models, return the performances over n simulations
import time
import pandas as pd
import numpy as np

# import os, sys, inspect
#
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)
# sys.path.insert(0, parent_dir + '/code')

from synthetic_dgp.simulate import SIMULATE
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ as DFM
from models.ddfm import DDFM

import warnings

warnings.filterwarnings("ignore")


def run_sims(this_comb: tuple, n_mc_sim: int = 100) -> None:
    """
    Method to make a call to simulate a factor model, estimate and evaluate a DFM and a DDFM.
    Args:
        this_comb: configation of the DGP for the factor model
        n_mc_sim: number of simulations (for each simulation, a new DGP is generate together with its data and on it a
            DFM and a DDFM are simulated and estimated)

    Returns:
        None, it prints the results to a csv file with the name of the configuration taken from "this_comb".
    """
    dir_name_file = str(this_comb).replace(",", "").replace("(", "").replace(")", "").replace(" ", "_") + ".csv"
    r = this_comb[0]
    n_obs = this_comb[1]
    portion_missings = this_comb[2]
    n = this_comb[3]
    non_linear = this_comb[4]
    rho = this_comb[5]  # 0.7
    alpha = this_comb[6]  # 0.2
    u = 0.1
    tau = 0  # cross correlation among idio
    if non_linear:
        poly_degree = 2
        sign_features = r
    else:
        poly_degree = 1
        sign_features = 0

    results_dfm = np.zeros((n_mc_sim, 2))
    results_ddfm = np.zeros((n_mc_sim, 2))
    for n_mc in range(n_mc_sim):
        out = run_single_sim(seed=n_mc + 1, portion_missings=portion_missings,
                                                                     n=n,
                                                                     r=r, poly_degree=poly_degree,
                                                                     sign_features=sign_features,
                                                                     n_obs=n_obs,
                                                                     rho=rho, alpha=alpha, u=u, tau=tau)
        results_dfm[n_mc, :] = out["results_dfm"]
        results_ddfm[n_mc, :] = out["results_ddfm"]
        del out
    df_results = pd.DataFrame(np.hstack((results_dfm, results_ddfm)), columns=['dfm smoothed', 'dfm filtered',
                                                                               'ddfm non-filtered', 'ddfm filtered'])
    df_results.to_csv(dir_name_file)


def run_single_sim(seed: int, n: int = 10, portion_missings: float = 0, r: int = 2, poly_degree: int = 1,
                   sign_features: int = 0, n_obs: int = 50, rho: float = 0.7, alpha: float = 0.2, u: float = 0.1,
                   tau: float = 0) -> dict:
    """
    Method to simulate a factor model, estimate and evaluate a DFM and a DDFM.
    Args:
        seed: seed setting for replicability
        n: number of observable components
        portion_missings: portion of missing data
        r: number of common factors
        poly_degree: polynomial degree (1 linear)
        sign_features: whether to add features based on sign (the first "sign_features" features, if 0 than not)
        n_obs: number of observations to simulate
        rho: parameter governing serial-correlation of the common factors
        alpha: parameter governing serial-correlation of the idiosyncratic components
        u: parameter governing the signal-to-noise ratio
        tau: parameter governing cross-correlation of the idiosyncratic components

    Returns:
        a dictionary with the evaluation scores of the DFM and the DDFM smoothed/non-filtered and filtered.
    """
    # random.seed(seed)
    # np.random.seed(seed)
    results_dfm = np.zeros(2)
    results_ddfm = np.zeros(2)

    # simulate DGP
    sim = SIMULATE(seed=seed, n=n, r=r, poly_degree=poly_degree, sign_features=sign_features, rho=rho, alpha=alpha, u=u,
                   tau=tau)
    x = sim.simulate(n_obs, portion_missings=portion_missings)
    r_hat = sim.f.shape[1]

    # estimate dfm
    dyn_fact_mdl = DFM(pd.DataFrame(x), factors=min(r_hat, x.shape[1]), factor_orders=1)
    res_dyn_fact_mdl = dyn_fact_mdl.fit(disp=10)
    results_dfm[0] = sim.evaluate(res_dyn_fact_mdl.factors.smoothed.values, f_true=sim.f)
    results_dfm[1] = sim.evaluate(res_dyn_fact_mdl.factors.filtered.values, f_true=sim.f)

    # estimate ddfm
    if poly_degree > 1:
        structure_encoder = (r_hat * 6, r_hat * 4, r_hat * 2, r_hat)
    else:
        structure_encoder = (r_hat,)
    deep_dyn_fact_mdl = DDFM(pd.DataFrame(x), seed=seed, structure_encoder=structure_encoder, factor_oder=1,
                             use_bias=False, link='relu')
    deep_dyn_fact_mdl.fit()
    results_ddfm[0] = sim.evaluate(np.mean(deep_dyn_fact_mdl.factors, axis=0), f_true=sim.f)
    results_ddfm[1] = sim.evaluate(deep_dyn_fact_mdl.factors_filtered, f_true=sim.f)
    # output dictionary
    out = {"results_dfm": results_dfm,
           "results_ddfm": results_ddfm}
    return out


def compute_elapsed_time(n_obs: list, n_vars: list, seed: int, n_sims: int = 30) -> dict:
    """
    Simulate a polynomial of order 2 DGP with 3 factors. Then estimate it with DFM and DDFM. Calculate statistics of the
    elapsed time.
    Args:
        n_obs: number of observations from a given DGP
        n_vars: number of variables from a given DGP
        seed: seed setting for replicability
        n_sims: number of simulations (each simulation correspond to a potentially different DGP)

    Returns:
        a dictionary with the following statistics about the time taken to init the object and estimate the model:
            - average_time_dfm
            - std_time_dfm
            - average_time_ddfm
            - std_time_ddfm
    """
    # # set seed
    # random.seed(seed)
    # np.random.seed(seed)
    # part fo the conf for DGP
    r = 3
    portion_missings = 0.2
    # init results collector
    average_time_dfm = np.zeros((len(n_vars), len(n_obs)))
    std_time_dfm = np.zeros_like(average_time_dfm)
    average_time_ddfm = np.zeros_like(average_time_dfm)
    std_time_ddfm = np.zeros_like(average_time_dfm)
    # loop
    for c_vars, v_vars in enumerate(n_vars):
        sim = SIMULATE(seed=seed, n=v_vars, r=r, poly_degree=2)
        for c_obs, v_obs in enumerate(n_obs):
            x = sim.simulate(v_obs, portion_missings=portion_missings)
            r_true = sim.f.shape[1]
            structure_encoder = (r_true * 6, r_true * 4, r_true * 2, r_true)
            # init arrays
            time_dfm = np.zeros(n_sims)
            time_ddfm = np.zeros(n_sims)
            for j in range(n_sims):
                # compute time DFM
                start_time = time.time()
                dyn_fact_mdl = DFM(pd.DataFrame(x), factors=min(r_true, x.shape[1]), factor_orders=1)
                dyn_fact_mdl.fit(disp=1000)
                end_time = time.time()
                time_dfm[j] = end_time - start_time
                # compute time DDFM
                start_time = time.time()
                deep_dyn_fact_mdl = DDFM(pd.DataFrame(x), seed=seed, structure_encoder=structure_encoder, factor_oder=1,
                                         use_bias=False, link='relu')
                deep_dyn_fact_mdl.fit()
                end_time = time.time()
                time_ddfm[j] = end_time - start_time
            # compute statistics
            average_time_dfm[c_vars, c_obs] = np.average(time_dfm)
            std_time_dfm[c_vars, c_obs] = np.std(time_dfm)
            average_time_ddfm[c_vars, c_obs] = np.average(time_ddfm)
            std_time_ddfm[c_vars, c_obs] = np.std(time_ddfm)
    # put results into a dictionary
    results = {"average_time_dfm": average_time_dfm,
               "std_time_dfm": std_time_dfm,
               "average_time_ddfm": average_time_ddfm,
               "std_time_ddfm": std_time_ddfm}
    return results
