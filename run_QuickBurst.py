#import packages
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import corner
import healpy as hp
import os, glob, json, pickle
import scipy.linalg as sl
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise_extensions import blocks
from enterprise_extensions import models as ee_models
from enterprise_extensions import model_utils as ee_model_utils
from enterprise_extensions import model_orfs
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from enterprise_extensions import sampler as ee_sampler
from enterprise.signals.signal_base import LogLikelihood
from QuickBurst import enterprise_wavelets as models
from enterprise.signals.parameter import function
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import re
#style
import cProfile


from QuickBurst import QuickBurst_MCMC

with open("/home/user/path_to/.../data.pkl", 'rb') as f:
    psrs = pickle.load(f)

noise_file = "/home/user/path_to.../noise_file.json"

with open(noise_file, 'r') as h:
    noise_params = json.load(h)


#Setting dataset max time and reference time
maximum = 0
minimum = np.inf
for psr in psrs:
    if psr.toas.max() > maximum:
        maximum = psr.toas.max()
    if psr.toas.min() < minimum:
        minimum = psr.toas.min()


#Sets reference time
tref = minimum

t0_max = (maximum - minimum)/365/24/3600
print(t0_max)


#Number of shape parameter updates
N_slow=int(1e5)

#How often to update fisher matrix proposals (based on shape parameter updates)
n_fish_update = int(N_slow/2)

#Ratio of projection parameter updates per shape parameter update
projection_updates = 10000

#Number of samples to thin (based on total samples N_slow*projection_updates)
thinning = projection_updates

T_max = 4 #2
n_chain = 5 #3

#Prior bounds on shape params
tau_min = 0.2
tau_max = 5.0 #3.0
f_max = 1e-7
f_min = 3.5e-9 #1e-8

#Load in tau scan proposal files
ts_file = "/home/user/.../path_to/GW_signal_wavelet_tau_scan.pkl"
glitch_ts_file = "/home/user/.../path_to/noise_transient_tau_scan.pkl"

filepath = "/home/user/.../save_dir/"
os.makedirs(filepath, exist_ok = True)
savepath = filepath + "some_file_name" #NOTE: DO NOT ADD FILE EXTENSION

samples, acc_fraction, swap_record, rj_record, ptas, log_likelihood, betas, PT_acc = QuickBurst_MCMC.run_qb(N_slow, T_max, n_chain, psrs,
                                                                    max_n_wavelet=5,
                                                                    min_n_wavelet=0,
                                                                    n_wavelet_start=2,
                                                                    RJ_weight=2,
                                                                    glitch_RJ_weight=2,
                                                                    regular_weight=2,
                                                                    noise_jump_weight=2,
                                                                    PT_swap_weight=2,
                                                                    tau_scan_proposal_weight=2,
                                                                    glitch_tau_scan_proposal_weight=2,
                                                                    tau_scan_file=ts_file,
                                                                    glitch_tau_scan_file=glitch_ts_file,
                                                                    #gwb_log_amp_range=[-18,-15],
                                                                    rn_log_amp_range=[-18,-11],
                                                                    wavelet_log_amp_range=[-10.0,-5.0],
                                                                    per_psr_rn_log_amp_range=[-18,-11],
                                                                    #rn_params = [noise_params['gw_crn_log10_A'],noise_params['gw_crn_gamma']],
                                                                    prior_recovery=False,
                                                                    #gwb_amp_prior='log-uniform',
                                                                    rn_amp_prior='log-uniform',
                                                                    wavelet_amp_prior='uniform',
                                                                    per_psr_rn_amp_prior='log-uniform',
                                                                    #gwb_on_prior=0.975,
                                                                    max_n_glitch=3,
                                                                    #n_glitch_start='random',
                                                                    glitch_log_amp_range=[-10.0,-5.0],
                                                                    glitch_amp_prior='uniform',
                                                                    f0_max = f_max,
                                                                    f0_min = f_min,
                                                                    tau_max_in = tau_max,
                                                                    tau_min_in = tau_min,
                                                                    t0_max=t0_max,
                                                                    tref = tref,
                                                                    vary_white_noise=False,
                                                                    include_rn=False, vary_rn=False,
                                                                    include_equad=True,
                                                                    include_ecorr=False,
                                                                    include_efac=True,
                                                                    wn_backend_selection=False,
                                                                    noisedict= noise_params,
                                                                    include_per_psr_rn=False,
                                                                    vary_per_psr_rn=False,
                                                                    #resume_from=savepath,
                                                                    #per_psr_rn_start_file=RN_start_file,
                                                                    n_fish_update = n_fish_update,
                                                                    savepath=savepath, save_every_n=100,
                                                                    n_fast_to_slow=projection_updates, thin = thinning)
