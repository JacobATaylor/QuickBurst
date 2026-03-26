#import packages
from __future__ import division
import numpy as np
import os, glob, json, pickle
from QuickBurst import QuickBurst_MCMC as QuickBurst_MCMC
from QuickBurst.QuickBurst_utils import ChainParams

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

t0_max = (maximum - minimum)/365.25/24/3600
print(t0_max)


#Number of shape parameter updates
N_slow=int(1e5)

#How often to update fisher matrix proposals (based on shape parameter updates)
n_fish_update = int(N_slow/10)

#Ratio of projection parameter updates per shape parameter update
projection_updates = 10000

#Proposal weights (must sum to 1)
DE_prob = 0.3
fisher_prob = 0.6 
prior_draw_prob = 0.1

#Number of samples to thin (based on total samples N_slow*projection_updates)
thinning = 10000

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
config_file_name = "config"

# Chain configuration parameters
chain_params = ChainParams(
    psrs, tref=tref,
    max_n_wavelet=3,
    min_n_wavelet=0,
    max_n_glitch=3,
    # White noise
    vary_white_noise=False,
    include_equad=True,
    include_ecorr=False,
    include_efac=True,
    wn_backend_selection=False,
    noisedict=noise_params,
    # Red noise
    include_rn=True,
    vary_rn=True,
    rn_amp_prior='log-uniform',
    rn_log_amp_range=[-18, -11],
    # Per-pulsar red noise
    include_per_psr_rn=True,
    vary_per_psr_rn=True,
    per_psr_rn_amp_prior='log-uniform',
    per_psr_rn_log_amp_range=[-20, -11],
    # Wavelet priors
    wavelet_amp_prior='uniform',
    wavelet_log_amp_range=[-10.0, -5.0],
    # Glitch priors
    glitch_amp_prior='uniform',
    glitch_log_amp_range=[-10.0, -5.0],
    # Shape parameter bounds
    f0_min=f_min, f0_max=f_max,
    tau_min=tau_min, tau_max=tau_max,
    t0_max=t0_max,
    # Misc
    prior_recovery=False)

# Start sampler
samples, acc_fraction, swap_record, rj_record, ptas, log_likelihood, betas, PT_acc = QuickBurst_MCMC.run_qb(N_slow, T_max, n_chain,
                                                                    chain_params=chain_params,
                                                                    n_wavelet_start=1,
                                                                    SNR_prior=True,
                                                                    RJ_weight=2,
                                                                    glitch_RJ_weight=2,
                                                                    regular_weight=2,
                                                                    noise_jump_weight=2,
                                                                    PT_swap_weight=2,
                                                                    tau_scan_proposal_weight=2,
                                                                    glitch_tau_scan_proposal_weight=2,
                                                                    DE_prob=DE_prob,
                                                                    fisher_prob=fisher_prob,
                                                                    prior_draw_prob=prior_draw_prob,
                                                                    de_history_size=5000,
                                                                    tau_scan_file=ts_file,
                                                                    glitch_tau_scan_file=glitch_ts_file,
                                                                    noisedict=noise_params,
                                                                    T_dynamic=False,
                                                                    T_dynamic_nu=100,
                                                                    T_dynamic_t0=10000,
                                                                    # resume_from=savepath,
                                                                    n_fish_update=n_fish_update,
                                                                    savepath=savepath, save_every_n=100,
                                                                    n_fast_to_slow=projection_updates, thin=thinning, 
                                                                    write_run_parameters_to_file=True,
                                                                    run_configuration_directory=filepath,
                                                                    run_configuration_file=config_file_name)