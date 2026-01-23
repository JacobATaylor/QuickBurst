"""
C 2024 Jacob Taylor, Rand Burnette, and Bence Becsy fast Burst MCMC

MCMC to utilize faster generic GW burst search likelihood.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import json
import h5py

import time

from numba import njit,prange
from numba.experimental import jitclass

import enterprise
import enterprise.signals.parameter as parameter
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import utils
from enterprise.signals import deterministic_signals
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise_extensions import blocks

from enterprise_extensions.frequentist import Fe_statistic

from QuickBurst import enterprise_wavelets as models
import pickle

import shutil
import os

from QuickBurst import QuickBurst_lnlike as Quickburst
from QuickBurst import QB_FastPrior


################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################
def run_qb(N_slow, T_max, n_chain, pulsars, max_n_wavelet=1, min_n_wavelet=0, n_wavelet_prior='flat', n_wavelet_start='random', RJ_weight=2, glitch_RJ_weight=2,
            regular_weight=2, noise_jump_weight=2, PT_swap_weight=2, DE_prob = 0.6, fisher_prob = 0.3, prior_draw_prob = 0.1, de_history_size = 5000, thin_de = 10000, T_ladder=None, T_dynamic=False, T_dynamic_nu=300, T_dynamic_t0=1000, PT_hist_length=100,
            tau_scan_proposal_weight=2, glitch_tau_scan_proposal_weight=2, tau_scan_file=None,
            prior_recovery=False, wavelet_amp_prior='uniform', rn_amp_prior='uniform', per_psr_rn_amp_prior='uniform',
            rn_log_amp_range=[-18,-11], per_psr_rn_log_amp_range=[-18,-11], wavelet_log_amp_range=[-18,-11],
            vary_white_noise=False, efac_start=None, include_equad=False, include_ecorr = False, include_efac = False, wn_backend_selection=False, noisedict=None,
            include_rn=False, vary_rn=False, rn_params=[-13.0, 1.0], include_per_psr_rn=False, vary_per_psr_rn=False, per_psr_rn_start_file=None, jupyter_notebook=False,
            max_n_glitch=1, glitch_amp_prior='uniform', glitch_log_amp_range=[-18, -11], equad_range = [-8.5, -5], ecorr_range = [-8.5, -5], n_glitch_prior='flat', n_glitch_start='random',
            t0_min=0.0, t0_max=10.0, tref=53000*86400, glitch_tau_scan_file=None, TF_prior_file=None, f0_min=3.5e-9, f0_max=1e-7, tau_min_in=0.2, tau_max_in=5.0,
            save_every_n=10, savepath=None, resume_from=None, start_from=None, n_status_update=100, n_fish_update=1000, n_fast_to_slow=1000, thin = 100, write_run_parameters_to_file = True, run_configuration_directory = "", run_configuration_file = ""):

    """
    Function to perform markov-chain monte carlo sampling with for generic GW burst signals. Utilizes Class Functions from QuickBurst_lnlike.

    :returns    samples[:,::n_fast_to_slow,:] (slow step samples), acc_fraction (acceptance rates),
                swap_record (PT swaps), rj_record (history of adding/removing wavelets), pta, log_likelihood[:,::n_fast_to_slow] (likelihood for slow steps),
                betas[:,::n_fast_to_slow] (temperatures for slow steps), PT_acc (acceptance rate for last PT_hist_length PT swaps)

    :param N_slow:
        Number of shape parameter updates to do in run.
    :param T_max:
        Max temperature allowed in Parallel Tempering (PT) chains.
    :param n_chain:
        Number of chains to include in a run.
    :param pulsars:
        List of pulsar objects.
    :param max_n_wavelet:
        Maximum number of GW signal wavelets to include in PTA model.
    :param min_n_wavelet:
        Minimum number of GW signal wavelets to include in PTA model.
    :param n_wavelet_prior:
        Type of signal GW signal wavelet prior to use. Either ''flat'' or you can specify list of weights of shape max_n_wavelet. [flat'] by default.
    :param n_wavelet_start:
        How many GW signal wavelets to start sampling with. ['random'] by default, which is a random draw between [0, max_n_wavelet].
    :param RJ_weight:
        Sampling weight for GW signal wavelet reversible jumps.
    :param glitch_RJ_weight:
        Sampling weight for noise transient wavelet reversible jumps.
    :param regular_weight:
        Sampling weight for regular jumps. [2] by default.
    :param noise_jump_weight:
        Sampling weight for noise jumps. [2] by default.
    :param PT_swap_weight:
        Sampling weight for PT chain swaps. [2] by default.
    :param T_ladder:
        Temperature ladder; if None, geometrically spaced ladder is made with n_chain chains reaching T_max. [None] by default.
    :param T_dynamic:
        If True, dynamically sets temperature ladder during sampling. [False] by default.
    :param T_dynamic_nu:
        ...
    :param T_dynamic_t0:
        ...
    :param PT_hist_length:
        Number of PT chain swaps used to calculate PT swap acceptance probability.
    :param tau_scan_proposal_weight:
        Sampling weight for GW signal tau scan proposal jumps. [2] by default.
    :param glitch_tau_scan_proposal_weight:
        Sampling weight for noise transient wavelet tau scan proposal jumps. [2] by default.
    :param tau_scan_file:
        Tau scan file containing tau scan proposal data for GW signal wavelet tau scan proposal jumps. If None, RJ_weight and
        tau_scan_proposal_weight must both be 0. [None] by default.
    :param prior_recovery:
        If True, return 1 for the likelihood for every step. Parameter recovery should return the specified priors. [False] by default.
    :param wavelet_amp_prior:
        GW signal wavelet prior on log10_h and log10_hcross. Choice can be ['uniform', 'log_uniform']. ['uniform'] by default.
    :param rn_amp_prior:
        CURN amplitude prior. Choices can be ['uniform', 'log_uniform']. ['uniform'] by default.
    :param per_psr_rn_amp_prior:
        Intrinsic pulsar RN amplitude prior. Choices can be ['uniform', 'log_uniform']. ['uniform'] by default.
    :param rn_log_amp_range:
        CURN amplitude prior range. [-18, -11] by default.
    :param per_psr_rn_log_amp_range:
        Intrinsic pulsar RN amplitude prior range: [-18, -11] by default.
    :param wavelet_log_amp_range:
        GW signal wavelet amplitude prior range. [-18, -11] by default.
    :param vary_white_noise:
        If True, vary intrinsic pulsar white noise. [False] by default.
    :param efac_start: NOT YET IMPLEMENTED
        If vary_white_noise = True, set initial sample for efac parameters to efac_start. [None] by default.
    :param include_equad:
        If True, will include t2equad models in PTA. If vary_white_noise = True, t2equad model parameters will be varied. [False] by default.
    :param include_ecorr:
        If True, will include ecorr models in PTA. If vary_white_noise = True, ecorr model parameters will be varied. [False] by default.
    :param include_efac:
        If True, will include efac models in PTA. If vary_white_noise = True, efac model parameters will be varied.
    :param wn_backend_selection:
        If True, use enterprise Selection based on backend. Usually use True for real data, False for simulated data. [False] by default.
    :param noisedict:
        Parameter noise dictionary for model parameters. Can be either a filepath or a dictionary. [None] by default.
    :param include_rn:
        If True, include CURN parameters in PTA model. If vary_rn = True, these parameters will be varied. [False] by default.
    :param vary_rn:
        If True, CURN parameters will be varied in PTA model. [False] by default.
    :param rn_params:
        If CURN parameters are fixed, rn_params will set the amplitude and spectral index. rn_params[0] sets
        the amplitude, while rn_params[1] sets the spectral index. [-13.0, 1] by default.
    :param include_per_psr_rn:
        If True, intrinsic pulsar red noise models will be included in PTA. [False] by default.
    :param vary_per_psr_rn:
        If True, intrinsic pulsar red noise will be varied. [False] by default.
    :param per_psr_rn_start_file: NOT YET IMPLEMENTED
        If vary_per_psr_rn = True, sets initial parameter values to values specified in file. Usually will be an empirical distribution. [None] by default.
    :param jupyter_notebook:
        ...
    :param max_n_glitch:
        Max number of noise transient wavelets allowed in PTA model. [1] by default.
    :param glitch_amp_prior:
        Prior on noise transient wavelet amplitudes. Choices can be ['uniform', 'log_uniform']. ['uniform'] by default.
    :param glitch_log_amp_range:
        Noise transient wavelet amplitude prior range. [-18, -11] by default.
    :param equad_range:
        If include_equad = True and vary_equad = True, equad_range sets the prior bounds on equad parameters. [-8.5, 5] by default.
    :param ecorr_range:
        If include_ecorr = True and vary_ecorr = True, ecorr_range sets the prior bounds on ecorr parameters. [-8.5, 5] by default.
    :param n_glitch_prior:
        Type of noise transient wavelet prior to use. Either ''flat'' or you can specify list of weights of shape max_n_wavelet. [flat'] by default.
    :param n_glitch_start:
        How many noise transient wavelets to start sampling with. ['random'] by default, which is a random draw between [0, max_n_glitch].
    :param t0_min:
        The minimum epoch time with reference to the beginning of the data set.
    :param t0_max:
        The maximum epoch time for the data set.
    :param glitch_tau_scan_file:
        Tau scan file containing tau scan proposal data for noise transient wavelet tau scan proposal jumps. If None, glitch_RJ_weight and
        glitch_tau_scan_proposal_weight must both be 0. [None] by default.
    :param TF_prior_file:
        ...
    :param f0_min:
        Lower bound on GW signal wavelet and noise transient wavelet frequency in Hz. [3.5e-9] by default.
    :param f0_max:
        Upper bound on GW signal wavelet and noise transient wavelet frequency in Hz. [1e-7] by default.
    :param tau_min_in:
        Lower bound on GW signal wavelet and noise transient wavelet width in years. [0.2] by default.
    :param tau_max_in:
        Upper bound on GW signal wavelet and noise transient wavelet width in years. [5] by default.
    :param save_every_n:
        Number of samples between saving the chain. This is multiplied by n_fast_to_slow. [10] by default.
    :param savepath:
        Path to save output to. [None] by default.
    :param resume_from:
        Resume from an existing chain. Must ensure parameters in chain match parameters in PTA model. [None] by default.
    :param start_from:
        Start from existing sample dictionary. Will check if param exists in pta.params. [None] by default.
    :param n_status_update:
        Number of N_slow samples between chain status updates. [100] by default.
    :param n_fish_update:
        Number of N_slow samples between fisher matrix updates. [1000] by default.
    :param n_fast_to_slow:
        Number of projection parameter updates for every shape parameter update. [1000] by default.
    :param thin:
        Spacing between saved samples. If 10, saves every 10th sample. [100] by default.
    :param write_run_parameters_to_file:
        Option if you want to write the run parameters to a file.
    :param run_configuration_directory:
        This is the directory you want to save the data about the run configurtion in. Make sure it has a slash at the end.
    :param run_configuration_file:
        The name of the file of that stores the run configuration. I recommend making it a similar name to the file the actual chain data is stored in. DONT ADD A FILE EXTENSION.
    """

    #scale steps to slow steps
    N = N_slow*n_fast_to_slow
    n_status_update = n_status_update*n_fast_to_slow
    n_fish_update = n_fish_update*n_fast_to_slow
    save_every_n = save_every_n*n_fast_to_slow

    #Get the names of the pulsars in order to store them
    psr_names = []
    for psr in pulsars:
        psr_names.append(psr.name)


    #If no wn or rn variance, shouldn't do any noise jumps
    if not vary_white_noise:
        if not vary_per_psr_rn:
            noise_jump_weight = 0
    if TF_prior_file is None:
        TF_prior = None
    else:
        with open(TF_prior_file, 'rb') as f:
            TF_prior = pickle.load(f)
    pta, QB_FP, QB_FPI, glitch_indx, wavelet_indx, per_puls_indx, per_puls_rn_indx, per_puls_wn_indx, rn_indx, all_noiseparam_idxs, num_per_puls_param_list = get_pta(pulsars, vary_white_noise=vary_white_noise, include_equad=include_equad,
                                                                                                    include_ecorr = include_ecorr, include_efac = include_efac,
                                                                                                    wn_backend_selection=wn_backend_selection,noisedict=noisedict, include_rn=include_rn, vary_rn=vary_rn,
                                                                                                    include_per_psr_rn=include_per_psr_rn, vary_per_psr_rn=vary_per_psr_rn,
                                                                                                    max_n_wavelet=max_n_wavelet, efac_start=efac_start, rn_amp_prior=rn_amp_prior,
                                                                                                    rn_log_amp_range=rn_log_amp_range, rn_params=rn_params, per_psr_rn_amp_prior=per_psr_rn_amp_prior,
                                                                                                    per_psr_rn_log_amp_range=per_psr_rn_log_amp_range, equad_range = equad_range,
                                                                                                    wavelet_amp_prior=wavelet_amp_prior, ecorr_range = ecorr_range,
                                                                                                    wavelet_log_amp_range=wavelet_log_amp_range, prior_recovery=prior_recovery,
                                                                                                    max_n_glitch=max_n_glitch, glitch_amp_prior=glitch_amp_prior, glitch_log_amp_range=glitch_log_amp_range,
                                                                                                    t0_min=t0_min, t0_max=t0_max, f0_min=f0_min, f0_max=f0_max, tau_min=tau_min_in, tau_max=tau_max_in,
                                                                                                    TF_prior=TF_prior, tref=tref)


    print('all noise param indexes: {}'.format(all_noiseparam_idxs))
    if n_chain < 2:
        print('Not enough chains for DE jumps. Make sure to set n_chain to 2 or more chains. Setting DE_prob = 0.')
        DE_prob = 0

    print('Number of pta params: ', len(pta.params))
    #setting up temperature ladder
    if n_chain > 1:
        if T_ladder is None:
            #using geometric spacing
            c = T_max**(1.0/(n_chain-1))
            Ts = c**np.arange(n_chain)

            print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\
     Temperature ladder is:\n".format(n_chain,c),Ts)
        else:
            Ts = np.array(T_ladder)
            n_chain = Ts.size

        print("Using {0} temperature chains with custom spacing: ".format(n_chain),Ts)
    else:
        Ts = T_max
    if T_dynamic:
        print("Dynamic temperature adjustment: ON")
    else:
        print("Dynamic temperature adjustment: OFF")

    #Create a dict of all the run parameters
    if write_run_parameters_to_file:
        run_configuration_data = {}
        run_configuration_data['N_slow'] = N_slow
        run_configuration_data['T_max'] = T_max
        run_configuration_data['n_chain'] = n_chain
        run_configuration_data['psr_names'] = psr_names
        run_configuration_data['max_n_wavelet'] = max_n_wavelet
        run_configuration_data['min_n_wavelet'] = min_n_wavelet
        run_configuration_data['n_wavelet_start'] = n_wavelet_start
        run_configuration_data['glitch_RJ_weight'] = glitch_RJ_weight
        run_configuration_data['regular_weight'] = regular_weight
        run_configuration_data['noise_jump_weight'] = noise_jump_weight
        run_configuration_data['PT_swap_weight'] = PT_swap_weight
        run_configuration_data['DE_prob'] = DE_prob
        run_configuration_data['fisher_prob'] = fisher_prob
        run_configuration_data['prior_draw_prob'] = prior_draw_prob
        run_configuration_data['de_history_size'] = de_history_size
        run_configuration_data['T_ladder'] = T_ladder
        run_configuration_data['T_dynamic'] = T_dynamic
        run_configuration_data['T_dynamic_nu'] = T_dynamic_nu
        run_configuration_data['T_dynamic_t0'] = T_dynamic_t0
        run_configuration_data['PT_hist_length'] = PT_hist_length
        run_configuration_data['tau_scan_proposal_weight'] = tau_scan_proposal_weight        
        run_configuration_data['glitch_tau_scan_proposal_weight'] = glitch_tau_scan_proposal_weight
        run_configuration_data['tau_scan_file'] = tau_scan_file
        run_configuration_data['prior_recovery'] = prior_recovery
        run_configuration_data['wavelet_amp_prior'] = wavelet_amp_prior
        run_configuration_data['rn_amp_prior'] = rn_amp_prior
        run_configuration_data['per_psr_rn_amp_prior'] = per_psr_rn_amp_prior
        run_configuration_data['rn_log_amp_range'] = rn_log_amp_range
        run_configuration_data['per_psr_rn_log_amp_range'] = per_psr_rn_log_amp_range
        run_configuration_data['wavelet_log_amp_range'] = wavelet_log_amp_range
        run_configuration_data['vary_white_noise'] = vary_white_noise
        run_configuration_data['efac_start'] = efac_start
        run_configuration_data['include_equad'] = include_equad
        run_configuration_data['include_ecorr'] = include_ecorr
        run_configuration_data['include_efac'] = include_efac
        run_configuration_data['wn_backend_selection'] = wn_backend_selection
        run_configuration_data['noisedict'] = noisedict
        run_configuration_data['include_rn'] = include_rn
        run_configuration_data['vary_rn'] = vary_rn
        run_configuration_data['per_psr_rn_start_file'] = per_psr_rn_start_file
        run_configuration_data['jupyter_notebook'] = jupyter_notebook
        run_configuration_data['rn_params'] = rn_params
        run_configuration_data['include_per_psr_rn'] = include_per_psr_rn
        run_configuration_data['vary_per_psr_rn'] = vary_per_psr_rn
        run_configuration_data['max_n_glitch'] = max_n_glitch
        run_configuration_data['glitch_amp_prior'] = glitch_amp_prior
        run_configuration_data['glitch_log_amp_range'] = glitch_log_amp_range
        run_configuration_data['equad_range'] = equad_range
        run_configuration_data['ecorr_range'] = ecorr_range
        run_configuration_data['n_glitch_prior'] = n_glitch_prior
        run_configuration_data['n_glitch_start'] = n_glitch_start
        run_configuration_data['t0_min'] = t0_min
        run_configuration_data['t0_max'] = t0_max
        run_configuration_data['tref'] = tref
        run_configuration_data['glitch_tau_scan_file'] = glitch_tau_scan_file
        run_configuration_data['TF_prior_file'] = TF_prior_file
        run_configuration_data['f0_min'] = f0_min
        run_configuration_data['f0_max'] = f0_max
        run_configuration_data['tau_min_in'] = tau_min_in
        run_configuration_data['save_every_n'] = save_every_n
        run_configuration_data['savepath'] = savepath
        run_configuration_data['resume_from'] = resume_from
        run_configuration_data['start_from'] = start_from
        run_configuration_data['n_status_update'] = n_status_update
        run_configuration_data['n_fish_update'] = n_fish_update
        run_configuration_data['n_fast_to_slow'] = n_fast_to_slow
        run_configuration_data['thin'] = thin
        with open(run_configuration_directory + run_configuration_file + ".json","w") as file:
            json.dump(run_configuration_data, file, indent=4)

    #This is a global variable which keeps track of the index in order to update the history array
    global de_arr_itr
    de_arr_itr = np.zeros(n_chain, dtype=int)

    #set up array to hold acceptance probabilities of last PT_hist_length PT swaps
    PT_hist = np.ones((n_chain-1,PT_hist_length))*np.nan #initiated with NaNs
    PT_hist_idx = np.array([0]) #index to keep track of which row to update in PT_hist

    #set up and print out prior on number of wavelets
    if max_n_wavelet!=0:
        if n_wavelet_prior=='flat':
            n_wavelet_prior = np.ones(max_n_wavelet+1)/(max_n_wavelet+1-min_n_wavelet)
            for i in range(min_n_wavelet):
                n_wavelet_prior[i] = 0.0
        else:
            n_wavelet_prior = np.array(n_wavelet_prior)
            n_wavelet_norm = np.sum(n_wavelet_prior)
            n_wavelet_prior *= 1.0/n_wavelet_norm
        print("Prior on number of wavelets: ", n_wavelet_prior)

    #set up and print out prior on number of glitches
    if max_n_glitch!=0:
        if n_glitch_prior=='flat':
            n_glitch_prior = np.ones(max_n_glitch+1)/(max_n_glitch+1)
        else:
            n_glitch_prior = np.array(n_glitch_prior)
            n_glitch_norm = np.sum(n_glitch_prior)
            n_glitch_prior *= 1.0/n_glitch_norm
        print("Prior on number of glitches: ", n_glitch_prior)

    #setting up array for the samples
    num_params = max_n_wavelet*10+max_n_glitch*6
    num_params += 2 #for keepeng a record of number of wavelets and glitches

    num_per_psr_params = 0
    num_noise_params = 0
    if vary_rn:
        num_noise_params += 2
    num_per_psr_params += sum(num_per_puls_param_list)
    num_noise_params += sum(num_per_puls_param_list)

    num_params += num_noise_params
    print('-'*5)
    print(num_params)
    print(num_noise_params)
    print(num_per_psr_params)
    print('-'*5)


    if resume_from is not None:
        print("Resuming from file: " + resume_from)
        resume_from += '.h5df'
        with h5py.File(resume_from, 'r+') as f:
            samples_resume = f['samples_cold'][()]
            print('samples_resume: ', samples_resume.shape[1])
            log_likelihood_resume = f['log_likelihood'][()]
                #If resuming a likelihood comparison run
            acc_frac_resume = f['acc_fraction'][()]
            param_names_resume = list(par.decode('utf-8') for par in f['par_names'][()])
            #param_names_resume = f['param_names'][()]
            swap_record_resume = f['swap_record'][()]
            print('resume swap record shape: ', swap_record_resume.shape)
            betas_resume = f['betas'][()]
            PT_acc_resume = f['PT_acc'][()]

        #Print for how many samples loading in.
        N_resume = samples_resume.shape[1]
        print("# of samples sucessfully read in: " + str(N_resume))

        samples = np.zeros((n_chain, save_every_n+1, num_params))
        samples[:,0,:] = np.copy(samples_resume[:, -1, :])

        swap_record = np.zeros((save_every_n+1, 1))
        swap_record[0] = swap_record_resume[-1]
        print('swap record length: ', swap_record.shape)

        log_likelihood = np.zeros((n_chain,save_every_n+1))
        log_likelihood[:,0] = np.copy(log_likelihood_resume[:, -1])

        print('Saving every {0} samples, total samples: {1} '.format(save_every_n, N+N_resume), '\n')
        print('Ending total saved samples: {}'.format(int(N/thin)+N_resume), '\n')

        betas = np.ones((n_chain, save_every_n+1))
        betas[:,0] = np.copy(betas_resume[:, -1])

        PT_acc = np.zeros((n_chain-1,save_every_n+1))
        PT_acc[:,0] = np.copy(PT_acc_resume[:, -1])

        QB_logl = []
        QB_Info = []
        for j in range(n_chain):
            n_wavelet = int(samples[j,0,0])# get_n_wavelet(samples, j, 0)
            n_glitch = int(samples[j,0,1])# get_n_glitch(samples, j, 0)
            first_sample = samples[j,0,2:]

            QB_logl.append(Quickburst.QuickBurst(pta = pta, psrs = pulsars, params = dict(zip(pta.param_names, first_sample)), Npsr = len(pulsars), tref=tref, Nglitch = n_glitch, Nwavelet = n_wavelet, Nglitch_max = max_n_glitch ,Nwavelet_max = max_n_wavelet, rn_vary = vary_rn, wn_vary = vary_white_noise, prior_recovery = prior_recovery))
            QB_Info.append(Quickburst.QuickBurst_info(Npsr=len(pulsars),pos = QB_logl[j].pos, resres_logdet = QB_logl[j].resres_logdet, Nglitch = n_glitch ,Nwavelet = n_wavelet, wavelet_prm = QB_logl[j].wavelet_prm, glitch_prm = QB_logl[j].glitch_prm, sigmas = QB_logl[j].sigmas, MMs = QB_logl[j].MMs, NN = QB_logl[j].NN, prior_recovery = prior_recovery, glitch_indx = QB_logl[j].glitch_indx, wavelet_indx = QB_logl[j].wavelet_indx, glitch_pulsars = QB_logl[j].glitch_pulsars))
    else:
        print('Saving every {0} samples, total samples: {1} '.format(save_every_n, N), '\n')
        print('Ending total saved samples: {}'.format(int(N/thin)), '\n')
        samples = np.zeros((n_chain, save_every_n+1, num_params))

        #set up log_likelihood array
        log_likelihood = np.zeros((n_chain,save_every_n+1))
        QB_logl = []
        QB_Info = []

        #set up betas array with PT inverse temperatures
        betas = np.ones((n_chain,save_every_n+1))
        #set first row with initial betas
        betas[:,0] = 1/Ts
        print("Initial beta (1/T) ladder is:\n",betas[:,0])

        #set up array holding PT acceptance rate for each iteration
        PT_acc = np.zeros((n_chain-1,save_every_n+1))

        #filling first sample at all temperatures with last sample of previous run's zero temperature chain (thus it works if n_chain is different)
        if start_from is not None:
            #set starting point from param dictionary
            samples_start = start_from
            for j in range(n_chain):
                for k,v in samples_start:
                    if k in pta.params:
                        #TODO: implement accounting for number of glitches/wavelets offset into starting sample
                        samples[j,0,2+k] = np.copy(v)
        #filling first sample with random draw
        else:
            for j in range(n_chain):
                #set up n_wavelet
                if n_wavelet_start == 'random':
                    n_wavelet = np.random.choice( np.arange(min_n_wavelet,max_n_wavelet+1) )
                else:
                    n_wavelet = n_wavelet_start
                #set up n_glitch
                if n_glitch_start == 'random':
                    n_glitch = np.random.choice(max_n_glitch+1)
                else:
                    n_glitch = n_glitch_start

                samples[j,0,0] = n_wavelet
                samples[j,0,1] = n_glitch
                QB_FPI.n_wavelet = n_wavelet #reset prior values after getting starting point
                QB_FPI.n_glitch = n_glitch
                samples[j,0,2:] =  np.hstack([p.sample() for p in pta.params])

                #Setting starting values based on M2A or noise run chain
                if noisedict is not None:
                    #load in params from dictionary
                    for idx, param in enumerate(pta.param_names):
                        if param in noisedict.keys():
                            samples[j, 0, 2+idx] = noisedict[param]

                #set all wavelet gw sources to same sky location
                if n_wavelet!=0:
                    for windx in range(n_wavelet):
                        samples[j,0,2+int(wavelet_indx[windx,0])] = samples[j,0,2+int(wavelet_indx[0,0])]
                        samples[j,0,2+int(wavelet_indx[windx,1])] = samples[j,0,2+int(wavelet_indx[0,1])]
                        samples[j,0,2+int(wavelet_indx[windx,2])] = samples[j,0,2+int(wavelet_indx[0,2])]

                ''' functionality to add
                if vary_white_noise and not vary_per_psr_rn:
                    if efac_start is not None:
                        for k in range(len(pulsars)):
                            samples[j,0,2:+wn_indx[k,0]] = 1*efac_start

                elif vary_per_psr_rn and not vary_white_noise:
                    if per_psr_rn_start_file is not None:
                        RN_noise_data = np.load(per_psr_rn_start_file)
                        samples[j,0,2:+wn_indx[k,0]] = RN_noise_data['RN_start']
                '''

        #printing info about initial parameters
        for j in range(n_chain):
            n_wavelet = int(samples[j,0,0]) #get_n_wavelet(samples, j, 0)
            n_glitch = int(samples[j,0,1]) #get_n_glitch(samples, j, 0)
            first_sample = np.copy(samples[j,0,2:])

            print('param list: ',pta.param_names)
            print('first_sample: ', first_sample)
            #Generate first sample param dictionary
            sample_dict = {}
            for i in range(len(first_sample)):
                sample_dict[pta.param_names[i]] = first_sample[i]
            rn_check = False

            if vary_per_psr_rn or vary_rn:
                rn_check = True
            print('QB logl object creation')
            QB_logl.append(Quickburst.QuickBurst(pta = pta, psrs = pulsars, params = sample_dict, Npsr = len(pulsars), tref=tref, Nglitch = n_glitch, Nwavelet = n_wavelet, Nglitch_max = max_n_glitch ,Nwavelet_max = max_n_wavelet, rn_vary = rn_check, wn_vary = vary_white_noise, prior_recovery=prior_recovery))
            QB_Info.append(Quickburst.QuickBurst_info(Npsr=len(pulsars),pos = QB_logl[j].pos, resres_logdet = QB_logl[j].resres_logdet, Nglitch = n_glitch,
                                                      Nwavelet = n_wavelet, wavelet_prm = QB_logl[j].wavelet_prm, glitch_prm = QB_logl[j].glitch_prm, sigmas = QB_logl[j].sigmas,
                                                      MMs = QB_logl[j].MMs, NN = QB_logl[j].NN, prior_recovery = prior_recovery, glitch_indx = QB_logl[j].glitch_indx, wavelet_indx = QB_logl[j].wavelet_indx,
                                                      glitch_pulsars = QB_logl[j].glitch_pulsars))
            print('QB logl calc for initial sample')
            log_likelihood[j,0] = QB_logl[j].get_lnlikelihood(first_sample, vary_white_noise = vary_white_noise, vary_red_noise = rn_check)

            #Save first step for ensuring wavelet parameters are initially populated
            QB_logl[j].save_values(accept_new_step=True, vary_white_noise = vary_white_noise, vary_red_noise = rn_check)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

    #setting up array for the fisher eigenvalues
    #Default case for fisher eigenvectors (only steps along one parameter at a time)
    eig = np.broadcast_to(np.eye(10)*0.1, (n_chain, max_n_wavelet, 10, 10) ).copy()
    #also one for the glitch parameters
    eig_glitch = np.broadcast_to(np.eye(6)*0.1, (n_chain, max_n_glitch, 6, 6) ).copy()
    #one for GWB and common rn parameters, which we will keep updating
    eig_rn = np.broadcast_to( np.array([[1.0,0], [0,0.3]]), (n_chain, 2, 2)).copy()
    #and one for white noise/red noise parameters, which we will also keep updating
    eig_per_psr = np.broadcast_to(np.eye(num_per_psr_params)*0.1, (n_chain, num_per_psr_params, num_per_psr_params) ).copy()

    #read in tau_scan data if we will need it
    if tau_scan_proposal_weight+RJ_weight>0:
        if tau_scan_file==None:
            raise Exception("tau-scan data file is needed for tau-scan global propsals")
        with open(tau_scan_file, 'rb') as f:
            tau_scan_data = pickle.load(f)
            print("Tau-scan data read in successfully!")

        tau_scan = tau_scan_data['tau_scan']

        TAU_list = list(tau_scan_data['tau_edges'])
        F0_list = tau_scan_data['f0_edges']
        T0_list = tau_scan_data['t0_edges']

        #check if same prior range was used
        log_f0_max = float(pta.params[wavelet_indx[0,3]]._typename.split('=')[2][:-1])
        log_f0_min = float(pta.params[wavelet_indx[0,3]]._typename.split('=')[1].split(',')[0])
        t0_max = float(pta.params[wavelet_indx[0,8]]._typename.split('=')[2][:-1])
        t0_min = float(pta.params[wavelet_indx[0,8]]._typename.split('=')[1].split(',')[0])
        tau_max = float(pta.params[wavelet_indx[0,9]]._typename.split('=')[2][:-1])
        tau_min = float(pta.params[wavelet_indx[0,9]]._typename.split('=')[1].split(',')[0])

        print("#"*70)
        print("Tau-scan and MCMC prior range check (they must be the same)")
        print("tau_min: ", TAU_list[0], tau_min)
        print("tau_max: ", TAU_list[-1], tau_max)
        print("t0_min: ", T0_list[0][0]/3600/24/365.25, t0_min)
        print("t0_max: ", T0_list[0][-1]/3600/24/365.25, t0_max)
        print("f0_min: ", F0_list[0][0], 10**log_f0_min)
        print("f0_max: ", F0_list[0][-1], 10**log_f0_max)
        print("#"*70)

        #normalization
        norm = 0.0
        for idx, TTT in enumerate(tau_scan):
            for kk in range(TTT.shape[0]):
                for ll in range(TTT.shape[1]):
                    df = np.log10(F0_list[idx][kk+1]/F0_list[idx][kk])
                    dt = (T0_list[idx][ll+1]-T0_list[idx][ll])/3600/24/365.25
                    dtau = (TAU_list[idx+1]-TAU_list[idx])
                    norm += TTT[kk,ll]*df*dt*dtau
        tau_scan_data['norm'] = norm #TODO: Implement some check to make sure this is normalized over the same range as the prior range used in the MCMC

    #read in glitch_tau_scan data if we will need it

    if glitch_tau_scan_proposal_weight+glitch_RJ_weight>0:
        if glitch_tau_scan_file==None:
            raise Exception("glitch-tau-scan data file is needed for glitch model tau-scan global propsals")
        with open(glitch_tau_scan_file, 'rb') as f:
            glitch_tau_scan_data = pickle.load(f)
            print("Glitch tau-scan data read in successfully!")

        TAU_list = list(glitch_tau_scan_data['tau_edges'])
        F0_list = glitch_tau_scan_data['f0_edges']
        T0_list = glitch_tau_scan_data['t0_edges']

        #check if same prior range was used
        log_f0_max = float(pta.params[glitch_indx[0,0]]._typename.split('=')[2][:-1])
        log_f0_min = float(pta.params[glitch_indx[0,0]]._typename.split('=')[1].split(',')[0])
        t0_max = float(pta.params[glitch_indx[0,4]]._typename.split('=')[2][:-1])
        t0_min = float(pta.params[glitch_indx[0,4]]._typename.split('=')[1].split(',')[0])
        tau_max = float(pta.params[glitch_indx[0,5]]._typename.split('=')[2][:-1])
        tau_min = float(pta.params[glitch_indx[0,5]]._typename.split('=')[1].split(',')[0])

        print("#"*70)
        print("Glitch tau--scan and MCMC prior range check (they must be the same)")
        print("tau_min: ", TAU_list[0], tau_min)
        print("tau_max: ", TAU_list[-1], tau_max)
        print("t0_min: ", T0_list[0][0]/3600/24/365.25, t0_min)
        print("t0_max: ", T0_list[0][-1]/3600/24/365.25, t0_max)
        print("f0_min: ", F0_list[0][0], 10**log_f0_min)
        print("f0_max: ", F0_list[0][-1], 10**log_f0_max)
        print("#"*70)

        #normalization
        glitch_tau_scan_data['psr_idx_proposal'] = np.ones(len(pulsars))
        for i in range(len(pulsars)):
            glitch_tau_scan = glitch_tau_scan_data['tau_scan'+str(i)]

            norm = 0.0
            for idx, TTT in enumerate(glitch_tau_scan):
                for kk in range(TTT.shape[0]):
                    for ll in range(TTT.shape[1]):
                        df = np.log10(F0_list[idx][kk+1]/F0_list[idx][kk])
                        dt = (T0_list[idx][ll+1]-T0_list[idx][ll])/3600/24/365.25
                        dtau = (TAU_list[idx+1]-TAU_list[idx])
                        norm += TTT[kk,ll]*df*dt*dtau
            glitch_tau_scan_data['norm'+str(i)] = norm #TODO: Implement some check to make sure this is normalized over the same range as the prior range used in the MCMC

            tau_scan_limit = 0.1#0 #--start form 1 to avoid having zeros in the proposal
            #check if we've read in a tau-scan file
            if tau_scan_proposal_weight+RJ_weight<=0:
                #make fake tau_scan_data to use in next step
                tau_scan_data = {}
                tau_scan_data['tau_scan'] = [ggg*0.0 for ggg in glitch_tau_scan]
            for g_TS, TS in zip(glitch_tau_scan, tau_scan_data['tau_scan']):
                TS_max = np.max( g_TS - TS/np.sqrt(float(len(pulsars))) )
                if TS_max>tau_scan_limit:
                    tau_scan_limit = TS_max
            glitch_tau_scan_data['psr_idx_proposal'][i] = tau_scan_limit

        glitch_tau_scan_data['psr_idx_proposal'] = glitch_tau_scan_data['psr_idx_proposal']/np.sum(glitch_tau_scan_data['psr_idx_proposal'])
        print('-'*20)
        print("Glitch psr index proposal:")
        print(glitch_tau_scan_data['psr_idx_proposal'])
        print(np.sum(glitch_tau_scan_data['psr_idx_proposal']))
        print('-'*20)

    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros((10, n_chain)) #columns: chain number; rows: proposal type (glitch_RJ, glitch_tauscan, wavelet_RJ, wavelet_tauscan,  PT, fast fisher, regular fisher, noise_jump DE, noise_jump fisher, noise_jump prior draw)
    a_no=np.zeros((10, n_chain))
    acc_fraction = a_yes/(a_no+a_yes)
    if resume_from is None:
        swap_record = np.zeros((save_every_n+1, 1))
    rj_record = []

    #set up probabilities of different proposals
    total_weight = (regular_weight + PT_swap_weight + tau_scan_proposal_weight +
                    RJ_weight + noise_jump_weight + glitch_tau_scan_proposal_weight + glitch_RJ_weight)
    swap_probability = PT_swap_weight/total_weight
    tau_scan_proposal_probability = tau_scan_proposal_weight/total_weight
    regular_probability = regular_weight/total_weight
    RJ_probability = RJ_weight/total_weight
    noise_jump_probability = noise_jump_weight/total_weight
    glitch_tau_scan_proposal_probability = glitch_tau_scan_proposal_weight/total_weight
    glitch_RJ_probability = glitch_RJ_weight/total_weight
    print("Percentage of steps doing different jumps:\nPT swaps: {0:.2f}%\nRJ moves: {3:.2f}%\nGlitch RJ moves: {6:.2f}%\n\
Tau-scan-proposals: {1:.2f}%\nGlitch tau-scan-proposals: {5:.2f}%\nJumps along Fisher eigendirections: {2:.2f}%\nNoise jump: {4:.2f}%".format(swap_probability*100,
          tau_scan_proposal_probability*100, regular_probability*100,
          RJ_probability*100, noise_jump_probability*100, glitch_tau_scan_proposal_probability*100, glitch_RJ_probability*100))

    #No longer need if/else, since we simply append to existing file. Should be from 0 to N always.
    start_iter = 0
    stop_iter = N


    t_start = time.time()

    N_Noise_Params_changed = 0 #tuning param for noise jumps

    #initially define de_history = None until initialized with samples in main loop
    de_history = None
    #Set thin_de = n_fast_to_slow to ensure de_history gets updated during runtime
    if thin_de > n_fast_to_slow:
        thin_de = n_fast_to_slow

    #if DE jumps on, initialize DE history array
    if DE_prob > 0:
        de_history = initialize_de_history(n_chain, samples, QB_FPI, num_params, de_history_size = de_history_size, n_fast_to_slow = n_fast_to_slow, pta_params = pta.param_names, verbose = False)

    #########################
    #MAIN MCMC LOOP
    #########################
    for i in range(int(start_iter), int(stop_iter)): #-1 because ith step here produces (i+1)th sample based on ith sample
        ########################################################
        #
        #write results to file every save_every_n iterations
        #
        ########################################################

        #Case where i == int(stop_iter)-1 saves last N/(save_every_n*n_fast_to_slow) samples to file.
        #Case where i%save_every_n==0 saves every other chunk of samples.
        if savepath is not None and (i%save_every_n==0 or i == int(stop_iter)-1) and i!=start_iter:

        #TODO: If resume_from == savepath, check if params are the same, otherwise return error.
            print('Saving at sample {}'.format(i))
            print('int(stop_iter) - 1: ', int(stop_iter)-1)
            """output to hdf5 at loop iteration"""
            if savepath is not None:
                savefile = savepath + '.h5df'
                #Saving intermediate chunks
                if i>save_every_n:
                    with h5py.File(savefile, 'a') as f:
                        #Create shape for samples
                        f['samples_cold'].resize((f['samples_cold'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
                        f['swap_record'].resize((f['swap_record'].shape[0] + int((swap_record.shape[0] - 1)/thin)), axis = 0)
                        f['betas'].resize((f['betas'].shape[1] + int((betas.shape[1] - 1)/thin)), axis=1)
                        f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
                        f['PT_acc'].resize((f['PT_acc'].shape[1] + int((PT_acc.shape[1]-1)/thin)),axis=1)
                        #Save samples
                        f['samples_cold'][:,-int((samples.shape[1]-1)/thin):,:] = samples[:,:-1:thin,:]
                        f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = log_likelihood[:,:-1:thin]
                        f['betas'][:,-int((log_likelihood.shape[1]-1)/thin):] = betas[:, :-1:thin]
                        f['PT_acc'][:,-int((log_likelihood.shape[1]-1)/thin):] = PT_acc[:, :-1:thin]
                        f['acc_fraction'][...] = np.copy(acc_fraction)
                        f['swap_record'][-int((log_likelihood.shape[1]-1)/thin):] = np.copy(swap_record[:-1:thin])

                else:
                    #Creating h5df file at start of sampling if not resuming.
                    if resume_from is None:
                        print('Writing file at {}'.format(savefile))
                        with h5py.File(savefile, 'w') as f:
                            f.create_dataset('samples_cold', data= samples[:,:-1:thin,:], compression="gzip", chunks=True, maxshape = (n_chain, None, samples.shape[2])) #maxshape=(n_chain,int(N/thin),samples.shape[2]))
                            f.create_dataset('log_likelihood', data=log_likelihood[:,:-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))
                            f.create_dataset('par_names', data=np.array(pta.param_names, dtype='S'))
                            f.create_dataset('acc_fraction', data=acc_fraction)
                            f.create_dataset('swap_record', data = swap_record[:-1:thin], compression="gzip", chunks=True, maxshape = (None,1))# maxshape=int(N/thin))
                            f.create_dataset('betas', data=betas[:, :-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))
                            f.create_dataset('PT_acc', data=PT_acc[:, :-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))

                    #If resuming from a file where resume_from != savefile
                    #May need 2 cases here.
                    elif resume_from != savefile and not os.path.exists(savefile):
                        print('Creating file at {}, \n Starting from {}'.format(savefile, resume_from))
                        with h5py.File(savefile, 'w') as f:
                            f.create_dataset('samples_cold', data= samples[:,:-1:thin,:], compression="gzip", chunks=True, maxshape = (n_chain, None, samples.shape[2])) #maxshape=(n_chain,int(N/thin),samples.shape[2]))
                            f.create_dataset('log_likelihood', data=log_likelihood[:,:-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))
                            f.create_dataset('par_names', data=np.array(pta.param_names, dtype='S'))
                            f.create_dataset('acc_fraction', data=acc_fraction)
                            f.create_dataset('swap_record', data = swap_record[:-1:thin], compression="gzip", chunks=True, maxshape = (None,1))# maxshape=int(N/thin))
                            f.create_dataset('betas', data=betas[:, :-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))
                            f.create_dataset('PT_acc', data=PT_acc[:, :-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))

                    else:
                        #If resuming from existing file, append to file.
                        with h5py.File(savefile, 'a') as f:
                            print('Appending to file {}'.format(savefile))
                            #Create shape for samples
                            f['samples_cold'].resize((f['samples_cold'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
                            f['swap_record'].resize((f['swap_record'].shape[0] + int((swap_record.shape[0] - 1)/thin)), axis = 0)
                            f['betas'].resize((f['betas'].shape[1] + int((betas.shape[1] - 1)/thin)), axis=1)
                            f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
                            f['PT_acc'].resize((f['PT_acc'].shape[1] + int((PT_acc.shape[1]-1)/thin)),axis=1)
                            #Save samples
                            f['samples_cold'][:,-int((samples.shape[1]-1)/thin):,:] = samples[:,:-1:thin,:]
                            f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = log_likelihood[:,:-1:thin]
                            f['betas'][:,-int((log_likelihood.shape[1]-1)/thin):] = betas[:, :-1:thin]
                            f['PT_acc'][:,-int((log_likelihood.shape[1]-1)/thin):] = PT_acc[:, :-1:thin]
                            f['acc_fraction'][...] = np.copy(acc_fraction)
                            f['swap_record'][-int((log_likelihood.shape[1]-1)/thin):] = np.copy(swap_record[:-1:thin])

            #clear out log_likelihood and samples arrays
            samples_now = samples[:,-1,:]
            log_likelihood_now = log_likelihood[:,-1]
            betas_now = betas[:, -1]
            PT_acc_now = PT_acc[:, -1]
            #Clearing out old arrays
            samples = np.zeros((n_chain, save_every_n+1, num_params))
            log_likelihood = np.zeros((n_chain, save_every_n+1))

            betas = np.zeros((n_chain, save_every_n+1))
            PT_acc = np.zeros((n_chain-1, save_every_n+1))

            #Setting first sample for new arrays
            samples[:,0,:] = samples_now
            log_likelihood[:,0] = log_likelihood_now
            betas[:, 0] = betas_now
            PT_acc[:, 0] = PT_acc_now

        
        #Updates temeratures (betas) for parallel tempering. We do this at the start of the loop at every step because the betas array with made to record the temperature at every single step.
        #If one is running with a dynamic temperature ladder, then this update will be overwritten when the temperature ladder is changed. This happens in the do_pt_swap() function.
        betas[:,(i%save_every_n)+1] = betas[:,i%save_every_n]      
        ########################################################
        #
        #logging PT acceptance fraction
        #
        ########################################################
        #logging mean acc probability over last PT_hist_length swaps
        if i%n_fast_to_slow == 0:
            PT_acc[:,i%save_every_n] = np.nanmean(PT_hist, axis=1) #nanmean so early on when we still have nans we only use the actual data
        else: #trying to minimize calculations by only trying to re-calc after slow steps
            if i%save_every_n != 0:
                PT_acc[:,i%save_every_n] = PT_acc[:,i%save_every_n -1]

        ########################################################
        #
        #print out run state every n_status_update iterations
        #
        ########################################################
        if i%n_status_update==0:
            acc_fraction = a_yes/(a_no+a_yes)
            if jupyter_notebook:
                print('Progress: {0:2.2f}% '.format(i/N*100) + '\r',end='')
            else:
                print('Progress: {0:2.2f}% '.format(i/N*100) +
                        'Acceptance fraction #columns: chain number; rows: proposal type (glitch_RJ, glitch_tauscan, wavelet_RJ, wavelet_tauscan, PT, fast_jump, regular_jump, noise_jump DE, noise_jump fisher, noise_jump prior):'+'\n')
                print('Run Time: {0}s'.format(time.time()-t_start))
                print(acc_fraction)
                print(PT_acc[:,i%save_every_n])
                print('Differential evolution array: {0}'.format(de_arr_itr))
        #################################################################################
        #
        #update our eigenvectors from the fisher matrix every n_fish_update iterations
        #
        #################################################################################

        if i%n_fish_update==0:
            #only update T>1 chains every 10th time
            if i%(n_fish_update*10)==0:
                max_j = n_chain
            else:
                max_j = 1

            for j in range(max_j):
                if prior_recovery == False:
                    n_wavelet = int(samples[j,i%save_every_n,0])
                    n_glitch = int(samples[j,i%save_every_n,1])

                    # Fisher Information Matrix: Calculates the covariances for each parameter associated with
                    # maximum likelihood estimates. This is used to inform jump proposals in parameter space
                    # for various kinds of jumps by pulling out eigenvectors from Fisher Matrix for particular
                    # parameters that are being updated.

                    #wavelet eigenvectors
                    if n_wavelet!=0:
                        eigenvectors = get_fisher_eigenvectors(np.copy(samples[j, i%save_every_n, 2:]), pta, QB_FP, QB_logl=QB_logl[j], T_chain=1/betas[j,i%save_every_n], n_sources=n_wavelet, array_index=wavelet_indx, flag = True)
                        if eigenvectors.size > 0:#np.all(eigenvectors[np.where(eigenvectors != 0)]):
                            eig[j,:n_wavelet,:,:] = eigenvectors
                        else:
                            print('wave bad')

                    #glitch eigenvectors
                    if n_glitch!=0:
                        eigen_glitch = get_fisher_eigenvectors(np.copy(samples[j, i%save_every_n, 2:]), pta, QB_FP, QB_logl=QB_logl[j], T_chain=1/betas[j,i%save_every_n], n_sources=n_glitch, dim=6, array_index=glitch_indx, flag = True)
                        if eigen_glitch.size > 0:
                            eig_glitch[j,:n_glitch,:,:] = eigen_glitch
                        else:
                            print('glitch bad')

                    #RN eigenvectors
                    if vary_rn:
                        eigvec_rn = get_fisher_eigenvectors(np.copy(samples[j, i%save_every_n, 2:]), pta, QB_FP, QB_logl=QB_logl[j], T_chain=1/betas[j,i%save_every_n], n_sources=1, dim=len(rn_indx), array_index=rn_indx, vary_rn = vary_rn)
                        if eigvec_rn.size > 0:
                            eig_rn[j,:,:] = eigvec_rn[0,:,:]
                        else:
                            print('rn bad')

                    #per PSR eigenvectors
                    if j == 0:
                        if vary_per_psr_rn or vary_white_noise:
                            #T_chain=1/betas[j,i%save_every_n]

                            ''' per_psr_eigvec indexes correspond to: [chain, pulsar, param, param]'''
                            if len(pulsars) == 1:
                                #if single pulsar runs, needs to use different index from per_puls_indx
                                per_psr_eigvec = get_fisher_eigenvectors(np.copy(samples[j, i%save_every_n, 2:]), pta, QB_FP, QB_logl=QB_logl[j], n_sources=len(pulsars), dim=len(per_puls_indx[0]), array_index=per_puls_indx, vary_white_noise = vary_white_noise, vary_psr_red_noise = vary_per_psr_rn)
                            else:
                                per_psr_eigvec = get_fisher_eigenvectors(np.copy(samples[j, i%save_every_n, 2:]), pta, QB_FP, QB_logl=QB_logl[j], n_sources=len(pulsars), dim=len(per_puls_indx[1]), array_index=per_puls_indx, vary_white_noise = vary_white_noise, vary_psr_red_noise = vary_per_psr_rn)
                            #set to 0 all params in each eigenvector that are less than highest val
                            #set small values to 0, sum together pulsar eigenvectors in jump to get more informative jump
                            #Add extra loop over pulsars in noise_jump to add together pulsar eigenvectors in jump

                            #Set the number of noise parameters changed per fisher call
                            N_Noise_Params_changed = int(per_psr_eigvec[0].shape[0]) #tuning param for noise jumps
                            if N_Noise_Params_changed < 10:
                                print('Varying all noise parameters! {} parameters.'.format(N_Noise_Params_changed))
                            if N_Noise_Params_changed > 10:
                                N_Noise_Params_changed = 10
                                print('Greater than 10 noise parameters. Varying {} noise parameters.'.format(N_Noise_Params_changed))
            #Approximation for eigenvectors (scale other eigenvectors by chain temps)
            if vary_per_psr_rn or vary_white_noise:
                for j in range(n_chain):
                    T_chain=1/betas[j,i%save_every_n]
                    if per_psr_eigvec.size > 0:
                        eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]*(T_chain)**(1/2)
                        print('eign_per_pulsar re-written')
                    else:
                        print('per_psr bad')
        ###########################################################
        #
        #Do the actual MCMC step
        #
        ###########################################################
        if i%n_fast_to_slow==0:
            #draw a random number to decide which jump to do

            #Choose jump to perform
            jump_decide = np.random.uniform()

            accept_jump_arr = np.zeros(n_chain)

            #i%save_every_n will check where we are in sample blocks
            if (jump_decide<swap_probability):
                do_pt_swap(n_chain, max_n_wavelet, max_n_glitch, pta, QB_FPI, QB_logl, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, swap_record, vary_white_noise, num_noise_params, log_likelihood, PT_hist, PT_hist_idx, n_fast_to_slow, save_every_n, i, T_dynamic, T_dynamic_nu, T_dynamic_t0)

            #global proposal based on tau_scan
            elif (jump_decide<swap_probability+tau_scan_proposal_probability):
                accept_jump_arr = do_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, pta, QB_FPI, QB_logl,  QB_Info, samples, i%save_every_n, betas, a_yes, a_no, vary_white_noise, num_noise_params, tau_scan_data, log_likelihood, wavelet_indx, glitch_indx)

            #jump to change number of wavelets
            elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability):
                accept_jump_arr = do_wavelet_rj_move(n_chain, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior, pta, QB_FPI, QB_logl, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, rj_record, vary_white_noise, num_noise_params, tau_scan_data, log_likelihood,  wavelet_indx, glitch_indx)

            #jump to change some noise parameters
            elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+noise_jump_probability):
                accept_jump_arr = noise_jump(n_chain, max_n_wavelet, max_n_glitch, pta, QB_FPI, QB_logl, QB_Info,
                    samples, i%save_every_n, betas, a_yes, a_no, eig_per_psr, per_puls_indx, per_puls_rn_indx, per_puls_wn_indx, all_noiseparam_idxs,
                    num_noise_params, vary_white_noise, vary_per_psr_rn, log_likelihood, wavelet_indx, glitch_indx, N_Noise_Params_changed, de_history, total_weight, DE_prob, fisher_prob, prior_draw_prob)
            #jump to change glitch params
            elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+noise_jump_probability+glitch_tau_scan_proposal_probability):
                accept_jump_arr = do_glitch_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, pta,  QB_FPI, QB_logl, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, vary_white_noise, num_noise_params, glitch_tau_scan_data, log_likelihood, wavelet_indx, glitch_indx)

            #jump to change number of glitches
            elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+noise_jump_probability+glitch_tau_scan_proposal_probability+glitch_RJ_probability):
                accept_jump_arr = do_glitch_rj_move(n_chain, max_n_wavelet, max_n_glitch, n_glitch_prior, pta,  QB_FPI, QB_logl, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, vary_white_noise, num_noise_params, glitch_tau_scan_data, log_likelihood, wavelet_indx, glitch_indx)

            #do regular jump
            else:
                accept_jump_arr = regular_jump(n_chain, max_n_wavelet, max_n_glitch, pta,  QB_FPI, QB_logl, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, eig, eig_glitch, eig_rn,
                             num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, rn_indx, log_likelihood, total_weight, DE_prob, fisher_prob, prior_draw_prob)

            #update de history after every shape parameter update
            if DE_prob > 0:
                if i != int(stop_iter):
                    de_history = update_de_history(n_chain, samples, de_history, QB_FPI, num_params, i, accept_jump_arr, de_history_size = de_history_size, n_fast_to_slow = n_fast_to_slow, save_every_n = save_every_n, thin_de = n_fast_to_slow)


        else:
            #For fast jumps, can't have wavelet_indx[i, 3, 8, 9] or glitch_indx[i, 0, 3, 4, 5] Otherwise M and N gets recalculated
            #Note: i%save_every_n will be 1 through 9 when i%n_fast_to_slow != 0.

            fast_jump(n_chain, max_n_wavelet, max_n_glitch, QB_FPI, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, eig, eig_glitch, eig_rn, num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, log_likelihood)

    acc_fraction = a_yes/(a_no+a_yes)

    return samples[:,::n_fast_to_slow,:], acc_fraction, swap_record, rj_record, pta, log_likelihood[:,::n_fast_to_slow], betas[:,::n_fast_to_slow], PT_acc

################################################################################
#
#GLOBAL PROPOSAL BASED ON TAU-SCAN
#
################################################################################
def do_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, pta, FPI, QB_logl, QB_Info,
                            samples, i, betas, a_yes, a_no, vary_white_noise, num_noise_params,
                            tau_scan_data, log_likelihood, wavelet_indx, glitch_indx):

    tau_scan = tau_scan_data['tau_scan']
    tau_scan_limit = 0
    for TS in tau_scan:
        TS_max = np.max(TS)
        if TS_max>tau_scan_limit:
            tau_scan_limit = TS_max

    TAU_list = list(tau_scan_data['tau_edges'])
    F0_list = tau_scan_data['f0_edges']
    T0_list = tau_scan_data['t0_edges']

    #Keeps track of if the jump was accepted or rejected. The array is an array of
    accept_jump_arr = np.zeros(n_chain)

    for j in range(n_chain):
        #check if there's any wavelet -- stay at given point if not
        n_wavelet = int(samples[j,i,0]) #get_n_wavelet(samples, j, i)
        n_glitch = int(samples[j,i,1]) #get_n_glitch(samples, j, i)

        if n_wavelet==0:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[3,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            continue

        log_f0_max = float(pta.params[wavelet_indx[0,3]]._typename.split('=')[2][:-1])
        log_f0_min = float(pta.params[wavelet_indx[0,3]]._typename.split('=')[1].split(',')[0])
        t0_max = float(pta.params[wavelet_indx[0,8]]._typename.split('=')[2][:-1])
        t0_min = float(pta.params[wavelet_indx[0,8]]._typename.split('=')[1].split(',')[0])
        tau_max = float(pta.params[wavelet_indx[0,9]]._typename.split('=')[2][:-1])
        tau_min = float(pta.params[wavelet_indx[0,9]]._typename.split('=')[1].split(',')[0])

        accepted = False
        while accepted==False:
            #propose new distribution for shape parameters
            log_f0_new = np.random.uniform(low=log_f0_min, high=log_f0_max)
            t0_new = np.random.uniform(low=t0_min, high=t0_max)
            tau_new = np.random.uniform(low=tau_min, high=tau_max)

            #bin up distributions based on tau scan data
            tau_idx = np.digitize(tau_new, np.array(TAU_list)) - 1
            f0_idx = np.digitize(10**log_f0_new, np.array(F0_list[tau_idx])) - 1
            t0_idx = np.digitize(t0_new, np.array(T0_list[tau_idx])/(365.25*24*3600)) - 1

            #pick new tau scan point
            tau_scan_new_point = tau_scan[tau_idx][f0_idx, t0_idx]

            #see if new tau scan point is accepted (normalized to max tau scan value from tau scan data)
            if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
                accepted = True

        #randomly select other parameters (except sky location and psi, which we won't change here)

        cos_gwtheta_old = np.copy(samples[j, i, 2+wavelet_indx[0,0]])
        gwphi_old =  np.copy(samples[j, i, 2+wavelet_indx[0,2]])
        psi_old = np.copy(samples[j, i, 2+wavelet_indx[0,1]])

        #Sample new wavelet parameters
        log10_h_new = pta.params[wavelet_indx[0,4]].sample()
        log10_h_cross_new = pta.params[wavelet_indx[0,5]].sample()
        phase0_new = pta.params[wavelet_indx[0,6]].sample()
        phase0_cross_new = pta.params[wavelet_indx[0,7]].sample()

        #select particular wavelet
        wavelet_select = np.random.randint(n_wavelet)

        samples_current = np.copy(samples[j, i, 2:]) #strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
        new_point = np.copy(samples_current)
        new_point[wavelet_indx[wavelet_select,0]] = cos_gwtheta_old
        new_point[wavelet_indx[wavelet_select,1]] = psi_old
        new_point[wavelet_indx[wavelet_select,2]] = gwphi_old
        new_point[wavelet_indx[wavelet_select,3]] = log_f0_new
        new_point[wavelet_indx[wavelet_select,4]] = log10_h_new
        new_point[wavelet_indx[wavelet_select,5]] = log10_h_cross_new
        new_point[wavelet_indx[wavelet_select,6]] = phase0_new
        new_point[wavelet_indx[wavelet_select,7]] = phase0_cross_new
        new_point[wavelet_indx[wavelet_select,8]] = t0_new
        new_point[wavelet_indx[wavelet_select,9]] = tau_new

        log_L = QB_logl[j].get_lnlikelihood(new_point)
        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)

        #getting ratio of proposal densities!
        tau_old = samples[j,i,2+wavelet_indx[wavelet_select,9]]
        f0_old = 10**samples[j,i,2+wavelet_indx[wavelet_select,3]]
        t0_old = samples[j,i,2+wavelet_indx[wavelet_select,8]]

        tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
        f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
        t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

        tau_scan_old_point = tau_scan[tau_idx_old][f0_idx_old, t0_idx_old]

        log10_h_old = samples[j,i,2+wavelet_indx[wavelet_select,4]]
        log10_h_cross_old = samples[j,i,2+wavelet_indx[wavelet_select,5]]

        hastings_extra_factor = pta.params[wavelet_indx[0,4]].get_pdf(log10_h_old) / pta.params[wavelet_indx[0,4]].get_pdf(log10_h_new)
        hastings_extra_factor *= pta.params[wavelet_indx[0,5]].get_pdf(log10_h_cross_old) / pta.params[wavelet_indx[0,5]].get_pdf(log10_h_cross_new)
        acc_ratio = np.exp(log_acc_ratio)*(tau_scan_old_point/tau_scan_new_point) * hastings_extra_factor

        #acc_ratio = 1
        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            a_yes[3,j]+=1
            log_likelihood[j,i+1] = log_L

            QB_logl[j].save_values(accept_new_step=True)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

            accept_jump_arr[j] = 1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[3,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            QB_logl[j].save_values(accept_new_step=False)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
    return accept_jump_arr

################################################################################
#
#GLITCH MODEL GLOBAL PROPOSAL BASED ON TAU-SCAN
#
################################################################################
def do_glitch_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, pta,
                                   FPI, QB_logl, QB_Info, samples, i, betas, a_yes,
                                   a_no, vary_white_noise, num_noise_params, glitch_tau_scan_data,
                                    log_likelihood, wavelet_indx, glitch_indx):

    TAU_list = list(glitch_tau_scan_data['tau_edges'])
    F0_list = glitch_tau_scan_data['f0_edges']
    T0_list = glitch_tau_scan_data['t0_edges']

    #Keeps track of if the jump was accepted or rejected. Need to keep track of each PT chain so we have an array
    accept_jump_arr = np.zeros(n_chain)

    for j in range(n_chain):
        #check if there's any wavelet -- stay at given point if not
        n_wavelet = int(samples[j,i,0]) #get_n_wavelet(samples, j, i)
        n_glitch = int(samples[j,i,1]) #get_n_glitch(samples, j, i)
        if n_glitch==0:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[1,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            continue

        #select which glitch to change
        glitch_select = np.random.randint(n_glitch)

        #pick which pulsar to move the glitch to (stay at where we are in 50% of the time) -- might be an issue with detailed balance
        if np.random.uniform()<=0.5:
            psr_idx = np.random.uniform(low=-0.5, high=len(pta.pulsars)-1 )
        else:
            psr_idx = samples[j,i,2+glitch_indx[glitch_select,3]]

        #load in the appropriate tau-scan
        tau_scan = glitch_tau_scan_data['tau_scan'+str(int(np.round(psr_idx)))]
        tau_scan_limit = 0
        for TS in tau_scan:
            TS_max = np.nanmax(TS)
            if TS_max>tau_scan_limit:
                tau_scan_limit = TS_max

        log_f0_max = float(pta.params[glitch_indx[0,0]]._typename.split('=')[2][:-1])
        log_f0_min = float(pta.params[glitch_indx[0,0]]._typename.split('=')[1].split(',')[0])
        t0_max = float(pta.params[glitch_indx[0,4]]._typename.split('=')[2][:-1])
        t0_min = float(pta.params[glitch_indx[0,4]]._typename.split('=')[1].split(',')[0])
        tau_max = float(pta.params[glitch_indx[0,5]]._typename.split('=')[2][:-1])
        tau_min = float(pta.params[glitch_indx[0,5]]._typename.split('=')[1].split(',')[0])

        accepted = False
        while accepted==False:
            log_f0_new = np.random.uniform(low=log_f0_min, high=log_f0_max)
            t0_new = np.random.uniform(low=t0_min, high=t0_max)
            tau_new = np.random.uniform(low=tau_min, high=tau_max)

            tau_idx = np.digitize(tau_new, np.array(TAU_list)) - 1
            f0_idx = np.digitize(10**log_f0_new, np.array(F0_list[tau_idx])) - 1
            t0_idx = np.digitize(t0_new, np.array(T0_list[tau_idx])/(365.25*24*3600)) - 1

            tau_scan_new_point = tau_scan[tau_idx][f0_idx, t0_idx]
            if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
                accepted = True

        #randomly select phase and amplitude
        phase0_new = pta.params[glitch_indx[0,2]].sample()
        log10_h_new = pta.params[glitch_indx[0,1]].sample()

        samples_current = np.copy(samples[j, i, 2:]) #strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
        new_point = np.copy(samples_current)

        #Only change glitch parameters between new and old point
        new_point[glitch_indx[glitch_select,0]] = log_f0_new
        new_point[glitch_indx[glitch_select,1]] = log10_h_new
        new_point[glitch_indx[glitch_select,2]] = phase0_new
        new_point[glitch_indx[glitch_select,3]] = psr_idx
        new_point[glitch_indx[glitch_select,4]] = t0_new
        new_point[glitch_indx[glitch_select,5]] = tau_new

        log_L = QB_logl[j].get_lnlikelihood(new_point)

        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)

        #getting ratio of proposal densities!
        tau_old = samples[j,i,2+glitch_indx[glitch_select,5]]
        f0_old = 10**samples[j,i,2+glitch_indx[glitch_select,0]]
        t0_old = samples[j,i,2+glitch_indx[glitch_select,4]]

        #get old psr index and load in appropriate tau scan
        psr_idx_old = samples[j,i,2+glitch_indx[glitch_select,3]]
        tau_scan_old = glitch_tau_scan_data['tau_scan'+str(int(np.round(psr_idx_old)))]
        tau_scan_limit_old = 0
        for TS in tau_scan_old:
            TS_max = np.nanmax(TS)
            if TS_max>tau_scan_limit_old:
                tau_scan_limit_old = TS_max

        tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
        f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
        t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

        tau_scan_old_point = tau_scan_old[tau_idx_old][f0_idx_old, t0_idx_old]

        log10_h_old = samples[j,i,2+glitch_indx[glitch_select,1]]
        hastings_extra_factor = pta.params[glitch_indx[0,1]].get_pdf(log10_h_old) / pta.params[glitch_indx[0,1]].get_pdf(log10_h_new)

        acc_ratio = np.exp(log_acc_ratio)*(tau_scan_old_point/tau_scan_new_point) * (tau_scan_limit/tau_scan_limit_old) * hastings_extra_factor

        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            a_yes[1,j]+=1
            log_likelihood[j,i+1] = log_L

            QB_logl[j].save_values(accept_new_step=True)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

            accept_jump_arr[j] = 1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[1,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]

            QB_logl[j].save_values(accept_new_step=False)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
   
    return accept_jump_arr

################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN CW, GWB AND RN PARAMETERS)
#
################################################################################
def regular_jump(n_chain, max_n_wavelet, max_n_glitch, pta, FPI, QB_logl, QB_Info,
                 samples, i, betas, a_yes, a_no, eig, eig_glitch, eig_rn,
                 num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx,
                 rn_indx, log_likelihood, total_weight, DE_prob, fisher_prob, prior_draw_prob):
    
    #Keeps track of if the jump was accepted or rejected. Need to keep track of each PT chain so we have an array
    accept_jump_arr = np.zeros(n_chain)

    for j in range(n_chain):
        n_wavelet = int(samples[j,i,0]) #get_n_wavelet(samples, j, i)
        n_glitch = int(samples[j,i,1]) #get_n_glitch(samples, j, i)
        rn_changed = False #flag for if the common rn gets varied
        wn_changed = False #Should always be false for regular jumps

        samples_current = np.copy(samples[j, i, 2:]) #strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

        #decide if moving in wavelet parameters, glitch parameters, or GWB/RN parameters
        #case #1: we can vary any of them
        if n_wavelet!=0 and n_glitch!=0 and vary_rn:
            vary_decide = np.random.uniform()
            if vary_decide <= 1.0/3.0:
                what_to_vary = 'WAVE'
            elif vary_decide <= 2.0/3.0:
                what_to_vary = 'GLITCH'
            else:
                what_to_vary = 'RN'
        #case #2: whe can vary two of them
        elif n_glitch!=0 and vary_rn:
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'GLITCH'
            else:
                what_to_vary = 'RN'
        elif n_wavelet!=0 and vary_rn:
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'WAVE'
            else:
                what_to_vary = 'RN'
        elif n_wavelet!=0 and n_glitch!=0:
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'GLITCH'
            else:
                what_to_vary = 'WAVE'
        #case #3: we can only vary one of them
        elif n_wavelet!=0:
            what_to_vary = 'WAVE'
        elif n_glitch!=0:
            what_to_vary = 'GLITCH'
        elif vary_rn:
            what_to_vary = 'RN'
        #case #4: nothing to vary
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            continue

        # which_jump = np.random.choice(3, p=[DE_prob/total_weight,
        #                                     fisher_prob/total_weight,
        #                                     prior_draw_prob/total_weight])
        if what_to_vary == 'WAVE':
            wavelet_select = np.random.randint(n_wavelet)
            jump_select = np.random.randint(10)
            jump_1wavelet = eig[j,wavelet_select,jump_select,:]
            jump = np.zeros(samples_current.size)
            #change intrinsic (and extrinsic) parameters of selected wavelet

            #This is meant to replace all values for a specific wavelet,
            #which means we need to index 1 beyond the end so it will see all values
            jump[wavelet_indx[wavelet_select,0]:wavelet_indx[wavelet_select,9]+1] = jump_1wavelet
            #and change sky location and polarization angle of all wavelets
            for which_wavelet in range(n_wavelet):
                jump[wavelet_indx[which_wavelet,0]] = jump_1wavelet[0]
                jump[wavelet_indx[which_wavelet,1]] = jump_1wavelet[1]
                jump[wavelet_indx[which_wavelet,2]] = jump_1wavelet[2]
        elif what_to_vary == 'GLITCH':
            glitch_select = np.random.randint(n_glitch)
            jump_select = np.random.randint(6)
            jump_1glitch = eig_glitch[j,glitch_select,jump_select,:]
            jump = np.zeros(samples_current.size)
            #change intrinsic (and extrinsic) parameters of selected wavelet

            #This is meant to replace all values for a specific glitch,
            #which means we need to index 1 beyond the end so it will see all values
            jump[glitch_indx[glitch_select,0]:glitch_indx[glitch_select,5]+1] = jump_1glitch
        elif what_to_vary == 'RN':
            rn_changed = True
            jump_select = np.random.randint(2)
            jump_rn = eig_rn[j,jump_select,:]
            jump = np.zeros(samples_current.size)

            jump[rn_indx[0]:rn_indx[1]+1] = jump_rn

        new_point = samples_current + jump*np.random.normal()#only sd of 1 for all parameter jumps

        #check if we are inside prior before calling likelihood, otherwise it throws an error
        new_point = correct_intrinsic(new_point, FPI, FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
        new_log_prior = QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)
        if new_log_prior==-np.inf: #check if prior is -inf - reject step if it is
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            continue

        log_L = QB_logl[j].get_lnlikelihood(new_point, vary_red_noise = rn_changed, vary_white_noise = wn_changed)
        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += new_log_prior
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)

        acc_ratio = np.exp(log_acc_ratio)

        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            a_yes[6,j]+=1
            log_likelihood[j,i+1] = log_L

            QB_logl[j].save_values(accept_new_step=True)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

            #step accepted
            accept_jump_arr[j] = 1    
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]


            QB_logl[j].save_values(accept_new_step=False, vary_red_noise = rn_changed, vary_white_noise = wn_changed)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

    return accept_jump_arr

################################################################################
#
#Fast MCMC JUMP ROUTINE (jumping in projection PARAMETERS)
#
################################################################################
@njit(fastmath=True,parallel=False)
def fast_jump(n_chain, max_n_wavelet, max_n_glitch, FPI, QB_Info, samples, i,
              betas, a_yes, a_no, eig, eig_glitch, eig_rn, num_noise_params,
              num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, log_likelihood):

    for j in range(n_chain):
        n_wavelet = int(samples[j,i,0])
        n_glitch = int(samples[j,i,1])

        samples_current = np.copy(samples[j,i,2:])

        #decide if moving in wavelet parameters, glitch parameters, or GWB/RN parameters
        #4/23/24: Implement prior draws for 10% of fast steps
        #case #1: we can vary any of them
        if n_wavelet!=0 and n_glitch!=0:
            vary_decide = np.random.random()
            if vary_decide <= 0.5:
                what_to_vary = 'GLITCH'
            else:
                what_to_vary = 'WAVE'
        #case #3: we can only vary one of them
        elif n_wavelet!=0:
            what_to_vary = 'WAVE'
        elif n_glitch!=0:
            what_to_vary = 'GLITCH'
        #case #4: nothing to vary
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[5,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            continue

        ####
        #could potentialy add more agressive jumps drawing from the prior if acceptance is to high for these jumps
        ####

        if what_to_vary == 'WAVE':
            wavelet_select = np.random.randint(n_wavelet)
            jump_select = np.random.randint(10)
            jump_1wavelet = eig[j,wavelet_select,jump_select,:]
            jump = np.zeros(samples_current.size)
            #change intrinsic (and extrinsic) parameters of selected wavelet

            #This is meant to replace all values for a specific glitch,
            #which means we need to index 1 beyond the end so it will see all values
            #For not varying shape parameters, we need to parse through jump and eigenvectors
            jump[wavelet_indx[wavelet_select,0]:wavelet_indx[wavelet_select,9]+1] = jump_1wavelet

            #to avoid all shape parameters for wavelets: wavelet_indx[i, 0, 6, or 7]
            #for glitches: glitch_indx[i, 0, 4, 5]
            jump[wavelet_indx[wavelet_select,3]] = 0 #f0
            jump[wavelet_indx[wavelet_select,8]] = 0 #t0
            jump[wavelet_indx[wavelet_select,9]] = 0 #tau

            #and change sky location and polarization angle of all wavelets
            for which_wavelet in range(n_wavelet):
                jump[wavelet_indx[which_wavelet,0]] = jump_1wavelet[0]
                jump[wavelet_indx[which_wavelet,1]] = jump_1wavelet[1]
                jump[wavelet_indx[which_wavelet,2]] = jump_1wavelet[2]
        elif what_to_vary == 'GLITCH':
            glitch_select = np.random.randint(n_glitch)
            jump_select = np.random.randint(6)
            jump_1glitch = eig_glitch[j,glitch_select,jump_select,:]
            jump = np.zeros(samples_current.size)
            #change intrinsic (and extrinsic) parameters of selected wavelet

            #This is meant to replace all values for a specific glitch,
            #which means we need to index 1 beyond the end so it will see all values
            jump[glitch_indx[glitch_select,0]:glitch_indx[glitch_select,5]+1] = jump_1glitch

            jump[glitch_indx[glitch_select,0]] = 0 #f0
            jump[glitch_indx[glitch_select,3]] = 0 #psr_idx
            jump[glitch_indx[glitch_select,4]] = 0 #t0
            jump[glitch_indx[glitch_select,5]] = 0 #tau

        new_point = samples_current + jump*np.random.normal()#only sd of 1 for all parameter jumps

        new_point = correct_intrinsic(new_point, FPI, FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
        new_log_prior = QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)

        if new_log_prior==-np.inf: #check if prior is -inf - reject step if it is
            samples[j,i+1,:] = samples[j,i,:]
            a_no[5,j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            continue

        log_L = QB_Info[j].get_lnlikelihood(new_point)

        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += new_log_prior
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)


        acc_ratio = np.exp(log_acc_ratio)

        if np.random.random()<=acc_ratio:
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            a_yes[5,j]+=1
            log_likelihood[j,i+1] = log_L

        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[5,j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]



################################################################################
#
#REVERSIBLE-JUMP (RJ, aka TRANS-DIMENSIONAL) MOVE -- adding or removing a wavelet
#
################################################################################
def do_wavelet_rj_move(n_chain, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior,
                       pta, FPI, QB_logl, QB_Info, samples, i, betas, a_yes, a_no,
                       rj_record, vary_white_noise, num_noise_params, tau_scan_data,
                       log_likelihood, wavelet_indx, glitch_indx):

    tau_scan = tau_scan_data['tau_scan']

    tau_scan_limit = 0
    for TS in tau_scan:
        TS_max = np.nanmax(TS)
        if TS_max>tau_scan_limit:
            tau_scan_limit = TS_max

    TAU_list = list(tau_scan_data['tau_edges'])
    F0_list = tau_scan_data['f0_edges']
    T0_list = tau_scan_data['t0_edges']

    #Keeps track of if the jump was accepted or rejected. The array is an array of
    accept_jump_arr = np.zeros(n_chain)

    for j in range(n_chain):
        n_wavelet = int(samples[j,i,0]) #get_n_wavelet(samples, j, i)
        n_glitch = int(samples[j,i,1]) #get_n_glitch(samples, j, i)

        add_prob = 0.5 #same propability of adding and removing
        #decide if we add or remove a signal
        direction_decide = np.random.uniform()

        if n_wavelet==min_n_wavelet or (direction_decide<add_prob and n_wavelet!=max_n_wavelet): #adding a wavelet------------------------------------------------------

            if j==0: rj_record.append(1)

            log_f0_max = float(pta.params[wavelet_indx[0,3]]._typename.split('=')[2][:-1])
            log_f0_min = float(pta.params[wavelet_indx[0,3]]._typename.split('=')[1].split(',')[0])
            t0_max = float(pta.params[wavelet_indx[0,8]]._typename.split('=')[2][:-1])
            t0_min = float(pta.params[wavelet_indx[0,8]]._typename.split('=')[1].split(',')[0])
            tau_max = float(pta.params[wavelet_indx[0,9]]._typename.split('=')[2][:-1])
            tau_min = float(pta.params[wavelet_indx[0,9]]._typename.split('=')[1].split(',')[0])

            accepted = False
            while accepted==False:
                log_f0_new = np.random.uniform(low=log_f0_min, high=log_f0_max)
                t0_new = np.random.uniform(low=t0_min, high=t0_max)
                tau_new = np.random.uniform(low=tau_min, high=tau_max)

                tau_idx = np.digitize(tau_new, np.array(TAU_list)) - 1
                f0_idx = np.digitize(10**log_f0_new, np.array(F0_list[tau_idx])) - 1
                t0_idx = np.digitize(t0_new, np.array(T0_list[tau_idx])/(365.25*24*3600)) - 1

                tau_scan_new_point = tau_scan[tau_idx][f0_idx, t0_idx]

                if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
                    accepted = True

            #randomly select other parameters
            log10_h_new = pta.params[wavelet_indx[0,4]].sample()
            log10_h_cross_new = pta.params[wavelet_indx[0,5]].sample()
            phase0_new = pta.params[wavelet_indx[0,6]].sample()
            phase0_cross_new = pta.params[wavelet_indx[0,7]].sample()
            #if this is the first wavelet, draw sky location and polarization angle too
            if n_wavelet==0:
                cos_gwtheta_new = pta.params[wavelet_indx[0,0]].sample()
                psi_new = pta.params[wavelet_indx[0,1]].sample()
                gwphi_new = pta.params[wavelet_indx[0,2]].sample()
            #if this is not the first wavelet, copy sky location and ellipticity from existing wavelet(s)
            else:
                cos_gwtheta_new = np.copy(samples[j,i,2+wavelet_indx[0,0]])
                psi_new = np.copy(samples[j,i,2+wavelet_indx[0,1]])
                gwphi_new = np.copy(samples[j,i,2+wavelet_indx[0,2]])

            prior_ext = (pta.params[wavelet_indx[0,0]].get_pdf(cos_gwtheta_new) * pta.params[wavelet_indx[0,1]].get_pdf(psi_new) *
                         pta.params[wavelet_indx[0,2]].get_pdf(gwphi_new) *
                         pta.params[wavelet_indx[0,4]].get_pdf(log10_h_new) * pta.params[wavelet_indx[0,5]].get_pdf(log10_h_cross_new) *
                         pta.params[wavelet_indx[0,6]].get_pdf(phase0_new) * pta.params[wavelet_indx[0,7]].get_pdf(phase0_cross_new))

            samples_current = np.copy(samples[j,i,2:])

            new_point = np.copy(samples[j,i,2:])

            new_point[wavelet_indx[n_wavelet,0]] = cos_gwtheta_new
            new_point[wavelet_indx[n_wavelet,1]] = psi_new
            new_point[wavelet_indx[n_wavelet,2]] = gwphi_new
            new_point[wavelet_indx[n_wavelet,3]] = log_f0_new
            new_point[wavelet_indx[n_wavelet,4]] = log10_h_new
            new_point[wavelet_indx[n_wavelet,5]] = log10_h_cross_new
            new_point[wavelet_indx[n_wavelet,6]] = phase0_new
            new_point[wavelet_indx[n_wavelet,7]] = phase0_cross_new
            new_point[wavelet_indx[n_wavelet,8]] = t0_new
            new_point[wavelet_indx[n_wavelet,9]] = tau_new

            #Run the update to the QB_logL and QB_FP here

            log_L = QB_logl[j].M_N_RJ_helper(new_point, n_wavelet+1, n_glitch, adding = True, wavelet_change = True)
            log_acc_ratio = log_L*betas[j,i]

            new_point = correct_intrinsic(new_point, FPI, FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
            log_acc_ratio += QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   FPI.wave_le_highs, n_wavelet+1,n_glitch, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch) #I think this is the prior that needs n_wavelet shifted
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   FPI.wave_le_highs, n_wavelet,n_glitch, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch) #this should keep n_wavelet consistend with previous likelihood

            #apply normalization
            tau_scan_new_point_normalized = tau_scan_new_point/tau_scan_data['norm']

            acc_ratio = np.exp(log_acc_ratio)/prior_ext/tau_scan_new_point_normalized
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_wavelet==min_n_wavelet:
                acc_ratio *= 0.5
            if n_wavelet==max_n_wavelet-1:
                acc_ratio *= 2.0
            #accounting for n_wavelet prior
            acc_ratio *= n_wavelet_prior[int(n_wavelet)+1]/n_wavelet_prior[int(n_wavelet)] #not done in FPI at the moment because these parameters are irelevent to most steps

            if np.random.uniform()<=acc_ratio:

                samples[j,i+1,0] = n_wavelet+1
                samples[j,i+1,1] = n_glitch
                samples[j,i+1,2:] = new_point[:]

                a_yes[2,j] += 1
                log_likelihood[j,i+1] = log_L
                QB_logl[j].save_values(accept_new_step=True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

                accept_jump_arr[j] = 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[2,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
                QB_logl[j].save_values(accept_new_step=False, rj_jump = True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)


        elif n_wavelet==max_n_wavelet or (direction_decide>add_prob and n_wavelet!=min_n_wavelet):   #removing a wavelet----------------------------------------------------------

            if j==0: rj_record.append(-1)

            #choose which wavelet to remove
            remove_index = np.random.randint(n_wavelet)

            #copy samples slice
            samples_current = np.copy(samples[j,i,2:])#strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

            #Wavelet params from sample slice to remove
            samples_removed = np.copy(samples_current[wavelet_indx[remove_index,0]:wavelet_indx[remove_index,-1]+1])

            #Copy wavelet parameters from current sample slice
            wavelet_params_copied = np.copy(samples_current[wavelet_indx[0,0]:wavelet_indx[max_n_wavelet-1,-1]+1])

            #Make copy of current sample slice to modify later
            new_point = np.copy(samples_current)

            #delete wavelet params we don't need from samples slice
            wavelet_params_new = np.delete(wavelet_params_copied,list(range(remove_index*10,remove_index*10+10)))
            #Append new
            wavelet_params_new = np.append(wavelet_params_new, samples_removed)
            #arranged so removed wavelet is shifted to the end and all following wavelets are shifted over
            new_point[wavelet_indx[0,0]:wavelet_indx[max_n_wavelet-1,-1]+1] = np.copy(wavelet_params_new)


            log_L = QB_logl[j].M_N_RJ_helper(new_point, n_wavelet-1, n_glitch, remove_index = remove_index, wavelet_change = True)

            log_acc_ratio = log_L*betas[j,i]

            new_point = correct_intrinsic(new_point, FPI, FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
            log_acc_ratio += QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   FPI.wave_le_highs, n_wavelet-1, n_glitch, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch)

            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]

            log_acc_ratio += -QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   FPI.wave_le_highs, n_wavelet,n_glitch, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch)



            #getting tau_scan at old point
            tau_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,9]])
            f0_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,3]])
            t0_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,8]])

            tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
            f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
            t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

            tau_scan_old_point = tau_scan[tau_idx_old][f0_idx_old, t0_idx_old]

            #apply normalization
            tau_scan_old_point_normalized = tau_scan_old_point/tau_scan_data['norm']

            #getting external parameter priors
            log10_h_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,4]])
            log10_h_cross_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,5]])
            phase0_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,6]])
            phase0_cross_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,7]])
            cos_gwtheta_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,0]])
            gwphi_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,2]])
            psi_old = np.copy(samples[j, i, 2+wavelet_indx[remove_index,1]])

            prior_ext = (pta.params[wavelet_indx[0,0]].get_pdf(cos_gwtheta_old) * pta.params[wavelet_indx[0,1]].get_pdf(psi_old) *
                         pta.params[wavelet_indx[0,2]].get_pdf(gwphi_old) *
                         pta.params[wavelet_indx[0,4]].get_pdf(log10_h_old) * pta.params[wavelet_indx[0,5]].get_pdf(log10_h_cross_old) *
                         pta.params[wavelet_indx[0,6]].get_pdf(phase0_old) * pta.params[wavelet_indx[0,7]].get_pdf(phase0_cross_old))

            acc_ratio = np.exp(log_acc_ratio)*prior_ext*tau_scan_old_point_normalized
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_wavelet==min_n_wavelet+1:
                acc_ratio *= 2.0
            if n_wavelet==max_n_wavelet:
                acc_ratio *= 0.5
            #accounting for n_wavelet prior
            acc_ratio *= n_wavelet_prior[int(n_wavelet)-1]/n_wavelet_prior[int(n_wavelet)]

            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,0] = n_wavelet-1
                samples[j,i+1,1] = n_glitch
                samples[j,i+1,2:] = new_point[:]
                a_yes[2,j] += 1
                log_likelihood[j,i+1] = log_L
                QB_logl[j].save_values(accept_new_step=True)
                #FPI.n_wavelet = n_wavelet-1
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

                accept_jump_arr[j] = 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[2,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
                QB_logl[j].save_values(accept_new_step=False, rj_jump = True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
    return accept_jump_arr

def do_glitch_rj_move(n_chain, max_n_wavelet, max_n_glitch, n_glitch_prior, pta,
                      FPI, QB_logl, QB_Info, samples, i, betas, a_yes, a_no,
                      vary_white_noise, num_noise_params, glitch_tau_scan_data, log_likelihood,
                      wavelet_indx, glitch_indx):

    TAU_list = list(glitch_tau_scan_data['tau_edges'])
    F0_list = glitch_tau_scan_data['f0_edges']
    T0_list = glitch_tau_scan_data['t0_edges']

    #Keeps track of if the jump was accepted or rejected. The array is an array of
    accept_jump_arr = np.zeros(n_chain)

    for j in range(n_chain):

        n_wavelet = int(samples[j,i,0])
        n_glitch = int(samples[j,i,1])

        add_prob = 0.5 #same propability of addind and removing
        #decide if we add or remove a signal
        direction_decide = np.random.uniform()
        
        if n_glitch==0 or (direction_decide<add_prob and n_glitch!=max_n_glitch): #adding a glitch------------------------------------------------------
            #pick which pulsar to add a glitch to
            psr_idx = np.random.choice(len(pta.pulsars), p=glitch_tau_scan_data['psr_idx_proposal'])

            #load in the appropriate tau-scan
            tau_scan = glitch_tau_scan_data['tau_scan'+str(psr_idx)]

            tau_scan_limit = 0
            for TS in tau_scan:
                TS_max = np.nanmax(TS)
                if TS_max>tau_scan_limit:
                    tau_scan_limit = TS_max


            log_f0_max = float(pta.params[glitch_indx[0,0]]._typename.split('=')[2][:-1])
            log_f0_min = float(pta.params[glitch_indx[0,0]]._typename.split('=')[1].split(',')[0])
            t0_max = float(pta.params[glitch_indx[0,4]]._typename.split('=')[2][:-1])
            t0_min = float(pta.params[glitch_indx[0,4]]._typename.split('=')[1].split(',')[0])
            tau_max = float(pta.params[glitch_indx[0,5]]._typename.split('=')[2][:-1])
            tau_min = float(pta.params[glitch_indx[0,5]]._typename.split('=')[1].split(',')[0])

            accepted = False
            while accepted==False:
                log_f0_new = np.random.uniform(low=log_f0_min, high=log_f0_max)
                t0_new = np.random.uniform(low=t0_min, high=t0_max)
                tau_new = np.random.uniform(low=tau_min, high=tau_max)
                # print('glitch rj add tau min, tau_max: {}'.format(tau_min, tau_max))
                tau_idx = np.digitize(tau_new, np.array(TAU_list)) - 1
                f0_idx = np.digitize(10**log_f0_new, np.array(F0_list[tau_idx])) - 1
                t0_idx = np.digitize(t0_new, np.array(T0_list[tau_idx])/(365.25*24*3600)) - 1

                tau_scan_new_point = tau_scan[tau_idx][f0_idx, t0_idx]
                if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
                    accepted = True

            #randomly select phase and amplitude
            phase0_new = pta.params[glitch_indx[0,2]].sample()
            log10_h_new = pta.params[glitch_indx[0,1]].sample()

            prior_ext = pta.params[glitch_indx[0,2]].get_pdf(phase0_new) * pta.params[glitch_indx[0,1]].get_pdf(log10_h_new)

            samples_current = np.copy(samples[j, i, 2:])
            new_point = np.copy(samples[j,i,2:])

            new_point[glitch_indx[n_glitch,0]] = log_f0_new
            new_point[glitch_indx[n_glitch,1]] = log10_h_new
            new_point[glitch_indx[n_glitch,2]] = phase0_new
            new_point[glitch_indx[n_glitch,3]] = psr_idx
            new_point[glitch_indx[n_glitch,4]] = t0_new
            new_point[glitch_indx[n_glitch,5]] = tau_new

            log_L = QB_logl[j].M_N_RJ_helper(new_point, n_wavelet, n_glitch+1, adding = True, glitch_change = True)

            log_acc_ratio = log_L*betas[j,i]

            new_point = correct_intrinsic(new_point, FPI, FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
            log_acc_ratio += QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   FPI.wave_le_highs, n_wavelet, n_glitch+1, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch)

            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]

            log_acc_ratio += -QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   FPI.wave_le_highs, n_wavelet,n_glitch, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch)

            #apply normalization
            tau_scan_new_point_normalized = tau_scan_new_point/glitch_tau_scan_data['norm'+str(psr_idx)]

            acc_ratio = np.exp(log_acc_ratio)/prior_ext/tau_scan_new_point_normalized/glitch_tau_scan_data['psr_idx_proposal'][int(np.round(psr_idx))]
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_glitch==0:
                acc_ratio *= 0.5
            if n_glitch==max_n_glitch-1:
                acc_ratio *= 2.0
            #accounting for n_glitch prior
            acc_ratio *= n_glitch_prior[int(n_glitch)+1]/n_glitch_prior[int(n_glitch)]
            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,0] = n_wavelet
                samples[j,i+1,1] = n_glitch+1
                samples[j,i+1,2:] = new_point[:]
                a_yes[0,j] += 1
                log_likelihood[j,i+1] = log_L

                QB_logl[j].save_values(accept_new_step=True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
                accept_jump_arr[j] = 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
                QB_logl[j].save_values(accept_new_step=False, rj_jump = True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)


        elif n_glitch==max_n_glitch or (direction_decide>add_prob and n_glitch!=0):   #removing a glitch----------------------------------------------------------
            #choose which glitch to remove
            remove_index = np.random.randint(n_glitch)

            samples_current = np.copy(samples[j,i,2:])
            samples_removed = np.copy(samples_current[glitch_indx[remove_index,0]:glitch_indx[remove_index,-1]+1]) #copy of the glitch to remove
            glitch_params_coppied = np.copy(samples_current[glitch_indx[0,0]:glitch_indx[max_n_glitch-1,-1]+1]) #copy of all wavelet params
            new_point = np.copy(samples_current)

            glitch_params_new = np.delete(glitch_params_coppied,list(range(remove_index*6,remove_index*6+6)))
            glitch_params_new = np.append(glitch_params_new, samples_removed)

            #arranged so removed glitch is shifted to the end and all following wavelets are shifted over
            new_point[glitch_indx[0,0]:glitch_indx[max_n_glitch-1,-1]+1] = np.copy(glitch_params_new)


            log_L = QB_logl[j].M_N_RJ_helper(new_point, n_wavelet, n_glitch-1, remove_index = remove_index, glitch_change = True)

            log_acc_ratio = log_L*betas[j,i]

            new_point = correct_intrinsic(new_point, FPI, FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
            log_acc_ratio += QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   FPI.wave_le_highs, n_wavelet,n_glitch-1, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch)

            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]

            log_acc_ratio += -QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   FPI.wave_le_highs, n_wavelet,n_glitch, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch)
            #getting old parameters
            tau_old = samples[j,i,2+glitch_indx[remove_index,5]]
            f0_old = 10**samples[j,i,2+glitch_indx[remove_index,0]]
            t0_old = samples[j,i,2+glitch_indx[remove_index,4]]
            log10_h_old = samples[j,i,2+glitch_indx[remove_index,1]]
            phase0_old = samples[j,i,2+glitch_indx[remove_index,2]]

            #get old psr index and load in appropriate tau scan
            psr_idx_old = samples[j,i,2+glitch_indx[remove_index,3]]
            tau_scan_old = glitch_tau_scan_data['tau_scan'+str(int(np.round(psr_idx_old)))]
            tau_scan_limit_old = 0
            for TS in tau_scan_old:
                TS_max = np.nanmax(TS)
                if TS_max>tau_scan_limit_old:
                    tau_scan_limit_old = TS_max

            #getting tau_scan at old point
            tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
            f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
            t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1


            tau_scan_old_point = tau_scan_old[tau_idx_old][f0_idx_old, t0_idx_old]

            #apply normalization
            tau_scan_old_point_normalized = tau_scan_old_point/glitch_tau_scan_data['norm'+str(int(np.round(psr_idx_old)))]

            prior_ext = pta.params[glitch_indx[0,2]].get_pdf(phase0_old) * pta.params[glitch_indx[0,1]].get_pdf(log10_h_old)

            acc_ratio = np.exp(log_acc_ratio)*prior_ext*tau_scan_old_point_normalized*glitch_tau_scan_data['psr_idx_proposal'][int(np.round(psr_idx_old))]
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_glitch==1:
                acc_ratio *= 2.0
            if n_glitch==max_n_glitch:
                acc_ratio *= 0.5
            #accounting for n_glitch prior
            acc_ratio *= n_glitch_prior[int(n_glitch)-1]/n_glitch_prior[int(n_glitch)]

            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,0] = n_wavelet
                samples[j,i+1,1] = n_glitch-1
                samples[j,i+1,2:] = new_point[:]
                a_yes[0,j] += 1
                log_likelihood[j,i+1] = log_L
                QB_logl[j].save_values(accept_new_step=True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

                #jump accepted
                accept_jump_arr[j] = 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]

                QB_logl[j].save_values(accept_new_step=False, rj_jump = True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

    return accept_jump_arr

################################################################################
#
#NOISE MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN WHITE NOISE PARAMETERS)
#
################################################################################

def noise_jump(n_chain, max_n_wavelet, max_n_glitch, pta, FPI, QB_logl, QB_Info,
                    samples, i, betas, a_yes, a_no, eig_per_psr, per_puls_indx, per_puls_rn_indx, per_puls_wn_indx, all_noiseparam_idxs,
                    num_noise_params, vary_white_noise, vary_per_psr_rn, log_likelihood, wavelet_indx, glitch_indx, N_Noise_Params_changed, de_history, total_weight, DE_prob, fisher_prob, prior_draw_prob):

    total_weight = (DE_prob + fisher_prob + prior_draw_prob)

    #Keeps track of if the jump was accepted or rejected. The array is an array of
    accept_jump_arr = np.zeros(n_chain)

    for j in range(n_chain):
        n_wavelet = int(samples[j,i,0])
        n_glitch = int(samples[j,i,1])

        samples_current = np.copy(samples[j,i,2:])

        #Decide which jump to do
        which_jump = np.random.choice(3, p=[DE_prob/total_weight,
                                            fisher_prob/total_weight,
                                            prior_draw_prob/total_weight])

        if which_jump == 0:
            #perform DE jump
            if vary_white_noise and vary_per_psr_rn:
                ndim = len(all_noiseparam_idxs)
                new_point = DE_proposal(j, samples_current, de_history, all_noiseparam_idxs, ndim)

            if vary_white_noise and not vary_per_psr_rn:
                ndim = sum(len([pulsar_wn]) for pulsar_wn in per_puls_wn_indx)
                white_noise_idxs = [wn_idxs for psr_wn in per_puls_wn_indx for wn_idxs in [psr_wn]]
                new_point = DE_proposal(j, samples_current, de_history, white_noise_idxs, ndim)

            if vary_per_psr_rn and not vary_white_noise:
                ndim = sum(len([pulsar_rn]) for pulsar_rn in per_puls_rn_indx)
                red_noise_idxs = [rn_idxs for psr_rn in per_puls_rn_indx for rn_idxs in [psr_rn]]
                new_point = DE_proposal(j, samples_current, de_history, red_noise_idxs, ndim)

        elif which_jump == 1:
            #Pick wn or rn parameter eigenvectors to jump along
            jump_noise = np.zeros(eig_per_psr[j,0,:].shape)
            for nnn in range(N_Noise_Params_changed):
                jump_select = np.random.randint(eig_per_psr.shape[1])

                #randomly scale eigenvectors in each parameter
                jump_noise += eig_per_psr[j,jump_select,:]*np.random.normal()


            jump = np.zeros(samples_current.size)
            param_count = 0

            #Loop through all pulsars and pulsar noise params
            param_indexes = []

            #
            for ii in range(len(per_puls_indx)):
                for jj in range(len(per_puls_indx[ii])):
                    param_indexes.append(per_puls_indx[ii][jj])
                    #Jump through noise params (which should correspond to noise eigenvector indexes)
                    #if param_count < num_noise_params:
                    jump[per_puls_indx[ii][jj]] = jump_noise[param_count]
                    param_count += 1

            new_point = samples_current + jump

        elif which_jump == 2:
            #Pick random pulsar
            pulsar_idx = np.random.randint(len(per_puls_indx))
            #Choose 10 random noise parameters
            if len(per_puls_indx[pulsar_idx]) <= 10:
                pulsar_noise_idxs = per_puls_indx[pulsar_idx]
            if len(per_puls_indx[pulsar_idx]) > 10:
                #pick 5 random parameters for a pulsar
                pulsar_noise_idxs = np.random.choice(per_puls_indx[pulsar_idx], size=10, replace=False)
            #Draw random value for one pulsar for all its noise params
            prior_draws = []
            idx = []
            new_point = QB_FastPrior.get_sample_idxs(samples_current.copy(), pulsar_noise_idxs, FPI)

        new_point = correct_intrinsic(new_point, FPI, FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)
        new_log_prior = QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)

        if new_log_prior==-np.inf: #check if prior is -inf - reject step if it is
            samples[j,i+1,:] = samples[j,i,:]
            a_no[7,j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            continue

        log_L = QB_logl[j].get_lnlikelihood(new_point, vary_white_noise = vary_white_noise, vary_red_noise = vary_per_psr_rn)

        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += new_log_prior
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]

        log_acc_ratio += -QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                               FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                               FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                               FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                               FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                               FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                               FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                               FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                               FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                               FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                               FPI.wave_le_highs, n_wavelet,n_glitch, \
                                               FPI.max_n_wavelet, FPI.max_n_glitch)
        acc_ratio = np.exp(log_acc_ratio)

        if np.random.uniform()<=acc_ratio:

            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            #DE proposals
            if which_jump == 0:
                a_yes[7,j]+=1
            #Fisher proposals
            if which_jump == 1:
                a_yes[8,j]+=1
            #prior proposals
            if which_jump == 2:
                a_yes[9,j]+=1
            log_likelihood[j,i+1] = log_L

            QB_logl[j].save_values(accept_new_step=True, vary_white_noise = vary_white_noise, vary_red_noise = vary_per_psr_rn)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

            accept_jump_arr[j] = 1

        else:
            samples[j,i+1,:] = samples[j,i,:]
            #DE proposals
            if which_jump == 0:
                a_no[7,j]+=1
            #Fisher proposals
            if which_jump == 1:
                a_no[8,j]+=1
            #prior proposals
            if which_jump == 2:
                a_no[9,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            QB_logl[j].save_values(accept_new_step=False, vary_white_noise = vary_white_noise, vary_red_noise = vary_per_psr_rn)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
    
    return accept_jump_arr

################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, max_n_wavelet, max_n_glitch, pta, FPI, QB_logl, QB_Info, samples, i, betas, a_yes, a_no, swap_record, vary_white_noise, num_noise_params, log_likelihood, PT_hist, PT_hist_idx, n_fast_to_slow, save_every_n, QB_itteration, T_dynamic, T_dynamic_nu, T_dynamic_t0):
    #set up map to help keep track of swaps
    swap_map = list(range(n_chain))

    #get log_Ls from all the chains
    log_Ls = []
    for j in range(n_chain):
        log_Ls.append(log_likelihood[j,i])
    #loop through and propose a swap at each chain (starting from hottest chain and going down in T) and keep track of results in swap_map
    for swap_chain in reversed(range(n_chain-1)):
        log_acc_ratio = -log_Ls[swap_map[swap_chain]] * betas[swap_chain,i]
        log_acc_ratio += -log_Ls[swap_map[swap_chain+1]] * betas[swap_chain+1,i]
        log_acc_ratio += log_Ls[swap_map[swap_chain+1]] * betas[swap_chain,i]
        log_acc_ratio += log_Ls[swap_map[swap_chain]] * betas[swap_chain+1,i]

        acc_ratio = np.exp(log_acc_ratio)
        PT_hist[swap_chain,PT_hist_idx[0]%PT_hist.shape[1]] = np.minimum(acc_ratio, 1.0)

        if np.random.uniform()<=acc_ratio:
            swap_map[swap_chain], swap_map[swap_chain+1] = swap_map[swap_chain+1], swap_map[swap_chain]
            a_yes[4,swap_chain]+=1
            swap_record[i] = swap_chain
        else:
            a_no[4,swap_chain]+=1


    #Implementaiton of dynamic temperature ladder
    #Only update the temperatures 10 PT steps to let the last change of temperatures take some effect.
    if PT_hist_idx[0] % 10 == 0 and PT_hist_idx[0] != 0 and T_dynamic:
        #get which itteraiton of noise jump we are on
        noise_itteration = QB_itteration / n_fast_to_slow

        #Work in terms of temperatures because it is conceptually easier to think about
        tempuratures = 1/betas[:,i%save_every_n]

        #updates temperature ladder. Please reference "dynamic temperature ladder Quick Burst" in the docs directory to understand.
        for j in range(1,n_chain-1):
            #compute the kappa which essentially scales how fast the temperatures change
            k = 1/T_dynamic_nu * (T_dynamic_t0/(noise_itteration+T_dynamic_t0))

            #calculate the current value of s. We need to put a condition in here that handles if there is a negative in the log. Clearly a negative in the log will return a nan and basically crash the program.
            curr_s = np.log(tempuratures[j] - tempuratures[j-1])
            if np.isnan(curr_s):
                continue
            
            #calculate the N I defined in the document referenced above
            my_n = curr_s + k*(np.nanmean(PT_hist[j-1,:])- np.nanmean(PT_hist[j,:]))
            tempuratures[j] = tempuratures[j-1]+np.exp(my_n)

            #If it ever happens we change a temperature such that a lower temperature becomes higher than a higher temperature that is an issue. If that happens we force the higher temperature to be higher than the lower again.
            if tempuratures[j] > tempuratures[j+1]:
                tempuratures[j+1] = tempuratures[j] + 0.01

        #Actually set the new temperatures.
        for j in range(n_chain):
            betas[j,(i%save_every_n)+1] = 1/tempuratures[j]


    PT_hist_idx += 1
    QB_logl_map = []
    QB_Info_map = []
    for j in range(n_chain):

        QB_logl_map.append(QB_logl[swap_map[j]])
        QB_Info_map.append(QB_Info[swap_map[j]])

    #loop through the chains and record the new samples and log_Ls
    for j in range(n_chain):
        QB_logl[j] = QB_logl_map[j]
        QB_Info[j] = QB_Info_map[j]
        samples[j,i+1,:] = samples[swap_map[j],i,:]
        log_likelihood[j,i+1] = log_likelihood[swap_map[j],i]

################################################################################
#
#FISHER EIGENVALUE CALCULATION
#
################################################################################
def get_fisher_eigenvectors(params, pta, QB_FP, QB_logl, T_chain=1, epsilon=1e-2, n_sources=1, dim=10, array_index=None, use_prior=False, flag = False, vary_intrinsic_noise = False, vary_white_noise = False, vary_psr_red_noise = False, vary_rn = False):
    n_source=n_sources # this needs to not be used for the non-wavelet/glitch indexing (set to 1)
    eig = []
    index_rows = len(array_index)
    if flag or vary_rn:
        fisher = np.zeros((n_source,dim,dim))
    elif vary_white_noise or vary_psr_red_noise:
        offset_array = []
        dim = 0
        for psr in range(len(array_index)):
            offset_array.append(len(array_index[psr]))
            dim += len(array_index[psr])
        fisher = np.zeros((n_source, dim, dim))
    else:
        fisher = np.zeros((n_source,dim*index_rows,dim*index_rows))
    #lnlikelihood at specified point
    if use_prior:
        nn = QB_logl.get_lnlikelihood(params,no_step = True) + QB_FP.get_lnprior(params)
    else:
        nn = QB_logl.get_lnlikelihood(params,no_step = True)

    print('fish n_source {0}: dim {1}: params len {2}: array_index {3}'.format(n_source, dim, len(params), array_index))
    #flag = True if doing wavelet/glitch fisher matrix calculations
    if flag == True:
        for k in range(n_source):
            #calculate diagonal elements
            for i in range(dim):
                #create parameter vectors with +-epsilon in the ith component
                paramsPP = np.copy(params)
                paramsMM = np.copy(params)
                #changing wavelet or glitch params only
                paramsPP[array_index[k,i]] += 2*epsilon
                paramsPP[array_index[k,i]] += 2*epsilon
                #otherwise, change all other params by 2*epsilon

                #lnlikelihood at +-epsilon positions
                if use_prior:
                    pp = QB_logl.get_lnlikelihood(paramsPP,no_step = True) + QB_FP.get_lnprior(paramsPP)
                    mm = QB_logl.get_lnlikelihood(paramsMM,no_step = True) + QB_FP.get_lnprior(paramsMM)
                else:
                    pp = QB_logl.get_lnlikelihood(paramsPP,no_step = True)
                    mm = QB_logl.get_lnlikelihood(paramsMM,no_step = True)

                #calculate diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                fisher[k,i,i] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

            #calculate off-diagonal elements
            for i in range(dim):
                for j in range(i+1,dim):
                    #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
                    paramsPP = np.copy(params)
                    paramsMM = np.copy(params)
                    paramsPM = np.copy(params)
                    paramsMP = np.copy(params)
                    #looping through wavelets or glitch terms only and perturbing by small amount
                    paramsPP[array_index[k,i]] += epsilon
                    paramsPP[array_index[k,j]] += epsilon
                    paramsMM[array_index[k,i]] -= epsilon
                    paramsMM[array_index[k,j]] -= epsilon
                    paramsPM[array_index[k,i]] += epsilon
                    paramsPM[array_index[k,j]] -= epsilon
                    paramsMP[array_index[k,i]] -= epsilon
                    paramsMP[array_index[k,j]] += epsilon

                    #lnlikelihood at those positions
                    if use_prior:
                        pp = QB_logl.get_lnlikelihood(paramsPP,no_step = True) + QB_FP.get_lnprior(paramsPP)
                        mm = QB_logl.get_lnlikelihood(paramsMM,no_step = True) + QB_FP.get_lnprior(paramsMM)
                        pm = QB_logl.get_lnlikelihood(paramsPM,no_step = True) + QB_FP.get_lnprior(paramsPM)
                        mp = QB_logl.get_lnlikelihood(paramsMP,no_step = True) + QB_FP.get_lnprior(paramsMP)
                    else:
                        pp = QB_logl.get_lnlikelihood(paramsPP,no_step = True)
                        mm = QB_logl.get_lnlikelihood(paramsMM,no_step = True)
                        pm = QB_logl.get_lnlikelihood(paramsPM,no_step = True)
                        mp = QB_logl.get_lnlikelihood(paramsMP,no_step = True)

                    #calculate off-diagonal elements of the Hessian from a central finite element scheme
                    #note the minus sign compared to the regular Hessian
                    fisher[k,i,j] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                    fisher[k,j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)

            #correct for the given temperature of the chain (11/16: Moved temp scaling to main loop)
            fisher = fisher/T_chain

            try:
                #Filter nans and infs and replace them with 1s
                #this will imply that we will set the eigenvalue to 100 a few lines below
                #UPDATED so that 0s are also replaced with 1.0
                FISHER = np.where(np.isfinite(fisher[k,:,:]) * (fisher[k,:,:]!=0.0), fisher[k,:,:], 1.0)
                if not np.array_equal(FISHER, fisher[k,:,:]):
                    print("Changed some nan elements in the Fisher matrix to 1.0")

                #Find eigenvalues and eigenvectors of the Fisher matrix
                w, v = np.linalg.eig(FISHER)

                #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
                eig_limit = 1.0

                W = np.where(np.abs(w)>eig_limit, w, eig_limit)

                eig.append( (np.sqrt(1.0/np.abs(W))*v).T )

            except:
                print("An Error occured in the eigenvalue calculation")
                #eig.append( np.array(False) )

    #Run this if not doing wavelet/glitch stuff
    elif vary_psr_red_noise or vary_white_noise:
        #diagonal terms in fisher matrices
        for n in range(index_rows):
            dim = len(array_index[n])
            offset = sum(offset_array[:n])

            for i in range(dim):
                #create parameter vectors with +-epsilon in the ith component
                paramsPP = np.copy(params)
                paramsMM = np.copy(params)
                #changing wavelet or glitch params only

                paramsPP[array_index[n][i]] += 2*epsilon
                paramsPP[array_index[n][i]] += 2*epsilon
                #otherwise, change all other params by 2*epsilon

                #lnlikelihood at +-epsilon positions
                if use_prior:
                    pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsPP)
                    mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsMM)
                else:
                    pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True)
                    mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True)

                #calculate diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                if len(QB_logl.psrs) == 1: #if only 1 pulsar, run this
                    fisher[0,i+offset,i+offset] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)
                else: #else, run as normal
                    fisher[1,i+offset,i+offset] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)
        for n in range(index_rows):
            #calculate off-diagonal elements
            dim = len(array_index[n])
            offset = sum(offset_array[:n])
            for i in range(dim):
                for j in range(i+1,dim):
                    #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
                    paramsPP = np.copy(params)
                    paramsMM = np.copy(params)
                    paramsPM = np.copy(params)
                    paramsMP = np.copy(params)

                    paramsPP[array_index[n][i]] += epsilon
                    paramsPP[array_index[n][j]] += epsilon
                    paramsMM[array_index[n][i]] -= epsilon
                    paramsMM[array_index[n][j]] -= epsilon
                    paramsPM[array_index[n][i]] += epsilon
                    paramsPM[array_index[n][j]] -= epsilon
                    paramsMP[array_index[n][i]] -= epsilon
                    paramsMP[array_index[n][j]] += epsilon

                    #lnlikelihood at those positions
                    if use_prior:
                        pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsPP)
                        mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsMM)
                        pm = QB_logl.get_lnlikelihood(paramsPM, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsPM)
                        mp = QB_logl.get_lnlikelihood(paramsMP, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsMP)
                    else:
                        pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True)
                        mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True)
                        pm = QB_logl.get_lnlikelihood(paramsPM, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True)
                        mp = QB_logl.get_lnlikelihood(paramsMP, vary_white_noise = vary_white_noise, vary_red_noise = vary_psr_red_noise, no_step = True)

                    #calculate off-diagonal elements of the Hessian from a central finite element scheme
                    #note the minus sign compared to the regular Hessian
                    if len(QB_logl.psrs) == 1: #if only 1 pulsar, run this
                        fisher[0,i+offset,j+offset] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                        fisher[0,j+offset,i+offset] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                    else: #else, run as normal
                        fisher[1,i+offset,j+offset] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                        fisher[1,j+offset,i+offset] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
        try:
            #Filter nans and infs and replace them with 1s
            #this will imply that we will set the eigenvalue to 100 a few lines below
            #UPDATED so that 0s are also replaced with 1.0
            if len(QB_logl.psrs) == 1: #if only 1 pulsar, run this
                FISHER = np.where(np.isfinite(fisher[0,:,:]) * (fisher[0,:,:]!=0.0), fisher[0,:,:], 1.0)
                if not np.array_equal(FISHER, fisher[0,:,:]):
                    print("Changed some nan elements in the Fisher matrix to 1.0")
            else: #else, run as normal
                FISHER = np.where(np.isfinite(fisher[1,:,:]) * (fisher[1,:,:]!=0.0), fisher[1,:,:], 1.0)
                if not np.array_equal(FISHER, fisher[1,:,:]):
                    print("Changed some nan elements in the Fisher matrix to 1.0")

            #Find eigenvalues and eigenvectors of the Fisher matrix
            w, v = np.linalg.eig(FISHER)

            #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
            eig_limit = 1.0

            W = np.where(np.abs(w)>eig_limit, w, eig_limit)

            eig_temp = (np.sqrt(1.0/np.abs(W))*v).T #the eigenvalues we would normally output

            for iii in range(eig_temp.shape[0]):
                eig_max = np.max(eig_temp[iii])
                eig_temp[iii] = np.where(eig_temp[iii] < eig_max/10, 0, eig_temp[iii]) #set non-max small values to 0

            eig.append(eig_temp)

        except:
            print("An Error occured in the eigenvalue calculation")


    elif vary_rn:
        print('CURN is varied! Fisher steps!')
        #diagonal terms in fisher matrices
        for n in range(index_rows):
            #create parameter vectors with +-epsilon in the ith component
            paramsPP = np.copy(params)
            paramsMM = np.copy(params)
            #changing CRN terms only

            #Only two elements in one row (log10_A, log10_gamma).
            #Cheat to keep notation for only needing one loop for CRN.
            paramsPP[array_index[n]] += 2*epsilon
            paramsPP[array_index[n]] += 2*epsilon
            #otherwise, change all other params by 2*epsilon

            #lnlikelihood at +-epsilon positions
            if use_prior:
                pp = QB_logl.get_lnlikelihood(paramsPP, vary_red_noise = vary_rn, no_step = True) + QB_FP.get_lnprior(paramsPP)
                mm = QB_logl.get_lnlikelihood(paramsMM, vary_red_noise = vary_rn, no_step = True) + QB_FP.get_lnprior(paramsMM)
            else:
                pp = QB_logl.get_lnlikelihood(paramsPP, vary_red_noise = vary_rn, no_step = True)
                mm = QB_logl.get_lnlikelihood(paramsMM, vary_red_noise = vary_rn, no_step = True)

            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            fisher[0,n,n] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

        print('Calculating off-diagonal CURN fisher elements')
        for i in range(index_rows):
            #calculate off-diagonal elements
            for j in range(i+1, dim):
                #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
                paramsPP = np.copy(params)
                paramsMM = np.copy(params)
                paramsPM = np.copy(params)
                paramsMP = np.copy(params)

                paramsPP[array_index[i]] += epsilon
                paramsPP[array_index[j]] += epsilon
                paramsMM[array_index[i]] -= epsilon
                paramsMM[array_index[j]] -= epsilon
                paramsPM[array_index[i]] += epsilon
                paramsPM[array_index[j]] -= epsilon
                paramsMP[array_index[i]] -= epsilon
                paramsMP[array_index[j]] += epsilon

                #lnlikelihood at those positions
                if use_prior:
                    pp = QB_logl.get_lnlikelihood(paramsPP, vary_red_noise = vary_rn, no_step = True) + QB_FP.get_lnprior(paramsPP)
                    mm = QB_logl.get_lnlikelihood(paramsMM, vary_red_noise = vary_rn, no_step = True) + QB_FP.get_lnprior(paramsMM)
                    pm = QB_logl.get_lnlikelihood(paramsPM, vary_red_noise = vary_rn, no_step = True) + QB_FP.get_lnprior(paramsPM)
                    mp = QB_logl.get_lnlikelihood(paramsMP, vary_red_noise = vary_rn, no_step = True) + QB_FP.get_lnprior(paramsMP)
                else:
                    pp = QB_logl.get_lnlikelihood(paramsPP, vary_red_noise = vary_rn, no_step = True)
                    mm = QB_logl.get_lnlikelihood(paramsMM, vary_red_noise = vary_rn, no_step = True)
                    pm = QB_logl.get_lnlikelihood(paramsPM, vary_red_noise = vary_rn, no_step = True)
                    mp = QB_logl.get_lnlikelihood(paramsMP, vary_red_noise = vary_rn, no_step = True)

                #calculate off-diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                fisher[0,i,j] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                fisher[0,j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
        #correct for the given temperature of the chain (11/16: Moved temp scaling in main loop)
        fisher = fisher/T_chain

        try:
            #Filter nans and infs and replace them with 1s
            #this will imply that we will set the eigenvalue to 100 a few lines below
            #UPDATED so that 0s are also replaced with 1.0
            FISHER = np.where(np.isfinite(fisher[1,:,:]) * (fisher[1,:,:]!=0.0), fisher[1,:,:], 1.0)
            if not np.array_equal(FISHER, fisher[1,:,:]):
                print("Changed some nan elements in the Fisher matrix to 1.0")

            #Find eigenvalues and eigenvectors of the Fisher matrix
            w, v = np.linalg.eig(FISHER)

            #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
            eig_limit = 1.0

            W = np.where(np.abs(w)>eig_limit, w, eig_limit)

            eig.append( (np.sqrt(1.0/np.abs(W))*v).T )

        except:
            print("An Error occured in the eigenvalue calculation")


    return np.array(eig)


################################################################################
#
#FUNCTION TO EASILY SET UP A LIST OF PTA OBJECTS
#
################################################################################
def get_pta(pulsars, vary_white_noise=True, include_equad = False, include_ecorr = False, include_efac = False, wn_backend_selection=False, noisedict=None, include_rn=True, vary_rn=True, include_per_psr_rn=False, vary_per_psr_rn=False, max_n_wavelet=1, efac_start=1.0, rn_amp_prior='uniform', rn_log_amp_range=[-18,-11], rn_params=[-14.0,1.0], wavelet_amp_prior='uniform', wavelet_log_amp_range=[-18,-11], per_psr_rn_amp_prior='uniform', per_psr_rn_log_amp_range=[-18,-11], equad_range = [-8.5, -5], ecorr_range = [-8.5, -5], prior_recovery=False, max_n_glitch=1, glitch_amp_prior='uniform', glitch_log_amp_range=[-18, -11], t0_min=0.0, t0_max=10.0, f0_min=3.5e-9, f0_max=1e-7, tau_min=0.2, tau_max=5.0, TF_prior=None, use_svd_for_timing_gp=True, tref=53000*86400):
    '''
    Function to make PTA including deterministic models.

    :returns pta, QB_FP (prior object), QB_FPI (prior info object), glitch_indx (noise transient parameter indexes),
             wavelet_indx (GW signal wavelet parameter indexes), per_puls_indx (intrinsic pulsar noise indexes),
             rn_indx (CURN parameter indexes), num_per_puls_param_list (number of params per pulsar)

    :param pulsars:
        Pulsar pickle file
    :param vary_white_noise:
        If True, vary any included WN models in PTA model. [False] by default.
    :param include_equad:
        If True, include equad WN models in PTA. Currently only works with t2equad. [False] by default.
    :param include_ecorr:
        If True, include correlated WN models in PTA. [False] by default.
    :param include_efac:
        If True, include efac WN models in PTA. [False] by default.
    :param wn_backend_selection:
        If True, use enterprise Selection based on backend. Usually use True for real data, False for simulated data. [False] by default.
    :param noisedict:
        Parameter noise dictionary for model parameters. Can be either a filepath or a dictionary. [None] by default.
    :param include_rn:
        If True, include CURN parameters in PTA model. If vary_rn = True, these parameters will be varied. [False] by default.
    :param vary_rn:
        If True, CURN parameters will be varied in PTA model. [False] by default.
    :param include_per_psr_rn:
        If True, intrinsic pulsar red noise models will be included in PTA. [False] by default.
    :param vary_per_psr_rn:
        If True, intrinsic pulsar red noise will be varied. [False] by default.
    :param max_n_wavelet:
        Maximum number of GW signal wavelets to include in PTA model.
    :param efac_start: NOT YET IMPLEMENTED
        If vary_white_noise = True, set initial sample for efac parameters to efac_start. [None] by default.
    :param rn_amp_prior:
        CURN amplitude prior. Choices can be ['uniform', 'log_uniform']. ['uniform'] by default.
    :param rn_log_amp_range:
        CURN amplitude prior range. [-18, -11] by default.
    :param rn_params:
        If CURN parameters are fixed, rn_params will set the amplitude and spectral index. rn_params[0] sets
        the amplitude, while rn_params[1] sets the spectral index. [-13.0, 1] by default.
    :param wavelet_amp_prior:
        GW signal wavelet prior on log10_h and log10_hcross. Choice can be ['uniform', 'log_uniform']. ['uniform'] by default.
    :param wavelet_log_amp_range:
        GW signal wavelet amplitude prior range. [-18, -11] by default.
    :param per_psr_rn_amp_prior:
        Intrinsic pulsar RN amplitude prior. Choices can be ['uniform', 'log_uniform']. ['uniform'] by default.
    :param per_psr_rn_log_amp_range:
        Intrinsic pulsar RN amplitude prior range: [-18, -11] by default.
    :param equad_range:
        If include_equad = True and vary_equad = True, equad_range sets the prior bounds on equad parameters. [-8.5, -5] by default.
    :param ecorr_range:
        If include_ecorr = True and vary_ecorr = True, ecorr_range sets the prior bounds on ecorr parameters. [-8.5, -5] by default.
    :param prior_recovery:
        If True, return 1 for the likelihood for every step. Parameter recovery should return the specified priors. [False] by default.
    :param max_n_glitch:
        Max number of noise transient wavelets allowed in PTA model. [1] by default.
    :param glitch_amp_prior:
        Prior on noise transient wavelet amplitudes. Choices can be ['uniform', 'log_uniform']. ['uniform'] by default.
    :param glitch_log_amp_range:
        Noise transient wavelet amplitude prior range. [-18, -11] by default.
    :param t0_min:
        The minimum epoch time with reference to the beginning of the data set.
    :param t0_max:
        The maximum epoch time for the data set.
    :param f0_min:
        Lower bound on GW signal wavelet and noise transient wavelet frequency in Hz. [3.5e-9] by default.
    :param f0_max:
        Upper bound on GW signal wavelet and noise transient wavelet frequency in Hz. [1e-7] by default.
    :param tau_min:
        Lower bound on GW signal wavelet and noise transient wavelet width in years. [0.2] by default.
    :param tau_max:
        Upper bound on GW signal wavelet and noise transient wavelet width in years. [5] by default.
    :param TF_prior:
        ...
    :param use_svd_for_timing_gp:
        If True, use SVD decomposition for timing model parameter matrix M. [True] by default.
    :param tref:
        Reference time for the beginning of the data set. Given in seconds. [50000*86400] by default.
    '''


    #setting up base model

    if vary_white_noise:
        #Include constant efac in both places, in case we want to fix efac but vary other things.
        if include_efac:
            efac = parameter.Uniform(0.01, 10.0)
        if include_equad:
            equad = parameter.Uniform(equad_range[0], equad_range[1])
        if include_ecorr:
            ecorr = parameter.Uniform(ecorr_range[0], ecorr_range[1])

    else:
        if include_efac:
            efac = parameter.Constant(efac_start)
        if include_equad:
            equad = parameter.Constant()
        if include_ecorr:
            ecorr = parameter.Constant()

    if wn_backend_selection:
        selection = selections.Selection(selections.by_backend)
        if include_equad:
            ef = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=selection)
        else:
            ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        wn = ef
        if include_ecorr:
            #ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
            ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection, name='')
            wn += ec
    else:
        if include_equad:
            ef = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad)
        else:
            ef = white_signals.MeasurementNoise(efac = efac)
        wn = ef
        if include_ecorr:
            #ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr)
            ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, name='')
            wn += ec
    tm = gp_signals.TimingModel(use_svd=use_svd_for_timing_gp)

    #adding per psr RN if included
    if include_per_psr_rn:
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)

        if vary_per_psr_rn:
            if per_psr_rn_amp_prior == 'uniform':
                log10_A = parameter.LinearExp(per_psr_rn_log_amp_range[0], per_psr_rn_log_amp_range[1])
            elif per_psr_rn_amp_prior == 'log-uniform':
                log10_A = parameter.Uniform(per_psr_rn_log_amp_range[0], per_psr_rn_log_amp_range[1])

            gamma = parameter.Uniform(0, 7)
        else:
            log10_A = parameter.Constant()
            gamma = parameter.Constant()

        #This should be setting amplitude and gamma to default values
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        per_psr_rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan)

    #adding red noise if included
    if include_rn:
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)

        if vary_rn:
            #If varying CURN, need to have param names match in noise dictionary. Otherwise, will get error.
            amp_name = 'gw_crn_log10_A'
            if rn_amp_prior == 'uniform':
                log10_Arn = parameter.LinearExp(rn_log_amp_range[0], rn_log_amp_range[1])(amp_name)
            elif rn_amp_prior == 'log-uniform':
                log10_Arn = parameter.Uniform(rn_log_amp_range[0], rn_log_amp_range[1])(amp_name)
            gam_name = 'gw_crn_gamma'
            gamma_rn = parameter.Uniform(0, 7)(gam_name)
            pl = utils.powerlaw(log10_A=log10_Arn, gamma=gamma_rn)
            rn = gp_signals.FourierBasisGP(spectrum=pl, coefficients=False, components=30, Tspan=Tspan,
                                           modes=None, name='com_rn')
        else:
            #Why these values for the common process? rn_params is hard coded in run_bhb() as rn_params = [-13.0, 1.0]
            amp_name = 'gw_crn_log10_A'
            log10_A = parameter.Constant(rn_params[0])(amp_name)
            gam_name = 'gw_crn_gamma'
            gamma = parameter.Constant(rn_params[1])(gam_name)
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan, name = 'com_rn')

    #wavelet models
    wavelets = []
    for i in range(max_n_wavelet):
        log10_f0 = parameter.Uniform(np.log10(f0_min), np.log10(f0_max))("wavelet_"+str(i)+'_'+'log10_f0')
        cos_gwtheta = parameter.Uniform(-1, 1)("wavelet_"+ str(i)+'_'+'cos_gwtheta')
        gwphi = parameter.Uniform(0, 2*np.pi)("wavelet_" + str(i)+'_'+'gwphi')
        psi = parameter.Uniform(0, np.pi)("wavelet_" + str(i)+'_'+'gw_psi')
        phase0 = parameter.Uniform(0, 2*np.pi)("wavelet_" + str(i)+'_'+'phase0')
        phase0_cross = parameter.Uniform(0, 2*np.pi)("wavelet_" + str(i)+'_'+'phase0_cross')
        tau = parameter.Uniform(tau_min, tau_max)("wavelet_" + str(i)+'_'+'tau')
        t0 = parameter.Uniform(t0_min, t0_max)("wavelet_" + str(i)+'_'+'t0')
        if wavelet_amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(wavelet_log_amp_range[0], wavelet_log_amp_range[1])("wavelet_" + str(i)+'_'+'log10_h')
            log10_h_cross = parameter.Uniform(wavelet_log_amp_range[0], wavelet_log_amp_range[1])("wavelet_" + str(i)+'_'+'log10_h_cross')
        elif wavelet_amp_prior == 'uniform':
            log10_h = parameter.LinearExp(wavelet_log_amp_range[0], wavelet_log_amp_range[1])("wavelet_" + str(i)+'_'+'log10_h')
            log10_h_cross = parameter.LinearExp(wavelet_log_amp_range[0], wavelet_log_amp_range[1])("wavelet_" + str(i)+'_'+'log10_h_cross')
        else:
            print("Wavelet amplitude prior of {0} not available".format(wavelet_amp_prior))
        wavelet_wf = models.wavelet_delay(cos_gwtheta = cos_gwtheta, gwphi = gwphi, log10_h = log10_h, log10_h2 = log10_h_cross,
                                          tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0, phase02 = phase0_cross,
                                          epsilon = None, psi = psi, tref = tref)
        wavelets.append(deterministic_signals.Deterministic(wavelet_wf, name='wavelet'+str(i)))
    #glitch models
    glitches = []
    for i in range(max_n_glitch):
        log10_f0 = parameter.Uniform(np.log10(f0_min), np.log10(f0_max))("Glitch_"+str(i)+'_'+'log10_f0')
        phase0 = parameter.Uniform(0, 2*np.pi)("Glitch_"+str(i)+'_'+'phase0')
        tau = parameter.Uniform(tau_min, tau_max)("Glitch_"+str(i)+'_'+'tau')
        t0 = parameter.Uniform(t0_min, t0_max)("Glitch_"+str(i)+'_'+'t0')
        psr_idx = parameter.Uniform(-0.5, len(pulsars)-0.5)("Glitch_"+str(i)+'_'+'psr_idx')
        if glitch_amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(glitch_log_amp_range[0], glitch_log_amp_range[1])("Glitch_"+str(i)+'_'+'log10_h')
        elif glitch_amp_prior == 'uniform':
            log10_h = parameter.LinearExp(glitch_log_amp_range[0], glitch_log_amp_range[1])("Glitch_"+str(i)+'_'+'log10_h')
        else:
            print("Glitch amplitude prior of {0} not available".format(glitch_amp_prior))
        glitch_wf = models.glitch_delay(log10_h = log10_h, tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0, tref=tref,
                                        psr_float_idx = psr_idx, pulsars=pulsars)
        glitches.append(deterministic_signals.Deterministic(glitch_wf, name='Glitch'+str(i) ))
    pta = []

    #Need to make sure to add red noise if included
    s =  tm + wn

    #Check what to include in model based on flags
    if include_per_psr_rn:
        s += per_psr_rn
    if include_rn:
        s += rn
    for i in range(max_n_glitch):
        s += glitches[i]
    for i in range(max_n_wavelet):
        s += wavelets[i]
    model = []
    for p in pulsars:
        model.append(s(p))

    #If noisedict is filepath, do this.
    #elif noisedict is not None:
    if noisedict is not None:
        if isinstance(noisedict, str):
            with open(noisedict, 'r') as fp: #rb
                noisedict = json.load(fp)
                pta_load = signal_base.PTA(model)
                pta_load.set_default_params(noisedict)
            if TF_prior is None:
                pta = pta_load
            else:
                pta = get_tf_prior_pta(pta, TF_prior, n_wavelet)
        #if noisedict is dictionary, do this.
        else:
            pta_load = signal_base.PTA(model)
            pta_load.set_default_params(noisedict)
            if TF_prior is None:
                pta = pta_load
            else:
                pta = get_tf_prior_pta(pta, TF_prior, n_wavelet)
    else:
        if TF_prior is None:
            pta = signal_base.PTA(model)
        else:
            pta = get_tf_prior_pta(signal_base.PTA(model), TF_prior, n_wavelet)

    #Parsing through parameters in pta -> able to grab indexes for reference later
    #Will (hopefully) be easier to read the code this way.
    key_list = pta.param_names
    glitch_indx = np.zeros((max_n_glitch,6), dtype = 'int')
    wavelet_indx = np.zeros((max_n_wavelet,10), dtype = 'int')
    #glitch models
    for i in range(max_n_glitch):
        glitch_indx[i,0] = key_list.index('Glitch_'+str(i)+'_log10_f0')
        glitch_indx[i,1] = key_list.index('Glitch_'+str(i)+'_log10_h')
        glitch_indx[i,2] = key_list.index('Glitch_'+str(i)+'_phase0')
        glitch_indx[i,3] = key_list.index('Glitch_'+str(i)+'_psr_idx')
        glitch_indx[i,4] = key_list.index('Glitch_'+str(i)+'_t0')
        glitch_indx[i,5] = key_list.index('Glitch_'+str(i)+'_tau')
    #wavelet models
    for j in range(max_n_wavelet):
        wavelet_indx[j,0] = key_list.index('wavelet_'+str(j)+'_cos_gwtheta')
        wavelet_indx[j,1] = key_list.index('wavelet_'+str(j)+'_gw_psi')
        wavelet_indx[j,2] = key_list.index('wavelet_'+str(j)+'_gwphi')
        wavelet_indx[j,3] = key_list.index('wavelet_'+str(j)+'_log10_f0')
        wavelet_indx[j,4] = key_list.index('wavelet_'+str(j)+'_log10_h')
        wavelet_indx[j,5] = key_list.index('wavelet_'+str(j)+'_log10_h_cross')
        wavelet_indx[j,6] = key_list.index('wavelet_'+str(j)+'_phase0')
        wavelet_indx[j,7] = key_list.index('wavelet_'+str(j)+'_phase0_cross')
        wavelet_indx[j,8] = key_list.index('wavelet_'+str(j)+'_t0')
        wavelet_indx[j,9] = key_list.index('wavelet_'+str(j)+'_tau')

    # number of wn params for each pulsar
    num_per_puls_param_list = []
    #List of lists of all wn/rn params per pulsar
    per_puls_indx = []
    #List of wn params
    per_puls_wn_indx = []
    #List on intrinsic rn params
    per_puls_rn_indx = []
    #All noise param indexes not separated by pulsar
    all_noiseparam_idxs = []
    #For each pulsar
    for i in range(len(pulsars)):
        param_list = pta.pulsarmodels[i].param_names
        psr_noise_indx = []
        #Search through all parameters to get indexes for rn and wn params for each pulsar
        for ct, par in enumerate(param_list):
            #Skip common rn terms
            if pulsars[i].name in par:
                if 'ecorr' in par or 'efac' in par or 'equad' in par or 'log10_A' in par or 'gamma' in par:
                    #get indexes for each pulsar from overall pta params
                    psr_noise_indx.append(key_list.index(par))
                    all_noiseparam_idxs.append(key_list.index(par))
                    #get indexes for each pulsar for white noise from pta params
                    if 'log10_A' in par or 'gamma' in par:
                        per_puls_rn_indx.append(key_list.index(par))
                    #get indexes for each pulsar for red noise from pta params
                    if 'ecorr' in par or 'efac' in par or 'equad' in par:
                        per_puls_wn_indx.append(key_list.index(par))

        #append to overall list of lists
        per_puls_indx.append(psr_noise_indx)
        num_per_puls_param_list.append(len(psr_noise_indx))
    print('Number of params per pulsar: ', num_per_puls_param_list)

    #Generate the lnPrior object for this PTA
    QB_FP = QB_FastPrior.FastPrior(pta, pulsars)
    QB_FPI = QB_FastPrior.get_FastPriorInfo(pta, pulsars, max_n_glitch, max_n_wavelet)

    rn_indx = np.zeros((2), dtype = 'int')
    if vary_rn:
        rn_indx[0] = key_list.index('gw_crn_gamma')
        rn_indx[1] = key_list.index('gw_crn_log10_A')

    # print(pta.summary())
    return pta, QB_FP, QB_FPI, glitch_indx, wavelet_indx, per_puls_indx, per_puls_rn_indx, per_puls_wn_indx, rn_indx, all_noiseparam_idxs, num_per_puls_param_list

################################################################################
#
#MAKE PTA OBJECT FOR PRIOR RECOVERY
#
################################################################################
def get_prior_recovery_pta(pta):
    class prior_recovery_pta:
        def __init__(self, pta):
            self.pta = pta
            self.params = pta.params
            self.pulsars = pta.pulsars
            self.summary = pta.summary
        def get_lnlikelihood(self, x):
            return 0.0
        def get_lnprior(self, x):
            return self.pta.get_lnprior(x)

    return prior_recovery_pta(pta)

################################################################################
#
#MAKE PTA OBJECT WITH CUSTOM T0-F0 PRIOR FOR ZOOM-IN RERUNS
#
################################################################################
def get_tf_prior_pta(pta, TF_prior, n_wavelet, prior_recovery=False):
    class tf_prior_pta:
        def __init__(self, pta):
            self.pta = pta
            self.params = pta.params
            self.pulsars = pta.pulsars
            self.summary = pta.summary
        def get_lnlikelihood(self, x):
            if prior_recovery:
                return 0.0
            else:
                return self.pta.get_lnlikelihood(x)
        def get_lnprior(self, x):
            within_prior = True
            for i in range(n_wavelet):
                t0 = x[8+10*i]
                log10_f0 = x[3+10*i]
                t_idx = int( np.digitize(t0, TF_prior['t_bins']) )
                f_idx = int( np.digitize(log10_f0, TF_prior['lf_bins']) )

                if (t_idx, f_idx) not in TF_prior['on_idxs']:
                    within_prior = False
            if within_prior:
                return self.pta.get_lnprior(x)
            else:
                return -np.inf

    return tf_prior_pta(pta)

################################################################################
#
#SOME HELPER FUNCTIONS
#
################################################################################

def remove_params(samples, j, i, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch, params_slice = False):
    #"Special" indexing for handling edge cases when max_n_wavelet = n_wavelet or max_n_glitch = n_glitch
    #If at max, offset index by 1.
    wave_start = 0
    wave_end = 0
    if max_n_wavelet != 0 and n_wavelet != max_n_wavelet:
        wave_end = wavelet_indx[max_n_wavelet-1][9]+1 # used to be (#-1][5]+1)
        wave_start = wavelet_indx[n_wavelet][0]

    glitch_start = 0
    glitch_end = 0
    if max_n_glitch != 0 and n_glitch != max_n_glitch:
        glitch_end = glitch_indx[max_n_glitch-1][5]+1
        glitch_start = glitch_indx[n_glitch][0]
    if params_slice:
        return np.delete(samples, list(range(wave_start,wave_end))+list(range(glitch_start, glitch_end)))
    else:
        return np.delete(samples[j,i,2:], list(range(wave_start,wave_end))+list(range(glitch_start, glitch_end)))

################################################################################
#
#MATCH CALCULATION ROUTINES
#
################################################################################
#@profile
def get_similarity_matrix(pta, psrs, delays_list, noise_param_dict=None):
    """
    Function to calculate all inner product combinations of delays_list.

    :param pta:
        Enterprise pta object
    :param psrs:
        Pickled pulsar object
    :param delays_list:
        List of signals to calculate inner products between. Each element of delays_list should be shaped (N_psr, N_toa), where N_toa is the number of toas
        for a particular pulsar.
    :param noise_param_dict:
        Noise dictionary used to set default pta parameter values. [None] by default.

    :returns: Numpy array of shape (len(delays_list), len(delays_list)), where diagonal elements are inner products of the i^th list
              with itself (SNR^2 for each), while off-diagonal are the match statistic (squared) of each signal with every other signal.
    """
    if noise_param_dict is None:
        print('No noise dictionary provided!...')
    else:
        pta.set_default_params(noise_param_dict)

    phiinvs = pta.get_phiinv([], logdet=False, method = 'partition')
    TNTs = pta.get_TNT([])
    Ts = pta.get_basis()
    #Using Nvecs due to the N matrix being diagonal. Only need vectors to compute inner products (Ecorr is in Sigma, not Nmat)
    Nvecs = pta.get_ndiag([])
    #Nmats = [make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, Nvecs, Ts)] #call the Nmatt calc in Tau_scans_pta
    #number of waveforms
    n_wf = len(delays_list)

    S = np.zeros((n_wf,n_wf))
    for idx, (psr, Nvec, TNT, phiinv, T) in enumerate(zip(pta.pulsars, Nvecs,
                                                          TNTs, phiinvs, Ts)):
        #Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        # sigmainv = np.linalg.pinv(TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv))
        cf_sigmainv = sl.cho_factor(TNT + np.diag(phiinv)) #for TNT, phiinv in zip(self.TNTs, self.phiinvs)

        for i in range(n_wf):
            for j in range(n_wf):
                delay_i = delays_list[i][idx]
                delay_j = delays_list[j][idx]

                #Now mask to only include nonzero delays, which go from 0 (starting MJD) to max MJD (i.e. end of dataset)
                masked_delay_i = np.copy(delay_i[0:len(psrs[idx].toas)])
                masked_delay_j = np.copy(delay_j[0:len(psrs[idx].toas)])
                S[i,j] += innerprod_cho(Nvec, T, cf_sigmainv, masked_delay_i, masked_delay_j)#innerprod(Nvec, T, sigmainv, TNT, masked_delay_i, masked_delay_j)
    return S

def get_match_matrix(pta, psrs, delays_list, noise_param_dict=None):
    """
    Function to calculate all inner product combinations of delays_list.

    :param pta:
        Enterprise pta
    :param psrs:
        Pickled pulsar object
    :param delays_list:
        List of signals to calculate inner products between. Each element of delays_list should be shaped (N_psr, N_toa), where N_toa is the number of toas
        for a particular pulsar.
    :param noise_param_dict:
        Noise dictionary used to set default pta parameter values. [None] by default.

    :returns: Numpy array of shape (len(delays_list), len(delays_list)). Diagonal elements are inner products of the i^th list
              with itself (SNR for each), while off-diagonal are the match statistic of each signal with every other signal.
    """
    S = get_similarity_matrix(pta, psrs, delays_list, noise_param_dict=noise_param_dict)

    M = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            M[i,j] = S[i,j]/np.sqrt(S[i,i]*S[j,j])
    return M



################################################################################
#
#INNER PRODUCT ROUTINES (based on enterprise Fp_statistic)
#
################################################################################

def innerprod_cho(Nvec, T, cf, x, y):
    """
    Computes the inner product of two vectors x and y.
    :param Nvec: n_diag from PTA
    :param T: PTA get_basis
    :param cf: sigma inverse calculated using Cholesky
    :param x:left vector
    :param y:right vector
    :returns:
    Inner product, xCy
    """
    TNx = Nvec.solve(x, left_array=T)
    TNy = Nvec.solve(y, left_array=T)
    xNy = Nvec.solve(y, left_array=x)

    expval = sl.cho_solve(cf, TNy)
    return xNy - TNx @ expval


def make_Nmat(phiinv, TNT, Nvec, T):

    Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
    cf = sl.cho_factor(Sigma)
    Nshape = np.shape(T)[0]

    TtN = Nvec.solve(other = np.eye(Nshape),left_array = T)

    #Put pulsar's autoerrors in a diagonal matrix
    Ndiag = Nvec.solve(other = np.eye(Nshape),left_array = np.eye(Nshape))

    expval2 = sl.cho_solve(cf,TtN)
    #TtNt = np.transpose(TtN)

    #An Ntoa by Ntoa noise matrix to be used in expand dense matrix calculations earlier
    return Ndiag - np.dot(TtN.T,expval2)

######### DE jump proposal routines ###########

def initialize_de_history(n_chain, samples, FPI, num_params, de_history_size = 5000, n_fast_to_slow = 10000, pta_params = None, verbose = False):
    """
    Create differential evolution (DE) history array
    :param n_chain: Number of parallel tempering chains
    :param samples: Samples array for all parallel tempering chains
    :param FPI: QuickBurst FastPrior class objects for initial sample draws
    :param num_params:

    :return de_history: Array containing an initial differential evolution (DE) buffer

    """

    de_history = np.zeros((n_chain, de_history_size, num_params-2))

    #Loop through each chain and create buffer from prior draws
    for chain in range(n_chain):
        for idx in range(de_history_size):
            #Subtract 2 parameters from total number of parameters (Number of wavelets/noise transients)
            new_point = QB_FastPrior.get_sample_full(num_params-2, FPI)
            new_point = correct_intrinsic(new_point, FPI, FPI.cut_par_ids, FPI.cut_lows, FPI.cut_highs)

            de_history[chain, idx, :] = new_point

    if verbose and pta_params is not None:
        for param in range(num_params-2):
            plt.hist(de_history[0, :, param])
            plt.title(pta_params[param])
            plt.show()
    return de_history

def update_de_history(n_chain, samples, de_history, FPI, num_params, sample_idx, accept_jump_arr, de_history_size = 5000, n_fast_to_slow = 10000, save_every_n = 10000, thin_de = 10000):
    """
    Update differential evolution (DE) history array




    :return de_history: Array containing an initial differential evolution (DE) buffer
    """
    old_de_history = np.copy(de_history)
    for j in range(n_chain):
        #only want to update the history array for the particular chain if the step was accepted
        if accept_jump_arr[j]==0:
            continue
        #Number of de history samples to update. Default is 1.
        n_de_update = n_fast_to_slow//thin_de
        for itrd in range(0,n_de_update):
            #itrn%save_every_n tracks how far you are into the current block of samples
            itrbd = sample_idx%save_every_n+itrd*thin_de
            assert not np.all(samples[j,itrbd,:]==0.)
            de_history[j,de_arr_itr[j]%de_history_size] = samples[j,itrbd,2:]
            de_arr_itr[j] += 1

    #See where the two have changed (should only be one sample set each time)
    de_mask = old_de_history[0, :, :] != de_history[0, :, :]
    return de_history

# @njit()
def DE_proposal(chain_num, sample, de_history, param_ids, ndim):
    """
    Perform differential evolution (DE) jump proposal during MCMC sampling

    :param chain_num: Chain index
    :param sample: Current MCMC sample
    :param de_history: DE history array
    :param param_ids: list of parameter indexes for parameters being varied
    :param ndim: number of parameter indexes

    :return new_point: new sample proposal
    new_point
    """
    #pick two random samples
    de_indices = np.random.choice(de_history.shape[1], size=2, replace=False)

    #copy two sets of samples for intrinsic noise
    x1 = np.copy(de_history[chain_num,de_indices[0],param_ids])
    x2 = np.copy(de_history[chain_num,de_indices[1],param_ids])
    new_point = np.copy(sample)
    #for 10% of DE jumps, do a big DE jump
    if np.random.uniform() < 0.1:
        gamma = 1
        new_point[param_ids] += gamma*(x2 - x1)
    #otherwise, do a regular DE jump
    else:
        alpha = 2.38/np.sqrt(2*ndim)
        gamma = np.random.normal(loc=0,scale=alpha)
        new_point[param_ids] += gamma*(x2 - x1)

    return new_point

#Adopted from QuickCW
"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)
utils for correcting parameters to nominal ranges"""
@njit()
def reflect_into_range(x, x_low, x_high):
    """reflect an arbitrary parameter into a nominal range

    :param x:       Value of parameter to reflect in range
    :param x_low:   Lower bound of parameter
    :param x_high:  Upper bound of prameter

    :return res:    Value of parameter reflected into range
    """
    #ensure always returns something in range (i.e. do an arbitrary number of reflections) similar to reflect_cosines but does not need to track angles
    x_range = x_high-x_low
    res = x
    if res<x_low:
        res = x_low+(-(res-x_low))%(2*x_range)  # 2*x_low - x
    if res>x_high:
        res = x_high-(res-x_high)%(2*x_range)  # 2*x_high - x
        if res<x_low:
            res = x_low+(-(res-x_low))%(2*x_range)  # 2*x_low - x
    return res

@njit()
def correct_intrinsic(sample,x0, cut_par_ids, cut_lows, cut_highs):
    """correct intrinsic parameters for phases and cosines

    :param sample:      Array with parameters
    :param x0:          QBPriorInfo object
    :param cut_par_ids: Indices of parameters needing extra check
    :param cut_lows:    Lower bounds of parameters needing extra check
    :param cut_highs:   Upper bounds of parameters needing extra check

    :return sample:     Corrected sample array
    """

    for itr in range(cut_par_ids.size):
        idx = cut_par_ids[itr]
        sample[idx] = reflect_into_range(sample[idx],cut_lows[itr],cut_highs[itr])

    return sample
