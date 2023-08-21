################################################################################
#
#BayesWavePTA -- Bayesian search for burst GW signals in PTA data based on the BayesWave algorithm
#
#Bence Bécsy (bencebecsy@montana.edu) -- 2020
################################################################################

import numpy as np
import matplotlib.pyplot as plt
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

import enterprise_wavelets as models
import pickle

import shutil
import os

import QuickBurst_lnlike as Quickburst
import QB_FastPrior
import line_profiler

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################
def run_bhb(N_slow, T_max, n_chain, pulsars, max_n_wavelet=1, min_n_wavelet=0, n_wavelet_prior='flat', n_wavelet_start='random', RJ_weight=0, glitch_RJ_weight=0,
            regular_weight=3, noise_jump_weight=3, PT_swap_weight=1, T_ladder=None, T_dynamic=False, T_dynamic_nu=300, T_dynamic_t0=1000, PT_hist_length=100,
            tau_scan_proposal_weight=0, tau_scan_file=None, draw_from_prior_weight=0,
            de_weight=0, prior_recovery=False, wavelet_amp_prior='uniform', gwb_amp_prior='uniform', rn_amp_prior='uniform', per_psr_rn_amp_prior='uniform',
            gwb_log_amp_range=[-18,-11], rn_log_amp_range=[-18,-11], per_psr_rn_log_amp_range=[-18,-11], wavelet_log_amp_range=[-18,-11],
            vary_white_noise=False, efac_start=None, include_equad=False, include_ecorr = False, wn_backend_selection=False, noisedict=None, gwb_switch_weight=0,
            include_rn=False, vary_rn=False, num_total_wn_params=None, rn_params=[-13.0,1.0], include_per_psr_rn=False, vary_per_psr_rn=False, per_psr_rn_start_file=None,
            jupyter_notebook=False, gwb_on_prior=0.5,
            max_n_glitch=1, glitch_amp_prior='uniform', glitch_log_amp_range=[-18, -11], n_glitch_prior='flat', n_glitch_start='random', t0_min=0.0, t0_max=10.0, tref=53000*86400,
            glitch_tau_scan_proposal_weight=0, glitch_tau_scan_file=None, TF_prior_file=None, f0_min=3.5e-9, f0_max=1e-7,
            save_every_n=10, savepath=None, safe_save=False, resume_from=None, start_from=None, n_status_update=100, n_fish_update=1000, n_fast_to_slow=1000, thin = 100,
            ent_lnlike_test = False, QB_attributes = None):
    #scale steps to slow steps
    N = N_slow*n_fast_to_slow
    n_status_update = n_status_update*n_fast_to_slow
    n_fish_update = n_fish_update*n_fast_to_slow
    save_every_n = save_every_n*n_fast_to_slow

    print('Saving every {0} samples, total samples: {1} '.format(save_every_n, N), '\n')
    print('Ending total saved samples: {}'.format(int(N/thin)), '\n')

    #If no wn or rn variance, shouldn't do any noise jumps
    if not vary_white_noise:
        if not vary_per_psr_rn:
            noise_jump_weight = 0
    if TF_prior_file is None:
        TF_prior = None
    else:
        with open(TF_prior_file, 'rb') as f:
            TF_prior = pickle.load(f)
    pta, ent_ptas, QB_FP, QB_FPI, glitch_indx, wavelet_indx, per_puls_indx, rn_indx, num_per_puls_param_list = get_pta(pulsars, vary_white_noise=vary_white_noise, include_equad=include_equad,
                                                                                                    include_ecorr = include_ecorr, wn_backend_selection=wn_backend_selection,
                                                                                                    noisedict=noisedict, include_rn=include_rn, vary_rn=vary_rn,
                                                                                                    include_per_psr_rn=include_per_psr_rn, vary_per_psr_rn=vary_per_psr_rn,
                                                                                                    max_n_wavelet=max_n_wavelet, efac_start=efac_start, rn_amp_prior=rn_amp_prior,
                                                                                                    rn_log_amp_range=rn_log_amp_range, rn_params=rn_params, per_psr_rn_amp_prior=per_psr_rn_amp_prior,
                                                                                                    per_psr_rn_log_amp_range=per_psr_rn_log_amp_range, gwb_amp_prior=gwb_amp_prior,
                                                                                                    gwb_log_amp_range=gwb_log_amp_range, wavelet_amp_prior=wavelet_amp_prior,
                                                                                                    wavelet_log_amp_range=wavelet_log_amp_range, prior_recovery=prior_recovery, ent_lnlike_test = ent_lnlike_test,
                                                                                                    max_n_glitch=max_n_glitch, glitch_amp_prior=glitch_amp_prior, glitch_log_amp_range=glitch_log_amp_range,
                                                                                                    t0_min=t0_min, t0_max=t0_max, f0_min=f0_min, f0_max=f0_max,
                                                                                                    TF_prior=TF_prior, tref=tref)


    print('Number of pta params: ', len(pta.params))
    print(pta.param_names)#summary())

    #setting up temperature ladder
    if n_chain > 1:
        if T_ladder is None:
            #using geometric spacing
            c = T_max**(1.0/(n_chain-1))
            Ts = c**np.arange(n_chain)

            #make highest temperature inf if dynamic T ladder is used
            if T_dynamic:
                Ts[-1] = np.inf

            print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\
     Temperature ladder is:\n".format(n_chain,c),Ts)
        else:
            Ts = np.array(T_ladder)
            n_chain = Ts.size

            #make highest temperature inf if dynamic T ladder is used
            if T_dynamic:
                Ts[-1] = np.inf

        print("Using {0} temperature chains with custom spacing: ".format(n_chain),Ts)
    else:
        Ts = T_max
    if T_dynamic:
        print("Dynamic temperature adjustment: ON")
    else:
        print("Dynamic temperature adjustment: OFF")

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

    #testing number of steps per noise params
    noise_steps = np.zeros((num_per_psr_params))

    if resume_from is not None:
        print("Resuming from file: " + resume_from)
        resume_from += '.h5df'
        with h5py.File(resume_from, 'r+') as f:
            samples_resume = f['samples_cold'][()]
            print('samples_resume: ', samples_resume.shape[1])
            log_likelihood_resume = f['log_likelihood'][()]
                #If resuming a likelihood comparison run
            if ent_lnlike_test:
                ent_lnlikelihood_resume = f['ent_lnlikelihood'][()]
            acc_frac_resume = f['acc_fraction'][()]
            param_names_resume = list(par.decode('utf-8') for par in f['par_names'][()])
            #param_names_resume = f['param_names'][()]
            swap_record_resume = f['swap_record'][()]
            print('resume swap record shape: ', swap_record_resume.shape)
            betas_resume = f['betas'][()]
            PT_acc_resume = f['PT_acc'][()]

        #Print for how many samples loading in.
        N_resume = samples_resume.shape[1]*n_fast_to_slow
        print("# of samples sucessfully read in: " + str(N_resume))

        samples = np.zeros((n_chain, save_every_n+1, num_params))
        samples[:,0,:] = np.copy(samples_resume[:, -1, :])

        swap_record = np.zeros((save_every_n+1, 1))
        swap_record[0] = swap_record_resume[-1]
        print('swap record length: ', swap_record.shape)

        log_likelihood = np.zeros((n_chain,save_every_n+1))
        log_likelihood[:,0] = np.copy(log_likelihood_resume[:, -1])

        ent_lnlikelihood = np.zeros((n_chain,save_every_n+1))
        if ent_lnlike_test:
            ent_lnlikelihood[:,0] = np.copy(ent_lnlikelihood_resume[:, -1])


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
            #first_sample = strip_samples(samples, j, 0, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
            QB_logl.append(Quickburst.QuickBurst(pta = pta, psrs = pulsars, params = dict(zip(pta.param_names, first_sample)), Npsr = len(pulsars), tref=53000*86400, Nglitch = n_glitch, Nwavelet = n_wavelet, Nglitch_max = max_n_glitch ,Nwavelet_max = max_n_wavelet, rn_vary = vary_rn, wn_vary = vary_white_noise, prior_recovery = prior_recovery))
            QB_Info.append(Quickburst.QuickBurst_info(Npsr=len(pulsars),pos = QB_logl[j].pos, resres_logdet = QB_logl[j].resres_logdet, Nglitch = n_glitch ,Nwavelet = n_wavelet, wavelet_prm = QB_logl[j].wavelet_prm, glitch_prm = QB_logl[j].glitch_prm, sigmas = QB_logl[j].sigmas, MMs = QB_logl[j].MMs, NN = QB_logl[j].NN, prior_recovery = prior_recovery, glitch_indx = QB_logl[j].glitch_indx, wavelet_indx = QB_logl[j].wavelet_indx, glitch_pulsars = QB_logl[j].glitch_pulsars))
    else:
        samples = np.zeros((n_chain, save_every_n+1, num_params))

        #set up log_likelihood array
        log_likelihood = np.zeros((n_chain,save_every_n+1))
        ent_lnlikelihood = np.zeros((n_chain,save_every_n+1))
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
            samples_start = start_from
            for j in range(n_chain):
                samples[j,0,:] = np.copy(samples_start[:])
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
                samples[j,0,2:] =  np.hstack(p.sample() for p in pta.params)

                #Setting starting values based on M2A or noise run chain
                if noisedict is not None:
                    #load in params from dictionary
                    for param, idx in enumerate(pta.param_names):
                        if param in noisedict.keys():
                            samples[j, 0, 2+idx] = noisedict[param]

                #set all wavelet gw sources to same sky location
                if n_wavelet!=0:
                    for windx in range(n_wavelet):
                        samples[j,0,2+int(wavelet_indx[windx,0])] = samples[j,0,2+int(wavelet_indx[0,0])]
                        samples[j,0,2+int(wavelet_indx[windx,1])] = samples[j,0,2+int(wavelet_indx[0,1])]
                        samples[j,0,2+int(wavelet_indx[windx,2])] = samples[j,0,2+int(wavelet_indx[0,2])]

                if vary_white_noise and not vary_per_psr_rn:
                    if efac_start is not None:
                        for k in range(len(pulsars)):
                            samples[j,0,2:+wn_indx[k,0]] = 1*efac_start
                ''' functionality to add
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
            #first_sample = strip_samples(samples, j, 0, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

            #Generate first sample param dictionary
            sample_dict = {}
            for i in range(len(first_sample)):
                sample_dict[pta.param_names[i]] = first_sample[i]
            rn_check = False

            if vary_per_psr_rn or vary_rn:
                rn_check = True
            QB_logl.append(Quickburst.QuickBurst(pta = pta, psrs = pulsars, params = sample_dict, Npsr = len(pulsars), tref=53000*86400, Nglitch = n_glitch, Nwavelet = n_wavelet, Nglitch_max = max_n_glitch ,Nwavelet_max = max_n_wavelet, rn_vary = rn_check, wn_vary = vary_white_noise, prior_recovery=prior_recovery))
            QB_Info.append(Quickburst.QuickBurst_info(Npsr=len(pulsars),pos = QB_logl[j].pos, resres_logdet = QB_logl[j].resres_logdet, Nglitch = n_glitch,
                                                      Nwavelet = n_wavelet, wavelet_prm = QB_logl[j].wavelet_prm, glitch_prm = QB_logl[j].glitch_prm, sigmas = QB_logl[j].sigmas,
                                                      MMs = QB_logl[j].MMs, NN = QB_logl[j].NN, prior_recovery = prior_recovery, glitch_indx = QB_logl[j].glitch_indx, wavelet_indx = QB_logl[j].wavelet_indx,
                                                      glitch_pulsars = QB_logl[j].glitch_pulsars))
            log_likelihood[j,0] = QB_logl[j].get_lnlikelihood(first_sample, vary_white_noise = vary_white_noise, vary_red_noise = rn_check)
            if ent_lnlike_test:
                ent_lnlikelihood[j,0] = ent_ptas[n_wavelet][n_glitch].get_lnlikelihood(remove_params(first_sample, 0, 0, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch, params_slice = True))
                print('our like: ', log_likelihood[j,0])
                print('PTA like: ', ent_ptas[n_wavelet][n_glitch].get_lnlikelihood(remove_params(first_sample, 0, 0, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch, params_slice = True)))
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
    #and one for white noise parameters, which we will also keep updating
    eig_per_psr = np.broadcast_to(np.eye(num_per_psr_params)*0.1, (n_chain, num_per_psr_params, num_per_psr_params) ).copy()

    #read in tau_scan data if we will need it
    if tau_scan_proposal_weight+RJ_weight>0:
        if tau_scan_file==None:
            raise Exception("tau-scan data file is needed for tau-scan global propsals")
        with open(tau_scan_file, 'rb') as f:
            tau_scan_data = pickle.load(f)
            print("Tau-scan data read in successfully!")

        tau_scan = tau_scan_data['tau_scan']
        print(len(tau_scan))

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
        print(norm)

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
    a_yes=np.zeros((9, n_chain)) #columns: chain number; rows: proposal type (glitch_RJ, glitch_tauscan, wavelet_RJ, wavelet_tauscan, gwb_RJ, PT, fast fisher, regular fisher, noise_jump)
    a_no=np.zeros((9, n_chain))
    acc_fraction = a_yes/(a_no+a_yes)
    if resume_from is None:
        swap_record = np.zeros((save_every_n+1, 1))
    rj_record = []

    #set up probabilities of different proposals
    total_weight = (regular_weight + PT_swap_weight + tau_scan_proposal_weight +
                    RJ_weight + gwb_switch_weight + noise_jump_weight + glitch_tau_scan_proposal_weight + glitch_RJ_weight)
    swap_probability = PT_swap_weight/total_weight
    tau_scan_proposal_probability = tau_scan_proposal_weight/total_weight
    regular_probability = regular_weight/total_weight
    RJ_probability = RJ_weight/total_weight
    gwb_switch_probability = gwb_switch_weight/total_weight
    noise_jump_probability = noise_jump_weight/total_weight
    glitch_tau_scan_proposal_probability = glitch_tau_scan_proposal_weight/total_weight
    glitch_RJ_probability = glitch_RJ_weight/total_weight
    print("Percentage of steps doing different jumps:\nPT swaps: {0:.2f}%\nRJ moves: {3:.2f}%\nGlitch RJ moves: {7:.2f}%\nGWB-switches: {4:.2f}%\n\
Tau-scan-proposals: {1:.2f}%\nGlitch tau-scan-proposals: {6:.2f}%\nJumps along Fisher eigendirections: {2:.2f}%\nNoise jump: {5:.2f}%".format(swap_probability*100,
          tau_scan_proposal_probability*100, regular_probability*100,
          RJ_probability*100, gwb_switch_probability*100, noise_jump_probability*100, glitch_tau_scan_proposal_probability*100, glitch_RJ_probability*100))

    #No longer need if/else, since we simply append to existing file. Should be from 0 to N always.
    start_iter = 0
    stop_iter = N


    fast_declines_prior =  np.zeros((1, n_chain))
    regular_declines_prior =  np.zeros((1, n_chain))

    #Generate first step and likelihood_attributes regardless of testing or not.
    likelihood_attributes = {}
    attributes_to_test = []
    step_array = []
    if ent_lnlike_test == True:

        #Sort alphabetically if there is a list to sort.
        if QB_attributes is not None:
            #Get attributes to save from run_bhb kwarg
            attributes_to_test = QB_attributes
            attributes_to_test.sort()

            if resume_from == True:
                likelihood_attributes = json.load(savepath + '.json')
            else:
                #Create dictionary for each chain, and append a list to each parameter in each
                #temp_dict for each chain
                for jj in range(n_chain):
                    temp_dict = {}

                    #While here, also create step_array list for each chain.
                    #step_array.append(['first_step'])
                    for kk in attributes_to_test:
                        temp_dict[kk] = [np.copy(getattr(QB_logl[jj], kk))]
                    #added so this array matches with the params and likelihood indexes
                    likelihood_attributes['chain_{}'.format(jj)] = temp_dict
                    print(likelihood_attributes['chain_{}'.format(jj)].keys())
            print(likelihood_attributes.keys())

        else:
            ValueError('Must pass in list of Quickburst object properties!')

        for jj in range(n_chain):
            step_array.append(['first_step'])
        print(step_array)
        # #Create empty dictionary for storing object properties to
        # if resume_from == True:
        #     likelihood_attributes = json.load(savepath + '.json')
        # else:
        #     #Create dictionary for each chain, and append a list to each parameter in each
        #     #temp_dict for each chain
        #     for jj in range(n_chain):
        #         temp_dict = {}
        #         for kk in attributes_to_test:
        #             temp_dict[kk] = [np.copy(getattr(QB_logl[jj], kk))]
        #         #added so this array matches with the params and likelihood indexes
        #         likelihood_attributes['chain_{}'.format(jj)] = temp_dict
        #         print(likelihood_attributes['chain_{}'.format(jj)].keys())
        # print(likelihood_attributes.keys())
    #print('Likelihood Attributes shape: ', likelihood_attributes)

    t_start = time.time()



    #########################
    #MAIN MCMC LOOP
    #########################
    for i in range(int(start_iter), int(stop_iter-1)): #-1 because ith step here produces (i+1)th sample based on ith sample
        ########################################################
        #
        #write results to file every save_every_n iterations
        #
        ########################################################
        if savepath is not None and i%save_every_n==0 and i!=start_iter:
            """output to hdf5 at loop iteration"""
            if savepath is not None:
                savefile = savepath + '.h5df'
                if i>save_every_n:
                    with h5py.File(savefile, 'a') as f:
                        #Create shape for samples
                        f['samples_cold'].resize((f['samples_cold'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
                        f['swap_record'].resize((f['swap_record'].shape[0] + int((swap_record.shape[0] - 1)/thin)), axis = 0)
                        f['betas'].resize((f['betas'].shape[1] + int((betas.shape[1] - 1)/thin)), axis=1)
                        f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
                        if ent_lnlike_test:
                            f['ent_lnlikelihood'].resize((f['ent_lnlikelihood'].shape[1] + int((ent_lnlikelihood.shape[1] - 1)/thin)), axis=1)
                            f['ent_lnlikelihood'][:,-int((ent_lnlikelihood.shape[1]-1)/thin):] = ent_lnlikelihood[:,:-1:thin]
                        #Save samples
                        f['samples_cold'][:,-int((samples.shape[1]-1)/thin):,:] = samples[:,:-1:thin,:]
                        f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = log_likelihood[:,:-1:thin]
                        #Create shape for Betas
                        f['betas'][:,-int((log_likelihood.shape[1]-1)/thin):] = betas[:, :-1:thin]
                        f['PT_acc'][:,-int((log_likelihood.shape[1]-1)/thin):] = PT_acc[:, :-1:thin]
                        f['acc_fraction'][...] = np.copy(acc_fraction)
                        f['swap_record'][-int((log_likelihood.shape[1]-1)/thin):] = np.copy(swap_record[:-1:thin])

                    if ent_lnlike_test:
                        with open(savepath + '.pkl', 'wb') as f:
                            #Convert our properties back to lists, not np arrays
                            for n in range(n_chain):
                                for jj in attributes_to_test:
                                    #print('property {} type: '.format(jj), str(type(likelihood_attributes['chain_{}'.format(n)][jj][0])))
                                    if str(type(likelihood_attributes['chain_{}'.format(n)][jj][0])) == "<class 'numpy.ndarray'>":
                                        #print('if statement triggered! \n')
                                        #If ndarray, go through all samples and change each element to list
                                        # print('# of Samples for attribute {}: '.format(jj), len(likelihood_attributes['chain_{}'.format(n)][jj]))
                                        for kk in range(len(likelihood_attributes['chain_{}'.format(n)][jj])):
                                            #Save over ndarray with list type object

                                            ##### SECONDARY IDEA: We create a temporary thing that takes the elements out at each step for each attribute, and saves them to a list,
                                            ##### and that gets pickled in the end. This would work, but we would need to be VERY careful....
                                            temp_list = likelihood_attributes['chain_{}'.format(n)][jj][kk].tolist()
                                            likelihood_attributes['chain_{}'.format(n)][jj][kk] = temp_list
                            pickle.dump(likelihood_attributes, f)

                else:
                    if resume_from is None:
                        with h5py.File(savefile, 'w') as f:
                            f.create_dataset('samples_cold', data= samples[:,:-1:thin,:], compression="gzip", chunks=True, maxshape = (n_chain, None, samples.shape[2])) #maxshape=(n_chain,int(N/thin),samples.shape[2]))
                            f.create_dataset('log_likelihood', data=log_likelihood[:,:-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))
                            if ent_lnlike_test:
                                f.create_dataset('ent_lnlikelihood', data=ent_lnlikelihood[:,:-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))
                            f.create_dataset('par_names', data=np.array(pta.param_names, dtype='S'))
                            f.create_dataset('acc_fraction', data=acc_fraction)
                            f.create_dataset('swap_record', data = swap_record[:-1:thin], compression="gzip", chunks=True, maxshape = (None,1))# maxshape=int(N/thin))
                            f.create_dataset('betas', data=betas[:, :-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))
                            f.create_dataset('PT_acc', data=PT_acc[:, :-1:thin], compression="gzip", chunks=True, maxshape = (n_chain, None)) #maxshape=(samples.shape[0],int(N/thin)))
                            print('max shape: ', f['swap_record'].maxshape)

                        if ent_lnlike_test:
                            with open(savepath + '.pkl', 'wb') as f:
                                #Convert our properties back to lists, not np arrays
                                for n in range(n_chain):
                                    for jj in attributes_to_test:
                                        #print('property {} type: '.format(jj), str(type(likelihood_attributes['chain_{}'.format(n)][jj][0])))
                                        if str(type(likelihood_attributes['chain_{}'.format(n)][jj][0])) == "<class 'numpy.ndarray'>":
                                            #print('if statement triggered! \n')
                                            #If ndarray, go through all samples and change each element to list
                                            # print('# of Samples for attribute {}: '.format(jj), len(likelihood_attributes['chain_{}'.format(n)][jj]))
                                            for kk in range(len(likelihood_attributes['chain_{}'.format(n)][jj])):
                                                #Save over ndarray with list type object

                                                ##### SECONDARY IDEA: We create a temporary thing that takes the elements out at each step for each attribute, and saves them to a list,
                                                ##### and that gets pickled in the end. This would work, but we would need to be VERY careful....
                                                temp_list = likelihood_attributes['chain_{}'.format(n)][jj][kk].tolist()
                                                likelihood_attributes['chain_{}'.format(n)][jj][kk] = temp_list
                                pickle.dump(likelihood_attributes, f)

                    else:
                        with h5py.File(savefile, 'a') as f:
                            #Create shape for samples
                            print('samples shape:', f['samples_cold'].shape[1])
                            print('samples.shape[1]', int(samples.shape[1]-1)/thin)
                            f['samples_cold'].resize((f['samples_cold'].shape[1] + int((samples.shape[1] - 1)/thin)), axis=1)
                            f['swap_record'].resize((f['swap_record'].shape[0] + int((swap_record.shape[0] - 1)/thin)), axis = 0)
                            f['betas'].resize((f['betas'].shape[1] + int((betas.shape[1] - 1)/thin)), axis=1)
                            f['log_likelihood'].resize((f['log_likelihood'].shape[1] + int((log_likelihood.shape[1] - 1)/thin)), axis=1)
                            if ent_lnlike_test:
                                f['ent_lnlikelihood'].resize((f['ent_lnlikelihood'].shape[1] + int((ent_lnlikelihood.shape[1] - 1)/thin)), axis=1)
                                f['ent_lnlikelihood'][:,-int((ent_lnlikelihood.shape[1]-1)/thin):] = ent_lnlikelihood[:,:-1:thin]
                            #Save samples
                            f['samples_cold'][:,-int((samples.shape[1]-1)/thin):,:] = samples[:,:-1:thin,:]
                            f['log_likelihood'][:,-int((log_likelihood.shape[1]-1)/thin):] = log_likelihood[:,:-1:thin]
                            #Create shape for Betas
                            f['betas'][:,-int((log_likelihood.shape[1]-1)/thin):] = betas[:, :-1:thin]
                            f['PT_acc'][:,-int((log_likelihood.shape[1]-1)/thin):] = PT_acc[:, :-1:thin]
                            f['acc_fraction'][...] = np.copy(acc_fraction)
                            f['swap_record'][-int((log_likelihood.shape[1]-1)/thin):] = np.copy(swap_record[:-1:thin])
                        if ent_lnlike_test:
                            with open(savepath + '.pkl', 'wb') as f:
                                #Convert our properties back to lists, not np arrays
                                for n in range(n_chain):
                                    for jj in attributes_to_test:
                                        #print('property {} type: '.format(jj), str(type(likelihood_attributes['chain_{}'.format(n)][jj][0])))
                                        if str(type(likelihood_attributes['chain_{}'.format(n)][jj][0])) == "<class 'numpy.ndarray'>":
                                            #print('if statement triggered! \n')
                                            #If ndarray, go through all samples and change each element to list
                                            # print('# of Samples for attribute {}: '.format(jj), len(likelihood_attributes['chain_{}'.format(n)][jj]))
                                            for kk in range(len(likelihood_attributes['chain_{}'.format(n)][jj])):
                                                #Save over ndarray with list type object

                                                ##### SECONDARY IDEA: We create a temporary thing that takes the elements out at each step for each attribute, and saves them to a list,
                                                ##### and that gets pickled in the end. This would work, but we would need to be VERY careful....
                                                temp_list = likelihood_attributes['chain_{}'.format(n)][jj][kk].tolist()
                                                likelihood_attributes['chain_{}'.format(n)][jj][kk] = temp_list
                                pickle.dump(likelihood_attributes, f)

            #For comparing likelihood values to QuickBurst likelihoods
            if ent_lnlike_test:
                ent_lnlikelihood_now = ent_lnlikelihood[:, -1]
                ent_lnlikelihood = np.zeros((n_chain, save_every_n+1))
                ent_lnlikelihood[:,0] = ent_lnlikelihood_now
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
        #update temperature ladder
        #
        ########################################################
        '''
        if i%save_every_n>0:
            if T_dynamic and PT_hist_idx>0: #based on arXiv:1501.05823 and https://github.com/willvousden/ptemcee
                kappa = 1.0/T_dynamic_nu * T_dynamic_t0/(PT_hist_idx+T_dynamic_t0)
                #dSs = kappa * (acc_fraction[5,:-2] - acc_fraction[5,1:-1])
                dSs = kappa * (PT_acc[:-1,i%save_every_n] - PT_acc[1:,i%save_every_n])
                #print(dSs)
                deltaTs = np.diff(1 / betas[:-1,i%save_every_n-1])
                #print(deltaTs)
                deltaTs *= np.exp(dSs)
                #print(deltaTs)

                new_betas = 1 / (np.cumsum(deltaTs) + 1 / betas[0,i%save_every_n-1])
                #print(new_betas)

                #set new betas
                betas[-1,i%save_every_n] = 0.0
                betas[1:-1,i%save_every_n] = np.copy(new_betas)
            else:
                #copy betas from previous iteration
                betas[:,i%save_every_n] = betas[:,i%save_every_n-1]
        '''
        #for now, just update the next step to have the current temp
        betas[:,i%save_every_n+1] = betas[:,i%save_every_n]

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
                        'Acceptance fraction #columns: chain number; rows: proposal type (glitch_RJ, glitch_tauscan, wavelet_RJ, wavelet_tauscan, GWB_RJ, PT, fisher, noise_jump):'+'\n')
                print('Run Time: {0}s'.format(time.time()-t_start))
                print(noise_steps)
                print(acc_fraction)
                fast_decs_prior = []
                regular_decs_prior = []
                for n in range(n_chain):
                    temp_fast = 0
                    temp_reg = 0
                    for k in range(len(fast_declines_prior)):
                        temp_fast += fast_declines_prior[k]
                        temp_reg += regular_declines_prior[k]
                    fast_decs_prior.append(temp_fast)
                    regular_decs_prior.append(temp_reg)
                print('fast_declines_prior chain1/chain2/chain3: ', fast_decs_prior, '\n')
                print('regular_declines_prior chain1/chain2/chain3: ', regular_decs_prior, '\n')
                #print(PT_hist)
                print(PT_acc[:,i%save_every_n])
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
                    n_wavelet = int(samples[j,i%save_every_n,0]) #get_n_wavelet(samples, j, i%save_every_n)
                    n_glitch = int(samples[j,i%save_every_n,1]) #get_n_glitch(samples, j, i%save_every_n)

                    # Fisher Information Matrix: Calculates the covariances for each parameter associated with
                    # maximum likelihood estimates. This is used to inform jump proposals in parameter space
                    # for various kinds of jumps by pulling out eigenvectors from Fisher Matrix for particular
                    # parameters that are being updated.

                    #wavelet eigenvectors
                    if n_wavelet!=0:
                        eigenvectors = get_fisher_eigenvectors(np.copy(samples[j, i%save_every_n, 2:]), pta, QB_FP, QB_logl=QB_logl[j], T_chain=1/betas[j,i%save_every_n], n_sources=n_wavelet, array_index=wavelet_indx, flag = True)
                        if np.all(eigenvectors):
                            eig[j,:n_wavelet,:,:] = eigenvectors

                    #glitch eigenvectors
                    if n_glitch!=0:
                        eigen_glitch = get_fisher_eigenvectors(np.copy(samples[j, i%save_every_n, 2:]), pta, QB_FP, QB_logl=QB_logl[j], T_chain=1/betas[j,i%save_every_n], n_sources=n_glitch, dim=6, array_index=glitch_indx, flag = True)
                        if np.all(eigen_glitch):
                            eig_glitch[j,:n_glitch,:,:] = eigen_glitch

                    #RN eigenvectors
                    if vary_rn:
                        eigvec_rn = get_fisher_eigenvectors(np.copy(samples[j, i%save_every_n, 2:]), pta, QB_FP, QB_logl=QB_logl[j], T_chain=1/betas[j,i%save_every_n], n_sources=1, dim=len(rn_indx), array_index=rn_indx, vary_rn = vary_rn)
                        if np.all(eigvec_rn):
                            eig_rn[j,:,:] = eigvec_rn[0,:,:]

                    #per PSR eigenvectors
                    if vary_per_psr_rn or vary_white_noise:
                        per_psr_eigvec = get_fisher_eigenvectors(np.copy(samples[j, i%save_every_n, 2:]), pta, QB_FP, QB_logl=QB_logl[j], T_chain=1/betas[j,i%save_every_n], n_sources=len(pulsars), dim=len(per_puls_indx[1]), array_index=per_puls_indx, vary_white_noise = vary_white_noise, vary_psr_red_noise = vary_per_psr_rn)
                        if np.all(per_psr_eigvec):
                            eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]

                    #print('QB_Info[j].resres_logdet: ',QB_Info[j].NN)
        ###########################################################
        #
        #Do the actual MCMC step
        #
        ###########################################################
        #print('iterable: ',i)
        if i%n_fast_to_slow==0:
            #draw a random number to decide which jump to do
            jump_decide = np.random.uniform()
            #print('jump_decide: ',jump_decide)
            #i%save_every_n will check where we are in sample blocks
            if (jump_decide<swap_probability):
                do_pt_swap(n_chain, max_n_wavelet, max_n_glitch, pta, QB_FPI, QB_logl, likelihood_attributes, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, swap_record, vary_white_noise, num_noise_params, log_likelihood, ent_lnlikelihood, PT_hist, PT_hist_idx, ent_lnlike_test, attributes_to_test, step_array)
                #step_array.append('PT_SWAP')
            #global proposal based on tau_scan
            elif (jump_decide<swap_probability+tau_scan_proposal_probability):
                do_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, pta,  ent_ptas, QB_FPI, QB_logl, likelihood_attributes, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, vary_white_noise, num_noise_params, tau_scan_data, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx)
                #step_array.append('TAU_GLOBAL')
            #jump to change number of wavelets
            elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability):
                do_wavelet_rj_move(n_chain, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior, pta,  ent_ptas, QB_FPI, QB_logl, likelihood_attributes, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, rj_record, vary_white_noise, num_noise_params, tau_scan_data, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx)
                #step_array.append('RJ')
            #jump to change some noise parameters
            elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+noise_jump_probability):
                noise_jump(n_chain, max_n_wavelet, max_n_glitch, pta, ent_ptas, QB_FPI, QB_logl, likelihood_attributes, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, eig_per_psr, per_puls_indx, num_noise_params, vary_white_noise, vary_per_psr_rn, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx, noise_steps)
                #step_array.append('NOISE')
            #jump to change glitch params
            elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+noise_jump_probability+glitch_tau_scan_proposal_probability):
                do_glitch_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, pta,  ent_ptas, QB_FPI, QB_logl, likelihood_attributes, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, vary_white_noise, num_noise_params, glitch_tau_scan_data, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx)
                #step_array.append('GLITCH_TAU_GLOBAL')
            #jump to change number of glitches
            elif (jump_decide<swap_probability+tau_scan_proposal_probability+RJ_probability+noise_jump_probability+glitch_tau_scan_proposal_probability+glitch_RJ_probability):
                do_glitch_rj_move(n_chain, max_n_wavelet, max_n_glitch, n_glitch_prior, pta,  ent_ptas, QB_FPI, QB_logl, likelihood_attributes, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, vary_white_noise, num_noise_params, glitch_tau_scan_data, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx)
                #step_array.append('glitch_RJ')
            #do regular jump
            else:
                regular_jump(n_chain, max_n_wavelet, max_n_glitch, pta,  ent_ptas, QB_FPI, QB_logl, likelihood_attributes, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, eig, eig_glitch, eig_rn, num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, rn_indx, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, regular_declines_prior)
                #step_array.append('Regular_jump')

        else:
            #For fast jumps, can't have wavelet_indx[i, 3, 8, 9] or glitch_indx[i, 0, 3, 4, 5] Otherwise M and N gets recalculated
            #Note: i%save_every_n will be 1 through 9 when i%n_fast_to_slow != 0.
            fast_jump(n_chain, max_n_wavelet, max_n_glitch, QB_FPI, QB_Info, samples, i%save_every_n, betas, a_yes, a_no, eig, eig_glitch, eig_rn, num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, log_likelihood, fast_declines_prior)
            #step_array.append('fast_jump')
            #fast step goes here
            if ent_lnlike_test:
                for jj in range(n_chain):
                    step_array[jj].append('fast_jump')
                    #print('Still doing fast jump ent lnlike test for chain {}! \n'.format(jj))
                    n_wavelet = int(samples[jj,i%save_every_n,0])
                    n_glitch = int(samples[jj,i%save_every_n,1])
                    temp_entlike = ent_ptas[n_wavelet][n_glitch].get_lnlikelihood(remove_params(samples, jj, i%save_every_n+1, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch))
                    #print('fast new_enterprise likelihood: ', temp_entlike)
                    #print('fast old_enterprise likelihood: ', pta.get_lnlikelihood(samples[jj,i%save_every_n,2:]))
                    ent_lnlikelihood[jj, i%save_every_n+1] = temp_entlike
                    for kk in attributes_to_test:
                        #Need to np.copy(getattr()), as getattr() points to object parameter at all steps. Changing subsequent steps will change all steps.
                        likelihood_attributes['chain_{}'.format(jj)][kk].append(np.copy(getattr(QB_logl[jj], kk)))
                    #likelihood_attributes[jj].append(np.array([QB_Info[jj].resres_logdet, QB_logl[jj].Nvecs_previous, QB_logl[jj].Nvecs, QB_Info[jj].glitch_prm, QB_Info[jj].wavelet_prm, QB_Info[jj].glitch_pulsars, QB_logl[jj].params, QB_Info[jj].MMs, QB_Info[jj].NN, QB_Info[jj].sigmas]))
    acc_fraction = a_yes/(a_no+a_yes)

    if ent_lnlike_test:
        return samples[:,::n_fast_to_slow,:], acc_fraction, swap_record, rj_record, pta, log_likelihood[:,::n_fast_to_slow], ent_lnlikelihood[:, ::n_fast_to_slow], betas[:,::n_fast_to_slow], PT_acc, noise_steps, step_array, likelihood_attributes
    else:
        return samples[:,::n_fast_to_slow,:], acc_fraction, swap_record, rj_record, pta, log_likelihood[:,::n_fast_to_slow], betas[:,::n_fast_to_slow], PT_acc, noise_steps

################################################################################
#
#GLOBAL PROPOSAL BASED ON TAU-SCAN
#
################################################################################
def do_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, pta,  ent_ptas, FPI, QB_logl, likelihood_attributes, QB_Info, samples, i, betas, a_yes, a_no, vary_white_noise, num_noise_params, tau_scan_data, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx):
    print("TAU-GLOBAL")
    tau_scan = tau_scan_data['tau_scan']
    tau_scan_limit = 0
    for TS in tau_scan:
        TS_max = np.max(TS)
        if TS_max>tau_scan_limit:
            tau_scan_limit = TS_max

    TAU_list = list(tau_scan_data['tau_edges'])
    F0_list = tau_scan_data['f0_edges']
    T0_list = tau_scan_data['t0_edges']

    for j in range(n_chain):
        #check if there's any wavelet -- stay at given point if not
        n_wavelet = int(samples[j,i,0]) #get_n_wavelet(samples, j, i)
        n_glitch = int(samples[j,i,1]) #get_n_glitch(samples, j, i)

        if n_wavelet==0:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[3,j]+=1
            ent_lnlikelihood[j, i+1] = ent_lnlikelihood[j, i]
            log_likelihood[j,i+1] = log_likelihood[j,i]
            if ent_lnlike_test:
                step_array[j].append('TAU_GLOBAL_rejected_0wavelet')
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
            #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "TAU-GLOBAL"]))
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
        #print('Wavelet tau scan QB logl: ', log_L)
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
            if ent_lnlike_test:
                step_array[j].append('TAU_GLOBAL_accepted')
                temp_entlike = ent_ptas[n_wavelet][n_glitch].get_lnlikelihood(remove_params(samples, j, i+1, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch))
                #print('Wavelet tau scan enterprise likelihood: ', temp_entlike)
                ent_lnlikelihood[j, i+1] = temp_entlike
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "TAU-GLOBAL"]))

            #print("accept step")
            QB_logl[j].save_values(accept_new_step=True)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[3,j]+=1
            ent_lnlikelihood[j, i+1] = ent_lnlikelihood[j, i]
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #print("reject step")
            QB_logl[j].save_values(accept_new_step=False)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
            if ent_lnlike_test:
                step_array[j].append('TAU_GLOBAL_rejected')
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
            #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "TAU-GLOBAL"]))
################################################################################
#
#GLITCH MODEL GLOBAL PROPOSAL BASED ON TAU-SCAN
#
################################################################################
def do_glitch_tau_scan_global_jump(n_chain, max_n_wavelet, max_n_glitch, pta,  ent_ptas, FPI, QB_logl, likelihood_attributes, QB_Info, samples, i, betas, a_yes, a_no, vary_white_noise, num_noise_params, glitch_tau_scan_data, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx):
    print("GLITCH TAU-GLOBAL")

    TAU_list = list(glitch_tau_scan_data['tau_edges'])
    F0_list = glitch_tau_scan_data['f0_edges']
    T0_list = glitch_tau_scan_data['t0_edges']

    for j in range(n_chain):
        #check if there's any wavelet -- stay at given point if not
        n_wavelet = int(samples[j,i,0]) #get_n_wavelet(samples, j, i)
        n_glitch = int(samples[j,i,1]) #get_n_glitch(samples, j, i)
        if n_glitch==0:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[1,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            ent_lnlikelihood[j, i+1] = ent_lnlikelihood[j, i]
            if ent_lnlike_test:
                step_array[j].append('GLITCH_TAU_GLOBAL_rejected_0glitch')
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
            #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "GLITCH TAU-GLOBAL"]))
            #print("No glitch to vary!")
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
            TS_max = np.max(TS)
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
        #print('Glitch tau scan QB logl: ', log_L)
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
            TS_max = np.max(TS)
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
            if ent_lnlike_test:
                step_array[j].append('GLITCH_TAU_GLOBAL_accepted')
                temp_entlike = ent_ptas[n_wavelet][n_glitch].get_lnlikelihood(remove_params(samples, j, i+1, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch))
                #print('Glitch tau scan enterprise likelihood: ', temp_entlike)
                ent_lnlikelihood[j, i+1] = temp_entlike
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "GLITCH TAU-GLOBAL"]))
            #print("accept step")
            QB_logl[j].save_values(accept_new_step=True)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[1,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            ent_lnlikelihood[j, i+1] = ent_lnlikelihood[j,i]
            #print("reject step")
            QB_logl[j].save_values(accept_new_step=False)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
            if ent_lnlike_test:
                step_array[j].append('GLITCH_TAU_GLOBAL_rejected')
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
            #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "GLITCH TAU-GLOBAL"]))

################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN CW, GWB AND RN PARAMETERS)
#
################################################################################
def regular_jump(n_chain, max_n_wavelet, max_n_glitch, pta,  ent_ptas, FPI, QB_logl, likelihood_attributes, QB_Info, samples, i, betas, a_yes, a_no, eig, eig_glitch, eig_rn, num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, rn_indx, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, regular_declines_prior):
    print("Regular_jump")
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
            a_no[7,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            ent_lnlikelihood[j,i+1] = ent_lnlikelihood[j,i]
            if ent_lnlike_test:
                step_array[j].append('Regular_jump_rejected')
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
            #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "Regular_jump"]))
            #print("Nothing to vary!")
            continue
        #print('Regular jump: Varying {}'.format(what_to_vary), '\n')

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
            #print('Regular jump prior is infinite')
            samples[j,i+1,:] = samples[j,i,:]
            a_no[7,j] += 1
            regular_declines_prior[0, j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            ent_lnlikelihood[j, i+1] = ent_lnlikelihood[j,i]
            if ent_lnlike_test:
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
            #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "Regular_jump"]))
            continue

        log_L = QB_logl[j].get_lnlikelihood(new_point, vary_red_noise = rn_changed, vary_white_noise = wn_changed)
        #print('Regular jump QB log_l: ', log_L)
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
        #print('Regular acc_ratio: ',acc_ratio)
        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            a_yes[7,j]+=1
            log_likelihood[j,i+1] = log_L
            if ent_lnlike_test:
                step_array[j].append('Regular_jump_accepted')
                temp_entlike = ent_ptas[n_wavelet][n_glitch].get_lnlikelihood(remove_params(samples, j, i+1, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch))
                #print('Regular jump enterprise likelihood: ', temp_entlike)
                ent_lnlikelihood[j, i+1] = temp_entlike
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "Regular_jump"]))
                #print('Regular jump QB_logl[{}].NN: '.format(j), QB_logl[j].NN)
            #print("accept step")
            QB_logl[j].save_values(accept_new_step=True)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[7,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            ent_lnlikelihood[j,i+1] = ent_lnlikelihood[j,i]
            #print("reject step")
            QB_logl[j].save_values(accept_new_step=False, vary_red_noise = rn_changed, vary_white_noise = wn_changed)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
            if ent_lnlike_test:
                step_array[j].append('Regular_jump_rejected')
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
            #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "Regular_jump"]))
            #print('Regular jump QB_logl[{}].NN: '.format(j), QB_logl[j].NN)

################################################################################
#
#Fast MCMC JUMP ROUTINE (jumping in projection PARAMETERS)
#
################################################################################
@njit(fastmath=True,parallel=False)
def fast_jump(n_chain, max_n_wavelet, max_n_glitch, FPI, QB_Info, samples, i, betas, a_yes, a_no, eig, eig_glitch, eig_rn, num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, log_likelihood, fast_declines_prior):
    #print("fast_jump")
    for j in range(n_chain):
        n_wavelet = int(samples[j,i,0])
        n_glitch = int(samples[j,i,1])

        samples_current = np.copy(samples[j,i,2:]) #strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

        #decide if moving in wavelet parameters, glitch parameters, or GWB/RN parameters
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
            #print('no fast step chosen!')
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #print("Nothing to vary!")
            continue

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
            #to avoid all shape parametersfor wavelets: wavelet_indx[i, 0, 6, or 7]
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
        #check if we are inside prior before calling likelihood, otherwise it throws an error
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
            #print('Fast jump: Prior is infinite at step {0}'.format(iter))
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j] += 1
            fast_declines_prior[0, j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #ent_lnlikelihood[j, i+1] = ent_lnlikelihood[j, i]
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
        #print('Fast acc_ratio: ',acc_ratio)
        if np.random.random()<=acc_ratio:
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            a_yes[6,j]+=1
            log_likelihood[j,i+1] = log_L
            #print("accept step")
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #ent_lnlikelihood[j, i+1] = ent_lnlikelihood[j,i]
            #print("reject step")


################################################################################
#
#REVERSIBLE-JUMP (RJ, aka TRANS-DIMENSIONAL) MOVE -- adding or removing a wavelet
#
################################################################################
def do_wavelet_rj_move(n_chain, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior, pta, ent_ptas, FPI, QB_logl, likelihood_attributes, QB_Info, samples, i, betas, a_yes, a_no, rj_record, vary_white_noise, num_noise_params, tau_scan_data, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx):
    print("RJ")
    tau_scan = tau_scan_data['tau_scan']

    tau_scan_limit = 0
    for TS in tau_scan:
        TS_max = np.max(TS)
        if TS_max>tau_scan_limit:
            tau_scan_limit = TS_max

    TAU_list = list(tau_scan_data['tau_edges'])
    F0_list = tau_scan_data['f0_edges']
    T0_list = tau_scan_data['t0_edges']
    #print(samples[0,i,:])
    #print(i)

    for j in range(n_chain):
        n_wavelet = int(samples[j,i,0]) #get_n_wavelet(samples, j, i)
        n_glitch = int(samples[j,i,1]) #get_n_glitch(samples, j, i)

        add_prob = 0.5 #same propability of addind and removing
        #decide if we add or remove a signal
        direction_decide = np.random.uniform()
        if n_wavelet==min_n_wavelet or (direction_decide<add_prob and n_wavelet!=max_n_wavelet): #adding a wavelet------------------------------------------------------
            print('adding wavelet')
            if j==0: rj_record.append(1)
            #if j==0: print("Propose to add a wavelet")
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
                #print(tau_scan_new_point/tau_scan_limit)
                if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
                    accepted = True
                    #print("Yeeeh!")
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

            samples_current = np.copy(samples[j,i,2:])#strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

            new_point = np.copy(samples[j,i,2:])#strip_samples(samples, j, i, n_wavelet+1, max_n_wavelet, n_glitch, max_n_glitch)
            # new_wavelet = np.array([cos_gwtheta_new, psi_new, gwphi_new, log_f0_new, log10_h_new, log10_h_cross_new,
            #                         phase0_new, phase0_cross_new, t0_new, tau_new])
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
            #print('Adding wavelet QB Log_L: ', log_L)
            #print('Adding wavelet ent Log_L: ', ent_ptas[n_wavelet+1][n_glitch].get_lnlikelihood(remove_params(new_point, 0, 0, wavelet_indx, glitch_indx, n_wavelet+1, max_n_wavelet, n_glitch, max_n_glitch, params_slice = True)))
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

            #If proposing adding wavelet, always accept (FOR TESTING ONLY)
            #acc_ratio = 1
            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Yeeeh")
                #print('accepted')
                samples[j,i+1,0] = n_wavelet+1
                samples[j,i+1,1] = n_glitch
                samples[j,i+1,2:] = new_point[:]
                #samples[j,i+1,2:2+(n_wavelet+1)*10] = new_point[:(n_wavelet+1)*10]
                #samples[j,i+1,2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[(n_wavelet+1)*10:(n_wavelet+1)*10+n_glitch*6]
                #samples[j,i+1,2+max_n_wavelet*10+max_n_glitch*6:] = new_point[(n_wavelet+1)*10+n_glitch*6:]
                a_yes[2,j] += 1
                log_likelihood[j,i+1] = log_L
                if ent_lnlike_test:
                    step_array[j].append('RJ_adding_accepted')
                    temp_entlike = ent_ptas[n_wavelet+1][n_glitch].get_lnlikelihood(remove_params(samples, j, i+1, wavelet_indx, glitch_indx, n_wavelet+1, max_n_wavelet, n_glitch, max_n_glitch))
                    ent_lnlikelihood[j, i+1] = temp_entlike
                    for kk in attributes_to_test:
                        likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                    #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "RJ"]))
                    #print("accept step dif: ",log_L-temp_entlike)
                QB_logl[j].save_values(accept_new_step=True)
                #FPI.n_wavelet = n_wavelet+1 #similar to save values in that it updates FPI when changing number of wavelets
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
            else:
                #print('rejected')
                samples[j,i+1,:] = samples[j,i,:]
                a_no[2,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
                ent_lnlikelihood[j,i+1] = ent_lnlikelihood[j,i]
                QB_logl[j].save_values(accept_new_step=False, rj_jump = True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
                if ent_lnlike_test:
                    step_array[j].append('RJ_adding_rejected')
                    for kk in attributes_to_test:
                        likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "RJ"]))

        elif n_wavelet==max_n_wavelet or (direction_decide>add_prob and n_wavelet!=min_n_wavelet):   #removing a wavelet----------------------------------------------------------
            print('removing wavelet')
            if j==0: rj_record.append(-1)
            #if j==0: print("Propose to remove a wavelet")
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

            # #Place last wavelet in place of wavelet being removed (won't be sampling over wavelet being removed)
            # new_point[wavelet_indx[remove_index,0]:wavelet_indx[remove_index,-1]+1] = np.copy(new_point[wavelet_indx[n_wavelet-1,0]:wavelet_indx[n_wavelet-1,-1]+1])
            # new_point[wavelet_indx[n_wavelet-1,0]:wavelet_indx[n_wavelet-1,-1]+1] = np.copy(samples_removed)


            log_L = QB_logl[j].M_N_RJ_helper(new_point, n_wavelet-1, n_glitch, remove_index = remove_index, wavelet_change = True)
            #print('Removing wavelet QB Log_L: ', log_L)
            #print('Removing wavelet ent Log_L: ', ent_ptas[n_wavelet-1][n_glitch].get_lnlikelihood(remove_params(new_point, 0, 0, wavelet_indx, glitch_indx, n_wavelet-1, max_n_wavelet, n_glitch, max_n_glitch, params_slice = True)))
            #print('rej_wave_logl: ', log_L)
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
                                                   FPI.wave_le_highs, n_wavelet-1, n_glitch, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch)
            #print('log_acc_ratio1: ', log_acc_ratio)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]

            #TODO: need to account for samples_current having one less glitch than in new_point
            #(will need new params in QB_FastPrior for n_glitch and n_wavelet)
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
            #print('log_acc_ratio2: ', log_acc_ratio)


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
            #print('acc_ratio: ', acc_ratio)
            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Ohhhhhhhhhhhhh")
                #print('accepted')
                samples[j,i+1,0] = n_wavelet-1
                samples[j,i+1,1] = n_glitch
                samples[j,i+1,2:] = new_point[:]
                a_yes[2,j] += 1
                log_likelihood[j,i+1] = log_L
                if ent_lnlike_test:
                    step_array[j].append('RJ_removing_accepted')
                    temp_entlike = ent_ptas[n_wavelet-1][n_glitch].get_lnlikelihood(remove_params(samples, j, i+1, wavelet_indx, glitch_indx, n_wavelet-1, max_n_wavelet, n_glitch, max_n_glitch))
                    ent_lnlikelihood[j, i+1] = temp_entlike
                    for kk in attributes_to_test:
                        likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                    #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "RJ"]))
                QB_logl[j].save_values(accept_new_step=True)
                #FPI.n_wavelet = n_wavelet-1
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
            else:
                #print('rejected')
                samples[j,i+1,:] = samples[j,i,:]
                a_no[2,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
                ent_lnlikelihood[j,i+1] = ent_lnlikelihood[j,i]
                QB_logl[j].save_values(accept_new_step=False, rj_jump = True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
                if ent_lnlike_test:
                    step_array[j].append('RJ_removing_rejected')
                    for kk in attributes_to_test:
                        likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "RJ"]))

def do_glitch_rj_move(n_chain, max_n_wavelet, max_n_glitch, n_glitch_prior, pta,  ent_ptas, FPI, QB_logl, likelihood_attributes, QB_Info, samples, i, betas, a_yes, a_no, vary_white_noise, num_noise_params, glitch_tau_scan_data, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx):
    TAU_list = list(glitch_tau_scan_data['tau_edges'])
    F0_list = glitch_tau_scan_data['f0_edges']
    T0_list = glitch_tau_scan_data['t0_edges']
    #print('glitch RJ')
    for j in range(n_chain):
        #print("-- ", j)
        n_wavelet = int(samples[j,i,0])#get_n_wavelet(samples, j, i)
        n_glitch = int(samples[j,i,1])#get_n_glitch(samples, j, i)

        add_prob = 0.5 #same propability of addind and removing
        #decide if we add or remove a signal
        direction_decide = np.random.uniform()
        if n_glitch==0 or (direction_decide<add_prob and n_glitch!=max_n_glitch): #adding a glitch------------------------------------------------------
            #pick which pulsar to add a glitch to
            print('adding glitch')
            psr_idx = np.random.choice(len(pta.pulsars), p=glitch_tau_scan_data['psr_idx_proposal'])

            #load in the appropriate tau-scan
            tau_scan = glitch_tau_scan_data['tau_scan'+str(psr_idx)]
            #print(i)
            tau_scan_limit = 0
            for TS in tau_scan:
                TS_max = np.max(TS)
                if TS_max>tau_scan_limit:
                    tau_scan_limit = TS_max
            #print(tau_scan_limit)

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

            prior_ext = pta.params[glitch_indx[0,2]].get_pdf(phase0_new) * pta.params[glitch_indx[0,1]].get_pdf(log10_h_new)# * ptas[0][1][gwb_on].params[3].get_pdf(float(psr_idx))

            samples_current = np.copy(samples[j, i, 2:])
            new_point = np.copy(samples[j,i,2:])#strip_samples(samples, j, i, n_wavelet+1, max_n_wavelet, n_glitch, max_n_glitch)
            # new_wavelet = np.array([cos_gwtheta_new, psi_new, gwphi_new, log_f0_new, log10_h_new, log10_h_cross_new,
            #                         phase0_new, phase0_cross_new, t0_new, tau_new])
            new_point[glitch_indx[n_glitch,0]] = log_f0_new
            new_point[glitch_indx[n_glitch,1]] = log10_h_new
            new_point[glitch_indx[n_glitch,2]] = phase0_new
            new_point[glitch_indx[n_glitch,3]] = psr_idx
            new_point[glitch_indx[n_glitch,4]] = t0_new
            new_point[glitch_indx[n_glitch,5]] = tau_new


            # log_L = ptas[n_wavelet][(n_glitch+1)][gwb_on].get_lnlikelihood(new_point)
            # log_acc_ratio = log_L*betas[j,i]
            # log_acc_ratio += ptas[n_wavelet][(n_glitch+1)][gwb_on].get_lnprior(new_point)
            # log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            # log_acc_ratio += -ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(samples_current)

            log_L = QB_logl[j].M_N_RJ_helper(new_point, n_wavelet, n_glitch+1, adding = True, glitch_change = True)
            #print('Adding glitch QB Log_L: ', log_L)
            #print('Adding glitch ent Log_L: ', ent_ptas[n_wavelet][n_glitch+1].get_lnlikelihood(remove_params(new_point, 0, 0, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch+1, max_n_glitch, params_slice = True)))
            #print('rej_wave_logl: ', log_L)
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
                                                   FPI.wave_le_highs, n_wavelet, n_glitch+1, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch)
            #print('QB_Fast_prior: ', QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   # FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   # FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   # FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   # FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   # FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   # FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   # FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   # FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   # FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   # FPI.wave_le_highs, n_wavelet, n_glitch+1, \
                                                   # FPI.max_n_wavelet, FPI.max_n_glitch))
            #print('Ent_Fast_prior: ', ent_ptas[n_wavelet][n_glitch+1].get_lnprior(remove_params(new_point, 0, 0, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch+1, max_n_glitch,params_slice = True)))

            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]

            #TODO: need to account for samples_current having one less glitch than in new_point
            #(will need new params in QB_FastPrior for n_glitch and n_wavelet)
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
            #print('QB_Fast_prior_old: ', QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   # FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   # FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   # FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   # FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   # FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   # FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   # FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   # FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   # FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   # FPI.wave_le_highs, n_wavelet,n_glitch, \
                                                   # FPI.max_n_wavelet, FPI.max_n_glitch))
            #print('Ent_Fast_prior_old: ', ent_ptas[n_wavelet][n_glitch].get_lnprior(remove_params(samples_current, 0, 0, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch,params_slice = True)))

            #apply normalization
            tau_scan_new_point_normalized = tau_scan_new_point/glitch_tau_scan_data['norm'+str(psr_idx)]
            #print('prior_ext: ', prior_ext)
            #print('tau_scan_old_point_normalized: ', tau_scan_new_point_normalized)
            #print('psr_idx_proposal: ',glitch_tau_scan_data['psr_idx_proposal'][int(np.round(psr_idx))])
            acc_ratio = np.exp(log_acc_ratio)/prior_ext/tau_scan_new_point_normalized/glitch_tau_scan_data['psr_idx_proposal'][int(np.round(psr_idx))]
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_glitch==0:
                acc_ratio *= 0.5
            if n_glitch==max_n_glitch-1:
                acc_ratio *= 2.0
            #accounting for n_glitch prior
            acc_ratio *= n_glitch_prior[int(n_glitch)+1]/n_glitch_prior[int(n_glitch)]
            #print('acc_ratio: ', acc_ratio)
            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Yeeeh")
                #print('accepted')
                samples[j,i+1,0] = n_wavelet
                samples[j,i+1,1] = n_glitch+1
                samples[j,i+1,2:] = new_point[:]
                a_yes[0,j] += 1
                log_likelihood[j,i+1] = log_L
                if ent_lnlike_test:
                    step_array[j].append('glitch_RJ_adding_accepted')
                    temp_entlike = ent_ptas[n_wavelet][n_glitch+1].get_lnlikelihood(remove_params(samples, j, i+1, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch+1, max_n_glitch))
                    ent_lnlikelihood[j, i+1] = temp_entlike
                    for kk in attributes_to_test:
                        likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                    #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, 'glitch RJ']))
                    #print('glitch RJ jump QB_logl[{}].NN: '.format(j), QB_logl[j].NN)
                QB_logl[j].save_values(accept_new_step=True)
                #FPI.n_glitch = n_glitch+1
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
            else:
                #print('rejected')
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
                ent_lnlikelihood[j,i+1] = ent_lnlikelihood[j,i]
                QB_logl[j].save_values(accept_new_step=False, rj_jump = True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
                if ent_lnlike_test:
                    step_array[j].append('glitch_RJ_adding_rejected')
                    for kk in attributes_to_test:
                        likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, 'glitch RJ']))
                #print('glitch RJ jump QB_logl[{}].NN: '.format(j), QB_logl[j].NN)

        elif n_glitch==max_n_glitch or (direction_decide>add_prob and n_glitch!=0):   #removing a glitch----------------------------------------------------------
            #choose which glitch to remove
            print('removing glitch')
            remove_index = np.random.randint(n_glitch)

            samples_current = np.copy(samples[j,i,2:])#strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
            samples_removed = np.copy(samples_current[glitch_indx[remove_index,0]:glitch_indx[remove_index,-1]+1]) #copy of the glitch to remove
            glitch_params_coppied = np.copy(samples_current[glitch_indx[0,0]:glitch_indx[max_n_glitch-1,-1]+1]) #copy of all wavelet params
            new_point = np.copy(samples_current)

            glitch_params_new = np.delete(glitch_params_coppied,list(range(remove_index*6,remove_index*6+6)))
            glitch_params_new = np.append(glitch_params_new, samples_removed)
            new_point[glitch_indx[0,0]:glitch_indx[max_n_glitch-1,-1]+1] = np.copy(glitch_params_new) #arranged so removed glitch is shifted to the end and all following wavelets are shifted over

            # #Place last wavelet in place of wavelet being removed (won't be sampling over wavelet being removed)
            # new_point[glitch_indx[remove_index,0]:glitch_indx[remove_index,-1]+1] = np.copy(new_point[glitch_indx[n_wavelet-1,0]:glitch_indx[n_wavelet-1,-1]+1])
            # new_point[glitch_indx[n_wavelet-1,0]:glitch_indx[n_wavelet-1,-1]+1] = np.copy(samples_removed)

            log_L = QB_logl[j].M_N_RJ_helper(new_point, n_wavelet, n_glitch-1, remove_index = remove_index, glitch_change = True)
            #print('Removing glitch QB Log_L: ', log_L)
            #print('Removing glitch ent Log_L: ', ent_ptas[n_wavelet][n_glitch-1].get_lnlikelihood(remove_params(new_point, 0, 0, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch-1, max_n_glitch, params_slice = True)))
            #print('rej_wave_logl: ', log_L)
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
                                                   FPI.wave_le_highs, n_wavelet,n_glitch-1, \
                                                   FPI.max_n_wavelet, FPI.max_n_glitch)
            #print('log_acc_ratio1: ', log_acc_ratio)
            #print('QB_Fast_prior: ', QB_FastPrior.get_lnprior_helper(new_point, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   # FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   # FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   # FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   # FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   # FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   # FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   # FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   # FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   # FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   # FPI.wave_le_highs, n_wavelet, n_glitch-1, \
                                                   # FPI.max_n_wavelet, FPI.max_n_glitch))
            #print('Before remove_params: ', np.shape(new_point))
            #print('Ent_Fast_prior: ', ent_ptas[n_wavelet][n_glitch-1].get_lnprior(remove_params(new_point, 0, 0, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch-1, max_n_glitch,params_slice = True)))
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]

            #TODO: need to account for samples_current having one less glitch than in new_point
            #(will need new params in QB_FastPrior for n_glitch and n_wavelet)
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
            #print('log_acc_ratio2: ', log_acc_ratio)
            #print('QB_Fast_prior_old: ', QB_FastPrior.get_lnprior_helper(samples_current, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                                   # FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                                   # FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                                   # FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,FPI.px_par_ids,\
                                                   # FPI.px_mus, FPI.px_errs, FPI.global_common, \
                                                   # FPI.glitch_uf_par_ids, FPI.glitch_uf_lows, \
                                                   # FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                                   # FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                                   # FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                                   # FPI.wave_uf_highs, FPI.wave_le_par_ids,FPI.wave_le_lows, \
                                                   # FPI.wave_le_highs, n_wavelet,n_glitch, \
                                                   # FPI.max_n_wavelet, FPI.max_n_glitch))
            #print('Before remove_params: ', np.shape(samples_current))
            #print('Ent_Fast_prior_old: ', ent_ptas[n_wavelet][n_glitch].get_lnprior(remove_params(samples_current, 0, 0, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch,params_slice = True)))
            #getting old parameters
            tau_old = samples[j,i,2+glitch_indx[remove_index,5]]
            f0_old = 10**samples[j,i,2+glitch_indx[remove_index,0]]
            t0_old = samples[j,i,2+glitch_indx[remove_index,4]]
            log10_h_old = samples[j,i,2+glitch_indx[remove_index,1]]
            phase0_old = samples[j,i,2+glitch_indx[remove_index,2]]

            #get old psr index and load in appropriate tau scan
            psr_idx_old = samples[j,i,2+glitch_indx[remove_index,3]]
            #print('psr_idx_old: ', psr_idx_old)
            tau_scan_old = glitch_tau_scan_data['tau_scan'+str(int(np.round(psr_idx_old)))]
            tau_scan_limit_old = 0
            for TS in tau_scan_old:
                TS_max = np.max(TS)
                if TS_max>tau_scan_limit_old:
                    tau_scan_limit_old = TS_max
            #print(tau_scan_limit_old)

            #getting tau_scan at old point
            tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
            f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
            t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

            #print(tau_old, TAU_list)
            #print(tau_idx_old, f0_idx_old, t0_idx_old)

            tau_scan_old_point = tau_scan_old[tau_idx_old][f0_idx_old, t0_idx_old]

            #apply normalization
            tau_scan_old_point_normalized = tau_scan_old_point/glitch_tau_scan_data['norm'+str(int(np.round(psr_idx_old)))]

            prior_ext = pta.params[glitch_indx[0,2]].get_pdf(phase0_old) * pta.params[glitch_indx[0,1]].get_pdf(log10_h_old)# * ptas[0][1][gwb_on].params[3].get_pdf(psr_idx_old)
            #print('prior_ext: ', prior_ext)
            #print('tau_scan_old_point_normalized: ', tau_scan_old_point_normalized)
            #print('psr_idx_proposal: ',glitch_tau_scan_data['psr_idx_proposal'][int(np.round(psr_idx_old))])
            acc_ratio = np.exp(log_acc_ratio)*prior_ext*tau_scan_old_point_normalized*glitch_tau_scan_data['psr_idx_proposal'][int(np.round(psr_idx_old))]
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_glitch==1:
                acc_ratio *= 2.0
            if n_glitch==max_n_glitch:
                acc_ratio *= 0.5
            #accounting for n_glitch prior
            acc_ratio *= n_glitch_prior[int(n_glitch)-1]/n_glitch_prior[int(n_glitch)]
            #print('acc_ratio: ', acc_ratio)
            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Ohhhhhhhhhhhhh")
                #print('accepted')
                samples[j,i+1,0] = n_wavelet
                samples[j,i+1,1] = n_glitch-1
                samples[j,i+1,2:] = new_point[:]
                a_yes[0,j] += 1
                log_likelihood[j,i+1] = log_L
                if ent_lnlike_test:
                    step_array[j].append('glitch_RJ_removing_accepted')
                    temp_entlike = ent_ptas[n_wavelet][n_glitch-1].get_lnlikelihood(remove_params(samples, j, i+1, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch-1, max_n_glitch))
                    ent_lnlikelihood[j, i+1] = temp_entlike
                    for kk in attributes_to_test:
                        likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                    #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, 'glitch RJ']))
                    #print('glitch RJ jump QB_logl[{}].NN: '.format(j), QB_logl[j].NN)
                QB_logl[j].save_values(accept_new_step=True)
                #FPI.n_glitch = n_glitch-1
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
                #These prints should match if things are working
                #print('Accepted number of glitches: ', n_glitch-1)
                #print('Saved number of glitches: ', QB_logl[j].Nglitch)
            else:
                #print('rejected')
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
                ent_lnlikelihood[j,i+1] = ent_lnlikelihood[j,i]
                QB_logl[j].save_values(accept_new_step=False, rj_jump = True)
                QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)

                #These prints should match if things are working
                #print('Rejected number of glitches: ', n_glitch-1)
                #print('Saved number of glitches: ', QB_logl[j].Nglitch)
                if ent_lnlike_test:
                    step_array[j].append('glitch_RJ_removing_rejected')
                    for kk in attributes_to_test:
                        likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, 'glitch RJ']))
                #print('glitch RJ jump QB_logl[{}].NN: '.format(j), QB_logl[j].NN)


################################################################################
#
#NOISE MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN WHITE NOISE PARAMETERS)
#
################################################################################
def noise_jump(n_chain, max_n_wavelet, max_n_glitch, pta, ent_ptas, FPI, QB_logl, likelihood_attributes, QB_Info, samples, i, betas, a_yes, a_no, eig_per_psr, per_puls_indx, num_noise_params, vary_white_noise, vary_rn, log_likelihood, ent_lnlikelihood, ent_lnlike_test, attributes_to_test, step_array, wavelet_indx, glitch_indx, noise_steps):
    print("NOISE")
    for j in range(n_chain):
        n_wavelet = int(samples[j,i,0]) #get_n_wavelet(samples, j, i) # samples[j, i]
        n_glitch = int(samples[j,i,1]) #get_n_glitch(samples, j, i)

        samples_current = np.copy(samples[j,i,2:])
        #do the wn jump
        jump_select = np.random.randint(eig_per_psr.shape[1])
        jump_wn = eig_per_psr[j,jump_select,:]

        jump = np.zeros(samples_current.size)
        param_count = 0

        #Loop through all pulsars and pulsar noise params
        for ii in range(len(per_puls_indx)):
            for jj in range(len(per_puls_indx[ii])):
                #Jump through noise params (which should correspond to noise eigenvector indexes)
                #if param_count < num_noise_params:
                jump[per_puls_indx[ii][jj]] = jump_wn[param_count]
                param_count += 1

        new_point = samples_current + jump*np.random.normal()

        #Random prior draw during 10% of noise jumps
        if np.random.uniform() < 0.1:
            #Pick random pulsar
            pulsar_idx = np.random.randint(len(per_puls_indx))

            #Draw random value for one pulsar for all its noise params
            prior_draws = []
            idx = []
            for u in range(len(per_puls_indx[pulsar_idx])):
                idx.append(per_puls_indx[pulsar_idx][u])
                prior_draws.append(pta.params[idx[u]].sample())

            #Get idx for each pulsar noise param
            for z in range(len(idx)):
                new_point[idx[z]] = prior_draws[z]


        log_L = QB_logl[j].get_lnlikelihood(new_point, vary_white_noise = vary_white_noise, vary_red_noise = vary_rn) #why was vary_red_noise false
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

        acc_ratio = np.exp(log_acc_ratio)
        #print('Noise acc_ratio: ',acc_ratio)
        if np.random.uniform()<=acc_ratio:
            noise_steps += jump_wn*10
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            a_yes[8,j]+=1
            log_likelihood[j,i+1] = log_L
            if ent_lnlike_test:
                step_array[j].append('NOISE_accepted')
                temp_entlike = ent_ptas[n_wavelet][n_glitch].get_lnlikelihood(remove_params(samples, j, i+1, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch))
                ent_lnlikelihood[j, i+1] = temp_entlike
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
                #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "NOISE"]))
                #print('Noise jump QB_logl[{}].NN: '.format(j), QB_logl[j].NN)

            #print("accept step")
            QB_logl[j].save_values(accept_new_step=True, vary_white_noise = vary_white_noise, vary_red_noise = vary_rn)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[8,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            ent_lnlikelihood[j,i+1] = ent_lnlikelihood[j,i]
            #print("reject step")
            QB_logl[j].save_values(accept_new_step=False, vary_white_noise = vary_white_noise, vary_red_noise = vary_rn)
            QB_Info[j].load_parameters(QB_logl[j].resres_logdet, QB_logl[j].Nglitch, QB_logl[j].Nwavelet, QB_logl[j].wavelet_prm, QB_logl[j].glitch_prm, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].glitch_pulsars)
            if ent_lnlike_test:
                step_array[j].append('NOISE_rejected')
                for kk in attributes_to_test:
                    likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
            #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "NOISE"]))
            #print('Noise jump QB_logl[{}].NN: '.format(j), QB_logl[j].NN)

################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, max_n_wavelet, max_n_glitch, pta, FPI, QB_logl, likelihood_attributes, QB_Info, samples, i, betas, a_yes, a_no, swap_record, vary_white_noise, num_noise_params, log_likelihood, ent_lnlikelihood, PT_hist, PT_hist_idx, ent_lnlike_test, attributes_to_test, step_array):
    print("SWAP")

    #set up map to help keep track of swaps
    swap_map = list(range(n_chain))

    #get log_Ls from all the chains
    log_Ls = []
    ent_logLs = []
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
            a_yes[5,swap_chain]+=1
            #PT_hist[swap_chain,PT_hist_idx[0]%PT_hist.shape[1]] = 1.0
            swap_record[i] = swap_chain
            #print('Swapping chains {0} and {1}'.format(swap_map[swap_chain], swap_map[swap_chain+1]))
            if ent_lnlike_test:
                step_array[j].append('PT_SWAP_accepted')
        else:
            a_no[5,swap_chain]+=1
            #PT_hist[swap_chain,PT_hist_idx[0]%PT_hist.shape[1]] = 0.0
            if ent_lnlike_test:
                step_array[j].append('PT_SWAP_rejected')

    PT_hist_idx += 1
    QB_logl_map = []
    QB_Info_map = []
    for j in range(n_chain):
        print('Swapping chain {0} w/ chain {1}'.format(j, swap_map[j]))
        QB_logl_map.append(QB_logl[swap_map[j]])
        QB_Info_map.append(QB_Info[swap_map[j]])

    #loop through the chains and record the new samples and log_Ls
    for j in range(n_chain):
        QB_logl[j] = QB_logl_map[j]
        QB_Info[j] = QB_Info_map[j]
        samples[j,i+1,:] = samples[swap_map[j],i,:]
        log_likelihood[j,i+1] = log_likelihood[swap_map[j],i]
        ent_lnlikelihood[j, i+1] = ent_lnlikelihood[swap_map[j], i]

        #Can leave normal indexing, since QB_logl objects are switched.
        #print('QB_logl[j].resres_logdet: ',QB_logl[j].resres_logdet)
        #print('QB_Info[j].resres_logdet: ',QB_Info[j].resres_logdet)
        if ent_lnlike_test:
            for kk in attributes_to_test:
                likelihood_attributes['chain_{}'.format(j)][kk].append(np.copy(getattr(QB_logl[j], kk)))
        #likelihood_attributes[j].append(np.array([QB_logl[j].resres_logdet, QB_logl[j].Nvecs_previous, QB_logl[j].Nvecs, QB_logl[j].glitch_prm, QB_logl[j].wavelet_prm, QB_logl[j].glitch_pulsars, QB_logl[j].params, QB_logl[j].MMs, QB_logl[j].NN, QB_logl[j].sigmas, "SWAP"]))

################################################################################
#
#FISHER EIGENVALUE CALCULATION
#
################################################################################
def get_fisher_eigenvectors(params, pta, QB_FP, QB_logl, T_chain=1, epsilon=1e-4, n_sources=1, dim=10, array_index=None, use_prior=False, flag = False, vary_white_noise = False, vary_psr_red_noise = False, vary_rn = False):#offset=0, use_prior=False):
    #print('FISHER STEP')
    n_source=n_sources # this needs to not be used for the non-wavelet/glitch indexing (set to 1)
    eig = []

    index_rows = len(array_index)
    if flag or vary_rn:
        fisher = np.zeros((n_source,dim,dim))
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

            #correct for the given temperature of the chain
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
                eig.append( np.array(False) )

    #Run this if not doing wavelet/glitch stuff
    elif vary_psr_red_noise or vary_white_noise:
        #diagonal terms in fisher matrices
        for n in range(index_rows):
            dim = len(array_index[n])
            #print('Dimensions!!!!!: ', dim)
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
                    pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise, vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsPP)
                    mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise, vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsMM)
                else:
                    pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise, vary_psr_red_noise, no_step = True)
                    mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise, vary_psr_red_noise, no_step = True)

                #calculate diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                fisher[1,i+n*dim,i+n*dim] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)
        for n in range(index_rows):
            #calculate off-diagonal elements
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
                        pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise, vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsPP)
                        mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise, vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsMM)
                        pm = QB_logl.get_lnlikelihood(paramsPM, vary_white_noise, vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsPM)
                        mp = QB_logl.get_lnlikelihood(paramsMP, vary_white_noise, vary_psr_red_noise, no_step = True) + QB_FP.get_lnprior(paramsMP)
                    else:
                        pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise, vary_psr_red_noise, no_step = True)
                        mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise, vary_psr_red_noise, no_step = True)
                        pm = QB_logl.get_lnlikelihood(paramsPM, vary_white_noise, vary_psr_red_noise, no_step = True)
                        mp = QB_logl.get_lnlikelihood(paramsMP, vary_white_noise, vary_psr_red_noise, no_step = True)

                    #calculate off-diagonal elements of the Hessian from a central finite element scheme
                    #note the minus sign compared to the regular Hessian
                    fisher[1,i+n*dim,j+n*dim] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                    fisher[1,j+n*dim,i+n*dim] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
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
            eig.append(np.array(False))

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
                pp = QB_logl.get_lnlikelihood(paramsPP, vary_rn, no_step = True) + QB_FP.get_lnprior(paramsPP)
                mm = QB_logl.get_lnlikelihood(paramsMM, vary_rn, no_step = True) + QB_FP.get_lnprior(paramsMM)
            else:
                pp = QB_logl.get_lnlikelihood(paramsPP, vary_rn, no_step = True)
                mm = QB_logl.get_lnlikelihood(paramsMM, vary_rn, no_step = True)

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
        #correct for the given temperature of the chain
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
            eig.append( np.array(False) )

    return np.array(eig)


################################################################################
#
#FUNCTION TO EASILY SET UP A LIST OF PTA OBJECTS
#
################################################################################
def get_pta(pulsars, vary_white_noise=True, include_equad = False, include_ecorr = False, wn_backend_selection=False, noisedict=None, include_rn=True, vary_rn=True, include_per_psr_rn=False, vary_per_psr_rn=False, max_n_wavelet=1, efac_start=1.0, rn_amp_prior='uniform', rn_log_amp_range=[-18,-11], rn_params=[-14.0,1.0], gwb_amp_prior='uniform', gwb_log_amp_range=[-18,-11], wavelet_amp_prior='uniform', wavelet_log_amp_range=[-18,-11], per_psr_rn_amp_prior='uniform', per_psr_rn_log_amp_range=[-18,-11], prior_recovery=False, ent_lnlike_test = False, max_n_glitch=1, glitch_amp_prior='uniform', glitch_log_amp_range=[-18, -11], t0_min=0.0, t0_max=10.0, f0_min=3.5e-9, f0_max=1e-7, TF_prior=None, use_svd_for_timing_gp=True, tref=53000*86400):
    #setting up base model
    if vary_white_noise:
        efac = parameter.Uniform(0.01, 10.0)
        if include_equad:
            equad = parameter.Uniform(-8.5, -5)
        if include_ecorr:
            ecorr = parameter.Uniform(-8.5, -5)

    else:
        #print('Constant efac!!')
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
            ef = white_signals.MeasurementNoise(efac=efac)
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
        else:# vary_per_psr_rn #== 'constant':
            #print('constant, non varying per pulsar rn!')
            '''
            #CURRENTLY DOES NOT SET AMPS/GAMMA VALUES
            #This defaults to val=None (value should be set later)
            #I'm also not sure this will ever be used, since we will want to see what per pulsar rn
            #gets put into wavelets (at least I think that's how this will work) ~ Jacob
            '''
            #For now, set amplitude to median of per_psr_rn_amp_prior and gamma to 3.2 (max likelihood in 15yr GWB search)
            log10_A = parameter.Constant(val = (per_psr_rn_log_amp_range[1]-per_psr_rn_log_amp_range[0])/2)
            gamma = parameter.Constant(val = 3.2)

        #This should be setting amplitude and gamma to default values
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        per_psr_rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan)

    #adding red noise if included
    if include_rn:
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)

        if vary_rn:
            #rn = ext_models.common_red_noise_block(prior='uniform', Tspan=Tspan, name='com_rn')
            amp_name = 'com_rn_log10_A'
            if rn_amp_prior == 'uniform':
                log10_Arn = parameter.LinearExp(rn_log_amp_range[0], rn_log_amp_range[1])(amp_name)
            elif rn_amp_prior == 'log-uniform':
                log10_Arn = parameter.Uniform(rn_log_amp_range[0], rn_log_amp_range[1])(amp_name)
            gam_name = 'com_rn_gamma'
            gamma_rn = parameter.Uniform(0, 7)(gam_name)
            pl = utils.powerlaw(log10_A=log10_Arn, gamma=gamma_rn)
            rn = gp_signals.FourierBasisGP(spectrum=pl, coefficients=False, components=30, Tspan=Tspan,
                                           modes=None, name='com_rn')
        else:
            #Why these values for the common process? rn_params is hard coded in run_bhb() as rn_params = [-13.0, 1.0]
            log10_A = parameter.Constant(rn_params[0])
            gamma = parameter.Constant(rn_params[1])
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

    #wavelet models
    wavelets = []
    for i in range(max_n_wavelet):
        log10_f0 = parameter.Uniform(np.log10(f0_min), np.log10(f0_max))("wavelet_"+str(i)+'_'+'log10_f0')
        cos_gwtheta = parameter.Uniform(-1, 1)("wavelet_"+ str(i)+'_'+'cos_gwtheta')
        gwphi = parameter.Uniform(0, 2*np.pi)("wavelet_" + str(i)+'_'+'gwphi')
        psi = parameter.Uniform(0, np.pi)("wavelet_" + str(i)+'_'+'gw_psi')
        phase0 = parameter.Uniform(0, 2*np.pi)("wavelet_" + str(i)+'_'+'phase0')
        phase0_cross = parameter.Uniform(0, 2*np.pi)("wavelet_" + str(i)+'_'+'phase0_cross')
        tau = parameter.Uniform(0.2, 5)("wavelet_" + str(i)+'_'+'tau')
        t0 = parameter.Uniform(t0_min, t0_max)("wavelet_" + str(i)+'_'+'t0')
        if wavelet_amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(wavelet_log_amp_range[0], wavelet_log_amp_range[1])("wavelet_" + str(i)+'_'+'log10_h')
            log10_h_cross = parameter.Uniform(wavelet_log_amp_range[0], wavelet_log_amp_range[1])("wavelet_" + str(i)+'_'+'log10_h_cross')
        elif wavelet_amp_prior == 'uniform':
            log10_h = parameter.LinearExp(wavelet_log_amp_range[0], wavelet_log_amp_range[1])("wavelet_" + str(i)+'_'+'log10_h')
            log10_h_cross = parameter.LinearExp(wavelet_log_amp_range[0], wavelet_log_amp_range[1])("wavelet_" + str(i)+'_'+'log10_h_cross')
        else:
            print("CW amplitude prior of {0} not available".format(cw_amp_prior))
        wavelet_wf = models.wavelet_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_h = log10_h, log10_h2=log10_h_cross,
                                          tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0, phase02=phase0_cross,
                                          epsilon = None, psi=psi, tref=tref)
        wavelets.append(deterministic_signals.Deterministic(wavelet_wf, name='wavelet'+str(i)))
    #glitch models
    glitches = []
    for i in range(max_n_glitch):
        log10_f0 = parameter.Uniform(np.log10(f0_min), np.log10(f0_max))("Glitch_"+str(i)+'_'+'log10_f0')
        phase0 = parameter.Uniform(0, 2*np.pi)("Glitch_"+str(i)+'_'+'phase0')
        tau = parameter.Uniform(0.2, 5)("Glitch_"+str(i)+'_'+'tau')
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
    if vary_rn:
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

    #enterprise ptas for testing against QuickBurst
    if ent_lnlike_test:
        ent_ptas = []
        for n_wavelets in range(max_n_wavelet+1):
            glitch_sub_ptas = []
            for n_glitches in range(max_n_glitch+1):
                #setting up the proper model
                s_ent = tm + wn
                if include_per_psr_rn:
                    s_ent += per_psr_rn
                if vary_rn:
                    s_ent += rn

                for ii in range(n_glitches):
                    #print('ohhh', ii)
                    s_ent += glitches[ii]
                for jj in range(n_wavelets):
                    #print('yeahhhh', jj)
                    s_ent += wavelets[jj]

                model_ent = []
                for p in pulsars:
                    model_ent.append(s_ent(p))

                #set the likelihood to unity if we are in prior recovery mode
                if prior_recovery:
                    if TF_prior is None:
                        glitch_sub_ptas.append(get_prior_recovery_pta(signal_base.PTA(model_ent)))
                    else:
                        glitch_sub_ptas.append(get_tf_prior_pta(signal_base.PTA(model_ent), TF_prior, n_wavelets, prior_recovery=True))
                elif noisedict is not None:
                    if isinstance(noisedict, str):
                        with open(noisedict, 'r') as fp:
                            noisedict_file = json.load(fp)
                            ent_pta = signal_base.PTA(model_ent)
                            ent_pta.set_default_params(noisedict_file)
                    else:
                        ent_pta = signal_base.PTA(model_ent)
                        ent_pta.set_default_params(noisedict)
                        if TF_prior is None:
                            glitch_sub_ptas.append(ent_pta)
                        else:
                            glitch_sub_ptas.append(get_tf_prior_pta(ent_pta, TF_prior, n_wavelets))
                else:
                    if TF_prior is None:
                        glitch_sub_ptas.append(signal_base.PTA(model_ent))
                    else:
                        glitch_sub_ptas.append(get_tf_prior_pta(signal_base.PTA(model_ent), TF_prior, n_wavelets))

            ent_ptas.append(glitch_sub_ptas)
    else:
        ent_ptas = None

    #print('ent_ptas.size: ', np.shape(ent_ptas))
    # if ent_lnlike_test:
    #     print('ent_pta params: ', ent_ptas[max_n_wavelet][max_n_glitch].params)
    #print('QB pta params: ', pta.params)
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

    #To account for backend/receiver combos, need to make per_puls_indx a list of lists
    #and append a list of indexes for each pulsar, which will be of varying size. I've commented
    #some mock code that could be a template for this.

    # number of wn params for each pulsar
    num_per_puls_param_list = []
    #List of lists of all wn params per pulsar
    per_puls_indx = []
    #For each pulsar
    for i in range(len(pulsars)):
        param_list = pta.pulsarmodels[i].param_names
        psr_wn_indx = []
        #Search through all parameters to get indexes for rn and wn params for each pulsar
        for ct, par in enumerate(param_list):
            #Skip common rn terms
            if pulsars[i].name in par:
                if 'ecorr' in par or 'efac' in par or 'equad' in par or 'log10_A' in par or 'gamma' in par:

                    #get indexes for each pulsar from overall pta params
                    psr_wn_indx.append(key_list.index(par))

        #append to overall list of lists
        per_puls_indx.append(psr_wn_indx)
        num_per_puls_param_list.append(len(psr_wn_indx))

    #Generate the lnPrior object for this PTA
    QB_FP = QB_FastPrior.FastPrior(pta, pulsars)
    QB_FPI = QB_FastPrior.get_FastPriorInfo(pta, pulsars, max_n_glitch, max_n_wavelet)
    #print('glitch_uf_lows: ',QB_FP.glitch_uf_lows)
    #print('glitch_uf_highs: ',QB_FP.glitch_uf_highs)
    rn_indx = np.zeros((2), dtype = 'int')
    if vary_rn:
        rn_indx[0] = key_list.index('com_rn_gamma')
        rn_indx[1] = key_list.index('com_rn_log10_A')

    return pta, ent_ptas, QB_FP, QB_FPI, glitch_indx, wavelet_indx, per_puls_indx, rn_indx, num_per_puls_param_list

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
                #print(t0, log10_f0)
                t_idx = int( np.digitize(t0, TF_prior['t_bins']) )
                f_idx = int( np.digitize(log10_f0, TF_prior['lf_bins']) )
                #print((t_idx, f_idx))
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
# def get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params):
#     return int(samples[j,i,2+max_n_wavelet*10+max_n_glitch*6+num_noise_params]!=0.0)

def remove_params(samples, j, i, wavelet_indx, glitch_indx, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch, params_slice = False):
    #"Special" indexing for handling dumb edge cases when max_n_wavelet = n_wavelet or max_n_glitch = n_glitch
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
    #print('Range of parameters to delete: ', list(range(wave_start,wave_end))+list(range(glitch_start, glitch_end)), '\n')
    if params_slice:
        #print('Remove params shape: ', np.shape(np.delete(samples, list(range(wave_start,wave_end))+list(range(glitch_start, glitch_end)))))
        return np.delete(samples, list(range(wave_start,wave_end))+list(range(glitch_start, glitch_end)))
    else:
        return np.delete(samples[j,i,2:], list(range(wave_start,wave_end))+list(range(glitch_start, glitch_end)))


'''why'''
# def get_n_wavelet(samples, j, i):
#     return int(samples[j,i,0])
#
# def get_n_glitch(samples, j, i):
#     return int(samples[j,i,1])

################################################################################
#
#MATCH CALCULATION ROUTINES
#
################################################################################

def get_similarity_matrix(pta, delays_list, noise_param_dict=None):

    if noise_param_dict is None:
        print('No noise dictionary provided!...')
    else:
        pta.set_default_params(noise_param_dict)

    #print(pta.summary())

    phiinvs = pta.get_phiinv([], logdet=False)
    TNTs = pta.get_TNT([])
    Ts = pta.get_basis()
    Nvecs = pta.get_ndiag([])
    Nmats = [ Fe_statistic.make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, Nvecs, Ts)]

    n_wf = len(delays_list)

    S = np.zeros((n_wf,n_wf))
    for idx, (psr, Nmat, TNT, phiinv, T) in enumerate(zip(pta.pulsars, Nmats,
                                                          TNTs, phiinvs, Ts)):
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

        for i in range(n_wf):
            for j in range(n_wf):
                delay_i = delays_list[i][idx]
                delay_j = delays_list[j][idx]
                #print(delay_i)
                #print(Nmat)
                #print(Nmat, T, Sigma)
                S[i,j] += Fe_statistic.innerProduct_rr(delay_i, delay_j, Nmat, T, Sigma)
    return S

def get_match_matrix(pta, delays_list, noise_param_dict=None):
    S = get_similarity_matrix(pta, delays_list, noise_param_dict=noise_param_dict)

    M = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            M[i,j] = S[i,j]/np.sqrt(S[i,i]*S[j,j])
    return M
