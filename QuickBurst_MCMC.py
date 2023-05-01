################################################################################
#
#BayesWavePTA -- Bayesian search for burst GW signals in PTA data based on the BayesWave algorithm
#
#Bence Bécsy (bencebecsy@montana.edu) -- 2020
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import json

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

import Fast_Burst_likelihood as Quickburst
import line_profiler

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################
#@profile
def run_bhb(N_slow, T_max, n_chain, pulsars, max_n_wavelet=1, min_n_wavelet=0, n_wavelet_prior='flat', n_wavelet_start='random', RJ_weight=0, glitch_RJ_weight=0,
            regular_weight=3, noise_jump_weight=3, PT_swap_weight=1, T_ladder=None, T_dynamic=False, T_dynamic_nu=300, T_dynamic_t0=1000, PT_hist_length=100,
            tau_scan_proposal_weight=0, tau_scan_file=None, draw_from_prior_weight=0,
            de_weight=0, prior_recovery=False, wavelet_amp_prior='uniform', gwb_amp_prior='uniform', rn_amp_prior='uniform', per_psr_rn_amp_prior='uniform',
            gwb_log_amp_range=[-18,-11], rn_log_amp_range=[-18,-11], per_psr_rn_log_amp_range=[-18,-11], wavelet_log_amp_range=[-18,-11],
            vary_white_noise=False, efac_start=None, include_equad=False, include_ecorr = False, wn_backend_selection=False, noisedict_file=None,
            include_gwb=False, gwb_switch_weight=0,
            include_rn=False, vary_rn=False, num_total_wn_params=None, rn_params=[-13.0,1.0], include_per_psr_rn=False, vary_per_psr_rn=False, per_psr_rn_start_file=None,
            jupyter_notebook=False, gwb_on_prior=0.5,
            max_n_glitch=1, glitch_amp_prior='uniform', glitch_log_amp_range=[-18, -11], n_glitch_prior='flat', n_glitch_start='random', t0_min=0.0, t0_max=10.0, tref=53000*86400,
            glitch_tau_scan_proposal_weight=0, glitch_tau_scan_file=None, TF_prior_file=None, f0_min=3.5e-9, f0_max=1e-7,
            save_every_n=10, savefile=None, safe_save=False, resume_from=None, start_from=None, n_status_update=100, n_fish_update=1000, n_fast_to_slow=1000):

    #scale steps to slow steps
    N = N_slow*n_fast_to_slow
    n_status_update = n_status_update*n_fast_to_slow
    n_fish_update = n_fish_update*n_fast_to_slow
    save_every_n = save_every_n*n_fast_to_slow
    #If no wn or rn variance, shouldn't do any noise jumps
    if not vary_white_noise:
        if not vary_per_psr_rn:
            noise_jump_weight = 0
    if TF_prior_file is None:
        TF_prior = None
    else:
        with open(TF_prior_file, 'rb') as f:
            TF_prior = pickle.load(f)
    pta, glitch_indx, wavelet_indx, per_puls_indx, rn_indx, gwb_indx, num_per_puls_param_list = get_pta(pulsars, vary_white_noise=vary_white_noise, include_equad=include_equad, include_ecorr = include_ecorr, wn_backend_selection=wn_backend_selection, noisedict_file=noisedict_file, include_rn=include_rn, vary_rn=vary_rn, include_per_psr_rn=include_per_psr_rn, vary_per_psr_rn=vary_per_psr_rn, include_gwb=include_gwb, max_n_wavelet=max_n_wavelet, efac_start=efac_start, rn_amp_prior=rn_amp_prior, rn_log_amp_range=rn_log_amp_range, rn_params=rn_params, per_psr_rn_amp_prior=per_psr_rn_amp_prior, per_psr_rn_log_amp_range=per_psr_rn_log_amp_range, gwb_amp_prior=gwb_amp_prior, gwb_log_amp_range=gwb_log_amp_range, wavelet_amp_prior=wavelet_amp_prior, wavelet_log_amp_range=wavelet_log_amp_range, prior_recovery=prior_recovery, max_n_glitch=max_n_glitch, glitch_amp_prior=glitch_amp_prior, glitch_log_amp_range=glitch_log_amp_range, t0_min=t0_min, t0_max=t0_max, f0_min=f0_min, f0_max=f0_max, TF_prior=TF_prior, tref=tref)

    '''
    #print(ptas)
    for i in range(len(ptas)):
        for j in range(len(ptas[i])):
            for k in range(len(ptas[i][j])):
                print(i,j,k)
                print(ptas[i][j][k].params)
                #point_to_test = np.tile(np.array([0.0, 0.54, 1.0, -8.0, -13.39, 2.0, 0.5]),i+1)
                '''
    print('Number of pta params: ', len(pta.params))
    #print(pta.param_names)#summary())

    #setting up temperature ladder
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

    if T_dynamic:
        print("Dynamic temperature adjustment: ON")
    else:
        print("Dynamic temperature adjustment: OFF")

    #set up array to hold acceptance probabilities of last PT_hist_length PT swaps
    PT_hist = np.ones((n_chain-1,PT_hist_length))*np.nan #initiated with NaNs
    PT_hist_idx = np.array([0]) #index to keep track of which row to update in PT_hist

    #printitng out the prior used on GWB on/off
    if include_gwb:
        print("Prior on GWB on/off: {0}%".format(gwb_on_prior*100))

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
    if include_gwb:
        num_params += 1

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
        npzfile = np.load(resume_from)
        swap_record = list(npzfile['swap_record'])
        log_likelihood_resume = npzfile['log_likelihood']
        betas_resume = npzfile['betas']
        PT_acc_resume = npzfile['PT_acc']
        samples_resume = npzfile['samples']

        N_resume = samples_resume.shape[1]
        print("# of samples sucessfully read in: " + str(N_resume))

        samples = np.zeros((n_chain, N_resume+N, num_params))
        samples[:,:N_resume,:] = np.copy(samples_resume)

        log_likelihood = np.zeros((n_chain,N_resume+N))
        log_likelihood[:,:N_resume] = np.copy(log_likelihood_resume)
        betas = np.ones((n_chain,N_resume+N))
        betas[:,:N_resume] = np.copy(betas_resume)
        PT_acc = np.zeros((n_chain-1,N_resume+N))
        PT_acc[:,:N_resume] = np.copy(PT_acc_resume)

        QB_logl = []
        for j in range(n_chain):
            print(j)
            print(samples[j,0,:])
            n_wavelet = get_n_wavelet(samples, j, 0)
            n_glitch = get_n_glitch(samples, j, 0)
            if include_gwb:
                gwb_on = get_gwb_on(samples, j, 0, max_n_wavelet, max_n_glitch, num_noise_params)
            else:
                gwb_on = 0
            first_sample = strip_samples(samples, j, 0, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
            #print(dict(zip(ptas[n_wavelet][n_glitch][gwb_on].param_names, first_sample)))
            QB_logl.append(Quickburst.FastBurst(pta = pta, psrs = pulsars, params = dict(zip(pta.param_names, first_sample)), Npsr = len(pulsars), tref=53000*86400, Nglitch = n_glitch, Nwavelet = n_wavelet, rn_vary = vary_rn, wn_vary = vary_white_noise))
            #log_likelihood[j,0] = ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(first_sample)
    else:
        samples = np.zeros((n_chain, N, num_params))

        #set up log_likelihood array
        log_likelihood = np.zeros((n_chain,N))
        QB_logl = []

        #set up betas array with PT inverse temperatures
        betas = np.ones((n_chain,N))
        #set first row with initial betas
        betas[:,0] = 1/Ts
        print("Initial beta (1/T) ladder is:\n",betas[:,0])

        #set up array holding PT acceptance rate for each iteration
        PT_acc = np.zeros((n_chain-1,N))

        #filling first sample at all temperatures with last sample of previous run's zero temperature chain (thus it works if n_chain is different)
        if start_from is not None:
            samples_start = start_from
            for j in range(n_chain):
                samples[j,0,:] = np.copy(samples_start[:])
            '''
            npzfile = np.load(start_from)
            samples_start = npzfile['samples']
            for j in range(n_chain):
                samples[j,0,:] = np.copy(samples_start[0,-1,:])
                '''
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
                samples[j,0,2:] =  np.hstack(p.sample() for p in pta.params)

                print("Starting with n_wavelet=",n_wavelet)
                print("Starting with n_glitch=",n_glitch)
                print('samples init: ', samples)
                #set all wavelet gw sources to same sky location
                if n_wavelet!=0:
                    for windx in range(n_wavelet):
                        #4/13/23 -> wavelet_indx is float -> need to turn into int. ~ Jacob
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
            #print(j)
            #print(samples[j,0,:])
            n_wavelet = get_n_wavelet(samples, j, 0) #int(samples[j,0,0])
            n_glitch = get_n_glitch(samples, j, 0)
            if include_gwb:
                gwb_on = get_gwb_on(samples, j, 0, max_n_wavelet, max_n_glitch, num_noise_params)
            else:
                gwb_on = 0
            first_sample = strip_samples(samples, j, 0, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
            #print(dict(zip(ptas[n_wavelet][n_glitch][gwb_on].param_names, first_sample)))
            #print('params to start QB with: ',dict(zip(pta.param_names, first_sample)))
            QB_logl.append(Quickburst.FastBurst(pta = pta, psrs = pulsars, params = dict(zip(pta.param_names, first_sample)), Npsr = len(pulsars), tref=53000*86400, Nglitch = n_glitch, Nwavelet = n_wavelet, rn_vary = vary_per_psr_rn, wn_vary = vary_white_noise))
            #print(first_sample)
            log_likelihood[j,0] = QB_logl[j].get_lnlikelihood(first_sample)

            #Save first step for ensuring wavelet parameters are initially populated
            QB_logl[j].save_values(accept_new_step=True)
            #log_likelihood[j,0] = ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(first_sample)
            #print(log_likelihood[j,0])
            #print(pta.get_lnprior(first_sample))

    #setting up array for the fisher eigenvalues
    #one for wavelet parameters which we will keep updating
    eig = np.ones((n_chain, max_n_wavelet, 10, 10))*0.1

    #also one for the glitch parameters
    eig_glitch = np.ones((n_chain, max_n_glitch, 6, 6))*0.03

    #one for GWB and common rn parameters, which we will keep updating
    if include_gwb:
        eig_gwb_rn = np.broadcast_to( np.array([[1.0,0,0], [0,0.3,0], [0,0,0.3]]), (n_chain, 3, 3)).copy()
    else:
        eig_gwb_rn = np.broadcast_to( np.array([[1.0,0], [0,0.3]]), (n_chain, 2, 2)).copy()

    #and one for white noise parameters, which we will also keep updating
    eig_per_psr = np.broadcast_to(np.eye(num_per_psr_params)*0.1, (n_chain, num_per_psr_params, num_per_psr_params) ).copy()
    #calculate wn eigenvectors

    for j in range(n_chain):
        n_wavelet = get_n_wavelet(samples, j, 0)
        n_glitch = get_n_glitch(samples, j, 0)
        per_psr_eigvec = get_fisher_eigenvectors(strip_samples(samples, j, 0, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), pta, QB_logl=QB_logl[j], T_chain=1/betas[j,0], n_sources=len(pulsars), dim=len(per_puls_indx[1]), array_index=per_puls_indx, vary_white_noise = vary_white_noise, vary_red_noise = vary_per_psr_rn)
        eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]

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
        log_f0_max = float(pta.params[3]._typename.split('=')[2][:-1])
        log_f0_min = float(pta.params[3]._typename.split('=')[1].split(',')[0])
        t0_max = float(pta.params[8]._typename.split('=')[2][:-1])
        t0_min = float(pta.params[8]._typename.split('=')[1].split(',')[0])
        tau_max = float(pta.params[9]._typename.split('=')[2][:-1])
        tau_min = float(pta.params[9]._typename.split('=')[1].split(',')[0])

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
        log_f0_max = float(pta.params[3]._typename.split('=')[2][:-1])
        log_f0_min = float(pta.params[3]._typename.split('=')[1].split(',')[0])
        t0_max = float(pta.params[8]._typename.split('=')[2][:-1])
        t0_min = float(pta.params[8]._typename.split('=')[1].split(',')[0])
        tau_max = float(pta.params[9]._typename.split('=')[2][:-1])
        tau_min = float(pta.params[9]._typename.split('=')[1].split(',')[0])

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
            print(i)
            glitch_tau_scan = glitch_tau_scan_data['tau_scan'+str(i)]
            print(len(glitch_tau_scan))

            norm = 0.0
            for idx, TTT in enumerate(glitch_tau_scan):
                for kk in range(TTT.shape[0]):
                    for ll in range(TTT.shape[1]):
                        df = np.log10(F0_list[idx][kk+1]/F0_list[idx][kk])
                        dt = (T0_list[idx][ll+1]-T0_list[idx][ll])/3600/24/365.25
                        dtau = (TAU_list[idx+1]-TAU_list[idx])
                        norm += TTT[kk,ll]*df*dt*dtau
            glitch_tau_scan_data['norm'+str(i)] = norm #TODO: Implement some check to make sure this is normalized over the same range as the prior range used in the MCMC
            print(norm)

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
            print(tau_scan_limit)
            glitch_tau_scan_data['psr_idx_proposal'][i] = tau_scan_limit

        glitch_tau_scan_data['psr_idx_proposal'] = glitch_tau_scan_data['psr_idx_proposal']/np.sum(glitch_tau_scan_data['psr_idx_proposal'])
        print('-'*20)
        print("Glitch psr index proposal:")
        print(glitch_tau_scan_data['psr_idx_proposal'])
        print(np.sum(glitch_tau_scan_data['psr_idx_proposal']))
        print('-'*20)

    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros((8, n_chain)) #columns: chain number; rows: proposal type (glitch_RJ, glitch_tauscan, wavelet_RJ, wavelet_tauscan, gwb_RJ, PT, fisher, noise_jump)
    a_no=np.zeros((8, n_chain))
    acc_fraction = a_yes/(a_no+a_yes)
    if resume_from is None:
        swap_record = []
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

    if resume_from is None:
        start_iter = 0
        stop_iter = N
    else:
        start_iter = N_resume-1 #-1 because if only 1 sample is read in that's the same as having a different starting point and start_iter should still be 0
        stop_iter = N_resume-1+N

    for i in range(int(start_iter), int(stop_iter-1)): #-1 because ith step here produces (i+1)th sample based on ith sample
        ########################################################
        #
        #logging PT acceptance fraction
        #
        ########################################################
        #logging cumulative acc fraction
        #acc_fraction = a_yes/(a_no+a_yes)
        #PT_acc[:,i] = np.copy(acc_fraction[5,:])

        #logging mean acc probability over last PT_hist_length swaps
        PT_acc[:,i] = np.nanmean(PT_hist, axis=1) #nanmean so early on when we still have nans we only use the actual data
        ########################################################
        #
        #update temperature ladder
        #
        ########################################################
        if i>0:
            if T_dynamic and PT_hist_idx>0: #based on arXiv:1501.05823 and https://github.com/willvousden/ptemcee
                kappa = 1.0/T_dynamic_nu * T_dynamic_t0/(PT_hist_idx+T_dynamic_t0)
                #dSs = kappa * (acc_fraction[5,:-2] - acc_fraction[5,1:-1])
                dSs = kappa * (PT_acc[:-1,i] - PT_acc[1:,i])
                #print(dSs)
                deltaTs = np.diff(1 / betas[:-1,i-1])
                #print(deltaTs)
                deltaTs *= np.exp(dSs)
                #print(deltaTs)

                new_betas = 1 / (np.cumsum(deltaTs) + 1 / betas[0,i-1])
                #print(new_betas)

                #set new betas
                betas[-1,i] = 0.0
                betas[1:-1,i] = np.copy(new_betas)
            else:
                #copy betas from previous iteration
                betas[:,i] = betas[:,i-1]
        ########################################################
        #
        #write results to file every save_every_n iterations
        #
        ########################################################
        if savefile is not None and i%save_every_n==0 and i!=start_iter:
            if not safe_save:
                np.savez(savefile, samples=samples[:,:i,:], acc_fraction=acc_fraction, swap_record=swap_record, log_likelihood=log_likelihood[:,:i],
                         betas=betas[:,:i], PT_acc=PT_acc[:,:i])
            else:
                #save to temporary file, copy it into permananet file and remove the temp file
                #prevents the loss of all data if the run is interupted when np.savez is running
                np.savez(savefile + ".temp.npz", samples=samples[:,:i,:], acc_fraction=acc_fraction, swap_record=swap_record, log_likelihood=log_likelihood[:,:i],
                         betas=betas[:,:i], PT_acc=PT_acc[:,:i])
                shutil.copy(savefile + ".temp.npz", savefile)
                os.remove(savefile + ".temp.npz")
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
                        'Acceptance fraction #columns: chain number; rows: proposal type (glitch_RJ, glitch_tauscan, wavelet_RJ, wavelet_tauscan, GWB_RJ, PT, fisher, noise_jump):')
                print(acc_fraction)
                #print(PT_hist)
                print(PT_acc[:,i])
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
                n_wavelet = get_n_wavelet(samples, j, i)
                n_glitch = get_n_glitch(samples, j, i)
                if include_gwb:
                    gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
                else:
                    gwb_on = 0



                # Fisher Information Matrix: Calculates the covariances for each parameter associated with
                # maximum likelihood estimates. This is used to inform jump proposals in parameter space
                # for various kinds of jumps by pulling out eigenvectors from Fisher Matrix for particular
                # parameters that are being updated.

                #wavelet eigenvectors
                if n_wavelet!=0:
                    eigenvectors = get_fisher_eigenvectors(strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), pta, QB_logl=QB_logl[j], T_chain=1/betas[j,i], n_sources=n_wavelet, array_index=wavelet_indx, flag = True)
                    if np.all(eigenvectors):
                        eig[j,:n_wavelet,:,:] = eigenvectors

                #glitch eigenvectors
                if n_glitch!=0:
                    eigen_glitch = get_fisher_eigenvectors(strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), pta, QB_logl=QB_logl[j], T_chain=1/betas[j,i], n_sources=n_glitch, dim=6, array_index=glitch_indx, flag = True)
                    if np.all(eigen_glitch):
                        eig_glitch[j,:n_glitch,:,:] = eigen_glitch

                #RN+G+WB eigenvectors

                # if include_gwb:
                #     eigvec_rn = get_fisher_eigenvectors(strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), pta, QB_logl=QB_logl[j], T_chain=1/betas[j,i], n_sources=1, dim=3, array_index=np.array((1,gwb_indx)))
                # else:
                #     eigvec_rn = get_fisher_eigenvectors(strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), pta, QB_logl=QB_logl[j], T_chain=1/betas[j,i], n_sources=1, dim=2, array_index=np.array((1,rn_indx)))
                # if np.all(eigvec_rn):
                #     eig_gwb_rn[j,:,:] = eigvec_rn[0,:,:]

                #per PSR eigenvectors
                if vary_rn or vary_white_noise:
                    per_psr_eigvec = get_fisher_eigenvectors(strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch), pta, QB_logl=QB_logl[j], T_chain=1/betas[j,0], n_sources=len(pulsars), dim=len(per_puls_indx[1]), array_index=per_puls_indx, vary_white_noise = vary_white_noise, vary_red_noise = vary_per_psr_rn)
                #if np.all(per_psr_eigvec):
                    eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]

        ###########################################################
        #
        #Do the actual MCMC step
        #
        ###########################################################
        if i%n_fast_to_slow==0:
            #draw a random number to decide which jump to do
            jump_decide = np.random.uniform()
            #print("-"*50)
            #print(samples[0,i,:])
            #PT swap move
            if jump_decide<swap_probability:
                do_pt_swap(n_chain, max_n_wavelet, max_n_glitch, pta, samples, i, betas, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_noise_params, log_likelihood, PT_hist, PT_hist_idx)

            elif jump_decide<swap_probability + noise_jump_probability:
                noise_jump(n_chain, max_n_wavelet, max_n_glitch, pta, samples, i, betas, a_yes, a_no, eig_per_psr, per_puls_indx, include_gwb, num_noise_params, vary_white_noise, vary_per_psr_rn, log_likelihood, QB_logl)
            #do regular jump
            else:
                regular_jump(n_chain, max_n_wavelet, max_n_glitch, pta, samples, i, betas, a_yes, a_no, eig, eig_glitch, eig_gwb_rn, include_gwb, num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, log_likelihood, QB_logl)
            #print(samples[0,i+1,:])
            #print("-"*50)
        else:
            #For fast jumps, can't have wavelet_indx[i, 3, 8, 9] or glitch_indx[i, 0, 3, 4, 5] Otherwise M and N gets recalculated
            fast_jump(n_chain, max_n_wavelet, max_n_glitch, pta, samples, i, betas, a_yes, a_no, eig, eig_glitch, eig_gwb_rn, include_gwb, num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, log_likelihood, QB_logl)
            #fast step goes here

    acc_fraction = a_yes/(a_no+a_yes)
    return samples, acc_fraction, swap_record, rj_record, pta, log_likelihood, betas, PT_acc


################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN CW, GWB AND RN PARAMETERS)
#
################################################################################
def regular_jump(n_chain, max_n_wavelet, max_n_glitch, pta, samples, i, betas, a_yes, a_no, eig, eig_glitch, eig_gwb_rn, include_gwb, num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, log_likelihood, QB_logl):
    #print("Regular_jump")
    for j in range(n_chain):
        n_wavelet = get_n_wavelet(samples, j, i)
        n_glitch = get_n_glitch(samples, j, i)
        #if j==0:
        #    print(n_wavelet)
        #    print(n_glitch)


        samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

        #decide if moving in wavelet parameters, glitch parameters, or GWB/RN parameters
        #case #1: we can vary any of them
        if n_wavelet!=0 and n_glitch!=0 and (include_gwb or vary_rn):
            vary_decide = np.random.uniform()
            if vary_decide <= 1.0/3.0:
                what_to_vary = 'WAVE'
            elif vary_decide <= 2.0/3.0:
                what_to_vary = 'GLITCH'
            else:
                what_to_vary = 'GWB'
        #case #2: whe can vary two of them
        elif n_glitch!=0 and (include_gwb or vary_rn):
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'GLITCH'
            else:
                what_to_vary = 'GWB'
        elif n_wavelet!=0 and (include_gwb or vary_rn):
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'WAVE'
            else:
                what_to_vary = 'GWB'
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
        elif gwb_on==1 or vary_rn:
            what_to_vary = 'GWB'
        #case #4: nothing to vary
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[5,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #print("Nothing to vary!")
            continue

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
            #print('cw')
            #print(jump)
        elif what_to_vary == 'GLITCH':
            glitch_select = np.random.randint(n_glitch)
            jump_select = np.random.randint(6)
            jump_1glitch = eig_glitch[j,glitch_select,jump_select,:]
            jump = np.zeros(samples_current.size)
            #change intrinsic (and extrinsic) parameters of selected wavelet

            #This is meant to replace all values for a specific glitch,
            #which means we need to index 1 beyond the end so it will see all values
            jump[glitch_indx[glitch_select,0]:glitch_indx[glitch_select,5]+1] = jump_1glitch
        elif what_to_vary == 'GWB':
            if include_gwb:
                jump_select = np.random.randint(3)
            else:
                jump_select = np.random.randint(2)
            jump_gwb = eig_gwb_rn[j,jump_select,:]
            if gwb_on==0 and include_gwb:
                jump_gwb[-1] = 0
            #this is a stupid mask for GWB params
            if include_gwb:
                jump = np.array([jump_gwb[int(i-n_wavelet*10-n_glitch*6-num_per_psr_params)] if i>=n_wavelet*10+n_glitch*6+num_per_psr_params and i<n_wavelet*10+n_glitch*6+num_noise_params+1 else 0.0 for i in range(samples_current.size)])
            else:
                jump = np.array([jump_gwb[int(i-n_wavelet*10-n_glitch*6-num_per_psr_params)] if i>=n_wavelet*10+n_glitch*6+num_per_psr_params and i<n_wavelet*10+n_glitch*6+num_noise_params else 0.0 for i in range(samples_current.size)])
            #if j==0: print('gwb+rn')
            #if j==0: print(i)
            #if j==0: print(jump)

        new_point = samples_current + jump*np.random.normal()#only sd of 1 for all parameter jumps
        #print('jump ', jump)
        #if j==0:
        #    print(samples_current)
        #    print(new_point)

        #check if we are inside prior before calling likelihood, otherwise it throws an error
        new_log_prior = pta.get_lnprior(new_point)
        if new_log_prior==-np.inf: #check if prior is -inf - reject step if it is
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            continue

        #print(j)
        #print(n_wavelet, n_glitch, gwb_on)
        #print(ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(new_point))
        #print(ptas[n_wavelet][n_glitch][gwb_on].params)
        #print(new_point)
        #print(new_point-samples_current)
        #print('new point: ', new_point)
        log_L = QB_logl[j].get_lnlikelihood(new_point)
        #log_L = ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(new_point)
        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += new_log_prior
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -pta.get_lnprior(samples_current)

        acc_ratio = np.exp(log_acc_ratio)
        #if j==0: print(acc_ratio)
        if np.random.uniform()<=acc_ratio:
            #if j==0: print("ohh jeez")
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            #print(samples[j,i+1,:])
            #samples[j,i+1,2:2+n_wavelet*10] = new_point[:n_wavelet*10]
            #samples[j,i+1,2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[n_wavelet*10:n_wavelet*10+n_glitch*6]
            #samples[j,i+1,2+max_n_wavelet*10+max_n_glitch*6:] = new_point[n_wavelet*10+n_glitch*6:]
            a_yes[6,j]+=1
            log_likelihood[j,i+1] = log_L
            #print("accept step")
            QB_logl[j].save_values(accept_new_step=True)
        else:
            samples[j,i+1,:] = samples[j,i,:]
            #print(samples[j,i+1,:])
            a_no[6,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #print("rejectS step")
            QB_logl[j].save_values(accept_new_step=False)


################################################################################
#
#Fast MCMC JUMP ROUTINE (jumping in projection PARAMETERS)
#
################################################################################
def fast_jump(n_chain, max_n_wavelet, max_n_glitch, pta, samples, i, betas, a_yes, a_no, eig, eig_glitch, eig_gwb_rn, include_gwb, num_noise_params, num_per_psr_params, vary_rn, wavelet_indx, glitch_indx, log_likelihood, QB_logl):
    #print("fast_jump")
    for j in range(n_chain):
        n_wavelet = get_n_wavelet(samples, j, i)
        n_glitch = get_n_glitch(samples, j, i)
        #if j==0:
        #    print(n_wavelet)
        #    print(n_glitch)

        samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

        #decide if moving in wavelet parameters, glitch parameters, or GWB/RN parameters
        #case #1: we can vary any of them
        if n_wavelet!=0 and n_glitch!=0:
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
        #case #4: nothing to vary
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[5,j]+=1
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
            #print('cw')
            #print(jump)
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
        #print('jump ', jump)
        #if j==0:
        #    print(samples_current)
        #    print(new_point)

        #check if we are inside prior before calling likelihood, otherwise it throws an error
        new_log_prior = pta.get_lnprior(new_point)
        if new_log_prior==-np.inf: #check if prior is -inf - reject step if it is
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            continue

        #print(j)
        #print(n_wavelet, n_glitch, gwb_on)
        #print(ptas[n_wavelet][n_glitch][gwb_on].get_lnprior(new_point))
        #print(ptas[n_wavelet][n_glitch][gwb_on].params)
        #print(new_point)
        #print(new_point-samples_current)
        #print('new point: ', new_point)
        log_L = QB_logl[j].get_lnlikelihood(new_point)
        #log_L = ptas[n_wavelet][n_glitch][gwb_on].get_lnlikelihood(new_point)
        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += new_log_prior
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -pta.get_lnprior(samples_current)

        acc_ratio = np.exp(log_acc_ratio)
        #if j==0: print(acc_ratio)
        if np.random.uniform()<=acc_ratio:
            #if j==0: print("ohh jeez")
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            #print(samples[j,i+1,:])
            #samples[j,i+1,2:2+n_wavelet*10] = new_point[:n_wavelet*10]
            #samples[j,i+1,2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[n_wavelet*10:n_wavelet*10+n_glitch*6]
            #samples[j,i+1,2+max_n_wavelet*10+max_n_glitch*6:] = new_point[n_wavelet*10+n_glitch*6:]
            a_yes[6,j]+=1
            log_likelihood[j,i+1] = log_L
            #print("accept step")
            QB_logl[j].save_values(accept_new_step=True)
        else:
            samples[j,i+1,:] = samples[j,i,:]
            #print(samples[j,i+1,:])
            a_no[6,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #print("rejectS step")
            QB_logl[j].save_values(accept_new_step=False)


################################################################################
#
#REVERSIBLE-JUMP (RJ, aka TRANS-DIMENSIONAL) MOVE -- adding or removing a wavelet
#
################################################################################
#@profile
def do_rj_move_wavelet(n_chain, max_n_wavelet, min_n_wavelet, max_n_glitch, n_wavelet_prior, pta, samples, i, betas, a_yes, a_no, rj_record, vary_white_noise, include_gwb, num_noise_params, tau_scan_data, log_likelihood, wavelet_indx):
    #print("RJ")
    '''
    tau_scan = tau_scan_data['tau_scan']

    tau_scan_limit = 0
    for TS in tau_scan:
        TS_max = np.max(TS)
        if TS_max>tau_scan_limit:
            tau_scan_limit = TS_max

    TAU_list = list(tau_scan_data['tau_edges'])
    F0_list = tau_scan_data['f0_edges']
    T0_list = tau_scan_data['t0_edges']
    '''
    #print(samples[0,i,:])
    #print(i)

    for j in range(n_chain):
        n_wavelet = get_n_wavelet(samples, j, i)
        n_glitch = get_n_glitch(samples, j, i)

        if include_gwb:
            gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
        else:
            gwb_on = 0

        add_prob = 0.5 #same propability of addind and removing
        #decide if we add or remove a signal
        direction_decide = np.random.uniform()
        if n_wavelet==min_n_wavelet or (direction_decide<add_prob and n_wavelet!=max_n_wavelet): #adding a wavelet------------------------------------------------------
            if j==0: rj_record.append(1)
            #if j==0: print("Propose to add a wavelet")
            '''
            log_f0_max = float(ptas[-1][0][gwb_on].params[3]._typename.split('=')[2][:-1])
            log_f0_min = float(ptas[-1][0][gwb_on].params[3]._typename.split('=')[1].split(',')[0])
            t0_max = float(ptas[-1][0][gwb_on].params[8]._typename.split('=')[2][:-1])
            t0_min = float(ptas[-1][0][gwb_on].params[8]._typename.split('=')[1].split(',')[0])
            tau_max = float(ptas[-1][0][gwb_on].params[9]._typename.split('=')[2][:-1])
            tau_min = float(ptas[-1][0][gwb_on].params[9]._typename.split('=')[1].split(',')[0])
            '''
            '''
            accepted = False
            while accepted==False:
                log_f0_new = np.random.uniform(low=log_f0_min, high=log_f0_max)
                t0_new = np.random.uniform(low=t0_min, high=t0_max)
                tau_new = np.random.uniform(low=tau_min, high=tau_max)

                tau_idx = np.digitize(tau_new, np.array(TAU_list)) - 1
                f0_idx = np.digitize(10**log_f0_new, np.array(F0_list[tau_idx])) - 1
                t0_idx = np.digitize(t0_new, np.array(T0_list[tau_idx])/(365.25*24*3600)) - 1

                #print(tau_new, t0_new, 10**log_f0_new)
                #print(tau_idx, t0_idx, f0_idx)

                tau_scan_new_point = tau_scan[tau_idx][f0_idx, t0_idx]
                #print(tau_scan_new_point/tau_scan_limit)
                if np.random.uniform()<(tau_scan_new_point/tau_scan_limit):
                    accepted = True
                    #print("Yeeeh!")
            '''
            #randomly select other parameters
            log_f0_new = pta.params[wavelet_indx[0,3]].sample()
            t0_new = pta.params[wavelet_indx[0,8]].sample()
            tau_new = pta.params[wavelet_indx[0,9]].sample()
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

            samples_current = samples[j,i,2:]#strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)

            new_point = samples[j,i,2:]#strip_samples(samples, j, i, n_wavelet+1, max_n_wavelet, n_glitch, max_n_glitch)
            # new_wavelet = np.array([cos_gwtheta_new, psi_new, gwphi_new, log_f0_new, log10_h_new, log10_h_cross_new,
            #                         phase0_new, phase0_cross_new, t0_new, tau_new])
            new_point[wavelet_indx[n_wavelet+1,0]] = cos_gwtheta_new
            new_point[wavelet_indx[n_wavelet+1,1]] = psi_new
            new_point[wavelet_indx[n_wavelet+1,2]] = gwphi_new
            new_point[wavelet_indx[n_wavelet+1,3]] = log_f0_new
            new_point[wavelet_indx[n_wavelet+1,4]] = log10_h_new
            new_point[wavelet_indx[n_wavelet+1,5]] = log10_h_cross_new
            new_point[wavelet_indx[n_wavelet+1,6]] = phase0_new
            new_point[wavelet_indx[n_wavelet+1,7]] = phase0_cross_new
            new_point[wavelet_indx[n_wavelet+1,8]] = t0_new
            new_point[wavelet_indx[n_wavelet+1,9]] = tau_new

            log_L = QB_logl[j].get_lnlikelihood(new_point)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += pta.get_lnprior(new_point) #this will need more attention
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -pta.get_lnprior(samples_current)


            #apply normalization
            #tau_scan_new_point_normalized = tau_scan_new_point/tau_scan_data['norm']

            acc_ratio = np.exp(log_acc_ratio)/prior_ext#/tau_scan_new_point_normalized
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_wavelet==min_n_wavelet:
                acc_ratio *= 0.5
            if n_wavelet==max_n_wavelet-1:
                acc_ratio *= 2.0
            #accounting for n_wavelet prior
            acc_ratio *= n_wavelet_prior[int(n_wavelet)+1]/n_wavelet_prior[int(n_wavelet)]
            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Yeeeh")
                samples[j,i+1,0] = n_wavelet+1
                samples[j,i+1,1] = n_glitch
                samples[j,i+1,2:] = new_point[:]
                #samples[j,i+1,2:2+(n_wavelet+1)*10] = new_point[:(n_wavelet+1)*10]
                #samples[j,i+1,2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[(n_wavelet+1)*10:(n_wavelet+1)*10+n_glitch*6]
                #samples[j,i+1,2+max_n_wavelet*10+max_n_glitch*6:] = new_point[(n_wavelet+1)*10+n_glitch*6:]
                a_yes[2,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[2,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
        '''
        We need to decide how to move around terms when removing a wavelet
        '''
        '''
        elif n_wavelet==max_n_wavelet or (direction_decide>add_prob and n_wavelet!=min_n_wavelet):   #removing a wavelet----------------------------------------------------------
            if j==0: rj_record.append(-1)
            #if j==0: print("Propose to remove a wavelet")
            #choose which wavelet to remove
            remove_index = np.random.randint(n_wavelet)

            #Call M_N helper function to change M and N matrices to account for changing wavelet
            wavelet_to_remove = wavelet_indx[remove_index, :]

            QB_logl.M_N_helper(remove_index, wavelet_change = True)

            samples_current = samples[j,i,2:]#strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
            #new_point = np.delete(samples_current, range(remove_index*10,(remove_index+1)*10))

            #Removing proper wavelet from samples_current. Starts from first parameter, go to last parameter for that wavelet.
            #new_point = np.delete(samples_current, range(wavelet_indx[remove_index,0]*10, wavelet_indx[remove_indx, -1]+1))
            #if j==0:
            #    print(samples_current)
            #    print(new_point)

            log_L = QB_logl[j].get_lnlikelihood(new_point)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += pta.get_lnprior(new_point)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -pta.get_lnprior(samples_current)

            #getting tau_scan at old point
            tau_old = samples[j,i,2+9+remove_index*10]
            f0_old = 10**samples[j,i,2+3+remove_index*10]
            t0_old = samples[j,i,2+8+remove_index*10]

            tau_idx_old = np.digitize(tau_old, np.array(TAU_list)) - 1
            f0_idx_old = np.digitize(f0_old, np.array(F0_list[tau_idx_old])) - 1
            t0_idx_old = np.digitize(t0_old, np.array(T0_list[tau_idx_old])/(365.25*24*3600)) - 1

            tau_scan_old_point = tau_scan[tau_idx_old][f0_idx_old, t0_idx_old]

            #apply normalization
            tau_scan_old_point_normalized = tau_scan_old_point/tau_scan_data['norm']

            #getting external parameter priors
            log10_h_old = np.copy(samples[j,i,2+4+remove_index*10])
            log10_h_cross_old = np.copy(samples[j,i,2+5+remove_index*10])
            phase0_old = np.copy(samples[j,i,2+6+remove_index*10])
            phase0_cross_old = np.copy(samples[j,i,2+7+remove_index*10])
            cos_gwtheta_old = np.copy(samples[j,i,2+0+remove_index*10])
            gwphi_old = np.copy(samples[j,i,2+2+remove_index*10])
            psi_old = np.copy(samples[j,i,2+1+remove_index*10])

            prior_ext = (ptas[-1][0][gwb_on].params[0].get_pdf(cos_gwtheta_old) * ptas[-1][0][gwb_on].params[1].get_pdf(psi_old) *
                         ptas[-1][0][gwb_on].params[2].get_pdf(gwphi_old) *
                         ptas[-1][0][gwb_on].params[4].get_pdf(log10_h_old) * ptas[-1][0][gwb_on].params[5].get_pdf(log10_h_cross_old) *
                         ptas[-1][0][gwb_on].params[6].get_pdf(phase0_old) * ptas[-1][0][gwb_on].params[7].get_pdf(phase0_cross_old))

            acc_ratio = np.exp(log_acc_ratio)*prior_ext*tau_scan_old_point_normalized
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_wavelet==min_n_wavelet+1:
                acc_ratio *= 2.0
            if n_wavelet==max_n_wavelet:
                acc_ratio *= 0.5
            #accounting for n_wavelet prior
            acc_ratio *= n_wavelet_prior[int(n_wavelet)-1]/n_wavelet_prior[int(n_wavelet)]
            if np.random.uniform()<=acc_ratio:
                #if j==0: print("Ohhhhhhhhhhhhh")
                samples[j,i+1,0] = n_wavelet-1
                samples[j,i+1,1] = n_glitch
                samples[j,i+1,2:2+(n_wavelet-1)*10] = new_point[:(n_wavelet-1)*10]
                samples[j,i+1,2+max_n_wavelet*10:2+max_n_wavelet*10+n_glitch*6] = new_point[(n_wavelet-1)*10:(n_wavelet-1)*10+n_glitch*6]
                samples[j,i+1,2+max_n_wavelet*10+max_n_glitch*6:] = new_point[(n_wavelet-1)*10+n_glitch*6:]
                a_yes[2,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[2,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
        '''
################################################################################
#
#NOISE MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN WHITE NOISE PARAMETERS)
#
################################################################################
def noise_jump(n_chain, max_n_wavelet, max_n_glitch, pta, samples, i, betas, a_yes, a_no, eig_per_psr, per_puls_indx, include_gwb, num_noise_params, vary_white_noise, vary_rn, log_likelihood, QB_logl):
    #print("NOISE")
    for j in range(n_chain):
        n_wavelet = get_n_wavelet(samples, j, i) # samples[j, i]
        n_glitch = get_n_glitch(samples, j, i)
        #if j==0:
        #print(n_wavelet)
        #print(n_glitch)

        if include_gwb:
            gwb_on = get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params)
        else:
            gwb_on = 0

        #samples_current = strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch)
        samples_current = np.copy(samples[j,i,2:])
        #do the wn jump
        jump_select = np.random.randint(eig_per_psr.shape[1])
        #print(jump_select)
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

        #jump = np.array([jump_wn[int(i-n_wavelet*10-n_glitch*6)] if i>=n_wavelet*10+n_glitch*6 and i<n_wavelet*10+n_glitch*6+eig_per_psr.shape[1] else 0.0 for i in range(samples_current.size)])
        #if j==0: print('noise')
        #if j==0: print(jump)

        new_point = samples_current + jump*np.random.normal()

        log_L = QB_logl[j].get_lnlikelihood(new_point, vary_white_noise = vary_white_noise, vary_red_noise = vary_rn)
        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += pta.get_lnprior(new_point)
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -pta.get_lnprior(samples_current)

        acc_ratio = np.exp(log_acc_ratio)
        #if j==0: print(acc_ratio)

        if np.random.uniform()<=acc_ratio:
            #if j==0: print("Ohhhh")
            samples[j,i+1,0] = n_wavelet
            samples[j,i+1,1] = n_glitch
            samples[j,i+1,2:] = new_point[:]
            a_yes[7,j]+=1
            log_likelihood[j,i+1] = log_L
            #print("accept step")
            QB_logl[j].save_values(accept_new_step=True)
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[7,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #print("rejectS step")
            QB_logl[j].save_values(accept_new_step=False)

################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, max_n_wavelet, max_n_glitch, pta, samples, i, betas, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_noise_params, log_likelihood, PT_hist, PT_hist_idx):
    #print("SWAP")

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
            a_yes[5,swap_chain]+=1
            #PT_hist[swap_chain,PT_hist_idx[0]%PT_hist.shape[1]] = 1.0
            swap_record.append(swap_chain)
        else:
            a_no[5,swap_chain]+=1
            #PT_hist[swap_chain,PT_hist_idx[0]%PT_hist.shape[1]] = 0.0

    #print(PT_hist_idx[0])
    PT_hist_idx += 1
    #print(swap_map)

    #loop through the chains and record the new samples and log_Ls
    for j in range(n_chain):
        samples[j,i+1,:] = samples[swap_map[j],i,:]
        log_likelihood[j,i+1] = log_likelihood[swap_map[j],i]

################################################################################
#
#FISHER EIGENVALUE CALCULATION
#
################################################################################
def get_fisher_eigenvectors(params, pta, QB_logl, T_chain=1, epsilon=1e-4, n_sources=1, dim=10, array_index=None, use_prior=False, flag = False, vary_white_noise = False, vary_red_noise = False):#offset=0, use_prior=False):
    n_source=n_sources # this needs to not be used for the non-wavelet/glitch indexing (set to 1)
    eig = []

    #print(params)
    index_rows = len(array_index)
    print('index_rows: ', index_rows)
    if flag:
        fisher = np.zeros((n_source,dim,dim))
    else:
        fisher = np.zeros((n_source,dim*index_rows,dim*index_rows))
    #lnlikelihood at specified point
    if use_prior:
        nn = QB_logl.get_lnlikelihood(params) + pta.get_lnprior(params)
    else:
        nn = QB_logl.get_lnlikelihood(params)

    print('fish n_source {0}: dim {1}: params len {2}: array_index {3}'.format(n_source, dim, len(params), array_index))
    if flag == True:
        for k in range(n_source):
            #print(k)
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
                    pp = QB_logl.get_lnlikelihood(paramsPP) + pta.get_lnprior(paramsPP)
                    mm = QB_logl.get_lnlikelihood(paramsMM) + pta.get_lnprior(paramsMM)
                else:
                    pp = QB_logl.get_lnlikelihood(paramsPP)
                    mm = QB_logl.get_lnlikelihood(paramsMM)

                #print(pp, nn, mm)

                #calculate diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                #print('diagonal')
                #print(pp,nn,mm)
                #print(-(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon))
                fisher[k,i,i] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

            # #calculate off-diagonal elements
            # for i in range(dim):
            #     for j in range(i+1,dim):
            #         #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
            #         paramsPP = np.copy(params)
            #         paramsMM = np.copy(params)
            #         paramsPM = np.copy(params)
            #         paramsMP = np.copy(params)
            #         #looping through wavelets or glitch terms only and perturbing by small amount
            #         paramsPP[array_index[k,i]] += epsilon
            #         paramsPP[array_index[k,j]] += epsilon
            #         paramsMM[array_index[k,i]] -= epsilon
            #         paramsMM[array_index[k,j]] -= epsilon
            #         paramsPM[array_index[k,i]] += epsilon
            #         paramsPM[array_index[k,j]] -= epsilon
            #         paramsMP[array_index[k,i]] -= epsilon
            #         paramsMP[array_index[k,j]] += epsilon
            #
            #         #lnlikelihood at those positions
            #         if use_prior:
            #             pp = QB_logl.get_lnlikelihood(paramsPP) + pta.get_lnprior(paramsPP)
            #             mm = QB_logl.get_lnlikelihood(paramsMM) + pta.get_lnprior(paramsMM)
            #             pm = QB_logl.get_lnlikelihood(paramsPM) + pta.get_lnprior(paramsPM)
            #             mp = QB_logl.get_lnlikelihood(paramsMP) + pta.get_lnprior(paramsMP)
            #         else:
            #             pp = QB_logl.get_lnlikelihood(paramsPP)
            #             mm = QB_logl.get_lnlikelihood(paramsMM)
            #             pm = QB_logl.get_lnlikelihood(paramsPM)
            #             mp = QB_logl.get_lnlikelihood(paramsMP)
            #
            #         #calculate off-diagonal elements of the Hessian from a central finite element scheme
            #         #note the minus sign compared to the regular Hessian
            #         #print('off-diagonal')
            #         #print(pp,mp,pm,mm)
            #         #print(-(pp - mp - pm + mm)/(4.0*epsilon*epsilon))
            #         fisher[k,i,j] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
            #         fisher[k,j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)

            #print(fisher)
            #correct for the given temperature of the chain
            fisher = fisher/T_chain
            print('fisher matrix: ', fisher)
            #print("---")
            #print(fisher)

            try:
                #Filter nans and infs and replace them with 1s
                #this will imply that we will set the eigenvalue to 100 a few lines below
                #UPDATED so that 0s are also replaced with 1.0
                FISHER = np.where(np.isfinite(fisher[k,:,:]) * (fisher[k,:,:]!=0.0), fisher[k,:,:], 1.0)
                if not np.array_equal(FISHER, fisher[k,:,:]):
                    print("Changed some nan elements in the Fisher matrix to 1.0")
                    #print(fisher[k,:,:])
                    #print(FISHER)

                #Find eigenvalues and eigenvectors of the Fisher matrix
                w, v = np.linalg.eig(FISHER)

                #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
                eig_limit = 1.0

                W = np.where(np.abs(w)>eig_limit, w, eig_limit)
                #print(W)
                #print(np.sum(v**2, axis=0))
                #if T_chain==1.0: print(W)
                #if T_chain==1.0: print(v)

                eig.append( (np.sqrt(1.0/np.abs(W))*v).T )
                #print(np.sum(eig**2, axis=1))
                #if T_chain==1.0: print(eig)

            except:
                print("An Error occured in the eigenvalue calculation")
                eig.append( np.array(False) )

            #import matplotlib.pyplot as plt
            #plt.figure()
            #plt.imshow(np.log10(np.abs(np.real(np.array(FISHER)))))
            #plt.imshow(np.real(np.array(FISHER)))
            #plt.colorbar()

            #plt.figure()
            #plt.imshow(np.log10(np.abs(np.real(np.array(eig)[0,:,:]))))
            #plt.imshow(np.real(np.array(eig)[0,:,:]))
            #plt.colorbar()

    #Run this if not doing wavelet/glitch stuff
    else:
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
                    pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise, vary_red_noise) + pta.get_lnprior(paramsPP)
                    mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise, vary_red_noise) + pta.get_lnprior(paramsMM)
                else:
                    pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise, vary_red_noise)
                    mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise, vary_red_noise)

                #print(pp, nn, mm)

                #calculate diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                #print('diagonal')
                #print(pp,nn,mm)
                #print(-(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon))
                fisher[1,i+n*dim,i+n*dim] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)
        # for n in range(index_rows):
        #     dim = len(array_index[n])
        #     #print('Dimensions!!!!!: ', dim)
        #     #calculate off-diagonal elements
        #     for i in range(dim):
        #         for j in range(i+1,dim):
        #             #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
        #             paramsPP = np.copy(params)
        #             paramsMM = np.copy(params)
        #             paramsPM = np.copy(params)
        #             paramsMP = np.copy(params)
        #
        #             paramsPP[array_index[n][i]] += epsilon
        #             paramsPP[array_index[n][j]] += epsilon
        #             paramsMM[array_index[n][i]] -= epsilon
        #             paramsMM[array_index[n][j]] -= epsilon
        #             paramsPM[array_index[n][i]] += epsilon
        #             paramsPM[array_index[n][j]] -= epsilon
        #             paramsMP[array_index[n][i]] -= epsilon
        #             paramsMP[array_index[n][j]] += epsilon
        #
        #             #lnlikelihood at those positions
        #             if use_prior:
        #                 pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise, vary_red_noise) + pta.get_lnprior(paramsPP)
        #                 mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise, vary_red_noise) + pta.get_lnprior(paramsMM)
        #                 pm = QB_logl.get_lnlikelihood(paramsPM, vary_white_noise, vary_red_noise) + pta.get_lnprior(paramsPM)
        #                 mp = QB_logl.get_lnlikelihood(paramsMP, vary_white_noise, vary_red_noise) + pta.get_lnprior(paramsMP)
        #             else:
        #                 pp = QB_logl.get_lnlikelihood(paramsPP, vary_white_noise, vary_red_noise)
        #                 mm = QB_logl.get_lnlikelihood(paramsMM, vary_white_noise, vary_red_noise)
        #                 pm = QB_logl.get_lnlikelihood(paramsPM, vary_white_noise, vary_red_noise)
        #                 mp = QB_logl.get_lnlikelihood(paramsMP, vary_white_noise, vary_red_noise)
        #
        #             #calculate off-diagonal elements of the Hessian from a central finite element scheme
        #             #note the minus sign compared to the regular Hessian
        #             #print('off-diagonal')
        #             #print(pp,mp,pm,mm)
        #             #print(-(pp - mp - pm + mm)/(4.0*epsilon*epsilon))
        #             fisher[1,i+n*dim,j+n*dim] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
        #             fisher[1,j+n*dim,i+n*dim] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)


        #correct for the given temperature of the chain
        fisher = fisher/T_chain
        print('fisher matrix: ', fisher)


        try:
            #Filter nans and infs and replace them with 1s
            #this will imply that we will set the eigenvalue to 100 a few lines below
            #UPDATED so that 0s are also replaced with 1.0
            FISHER = np.where(np.isfinite(fisher[1,:,:]) * (fisher[1,:,:]!=0.0), fisher[1,:,:], 1.0)
            if not np.array_equal(FISHER, fisher[1,:,:]):
                print("Changed some nan elements in the Fisher matrix to 1.0")
                #print(fisher[k,:,:])
                #print(FISHER)

            #Find eigenvalues and eigenvectors of the Fisher matrix
            w, v = np.linalg.eig(FISHER)

            #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
            eig_limit = 1.0

            W = np.where(np.abs(w)>eig_limit, w, eig_limit)
            #print(W)
            #print(np.sum(v**2, axis=0))
            #if T_chain==1.0: print(W)
            #if T_chain==1.0: print(v)

            eig.append( (np.sqrt(1.0/np.abs(W))*v).T )
            #print(np.sum(eig**2, axis=1))
            #if T_chain==1.0: print(eig)

        except:
            print("An Error occured in the eigenvalue calculation")
            eig.append( np.array(False) )

    return np.array(eig)


################################################################################
#
#FUNCTION TO EASILY SET UP A LIST OF PTA OBJECTS
#
################################################################################
def get_pta(pulsars, vary_white_noise=True, include_equad = False, include_ecorr = False, wn_backend_selection=False, noisedict_file=None, include_rn=True, vary_rn=True, include_per_psr_rn=False, vary_per_psr_rn=False, include_gwb=True, max_n_wavelet=1, efac_start=1.0, rn_amp_prior='uniform', rn_log_amp_range=[-18,-11], rn_params=[-14.0,1.0], gwb_amp_prior='uniform', gwb_log_amp_range=[-18,-11], wavelet_amp_prior='uniform', wavelet_log_amp_range=[-18,-11], per_psr_rn_amp_prior='uniform', per_psr_rn_log_amp_range=[-18,-11], prior_recovery=False, max_n_glitch=1, glitch_amp_prior='uniform', glitch_log_amp_range=[-18, -11], t0_min=0.0, t0_max=10.0, f0_min=3.5e-9, f0_max=1e-7, TF_prior=None, use_svd_for_timing_gp=True, tref=53000*86400):
    #setting up base model
    if vary_white_noise:
        efac = parameter.Uniform(0.01, 10.0)
        if include_equad:
            equad = parameter.Uniform(-8.5, -5)
        if include_ecorr:
            ecorr = parameter.Uniform(-8.5, -5)

    else:
        print('Constant efac!!')
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
    # if wn_backend_selection == True:
    #     wn_backend_selection = 'backend'
    #     print('Backend selected!!!!!!!!!!!!!!' + wn_backend_selection)
    tm = gp_signals.TimingModel(use_svd=use_svd_for_timing_gp)
    #wn = blocks.white_noise_block(vary=vary_white_noise, inc_ecorr=include_ecorr, gp_ecorr=True, select=wn_backend_selection)

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
            print('constant, non varying per pulsar rn!')
            '''
            #CURRENTLY DOES NOT SET AMPS/GAMMA VALUES
            #This defaults to val=None (value should be set later)
            #I'm also not sure this will ever be used, since we will want to see what per pulsar rn
            #gets put into wavelets (at least I think that's how this will work)
            '''
            #For now, set amplitude to median of per_psr_rn_amp_prior and gamma to 3.2 (max likelihood in 15yr GWB search)
            log10_A = parameter.Constant(val = (per_psr_rn_log_amp_range[1]-per_psr_rn_log_amp_range[0])/2)
            gamma = parameter.Constant(val = 3.2)
        #else:
            #raise ValueError('vary_per_psr_rn must one of the following list: "uniform", "log-uniform", or "constant"')
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

    #make base models including GWB
    if include_gwb:
        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)
        amp_name = 'gw_log10_A'
        if gwb_amp_prior == 'uniform':
            log10_Agw = parameter.LinearExp(gwb_log_amp_range[0], gwb_log_amp_range[1])(amp_name)
        elif gwb_amp_prior == 'log-uniform':
            log10_Agw = parameter.Uniform(gwb_log_amp_range[0], gwb_log_amp_range[1])(amp_name)

        gam_name = 'gw_gamma'
        gamma_val = 13.0/3
        gamma_gw = parameter.Constant(gamma_val)(gam_name)

        cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        gwb = gp_signals.FourierBasisCommonGP(cpl, utils.hd_orf(), coefficients=False,
                                              components=30, Tspan=Tspan,
                                              modes=None, name='gw')

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

    s =  tm + wn

    #Check what to include in model based on flags
    if include_per_psr_rn:
        s += per_psr_rn
    if vary_rn:
        s += rn
    if include_gwb:
        s += gwb
    for i in range(max_n_glitch):
        s += glitches[i]
    for i in range(max_n_wavelet):
        s += wavelets[i]
    model = []
    for p in pulsars:
        model.append(s(p))

    #set the likelihood to unity if we are in prior recovery mode
    if prior_recovery:
        if TF_prior is None:
            pta = get_prior_recovery_pta(signal_base.PTA(model))
        else:
            pta = get_tf_prior_pta(signal_base.PTA(model), TF_prior, n_wavelet, prior_recovery=True)

    #If noisedict_file is filepath, do this.
    elif noisedict_file is not None:
        #print('found noise dict: ', noisedict_file)
        if isinstance(noisedict_file, str):
            with open(noisedict_file, 'r') as fp: #rb
                noisedict = json.load(fp)
                #print('Noise dictionary: ', noisedict.keys(), "\n")
                pta_load = signal_base.PTA(model)
                #print('param names', pta_load.param_names, '\n')
                pta_load.set_default_params(noisedict)
                #print('PTA Summary: ', pta_load.summary())
            if TF_prior is None:
                pta = pta_load
            else:
                pta = get_tf_prior_pta(pta, TF_prior, n_wavelet)
        #if noisedict_file is dictionary, do this.
        else:
            pta_load = signal_base.PTA(model)
            #print('param names', pta_load.param_names, '\n')
            pta_load.set_default_params(noisedict_file)
                #print('PTA Summary: ', pta_load.summary())
            if TF_prior is None:
                pta = pta_load
            else:
                pta = get_tf_prior_pta(pta, TF_prior, n_wavelet)
        #print('PTA params: ', pta.params, "\n")
    else:
        if TF_prior is None:
            pta = signal_base.PTA(model)
        else:
            pta = get_tf_prior_pta(signal_base.PTA(model), TF_prior, n_wavelet)

    #Parsing through parameters in pta -> able to grab indexes for reference later
    #Will (hopefully) be easier to read the code this way.


    key_list = pta.param_names
    #print('key list: ', key_list)
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
    #
    # per_puls_indx = np.zeros((len(pulsars),num_wn_terms+num_rn_terms), dtype = 'int')
    # if vary_per_psr_rn and not vary_white_noise:
    #     for k in range(len(pulsars)):
    #         per_puls_indx[k,0] = key_list.index(pulsars[k].name + '_red_noise_gamma')
    #         per_puls_indx[k,1] = key_list.index(pulsars[k].name + '_red_noise_log10_A')
    # elif vary_white_noise and not vary_per_psr_rn:
    #     for k in range(len(pulsars)):
    #         wn_indx[k,0] = key_list.index(pulsars[l].name + '_efac')
    #         if include_equad_ecorr:
    #             wn_indx[k,1] = key_list.index(pulsars[l].name + '_log10_ecorr')
    #             wn_indx[k,2] = key_list.index(pulsars[l].name + '_log10_tnequad')
    #         #ecor and equad terms
    # elif vary_white_noise and vary_per_psr_rn:
    #     for k in range(len(pulsars)):
    #         per_puls_indx[k,0] = key_list.index(pulsars[k].name + '_red_noise_gamma')
    #         per_puls_indx[k,1] = key_list.index(pulsars[k].name + '_red_noise_log10_A')
    #         per_puls_indx[k,2] = key_list.index(pulsars[k].name + '_efac')
    #         if include_equad_ecorr:
    #             per_puls_indx[k,3] = key_list.index(pulsars[k].name + '_log10_ecorr')
    #             per_puls_indx[k,4] = key_list.index(pulsars[k].name + '_log10_tnequad')

    #Issue with indexing over wn params when including backend. Parsing receiver/frequency combinations
    #for all pulsars will introduce a lot of headache.

    rn_indx = np.zeros((2), dtype = 'int')
    if vary_rn:
        rn_indx[0] = key_list.index('com_rn_gamma')
        rn_indx[1] = key_list.index('com_rn_log10_A')
    gwb_indx = np.zeros((1), dtype = 'int')
    if include_gwb:
        gwb_indx[0] = key_list.index('gw_log10_A')

    return pta, glitch_indx, wavelet_indx, per_puls_indx, rn_indx, gwb_indx, num_per_puls_param_list

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
def get_gwb_on(samples, j, i, max_n_wavelet, max_n_glitch, num_noise_params):
    return int(samples[j,i,2+max_n_wavelet*10+max_n_glitch*6+num_noise_params]!=0.0)

def strip_samples(samples, j, i, n_wavelet, max_n_wavelet, n_glitch, max_n_glitch):
    return np.delete(samples[j,i,2:], list(range(n_wavelet*10,max_n_wavelet*10))+list(range(max_n_wavelet*10+n_glitch*6,max_n_wavelet*10+max_n_glitch*6)) )
'''why'''
def get_n_wavelet(samples, j, i):
    return int(samples[j,i,0])

def get_n_glitch(samples, j, i):
    return int(samples[j,i,1])

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
