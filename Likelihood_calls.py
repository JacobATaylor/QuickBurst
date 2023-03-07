#import packages
from __future__ import division

#%load_ext line_profiler
import numpy as np
import glob, json
import pickle
import os as os_pack
import matplotlib.pyplot as plt
import corner
#%matplotlib inline\n",
#%config InlineBackend.figure_format = 'retina'
import healpy as hp
import os, glob, json, pickle
import matplotlib.pyplot as plt
import numpy as np
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
import enterprise_wavelets as models
from enterprise.signals.deterministic_signals import Deterministic
from enterprise.signals.parameter import function
from la_forge.core import Core
from la_forge.diagnostics import plot_chains
from la_forge import rednoise
import la_forge
import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import re

import Fast_Burst_likelihood as FB

@profile
def run_function():
    #Loading in pickle and noise files
    pint_pickle = '/home/reyna/OS_15yr/15yr_data/v1p1_de440_pint_bipm2019.pkl'
    noise_file = '/home/reyna/OS_15yr/15yr_data/v1p1_wn_dict.json'
    #psrlist = np.loadtxt('/home/reyna/15yr_v1p0/15yr_v1-20211001T235643Z-001/15yr_v1/psrlist_15yr_pint.txt', dtype = str)
    with open(noise_file, 'r') as h:
        noise_params = json.load(h)
    with open(pint_pickle,'rb') as f:
        allpsrs = pickle.load(f)
    psrs = []
    for ii,p in enumerate(allpsrs):
        psrs.append(p)
    #Temporary to get code to not crash
    psrs = psrs[0:]
    psrlist = [psr.name for psr in psrs]

    glitches = []
    N_glitches = 5
    for i in range(N_glitches):
        log10_f0 = parameter.Uniform(np.log10(3.5e-9), np.log10(1e-7))("Glitch_"+str(i)+'_'+'log10_f0')
        phase0 = parameter.Uniform(0, 2*np.pi)("Glitch_"+str(i)+'_'+'phase0')
        tau = parameter.Uniform(0.2, 5)("Glitch_"+str(i)+'_'+'tau')
        t0 = parameter.Uniform(0.0, 10.0)("Glitch_"+str(i)+'_'+'t0')
        psr_idx = parameter.Uniform(-0.5, len(psrs)-0.5)("Glitch_"+str(i)+'_'+'psr_idx')
        log10_h = parameter.LinearExp(-10.5, -9)("Glitch_"+str(i)+'_'+'log10_h')
        glitch_wf = models.glitch_delay(log10_h = log10_h, tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0, tref=53000*86400,
                                                psr_float_idx = psr_idx, pulsars=psrs)
        glitches.append(deterministic_signals.Deterministic(glitch_wf, name='Glitch'+str(i) ))

    #wavelet models
    wavelets = [] #added "wavelet_" to names to help with seperation
    N_wavelets = 5
    for i in range(N_wavelets):
        log10_f0 = parameter.Uniform(np.log10(3.5e-9), np.log10(1e-7))("wavelet_"+str(i)+'_'+'log10_f0')
        cos_gwtheta = parameter.Uniform(-1, 1)("wavelet_"+str(i)+'_'+'cos_gwtheta')
        gwphi = parameter.Uniform(0, 2*np.pi)("wavelet_"+str(i)+'_'+'gwphi')
        psi = parameter.Uniform(0, np.pi)("wavelet_"+str(i)+'_'+'gw_psi')
        phase0 = parameter.Uniform(0, 2*np.pi)("wavelet_"+str(i)+'_'+'phase0')
        phase0_cross = parameter.Uniform(0, 2*np.pi)("wavelet_"+str(i)+'_'+'phase0_cross')
        tau = parameter.Uniform(0.2, 5)("wavelet_"+str(i)+'_'+'tau')
        t0 = parameter.Uniform(0.0, 10.0)("wavelet_"+str(i)+'_'+'t0')
        log10_h = parameter.LinearExp(-18,-11)("wavelet_"+str(i)+'_'+'log10_h')
        log10_h_cross = parameter.LinearExp(-18,-11)("wavelet_"+str(i)+'_'+'log10_h_cross')
        wavelet_wf = models.wavelet_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_h = log10_h, log10_h2=log10_h_cross,
                                          tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0, phase02=phase0_cross,
                                          epsilon = None, psi=psi, tref=53000*86400)
        wavelets.append(deterministic_signals.Deterministic(wavelet_wf, name='wavelet'+str(i)))

    tm = gp_signals.TimingModel(use_svd=True)
    wn = blocks.white_noise_block(vary=False, inc_ecorr=False)

    Tspan = ee_model_utils.get_tspan(psrs)
    rn = blocks.red_noise_block(psd='powerlaw', prior = 'log-uniform', Tspan=Tspan, components = 30)

    #s = base_model
    s = tm + wn + rn
    for i in range(N_glitches):
        s += glitches[i]
    for j in range(N_wavelets):
        s += wavelets[j]
    model = []
    for p in psrs:
        model.append(s(p))
    with open(noise_file, 'r') as fp:
        noisedict = json.load(fp)
        pta = signal_base.PTA(model)
        pta.set_default_params(noisedict)

    #run1
    d0_15y = parameter.sample(pta.params)
    x0_15y = np.array([d0_15y[par.name] for par in pta.params])
    FB_15y = FB.FastBurst(pta = pta, psrs = psrs, params = d0_15y, Npsr = len(psrs), tref=53000*86400, Nglitch = N_glitches, Nwavelet = N_wavelets, rn_vary = True)
    FB_15y.get_lnlikelihood(x0_15y)
    #run2
    d0_15y = parameter.sample(pta.params)
    x0_15y = np.array([d0_15y[par.name] for par in pta.params])
    FB_15y.get_lnlikelihood(x0_15y)
    #run3
    d0_15y = parameter.sample(pta.params)
    x0_15y = np.array([d0_15y[par.name] for par in pta.params])
    FB_15y.get_lnlikelihood(x0_15y)

    # NN = 20
    # log_L_Ent_15y = []
    # log_L_Fast_15y = []
    # for n in range(NN):
    #     d0_15y = parameter.sample(pta.params)
    #     x0_15y = np.array([d0_15y[par.name] for par in pta.params])
    #
    #     log_L_Ent_15y.append(pta.get_lnlikelihood(x0_15y))
    #     log_L_Fast_15y.append(FB_15y.get_lnlikelihood(x0_15y))
    #     print('run ',n)
    # log_L_Ent_15y = np.array(log_L_Ent_15y)
    # log_L_Fast_15y = np.array(log_L_Fast_15y)

run_function()
