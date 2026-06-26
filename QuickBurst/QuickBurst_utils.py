"""
QuickBurst utility classes for shared chain parameters.

ChainParams holds the PTA object, pulsar data, and other immutable
parameters that are shared across all parallel tempering chains.
This avoids unnecessary memory duplication when multiple QB_logl
objects are instantiated (one per chain).

Note on Python reference semantics:
    When QB_logl stores ``self.pta = pta``, Python stores a *reference*
    to the same PTA object — it does NOT copy it.  ChainParams is the
    single owner; every QB_logl chain points to the same underlying
    object in memory.
"""
from __future__ import division

import numpy as np
import json
from numba.typed import List
from QuickBurst.QuickBurst_MCMC import get_pta


class ChainParams:
    """Container for shared, immutable data used by all MCMC chains.

    Container for shared, immutable parameters for all MCMC chains.
    Builds the PTA and stores all model-definition settings.

    :param pulsars:
        List of pulsar objects.
    :param max_n_wavelet:
        Maximum number of GW signal wavelets to include in PTA model.
    :param min_n_wavelet:
        Minimum number of GW signal wavelets to include in PTA model.
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
    :param t0_min:
        The minimum epoch time with reference to the beginning of the data set.
    :param t0_max:
        The maximum epoch time for the data set.
    :param TF_prior_file:
        Custom prior for t0 and f0 shape parameters. [None] by default.
    :param f0_min:
        Lower bound on GW signal wavelet and noise transient wavelet frequency in Hz. [3.5e-9] by default.
    :param f0_max:
        Upper bound on GW signal wavelet and noise transient wavelet frequency in Hz. [1e-7] by default.
    :param tau_min_in:
        Lower bound on GW signal wavelet and noise transient wavelet width in years. [0.2] by default.
    :param tau_max_in:
        Upper bound on GW signal wavelet and noise transient wavelet width in years. [5] by default.
    """

    def __init__(self, psrs, tref,
                 max_n_wavelet=1, min_n_wavelet=0, 
                 max_n_glitch=1,
                 # White noise options
                 vary_white_noise=False,
                 include_equad=False, include_ecorr=False,
                 include_efac=False, wn_backend_selection=False,
                 noisedict=None, efac_start=1.0,
                 equad_range=(-8.5, -5), ecorr_range=(-8.5, -5),
                 # Red noise options
                 include_rn=False, vary_rn=False,
                 rn_amp_prior='uniform', rn_log_amp_range=(-18, -11),
                 rn_params=(-14.0, 1.0),
                 # Per-pulsar red noise
                 include_per_psr_rn=False, vary_per_psr_rn=False,
                 per_psr_rn_start_file=None,
                 per_psr_rn_amp_prior='uniform',
                 per_psr_rn_log_amp_range=(-18, -11),
                 # Wavelet priors
                 wavelet_amp_prior='uniform',
                 wavelet_log_amp_range=(-18, -11),
                 # Glitch priors
                 glitch_amp_prior='uniform',
                 glitch_log_amp_range=(-18, -11),
                 # Shape parameter bounds
                 t0_min=0.0, t0_max=10.0,
                 f0_min=3.5e-9, f0_max=1e-7,
                 tau_min=0.2, tau_max=5.0,
                 # Misc
                 TF_prior_file=None,
                 TF_prior=None,
                 use_svd_for_timing_gp=True,
                 prior_recovery=False):

        #Pulsars
        self.psrs = psrs
        self.Npsr = len(psrs)
        self.tref = tref

        #Store pulsar information
        self.toas = List([psr.toas - tref for psr in psrs])
        self.residuals = List([psr.residuals for psr in psrs])
        self.pos = np.zeros((self.Npsr, 3))
        
        for i in range(self.Npsr):
            self.pos[i] = psrs[i].pos

        #Model configuration
        self.max_n_wavelet = max_n_wavelet
        self.min_n_wavelet = min_n_wavelet
        self.max_n_glitch = max_n_glitch
        self.wn_backend_selection = wn_backend_selection
        self.vary_white_noise = vary_white_noise
        self.vary_rn = vary_rn
        self.include_rn = include_rn
        self.include_per_psr_rn = include_per_psr_rn
        self.vary_per_psr_rn = vary_per_psr_rn
        self.prior_recovery = prior_recovery
        self.wavelet_amp_prior = wavelet_amp_prior
        self.wavelet_log_amp_range = list(wavelet_log_amp_range)
        self.glitch_amp_prior = glitch_amp_prior
        self.glitch_log_amp_range = list(glitch_log_amp_range)
        self.TF_prior = TF_prior
        self.TF_prior_file = TF_prior_file
        self.per_psr_rn_start_file = per_psr_rn_start_file
        
        #Prior information
        self.rn_amp_prior = rn_amp_prior
        self.rn_log_amp_range = rn_log_amp_range
        self.per_psr_rn_amp_prior = per_psr_rn_amp_prior
        self.per_psr_rn_log_amp_range = per_psr_rn_log_amp_range
        self.include_equad = include_equad
        self.equad_range = equad_range
        self.include_ecorr = include_ecorr
        self.ecorr_range = ecorr_range
        self.include_efac = include_efac
        self.rn_params = rn_params
        self.efac_start = efac_start

        #Shape param ranges
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.t0_max = t0_max
        self.t0_min = t0_min
        self.tau_max = tau_max
        self.tau_min = tau_min

        #Make PTA
        get_pta_result = get_pta(
            psrs,
            vary_white_noise=vary_white_noise,
            include_equad=include_equad,
            include_ecorr=include_ecorr,
            include_efac=include_efac,
            wn_backend_selection=wn_backend_selection,
            noisedict=noisedict,
            efac_start=efac_start,
            equad_range=equad_range,
            ecorr_range=ecorr_range,
            include_rn=include_rn,
            vary_rn=vary_rn,
            rn_amp_prior=rn_amp_prior,
            rn_log_amp_range=rn_log_amp_range,
            rn_params=rn_params,
            include_per_psr_rn=include_per_psr_rn,
            vary_per_psr_rn=vary_per_psr_rn,
            per_psr_rn_amp_prior=per_psr_rn_amp_prior,
            per_psr_rn_log_amp_range=per_psr_rn_log_amp_range,
            max_n_wavelet=max_n_wavelet,
            wavelet_amp_prior=wavelet_amp_prior,
            wavelet_log_amp_range=wavelet_log_amp_range,
            max_n_glitch=max_n_glitch,
            glitch_amp_prior=glitch_amp_prior,
            glitch_log_amp_range=glitch_log_amp_range,
            t0_min=t0_min, t0_max=t0_max,
            f0_min=f0_min, f0_max=f0_max,
            tau_min=tau_min, tau_max=tau_max,
            TF_prior=TF_prior,
            use_svd_for_timing_gp=use_svd_for_timing_gp,
            tref=tref,
            prior_recovery=prior_recovery,
        )

        (self.pta, self.QB_FP, self.QB_FPI,
         self.glitch_indx, self.wavelet_indx,
         self.per_puls_indx, self.per_puls_rn_indx,
         self.per_puls_wn_indx, self.rn_indx,
         self.all_noiseparam_idxs,
         self.num_per_puls_param_list) = get_pta_result

        #parameter names
        self.param_names = self.pta.param_names
        self.num_params = len(self.param_names)
        self.Ts = List([np.asfortranarray(x) for x in self.pta.get_basis()])

    def summary(self):
        """Print a summary of shared chain parameters."""
        print(f"ChainParams summary:")
        print(f"  Pulsars:        {self.Npsr}")
        print(f"  Max wavelets:   {self.max_n_wavelet}")
        print(f"  Max glitches:   {self.max_n_glitch}")
        print(f"  PTA params:     {self.num_params}")
        print(f"  tref:           {self.tref:.1f} s")
        print(f"  Vary WN:        {self.vary_white_noise}")
        print(f"  Vary RN:        {self.vary_rn}")
        print(f"  Vary per-psr RN:{self.vary_per_psr_rn}")
        print(f"  Prior recovery: {self.prior_recovery}")
