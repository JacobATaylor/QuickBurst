"""
C 2024 Jacob, Rand, and Bence fast Burst likelihood

QuickBurst likelihood for generic GW burst searches.

"""

import numpy as np
import numba as nb
from numba import njit,prange
from numba.experimental import jitclass
from numba.typed import List
import scipy.linalg as sl
from QuickBurst.lapack_wrappers import solve_triangular
import copy

from enterprise import constants as const
from enterprise_extensions.frequentist import Fe_statistic as FeStat
from enterprise.signals import utils


#########
#strucutre overview
#
#Class to hold info about pta, params, psrs arrays
#   function calc M and N matrixies to be used
#   function to calculate likelihoods
########

class QuickBurst:
    """Class for generating the fast generic burst likelihood. """
    def __init__(self, pta, psrs, params, Npsr, tref, Nglitch, Nwavelet, Nglitch_max, Nwavelet_max, rn_vary, wn_vary,  prior_recovery=False):
        """
        get Class for generating fast generic burst likelihood.

        :param pta:
            'enterprise' pta object
        :param psrs:
            Pickled pulsar par/tim objects as returned by enterprise.Pulsar(psr) for all pulsars in pta.
        :param params:
            Parameter dictionary as generated from pta.params. Usually instantiated in QuickBurst_MCMC.
        :param Npsr:
            Number of pulsars in the pta.
        :param tref:
            Reference time for the beginning MJD of the dataset.
        :param Nglitch:
            Number of starting noise transient wavelets in the PTA model. If not resuming, random draw between 0 and Nglitch_max.
        :param Nwavelet:
            Number of starting GW signal wavelets in the PTA model. If not resuming from a chain, random draw between 0 and Nwavelet_max.
        :param rn_vary:
            If True, either varying intrinsic pulsar RN or CURN. [False] by default.
        :param wn_vary:
            If true, varying intrinsic pulsar WN. [False] by default.
        :param prior_recovery:
            If True, we return constant likelihood to be used for prior recovery diagnostic tests. [False] by default.
        """
        #model parameters that shouldn't change for a run
        self.pta = pta
        self.psrs = psrs
        self.Npsr = Npsr
        self.params = params
        #important matrixies
        self.Nvecs = list(self.pta.get_ndiag(self.params))
        self.TNTs = self.pta.get_TNT(self.params)
        self.Ts = self.pta.get_basis()
        #save values that may need to be recovered when doing MCMC
        self.params_previous = params
        self.Nvecs_previous = list(self.pta.get_ndiag(self.params))
        self.TNTs_previous = self.pta.get_TNT(self.params)

        #pulsar information
        self.toas = list([psr.toas - tref for psr in psrs])
        self.residuals = list([psr.residuals for psr in psrs])
        self.pos = np.zeros((self.Npsr,3))
        for i in range(self.Npsr):
            self.pos[i] = self.psrs[i].pos

        self.logdet = 0
        for (l,m) in self.pta.get_rNr_logdet(self.params): #Only using this for logdet term because the rNr term removes the deterministic signal durring generation
            self.logdet += m
        self.logdet_previous = np.copy(self.logdet)
        #terms used in cholesky component of the dot product (only needs to be updated per-pulsar)
        self.invTN = []
        self.CholSigma = []
        self.Ndiag = []

        phiinvs = self.pta.get_phiinv(self.params, logdet=True, method='partition')
        temp_logdetphi = []
        for ii in range(self.Npsr):
            TNT = self.TNTs[ii]
            T = self.Ts[ii].T
            #phiinv = phiinvs[ii]
            phiinv, logdetphi_loc = phiinvs[ii]
            temp_logdetphi.append(logdetphi_loc)
            Ndiag = 1/self.Nvecs[ii]
            self.Ndiag.append(Ndiag)
            self.invTN.append(T * Ndiag)
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
            chol_Sigma,lower = sl.cho_factor(Sigma.T,lower=True,overwrite_a=True,check_finite=False)
            self.CholSigma.append(chol_Sigma)

        self.CholSigma_previous = np.copy(self.CholSigma)
        self.invTN_previous = np.copy(self.invTN)
        self.Ndiag_previous = np.copy(self.Ndiag)
        #save likelihood terms if updating is not necessary
        self.wn_vary = wn_vary
        self.rn_vary = rn_vary
        self.no_step = False
        self.resres_logdet = np.copy(self.logdet + resres_logdet_calc(self.Npsr, self.residuals, self.invTN, self.Ndiag, temp_logdetphi, self.CholSigma))
        self.resres_logdet_previous = np.copy(self.resres_logdet)

        #check if we are doing a prior recovery runs
        self.prior_recovery=prior_recovery

        #max number of glitches and signals that can be handeled
        self.Nglitch = Nglitch
        self.Nwavelet = Nwavelet
        self.Nglitch_previous = Nglitch
        self.Nwavelet_previous = Nwavelet
        #holds parameters parsed from current run
        self.wavelet_prm = np.zeros((Nwavelet_max, 10))# in order GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, f0_w, tau_w, t0_w
        self.glitch_prm = np.zeros((Nglitch_max, 6))# in order f0, log10_h, phase0, psr_idx, t0, tau
        #parse the pulsar indexes from parameters
        #these are set to -1 by default because 0 would make us recalculate the first pulasr every time even if there is now glitch present
        self.glitch_pulsars = -1*np.ones((len(self.glitch_prm[:,3]))) #np.zeros((len(self.glitch_prm[:,3])))
        for el in range(Nglitch):
            self.glitch_pulsars[el] = round(self.glitch_prm[el,3])
        self.glitch_pulsars_previous = np.copy(self.glitch_pulsars)
        #holds previous updated shape params
        self.wavelet_saved = np.zeros((Nwavelet_max, 10))# these should only be saved over if a step is accepeted
        self.glitch_saved = np.zeros((Nglitch_max, 6))# these should only be saved over if a step is accepeted
        #max possible size of NN and MMs
        self.MMs = np.zeros((self.Npsr,2*Nwavelet_max + 2*Nglitch_max,2*Nwavelet_max + 2*Nglitch_max))
        self.NN = np.zeros((self.Npsr,2*Nwavelet_max + 2*Nglitch_max))
        #most recent accepeted NN and MMs
        self.MMs_previous = np.zeros((self.Npsr,2*Nwavelet_max + 2*Nglitch_max,2*Nwavelet_max + 2*Nglitch_max))
        self.NN_previous = np.zeros((self.Npsr,2*Nwavelet_max + 2*Nglitch_max))
        #space to story current shape parameters to decided it we need to update NN and MM
        self.Saved_Shape = np.zeros((3*Nwavelet_max + 3*Nglitch_max))
        self.key_list = list(self.params)#makes a list of the keys in the params Dictionary
        self.pta_param_names = self.pta.param_names

        self.glitch_indx = np.zeros((Nglitch_max,6)) # in order f0, log10_h, phase0, psr_idx, t0, tau
        self.wavelet_indx = np.zeros((Nwavelet_max,10)) # in order GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, f0_w, tau_w, t0_w
        for i in range(Nglitch_max):
            self.glitch_indx[i,0] = self.key_list.index('Glitch_'+str(i)+'_log10_f0')
            self.glitch_indx[i,1] = self.key_list.index('Glitch_'+str(i)+'_log10_h')
            self.glitch_indx[i,2] = self.key_list.index('Glitch_'+str(i)+'_phase0')
            self.glitch_indx[i,3] = self.key_list.index('Glitch_'+str(i)+'_psr_idx')
            self.glitch_indx[i,4] = self.key_list.index('Glitch_'+str(i)+'_t0')
            self.glitch_indx[i,5] = self.key_list.index('Glitch_'+str(i)+'_tau')

        #wavelet models
        for j in range(Nwavelet_max):
            self.wavelet_indx[j,0] = self.key_list.index('wavelet_'+str(j)+'_log10_f0')
            self.wavelet_indx[j,1] = self.key_list.index('wavelet_'+str(j)+'_cos_gwtheta')
            self.wavelet_indx[j,2] = self.key_list.index('wavelet_'+str(j)+'_gwphi')
            self.wavelet_indx[j,3] = self.key_list.index('wavelet_'+str(j)+'_gw_psi')
            self.wavelet_indx[j,4] = self.key_list.index('wavelet_'+str(j)+'_log10_h')
            self.wavelet_indx[j,5] = self.key_list.index('wavelet_'+str(j)+'_log10_h_cross')
            self.wavelet_indx[j,6] = self.key_list.index('wavelet_'+str(j)+'_phase0')
            self.wavelet_indx[j,7] = self.key_list.index('wavelet_'+str(j)+'_phase0_cross')
            self.wavelet_indx[j,8] = self.key_list.index('wavelet_'+str(j)+'_t0')
            self.wavelet_indx[j,9] = self.key_list.index('wavelet_'+str(j)+'_tau')

        self.sigmas = np.zeros((self.Npsr, Nwavelet_max + Nglitch_max, 2))
        self.sigmas_previous = np.zeros((self.Npsr, Nwavelet_max + Nglitch_max, 2))

    ######
    #Function that may be useful to help modify the M and N matrices without recalculating them
    #Specifically for cases of changing/adding glitches and wavelets
    ######
    def M_N_RJ_helper(self, x0, n_wavelet, n_glitch, remove_index = 0, adding = False, wavelet_change = False, glitch_change = False):
        """
        Function for calculating likelihood when pta parameter space changes mid sampling.

        :param x0:
            List of parameter values in proposed MCMC step. Should have dimensions of [len(pta.params)].
        :param n_wavelet:
            Number of GW signal wavelets proposed in current MCMC step.
        :param n_glitch:
            Number of noise transient wavelets (glitches) proposed in current MCMC step.
        :param remove_index:
            If adding = False, corresponds to the GW signal wavelet or noise transient (glitch) wavelet being removed from the PTA model.
        :param adding:
            If True, add GW signal wavelet or noise transient (glitch) wavelet to the PTA model. [False] by default.
        :param wavelet_change:
            If True, changing the number of GW signal wavelets in PTA model. [False] by default.
        :param glitch_change:
            If True, changing the number of noise transient (glitch) wavelets in PTA model. [False] by default.

        :return temp_like:      Bayesian likelihood given sample x0.
        """
        self.Nglitch_previous = self.Nglitch
        self.Nwavelet_previous = self.Nwavelet

        self.rn_vary = False
        self.wn_vary = False
        #Need to resize glitch_prm, wavelet_prm here when adding or removing glitch or wavelet. Which means also updating glitch_indx and wavelet_indx
        #in this edge case.
        self.glitch_prm, self.wavelet_prm = get_parameters(x0, self.glitch_prm, self.wavelet_prm, self.glitch_indx, self.wavelet_indx, n_glitch, n_wavelet)
        #Save previous M and N states
        self.NN_previous = np.copy(self.NN)
        self.MMs_previous = np.copy(self.MMs)
        self.CholSigma_previous = np.copy(self.CholSigma)
        self.invTN_previous = np.copy(self.invTN)
        self.Ndiag_previous = np.copy(self.Ndiag)

        #Need to make sure we update inverse cholesky matrix when in the edge case of 0 wavelets and 0 glitches
        #Cholesky matrix changes when we add wavelets/glitches, but still doesn't trigger when varying noise in this case.
        ####### start of cov calc stuff #######
        if self.Nwavelet + self.Nglitch == 0:

            d0 = dict((k, v) for k, v in zip(self.pta_param_names, x0))
            self.params_previous = np.copy(self.params)
            self.params = d0

            _ = self.TN_calculator()

        self.glitch_pulsars_previous = np.copy(self.glitch_pulsars)
        if glitch_change:
            self.glitch_pulsars = -1*np.ones((len(self.glitch_prm[:,3])))
            for el in range(n_glitch):#len(self.glitch_prm[:,3])):
                self.glitch_pulsars[el] = round(self.glitch_prm[el,3])

        #Set to new numbers of wavelets and glitches. Gets reset in save_values to previous values if step is rejected.
        self.Nglitch = n_glitch
        self.Nwavelet = n_wavelet

        #Set dif_flag to zeros initially and populate if things change.
        dif_flag = np.zeros(((self.Nwavelet + self.Nglitch)))

        if wavelet_change:
            if not adding:
                #Removing wavelet, delete row/column and add back to the end.
                for ii in range(self.Npsr):
                    #Copying both sin/cos terms in NN matrix
                    wavelet_copy_NN = np.copy(self.NN[ii, 2*remove_index:2*remove_index+2])*0
                    #Deleting relevant sin/cos filter function terms from NN
                    self.NN[ii, :-2] = np.delete(self.NN[ii], (2*remove_index,2*remove_index+1))
                    #Append terms to end of NN matrix
                    self.NN[ii, -2:] = wavelet_copy_NN
                    #copying both sin/cos terms (and cross terms) for MM matrix
                    #Need addition of 2 to get both terms, indexing w/ +2 doesn't include last term
                    wavelet_copy_MM_1 = np.copy(self.MMs[ii, 2*remove_index:2*remove_index+2,:])*0
                    #Deleting relevant sin/cos filter function terms from MM (along with cross terms for both row)
                    self.MMs[ii, :-2, :] = np.delete(self.MMs[ii], (2*remove_index,2*remove_index+1),axis=0)
                    #Append row to end
                    self.MMs[ii, -2:] = wavelet_copy_MM_1
                    wavelet_copy_MM_2 = np.copy(self.MMs[ii, :, 2*remove_index:2*remove_index+2])*0
                    #Deleting relevant sin/cos filter function terms from MM (along with cross terms for both col)
                    self.MMs[ii, :, :-2] = np.delete(self.MMs[ii], (2*remove_index,2*remove_index+1),axis=1)
                    #Append col to end
                    self.MMs[ii, :, -2:] = wavelet_copy_MM_2



            elif adding:
                #in M and N matrices, shift over all glitch terms in matrices, then dif flag index is only for new wavelet.
                #Adding wavelet
                for ii in range(self.Npsr):
                    #Copying all glitch sin/cos terms in NN matrix
                    wavelet_copy_NN = np.copy(self.NN[ii, 2*self.Nwavelet_previous:2*(self.Nglitch_previous+self.Nwavelet_previous)])
                    #Append terms shifted by one set of sin and cos terms
                    self.NN[ii, 2*self.Nwavelet_previous+2:2*(self.Nglitch_previous+self.Nwavelet_previous)+2] = wavelet_copy_NN
                    #copying both sin/cos terms (and cross terms) for MM matrix
                    #Need addition of 2 to get both terms
                    wavelet_copy_MM_1 = np.copy(self.MMs[ii, 2*self.Nwavelet_previous:2*(self.Nglitch_previous+self.Nwavelet_previous),:])
                    #shift row over by 2
                    self.MMs[ii, 2*self.Nwavelet_previous+2:2*(self.Nglitch_previous+self.Nwavelet_previous)+2, :] = wavelet_copy_MM_1
                    wavelet_copy_MM_2 = np.copy(self.MMs[ii, :, 2*self.Nwavelet_previous:2*(self.Nglitch_previous+self.Nwavelet_previous)])
                    #shift col over by 2
                    self.MMs[ii, :, 2*self.Nwavelet_previous+2:2*(self.Nglitch_previous+self.Nwavelet_previous)+2] = wavelet_copy_MM_2

                dif_flag[self.Nwavelet_previous] = 1
                #recalculate parts of MM and NN if adding wavelet, or shift around stuff if removing
                self.NN, self.MMs = get_M_N(self.toas, self.residuals, self.Npsr, self.MMs, self.NN, self.invTN, self.CholSigma, self.Ndiag,
                                            self.Nwavelet, self.Nglitch, self.wavelet_prm, self.glitch_prm, self.glitch_pulsars, self.glitch_pulsars_previous, dif_flag)
                #print('after self.NN: ', self.NN)
                #print('After self.MMs: ', self.MMs)
        elif glitch_change:
            remove_index += n_wavelet
            if not adding:
                #Removing wavelet, delete row/column and add back to the end.
                for ii in range(self.Npsr):
                    #Copying both sin/cos terms in NN matrix
                    glitch_copy_NN = np.copy(self.NN[ii, 2*remove_index:2*remove_index+2])*0
                    #Deleting relevant sin/cos filter function terms from NN
                    self.NN[ii, :-2] = np.delete(self.NN[ii], (2*remove_index,2*remove_index+1))
                    #Append terms to end of NN matrix
                    self.NN[ii, -2:] = glitch_copy_NN

                    #copying both sin/cos terms (and cross terms) for MM matrix
                    #Need addition of 2 to get both terms, indexing w/ +2 doesn't include last term
                    glitch_copy_MM_1 = np.copy(self.MMs[ii, 2*remove_index:2*remove_index+2,:])*0
                    #Deleting relevant sin/cos filter function terms from MM (along with cross terms for both row)
                    self.MMs[ii, :-2, :] = np.delete(self.MMs[ii], (2*remove_index,2*remove_index+1),axis=0)
                    #Append row to end
                    self.MMs[ii, -2:] = glitch_copy_MM_1
                    glitch_copy_MM_2 = np.copy(self.MMs[ii, :, 2*remove_index:2*remove_index+2])*0
                    #Deleting relevant sin/cos filter function terms from MM (along with cross terms for both col)
                    self.MMs[ii, :, :-2] = np.delete(self.MMs[ii], (2*remove_index,2*remove_index+1),axis=1)
                    #Append col to end
                    self.MMs[ii, :, -2:] = glitch_copy_MM_2
            elif adding:
                dif_flag[self.Nwavelet_previous+self.Nglitch_previous] = 1 #should index to the value after the currently active parts
                self.NN, self.MMs = get_M_N(self.toas, self.residuals, self.Npsr, self.MMs, self.NN, self.invTN, self.CholSigma, self.Ndiag,
                                            self.Nwavelet, self.Nglitch, self.wavelet_prm, self.glitch_prm, self.glitch_pulsars, self.glitch_pulsars_previous, dif_flag)
            #recalculate parts of MM and NN if adding glitch, or shift around stuff if removing
        #fast calculation of the lnlikelihood
        if self.prior_recovery:
            return 1.0

        self.sigmas_previous = np.copy(self.sigmas)
        self.sigmas = get_sigmas_helper(self.pos, self.sigmas, self.glitch_pulsars, self.Npsr, self.Nwavelet, self.Nglitch, self.wavelet_prm, self.glitch_prm)

        temp_like = likelihood_helper(self.sigmas, self.glitch_pulsars, self.resres_logdet, self.Npsr, self.Nwavelet, self.Nglitch, self.NN, self.MMs)
        return temp_like


    ######
    #calculates amplitudes for Signals
    ######
    def get_sigmas(self, glitch_pulsars):
        "Function that calls a numba jitted function for calculating filter coefficients."
        return get_sigmas_helper(self.pos, self.sigmas, glitch_pulsars, self.Npsr, self.Nwavelet, self.Nglitch, self.wavelet_prm, self.glitch_prm)


    ######
    #calculate the cholesky components for dot products
    ######
    def TN_calculator(self):
        """
        Function for calculating elements of the noise weighted inner product of two vectors (a|b).


        :return temp_logdetphi:
        """
        phiinvs = self.pta.get_phiinv(self.params, logdet=True, method='partition')
        self.Ndiag = []
        self.invTN = []
        self.CholSigma = []
        temp_logdetphi = []
        for ii in range(self.Npsr):
            Ndiag = 1 / self.Nvecs[ii]
            self.Ndiag.append(Ndiag)
            TNT = self.TNTs[ii]
            T = self.Ts[ii].T
            (phiinv, logdetphi) = phiinvs[ii]
            temp_logdetphi.append(logdetphi)
            self.invTN.append(T * Ndiag)
            Sigma = TNT + np.diag(phiinv) if phiinv.ndim == 1 else phiinv
            (chol_Sigma, lower) = sl.cho_factor(Sigma.T, lower = True, overwrite_a = True, check_finite = False)
            self.CholSigma.append(chol_Sigma)
        return temp_logdetphi

    #####
    #calculates lnliklihood for a set of signal parameters
    #####
    def get_lnlikelihood(self, x0, vary_white_noise = False, vary_red_noise = False, no_step = False):

        '''
        Class function to calculate check what parameters are changing in model, and to calculate likelihood.

        :param x0:
            List of parameter values to calculate likelihood with.
        :param vary_white_noise:
            If True, recalculates all inner products in likelihood. [False] by default.
        :param vary_red_noise:
            If True, recalculates all inner products in likelihood. [False] by default.
        :param no_step:
            If True, does not save overwrite Class attributes. Primarily used during Fisher matrix calculations. [False] by default.

        :return temp_like:
            Bayesian likelihood given x0.
        '''
        #prior recovery condition
        if self.prior_recovery:
            return 1.0

        #updating terms needed to calculate phiinv and logdet when varying RN
        self.rn_vary = vary_red_noise
        self.wn_vary = vary_white_noise
        self.no_step = no_step

        if self.rn_vary or self.wn_vary:

            d0 = dict((k, v) for k, v in zip(self.pta_param_names, x0))
            self.params_previous = np.copy(self.params)
            self.params = d0

        if self.wn_vary:
            self.Nvecs_previous = np.copy(self.Nvecs)
            self.TNTs_previous = np.copy(self.TNTs)
            self.Nvecs = list(self.pta.get_ndiag(self.params))
            self.TNTs = self.pta.get_TNT(self.params)
        #parse current parameters using dictionary
        #get_parameters needs to change to account for changing indexes for wavelet/glitch parameters
        self.glitch_prm, self.wavelet_prm = get_parameters(x0, self.glitch_prm, self.wavelet_prm, self.glitch_indx, self.wavelet_indx, self.Nglitch, self.Nwavelet)

        #check if shape params have changed between runs
        dif_flag = np.zeros((self.Nwavelet + self.Nglitch))
        for i in range(self.Nwavelet):
            if self.wavelet_saved[i,0] != self.wavelet_prm[i,0]:
                dif_flag[i] = 1
            if self.wavelet_saved[i,7] != self.wavelet_prm[i,7]:
                dif_flag[i] = 1
            if self.wavelet_saved[i,6] != self.wavelet_prm[i,6]:
                dif_flag[i] = 1
        for i in range(self.Nglitch):
            if self.glitch_saved[i,0] != self.glitch_prm[i,0]:
                dif_flag[self.Nwavelet+i] = 1
            if self.glitch_saved[i,4] != self.glitch_prm[i,4]:
                dif_flag[self.Nwavelet+i] = 1
            if self.glitch_saved[i,5] != self.glitch_prm[i,5]:
                dif_flag[self.Nwavelet+i] = 1
            if self.glitch_saved[i,3] != self.glitch_prm[i,3]:
                dif_flag[self.Nwavelet+i] = 1


        #parse the pulsar indexes from parameters
        self.glitch_pulsars_previous = np.copy(self.glitch_pulsars)
        self.glitch_pulsars = -1*np.ones((len(self.glitch_prm[:,3])))
        for el in range(self.Nglitch):
            self.glitch_pulsars[el] = round(self.glitch_prm[el,3])

        #calculate the amplitudes of noise transients and wavelets
        self.sigmas_previous = np.copy(self.sigmas)
        self.sigmas = self.get_sigmas(self.glitch_pulsars)

        #if we have new shape parameters, find the NN and MM matricies from filter functions
        self.NN_previous = np.copy(self.NN)
        self.MMs_previous = np.copy(self.MMs)
        self.CholSigma_previous = np.copy(self.CholSigma)
        self.invTN_previous = np.copy(self.invTN)
        self.Ndiag_previous = np.copy(self.Ndiag)

        #If varying any noise, recalculate all of M and N
        if self.rn_vary  or self.wn_vary:
            temp_logdetphi = self.TN_calculator()
            dif_flag = np.ones((self.Nwavelet + self.Nglitch))
        if 1 in dif_flag:
            self.NN_previous = np.copy(self.NN)
            self.MMs_previous = np.copy(self.MMs)
            self.NN, self.MMs = get_M_N(self.toas, self.residuals, self.Npsr, self.MMs, self.NN, self.invTN, self.CholSigma, self.Ndiag,
                                            self.Nwavelet, self.Nglitch, self.wavelet_prm, self.glitch_prm, self.glitch_pulsars, self.glitch_pulsars_previous, dif_flag)

        #update intrinsic likelihood terms when updating RN
        if self.wn_vary:
            self.logdet_previous = np.copy(self.logdet)
            self.logdet = 0
            for (l,m) in self.pta.get_rNr_logdet(self.params): #Only using this for logdet term because the rNr term removes the deterministic signal during generation
                self.logdet += m
        if self.rn_vary or self.wn_vary:
            resres_logdet = np.copy(self.logdet + resres_logdet_calc(self.Npsr, self.residuals, self.invTN, self.Ndiag, temp_logdetphi, self.CholSigma))
            self.resres_logdet_previous = np.copy(self.resres_logdet)
            self.resres_logdet = np.copy(resres_logdet)
        else:
            resres_logdet = np.copy(self.resres_logdet)
        #calls jitted function that compiles all likelihood contributions
        temp_like = likelihood_helper(self.sigmas, self.glitch_pulsars, resres_logdet, self.Npsr, self.Nwavelet, self.Nglitch, self.NN, self.MMs)
        #If fisher calculations, do not need to save params
        if self.no_step:
            self.save_values(accept_new_step=False, vary_white_noise = self.wn_vary, vary_red_noise = self.rn_vary, rj_jump = False)
        return temp_like

    #####
    #replaces saved values when deciding on MCMC step
    #####
    def save_values(self, accept_new_step=False, vary_white_noise = False, vary_red_noise = False, rj_jump = False):
        """
        Function to save Class attributes if current step is accepted in the MCMC.

        """
        #if the test point is being steped to, save it's parameter values to compare against in the future
        if accept_new_step:
            self.wavelet_saved = np.copy(self.wavelet_prm)
            self.glitch_saved = np.copy(self.glitch_prm)
        #if we stay at the original point, re-load all the values from before the step
        else:
            #if statments to check if the "params previous have actually been updated"
            if vary_red_noise or vary_white_noise:
                self.params = np.copy(self.params_previous)
                self.resres_logdet = np.copy(self.resres_logdet_previous)
                self.Ndiag = list(np.copy(self.Ndiag_previous))
            if vary_white_noise:
                self.Nvecs = np.copy(self.Nvecs_previous)
                self.TNTs = np.copy(self.TNTs_previous)
                self.logdet = np.copy(self.logdet_previous)
            self.NN = np.copy(self.NN_previous)
            self.MMs = np.copy(self.MMs_previous)

            self.sigmas = np.copy(self.sigmas_previous)

            self.CholSigma = list(np.copy(self.CholSigma_previous))
            self.invTN = list(np.copy(self.invTN_previous))

            self.glitch_pulsars = np.copy(self.glitch_pulsars_previous)
            if rj_jump:
                #Resave number of glitches and wavelets
                self.Nglitch = self.Nglitch_previous
                self.Nwavelet = self.Nwavelet_previous


#####
#generates the MM and NN matrixies from filter functions
#####
@njit(parallel=False,fastmath=True)
def get_M_N(toas, residuals, Npsr, MMs, NN, invTN, CholSigma, Ndiag_array, Nwavelet, Nglitch, wavelet_prm, glitch_prm, glitch_pulsars, glitch_pulsars_previous, dif_flag):
    """
    numba jitted Function to calculate inner products between pulsar residuals and filter functions, as well as between filter functions.

    :param toas:
        Array that contains lists of TOAs for each pulsar in pta.
    :param residuals:
        Array that contains lists of residuals for each pulsar in pta.
    :param Npsr:
        Number of pulsars in the pta.
    :param MMs:
        Likelihood Class attribute that contains the 3D matrix (Npsr, Nwavelet+Nglitch, Nwavelet + Nglitch) of inner products between all pairs of filter functions.
    :param NN:
        Likelihood Class attribute that contains the 2D matrix (N_psr, Nwavelet+Nglitch) of inner products between pulsar residuals and filter functions.
    :param CholSigma:
        Likelihood Class attribute that contains elements of the noise covariance matrix used in noise weighted inner products.
    :param Ndiag_array:
        Likelihood Class attribute that contains the diagonalized N matrix in the covariance matrix used in noise weighted inner products.
    :param Nwavelet:
        Number of wavelets in the pta model at the current MCMC step.
    :param Nglitch:
        Number of noise transient (glitches) in the pta model at the current MCMC step.
    :param wavelet_prm:
        GW signal wavelet parameter array for all GW signal wavelets in pta model at the current MCMC step.
    :param glitch_prm:
        Noise transient wavelet parameter array for all noise transient wavelets in pta model at the current MCMC step.
    :param glitch_pulsars:
        List on indexes referring to pulsars which currently have a noise transient wavelet present in the pta model.
    :param glitch_pulsars_previous:
        List of indexes referring to pulsars which had a noise transient wavelet in them previously. We recalculate these in the event of reverting to
        the previous MCMC step.
    :param dif_flag:
        List that checks if GW signal wavelet and/or noise transient wavelet shape parameters changed in current MCMC step.
    """
    for ii in range(Npsr):
        invCholSigmaTNfilter = np.zeros((Nwavelet + Nglitch, 2, len(np.asarray(CholSigma[ii])[:, 0])))
        Filt_cos = np.zeros((Nwavelet + Nglitch, len(toas[ii])))
        Filt_sin = np.zeros((Nwavelet + Nglitch, len(toas[ii])))

        for s in range(Nwavelet):
            Filt_cos[s] = np.exp(-1 * ((toas[ii] - wavelet_prm[(s, 7)]) / wavelet_prm[(s, 6)]) ** 2) * np.cos(2 * np.pi * wavelet_prm[(s, 0)] * (toas[ii] - wavelet_prm[(s, 7)]))
            Filt_sin[s] = np.exp(-1 * ((toas[ii] - wavelet_prm[(s, 7)]) / wavelet_prm[(s, 6)]) ** 2) * np.sin(2 * np.pi * wavelet_prm[(s, 0)] * (toas[ii] - wavelet_prm[(s, 7)]))
            invCholSigmaTNfilter[s, 0] = solve_triangular(np.asarray(CholSigma[ii]), np.asarray(invTN[ii]) @ Filt_cos[s], lower_a = True, trans_a = False, overwrite_b = False)
            invCholSigmaTNfilter[s, 1] = solve_triangular(np.asarray(CholSigma[ii]), np.asarray(invTN[ii]) @ Filt_sin[s], lower_a = True, trans_a = False, overwrite_b = False)

        for j in range(Nglitch):
            if (ii - 0.5 <= glitch_prm[j, 3] <= ii + 0.5):
                Filt_cos[j + Nwavelet] = np.exp(-1 * ((toas[ii] - glitch_prm[(j, 4)]) / glitch_prm[(j, 5)]) ** 2) * np.cos(2 * np.pi * glitch_prm[(j, 0)] * (toas[ii] - glitch_prm[(j, 4)]))
                Filt_sin[j + Nwavelet] = np.exp(-1 * ((toas[ii] - glitch_prm[(j, 4)]) / glitch_prm[(j, 5)]) ** 2) * np.sin(2 * np.pi * glitch_prm[(j, 0)] * (toas[ii] - glitch_prm[(j, 4)]))
                invCholSigmaTNfilter[j + Nwavelet, 0] = solve_triangular(np.asarray(CholSigma[ii]), np.asarray(invTN[ii]) @ Filt_cos[j + Nwavelet], lower_a = True, trans_a = False, overwrite_b = False)
                invCholSigmaTNfilter[j + Nwavelet, 1] = solve_triangular(np.asarray(CholSigma[ii]), np.asarray(invTN[ii]) @ Filt_sin[j + Nwavelet], lower_a = True, trans_a = False, overwrite_b = False)

        invCholSigmaTNres = solve_triangular(np.asarray(CholSigma[ii]), np.asarray(invTN[ii]) @ residuals[ii], lower_a = True, trans_a = False, overwrite_b = False)

        #update the full N and M when we are looking at a pulsar that contains some glitches (due to cross terms)
        if ii in glitch_pulsars or ii in glitch_pulsars_previous:
            #populate MM,NN with wavelets and glitches (including cross terms)
            for k in range(Nwavelet + Nglitch):
                if dif_flag[k] == 1: #check which terms have actualy changed before changing M and N
                    NN[ii, 0+2*k] = residuals[ii]*Ndiag_array[ii]@Filt_cos[k] - invCholSigmaTNres.T@invCholSigmaTNfilter[k,0] #manualy calculate dot product of aNb - aNTSigmaTNb
                    NN[ii, 1+2*k] = residuals[ii]*Ndiag_array[ii]@Filt_sin[k] - invCholSigmaTNres.T@invCholSigmaTNfilter[k,1]
                for l in range(Nwavelet + Nglitch):
                    if dif_flag[k] == 1 or dif_flag[l] == 1:
                        MMs[ii, 0+2*k, 0+2*l] = Filt_cos[k]*Ndiag_array[ii]@Filt_cos[l] - invCholSigmaTNfilter[k,0].T@invCholSigmaTNfilter[l,0]
                        MMs[ii, 1+2*k, 0+2*l] = Filt_sin[k]*Ndiag_array[ii]@Filt_cos[l] - invCholSigmaTNfilter[k,1].T@invCholSigmaTNfilter[l,0]
                        MMs[ii, 0+2*k, 1+2*l] = Filt_cos[k]*Ndiag_array[ii]@Filt_sin[l] - invCholSigmaTNfilter[k,0].T@invCholSigmaTNfilter[l,1]
                        MMs[ii, 1+2*k, 1+2*l] = Filt_sin[k]*Ndiag_array[ii]@Filt_sin[l] - invCholSigmaTNfilter[k,1].T@invCholSigmaTNfilter[l,1]
        #only update wavelets if there is now glitch in this pulsar
        else:
            #populate just wavelet parts of MM,NN
            for k in range(Nwavelet):
                if dif_flag[k] == 1: #check which terms have actually changed before changing M and N
                    NN[ii, 0+2*k] = residuals[ii]*Ndiag_array[ii]@Filt_cos[k] - invCholSigmaTNres.T@invCholSigmaTNfilter[k,0]
                    NN[ii, 1+2*k] = residuals[ii]*Ndiag_array[ii]@Filt_sin[k] - invCholSigmaTNres.T@invCholSigmaTNfilter[k,1]
                for l in range(Nwavelet):
                    if dif_flag[k] == 1 or dif_flag[l] == 1:
                        MMs[ii, 0+2*k, 0+2*l] = Filt_cos[k]*Ndiag_array[ii]@Filt_cos[l] - invCholSigmaTNfilter[k,0].T@invCholSigmaTNfilter[l,0]
                        MMs[ii, 1+2*k, 0+2*l] = Filt_sin[k]*Ndiag_array[ii]@Filt_cos[l] - invCholSigmaTNfilter[k,1].T@invCholSigmaTNfilter[l,0]
                        MMs[ii, 0+2*k, 1+2*l] = Filt_cos[k]*Ndiag_array[ii]@Filt_sin[l] - invCholSigmaTNfilter[k,0].T@invCholSigmaTNfilter[l,1]
                        MMs[ii, 1+2*k, 1+2*l] = Filt_sin[k]*Ndiag_array[ii]@Filt_sin[l] - invCholSigmaTNfilter[k,1].T@invCholSigmaTNfilter[l,1]
    return NN, MMs


#####
#needed to properly calculated logdet of the covariance matrix
#####
@njit(parallel=True,fastmath=True)
def logdet_Sigma_helper(chol_Sigma):
    """Numba jitted Function to calculate log10(2*pi*C) in likelihood"""
    #get logdet sigma from cholesky
    res = 0.
    for itrj in prange(0,chol_Sigma.shape[0]):
        res += np.log(chol_Sigma[itrj,itrj])
    return 2*res

#####
#Function to generate wavelets w/ uniform priors
#####
@njit(parallel=False,fastmath=True)
def get_parameters(x0, glitch_prm, wavelet_prm, glitch_indx, wavelet_indx, Nglitch, Nwavelet):
    """
    Numba jitted Function to store parameter values for all GW signal wavelets and noise transient wavelets in PTA model.

    :param x0:
        Current sample array in MCMC step.
    :param glitch_prm:
        Likelihood Class attribute containing noise transient wavelet parameters to be updated.
    :param wavelet_prm:
        Likelihood Class attribute containing GW signal wavelet parameters to be updated.
    :param glitch_indx:
        Likelihood Class attribute containing indexes of noise transient wavelet parameters in PTA.params().
    :param wavelet_indx:
        Likelihood Class attribute containing indexes of GW signal wavelet parameters in PTA.params().
    :param Nglitch:
        Number of noise transient wavelets in PTA model at current MCMC step.
    :param Nwavelet:
        Number of GW signal wavelets in PTA model at current MCMC step.

    :return wavelet_prm, glitch_prm:
        Returns 2D array of GW signal wavelet parameters and noise transient wavelet parameters, respectively.
    """
    #glitch models
    #start by re-setting parameters to zeros
    glitch_prm = np.copy(glitch_prm)*0
    wavelet_prm = np.copy(wavelet_prm)*0

    # self.wavelet_prm in order: GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, f0_w, tau_w, t0_w
    # self.glitch_prm in order: f0, log10_h, phase0, psr_idx, t0, tau
    for i in range(Nglitch):
        glitch_prm[i,0] = 10**(x0[int(glitch_indx[i,0])])
        glitch_prm[i,1] = 10**(x0[int(glitch_indx[i,1])])
        glitch_prm[i,2] = x0[int(glitch_indx[i,2])]
        glitch_prm[i,3] = x0[int(glitch_indx[i,3])]
        glitch_prm[i,4] = (365.25*24*3600)*x0[int(glitch_indx[i,4])]
        glitch_prm[i,5] = (365.25*24*3600)*x0[int(glitch_indx[i,5])]
    #wavelet models
    for j in range(Nwavelet):
        wavelet_prm[j,0] = 10**(x0[int(wavelet_indx[j,0])])
        wavelet_prm[j,1] = np.arccos(x0[int(wavelet_indx[j,1])])
        wavelet_prm[j,2] = x0[int(wavelet_indx[j,2])]
        wavelet_prm[j,3] = x0[int(wavelet_indx[j,3])]
        wavelet_prm[j,4] = x0[int(wavelet_indx[j,6])]
        wavelet_prm[j,5] = x0[int(wavelet_indx[j,7])]
        wavelet_prm[j,6] = (365.25*24*3600)*x0[int(wavelet_indx[j,9])]
        wavelet_prm[j,7] = (365.25*24*3600)*x0[int(wavelet_indx[j,8])]
        wavelet_prm[j,8] = 10**(x0[int(wavelet_indx[j,4])])
        wavelet_prm[j,9] = 10**(x0[int(wavelet_indx[j,5])])

    return glitch_prm, wavelet_prm

#####
#updating non-signal likelihood terms as we go
#####
    #@njit(parallel=True,fastmath=True) #was left unjitted
def resres_logdet_calc(Npsr, residuals, invTN, Ndiag, temp_logdetphi, chol_Sigma):
    '''
    Function to calculate terms in (res|res) + log(2*pi*C) in likelihood.
    '''
    rNr_loc = np.zeros(Npsr)
    logdet_array = np.zeros(Npsr)
    for i in range(Npsr):
        aNb = residuals[i] * Ndiag[i] @ residuals[i]

        SigmaTNaProd = solve_triangular(chol_Sigma[i], invTN[i] @ residuals[i], lower_a = True, trans_a = False, overwrite_b = False)
        SigmaTNbProd = solve_triangular(chol_Sigma[i], invTN[i] @ residuals[i], lower_a = True, trans_a = False, overwrite_b = False)
        dotSigmaTNr = SigmaTNaProd.T @ SigmaTNbProd
        rNr_loc[i] = aNb - dotSigmaTNr

        logdet_Sigma_loc = logdet_Sigma_helper(chol_Sigma[i])
        logdet_array[i] = temp_logdetphi[i] + logdet_Sigma_loc
    return np.sum(rNr_loc) + np.sum(logdet_array)

######
#calculates amplitudes for Signals
######
@njit(fastmath=True,parallel=False)
def get_sigmas_helper(pos, sigmas, glitch_pulsars, Npsr, Nwavelet, Nglitch, wavelet_prm, glitch_prm):
    '''
    Function to calculate coefficients for both GW signal wavelets and noise transient wavelets.

    '''
    #coefficients for wavelets and glitches
    sigma = np.copy(sigmas)*0

    #terms used in antenna pattern calculation
    m = np.zeros((Nwavelet, 3))
    n = np.zeros((Nwavelet, 3))
    omhat = np.zeros((Nwavelet, 3))
    for j in range(Nwavelet):
        sin_gwtheta = np.sin(wavelet_prm[j,1])
        cos_gwtheta = np.cos(wavelet_prm[j,1])
        sin_gwphi = np.sin(wavelet_prm[j,2])
        cos_gwphi = np.cos(wavelet_prm[j,2])

        m[j] = np.array([sin_gwphi, -cos_gwphi, 0.0])
        n[j] = np.array([-cos_gwtheta * cos_gwphi, -cos_gwtheta * sin_gwphi, sin_gwtheta])
        omhat[j] = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -cos_gwtheta])

    #4/30/24 May need to address edge case when 1-cosMu = 1 (divizion by zero error)
    for i in range(Npsr):
        for g in range(Nwavelet):
            m_pos = 0.
            n_pos = 0.
            cosMu = 0.
            for j in range(0,3):
                m_pos += m[g,j]*pos[i,j]
                n_pos += n[g,j]*pos[i,j]
                cosMu -= omhat[g,j]*pos[i,j]


            #Calculating the antenna response for the + and x GW modes. There is
            #a different response for each wavelet, so we compute a new antenna pattern for each.
            F_p = 0.5 * (m_pos ** 2 - n_pos ** 2) / (1 - cosMu)
            F_c = (m_pos * n_pos) / (1 - cosMu)

            #Calculating the angles once used to calculate coefficients for wavelet signal (see PDF for derivation)
            cos_0=np.cos(2*wavelet_prm[g,3]) #shouldn't depend on wavelet if they all are for the same GW source?
            cos_p=np.cos(wavelet_prm[g,4])
            cos_c=np.cos(wavelet_prm[g,5])
            sin_0=np.sin(2*wavelet_prm[g,3]) #shouldn't depend on wavelet if they all are for the same GW source?
            sin_p=np.sin(wavelet_prm[g,4])
            sin_c=np.sin(wavelet_prm[g,5])

            #Calculating wavelet signal coefficients
            sigma[i,g,0] = -F_p*wavelet_prm[g,8]*cos_p*cos_0 + F_p*wavelet_prm[g,9]*cos_c*sin_0 - F_c*wavelet_prm[g,8]*cos_p*sin_0 - F_c*wavelet_prm[g,9]*cos_c*cos_0
            sigma[i,g,1] = F_p*wavelet_prm[g,8]*sin_p*cos_0 - F_p*wavelet_prm[g,9]*sin_c*sin_0 + F_c*wavelet_prm[g,8]*sin_p*sin_0 + F_c*wavelet_prm[g,9]*sin_c*cos_0

        for k in range(Nglitch): #re-ordered so we only fill in the relevent glitch, to stop crossover with backfilled N,M
            if glitch_pulsars[k] == i:
                sigma[i,Nwavelet + k,0] = glitch_prm[k,1]*np.cos(glitch_prm[k,2])
                sigma[i,Nwavelet + k,1] = -glitch_prm[k,1]*np.sin(glitch_prm[k,2])
    return sigma

#####
#Calculate wavelet and glitch contributions to the likelihood
#####
@njit(fastmath=True,parallel=False)
def likelihood_helper(sigma, glitch_pulsars, resres_logdet, Npsr, Nwavelet, Nglitch, NN, MMs):
    '''
    Numba jitted Function to use likelihood class attributes to calculate likelihood for a given MCMC sample.

    '''
    #start calculating the LogLikelihood
    LogL = 0
    LogL += -1/2*resres_logdet
    #loop over the pulsars to add in the noise transients
    for i in range(Npsr):
        if i in glitch_pulsars:
            #step over wavelets and glitches
            for k in range(Nwavelet + Nglitch):
                LogL += (sigma[i,k,0]*NN[i, 0+2*k] + sigma[i,k,1]*NN[i, 1+2*k]) #adding in NN term in sum
                for l in range(Nwavelet + Nglitch):
                    LogL += -1/2*(sigma[i,k,0]*(sigma[i,l,0]*MMs[i, 0+2*k, 0+2*l] + sigma[i,l,1]*MMs[i, 0+2*k, 1+2*l]) + sigma[i,k,1]*(sigma[i,l,0]*MMs[i, 1+2*k, 0+2*l] + sigma[i,l,1]*MMs[i, 1+2*k, 1+2*l]))
        else:
            #just step over wavelets
            for k in range(Nwavelet):
                LogL += (sigma[i,k,0]*NN[i, 0+2*k] + sigma[i,k,1]*NN[i, 1+2*k]) #adding in NN term in sum
                for l in range(Nwavelet):
                    LogL += -1/2*(sigma[i,k,0]*(sigma[i,l,0]*MMs[i, 0+2*k, 0+2*l] + sigma[i,l,1]*MMs[i, 0+2*k, 1+2*l]) + sigma[i,k,1]*(sigma[i,l,0]*MMs[i, 1+2*k, 0+2*l] + sigma[i,l,1]*MMs[i, 1+2*k, 1+2*l]))
    return LogL


#QB class for sampling through projection parameters
@jitclass([('Npsr',nb.types.int64),('pos',nb.types.float64[:,::1]),('resres_logdet',nb.types.float64),('Nglitch',nb.types.int64),('Nwavelet',nb.types.int64),
            ('wavelet_prm',nb.types.float64[:,::1]),('glitch_prm',nb.types.float64[:,::1]),('MMs',nb.types.float64[:,:,::1]),('NN',nb.types.float64[:,::1]),('prior_recovery',nb.boolean),
            ('glitch_indx',nb.types.float64[:,::1]),('wavelet_indx',nb.types.float64[:,::1]),('glitch_pulsars',nb.types.float64[::1]), ('sigmas', nb.types.float64[:,:,::1])])#nb.types.ListType(nb.types.int64[::1])
class QuickBurst_info:
    '''QuickBurst info Class to store parameters used in projection parameter updates.'''
    def __init__(self, Npsr, pos, resres_logdet, Nglitch ,Nwavelet, wavelet_prm, glitch_prm, sigmas, MMs, NN, prior_recovery, glitch_indx, wavelet_indx, glitch_pulsars):
        #loading in parameters for the class to hold onto
        self.Npsr = Npsr
        self.pos = pos
        self.resres_logdet = resres_logdet
        #max number of glitches and signals that can be handeled
        self.Nglitch = Nglitch
        self.Nwavelet = Nwavelet
        #holds parameters parsed from current run
        self.wavelet_prm = wavelet_prm# in order GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, f0_w, tau_w, t0_w
        self.glitch_prm = glitch_prm# in order A, phi0, f0, tau, t0, glitch_idx

        self.wavelet_indx = wavelet_indx
        self.glitch_indx = glitch_indx

        self.sigmas = sigmas
        self.MMs = MMs
        self.NN = NN

        self.glitch_pulsars = glitch_pulsars

        self.prior_recovery = prior_recovery

    def load_parameters(self, resres_logdet, Nglitch ,Nwavelet, wavelet_prm, glitch_prm, MMs, NN, glitch_pulsars):
        '''Function to store parameters between projection parameter updates.'''

        self.resres_logdet = resres_logdet
        #max number of glitches and signals that can be handeled
        self.Nglitch = Nglitch
        self.Nwavelet = Nwavelet
        #holds parameters parsed from current run
        self.wavelet_prm = wavelet_prm# in order GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, f0_w, tau_w, t0_w
        self.glitch_prm = glitch_prm# in order A, phi0, f0, tau, t0, glitch_idx

        self.MMs = MMs
        self.NN = NN


        self.glitch_pulsars = glitch_pulsars

    def get_lnlikelihood(self, x0):
        '''Function to calculate the likelihood for projection parameter updates.'''
        if self.prior_recovery:
            return 1
        self.glitch_prm, self.wavelet_prm = get_parameters(x0, self.glitch_prm, self.wavelet_prm, self.glitch_indx, self.wavelet_indx, self.Nglitch, self.Nwavelet)
        #fast calculation of the lnlikelihood
        self.sigmas = get_sigmas_helper(self.pos, self.sigmas, self.glitch_pulsars, self.Npsr, self.Nwavelet, self.Nglitch, self.wavelet_prm, self.glitch_prm)
        temp_like = likelihood_helper(self.sigmas, self.glitch_pulsars, self.resres_logdet, self.Npsr, self.Nwavelet, self.Nglitch, self.NN, self.MMs)
        return temp_like
