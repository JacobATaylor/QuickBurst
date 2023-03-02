"""C 2023 Jacob, Rand, and Bence fast Burst likelihood"""

import numpy as np
import numba as nb
from numba import njit,prange
from numba.experimental import jitclass
from numba.typed import List
import scipy.linalg as sl
from lapack_wrappers import solve_triangular

from enterprise import constants as const
from enterprise_extensions.frequentist import Fe_statistic as FeStat
from enterprise.signals import utils

import line_profiler

#########
#strucutre overview
#
#Class to hold info about pta, params, psrs arrays
#   function calc M and N matrixies to be used
#   function to calculate likelihoods
########

class FastBurst:
    #####
    #generate object with the res|res and logdet terms that only depend on non-signal based parameters pre-calculated
    #####
    def __init__(self,pta,psrs,params,Npsr,tref,Nglitch, Nwavelet):

        #model parameters that shouldn't change for a run
        self.pta = pta
        self.psrs = psrs
        self.Npsr = Npsr
        self.params = params
        #important matrixies
        self.Nvecs = List(self.pta.get_ndiag(self.params))
        self.TNTs = self.pta.get_TNT(self.params)
        self.Ts = self.pta.get_basis()

        #pulsar information
        self.toas = List([psr.toas - tref for psr in psrs])
        self.residuals = List([psr.residuals for psr in psrs])
        self.pos = np.zeros((self.Npsr,3))
        for i in range(self.Npsr):
            self.pos[i] = self.psrs[i].pos

        self.logdet = 0
        for (l,m) in self.pta.get_rNr_logdet(params): #Only using this for logdet term because the rNr term removes the deterministic signal durring generation
            self.logdet += m

        #generate arrays to store res|res and logdet(2*Pi*N) terms
        rNr_loc = np.zeros(self.Npsr)
        logdet_array = np.zeros(self.Npsr)
        pls_temp = self.pta.get_phiinv(self.params, logdet=True, method='partition')

        for i in range(self.Npsr):

            phiinv_loc,logdetphi_loc = pls_temp[i]
            TNT = self.TNTs[i]
            T = self.Ts[i]
            Sigma = self.TNTs[i]+phiinv_loc

            Ndiag = 1/self.Nvecs[i]
            chol_Sigma,lower = sl.cho_factor(Sigma.T,lower=True,overwrite_a=True,check_finite=False)
            invchol_Sigma_T_loc = solve_triangular(chol_Sigma,T.T,lower_a=True,trans_a=False,overwrite_b=False)

            invCholSigmaTN = invchol_Sigma_T_loc*Ndiag

            #first term in the dot product
            rNr_loc[i] = dot_product(self.residuals[i],self.residuals[i],invCholSigmaTN,Ndiag)

            logdet_Sigma_loc = logdet_Sigma_helper(chol_Sigma)
            #add the necessary component to logdet
            logdet_array[i] =  logdetphi_loc + logdet_Sigma_loc

        #Non-signal dependent terms
        self.resres_logdet = self.logdet + np.sum(rNr_loc) + np.sum(logdet_array)

        #max number of glitches and signals that can be handeled
        self.Nglitch = Nglitch
        self.Nwavelet = Nwavelet
        #holds parameters parsed from current run
        self.wavelet_prm = np.zeros((self.Nwavelet, 10))# in order GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, f0_w, tau_w, t0_w
        self.glitch_prm = np.zeros((self.Nglitch, 6))# in order A, phi0, f0, tau, t0, glitch_idx
        #holds previous updated shape params
        self.wavelet_shape_saved = np.zeros((self.Nwavelet, 3))# f0, t0, tau
        self.glitch_shape_saved = np.zeros((self.Nglitch, 3))# f0, t0, tau
        #max possible size of NN and MMs
        self.MMs = np.zeros((self.Npsr,2*self.Nwavelet + 2*self.Nglitch,2*self.Nwavelet + 2*self.Nglitch))
        self.NN = np.zeros((self.Npsr,2*self.Nwavelet + 2*self.Nglitch))

        self.Saved_Shape = np.zeros((3*self.Nwavelet + 3*self.Nglitch)) #space to story current shape parameters to decided it we need to update NN and MM


    #####
    #generates the MM and NN matrixies from filter functions
    #####
    #@profile
    def get_M_N(self, glitch_pulsars):

        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition') #use enterprise to calculate phiinvs

        #reset MM and NN to zeros when running this function
        self.MMs = np.zeros((self.Npsr,2*self.Nwavelet + 2*self.Nglitch,2*self.Nwavelet + 2*self.Nglitch))
        self.NN = np.zeros((self.Npsr,2*self.Nwavelet + 2*self.Nglitch))

        for ii in range(self.Npsr):

            TNT = self.TNTs[ii]
            T = self.Ts[ii]
            phiinv = phiinvs[ii]

            Ndiag = 1/self.Nvecs[ii]
            Sigma = TNT + phiinv
            #mutate inplace to avoid memory allocation overheads
            chol_Sigma,lower = sl.cho_factor(Sigma.T,lower=True,overwrite_a=True,check_finite=False)
            invchol_Sigma_T_loc = solve_triangular(chol_Sigma,T.T,lower_a=True,trans_a=False,overwrite_b=False)

            invCholSigmaTN = invchol_Sigma_T_loc*Ndiag

            #print(invCholSigmaTN[:,0].shape)

            invCholSigmaTNfilter = np.zeros((self.Nwavelet + self.Nglitch,2,len(invCholSigmaTN[:,0]))) #stores the cholesky terms for use in dot products

            #filter fuctions
            Filt_cos = np.zeros((self.Nwavelet + self.Nglitch,len(self.toas[ii])))
            Filt_sin = np.zeros((self.Nwavelet + self.Nglitch,len(self.toas[ii])))

            #first half of the NN and MM  will be wavelets
            for s in range(self.Nwavelet):
                Filt_cos[s] = np.exp(-1*((self.toas[ii] - self.wavelet_prm[s,7])/self.wavelet_prm[s,6])**2)*np.cos(2*np.pi*self.wavelet_prm[s,0]*(self.toas[ii] - self.wavelet_prm[s,7])) #see PDF for derivation
                Filt_sin[s] = np.exp(-1*((self.toas[ii] - self.wavelet_prm[s,7])/self.wavelet_prm[s,6])**2)*np.sin(2*np.pi*self.wavelet_prm[s,0]*(self.toas[ii] - self.wavelet_prm[s,7]))
                invCholSigmaTNfilter[s,0] = invCholSigmaTN@Filt_cos[s]
                invCholSigmaTNfilter[s,1] = invCholSigmaTN@Filt_sin[s]
            #second half are glitches
            for j in range(self.Nglitch):
                if (ii-0.5 <= self.glitch_prm[j,3] <= ii+0.5): #only populate filter functions for pulsar with glitcvh in it
                    Filt_cos[j + self.Nwavelet] = np.exp(-1*((self.toas[ii] - self.glitch_prm[j,4])/self.glitch_prm[j,5])**2)*np.cos(2*np.pi*self.glitch_prm[j,0]*(self.toas[ii] - self.glitch_prm[j,4])) #see PDF for derivation
                    Filt_sin[j + self.Nwavelet] = np.exp(-1*((self.toas[ii] - self.glitch_prm[j,4])/self.glitch_prm[j,5])**2)*np.sin(2*np.pi*self.glitch_prm[j,0]*(self.toas[ii] - self.glitch_prm[j,4]))
                    invCholSigmaTNfilter[j + self.Nwavelet,0] = invCholSigmaTN@Filt_cos[j + self.Nwavelet]
                    invCholSigmaTNfilter[j + self.Nwavelet,1] = invCholSigmaTN@Filt_sin[j + self.Nwavelet]

            invCholSigmaTNres = invCholSigmaTN@self.residuals[ii]

            if ii in glitch_pulsars:
                #populate MM,NN with wavelets and glitches (including cross terms)
                for k in range(self.Nwavelet + self.Nglitch):
                    self.NN[ii, 0+2*k] = self.residuals[ii]*Ndiag@Filt_cos[k] - invCholSigmaTNres.T@invCholSigmaTNfilter[k,0]
                    self.NN[ii, 1+2*k] = self.residuals[ii]*Ndiag@Filt_sin[k] - invCholSigmaTNres.T@invCholSigmaTNfilter[k,1]
                    for l in range(self.Nwavelet + self.Nglitch):
                        self.MMs[ii, 0+2*k, 0+2*l] = Filt_cos[k]*Ndiag@Filt_cos[l] - invCholSigmaTNfilter[k,0].T@invCholSigmaTNfilter[l,0]
                        self.MMs[ii, 1+2*k, 0+2*l] = Filt_sin[k]*Ndiag@Filt_cos[l] - invCholSigmaTNfilter[k,1].T@invCholSigmaTNfilter[l,0]
                        self.MMs[ii, 0+2*k, 1+2*l] = Filt_cos[k]*Ndiag@Filt_sin[l] - invCholSigmaTNfilter[k,0].T@invCholSigmaTNfilter[l,1]
                        self.MMs[ii, 1+2*k, 1+2*l] = Filt_sin[k]*Ndiag@Filt_sin[l] - invCholSigmaTNfilter[k,1].T@invCholSigmaTNfilter[l,1]
            else:
                #populate just wavelet parts of MM,NN
                for k in range(self.Nwavelet):
                    self.NN[ii, 0+2*k] = self.residuals[ii]*Ndiag@Filt_cos[k] - invCholSigmaTNres.T@invCholSigmaTNfilter[k,0]
                    self.NN[ii, 1+2*k] = self.residuals[ii]*Ndiag@Filt_sin[k] - invCholSigmaTNres.T@invCholSigmaTNfilter[k,1]
                    for l in range(self.Nwavelet):
                        self.MMs[ii, 0+2*k, 0+2*l] = Filt_cos[k]*Ndiag@Filt_cos[l] - invCholSigmaTNfilter[k,0].T@invCholSigmaTNfilter[l,0]
                        self.MMs[ii, 1+2*k, 0+2*l] = Filt_sin[k]*Ndiag@Filt_cos[l] - invCholSigmaTNfilter[k,1].T@invCholSigmaTNfilter[l,0]
                        self.MMs[ii, 0+2*k, 1+2*l] = Filt_cos[k]*Ndiag@Filt_sin[l] - invCholSigmaTNfilter[k,0].T@invCholSigmaTNfilter[l,1]
                        self.MMs[ii, 1+2*k, 1+2*l] = Filt_sin[k]*Ndiag@Filt_sin[l] - invCholSigmaTNfilter[k,1].T@invCholSigmaTNfilter[l,1]
            # if ii in glitch_pulsars:
            #     #populate MM,NN with wavelets and glitches (including cross terms)
            #     for k in range(self.Nwavelet + self.Nglitch):
            #         self.NN[ii, 0+2*k] = dot_product(self.residuals[ii],Filt_cos[k],invCholSigmaTN,Ndiag)#,phiinv,T,TNT,Sigma,self.Nvecs[ii])
            #         self.NN[ii, 1+2*k] = dot_product(self.residuals[ii],Filt_sin[k],invCholSigmaTN,Ndiag)
            #         for l in range(self.Nwavelet + self.Nglitch):
            #             self.MMs[ii, 0+2*k, 0+2*l] = dot_product(Filt_cos[k], Filt_cos[l],invCholSigmaTN,Ndiag)
            #             self.MMs[ii, 1+2*k, 0+2*l] = dot_product(Filt_sin[k], Filt_cos[l],invCholSigmaTN,Ndiag)
            #             self.MMs[ii, 0+2*k, 1+2*l] = dot_product(Filt_cos[k], Filt_sin[l],invCholSigmaTN,Ndiag)
            #             self.MMs[ii, 1+2*k, 1+2*l] = dot_product(Filt_sin[k], Filt_sin[l],invCholSigmaTN,Ndiag)
            # else:
            #     #populate just wavelet parts of MM,NN
            #     for k in range(self.Nwavelet):
            #         self.NN[ii, 0+2*k] = dot_product(self.residuals[ii],Filt_cos[k],invCholSigmaTN,Ndiag)
            #         self.NN[ii, 1+2*k] = dot_product(self.residuals[ii],Filt_sin[k],invCholSigmaTN,Ndiag)
            #         for l in range(self.Nwavelet):
            #             self.MMs[ii, 0+2*k, 0+2*l] = dot_product(Filt_cos[k], Filt_cos[l],invCholSigmaTN,Ndiag)
            #             self.MMs[ii, 1+2*k, 0+2*l] = dot_product(Filt_sin[k], Filt_cos[l],invCholSigmaTN,Ndiag)
            #             self.MMs[ii, 0+2*k, 1+2*l] = dot_product(Filt_cos[k], Filt_sin[l],invCholSigmaTN,Ndiag)
            #             self.MMs[ii, 1+2*k, 1+2*l] = dot_product(Filt_sin[k], Filt_sin[l],invCholSigmaTN,Ndiag)

    #####
    #Function to generate wavelets w/ uniform priors
    #####
    def get_parameters(self, x0):
        '''
        we should be able to parse which positional values are which in the x0 by looking at the order of pta.params
        ahh yeah, that's smart.
        '''
        key_list = list(self.params)#makes a list of the keys in the params Dictionary
        #glitch models
        for i in range(self.Nglitch):
            self.glitch_prm[i,0] = 10**(x0[key_list.index('Glitch_'+str(i)+'_log10_f0')])
            self.glitch_prm[i,1] = 10**(x0[key_list.index('Glitch_'+str(i)+'_log10_h')])
            self.glitch_prm[i,2] = x0[key_list.index('Glitch_'+str(i)+'_phase0')]
            self.glitch_prm[i,3] = x0[key_list.index('Glitch_'+str(i)+'_psr_idx')]
            self.glitch_prm[i,4] = (365.25*24*3600)*x0[key_list.index('Glitch_'+str(i)+'_t0')]
            self.glitch_prm[i,5] = (365.25*24*3600)*x0[key_list.index('Glitch_'+str(i)+'_tau')]
        #wavelet models
        for j in range(self.Nwavelet):
            self.wavelet_prm[j,0] = 10**(x0[key_list.index('wavelet_'+str(j)+'_log10_f0')])
            self.wavelet_prm[j,1] = np.arccos(x0[key_list.index('wavelet_'+str(j)+'_cos_gwtheta')])
            self.wavelet_prm[j,2] = x0[key_list.index('wavelet_'+str(j)+'_gwphi')]
            self.wavelet_prm[j,3] = x0[key_list.index('wavelet_'+str(j)+'_gw_psi')]
            self.wavelet_prm[j,4] = x0[key_list.index('wavelet_'+str(j)+'_phase0')]
            self.wavelet_prm[j,5] = x0[key_list.index('wavelet_'+str(j)+'_phase0_cross')]
            self.wavelet_prm[j,6] = (365.25*24*3600)*x0[key_list.index('wavelet_'+str(j)+'_tau')]
            self.wavelet_prm[j,7] = (365.25*24*3600)*x0[key_list.index('wavelet_'+str(j)+'_t0')]
            self.wavelet_prm[j,8] = 10**(x0[key_list.index('wavelet_'+str(j)+'_log10_h')])
            self.wavelet_prm[j,9] = 10**(x0[key_list.index('wavelet_'+str(j)+'_log10_h_cross')])


    ######
    #calculates amplitudes for Signals
    ######
    def get_sigmas(self, glitch_pulsars):
        return get_sigmas_helper(self.pos, glitch_pulsars, self.Npsr, self.Nwavelet, self.Nglitch, self.wavelet_prm, self.glitch_prm)

    #####
    #calculates lnliklihood for a set of signal parameters
    #####
    #@profile
    def get_lnlikelihood(self, x0):#A, phi0, f0, tau, t0, glitch_idx, GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, f0_w, tau_w, t0_w):

        '''
        ######Understanding the components of logdet######
        logdet = logdet(2*pi*C) = log(det(phi)*det(sigma)*det(N))
        N = white noise covariance matrix
        NN is matrix of size (Npsr, 2), where 2 is the # of filter functions used to model transient wavelet. sigma_k[i] are coefficients on filter functions.
        phi = prior matrix
        sigma = inverse(phi) - transpose(T)*inverse(N)*T (first term -> phiinv, second term -> TNT)
        T = [M F]
        M = Design matrix
        F = Fourier matrix (matrix of fourier coefficients and sin/cos terms)
        '''

        '''Have you checked that this is rounding correctly (and matches the parameters for a glitch pulsar)?
        yes, I prited out these values and they were rounded correctly.
        '''
        self.get_parameters(x0)
        dif_flag = 0
        for i in range(self.Nglitch):
            if self.glitch_shape_saved[i,0] != self.glitch_prm[i,0] or self.glitch_shape_saved[i,1] != self.glitch_prm[i,4] or self.glitch_shape_saved[i,2] != self.glitch_prm[i,5]:
                #if current shape params don't match, set them to match and set off flag to update MM and NN
                dif_flag += 1
                self.glitch_shape_saved[i,0] = self.glitch_prm[i,0]
                self.glitch_shape_saved[i,1] = self.glitch_prm[i,4]
                self.glitch_shape_saved[i,2] = self.glitch_prm[i,5]
            if self.wavelet_shape_saved[i,0] != self.wavelet_prm[i,0] or self.wavelet_shape_saved[i,1] != self.wavelet_prm[i,7] or self.wavelet_shape_saved[i,2] != self.wavelet_prm[i,6]:
                dif_flag += 1
                self.wavelet_shape_saved[i,0] = self.wavelet_prm[i,0]
                self.wavelet_shape_saved[i,1] = self.wavelet_prm[i,7]
                self.wavelet_shape_saved[i,2] = self.wavelet_prm[i,6]

        #parse the pulsar indexes from parameters
        glitch_pulsars = np.zeros((len(self.glitch_prm[:,3])))
        for el in range(len(self.glitch_prm[:,3])):
            glitch_pulsars[el] = round(self.glitch_prm[el,3])

        sigma = self.get_sigmas(glitch_pulsars) #calculate the amplitudes of noise transients
        if dif_flag > 0:
            self.get_M_N(glitch_pulsars) #find the NN and MM matrixies from filter functions

        return liklihood_helper(sigma, glitch_pulsars, self.resres_logdet, self.Npsr, self.Nwavelet, self.Nglitch, self.NN, self.MMs)

#####
#needed to properly calculated logdet of the covariance matrix
#####
@njit(parallel=True,fastmath=True)
def logdet_Sigma_helper(chol_Sigma):
    """get logdet sigma from cholesky"""
    res = 0.
    for itrj in prange(0,chol_Sigma.shape[0]):
        res += np.log(chol_Sigma[itrj,itrj])
    return 2*res

######
#calculates amplitudes for Signals
######
@njit(fastmath=True,parallel=False)
def get_sigmas_helper(pos, glitch_pulsars, Npsr, Nwavelet, Nglitch, wavelet_prm, glitch_prm):
    #coefficients for wavelets and glitches
    sigma = np.zeros((Npsr, Nwavelet + Nglitch, 2))

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
            cos_0=np.cos(2*wavelet_prm[g,3])
            cos_p=np.cos(wavelet_prm[g,4])
            cos_c=np.cos(wavelet_prm[g,5])
            sin_0=np.sin(2*wavelet_prm[g,3])
            sin_p=np.sin(wavelet_prm[g,4])
            sin_c=np.sin(wavelet_prm[g,5])

            #Calculating wavelet signal coefficients
            sigma[i,g,0] = -F_p*wavelet_prm[g,8]*cos_p*cos_0 + F_p*wavelet_prm[g,9]*cos_c*sin_0 - F_c*wavelet_prm[g,8]*cos_p*sin_0 - F_c*wavelet_prm[g,9]*cos_c*cos_0
            sigma[i,g,1] = F_p*wavelet_prm[g,8]*sin_p*cos_0 - F_p*wavelet_prm[g,9]*sin_c*sin_0 + F_c*wavelet_prm[g,8]*sin_p*sin_0 + F_c*wavelet_prm[g,9]*sin_c*cos_0
        if i in glitch_pulsars:
            for k in range(Nglitch):
                sigma[i,Nwavelet + k,0] = glitch_prm[k,1]*np.cos(glitch_prm[k,2])
                sigma[i,Nwavelet + k,1] = -glitch_prm[k,1]*np.sin(glitch_prm[k,2])
    return sigma

#####
#combine all likelihood terms
#####
@njit(fastmath=True,parallel=False)
def liklihood_helper(sigma, glitch_pulsars, resres_logdet, Npsr, Nwavelet, Nglitch, NN, MMs):
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


#####
#function for taking the dot product of two tensors a and b
#See page 132 in https://arxiv.org/pdf/2105.13270.pdf, where (x|y) = x^T*C_inv*y
#####
@njit(fastmath=True,parallel=False)
def dot_product(a, b, invCholSigmaTN, Ndiag):

    '''@ decorator acts as a dot product operator'''
    aNb = a*Ndiag@b

    SigmaTNaProd = invCholSigmaTN@a
    SigmaTNbProd = invCholSigmaTN@b
    dotSigmaTNr = SigmaTNaProd.T@SigmaTNbProd

    dot_prod = aNb - dotSigmaTNr

    return dot_prod
