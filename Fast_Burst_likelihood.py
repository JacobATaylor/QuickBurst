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
#from memory_profiler import profile

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

            #first term in the dot product
            rNr_loc[i] = dot_product(self.residuals[i],self.residuals[i],phiinv_loc,T,TNT,Sigma,self.Nvecs[i])

            #mutate inplace to avoid memory allocation overheads
            chol_Sigma,lower = sl.cho_factor(Sigma.T,lower=True,overwrite_a=True,check_finite=False)
            logdet_Sigma_loc = logdet_Sigma_helper(chol_Sigma)
            #add the necessary component to logdet
            logdet_array[i] =  logdetphi_loc + logdet_Sigma_loc

        #Non-signal dependent terms
        self.resres_logdet = self.logdet + np.sum(rNr_loc) + np.sum(logdet_array)

        #singal model terms to be populated as we go
        '''
        Why is this 2 multiplier here again? I know why it is in self.sigmas (to seperate out the coefficients), but not here.
        I think you told me, but I don't remember the explanation.
        '''
        # self.MMs = np.zeros((Npsr,2*Nglitch,2*Nglitch))
        # self.NN = np.zeros((Npsr,2*Nglitch))
        # self.sigma = np.zeros((Nglitch, 2))
        self.Nglitch = Nglitch
        self.Nwavelet = Nwavelet

        self.MMs = np.zeros((self.Npsr,2*self.Nwavelet + 2*self.Nglitch,2*self.Nwavelet + 2*self.Nglitch))
        self.NN = np.zeros((self.Npsr,2*self.Nwavelet + 2*self.Nglitch))

        #self.MMs_wave = np.zeros((self.Npsr,2*self.Nwavelet,2*self.Nwavelet))
        #self.NN_wave = np.zeros((self.Npsr,2*self.Nwavelet))

    #####
    #generates the MM and NN matrixies from filter functions
    #####
    def get_M_N(self, f0, tau, t0, glitch_idx, f0_w, tau_w, t0_w):
        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition') #use enterprise to calculate phiinvs

        for ii in range(self.Npsr):

            TNT = self.TNTs[ii]
            T = self.Ts[ii]
            phiinv = phiinvs[ii]
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            #filter fuctions
            Filt_cos = np.zeros((self.Nwavelet + self.Nglitch,len(self.toas[ii])))
            Filt_sin = np.zeros((self.Nwavelet + self.Nglitch,len(self.toas[ii])))

            #Filt_cos_wave = np.zeros((self.Nwavelet,len(self.toas[ii])))
            #Filt_sin_wave = np.zeros((self.Nwavelet,len(self.toas[ii])))

            #first half of the NN and MM  will be wavelets
            for s in range(self.Nwavelet):
                Filt_cos[s] = np.exp(-1*((self.toas[ii] - t0_w[s])/tau_w[s])**2)*np.cos(2*np.pi*f0_w[s]*(self.toas[ii] - t0_w[s])) #see PDF for derivation
                Filt_sin[s] = np.exp(-1*((self.toas[ii] - t0_w[s])/tau_w[s])**2)*np.sin(2*np.pi*f0_w[s]*(self.toas[ii] - t0_w[s]))

            for j in range(self.Nglitch):
                #Single noise transient wavelet
                if (ii-0.5 <= glitch_idx[j] <= ii+0.5): #only populate filter functions for pulsar with glitcvh in it
                    Filt_cos[j + self.Nwavelet] = np.exp(-1*((self.toas[ii] - t0[j])/tau[j])**2)*np.cos(2*np.pi*f0[j]*(self.toas[ii] - t0[j])) #see PDF for derivation
                    Filt_sin[j + self.Nwavelet] = np.exp(-1*((self.toas[ii] - t0[j])/tau[j])**2)*np.sin(2*np.pi*f0[j]*(self.toas[ii] - t0[j]))

            for k in range(self.Nwavelet + self.Nglitch):
                self.NN[ii, 0+2*k] = dot_product(self.residuals[ii],Filt_cos[k],phiinv,T,TNT,Sigma,self.Nvecs[ii])
                self.NN[ii, 1+2*k] = dot_product(self.residuals[ii],Filt_sin[k],phiinv,T,TNT,Sigma,self.Nvecs[ii])
                for l in range(self.Nwavelet + self.Nglitch):
                    #populate MM,NN
                    self.MMs[ii, 0+2*k, 0+2*l] = dot_product(Filt_cos[k], Filt_cos[l],phiinv,T,TNT,Sigma,self.Nvecs[ii])
                    self.MMs[ii, 1+2*k, 0+2*l] = dot_product(Filt_sin[k], Filt_cos[l],phiinv,T,TNT,Sigma,self.Nvecs[ii])
                    self.MMs[ii, 0+2*k, 1+2*l] = dot_product(Filt_cos[k], Filt_sin[l],phiinv,T,TNT,Sigma,self.Nvecs[ii])
                    self.MMs[ii, 1+2*k, 1+2*l] = dot_product(Filt_sin[k], Filt_sin[l],phiinv,T,TNT,Sigma,self.Nvecs[ii])

    #Function to generate wavelets w/ uniform priors
    def get_wavelets(self, f0):
        #wavelet models


        self.wavelets = np.zeros((self.Nwavelet, 9))
        #xSort_wave_15y = [[],[],[],[],[],[],[],[],[],[]]

        xSort_wave_15y[0].append(10**(d0_15y['wavelet_'+str(j)+'_log10_f0']))
        xSort_wave_15y[1].append(np.arccos(d0_15y['wavelet_'+str(j)+'_cos_gwtheta']))
        xSort_wave_15y[2].append(d0_15y['wavelet_'+str(j)+'_gwphi'])
        xSort_wave_15y[3].append(d0_15y['wavelet_'+str(j)+'_gw_psi'])
        xSort_wave_15y[4].append(d0_15y['wavelet_'+str(j)+'_phase0'])
        xSort_wave_15y[5].append(d0_15y['wavelet_'+str(j)+'_phase0_cross'])
        xSort_wave_15y[6].append((365.25*24*3600)*d0_15y['wavelet_'+str(j)+'_tau'])
        xSort_wave_15y[7].append((365.25*24*3600)*d0_15y['wavelet_'+str(j)+'_t0'])
        xSort_wave_15y[8].append(10**(d0_15y['wavelet_'+str(j)+'_log10_h']))
        xSort_wave_15y[9].append(10**(d0_15y['wavelet_'+str(j)+'_log10_h_cross']))


    ######
    #calculates amplitudes for Signals
    ######
    def get_sigmas(self, A, phi0, GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, Fplus, Fcross, glitch_idx):
        #coefficients
        sigma = np.zeros((self.Npsr, self.Nwavelet + self.Nglitch, 2))

        #for d in glitch_idx
        #    if

        for i in range(len(self.psrs)):
            for g in range(self.Nwavelet):
                sigma[i,g,0] = -Fplus[i,g]*Ap[g]*np.cos(phi0p[g])*np.cos(2*pol[g]) + Fplus[i,g]*Ac[g]*np.cos(phi0c[g])*np.sin(2*pol[g]) - Fcross[i,g]*Ap[g]*np.cos(phi0p[g])*np.sin(2*pol[g]) - Fcross[i,g]*Ac[g]*np.cos(phi0c[g])*np.cos(2*pol[g])
                sigma[i,g,1] = Fplus[i,g]*Ap[g]*np.sin(phi0p[g])*np.cos(2*pol[g]) - Fplus[i,g]*Ac[g]*np.sin(phi0c[g])*np.sin(2*pol[g]) + Fcross[i,g]*Ap[g]*np.sin(phi0p[g])*np.sin(2*pol[g]) + Fcross[i,g]*Ac[g]*np.sin(phi0c[g])*np.cos(2*pol[g])
            for k in range(self.Nglitch):
                if (i-0.5 <= glitch_idx[k] <= i+0.5):
                    sigma[i,self.Nwavelet + k,0] = A[k]*np.cos(phi0[k])
                    sigma[i,self.Nwavelet + k,1] = -A[k]*np.sin(phi0[k])
        return sigma

    #####
    #calculates lnliklihood for a set of signal parameters
    #####
    def get_lnlikelihood(self, A, phi0, f0, tau, t0, glitch_idx, GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, f0_w, tau_w, t0_w):

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
        #GWtheta[0] = int(GWtheta[0])
        #GWphi[0] = int(GWphi[0])

        Fplus = np.zeros((self.Npsr,self.Nwavelet))
        Fcross = np.zeros((self.Npsr,self.Nwavelet))
        for i in range(len(self.psrs)):
            for g in range(self.Nwavelet):
                Fplus[i,g], Fcross[i,g], holding = utils.create_gw_antenna_pattern(self.pos[i], GWtheta[g], GWphi[g]) #Holding is for third varriable we don't use

        sigma = self.get_sigmas(A, phi0, GWtheta, GWphi, Ap, Ac, phi0p, phi0c, pol, Fplus, Fcross, glitch_idx) #calculate the amplitudes of noise transients
        self.get_M_N(f0,tau,t0,glitch_idx, f0_w, tau_w, t0_w) #find the NN and MM matrixies from filter functions

        #start calculating the LogLikelihood
        LogL = 0
        LogL += -1/2*self.resres_logdet
        #loop over the pulsars to add in the noise transients
        for i in range(len(self.psrs)):
            # if any(i-0.5 <= idx <= i+0.5 for idx in glitch_idx): #only calc terms if a glitch is present
            #     for k in range(self.Nglitch):
            #         LogL += (sigma[k,0]*self.NN[i, 0+2*k] + sigma[k,1]*self.NN[i, 1+2*k]) #adding in NN term in sum
            #         for l in range(self.Nglitch):
            #             LogL += -1/2*(sigma[k,0]*(sigma[l,0]*self.MMs[i, 0+2*k, 0+2*l] + sigma[l,1]*self.MMs[i, 0+2*k, 1+2*l]) + sigma[k,1]*(sigma[l,0]*self.MMs[i, 1+2*k, 0+2*l] + sigma[l,1]*self.MMs[i, 1+2*k, 1+2*l]))
            for k in range(self.Nwavelet + self.Nglitch):
                LogL += (sigma[i,k,0]*self.NN[i, 0+2*k] + sigma[i,k,1]*self.NN[i, 1+2*k]) #adding in NN term in sum
                for l in range(self.Nwavelet + self.Nglitch):
                    LogL += -1/2*(sigma[i,k,0]*(sigma[i,l,0]*self.MMs[i, 0+2*k, 0+2*l] + sigma[i,l,1]*self.MMs[i, 0+2*k, 1+2*l]) + sigma[i,k,1]*(sigma[i,l,0]*self.MMs[i, 1+2*k, 0+2*l] + sigma[i,l,1]*self.MMs[i, 1+2*k, 1+2*l]))
        return LogL

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

#####
#function for taking the dot product of two tensors a and b
#See page 132 in https://arxiv.org/pdf/2105.13270.pdf, where (x|y) = x^T*C_inv*y
#####
def dot_product(a, b, phiinv_loc, T, TNT, Sigma, Nvec):

    invchol_Sigma_TNs = List.empty_list(nb.types.float64[:,::1])
    Ndiag = np.diag(1/Nvec)

    #first term in the dot product
    aNb = np.dot(np.dot(a, Ndiag), b)

    #may need special case when phiinv_loc.ndim=1
    Sigma = TNT + phiinv_loc

    #mutate inplace to avoid memory allocation overheads
    chol_Sigma,lower = sl.cho_factor(Sigma.T,lower=True,overwrite_a=True,check_finite=False)
    invchol_Sigma_T_loc = solve_triangular(chol_Sigma,T.T,lower_a=True,trans_a=False)
    invchol_Sigma_TNs.append(np.ascontiguousarray(invchol_Sigma_T_loc/Nvec))

    invCholSigmaTN = invchol_Sigma_TNs[0]
    SigmaTNaProd = np.dot(invCholSigmaTN,a)
    SigmaTNbProd = np.dot(invCholSigmaTN,b)
    dotSigmaTNr = np.dot(SigmaTNaProd.T,SigmaTNbProd)

    dot_prod = aNb - dotSigmaTNr

    return dot_prod
