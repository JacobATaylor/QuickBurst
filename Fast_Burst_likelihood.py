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
    def __init__(self,pta,psrs,params,Npsr,tref):

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
        self.MMs = np.zeros((Npsr,2,2))
        self.NN = np.zeros((Npsr,2))
        self.sigma = np.zeros(2)

    #####
    #generates the MM and NN matrixies from filter functions
    #####
    def get_M_N(self, f0, tau, t0, glitch_idx):

        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition') #use enterprise to calculate phiinvs

        for ii in range(self.Npsr):

            TNT = self.TNTs[ii]
            T = self.Ts[ii]
            phiinv = phiinvs[ii]
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            #filter fuctions
            Filt_cos = np.zeros(len(self.toas[ii]))
            Filt_sin = np.zeros(len(self.toas[ii]))

            #Single noise transient wavelet
            if (ii-0.5 <= glitch_idx <= ii+0.5): #only populate filter functions for pulsar with glitcvh in it
                Filt_cos = np.exp(-1*((self.toas[ii] - t0)/tau)**2)*np.cos(2*np.pi*f0*(self.toas[ii] - t0)) #see PDF for derivation
                Filt_sin = np.exp(-1*((self.toas[ii] - t0)/tau)**2)*np.sin(2*np.pi*f0*(self.toas[ii] - t0))

            #populate MM,NN
            self.MMs[ii, 0, 0] = dot_product(Filt_cos, Filt_cos,phiinv,T,TNT,Sigma,self.Nvecs[ii])
            self.MMs[ii, 1, 0] = dot_product(Filt_sin, Filt_cos,phiinv,T,TNT,Sigma,self.Nvecs[ii])
            self.MMs[ii, 0, 1] = dot_product(Filt_cos, Filt_sin,phiinv,T,TNT,Sigma,self.Nvecs[ii])
            self.MMs[ii, 1, 1] = dot_product(Filt_sin, Filt_sin,phiinv,T,TNT,Sigma,self.Nvecs[ii])

            self.NN[ii, 0] = dot_product(self.residuals[ii],Filt_cos,phiinv,T,TNT,Sigma,self.Nvecs[ii])
            self.NN[ii, 1] = dot_product(self.residuals[ii],Filt_sin,phiinv,T,TNT,Sigma,self.Nvecs[ii])

    ######
    #calculates amplitudes for Signals
    ######
    def get_sigmas(self, A, phi0):
        #noise transient coefficients
        self.sigma[0] = A*np.cos(phi0)
        self.sigma[1] = -A*np.sin(phi0)

    #####
    #calculates lnliklihood for a set of signal parameters
    #####
    def get_lnlikelihood(self, A, phi0, f0, tau, t0, glitch_idx):

        '''
        ######Understanding the components of logdet######
        logdet = logdet(2*pi*C) = log(det(phi)*det(sigma)*det(N))
        N = white noise covariance matrix
        self.NN is matrix of size (Npsr, 2), where 2 is the # of filter functions used to model transient wavelet. sigma_k[i] are coefficients on filter functions.
        phi = prior matrix
        sigma = inverse(phi) - transpose(T)*inverse(N)*T (first term -> phiinv, second term -> TNT)
        T = [M F]
        M = Design matrix
        F = Fourier matrix (matrix of fourier coefficients and sin/cos terms)
        '''

        self.get_sigmas(A, phi0) #calculate the amplitudes of noise transients
        self.get_M_N(f0,tau,t0,glitch_idx) #find the NN and MM matrixies from filter functions

        #start calculating the LogLikelihood
        LogL = 0
        LogL += -1/2*self.resres_logdet
        #loop over the pulsars to add in the noise transients
        for i in range(len(self.psrs)):
            LogL += (self.sigma[0]*self.NN[i, 0] + self.sigma[1]*self.NN[i, 1]) #adding in NN term in sum
            LogL += -1/2*(self.sigma[0]*(self.sigma[0]*self.MMs[i, 0, 0] + self.sigma[1]*self.MMs[i, 0, 1]) + self.sigma[1]*(self.sigma[0]*self.MMs[i, 1, 0] + self.sigma[1]*self.MMs[i, 1, 1]))
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
