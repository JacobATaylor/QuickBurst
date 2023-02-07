"""C 2023 Jacob, Rand, and Bence fast Burst likelihood"""
import numpy as np
import numba as nb
from numba import njit,prange
from numba.experimental import jitclass
from numba.typed import List
#import scipy.linalg
import scipy.linalg as sl
from lapack_wrappers import solve_triangular

from enterprise import constants as const
from enterprise_extensions.frequentist import Fe_statistic as FeStat

from memory_profiler import profile

#########
#strucutre overview
#
#Class to hold info about pta, params, psrs arrays
#   function calc M and N matrixies to be used
#   function to calculate likelihoods
########

class FastBurst:
    #@profile
    def __init__(self,pta,psrs,params,Npsr, tref):

        self.pta = pta
        self.psrs = psrs
        self.Npsr = Npsr
        self.params = params
        '''used self.pta.params instead if self.params, might have been wrong'''
        self.Nvecs = List(self.pta.get_ndiag(self.params))
        #print('Nvecs arary: ', self.Nvecs)

        self.TNTs = self.pta.get_TNT(self.params)
        self.Ts = self.pta.get_basis()
        #self.Nmats = 0
        #self.Nmats = self.get_Nmats()

        self.MMs = np.zeros((Npsr,2,2))
        self.NN = np.zeros((Npsr,2))

        self.sigma = np.zeros(2)
        #invchol_Sigma_Ts = List()
        self.Nrs = List()
        self.isqrNvecs = List()


        self.toas = List([psr.toas - tref for psr in psrs])
        self.residuals = List([psr.residuals for psr in psrs])

        for i in range(self.Npsr):
            self.Nrs.append(self.residuals[i]/np.sqrt(self.Nvecs[i]))

        self.logdet = 0
        for (l,m) in self.pta.get_rNr_logdet(params): #can't keep the rNr term as it removes the deterministic signal already
            self.logdet += m

        #self.resres_logdet = np.sum([ell for ell in self.pta.get_rNr_logdet(params)])
        #print(-0.5*self.resres_logdet)
        rNr_loc = np.zeros(self.Npsr)

        logdet_array = np.zeros(self.Npsr)
        pls_temp = self.pta.get_phiinv(self.params, logdet=True, method='partition')
        #print(pls_temp)

        invchol_Sigma_TNs = List.empty_list(nb.types.float64[:,::1])

        dotSigmaTNr = np.zeros(self.Npsr)

        for i in range(self.Npsr):

            phiinv_loc,logdetphi_loc = pls_temp[i]
            #print(phiinv_loc)

            Ndiag = np.diag(1/self.Nvecs[i])
            #first term in the dot product
            rNr_loc[i] = np.dot(np.dot(self.psrs[i].residuals, Ndiag), self.psrs[i].residuals)

            '''may need special case when phiinv_loc.ndim=1'''
            Sigma = self.TNTs[i]+phiinv_loc#[i]

            #mutate inplace to avoid memory allocation overheads
            chol_Sigma,lower = sl.cho_factor(Sigma.T,lower=True,overwrite_a=True,check_finite=False)
            invchol_Sigma_T_loc = solve_triangular(chol_Sigma,self.Ts[i].T,lower_a=True,trans_a=False)
            invchol_Sigma_TNs.append(np.ascontiguousarray(invchol_Sigma_T_loc/np.sqrt(self.Nvecs[i]))) #half of the Nvecs here, other half in Nrs

            logdet_Sigma_loc = logdet_Sigma_helper(chol_Sigma)#2 * np.sum(np.log(np.diag(chol_Sigma)))

            #add the necessary component to logdet
            logdet_array[i] =  logdetphi_loc + logdet_Sigma_loc
            #print('logdet_phi: ',logdetphi_loc)
            #print('logdet_sigma: ',logdet_Sigma_loc)

            invCholSigmaTN = invchol_Sigma_TNs[i]
            SigmaTNrProd1 = np.dot(invCholSigmaTN,self.Nrs[i])
            SigmaTNrProd2 = np.dot(invCholSigmaTN,self.Nrs[i])

            dotSigmaTNr[i] = np.dot(SigmaTNrProd2.T,SigmaTNrProd1)
            #print('dotSigmaTNr: ',dotSigmaTNr[i])

        self.resres_logdet = self.logdet + np.sum(rNr_loc) + np.sum(logdet_array) - np.sum(dotSigmaTNr)
        #self.resres_logdet = self.resres_logdet + np.sum(logdet_array) - np.sum(dotSigmaTNr)
        print(-0.5*self.resres_logdet)
    #@profile
    def get_M_N(self, f0, tau, t0, glitch_idx):
        #call the enterprise inner product

        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition')

        print('Input time: ', t0/86400)

        for ii in range(self.Npsr):

            TNT = self.TNTs[ii]
            T = self.Ts[ii]
            phiinv = phiinvs[ii]
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            #Nmat = self.Nmats[ii]
            #filter fuctions
            Filt_cos = np.zeros(len(self.toas[ii]))
            Filt_sin = np.zeros(len(self.toas[ii]))
            #Single noise transient wavelet
            if (ii-0.5 <= glitch_idx <= ii+0.5):
                Filt_cos = np.exp(-1*((self.toas[ii] - t0)/tau)**2)*np.cos(2*np.pi*f0*(self.toas[ii] - t0))
                Filt_sin = np.exp(-1*((self.toas[ii] - t0)/tau)**2)*np.sin(2*np.pi*f0*(self.toas[ii] - t0))
            #print('Cosine: ', Filt_cos)
            #print('Sine: ', Filt_sin)
            #print('Exponential: ', -((self.psrs[ii].toas - t0)/tau)**2)
            #do dot product
            #populate MM,NN
            '''
            MMs = matrix of size (Npsr, N_filters, N_filters) that is defined as the dot product between filter functions
            '''
            '''
            self.MMs[ii, 0, 0] = FeStat.innerProduct_rr(Filt_cos,Filt_cos,Nmat,T,Sigma)
            self.MMs[ii, 1, 0] = FeStat.innerProduct_rr(Filt_sin, Filt_cos,Nmat,T,Sigma)
            self.MMs[ii, 0, 1] = FeStat.innerProduct_rr(Filt_cos, Filt_sin,Nmat,T,Sigma)
            self.MMs[ii, 1, 1] = FeStat.innerProduct_rr(Filt_sin, Filt_sin,Nmat,T,Sigma)
            print('MM matrix:', self.MMs)
            self.NN[ii, 0] = FeStat.innerProduct_rr(self.psrs[ii].residuals,Filt_cos,Nmat,T,Sigma)
            self.NN[ii, 1] = FeStat.innerProduct_rr(self.psrs[ii].residuals,Filt_sin,Nmat,T,Sigma)
            print('NN matrix:', self.NN)
            '''
            self.MMs[ii, 0, 0] = self.dot_product(Filt_cos,Filt_cos,T,Sigma,ii)
            self.MMs[ii, 1, 0] = self.dot_product(Filt_sin, Filt_cos,T,Sigma,ii)
            self.MMs[ii, 0, 1] = self.dot_product(Filt_cos, Filt_sin,T,Sigma,ii)
            self.MMs[ii, 1, 1] = self.dot_product(Filt_sin, Filt_sin,T,Sigma,ii)
            #print('MM matrix:', self.MMs)
            self.NN[ii, 0] = self.dot_product(self.psrs[ii].residuals,Filt_cos,T,Sigma,ii)#self.psrs[ii].residuals
            self.NN[ii, 1] = self.dot_product(self.psrs[ii].residuals,Filt_sin,T,Sigma,ii)
            #self.NN[ii, 0] = self.dot_product(self.Nrs[ii],Filt_cos,T,Sigma,ii)#self.psrs[ii].residuals
            #self.NN[ii, 1] = self.dot_product(self.Nrs[ii],Filt_sin,T,Sigma,ii)
            #print('NN matrix:', self.NN)


    def get_sigmas(self, A, phi0):

        #noise transient coefficients
        self.sigma[0] = A*np.cos(phi0)
        self.sigma[1] = -A*np.sin(phi0)
    #@profile
    def get_Nmats(self):
        '''Makes the Nmatrix used in the fstatistic'''
        TNTs = self.TNTs
        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition')
        # Get noise parameters for pta toaerr**2
        #Nvecs = self.pta.get_ndiag(self.params)
        # Get the basis matrix
        Ts = self.Ts

        Nmats = [make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, self.Nvecs, Ts)]

        return Nmats
    #@profile
    def dot_product(self, a, b, T, Sigma, psr_idx):

        dot_prod = 0

        pls_temp = self.pta.get_phiinv(self.params, logdet=True, method='partition')

        invchol_Sigma_TNs = List.empty_list(nb.types.float64[:,::1])
        dotSigmaTNr = 0

        Ndiag = np.diag(1/self.Nvecs[psr_idx])
        #first term in the dot product
        aNb = np.dot(np.dot(a, Ndiag), b)
        #print(Ndiag)
        '''this is the only place we use Nmat, every other aNb looks like it uses Nvec'''

        phiinv_loc,logdetphi_loc = pls_temp[psr_idx]

        '''may need special case when phiinv_loc.ndim=1'''
        Sigma = self.TNTs[psr_idx] + phiinv_loc

        #mutate inplace to avoid memory allocation overheads
        chol_Sigma,lower = sl.cho_factor(Sigma.T,lower=True,overwrite_a=True,check_finite=False)
        invchol_Sigma_T_loc = solve_triangular(chol_Sigma,self.Ts[psr_idx].T,lower_a=True,trans_a=False)
        invchol_Sigma_TNs.append(np.ascontiguousarray(invchol_Sigma_T_loc/self.Nvecs[psr_idx]))

        invCholSigmaTN = invchol_Sigma_TNs[0]
        SigmaTNaProd = np.dot(invCholSigmaTN,a)
        SigmaTNbProd = np.dot(invCholSigmaTN,b)
        '''
        dotTNa = self.Ts[psr_idx].T/np.sqrt(self.Nvecs[psr_idx])*a
        SigmaTNaProd = solve_triangular(chol_Sigma,dotTNa,lower_a=True,trans_a=False,overwrite_b=True)

        dotTNb = self.Ts[psr_idx].T/np.sqrt(self.Nvecs[psr_idx])*b
        SigmaTNbProd = solve_triangular(chol_Sigma,dotTNb,lower_a=True,trans_a=False,overwrite_b=True)
        '''

        dotSigmaTNr = np.dot(SigmaTNaProd.T,SigmaTNbProd)
        #dotSigmaTNr = SigmaTNaProd*SigmaTNbProd
        #print('SigmaTNbProd ',SigmaTNbProd)

        dot_prod = aNb - dotSigmaTNr
        #print(dot_prod)
        return dot_prod
    #@profile
    def get_lnlikelihood(self, A, phi0, f0, tau, t0, glitch_idx):
        #print('Amplitude: ', A)
        #print('Frequency: ', f0)
        """Function to do likelihood evaluations in QuickBurst, currently for a single noise transient wavelet"""

        '''
        x0: List of params
        resres: pulsar residuals
        logdet: logdet (2*pi*C) = log(det(phi)*det(sigma)*det(N))
        '''
        '''
        self.NN is matrix of size (Npsr, 2), where 2 is the # of filter functions used to model transient wavelet. sigma_k[i] are coefficients on filter functions.
        '''
        #print('glitch_index: ', glitch_idx)

        #print('Old Sigma: ', self.sigma)
        self.get_sigmas(A, phi0)
        #print('New sigma: ', self.sigma)

        #print('Old M and N: ', self.MMs[0, :, :], self.NN[0, :])
        self.get_M_N(f0,tau,t0,glitch_idx)
        #print('New M and N: ', self.MMs[0, :, :], self.NN[0, :])
        LogL = 0
        '''
        ######Understanding the components of logdet######
        logdet = logdet(2*pi*C) = log(det(phi)*det(sigma)*det(N))
        N = white noise covariance matrix
        phi = prior matrix
        sigma = inverse(phi) - transpose(T)*inverse(N)*T (first term -> phiinv, second term -> TNT)
        T = [M F]
        M = Design matrix
        F = Fourier matrix (matrix of fourier coefficients and sin/cos terms)
        We should print out these terms in both enterprise and our code and compare, rather than printing out multiple terms calculated (maybe?).
        '''
        LogL += -1/2*self.resres_logdet
        print('adding in resres_logdet', LogL)

        for i in range(len(self.psrs)):
            LogL += (self.sigma[0]*self.NN[i, 0] + self.sigma[1]*self.NN[i, 1])
            LogL += -1/2*(self.sigma[0]*(self.sigma[0]*self.MMs[i, 0, 0] + self.sigma[1]*self.MMs[i, 0, 1]) + self.sigma[1]*(self.sigma[0]*self.MMs[i, 1, 0] + self.sigma[1]*self.MMs[i, 1, 1]))
            #print('LogL: ', LogL)
        return LogL

'''Tried moving Nmat calc outside the class to match Fe stat code'''
#@profile
def make_Nmat(phiinv, TNT, Nvec, T):

    Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
    cf = sl.cho_factor(Sigma)
    # Nshape = np.shape(T)[0] # Not currently used in code

    TtN = np.multiply((1/Nvec)[:, None], T).T

    # Put pulsar's autoerrors in a diagonal matrix
    Ndiag = np.diag(1/Nvec)

    expval2 = sl.cho_solve(cf, TtN)
    # TtNt = np.transpose(TtN) # Not currently used in code

    # An Ntoa by Ntoa noise matrix to be used in expand dense matrix calculations earlier
    return Ndiag - np.dot(TtN.T, expval2)
#@profile
@njit(parallel=True,fastmath=True)
def logdet_Sigma_helper(chol_Sigma):
    """get logdet sigma from cholesky"""
    res = 0.
    for itrj in prange(0,chol_Sigma.shape[0]):
        res += np.log(chol_Sigma[itrj,itrj])
    return 2*res
