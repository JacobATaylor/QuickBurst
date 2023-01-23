"""C 2022 Jacob, Rand, and Bence fast Burst likelihood"""
import numpy as np
import numba as nb
from numba import njit,prange
from numba.experimental import jitclass
from numba.typed import List
#import scipy.linalg
import scipy.linalg as sl

from enterprise import constants as const
from enterprise_extensions.frequentist import Fe_statistic as FeStat

#########
#strucutre overview
#
#Class to hold info about pta, params, psrs arrays
#   function calc M and N matrixies to be used
#   function to calculate likelihoods
########

class FastBurst:
    def __init__(self,pta,psrs,params,Npsr, tref):

        self.pta = pta
        self.psrs = psrs
        self.Npsr = Npsr
        self.params = params

        self.Nmats = self.get_Nmats()

        self.MMs = np.zeros((Npsr,2,2))
        self.NN = np.zeros((Npsr,2))

        self.sigma = np.zeros(2)

        '''used self.pta.params instead if self.params, might have been wrong'''
        self.Nvecs = List(self.pta.get_ndiag(self.params))
        print('Nvecs arary: ', self.Nvecs)
        #get the part of the determinant that can be computed right now
        '''what does this function actualy return'''
        self.logdet = 0.0
        for (l,m) in self.pta.get_rNr_logdet(self.params):
            self.logdet += m
        #self.logdet += np.sum([m for (l,m) in self.pta.get_rNr_logdet(self.params)])
        print(self.logdet)
        #get the other pta results
        self.TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()

        #invchol_Sigma_Ts = List()
        self.Nrs = List()
        self.isqrNvecs = List()

        self.toas = List([psr.toas - tref for psr in psrs])
        self.residuals = List([psr.residuals for psr in psrs])

        self.resres_rNr = 0.
        self.TNvs = List()
        self.dotTNrs = List()
        for i in range(self.Npsr):
            self.isqrNvecs.append(1/np.sqrt(self.Nvecs[i]))
            self.Nrs.append(self.residuals[i]/np.sqrt(self.Nvecs[i]))
            print('Nrs: ', self.Nrs)
            self.resres_rNr += np.dot(self.Nrs[i], self.Nrs[i])
            print('resres_rNr: ', self.resres_rNr)
            self.TNvs.append((Ts[i].T/np.sqrt(self.Nvecs[i])).copy().T) #store F contiguous version
            self.dotTNrs.append(np.dot(self.Nrs[i],self.TNvs[i]))

        #put the rnr part of resres onto logdet
        '''why do we need to sum the rnr dot product'''

        self.logdet_base = self.resres_rNr+self.logdet
        '''Adds phi and sigma calculations to logdet, which is used to calculate covariance C. '''
        self.logdet = self.set_logdet()
        print('self.logdet: ', self.logdet)

    '''function pulled from various parts of Quick CW to finish calculating logdet for lnlikelihood calls'''
    def set_logdet(self):

        logdet_array = np.zeros(self.Npsr)
        pls_temp = self.pta.get_phiinv(self.params, logdet=True, method='partition')

        for i in range(self.Npsr):
            phiinv_loc,logdetphi_loc = pls_temp[i]

            '''may need special case when phiinv_loc.ndim=1'''
            Sigma = self.TNTs[i]+phiinv_loc

            #mutate inplace to avoid memory allocation overheads
            chol_Sigma,lower = sl.cho_factor(Sigma.T,lower=True,overwrite_a=True,check_finite=False)

            logdet_Sigma_loc = logdet_Sigma_helper(chol_Sigma)#2 * np.sum(np.log(np.diag(chol_Sigma)))

            #add the necessary component to logdet
            logdet_array[i] = - logdetphi_loc - logdet_Sigma_loc

        return self.logdet_base + np.sum(logdet_array)

    def get_M_N(self, f0, tau, t0):
        #call the enterprise inner product

        phiinvs = self.pta.get_phiinv(self.params, logdet=False)
        TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()

        print('Input time: ', t0/86400)

        for ii in range(self.Npsr):

            TNT = TNTs[ii]
            T = Ts[ii]
            phiinv = phiinvs[ii]
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

            Nmat = self.Nmats[ii]
            #filter fuctions
            Filt_cos = np.zeros(len(self.psrs[ii].toas))
            Filt_sin = np.zeros(len(self.psrs[ii].toas))

            print('Per pulsar toas: ', self.toas[ii]/86400)
            #Single noise transient wavelet
            Filt_cos = np.exp(-1*((self.toas[ii] - t0)/tau)**2)*np.cos(2*np.pi*f0*(self.toas[ii] - t0))
            Filt_sin = np.exp(-1*((self.toas[ii] - t0)/tau)**2)*np.sin(2*np.pi*f0*(self.toas[ii] - t0))
            print('Cosine: ', Filt_cos)
            print('Sine: ', Filt_sin)
            print('Exponential: ', -((self.psrs[ii].toas - t0)/tau)**2)
            #do dot product
            #populate MM,NN
            '''
            MMs = matrix of size (Npsr, N_filters, N_filters) that is defined as the dot product between filter functions
            '''
            self.MMs[ii, 0, 0] = FeStat.innerProduct_rr(Filt_cos,Filt_cos,Nmat,T,Sigma)
            self.MMs[ii, 1, 0] = FeStat.innerProduct_rr(Filt_sin, Filt_cos,Nmat,T,Sigma)
            self.MMs[ii, 0, 1] = FeStat.innerProduct_rr(Filt_cos, Filt_sin,Nmat,T,Sigma)
            self.MMs[ii, 1, 1] = FeStat.innerProduct_rr(Filt_sin, Filt_sin,Nmat,T,Sigma)

            self.NN[ii, 0] = FeStat.innerProduct_rr(Filt_cos,self.psrs[ii].residuals,Nmat,T,Sigma)
            self.NN[ii, 1] = FeStat.innerProduct_rr(Filt_sin,self.psrs[ii].residuals,Nmat,T,Sigma)


    def get_sigmas(self, A, phi0):

        #expects
        self.sigma[0] = A*np.cos(phi0)
        self.sigma[1] = -A*np.sin(phi0)

    def get_Nmats(self):
        '''Makes the Nmatrix used in the fstatistic'''
        TNTs = self.pta.get_TNT(self.params)
        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition')
        # Get noise parameters for pta toaerr**2
        Nvecs = self.pta.get_ndiag(self.params)
        # Get the basis matrix
        Ts = self.pta.get_basis(self.params)

        Nmats = [make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, Nvecs, Ts)]

        return Nmats

    def get_lnlikelihood(self, A, phi0, f0, tau, t0, glitch_idx):
        print('Amplitude: ', A)
        print('Frequency: ', f0)
        """Function to do likelihood evaluations in QuickBurst, currently for a single noise transient wavelet"""

        '''
        x0: List of params
        resres: pulsar residuals
        logdet: log(2*pi*C) in loglikelihood calculation
        '''
        '''
        self.NN is matrix of size (Npsr, 2), where 2 is the # of filter functions used to model transient wavelet. sigma_k[i] are coefficients on filter functions.
        '''
        print('glitch_index: ', glitch_idx)

        print('Old Sigma: ', self.sigma)
        self.get_sigmas(A, phi0)
        print('New sigma: ', self.sigma)

        print('Old M and N: ', self.MMs[0, :, :], self.NN[0, :])
        self.get_M_N(f0,tau,t0)
        print('New M and N: ', self.MMs[0, :, :], self.NN[0, :])
        LogL = 0
        LogL += -1/2*self.logdet
        for i in range(len(self.psrs)):
            #if statments to only include the glitch in the pulaser it is assigned to
            if (i-0.5 <= glitch_idx <= i+0.5):
                LogL += -1/2*np.dot(self.Nrs[i], self.Nrs[i]) + (self.sigma[0]*self.NN[i, 0] + self.sigma[1]*self.NN[i, 1])
                LogL += -1/2*(self.sigma[0]*(self.sigma[0]*self.MMs[i, 0, 0] + self.sigma[1]*self.MMs[i, 0, 1]) + self.sigma[1]*(self.sigma[0]*self.MMs[i, 1, 0] + self.sigma[1]*self.MMs[i, 1, 1]))
                print('reseres: ', np.dot(self.Nrs[i], self.Nrs[i]))
                print('logdet: ', self.logdet)
            else:
                print('skipped ', i)
                LogL += -1/2*np.dot(self.Nrs[i], self.Nrs[i])
                print('reseres: ', np.dot(self.Nrs[i], self.Nrs[i]))
                print('logdet: ', self.logdet)
        return LogL

'''Tried moving Nmat calc outside the class to match Fe stat code'''
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

@njit(parallel=True,fastmath=True)
def logdet_Sigma_helper(chol_Sigma):
    """get logdet sigma from cholesky"""
    res = 0.
    for itrj in prange(0,chol_Sigma.shape[0]):
        res += np.log(chol_Sigma[itrj,itrj])
    return 2*res

# for TNr, TNT, pl in zip(TNrs, TNTs, phiinvs):
#     if TNr is None:
#         continue
#
#     phiinv, logdet_phi = pl
#     Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
#
#     try:
#         cf = sl.cho_factor(Sigma)
#         expval = sl.cho_solve(cf, TNr)
#     except sl.LinAlgError:  # pragma: no cover
#         return -np.inf
#
#     logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
#
#     loglike += 0.5 * (np.dot(TNr, expval) - logdet_sigma - logdet_phi)

    # def cov_calc():
    #     cov = stuff
    #
    #
    #     return cov
    #     logL = np.sum(-1/2*)
    #     prior_ranges = {'amplitude': [0, 2]}
    #     amp_prior = [prior_ranges['amplitude'][0], prior_ranges['amplitude'][1]]
    #     Central_freq_prior = [prior_ranges['frequency'][0], prior_ranges['frequency'][1]]
    #     phase_prior = [prior_ranges['phase'][0], prior_ranges['phase'][1]]
    #
    #     if prior == 'Uniform':
    #         amp_prior = Parameter.Uniform(amp_prior[0], amp_prior[1])
    #         Central_freq_prior = Parameter.Uniform(Central_freq_prior[0], Central_freq_prior[1])
    #
    #     amp_value = amp_prior.sample()
    #
    #
    #     """jittable helper for calculating the log likelihood in CWFastLikelihood"""
    #     if prior_recovery:
    #         return 0.0
    #     else:
    #         fgw = 10.**x0.log10_fgw
    #         amp = 10.**x0.log10_h / (2*np.pi*fgw)
    #         mc = 10.**x0.log10_mc * const.Tsun
    #
    #
    #         sin_gwtheta = np.sqrt(1-x0.cos_gwtheta**2)
    #         sin_gwphi = np.sin(x0.gwphi)
    #         cos_gwphi = np.cos(x0.gwphi)
    #
    #         m = np.array([sin_gwphi, -cos_gwphi, 0.0])
    #         n = np.array([-x0.cos_gwtheta * cos_gwphi, -x0.cos_gwtheta * sin_gwphi, sin_gwtheta])
    #         omhat = np.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta * sin_gwphi, -x0.cos_gwtheta])
    #         sigma = np.zeros(4)
    #
    #         cos_phase0 = np.cos(x0.phase0)
    #         sin_phase0 = np.sin(x0.phase0)
    #         sin_2psi = np.sin(2*x0.psi)
    #         cos_2psi = np.cos(2*x0.psi)
    #
    #         log_L = -0.5*resres -0.5*logdet
    #
    #         if includeCW:
    #             for i in prange(0,x0.Npsr):
    #                 m_pos = 0.
    #                 n_pos = 0.
    #                 cosMu = 0.
    #                 for j in range(0,3):
    #                     m_pos += m[j]*pos[i,j]
    #                     n_pos += n[j]*pos[i,j]
    #                     cosMu -= omhat[j]*pos[i,j]
    #                 #m_pos = np.dot(m, pos[i])
    #                 #n_pos = np.dot(n, pos[i])
    #                 #cosMu = -np.dot(omhat, pos[i])
    #
    #                 F_p = 0.5 * (m_pos ** 2 - n_pos ** 2) / (1 - cosMu)
    #                 F_c = (m_pos * n_pos) / (1 - cosMu)
    #
    #                 p_dist = (pdist[i,0] + pdist[i,1]*x0.cw_p_dists[i])*(const.kpc/const.c)
    #
    #                 w0 = np.pi * fgw
    #                 omega_p0 = w0 *(1 + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1-cosMu))**(-3/8)
    #
    #                 amp_psr = amp * (w0/omega_p0)**(1.0/3.0)
    #                 phase0_psr = x0.cw_p_phases[i]
    #
    #                 cos_phase0_psr = np.cos(x0.phase0+phase0_psr*2.0)
    #                 sin_phase0_psr = np.sin(x0.phase0+phase0_psr*2.0)
    #
    #                 sigma[0] =  amp*(   cos_phase0 * (1+x0.cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Earth term sine
    #                                   2*sin_phase0 *     x0.cos_inc    * (+sin_2psi * F_p + cos_2psi * F_c)   )
    #                 sigma[1] =  amp*(   sin_phase0 * (1+x0.cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Earth term cosine
    #                                   2*cos_phase0 *     x0.cos_inc    * (-sin_2psi * F_p - cos_2psi * F_c)   )
    #                 sigma[2] =  -amp_psr*(   cos_phase0_psr * (1+x0.cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Pulsar term sine
    #                                   2*sin_phase0_psr *     x0.cos_inc    * (+sin_2psi * F_p + cos_2psi * F_c)   )
    #                 sigma[3] =  -amp_psr*(   sin_phase0_psr * (1+x0.cos_inc**2) * (-cos_2psi * F_p + sin_2psi * F_c) +    #Pulsar term cosine
    #                                   2*cos_phase0_psr *     x0.cos_inc    * (-sin_2psi * F_p - cos_2psi * F_c)   )
    #
    #                 for j in range(0,4):
    #                     log_L += sigma[j]*NN[i,j]
    #
    #                 prodMMPart = 0.
    #                 for j in range(0,4):
    #                     for k in range(0,4):
    #                         prodMMPart += sigma[j]*MMs[i,j,k]*sigma[k]
    #
    #                 log_L -= prodMMPart/2#np.dot(sigma,np.dot(MMs[i],sigma))/2
    #
    #        return log_L
