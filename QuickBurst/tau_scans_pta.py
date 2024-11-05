"""
C 2023 Jacob Taylor, Rand Burnette, and Bence Becsy fast Burst Tau Scan

Functions for claculating informative step distributions over some shape parameters (Tau Scans)
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.linalg as sl
import json

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

class TauScan(object):
    """
    Class for the Tau scans.
    :param psrs: List of `enterprise` Pulsar instances.
    :param params: Dictionary of noise parameters.
    """

    def __init__(self, psrs, params=None, pta=None):
        '''populate parameters and generate PTA if not provided'''
        if pta is None: #generates simple PTA assuming noise dictionary
            print('Initializing the model...')

            efac = parameter.Constant()
            equad = parameter.Constant()
            ef = white_signals.MeasurementNoise(efac=efac)
            eq = white_signals.EquadNoise(log10_equad=equad)

            tm = gp_signals.TimingModel(use_svd=True)

            s = eq + ef + tm

            model = []
            for p in psrs:
                model.append(s(p))
            self.pta = signal_base.PTA(model)

            # set white noise parameters
            if params is None:
                print('No noise dictionary provided!...')
            else:
                self.pta.set_default_params(params)
        else:
            self.pta = pta

        self.psrs = psrs
        self.params = params
        self.phiinvs = self.pta.get_phiinv(self.params, logdet=False)
        self.TNTs = self.pta.get_TNT(self.params)
        self.Ts = self.pta.get_basis()
        self.chol_sigmainvs = [sl.cho_factor(TNT + np.diag(phiinv)) for TNT, phiinv in zip(self.TNTs, self.phiinvs)]
        self.Nvecs = self.pta.get_ndiag(self.params)
        self.Nmats = None


    def get_Nmats(self):
        '''Makes the Nmatrix used in the fstatistic'''
        TNTs = self.pta.get_TNT(self.params)
        phiinvs = self.pta.get_phiinv(self.params, logdet=False, method='partition')
        #Get noise parameters for pta toaerr**2
        Nvecs = self.pta.get_ndiag(self.params)
        #Get the basis matrix
        Ts = self.pta.get_basis(self.params)

        Nmats = [ make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, Nvecs, Ts)]

        return Nmats

    def compute_TauScan(self, tau, t0, f0, tref=53000*86400):
        """
        Computes the Tau-scans.
        :param tau: tau of wavelet to use
        :param t0: central time of wavelet to use
        :param f0: central frequency of wavelet to use
        :returns:
        tau_scan: value of tau-scan at given tau, t0, f0 (map can be produced by looping over these
        """

        n_psr = len(self.psrs)
        tau_scan = 0

        for idx, (psr, Nvec, TNT, phiinv, T, sigmainv) in enumerate(zip(self.psrs, self.Nvecs,
                                             self.TNTs, self.phiinvs, self.Ts, self.chol_sigmainvs)):

            ntoa = len(psr.toas)

            wavelet_cos = MorletGaborWavelet(psr.toas-tref, 1.0, tau, f0, t0, 0.0)
            wavelet_sin = MorletGaborWavelet(psr.toas-tref, 1.0, tau, f0, t0, np.pi/2)

            cos_norm = np.sqrt(innerprod_cho(Nvec, T, sigmainv, wavelet_cos, wavelet_cos))
            sin_norm = np.sqrt(innerprod_cho(Nvec, T, sigmainv, wavelet_sin, wavelet_sin))
            #catch for missing data
            if cos_norm==0.0:
                cos_norm = 1.0
            if sin_norm==0.0:
                sin_norm = 1.0

            wavelet_cos = MorletGaborWavelet(psr.toas-tref, 1.0/cos_norm, tau, f0, t0, 0.0)
            wavelet_sin = MorletGaborWavelet(psr.toas-tref, 1.0/sin_norm, tau, f0, t0, np.pi/2)
            tau_scan += innerprod_cho(Nvec, T, sigmainv, wavelet_cos, psr.residuals)**2 + innerprod_cho(Nvec, T, sigmainv, wavelet_sin, psr.residuals)**2
            if tau_scan == np.nan:
                print('Cos: {}, Sin: {}'.format(cos_norm, sin_norm))
        return tau_scan


def make_Nmat(phiinv, TNT, Nvec, T):

    Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
    cf = sl.cho_factor(Sigma)
    Nshape = np.shape(T)[0]

    TtN = Nvec.solve(other = np.eye(Nshape),left_array = T)

    #Put pulsar's autoerrors in a diagonal matrix
    Ndiag = Nvec.solve(other = np.eye(Nshape),left_array = np.eye(Nshape))

    expval2 = sl.cho_solve(cf,TtN)

    #An Ntoa by Ntoa noise matrix to be used in expand dense matrix calculations earlier
    return Ndiag - np.dot(TtN.T,expval2)


def MorletGaborWavelet(t, A, tau, f0, t0, phi0): #functions used to construct signials and noise transients
    return A * np.exp(-(t-t0)**2/tau**2) * np.cos(2*np.pi*f0*(t-t0) + phi0)


def innerprod_cho(Nvec, T, cf, x, y):
    """
    Computes the inner product of two vectors x and y.
    :param Nvec: n_diag from PTA
    :param T: PTA get_basis
    :param cf: sigma inverse calculated using Cholesky
    :param x:left vector
    :param y:right vector
    :returns:
    Inner product, xCy
    """
    TNx = Nvec.solve(x, left_array=T)
    TNy = Nvec.solve(y, left_array=T)
    xNy = Nvec.solve(y, left_array=x)

    expval = sl.cho_solve(cf, TNy)
    return xNy - TNx @ expval


def make_tau_scan_map(TauScan, n_tau=5, f_min=None, f_max=None, t_min=None, t_max=None, tau_min=None, tau_max=None, tref=53000*86400):
    """
        Produce Tau-scan 3D (tau, t0, f0) map

        :param TauScan: TauScan object used to calculate TauScan at given values of tau, t0, f0
        :param n_tau: number of tau_slices to do
        :param f_min: minimum frequency [Hz]
        :param f_max: maximum frequency [Hz]
        :param t_min: minimum time [year]
        :param t_max: maximum time [year]
        :param tau_min: minimum tau [year]
        :param tau_max: maximum tau [year]
        :return: dictionary with the following entries:
                    'tau_scan': the actual tau scan as a list (for each tau) of 2D (t,f) numpy arrays
                    'tau_edges': numpy array with the edges of the tau binning used [year]
                    't0_edges': list (for different taus) of numpy arrays with the edges of the t0 binning used [s]
                    'f0_edges': list (for different taus) of numpy arrays with the edges of the f0 binning used [Hz]
        """

    #check if all prior boundaries are provided
    if (f_min is None) or (f_max is None) or (t_min is None) or (t_max is None) or (tau_min is None) or (tau_max is None):
        raise Exception("All 6 boundaries (max and min for f0, t0, tau) are needed to compute tau-scan map")

    #calculate tau bin boundaries to use
    tau_edges = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau+1)
    dtau = tau_edges[1]/tau_edges[0]

    #calculate bin centers
    taus = []
    for j in range(tau_edges.size-1):
        taus.append(tau_edges[j]*np.sqrt(dtau))

    taus = np.array(taus)*24*3600*365.25

    tau_scan = []
    T0_list = []
    F0_list = []
    #loop over different taus
    for k, tau in enumerate(taus):
        print(k+1, " / ", taus.size)

        #resolution needed to get some overlap between pixels (TODO: check what value was used for these)
        f_res = 1 / (np.sqrt(5)*np.pi*tau)
        N_f = int((f_max-f_min)/f_res)
        t_res = tau / np.sqrt(5)
        N_t = int((t_max-t_min)*365.25*24*3600/t_res)

        print(N_f)
        print(N_t)
        print("SUM: ", N_f*N_t)
        f0_edges = np.linspace(f_min, f_max, N_f+1)
        t0_edges = np.linspace(t_min, t_max, N_t+1)*365.25*24*3600
        T0_list.append(t0_edges)
        F0_list.append(f0_edges)

        #Calculate bin centers from bin edges
        f0s = []
        for i in range(f0_edges.size - 1):
            f0s.append( (f0_edges[i] + f0_edges[i+1])/2 )

        t0s = []
        for i in range(t0_edges.size - 1):
            t0s.append( (t0_edges[i] + t0_edges[i+1])/2 )

        #Loop over pixels and calculate tau scan map
        TS = np.zeros((N_f, N_t))
        for i, f0 in enumerate(f0s):
            for j, t0 in enumerate(t0s):
                #COS, SIN = TauScan.compute_TauScan(tau, t0, f0)
                #TS[i,j] = COS**2 + SIN**2
                TS[i,j] = TauScan.compute_TauScan(tau, t0, f0, tref=tref)
        tau_scan.append(TS)

    return {'tau_scan':tau_scan, 'tau_edges':tau_edges, 't0_edges':T0_list, 'f0_edges':F0_list}
