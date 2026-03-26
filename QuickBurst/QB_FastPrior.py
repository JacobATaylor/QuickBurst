"""
C 2024 Jacob Taylor, Rand Burnette, and Bence Becsy fast Burst prior object

MCMC to utilize faster generic GW burst search likelihood.

"""

import numpy as np
#np.seterr(all='raise')
import numba as nb
from numba import njit, prange
from numba.experimental import jitclass
from numba.typed import List
from QuickBurst import QuickBurst_lnlike as qb_like
################################################################################
#
#MY VERSION OF GETTING THE LOG PRIOR
#
################################################################################
class FastPrior:
    """helper class to set up information about priors"""
    def __init__(self, pta, psrs):
        """pta is an enterprise pta, and psrs is a list of pulsar objects."""
        self.pta = pta
        self.param_names = List(pta.param_names)
        uniform_pars = []
        uf_lows = []
        uf_highs = []
        lin_exp_pars = []
        le_lows = []
        le_highs = []
        normal_pars = []
        nm_mus = []
        nm_sigs = []
        dm_pars = []
        dm_dists = []
        dm_errs = []
        dm_dist_lows = []
        px_pars = []
        px_mus = []
        px_errs = []
        px_dist_lows = []
        #track parameters that are normal and distance so we can apply a cutoff to them
        normal_dist_pars = []
        nm_dist_lows = []

        #signal prior bounds
        wave_uf_lows = []
        wave_uf_highs = []
        wave_le_lows = []
        wave_le_highs = []
        wave_uniform_pars = []
        wave_lin_exp_pars = []

        #transient prior bounds
        glitch_uf_lows = []
        glitch_uf_highs = []
        glitch_le_lows = []
        glitch_le_highs = []
        glitch_uniform_pars = []
        glitch_lin_exp_pars = []


        for idx, par in enumerate(self.pta.params):
            #Special treatment for glitch priors
            if 'Glitch' in str(par):
                if "Uniform" in par._typename:
                    glitch_uniform_pars.append(par.name)
                    glitch_uf_lows.append(float(par._typename.split('=')[1].split(',')[0]))
                    glitch_uf_highs.append(float(par._typename.split('=')[2][:-1]))
                elif "LinearExp" in par._typename:
                    glitch_lin_exp_pars.append(par.name)
                    glitch_le_lows.append(float(par._typename.split('=')[1].split(',')[0]))
                    glitch_le_highs.append(float(par._typename.split('=')[2][:-1]))

            #Special treatment for wavelet priors
            elif 'wavelet' in str(par):
                if "Uniform" in par._typename:
                    wave_uniform_pars.append(par.name)
                    wave_uf_lows.append(float(par._typename.split('=')[1].split(',')[0]))
                    wave_uf_highs.append(float(par._typename.split('=')[2][:-1]))
                elif "LinearExp" in par._typename:
                    wave_lin_exp_pars.append(par.name)
                    wave_le_lows.append(float(par._typename.split('=')[1].split(',')[0]))
                    wave_le_highs.append(float(par._typename.split('=')[2][:-1]))
            else:
                if "Uniform" in par._typename:
                    uniform_pars.append(par.name)
                    uf_lows.append(float(par._typename.split('=')[1].split(',')[0]))
                    uf_highs.append(float(par._typename.split('=')[2][:-1]))
                elif "LinearExp" in par._typename:
                    lin_exp_pars.append(par.name)
                    le_lows.append(float(par._typename.split('=')[1].split(',')[0]))
                    le_highs.append(float(par._typename.split('=')[2][:-1]))
                elif "Normal" in par._typename:
                    normal_pars.append(par.name)
                    nm_mus.append(float(par._typename.split('=')[1].split(',')[0]))
                    nm_sigs.append(float(par._typename.split('=')[2][:-1]))
                    if "_cw0_p_dist" in par.name:
                        #find the corresponding pulsar distance so we can append it
                        normal_dist_pars.append(par.name)
                        for psr in psrs:
                            if psr.name in par.name:
                                #this should be the lower cutoff of dist_delta such that dist_mu + dist_sigma*dist_delta=0
                                nm_dist_lows.append(-psr.pdist[0]/psr.pdist[1])
                                break
                elif "DMDist" in par._typename:
                    dm_pars.append(par.name)
                    dm_dists.append(float(par._typename.split('=')[1].split(',')[0]))
                    dm_errs.append(float(par._typename.split('=')[2][:-1]))
                    dm_dist_lows.append(0.0)
                elif "PXDist" in par._typename:
                    px_pars.append(par.name)
                    px_dist = float(par._typename.split('=')[1].split(',')[0])
                    px_dist_err = float(par._typename.split('=')[2][:-1])
                    px_mus.append(1/px_dist)
                    px_errs.append(px_dist_err/px_dist**2)
                    px_dist_lows.append(0.0)


        self.uniform_lows = np.array(uf_lows)
        self.uniform_highs = np.array(uf_highs)
        self.lin_exp_lows = np.array(le_lows)
        self.lin_exp_highs = np.array(le_highs)
        self.normal_mus = np.array(nm_mus)
        self.normal_sigs = np.array(nm_sigs)
        self.dm_dists = np.array(dm_dists)
        self.dm_errs = np.array(dm_errs)
        self.px_mus = np.array(px_mus)
        self.px_errs = np.array(px_errs)
        self.uniform_par_ids = np.array([self.param_names.index(u_par) for u_par in uniform_pars], dtype='int')
        self.lin_exp_par_ids = np.array([self.param_names.index(l_par) for l_par in lin_exp_pars], dtype='int')
        self.normal_par_ids = np.array([self.param_names.index(n_par) for n_par in normal_pars], dtype='int')
        self.dm_par_ids = np.array([self.param_names.index(dm_par) for dm_par in dm_pars], dtype='int')
        self.px_par_ids = np.array([self.param_names.index(px_par) for px_par in px_pars], dtype='int')

        self.glitch_uf_lows = np.array(glitch_uf_lows)
        self.glitch_uf_highs = np.array(glitch_uf_highs)
        self.glitch_le_lows = np.array(glitch_le_lows)
        self.glitch_le_highs = np.array(glitch_le_highs)

        self.glitch_uf_par_ids = np.array([self.param_names.index(u_par) for u_par in glitch_uniform_pars], dtype = 'int')
        self.glitch_le_par_ids = np.array([self.param_names.index(le) for le in glitch_lin_exp_pars], dtype = 'int')

        self.wave_uf_lows = np.array(wave_uf_lows)
        self.wave_uf_highs = np.array(wave_uf_highs)
        self.wave_le_lows = np.array(wave_le_lows)
        self.wave_le_highs = np.array(wave_le_highs)

        self.wave_uf_par_ids = np.array([self.param_names.index(u_par) for u_par in wave_uniform_pars], dtype = 'int')
        self.wave_le_par_ids = np.array([self.param_names.index(le) for le in wave_lin_exp_pars], dtype = 'int')

        #logic for cutting off normally distributed distances so they don't go below 0
        self.normal_dist_par_ids = np.array([self.param_names.index(n_par) for n_par in normal_dist_pars], dtype='int')
        self.normal_dist_lows = np.array(nm_dist_lows)
        self.normal_dist_highs = np.full(self.normal_dist_lows.size,np.inf)

        self.dm_dist_lows = np.array(dm_dist_lows)
        self.dm_dist_highs = np.full(self.dm_dist_lows.size,np.inf)
        self.px_dist_lows = np.array(px_dist_lows)
        self.px_dist_highs = np.full(self.px_dist_lows.size,np.inf)

        self.cut_lows = np.hstack([self.uniform_lows,self.lin_exp_lows,self.normal_dist_lows,self.dm_dist_lows,self.px_dist_lows])
        self.cut_highs = np.hstack([self.uniform_highs,self.lin_exp_highs,self.normal_dist_highs,self.dm_dist_highs,self.px_dist_highs])
        self.cut_par_ids = np.hstack([self.uniform_par_ids,self.lin_exp_par_ids,self.normal_dist_par_ids,self.dm_par_ids,self.px_par_ids])

        #uniform prior is independent of value
        self.global_uniform = 0.
        for itrp in range(self.uniform_lows.size):
            low = self.uniform_lows[itrp]
            high = self.uniform_highs[itrp]
            self.global_uniform += -np.log(high-low)

        #linear exponential prior has component independent of value
        self.global_lin_exp = 0.
        for itrp in range(self.lin_exp_lows.size):
            low = self.lin_exp_lows[itrp]
            high = self.lin_exp_highs[itrp]
            self.global_lin_exp += np.log(np.log(10))-np.log(10 ** high - 10 ** low)


        #normal prior has component independent of value
        self.global_normal = 0.
        for itrp in range(self.normal_mus.size):
            self.global_normal += -np.log(2*np.pi)/2.

        #dm prior has component independent of the value
        self.global_dm = 0
        for itrp in range(self.dm_dists.size):
            dist = self.dm_dists[itrp]
            err = self.dm_errs[itrp]

            boxheight = 1/((dist+err)-(dist-err))
            gaussheight = 1/(np.sqrt(2*np.pi*(0.25*err)**2))

            area = 1+1*boxheight/gaussheight

            self.global_dm += np.log(boxheight/area)

        #part of the likelihood that is the same independent of the parameter values for all points with finite log prior
        self.global_common = self.global_uniform+self.global_lin_exp+self.global_normal+self.global_dm

    def get_lnprior(self, x0):
        """wrapper to get ln prior"""
        lprior = get_lnprior_helper(x0, self.uniform_par_ids, self.uniform_lows, self.uniform_highs,\
                                      self.lin_exp_par_ids, self.lin_exp_lows, self.lin_exp_highs,\
                                      self.normal_par_ids, self.normal_mus, self.normal_sigs,\
                                      self.dm_par_ids, self.dm_dists, self.dm_errs,\
                                      self.px_par_ids, self.px_mus, self.px_errs,\
                                      self.global_common)
        #self.param_rejections = reject_rates
        return lprior

    def get_sample(self, idx):
        """wrapper to quickly return random prior draw for the (idx)th parameter"""
        return get_sample_helper(idx, self.uniform_par_ids, self.uniform_lows, self.uniform_highs,\
                                      self.lin_exp_par_ids, self.lin_exp_lows, self.lin_exp_highs,\
                                      self.normal_par_ids, self.normal_mus, self.normal_sigs,\
                                      self.dm_par_ids, self.dm_dists, self.dm_errs,\
                                      self.px_par_ids, self.px_mus, self.px_errs)



@njit()
def get_sample_helper_full(n_par,uniform_par_ids, uniform_lows, uniform_highs,
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,
                           normal_par_ids, normal_mus, normal_sigs,
                           dm_par_ids, dm_dists, dm_errs,
                           px_par_ids, px_mus, px_errs):
    """jittable helper for prior draws"""
    res = np.zeros(n_par)
    for itrp,idx in enumerate(uniform_par_ids):
        res[idx] = np.random.uniform(uniform_lows[itrp], uniform_highs[itrp])
    for itrp,idx in enumerate(lin_exp_par_ids):
        res[idx] = np.log10(np.random.uniform(10**lin_exp_lows[itrp], 10**lin_exp_highs[itrp]))
    for itrp,idx in enumerate(normal_par_ids):
        res[idx] = np.random.normal(normal_mus[itrp], normal_sigs[itrp])
    for itrp,idx in enumerate(dm_par_ids):
        boxheight = 1/((dm_dists[itrp]+dm_errs[itrp])-(dm_dists[itrp]-dm_errs[itrp]))
        gaussheight = 1/(np.sqrt(2*np.pi*(0.25*dm_errs[itrp])**2))
        area = 1+1*boxheight/gaussheight

        #probability of being in the uniform part
        boxprob = 1/area

        #decide if we are in the middle or not
        alpha = np.random.uniform(0.0,1.0)
        if alpha<boxprob:
            res[idx] = np.random.uniform(dm_dists[itrp]-dm_errs[itrp], dm_dists[itrp]+dm_errs[itrp])
        else:
            x = np.random.normal(0, 0.25*dm_errs[itrp])
            if x>0.0:
                res[idx] = x+dm_dists[itrp]+dm_errs[itrp]
            else:
                res[idx] = x+dm_dists[itrp]-dm_errs[itrp]
    for itrp,idx in enumerate(px_par_ids):
        res[idx] = 1/np.random.normal(px_mus[itrp], px_errs[itrp])
    return res

def get_sample_helper(idx, uniform_par_ids, uniform_lows, uniform_highs,
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,
                           normal_par_ids, normal_mus, normal_sigs,
                           dm_par_ids, dm_dists, dm_errs,
                           px_par_ids, px_mus, px_errs):
    """jittable helper for prior draws"""
    if idx in uniform_par_ids:
        iii = np.argmax(uniform_par_ids==idx)
        return np.random.uniform(uniform_lows[iii], uniform_highs[iii])
    elif idx in lin_exp_par_ids:
        iii = np.argmax(lin_exp_par_ids==idx)
        return np.log10(np.random.uniform(10**lin_exp_lows[iii], 10**lin_exp_highs[iii]))
    elif idx in normal_par_ids:
        iii = np.argmax(normal_par_ids==idx)
        return np.random.normal(normal_mus[iii], normal_sigs[iii])
    elif idx in dm_par_ids:
        iii = np.argmax(dm_par_ids==idx)
        boxheight = 1/((dm_dists[iii]+dm_errs[iii])-(dm_dists[iii]-dm_errs[iii]))
        gaussheight = 1/(np.sqrt(2*np.pi*(0.25*dm_errs[iii])**2))
        area = 1+1*boxheight/gaussheight

        #probability of being in the uniform part
        boxprob = 1/area

        #decide if we are in the middle or not
        alpha = np.random.uniform(0.0,1.0)
        if alpha<boxprob:
            return np.random.uniform(dm_dists[iii]-dm_errs[iii], dm_dists[iii]+dm_errs[iii])
        else:
            x = np.random.normal(0, 0.25*dm_errs[iii])
            if x>0.0:
                return x+dm_dists[iii]+dm_errs[iii]
            else:
                return x+dm_dists[iii]-dm_errs[iii]
    else:
        iii = np.argmax(px_par_ids==idx)
        return 1/np.random.normal(px_mus[iii], px_errs[iii])

@njit()
def get_lnprior_helper(x0, uniform_par_ids, uniform_lows, uniform_highs,\
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,\
                           normal_par_ids, normal_mus, normal_sigs,\
                           dm_par_ids, dm_dists, dm_errs,px_par_ids, \
                           px_mus, px_errs, global_common, \
                           glitch_uf_par_ids, glitch_uf_lows, \
                           glitch_uf_highs, glitch_le_par_ids, \
                           glitch_le_lows, glitch_le_highs,\
                           wave_uf_par_ids, wave_uf_lows, \
                           wave_uf_highs, wave_le_par_ids, \
                           wave_le_lows, wave_le_highs, \
                           n_wavelet, n_glitch, max_n_wavelet, max_n_glitch):

    """jittable helper for calculating the log prior"""
    log_prior = global_common
    #loop through uniform parameters and make sure all are in range
    n = uniform_par_ids.size
    for itrp in range(n):
        low = uniform_lows[itrp]
        high = uniform_highs[itrp]
        par_id = uniform_par_ids[itrp]
        value = x0[par_id]
        if low>value or value>high:
            log_prior = -np.inf

    ##############
    #For checking glitch uniform priors are in prior range
    g = glitch_uf_par_ids.size
    for itrp in range(g):
        low = glitch_uf_lows[itrp]
        high = glitch_uf_highs[itrp]
        par_id = glitch_uf_par_ids[itrp]
        value = x0[par_id]
        if low>value or value>high:
            log_prior = -np.inf
    #For checking wavelet uniform priors are in prior range
    w = wave_uf_par_ids.size
    for itrp in range(w):
        low = wave_uf_lows[itrp]
        high = wave_uf_highs[itrp]
        par_id = wave_uf_par_ids[itrp]
        value = x0[par_id]
        if low>value or value>high:
            log_prior = -np.inf
    #for uniform prior wavelet/glitch contributions
    #Calculate global contribution for a single set of uniform glitch params
    glitch_global_uniform = 0.
    if max_n_glitch != 0:
        for itrp in range(int(glitch_uf_lows.size/max_n_glitch)):
            low = glitch_uf_lows[itrp]
            high = glitch_uf_highs[itrp]
            glitch_global_uniform += -np.log(high-low)
            #Count contribution for all glitches
        log_prior += glitch_global_uniform*n_glitch

    #Calculate global contribution for a single set of uniform wavelet params
    wave_global_uniform = 0.
    if max_n_wavelet != 0:
        for itrp in range(int(wave_uf_lows.size/max_n_wavelet)):
            low = wave_uf_lows[itrp]
            high = wave_uf_highs[itrp]
            wave_global_uniform += -np.log(high-low)
        #Count contribution for all wavelets
        log_prior += wave_global_uniform*n_wavelet

    #for linear exponential wavelet/glitch prior contributions
    #Calculate global contribution for all lin-exp glitch params (if any)
    if max_n_glitch != 0:
        for itrp in range(int(glitch_le_lows.size/max_n_glitch)*n_glitch):
            low = glitch_le_lows[itrp]
            high = glitch_le_highs[itrp]
            par_id = glitch_le_par_ids[itrp]
            value = x0[par_id]
            if low>value or value>high:
                log_prior = -np.inf #from enterprise

            else:
                log_prior += value*np.log(10)#from enterprise

    #Calculate global contribution for all lin-exp wavelet params (if any)
    if max_n_wavelet != 0:
        for itrp in range(int(wave_le_lows.size/max_n_wavelet)*n_wavelet):
            low = wave_le_lows[itrp]
            high = wave_le_highs[itrp]
            par_id = wave_le_par_ids[itrp]
            value = x0[par_id]
            if low>value or value>high:
                log_prior = -np.inf #from enterprise
            else:
                log_prior += value*np.log(10)#from enterprise

    #linear exponential prior has component independent of value
    glitch_global_lin_exp = 0.
    if max_n_glitch != 0:
        for itrp in range(int(glitch_le_lows.size/max_n_glitch)):
            low = glitch_le_lows[itrp]
            high = glitch_le_highs[itrp]
            glitch_global_lin_exp += np.log(np.log(10))-np.log(10 ** high - 10 ** low)
        log_prior += glitch_global_lin_exp*n_glitch

    #linear exponential prior has component independent of value
    wave_global_lin_exp = 0.
    if max_n_wavelet != 0:
        for itrp in range(int(wave_le_lows.size/max_n_wavelet)):
            low = wave_le_lows[itrp]
            high = wave_le_highs[itrp]
            wave_global_lin_exp += np.log(np.log(10))-np.log(10 ** high - 10 ** low)
        log_prior += wave_global_lin_exp*n_wavelet
    ################

    #loop through all other linear exponential parameters
    nn = lin_exp_par_ids.size
    for itrp in range(nn):
        low = lin_exp_lows[itrp]
        high = lin_exp_highs[itrp]
        par_id = lin_exp_par_ids[itrp]
        value = x0[par_id]
        if low>value or value>high:
            log_prior = -np.inf #from enterprise
        else:
            log_prior += value*np.log(10)#from enterprise

    #loop through normal parameters
    m = normal_par_ids.size
    for itrp in range(m):
        mu = normal_mus[itrp]
        sig = normal_sigs[itrp]
        par_id = normal_par_ids[itrp]
        value = x0[par_id]
        log_prior += -(value-mu)**2/(2*sig**2) #log_pdf_got

    #loop through dm distance parameters
    l = dm_par_ids.size
    for itrp in range(l):
        dist = dm_dists[itrp]
        err = dm_errs[itrp]
        par_id = dm_par_ids[itrp]

        value = x0[par_id]

        if value<=(dist-err):
            log_prior += -(value-(dist-err))**2/(2*(0.25*err)**2)
        elif value>=(dist+err):
            log_prior += -(value-(dist+err))**2/(2*(0.25*err)**2)

    #loop trhough parallax distance parameters
    ll = px_par_ids.size
    for itrp in range(ll):
        pi = px_mus[itrp]
        pi_err = px_errs[itrp]
        par_id = px_par_ids[itrp]

        value = x0[par_id]

        log_prior += np.log(1/(np.sqrt(2*np.pi)*pi_err*value**2)*np.exp(-(pi-value**(-1))**2/(2*pi_err**2)))


    return log_prior#, param_rejections

##################
# Method for computing snr prior value for individual signals
##################
@njit(fastmath=True, parallel=False)
def calculate_snr_prior_value(rho, wave_rho_star = 5, glitch_rho_star = 3, signal_type='wavelet'):

    denom_wave_const = 4.0 * (wave_rho_star**2)  
    denom_glitch_const = 2.0 * (glitch_rho_star**2) 

    log_p = 0.0
    if signal_type=='wavelet':
        #compute log snr value
                # Minimal numeric checks: rho must be finite and >= 0
        if (not np.isfinite(rho)) or (rho < 0.0):
            return -np.inf


        # inner term for wavelet prior: (1 + rho/(4*rho_star))
        inner_wave = 1.0 + (rho / (4.0 * wave_rho_star))
        if (not np.isfinite(inner_wave)) or (inner_wave <= 0.0):
            return -np.inf

        log_p = np.log(3.0) + np.log(rho) - np.log(denom_wave_const) - 5.0 * np.log(inner_wave)

        if not np.isfinite(log_p):
            return -np.inf
    
    if signal_type=='glitch':
        #compute log snr value
            # Minimal numeric checks
        if (not np.isfinite(rho)) or (rho < 0.0):
            return -np.inf

        # inner term for glitch prior: (1 + rho/(2*rho_star))
        inner_glitch = 1.0 + (rho / (2.0 * glitch_rho_star))
        if (not np.isfinite(inner_glitch)) or (inner_glitch <= 0.0):
            return -np.inf

        log_p = np.log(rho) - np.log(denom_glitch_const) - 3.0 * np.log(inner_glitch)

        if not np.isfinite(log_p):
            return -np.inf
    return log_p


##############################################
# Method for drawing from signal snr prior with rejection sampling
##############################################
@njit(fastmath=True, parallel=False)
def compute_signal_snr_prior(new_point, glitch_indx, wavelet_indx, n_wavelet, n_glitch, wavelet_iter, FPI, wavelet_amp_prior, wavelet_log_amp_range, 
                          toas, residuals, pos, sigmas, Npsr, MMs, NN, invTN, CholSigma, 
                          Ndiag_array, wavelet_prm, glitch_prm, glitch_pulsars, 
                          glitch_pulsars_previous, projection_step = False):
    #Set prior constants
    wave_rho_star   = 5.0
    denom_wave_const = 4.0 * (wave_rho_star**2)  
    log_snr_prior = 0.0
    
    #Draw signal snr and new point
    rho = sample_signal_snr(new_point, wave_rho_star, glitch_indx, wavelet_indx, n_wavelet, n_glitch, wavelet_iter, FPI,
                wavelet_amp_prior, wavelet_log_amp_range, toas,
                residuals, Npsr, pos, sigmas, MMs, NN, invTN, CholSigma, Ndiag_array,
                wavelet_prm, glitch_prm, glitch_pulsars, glitch_pulsars_previous,
                projection_step = projection_step)

    
    # Minimal numeric checks: rho must be finite and >= 0
    if (not np.isfinite(rho)) or (rho < 0.0):
        return rho, -np.inf


    # inner term for wavelet prior: (1 + rho/(4*rho_star))
    inner_wave = 1.0 + (rho / (4.0 * wave_rho_star))
    if (not np.isfinite(inner_wave)) or (inner_wave <= 0.0):
        return rho, -np.inf

    log_snr_prior = np.log(3.0) + np.log(rho) - np.log(denom_wave_const) - 5.0 * np.log(inner_wave)

    if not np.isfinite(log_snr_prior):
        return rho, -np.inf

    return rho, log_snr_prior

@njit(fastmath=True, parallel=False)
def compute_glitch_snr_prior(new_point, glitch_indx, wavelet_indx, n_wavelet, n_glitch, glitch_iter,glitch_amp_prior, glitch_log_amp_range, 
                    toas, residuals, Npsr, pos, sigmas, MMs, NN, wavelet_prm, glitch_prm,
                    glitch_pulsars, glitch_pulsars_previous, invTN, CholSigma, 
                    Ndiag, projection_step=False, prior_recovery=False):
    
    glitch_rho_star = 3.0
    log_p_glitch = 0.0
    denom_glitch_const = 2.0 * (glitch_rho_star**2) 

    rho = sample_glitch_snr(new_point, glitch_rho_star, glitch_indx, wavelet_indx, n_wavelet, n_glitch, glitch_iter, glitch_amp_prior,  glitch_log_amp_range, 
                    toas, residuals, Npsr, pos, sigmas, MMs, NN, wavelet_prm, glitch_prm, 
                    glitch_pulsars, glitch_pulsars_previous, invTN, CholSigma, Ndiag, 
                    projection_step=projection_step)

    # Minimal numeric checks
    if (not np.isfinite(rho)) or (rho <= 0.0):
        return rho, -np.inf

    # inner term for glitch prior: (1 + rho/(2*rho_star))
    inner_glitch = 1.0 + (rho / (2.0 * glitch_rho_star))
    if (not np.isfinite(inner_glitch)) or (inner_glitch <= 0.0):
        return rho, -np.inf

    log_p_glitch = np.log(rho) - np.log(denom_glitch_const) - 3.0 * np.log(inner_glitch)

    if not np.isfinite(log_p_glitch):
        return rho, -np.inf


    return rho, log_p_glitch

##############################################
# Method for drawing glitch SNR and amplitude
##############################################
@njit(fastmath={'reassoc': True, 'nsz': True, 'arcp': True, 'contract': True, 'afn': True}, parallel=False)
def sample_glitch_snr(new_point, SNRpeak, glitch_indx, wavelet_indx, n_wavelet, n_glitch, glitch_iter, glitch_amp_prior, glitch_log_amp_range, 
                    toas, residuals, Npsr, pos, sigmas, MMs, NN, wavelet_prm, glitch_prm, glitch_pulsars,
                    glitch_pulsars_previous, invTN, CholSigma, Ndiag,
                    projection_step = False):
    
    """
    Method for drawing joint amplitude and transient snr proposls. 
    
    """

    #Update psr_idx in glitch_prm and glitch_pulsars with updated index from glitch jump
    glitch_prm[glitch_iter, 3] = new_point[int(glitch_indx[glitch_iter, 3])]
    glitch_pulsars[glitch_iter] = round(new_point[int(glitch_indx[glitch_iter, 3])])


    # Pre-calculate constants for rejection sampling
    SNR4 = 4.0 * SNRpeak
    SNRsq = 4.0 * SNRpeak * SNRpeak
    
    dfac = 1.0 + SNRpeak / SNR4
    dfac5 = dfac * dfac * dfac * dfac * dfac
    max_val = (3.0 * SNRpeak) / (SNRsq * dfac5)
    invmax = 1.0 / max_val

    #update parameter vector
    glitch_prm, wavelet_prm = qb_like.get_parameters(new_point, glitch_prm, wavelet_prm, glitch_indx, 
                                                     wavelet_indx, n_glitch, n_wavelet)

    #Get new coefficients
    coeffs = qb_like.get_sigmas_helper(pos, sigmas, glitch_pulsars, Npsr, n_wavelet, n_glitch, wavelet_prm, glitch_prm)
    
    #If projection step, skip updating M matrix
    if not projection_step:
        dif_flag = np.zeros((n_wavelet + n_glitch))

        #Set current transient being updated to 1
        dif_flag[n_wavelet + glitch_iter] = 1.0
        
        _, MMs = qb_like.get_M_N(toas, residuals, Npsr, MMs, NN, invTN, CholSigma, Ndiag, n_wavelet, 
                               n_glitch, wavelet_prm, glitch_prm, glitch_pulsars, glitch_pulsars_previous, dif_flag)

    SNR = compute_glitch_snr(coeffs, round(new_point[int(glitch_indx[glitch_iter, 3])]), MMs, n_wavelet, glitch_iter)

    dfac = 1.0 + SNR / SNR4
    dfac5 = dfac * dfac * dfac * dfac * dfac
    den = (3.0 * SNR) / (SNRsq * dfac5)
    den *= invmax

    #Rejection sampling
    alpha = np.random.random()
    k = 0
    while alpha > den:
        
        if glitch_amp_prior == 'uniform':
            #Draw initial new amplitude
            new_h = np.log10(np.random.uniform(low=10**glitch_log_amp_range[0], 
                                        high=10**glitch_log_amp_range[1]))
            new_point[glitch_indx[glitch_iter, 1]] = new_h

        if glitch_amp_prior == 'log-uniform':
            #Draw initial new amplitude
            new_h = np.random.uniform(glitch_log_amp_range[0], glitch_log_amp_range[1])
            new_point[glitch_indx[glitch_iter, 1]] = new_h

        #update parameter vector
        glitch_prm[glitch_iter, 1] = 10**(new_h)

        #Get new coefficients
        coeffs = qb_like.get_sigmas_helper(pos, sigmas, glitch_pulsars, Npsr, n_wavelet, n_glitch, wavelet_prm, glitch_prm)
        
        SNR = compute_glitch_snr(coeffs, round(new_point[int(glitch_indx[glitch_iter, 3])]), MMs, n_wavelet, glitch_iter)
        
        dfac = 1.0 + SNR / SNR4
        dfac5 = dfac * dfac * dfac * dfac * dfac
        den = (3.0 * SNR) / (SNRsq * dfac5)
        den *= invmax
        alpha = np.random.uniform()
        
        k += 1
        
        # Escape hatch if rejection sampling takes too long
        if k > 10000:
            SNR = 0.0
            break
    return SNR

###############################################
# Method for drawing wavelet SNR and amplitudes
###############################################

@njit(fastmath=True, parallel=False)
def sample_signal_snr(new_point, SNRpeak, glitch_indx, wavelet_indx, n_wavelet, n_glitch, wavelet_iter, FPI, wavelet_amp_prior, wavelet_log_amp_range, toas, residuals, 
                    Npsr, pos, sigmas, MMs, NN, invTN, CholSigma, Ndiag, wavelet_prm,
                    glitch_prm, glitch_pulsars, glitch_pulsars_previous, projection_step = False):
    """
    Method for drawing joint amplitude and GW burst wavelet snr proposls. 
    """

    # Pre-calculate constants for rejection sampling
    SNR4 = 4.0 * SNRpeak
    SNRsq = 4.0 * SNRpeak * SNRpeak
    
    dfac = 1.0 + SNRpeak / SNR4
    dfac5 = dfac * dfac * dfac * dfac * dfac
    max_val = (3.0 * SNRpeak) / (SNRsq * dfac5)
    invmax = 1.0 / max_val

    #update parameter vector
    glitch_prm, wavelet_prm = qb_like.get_parameters(new_point, glitch_prm, wavelet_prm, glitch_indx, 
                                                     wavelet_indx, n_glitch, n_wavelet)
    
    #Get new coefficients

    coeffs = qb_like.get_sigmas_helper(pos, sigmas, glitch_pulsars, Npsr, n_wavelet, n_glitch, wavelet_prm, glitch_prm)

    #If projection step, skip updating M matrix
    if not projection_step:
        dif_flag = np.zeros((n_wavelet + n_glitch))

        #Set current transient being updated to 1
        dif_flag[wavelet_iter] = 1.0
        _, MMs = qb_like.get_M_N(toas, residuals, Npsr, MMs, NN, invTN, CholSigma, Ndiag, n_wavelet, 
                               n_glitch, wavelet_prm, glitch_prm, glitch_pulsars, glitch_pulsars_previous, dif_flag)
    
    SNR = compute_signal_snr(coeffs, MMs, wavelet_iter)

    dfac = 1.0 + SNR / SNR4
    dfac5 = dfac * dfac * dfac * dfac * dfac
    den = (3.0 * SNR) / (SNRsq * dfac5)
    den *= invmax
            
    alpha = np.random.random()
    k = 0

    while alpha > den:
        
        if wavelet_amp_prior == 'uniform':
            #Draw initial new amplitude
            new_h_plus = np.log10(np.random.uniform(low=10**wavelet_log_amp_range[0], 
                            high=10**wavelet_log_amp_range[1]))
            new_h_cross = np.log10(np.random.uniform(low=10**wavelet_log_amp_range[0], 
                                        high=10**wavelet_log_amp_range[1]))
            new_point[int(wavelet_indx[wavelet_iter, 4])] = new_h_plus
            new_point[int(wavelet_indx[wavelet_iter, 5])] = new_h_cross

        if wavelet_amp_prior == 'log-uniform':
            #Draw initial new amplitude
            new_h_plus = np.random.uniform(wavelet_log_amp_range[0], wavelet_log_amp_range[1])
            new_h_cross = np.random.uniform(wavelet_log_amp_range[0], wavelet_log_amp_range[1])
            new_point[int(wavelet_indx[wavelet_iter, 4])] = new_h_plus
            new_point[int(wavelet_indx[wavelet_iter, 5])] = new_h_cross

        #update parameter vector
        glitch_prm, wavelet_prm = qb_like.get_parameters(new_point, glitch_prm, wavelet_prm, 
                                                         glitch_indx, wavelet_indx, n_glitch, n_wavelet)

        #Get new coefficients
        coeffs = qb_like.get_sigmas_helper(pos, coeffs, glitch_pulsars, Npsr, n_wavelet, n_glitch, wavelet_prm, glitch_prm)

        SNR = compute_signal_snr(coeffs, MMs, wavelet_iter)

        dfac = 1.0 + SNR / SNR4
        dfac5 = dfac * dfac * dfac * dfac * dfac
        den = (3.0 * SNR) / (SNRsq * dfac5)
        den *= invmax
        
        alpha = np.random.uniform()
        
        k += 1
        # Escape hatch if rejection sampling takes too long
        if k > 10000:
            SNR = 0.0
            break

    return SNR#transient_snr, signal_snr, total_wave_snr

################################
# Method for computing total GW burst snr
################################
@njit(fastmath=True, parallel=False)
def compute_total_signal_snr(coeffs, M, Nwavelet):
    #Store total signal SNRs
    total_wave_snr = 0
    #Loop over Nwavelet
    signal_snrs = np.zeros((Nwavelet))
    if Nwavelet > 0:
        for k in range(Nwavelet):
            #Compute last term in equation 17 from QuickBurst paper
            temp_signal_snr = 0
            for l in range(Nwavelet):
                temp_signal_snr += np.sum(coeffs[:,k,0]*(coeffs[:,l,0]*M[:, 0+2*k, 0+2*l] + 
                                                            coeffs[:,l,1]*M[:, 0+2*k, 1+2*l]) + coeffs[:,k,1]*(
                                                            coeffs[:,l,0]*M[:, 1+2*k, 0+2*l] + 
                                                            coeffs[:,l,1]*M[:, 1+2*k, 1+2*l]))

            total_wave_snr += temp_signal_snr

        total_wave_snr = np.sqrt(total_wave_snr)

    return total_wave_snr

####################################
# Method for computing individual GW burst wavelet snr
####################################
@njit(fastmath=True, parallel=False)
def compute_signal_snr(coeffs, M, wavelet_iter):
    '''
    Jitted helper function to compute GW burst inner products.
    '''

    #glitch_pulsars has pulsar indexes for all wavelets up to n_glitch_max
    k = wavelet_iter
    #Compute last term in equation 17 from QuickBurst paper
    snr_sq = np.sum(coeffs[:,k,0]*(coeffs[:,k,0]*M[:, 0+2*k, 0+2*k] + 
                    coeffs[:,k,1]*M[:, 0+2*k, 1+2*k]) + coeffs[:,k,1]*(
                    coeffs[:,k,0]*M[:, 1+2*k, 0+2*k] + 
                    coeffs[:,k,1]*M[:, 1+2*k, 1+2*k])) 
    
    if snr_sq < 0.0:
        return 0.0

    return np.sqrt(snr_sq)

################################
# Method for computing glitch SNRs
################################
@njit(fastmath={'reassoc': True, 'nsz': True, 'arcp': True, 'contract': True, 'afn': True}, parallel=False)
def compute_glitch_snr(coeffs, psr_idx, M, n_wavelet, glitch_iter):
    '''
    Jitted helper function to compute noise transient inner products.
    '''
    
    #glitch_pulsars has pulsar indexes for all wavelets up to n_glitch_max
    #step over wavelets and glitches

    transient_snr = 0.0

    k_offset = n_wavelet + glitch_iter
    transient_snr += (
        coeffs[psr_idx, k_offset, 0] * (
            coeffs[psr_idx, k_offset, 0] * M[psr_idx, 0+2*k_offset, 0+2*k_offset] + 
            coeffs[psr_idx, k_offset, 1] * M[psr_idx, 0+2*k_offset, 1+2*k_offset]
        ) + 
        coeffs[psr_idx, k_offset, 1] * (
            coeffs[psr_idx, k_offset, 0] * M[psr_idx, 1+2*k_offset, 0+2*k_offset] + 
            coeffs[psr_idx, k_offset, 1] * M[psr_idx, 1+2*k_offset, 1+2*k_offset]
        )
    )

    if transient_snr < 0.0 or not np.isfinite(transient_snr):
        return 0.0
    return np.sqrt(transient_snr)
############ END OF SNR COMPUTATION


def get_lnprior(x0,FPI):
    
    """wrapper to get lnprior from jitted helper"""
    return get_lnprior_helper(x0, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                         FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                         FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                         FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,\
                                         FPI.px_par_ids, FPI.px_mus, FPI.px_errs,\
                                         FPI.global_common, FPI.glitch_uf_par_ids, \
                                         FPI.glitch_uf_lows, FPI.glitch_uf_highs, FPI.glitch_le_par_ids, \
                                         FPI.glitch_le_lows, FPI.glitch_le_highs, \
                                         FPI.wave_uf_par_ids, FPI.wave_uf_lows, \
                                         FPI.wave_uf_highs, FPI.wave_le_par_ids, \
                                         FPI.wave_le_lows, FPI.wave_le_highs, \
                                         FPI.n_wavelet, FPI.n_glitch, FPI.max_n_wavelet, FPI.max_n_glitch)



def get_lnprior_array(samples,FPI):
    """wrapper to get lnprior from jitted helper"""
    return get_lnprior_helper_array(samples, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                           FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                           FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                           FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,\
                                           FPI.px_par_ids, FPI.px_mus, FPI.px_errs,\
                                           FPI.global_common)

@njit()
def get_lnprior_helper_array(x0s, uniform_par_ids, uniform_lows, uniform_highs,\
                           lin_exp_par_ids, lin_exp_lows, lin_exp_highs,\
                           normal_par_ids, normal_mus, normal_sigs,\
                           dm_par_ids, dm_dists, dm_errs,\
                           px_par_ids, px_mus, px_errs,\
                           global_common): # n_wavelet, n_glitch,glitch_indx, wavelet_indx ):
    """jittable helper for calculating the log prior"""
    npoint = x0s.shape[0]

    log_priors = np.zeros(npoint)+global_common

    #loop through uniform parameters and make sure all are in range
    n = uniform_par_ids.size
    for itrp in range(n):
        low = uniform_lows[itrp]
        high = uniform_highs[itrp]
        par_id = uniform_par_ids[itrp]
        log_priors[(low>x0s[:,par_id])|(x0s[:,par_id]>high)] = -np.inf

    #loop through linear exponential parameters
    nn = lin_exp_par_ids.size
    for itrp in range(nn):
        low = lin_exp_lows[itrp]
        high = lin_exp_highs[itrp]
        par_id = lin_exp_par_ids[itrp]
        log_priors[:] += x0s[:,par_id]*np.log(10)#from enterprise
        log_priors[(low>x0s[:,par_id])|(x0s[:,par_id]>high)] = -np.inf

    #loop through normal parameters
    m = normal_par_ids.size
    for itrp in range(m):
        mu = normal_mus[itrp]
        sig = normal_sigs[itrp]
        par_id = normal_par_ids[itrp]
        log_priors[:] += -(x0s[:,par_id]-mu)**2/(2*sig**2) #log_pdf_got

    #loop through dm distance parameters
    l = dm_par_ids.size
    for itrp in range(l):
        dist = dm_dists[itrp]
        err = dm_errs[itrp]
        par_id = dm_par_ids[itrp]

        value = x0s[:,par_id]

        log_priors[value<=(dist-err)] += -(value-(dist-err))**2/(2*(0.25*err)**2)
        log_priors[value>=(dist+err)] += -(value-(dist+err))**2/(2*(0.25*err)**2)

    #loop trhough parallax distance parameters
    ll = px_par_ids.size
    for itrp in range(ll):
        pi = px_mus[itrp]
        pi_err = px_errs[itrp]
        par_id = px_par_ids[itrp]

        value = x0s[:,par_id]

        log_priors[:] += np.log(1/(np.sqrt(2*np.pi)*pi_err*value**2)*np.exp(-(pi-value**(-1))**2/(2*pi_err**2)))

    return log_priors

def get_sample_idxs(old_point,idx_choose,FPI):
    """get just some indexes reset for a uniform sample, actually just gets a whole new prior draw and picks the idxs needed"""
    new_point = old_point.copy()
    res = get_sample_full(new_point.size,FPI)
    for idx in idx_choose:
        new_point[idx] = res[idx]

    return new_point

def get_sample_full(n_par,FPI):
    """helper to get a sample with the specified indexes redrawn"""
    new_point = get_sample_helper_full(n_par, FPI.uniform_par_ids, FPI.uniform_lows, FPI.uniform_highs,\
                                              FPI.lin_exp_par_ids, FPI.lin_exp_lows, FPI.lin_exp_highs,\
                                              FPI.normal_par_ids, FPI.normal_mus, FPI.normal_sigs,\
                                              FPI.dm_par_ids, FPI.dm_dists, FPI.dm_errs,\
                                              FPI.px_par_ids, FPI.px_mus, FPI.px_errs)
    return new_point



@jitclass([('uniform_par_ids',nb.int64[:]),('uniform_lows',nb.float64[:]),('uniform_highs',nb.float64[:]),\
           ('lin_exp_par_ids',nb.int64[:]),('lin_exp_lows',nb.float64[:]),('lin_exp_highs',nb.float64[:]),\
           ('normal_par_ids',nb.int64[:]),('normal_mus',nb.float64[:]),('normal_sigs',nb.float64[:]),\
           ('dm_par_ids',nb.int64[:]),('dm_dists',nb.float64[:]),('dm_errs',nb.float64[:]),\
           ('px_par_ids',nb.int64[:]),('px_mus',nb.float64[:]),('px_errs',nb.float64[:]),\
           ('cut_par_ids',nb.int64[:]),('cut_lows',nb.float64[:]),('cut_highs',nb.float64[:]),\
           ('cw_ext_par_ids',nb.int64[:]),('cw_ext_lows',nb.float64[:]),('cw_ext_highs',nb.float64[:]),\
           ('global_common',nb.float64), ('glitch_uf_par_ids', nb.int64[:]), ('glitch_uf_lows', nb.float64[:]),\
           ('glitch_uf_highs', nb.float64[:]), ('glitch_le_par_ids', nb.int64[:]), ('glitch_le_lows', nb.float64[:]),\
           ('glitch_le_highs', nb.float64[:]), ('wave_uf_par_ids', nb.int64[:]), ('wave_uf_lows', nb.float64[:]),\
           ('wave_uf_highs', nb.float64[:]), ('wave_le_par_ids', nb.int64[:]), ('wave_le_lows', nb.float64[:]),\
           ('wave_le_highs', nb.float64[:]), ('n_wavelet', nb.int64), ('n_glitch', nb.int64), ('max_n_wavelet', nb.int64),\
           ('max_n_glitch', nb.int64)])
class FastPriorInfo:
    """simple jitclass to store the various elements of fast prior calculation in a way that can be 
       accessed quickly from a numba environment"""
    def __init__(self, uniform_par_ids, uniform_lows, uniform_highs, lin_exp_par_ids, lin_exp_lows, 
                 lin_exp_highs, normal_par_ids, normal_mus, normal_sigs, dm_par_ids, dm_dists, 
                 dm_errs, px_par_ids, px_mus, px_errs, cut_par_ids, cut_lows, cut_highs, global_common,
                 glitch_uf_par_ids, glitch_uf_lows, glitch_uf_highs, glitch_le_par_ids, glitch_le_lows,
                 glitch_le_highs, wave_uf_par_ids, wave_uf_lows, wave_uf_highs, wave_le_par_ids, 
                 wave_le_lows, wave_le_highs, n_wavelet, n_glitch, max_n_wavelet, max_n_glitch):
        #All other parameter attributes
        self.uniform_par_ids = uniform_par_ids
        self.uniform_lows = uniform_lows
        self.uniform_highs = uniform_highs
        self.lin_exp_par_ids = lin_exp_par_ids
        self.lin_exp_lows = lin_exp_lows
        self.lin_exp_highs = lin_exp_highs
        self.normal_par_ids = normal_par_ids
        self.normal_mus = normal_mus
        self.normal_sigs = normal_sigs
        self.dm_par_ids = dm_par_ids
        self.dm_dists = dm_dists
        self.dm_errs = dm_errs
        self.px_par_ids = px_par_ids
        self.px_mus = px_mus
        self.px_errs = px_errs
        self.cut_par_ids = cut_par_ids
        self.cut_lows = cut_lows
        self.cut_highs = cut_highs
        self.global_common = global_common

        #glitch attributes
        self.glitch_uf_par_ids = glitch_uf_par_ids
        self.glitch_uf_lows = glitch_uf_lows
        self.glitch_uf_highs = glitch_uf_highs
        self.glitch_le_par_ids = glitch_le_par_ids
        self.glitch_le_lows = glitch_le_lows
        self.glitch_le_highs = glitch_le_highs

        #wavelet attributes
        self.wave_uf_par_ids = wave_uf_par_ids
        self.wave_uf_lows = wave_uf_lows
        self.wave_uf_highs = wave_uf_highs
        self.wave_le_par_ids = wave_le_par_ids
        self.wave_le_lows = wave_le_lows
        self.wave_le_highs = wave_le_highs

        self.n_wavelet = n_wavelet
        self.n_glitch = n_glitch
        self.max_n_wavelet = max_n_wavelet
        self.max_n_glitch = max_n_glitch

def get_FastPriorInfo(pta,psrs,max_n_glitch,max_n_wavelet):
    """get FastPriorInfo object from pta"""
    fp_loc = FastPrior(pta,psrs)
    FPI = FastPriorInfo(fp_loc.uniform_par_ids, fp_loc.uniform_lows, fp_loc.uniform_highs,\
                        fp_loc.lin_exp_par_ids, fp_loc.lin_exp_lows, fp_loc.lin_exp_highs,\
                        fp_loc.normal_par_ids, fp_loc.normal_mus, fp_loc.normal_sigs,\
                        fp_loc.dm_par_ids, fp_loc.dm_dists, fp_loc.dm_errs,\
                        fp_loc.px_par_ids, fp_loc.px_mus, fp_loc.px_errs,\
                        fp_loc.cut_par_ids,fp_loc.cut_lows,fp_loc.cut_highs, \
                        fp_loc.global_common, fp_loc.glitch_uf_par_ids, fp_loc.glitch_uf_lows,\
                        fp_loc.glitch_uf_highs, fp_loc.glitch_le_par_ids, \
                        fp_loc.glitch_le_lows, fp_loc.glitch_le_highs, \
                        fp_loc.wave_uf_par_ids, fp_loc.wave_uf_lows,\
                        fp_loc.wave_uf_highs, fp_loc.wave_le_par_ids,\
                        fp_loc.wave_le_lows, fp_loc.wave_le_highs,
                        max_n_wavelet, max_n_glitch, max_n_wavelet, max_n_glitch)
    return FPI
