'''META DATA FOR LIKELIHOOD TESTS FOR QUICKBURST CODE'''
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "82238711",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "WARNING: version mismatch between CFITSIO header (v4.000999999999999) and linked library (v4.01).\n",
      "\n",
      "\n",
      "WARNING: version mismatch between CFITSIO header (v4.000999999999999) and linked library (v4.01).\n",
      "\n",
      "\n",
      "WARNING: version mismatch between CFITSIO header (v4.000999999999999) and linked library (v4.01).\n",
      "\n",
      "WARNING: AstropyDeprecationWarning: The private astropy._erfa module has been made into its own package, pyerfa, which is a dependency of astropy and can be imported directly using \"import erfa\" [astropy._erfa]\n"
     ]
    }
   ],
   "source": [
    "#chack for updated files\n",
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "#import packages\n",
    "from __future__ import division\n",
    "\n",
    "import numpy as np\n",
    "import glob, json\n",
    "import pickle\n",
    "\n",
    "import os as os_pack\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import corner\n",
    "#%matplotlib inline\n",
    "#%config InlineBackend.figure_format = 'retina'\n",
    "\n",
    "import healpy as hp\n",
    "\n",
    "import os, glob, json, pickle\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import scipy.linalg as sl\n",
    "\n",
    "import enterprise\n",
    "from enterprise.pulsar import Pulsar\n",
    "import enterprise.signals.parameter as parameter\n",
    "from enterprise.signals import utils\n",
    "from enterprise.signals import signal_base\n",
    "from enterprise.signals import selections\n",
    "from enterprise.signals.selections import Selection\n",
    "from enterprise.signals import white_signals\n",
    "from enterprise.signals import gp_signals\n",
    "from enterprise.signals import deterministic_signals\n",
    "import enterprise.constants as const\n",
    "\n",
    "from enterprise_extensions import blocks\n",
    "from enterprise_extensions import models as ee_models\n",
    "from enterprise_extensions import model_utils as ee_model_utils\n",
    "from enterprise_extensions import model_orfs\n",
    "from enterprise_extensions.frequentist import optimal_statistic as opt_stat\n",
    "from enterprise_extensions import sampler as ee_sampler\n",
    "from enterprise.signals.signal_base import LogLikelihood\n",
    "\n",
    "import enterprise_wavelets as models\n",
    "from enterprise.signals.deterministic_signals import Deterministic\n",
    "from enterprise.signals.parameter import function\n",
    "\n",
    "from la_forge.core import Core\n",
    "from la_forge.diagnostics import plot_chains\n",
    "from la_forge import rednoise\n",
    "import la_forge\n",
    "\n",
    "\n",
    "import corner\n",
    "from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc\n",
    "\n",
    "#style"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "5b433aab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['J0030+0451', 'J0340+4130', 'J0406+3039', 'J0437-4715']\n",
      "15.460752938661816\n",
      "8.097196263848735\n",
      "3.5471797714338695\n",
      "4.73231500896138\n"
     ]
    }
   ],
   "source": [
    "#Loading in pickle and noise files\n",
    "pint_pickle = '/home/gia/Jacob_Taylor/15yr_v1.1/Data/PINT/v1p1_de440_pint_bipm2019.pkl'\n",
    "\n",
    "#load noise dictionary\n",
    "noise_file = '/home/gia/Jacob_Taylor/15yr_v1.1/Data/PINT/v1p1_wn_dict.json'\n",
    "#psrlist = np.loadtxt('/home/reyna/15yr_v1p0/15yr_v1-20211001T235643Z-001/15yr_v1/psrlist_15yr_pint.txt', dtype = str)\n",
    "\n",
    "with open(noise_file, 'r') as h:\n",
    "    noise_params = json.load(h)\n",
    "\n",
    "with open(pint_pickle,'rb') as f:\n",
    "    allpsrs = pickle.load(f)\n",
    "\n",
    "psrs = []\n",
    "for ii,p in enumerate(allpsrs):\n",
    "    psrs.append(p)\n",
    "\n",
    "#Temporary to get code to not crash\n",
    "psrs = psrs[0:4]\n",
    "psrlist = [psr.name for psr in psrs]\n",
    "print(psrlist)\n",
    "for i in range(len(psrs)):\n",
    "    print((max(psrs[i].toas) - min(psrs[i].toas))/(3.17*10**(7)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "350c388c",
   "metadata": {},
   "outputs": [],
   "source": [
    "glitches = []\n",
    "log10_f0 = parameter.Uniform(np.log10(3.5e-9), np.log10(1e-7))(\"Glitch_\"+str(1)+'_'+'log10_f0')\n",
    "phase0 = parameter.Uniform(0, 2*np.pi)(\"Glitch_\"+str(1)+'_'+'phase0')\n",
    "tau = parameter.Uniform(0.2, 5)(\"Glitch_\"+str(1)+'_'+'tau')\n",
    "t0 = parameter.Uniform(0.0, 10.0)(\"Glitch_\"+str(1)+'_'+'t0')\n",
    "psr_idx = parameter.Uniform(-0.5, len(psrs)-0.5)(\"Glitch_\"+str(1)+'_'+'psr_idx')\n",
    "log10_h = parameter.LinearExp(-6.5, -5)(\"Glitch_\"+str(1)+'_'+'log10_h')\n",
    "glitch_wf = models.glitch_delay(log10_h = log10_h, tau = tau, log10_f0 = log10_f0, t0 = t0, phase0 = phase0, tref=53000*86400,\n",
    "                                        psr_float_idx = psr_idx, pulsars=psrs)\n",
    "glitches.append(deterministic_signals.Deterministic(glitch_wf, name='Glitch'+str(1) ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "3c33d91e",
   "metadata": {},
   "outputs": [],
   "source": [
    "tm = gp_signals.TimingModel(use_svd=True)\n",
    "wn = blocks.white_noise_block(vary=False, inc_ecorr=False)\n",
    "#s = base_model\n",
    "s = tm + wn + glitches[0]\n",
    "\n",
    "model = []\n",
    "for p in psrs:\n",
    "    model.append(s(p))\n",
    "\n",
    "with open(noise_file, 'r') as fp:\n",
    "    noisedict = json.load(fp)\n",
    "    pta = signal_base.PTA(model)\n",
    "    pta.set_default_params(noisedict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "95d4de0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "d0 = parameter.sample(pta.params)\n",
    "x0 = np.array([d0[par.name] for par in pta.params])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "c55eff73",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "phiinv_method  cliques\n",
      "resres_logdet  474270.19259698817\n",
      "resres_logdet+logsigprior 474270.19259698817\n",
      "resres_logdet+logsigprior+Tnr_sigma_phi2 459571.3997922353\n",
      "resres_logdet+logsigprior+Tnr_sigma_phi2 453862.07316676935\n",
      "resres_logdet+logsigprior+Tnr_sigma_phi2 450659.3819087744\n",
      "resres_logdet+logsigprior+Tnr_sigma_phi2 448886.27807438344\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "448886.27807438344"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pta.get_lnlikelihood(x0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "d5d1c095",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "444682.74780972634"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "444682.74780972634"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "6013ab44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[Glitch_1_log10_f0:Uniform(pmin=-8.455931955649724, pmax=-7.0),\n",
       " Glitch_1_log10_h:LinearExp(pmin=-6.5, pmax=-5),\n",
       " Glitch_1_phase0:Uniform(pmin=0, pmax=6.283185307179586),\n",
       " Glitch_1_psr_idx:Uniform(pmin=-0.5, pmax=3.5),\n",
       " Glitch_1_t0:Uniform(pmin=0.0, pmax=10.0),\n",
       " Glitch_1_tau:Uniform(pmin=0.2, pmax=5)]"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pta.params"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "5026c4b3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[array([ 2.08595371e-06, -5.90366388e-06, -5.92066508e-07, ...,\n",
      "        4.61103863e-06,  3.69812539e-06,  5.12630255e-06]), array([ 7.14780196e-06,  1.10163072e-05,  1.78848482e-05, ...,\n",
      "        7.06905167e-06,  9.67104343e-06, -2.91498660e-06]), array([-6.67558062e-06, -3.22204865e-06, -1.77474786e-06, ...,\n",
      "        2.54644130e-06, -8.93408535e-06,  1.76232555e-06]), array([-3.11563524e-06, -3.03258253e-06, -1.13872972e-06, ...,\n",
      "        7.70764237e-08,  4.37320048e-07,  8.57167085e-06])]\n"
     ]
    }
   ],
   "source": [
    "print(pta.get_residuals())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "b0f9d985",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Glitch_1_log10_f0',\n",
       " 'Glitch_1_log10_h',\n",
       " 'Glitch_1_phase0',\n",
       " 'Glitch_1_psr_idx',\n",
       " 'Glitch_1_t0',\n",
       " 'Glitch_1_tau']"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pta.param_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "d469d488",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0781907278052936"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x0[3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "b321b771",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import Fast_Burst_likelihood_ent_version as FB_ent\n",
    "import Fast_Burst_likelihood as FB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "215282cc",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Nvecs arary:  [[4.61695300e-12 5.66353407e-11 6.83697178e-12 ... 9.39287544e-12\n",
      " 1.01635764e-11 6.60318228e-12], [3.58685902e-11 2.77047195e-11 8.77407399e-11 ... 4.00248661e-11\n",
      " 7.61997117e-11 4.82576810e-11], [2.30848299e-11 2.25095542e-12 6.90732193e-12 ... 1.26507575e-11\n",
      " 3.65281122e-11 1.93290456e-11], [5.49729188e-12 4.98164427e-12 5.77752577e-12 ... 4.40088221e-12\n",
      " 4.86261323e-12 1.46668814e-09], ...]\n",
      "499865.0541270517\n",
      "474270.19259698817\n",
      "443227.88667172863\n"
     ]
    }
   ],
   "source": [
    "FB1 = FB_ent.FastBurst(pta = pta, psrs = psrs, params = d0, Npsr = len(psrs), tref=53000*86400)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "d5808805",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Amplitude:  7.275198722259484e-06\n",
      "Frequency:  7.424432949733411e-08\n",
      "glitch_index:  1.0781907278052936\n",
      "Old Sigma:  [0. 0.]\n",
      "New sigma:  [-5.66307284e-06 -4.56706935e-06]\n",
      "Old M and N:  [[0. 0.]\n",
      " [0. 0.]] [0. 0.]\n",
      "Input time:  2535.279674340166\n",
      "Per pulsar toas:  [ 394.85368635  394.85368641  394.85368641 ... 6067.37530852 6067.37530852\n",
      " 6067.37530852]\n",
      "Cosine:  [-0.03058962 -0.03058962 -0.03058962 ... -0.01215617 -0.01215617\n",
      " -0.01215617]\n",
      "Sine:  [ 0.24476505  0.24476505  0.24476505 ... -0.01847295 -0.01847295\n",
      " -0.01847295]\n",
      "Exponential:  [-790.28232761 -790.28232762 -790.28232762 ... -976.39827536 -976.39827536\n",
      " -976.39827536]\n",
      "Per pulsar toas:  [2972.08141782 2972.08141786 2972.08141793 ... 5942.92773218 5942.92773218\n",
      " 5942.92773222]\n",
      "Cosine:  [0.30252616 0.30252616 0.30252616 ... 0.01822269 0.01822269 0.01822269]\n",
      "Sine:  [-0.89355138 -0.89355138 -0.89355138 ... -0.02229015 -0.02229015\n",
      " -0.02229015]\n",
      "Exponential:  [-872.40420074 -872.40420074 -872.40420074 ... -972.10419592 -972.10419592\n",
      " -972.10419593]\n",
      "Per pulsar toas:  [4755.05880652 4755.05880655 4755.05880658 ... 6056.51249574 6056.5124958\n",
      " 6056.5124958 ]\n",
      "Cosine:  [ 0.01499291  0.01499291  0.01499291 ... -0.01928778 -0.01928778\n",
      " -0.01928778]\n",
      "Sine:  [ 0.22141805  0.22141805  0.22141805 ... -0.01185087 -0.01185087\n",
      " -0.01185087]\n",
      "Exponential:  [-931.59288749 -931.59288749 -931.59288749 ... -976.02307566 -976.02307567\n",
      " -976.02307567]\n",
      "Per pulsar toas:  [4221.64778504 4221.64778504 4221.64778504 ... 5957.92539828 5957.92539828\n",
      " 5957.92539828]\n",
      "Cosine:  [0.17275417 0.17275417 0.17275417 ... 0.02680951 0.02680951 0.02680951]\n",
      "Sine:  [-0.38220782 -0.38220782 -0.38220782 ... -0.00773776 -0.00773776\n",
      " -0.00773776]\n",
      "Exponential:  [-913.68184753 -913.68184753 -913.68184753 ... -972.62119079 -972.62119079\n",
      " -972.62119079]\n",
      "New M and N:  [[ 1.93051338e+14 -1.59295031e+12]\n",
      " [-1.59295031e+12  1.49306322e+14]] [ 2411000.58529601 25333377.81630716]\n",
      "adding in resres_logdet 443227.88667172863\n",
      "adding the signal\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "442986.5076212661"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "FB1.get_lnlikelihood(10**(x0[1]), x0[2], 10**(x0[0]), (3.15*10**7)*x0[5], (3.15*10**7)*x0[4], x0[3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e57ce25a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "475725.05379481724"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "442986.5076212661 #ent version"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "8cd7c925",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "482384.79280608444"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "482384.79280608444 #bence version"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e449eaf1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
