# Quick-start guide to QuickBurst

## Setup
First, we advise creating a new `conda` environment:
```
conda create --name QuickBurst python=3.9
```
Activate our new environment:
```
conda activate QuickBurst
```
Install enterprise, La Forge, and Jupyter:
```
conda install -c conda-forge enterprise-pulsar
pip install Jupyter
pip install la-forge
```

Clone the `QuickBurst` repo:
```
git clone https://github.com/JacobATaylor/QuickBurst.git
```
Move into the repo's folder:
```
cd QuickBurst
```
Install QuickBurst and all remaining requirements:
```
pip install -e .
pip install -r requirements.txt
```

## Running QuickBurst
You will need a pickle file with the data you are planning to analyize, along with it's noise parameter file. Before performing a run, you will need to create tau scans to inform jumps over shape parameters. To do so,
* Open `make_tau_scan_proposals.ipynb`
* Pass in your pickled pulsar object `data.pkl`.
* Pass in your noise dictionary.
* First create noise transient tau scans. Save these to a pickle file.
* Stitch together individual tau scan maps to create GW signal wavelet tau scans. Save these to a pickle file.

Once these tau scans have been created, we can start a run. The main analysis code can be found in `QuickBurst_MCMC.py` and `QuickBurst_lnlike.py`. `QuickBurst_MCMC.py` can be executed by either running `QuickBurst.ipynb` or `QuickBurst.py`, depending on your preferred format. Both scripts perform the same functions. To start a run:
* Open `QuickBurst.py` or `QuickBurst.ipynb`.
* Pass in: 1) `data.pkl`, 2) `noise_dictionary.json` (or as a filepath), 3) `noise_transient_tau_scan.pkl`, and 4) `GW_signal_wavelet_tau_scan.pkl`.

When starting your run, **check the `run_qb()` doc-string to ensure your model parameters are properly set (either fixed, varied, or not included)**. For modifying number of samples:
* `N_slow` is the number of shape parameter updates to do.
* `n_fast_to_slow` is the ratio of projection parameter to shape parameter updates. By default, this is set to `n_fast_to_slow = 10000`.
* `n_fish_update` is a ratio of fisher matrix proposal updates to shape parameter updates. It is advised to leave this to be once or twice for a run if varying intrinsic pulsar noise parameters (`n_fish_update = int(N_slow/2)`). By default, this is set to `n_fish_update = int(N_slow/2)`.

## Post processing
To begin post processing, your results will be saved in a single `.h5df` file. To analyze this file, follow the notebook `plot_results_QuickBurst.ipynb`, which has a number of scripts within to plot the likelihood trace, parameter histograms, sky location parameters, waveform recovery, and sky maps.
