# Formalization of the Freeze Index Computation

This repository contains the code inherent to Magnes' review and exploration of the freeze index first introduced
in [1]. The implementations reported here in are the ones introduced by Bachlin [2], Cockx [3], Zach [4], in addition
to Moore's [1] and the newly proposed multitaper method.

## Setup

### Requirements
1. Python >=3.9
2. (optional) Latex -- for paper-ready plots set `USE_TEX=True` in `aux/cfg.py`

### Daphnet dataset
The Daphnet Freezing of Gait dataset is used for comparisons. It has a permissive CC BY 4.0 license and the data can be found under `data/` folder.

Source (accessed on 19.08.2024): https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait

### Python environment
To manage Python library dependencies, Python virtual environment is used. Run the following from the root project directory:
```sh
# Create Python virtual environment
python3 -m venv venv
# Activate it
. ./venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

## Comparing Definitions
To compare FI definitions on the Daphnet dataset, run the script `run_variants_comparison.py` from root project directory as
```bash
python run_variants_comparison.py
```
The script will take care of parsing all data files, running the FI computations and comparisons, and save the
resulting plots in the `res/` subdirectory. The `res/` subdirectory is not tracked and automatically generated
if inexistent by the script. Results are sorted by input file and proxy choice.

## Multitaper Parameter Sweep
To run the multitaper parametric sweep and thus to inspect the effects of each parameter of the multitaper method
on the FI run
```bash
python run_multitaper_sweep.py
```

## Proxy Evaluation
To evaluate the effect of proxy choice on the FI for the multitaper definition, run
```bash
python run_proxy_sweep.py
```

## References
[1] Moore ST, MacDougall HG, Ondo WG. Ambulatory monitoring of freezing of gait in Parkinson's disease. Journal of neuroscience methods. 2008 Jan 30;167(2):340-8.

[2] Bachlin M, Plotnik M, Roggen D, Maidan I, Hausdorff JM, Giladi N, Troster G. Wearable assistant for Parkinson’s disease patients with the freezing of gait symptom. IEEE Transactions on Information Technology in Biomedicine. 2009 Nov 10;14(2):436-46.

[3] Cockx H, Nonnekes J, Bloem BR, van Wezel R, Cameron I, Wang Y. Dealing with the heterogeneous presentations of freezing of gait: how reliable are the freezing index and heart rate for freezing detection?. Journal of neuroengineering and rehabilitation. 2023 Apr 27;20(1):53.

[4] Zach H, Janssen AM, Snijders AH, Delval A, Ferraye MU, Auff E, Weerdesteyn V, Bloem BR, Nonnekes J. Identifying freezing of gait in Parkinson's disease during freezing provoking tasks using waist-mounted accelerometry. Parkinsonism & related disorders. 2015 Nov 1;21(11):1362-6.