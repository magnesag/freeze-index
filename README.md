# Formalization of the Freeze Index Computation

This repository contains the code inherent to Magnes' review and exploration
of the freeze index first introduced in [1]. The implementations reported herein
are the ones introduced by Bachlin [2], Cockx [3], Zach [4], in addition to
Moore's [1] and the newly proposed multitaper method. This work has been
published in Frontiers in Neurology, 16, 2025.

Please cite this work/repository as:

> Schaer A, Maurenbrecher H, Mangiante C, Sobkuliak R, Müsch K, Sanchez Lopez P, Moraud EM, Ergeneman O, Chatzipirpiridis G. Toward a unified gait freeze index: a standardized benchmark for clinical and regulatory evaluations. Frontiers in Neurology, 16:1528963, 2025.


## Setup

### Installation and Importing
Installation
```bash
pip install freeze-index
```
Typical import and use
```python
import numpy as np

from freezing import freezeindex as filib

x = np.random.randn(100)
t, fi = filib.compute_multitaper_fi(x, ...)
```

## Comparing Definitions [only available in source repo]
Source repo available at https://github.com/magnesag/freeze-index

### Requirements
1. Python >=3.10
2. (optional) Latex -- for paper-ready plots set `USE_TEX=True` in `aux/cfg.py`

### Daphnet Dataset
The Daphnet Freezing of Gait dataset is used for comparisons. It has a permissive
CC BY 4.0 license and the data can be found under `data/` folder.

Source (accessed on 19.08.2024): https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait

### Python Environment for Source Usage
To manage Python library dependencies, Python virtual environment is used. Run the
following from the root project directory (assuming `python --version >=3.10`):
```sh
# Create Python virtual environment
python -m venv venv
# Activate it
. ./venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

### Executable Comparisons `xcomparisons`
The `xcomparisons` folder contains executable scripts that perform comparisons.

The scripts take care of parsing all data files, running the FI computations
and comparisons, and save the resulting plots in the `res/` subdirectory. The `res/`
subdirectory is not tracked and automatically generated if inexistent by the script.
Results are sorted by input file and proxy choice.

#### Performance Evaluation on White Noise
To evaluate the FI definitions' theoretical performance when computing the FI on
white noise run the `simulation.py` script from the root directory as
```bash
python -m xcomparisons.simulation
```

#### Definitions Comparison
To compare FI definitions on the Daphnet dataset, run the script `run_variants_comparison.py`
from root project directory as
```bash
python -m xcomparisons.run_variants_comparison
```

#### Multitaper Parameter Sweep
To run the multitaper parametric sweep and thus to inspect the effects of each parameter
of the multitaper method on the FI run
```bash
python -m xcomparisons.run_multitaper_sweep
```

#### Proxy Evaluation
To evaluate the effect of proxy choice on the FI for the multitaper definition, run
```bash
python -m xcomparisons.run_proxy_sweep
```

## References
[1] Moore ST, MacDougall HG, Ondo WG. Ambulatory monitoring of freezing of gait in Parkinson's disease. Journal of neuroscience methods. 2008 Jan 30;167(2):340-8. <br>
[2] Bachlin M, Plotnik M, Roggen D, Maidan I, Hausdorff JM, Giladi N, Troster G. Wearable assistant for Parkinson’s disease patients with the freezing of gait symptom. IEEE Transactions on Information Technology in Biomedicine. 2009 Nov 10;14(2):436-46. <br>
[3] Cockx H, Nonnekes J, Bloem BR, van Wezel R, Cameron I, Wang Y. Dealing with the heterogeneous presentations of freezing of gait: how reliable are the freezing index and heart rate for freezing detection?. Journal of neuroengineering and rehabilitation. 2023 Apr 27;20(1):53. <br>
[4] Zach H, Janssen AM, Snijders AH, Delval A, Ferraye MU, Auff E, Weerdesteyn V, Bloem BR, Nonnekes J. Identifying freezing of gait in Parkinson's disease during freezing provoking tasks using waist-mounted accelerometry. Parkinsonism & related disorders. 2015 Nov 1;21(11):1362-6. <br>
