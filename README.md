# Formalization of the Freeze Index Computation

This repository contains the code inherent to Magnes' review and exploration of the freeze index first introduced
in [1]. The implementations reported here in are the ones introduced by Bachlin [2], Cockx [3], Zach [4], in addition
to Moore's [1] and the newly proposed multitaper method.

## Data
In order to be able to run the comparisons and evaluate the various configurations, the Daphnet data is required to
be stored in the subdirectory `data/`. The data files (TXT) files are gitignored. The data can be obtained by visiting

https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait

accessed on 19.08.2024.

## Comparing Definitions
To compare FI definitions on the Daphnet dataset, run the script `daphnet_evaluation.py` from root as
```bash
python3.9 run_variants_comparison.py
```
The script will take care of parsing all data files, running the FI computations and comparisons, and save the
resulting plots in the `res/` subdirectory. The `res/` subdirectory is not tracked and automatically generated
if inexistent by the script. Results are sorted by input file and proxy choice.

## Multitaper Parameter Sweep
To run the multitaper parametric sweep and thus to inspect the effects of each parameter of the multitaper method
on the FI run
```bash
python3.9 run_multitaper_sweep.py
```
**NOTE** TO BE ADDED

## Proxy Evaluation
To evaluate the effect of proxy choice on the FI for the multitaper definition, run
```bash
python3.9 run_proxy_sweep.py
```
**NOTE** TO BE ADDED

## References
[1] Moore, S. T., MacDougall, H. G., & Ondo, W. G. (2008). Ambulatory monitoring of freezing of gait in Parkinson's disease. Journal of Neuroscience Methods, 167(2), 340-348. doi:10.1016/j.jneumeth.2007.08.023

[2] Bachlin M, Plotnik M, Roggen D, Maidan I, Hausdorff JM, Giladi N, Troster G. Wearable assistant for Parkinson’s disease patients with the freezing of gait symptom. IEEE Transactions on Information Technology in Biomedicine. 2009 Nov 10;14(2):436-46.

[3] Cockx H, Nonnekes J, Bloem BR, van Wezel R, Cameron I, Wang Y. Dealing with the heterogeneous presentations of freezing of gait: how reliable are the freezing index and heart rate for freezing detection?. Journal of neuroengineering and rehabilitation. 2023 Apr 27;20(1):53.

[4] Zach H, Janssen AM, Snijders AH, Delval A, Ferraye MU, Auff E, Weerdesteyn V, Bloem BR, Nonnekes J. Identifying freezing of gait in Parkinson's disease during freezing provoking tasks using waist-mounted accelerometry. Parkinsonism & related disorders. 2015 Nov 1;21(11):1362-6.