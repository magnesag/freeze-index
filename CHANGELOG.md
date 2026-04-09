# CHANGELOG

## V1.0.1
### Added
- `dev` dependency group (managed through `uv`)

### Changed
- Docstrings in `freezing/**/*.py` and `xcomparisons/**/*.py` from Doxygen-style to Numpy-style and added missing docstrings, e.g. enumerations.
- `MIN_FFT_WINDOW_SIZE` from 128 to 256, to match Nushu gait-analysis implementation.
- `nmaf` argument in `compute_multitaper_fi()` default value from 5 to 11 to match Nushu gait-analysis implementation.
- `compute_multitaper_fi()` to raise a `ValueError` when window is empty.
- Version definition responsibility: from `freezing/__version__.py` to `pyproject.toml` as managed via `uv`
- Minimum Python version from 3.10 to 3.11, in order to have `tomlib` available, may be needed in the future to access/define `__version__` from `pyproject.toml`

### Fixed
- `combine_fis()` not to use `lcm` as this can yield gigantic numbers, e.g. in the case of two large prime numbers. Instead, the maximum length of the original sequences is used instead.

### Removed
- `freezing/__version__.py`, see version management comment in **Changed** section.
- `requirements.txt` as dependencies shall only be defined _once_ in `pyproject.toml`.

## V1.0.0
Initial version as per https://doi.org/10.3389/fneur.2025.1528963.