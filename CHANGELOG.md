# CHANGELOG

## V1.0.1
### Changed
- `MIN_FFT_WINDOW_SIZE` from 128 to 256.

### Fixed
- `combine_fis()` not to use `lcm` as this can yield gigantic numbers, e.g. in the case of two large prime numbers. Instead, the maximum length of the original sequences is used instead.

## V1.0.0
Initial version as per https://doi.org/10.3389/fneur.2025.1528963.