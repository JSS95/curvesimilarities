# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0a4] - 2024-08-13

### Fixed

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- `ifd_owp()` returns correct matching indices.

## [0.4.0a3] - 2024-08-13

### Fixed

**Fréchet distance (`curvesimilarities.frechet`)**

- `fd_matching()` returns correct matching indices.

## [0.4.0a2] - 2024-08-12

### Changed

**Fréchet distance (`curvesimilarities.frechet`)**

- `significant_events()` take *param_type* argument instead of *param*.
- `fd_matching()` take *param_type* argument instead of *param*.

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- `ifd_owp()` take *param_type* argument instead of *param*.

**Utility (`curvesimilarities.util`)**

- `sample_polyline()` no longer clips the parameters into the valid range.
- `sample_polyline()` is now numba-compiled and strictly takes numpy array as *vert* argument.
- `sample_polyline()` now takes *param_type* argument to specify the parametrization of *param* argument.

## [0.4.0a1] - 2024-08-09

### Added

**Fréchet distance (`curvesimilarities.frechet`)**

- Decision problem for Fréchet distance `decision_problem()`.
- Significant events in Fréchet distance `significant_events()`.
- Locally correct Fréchet matching `fd_matching()`.

### Changed

**Utility (`curvesimilarities.util`)**

- `curve_matching()` is renamed to `matching_pairs()`.

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- `ifd_owp()` now take `param` argument to specify the parametrization type.

### Removed

**Fréchet distance (`curvesimilarities.frechet`)**

- `fd_params()` is removed. Use `significant_events()` instead.

**Utility (`curvesimilarities.util`)**

- `refine_polyline()` is removed.

## [0.3.0] - 2024-07-26

### Added

**Fréchet distance (`curvesimilarities.frechet`)**

- Fréchet distance with parameters `fd_params()`.
- Discrete Fréchet distance with indices `dfd_idxs()`.

### Changed

**Fréchet distance (`curvesimilarities.frechet`)**

- `fd()` is now numba-compiled, and strictly takes numpy arrays as `P` and `Q`.
- `dfd()` is now numba-compiled, and strictly takes numpy arrays as `P` and `Q`.

**Dynamic time warping (`curvesimilarities.dtw`)**

- `dtw()` and `dtw_owp()` now take `dist` argument to specify the distance type.
- `dtw()` and `dtw_owp()` are now numba-compiled, and strictly take numpy arrays as `P` and `Q`.

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- `ifd()` and `ifd_owp()` now take `dist` argument to specify the distance type.
- `ifd()` and `ifd_owp()` now use the Euclidean distance as default (which is not implemented yet).
- `ifd()` and `ifd_owp()` are now numba-compiled, and strictly take numpy arrays as `P` and `Q`.

**Utility (`curvesimilarities.util`)**

- `curvespace_path()` is renamed to `curve_matching()`.

### Removed

**Average Fréchet distance (`curvesimilarities.averagefrechet`)**

- `curvesimilarities.averagefrechet` module is removed.

**Dynamic time warping (`curvesimilarities.dtw`)**

- `sdtw()` and `sdtw_owp()` are removed. Use `dtw()` and `dtw_owp()` with `dist` argument instead.

## [0.2.1] - 2024-06-24

### Fixed

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- Avoid duplicate points in optimal warping path.

## [0.2.0] - 2024-06-24

### Added

**Utility (`curvesimilarities.util`)**

- Polyline sampling function `sample_polyline()`.
- Polyline refining function `refine_polyline()`.
- Parameter space function `parameter_space()`.
- Function `curvespace_path()` which converts parameter space path to curve space point pairs.

### Changed

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- Squared Euclidean distance is used instead of Euclidean distance.

### Removed

**Average Fréchet distance (`curvesimilarities.averagefrechet`)**

- Average Fréchet distance `afd()` and its optimal warping path `afd_owp()`.

## [0.1.7] - 2024-06-20

### Added

**Fréchet distance (`curvesimilarities.frechet`)**

- `fd()` now accepts `rel_tol` and `abs_tol` arguments for its parametric search tolerance.

### Changed

- Every curve similarity function now returns the similarity in `float` type.

### Fixed

**Fréchet distance (`curvesimilarities.frechet`)**

- Convergence failure of `fd()` during its parametric search.

## [0.1.6] - 2024-06-15

### Fixed

**Fréchet distance (`curvesimilarities.frechet`)**

- Degenerate case during computing the XY-monotone axis of level set for `ifd()` and its variants.

## [0.1.5] - 2024-06-12

### Fixed

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- Bug during computing `ifd_owp()` and its variants.

## [0.1.4] - 2024-06-10

### Fixed

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- Another degenerate case in line-line integration is now dealt with.

## [0.1.3] - 2024-06-07

### Added

**Average Fréchet distance (`curvesimilarities.averagefrechet`)**

- Average Fréchet distance `afd()` and its optimal warping path `afd_owp()`.
- Quadratic average Fréchet distance `qafd()` and its optimal warping path `qafd_owp()`.

**Dynamic time warping (`curvesimilarities.dtw`)**

- Squared dynamic time warping distance `sdtw()` and its optimal warping path `sdtw_owp()`.

### Changed

**Dynamic time warping (`curvesimilarities.dtw`)**

- `dtw_owp()` now returns both the distance and the optimal warping path.

### Removed

**Dynamic time warping (`curvesimilarities.dtw`)**

- `dtw_acm()` is removed.

### Fixed

- Curves degenerated into a point are now dealt with.

## [0.1.2] - 2024-06-05

### Added

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- Integral Fréchet distance `ifd()` and its optimal warping path `ifd_owp()`.

## [0.1.1] - 2024-05-22

### Added

**Dynamic time warping (`curvesimilarities.dtw`)**

- Dynamic time warping distance `dtw()`, its accumulated cost matrix `dtw_acm()` and its optimal warping path `dtw_owp()`.

### Changed

**Fréchet distance (`curvesimilarities.frechet`)**

- Parametric search tolerance of `fd()` is now the machine epsilon of float64.

## [0.1.0] - 2024-05-19

### Added

**Fréchet distance (`curvesimilarities.frechet`)**

- (Continuous) Fréchet distance `fd()`.
- Discrete Fréchet distance `dfd()`.
