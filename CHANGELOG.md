# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

**Integral Fréchet distance (`curvesimilarities.integfrechet`)**

- `ifd()` and `ifd_owp()` now take `dist` argument to specify the distance type.
- `ifd()` and `ifd_owp()` now use the Euclidean distance as default (not implemented yet).

**Average Fréchet distance (`curvesimilarities.averagefrechet`)**

- `qafd()` and `qafd_owp()` now take `dist` argument to specify the distance type.
- `qafd()` and `qafd_owp()` now use the Euclidean distance as default.

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
