# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Polyline sampling function `sample_polyline()` in `curvesimilarities.util` module.
- Polyline refining function `refine_polyline()` in `curvesimilarities.util` module.

## [0.1.7] - 2024-06-20

### Added

- `fd()` now accepts `rel_tol` and `abs_tol` arguments for its parametric search tolerance.

### Changed

- Every curve similarity function now returns the similarity in `float` type.

### Fixed

- Convergence failure of `fd()` during its parametric search.

## [0.1.6] - 2024-06-15

### Fixed

- Degenerate case during computing the XY-monotone axis of level set for `ifd()` and its variants.

## [0.1.5] - 2024-06-12

### Fixed

- Bug during computing `ifd_owp()` and its variants.

## [0.1.4] - 2024-06-10

### Fixed

- Another degenerate case in line-line integration is now dealt with.

## [0.1.3] - 2024-06-07

### Added

- Average Fréchet distance `afd()` and its optimal warping path `afd_owp()`.
- Quadratic average Fréchet distance `qafd()` and its optimal warping path `qafd_owp()`.
- Squared dynamic time warping distance `sdtw()` and its optimal warping path `sdtw_owp()`.

### Changed

- `dtw_owp()` now returns both the distance and the optimal warping path.

### Removed

- `dtw_acm()` is removed.

### Fixed

- Curves degenerated into a point are now dealt with.

## [0.1.2] - 2024-06-05

### Added

- Integral Fréchet distance `ifd()` and its optimal warping path `ifd_owp()`.

## [0.1.1] - 2024-05-22

### Added

- Dynamic time warping distance `dtw()`, its accumulated cost matrix `dtw_acm()` and its optimal warping path `dtw_owp()`.

### Changed

- Parametric search tolerance of `fd()` is now the machine epsilon of float64.

## [0.1.0] - 2024-05-19

### Added

- (Continuous) Fréchet distance `fd()`.
- Discrete Fréchet distance `dfd()`.
