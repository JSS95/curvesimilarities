# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] - 2024-06-20

### Added

- `rel_tol` and `abs_tol` parameters to `fd()`.

### Changed

- Curve distances are now `float` type.

### Fixed

- Issue of Fréchet distance failling to converge is fixed.

## [0.1.6] - 2024-06-15

### Fixed

- xy-monotone axis of the level set of a cell from parallel lines.

## [0.1.5] - 2024-06-12

### Fixed

- Optimal warping path of integral Fréchet distance.

## [0.1.4] - 2024-06-10

### Fixed

- Degenerate case from line-line integration.

## [0.1.3] - 2024-06-07

### Added

- Average Fréchet distance.
- Quadratic average Fréchet distance.
- Squared dynamic time warping.

### Changed

- `dtw_owp()` now returns both the distance and the optimal warping path.

### Removed

- `dtw_acm()` is removed.

### Fixed

- Curves degenerated into a point are now dealt with.

## [0.1.2] - 2024-06-05

### Added

- Integral Fréchet distance.

## [0.1.1] - 2024-05-22

### Added

- Dynamic time warping.

### Changed

- Parametric search tolerance of Fréchet distance is now the machine epsilon of float64.

## [0.1.0] - 2024-05-19

### Added

- (Continuous) Fréchet distance.
- Discrete Fréchet distance.
