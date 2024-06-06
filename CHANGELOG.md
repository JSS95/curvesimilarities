# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Average Fréchet distance.
- Quadratic average Fréchet distance.

### Changed

- `dtw_owp()` now returns both the distance and the optimal warping path.

### Removed

- `dtw_acm()` is removed.

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
