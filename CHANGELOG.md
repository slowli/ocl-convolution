# Changelog

All notable changes to this project will be documented in this file.
The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Change dimension parameter types to `u32` to avoid potential runtime errors / UB
  when converting from `usize`.

### Changed

- Update dependencies.

### Removed

- Remove `WithParams` from the public interface of the crate.

## 0.2.0 - 2020-03-16

### Added

- Improve crate documentation, in particular, panic descriptions.

### Changed

- Update dependencies.

## 0.1.0 - 2019-07-21

The initial release of `ocl-convolution`.
