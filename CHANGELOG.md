# Changelog

All notable changes to this project will be documented in this file.
The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Bump minimum supported Rust version to 1.59.

## 0.3.0 - 2022-07-30

### Fixed

- Change dimension parameter types to `u32` to avoid potential runtime errors / undefined behavior
  when converting from `usize`.

### Changed

- Update dependencies.
- Bump minimum supported Rust version to 1.57 and switch to 2021 Rust edition.
- Re-license the project under dual MIT / Apache-2.0 license.

### Removed

- Remove `WithParams` from the public interface of the crate.

## 0.2.0 - 2020-03-16

### Added

- Improve crate documentation, in particular, panic descriptions.

### Changed

- Update dependencies.

## 0.1.0 - 2019-07-21

The initial release of `ocl-convolution`.
