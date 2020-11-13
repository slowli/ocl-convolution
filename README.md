# OpenCL-accelerated 2D convolutions for Rust

[![Build Status](https://github.com/slowli/ocl-convolution/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/ocl-convolution/actions) 
[![License: Apache-2.0](https://img.shields.io/github/license/slowli/ocl-convolution.svg)](https://github.com/slowli/ocl-convolution/blob/master/LICENSE)
![rust 1.39.0+ required](https://img.shields.io/badge/rust-1.39.0+-blue.svg?label=Required%20Rust) 

**Documentation:** [![Docs.rs](https://docs.rs/ocl-convolution/badge.svg)](https://docs.rs/ocl-convolution/)
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/ocl-convolution/ocl_convolution/) 

This library provides 2D [convolutions] accelerated with [OpenCL]. Convolutions
are particularly useful for deep learning tasks, such as image recognition;
they are a basic building block for [convolutional neural networks][cnn].

The library is intended mostly for quick-and-dirty hacking in deep learning research,
in which one needs a separate spatial convolution primitive. Note that full-scale
DL frameworks (TensorFlow, PyTorch, etc.) will most probably be a more robust and scalable
tool for more high-level tasks.

## Usage

Add this to your `Crate.toml`:

```toml
[dependencies]
ocl-convolution = "0.2.0"
``` 

See crate docs for the examples of usage.

### Installing OpenCL

OpenCL has [a variety of implementations](https://www.khronos.org/opencl/resources).
For quick testing, one may use [POCL](https://github.com/pocl/pocl); it is open source
and not tied to hardware (at the cost of being CPU-based, i.e., orders of magnitude
slower than OpenCL implementations by GPU vendors).
POCL may be installed from sources with the commands like these
(showcased here for Ubuntu Bionic):

```bash
# Install utils for build
apt-get install build-essential cmake pkg-config libhwloc-dev zlib1g-dev
# Install OpenCL-related utils
apt-get install ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev clinfo
# Install LLVM / Clang from the official APT repository
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - 
add-apt-repository 'deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main'
apt-get update
apt-get install clang-9 libclang-9-dev llvm-9 llvm-9-dev

# Get POCL sources
export POCL_VER=1.4 # latest stable version
curl -sSL "https://github.com/pocl/pocl/archive/v$POCL_VER.tar.gz" > pocl-$POCL_VER.tar.gz
tar xf "pocl-$POCL_VER.tar.gz"
# Build POCL from the sources
cd pocl-$POCL_VER
mkdir build && cd build
cmake -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-9 -DCMAKE_INSTALL_PREFIX=/usr ..
make

# Verify installation
clinfo
# If successful, `clinfo` should display information about the POCL platform.
```

## License

Licensed under [the Apache 2.0 license](LICENSE).

[convolutions]: https://en.wikipedia.org/wiki/Convolution
[opencl]: https://www.khronos.org/opencl/
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
