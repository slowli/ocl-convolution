# OpenCL-accelerated 2D convolutions for Rust

[![Build Status](https://github.com/slowli/ocl-convolution/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/ocl-convolution/actions) 
[![License: Apache-2.0](https://img.shields.io/github/license/slowli/ocl-convolution.svg)](https://github.com/slowli/ocl-convolution/blob/master/LICENSE)
![rust 1.49.0+ required](https://img.shields.io/badge/rust-1.49.0+-blue.svg?label=Required%20Rust) 

**Documentation:** [![Docs.rs](https://docs.rs/ocl-convolution/badge.svg)](https://docs.rs/ocl-convolution/)
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/ocl-convolution/ocl_convolution/) 

This library provides 2D [convolutions] accelerated with [OpenCL]. Convolutions
are particularly useful for deep learning tasks, such as image recognition;
they are a basic building block for [convolutional neural networks][cnn].

The library is intended mostly for quick-and-dirty hacking in deep learning research,
in which one needs a separate spatial convolution primitive. Full-scale
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
POCL [can be installed from sources](http://portablecl.org/docs/html/install.html)
with the commands like in the [installation script](install-pocl.sh)
(tested on Ubuntu 22.04).

## License

Licensed under [the Apache 2.0 license](LICENSE).

[convolutions]: https://en.wikipedia.org/wiki/Convolution
[opencl]: https://www.khronos.org/opencl/
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
