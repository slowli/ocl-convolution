# OpenCL-accelerated 2D convolutions for Rust

[![Travis Build Status](https://img.shields.io/travis/com/slowli/ocl-convolution/master.svg?label=Linux%20Build)](https://travis-ci.com/slowli/ocl-convolution) 
[![License: Apache-2.0](https://img.shields.io/github/license/slowli/ocl-convolution.svg)](https://github.com/slowli/ocl-convolution/blob/master/LICENSE)
![rust 1.34.0+ required](https://img.shields.io/badge/rust-1.34.0+-blue.svg?label=Required%20Rust) 

This library provides 2D [convolutions] accelerated with [OpenCL]. Convolutions
are particularly useful for deep learning tasks, such as image recognition;
they are a basic building block for [convolutional neural networks][cnn].

The library is intended mostly for quick-and-dirty hacking in deep learning research,
in which one needs a separate spatial convolution primitive. Note that full-scale
DL frameworks (TensorFlow, PyTorch, etc.) will most probably be a more robust and scalable
tool for more high-level tasks.

## Usage

See crate docs for the examples of usage.

## License

Licensed under [the Apache 2.0 license](LICENSE).

[convolutions]: https://en.wikipedia.org/wiki/Convolution
[opencl]: https://www.khronos.org/opencl/
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
