# OpenCL-accelerated 2D convolutions for Rust

[![Build Status](https://github.com/slowli/ocl-convolution/workflows/CI/badge.svg?branch=master)](https://github.com/slowli/ocl-convolution/actions) 
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](https://github.com/slowli/ocl-convolution#license)
![rust 1.65+ required](https://img.shields.io/badge/rust-1.65+-blue.svg?label=Required%20Rust) 

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
ocl-convolution = "0.3.0"
``` 

Basic floating-point convolution can be implemented as follows:

```rust
use ndarray::Array4;
use rand::{Rng, thread_rng};
use ocl_convolution::{Convolution, FeatureMap, Params};

let convolution = Convolution::f32(3)?.build(Params {
    strides: [1, 1],
    pads: [0; 4],
    dilation: [1, 1],
    groups: 1,
})?;

// Generate random signal with 6x6 spatial dims and 3 channels.
let mut rng = thread_rng();
let signal = Array4::from_shape_fn([1, 6, 6, 3], |_| rng.gen_range(-1.0..=1.0));
// Construct two 3x3 spatial filters.
let filters = Array4::from_shape_fn([2, 3, 3, 3], |_| rng.gen_range(-1.0..=1.0));
// Perform the convolution. The output must have 4x4 spatial dims
// and contain 2 channels (1 per each filter). The output layout will
// be the same as in the signal.
let output = convolution.compute(
    // `FeatureMap` wraps `ArrayView4` with information about
    // memory layout (which is "channels-last" / NHWC in this case).
    FeatureMap::nhwc(&signal),
    &filters,
)?;
assert_eq!(output.shape(), [1, 4, 4, 2]);

Ok::<_, ocl::Error>(())
```

See crate docs for more examples of usage.

### Installing OpenCL

OpenCL has [a variety of implementations](https://www.khronos.org/opencl/resources).
For quick testing, one may use [POCL](https://github.com/pocl/pocl); it is open source
and not tied to hardware (at the cost of being CPU-based, i.e., orders of magnitude
slower than OpenCL implementations by GPU vendors).
POCL [can be installed from sources](http://portablecl.org/docs/html/install.html)
with the commands like in the [installation script](install-pocl.sh)
(tested on Ubuntu 22.04).

## Contributing

All contributions are welcome! See [the contributing guide](CONTRIBUTING.md) to help
you get involved.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `ocl-convolution` by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.

[convolutions]: https://en.wikipedia.org/wiki/Convolution
[opencl]: https://www.khronos.org/opencl/
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
