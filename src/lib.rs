//! OpenCL-accelerated 2D convolutions.
//!
//! [Convolution] is a fundamental building block in signal processing. This crate is focused
//! on 2D convolutions (i.e., the signal is a still image) in the context of [deep learning]
//! (more precisely, [convolutional neural networks][cnn]).
//! The second requirement means that the convolution filter may contain many (order of hundreds)
//! filters; and the input may contain many channels (order of hundreds or thousands), rather
//! than traditional 3 or 4. Computing such convolutions is computationally heavy and can be
//! effectively accelerated with the help of [OpenCL].
//!
//! # Features
//!
//! The crate implements convolutions on two numerical formats:
//!
//! - Single-precision floats (`f32`)
//! - Signed 8-bit integers with 32-bit multiply-add accumulator (this format is frequently denoted
//!   `int8/32` in deep learning literature). Quantization parameters are applied uniformly
//!   to the entire layer.
//!
//! For both cases, dilated or grouped convolutions are supported.
//!
//! # Implementation details
//!
//! The implementation uses output-stationary workflow (see, e.g., [this paper] for
//! the definition); that is, each element of the output tensor is computed in a single run
//! of the OpenCL kernel. This minimizes memory overhead, but may not be the fastest algorithm.
//!
//! [Convolution]: https://en.wikipedia.org/wiki/Convolution
//! [deep learning]: https://en.wikipedia.org/wiki/Deep_learning
//! [cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
//! [OpenCL]: https://www.khronos.org/opencl/
//! [this paper]: https://dl.acm.org/citation.cfm?id=3001177
//!
//! # Examples
//!
//! ## Floating-point convolution
//!
//! ```
//! use ndarray::Array4;
//! use rand::Rng;
//! use ocl_convolution::{Convolution, FeatureMap, Params};
//!
//! # fn main() -> Result<(), ocl::Error> {
//! let convolution = Convolution::f32(3)?.build(Params {
//!     strides: [1, 1],
//!     pads: [0; 4],
//!     dilation: [1, 1],
//!     groups: 1,
//! })?;
//!
//! // Generate random signal with 6x6 spatial dims and 3 channels.
//! let mut rng = rand::rng();
//! let signal = Array4::from_shape_fn([1, 6, 6, 3], |_| rng.random_range(-1.0..=1.0));
//! // Construct two 3x3 spatial filters.
//! let filters = Array4::from_shape_fn([2, 3, 3, 3], |_| rng.random_range(-1.0..=1.0));
//! // Perform the convolution. The output must have 4x4 spatial dims
//! // and contain 2 channels (1 per each filter). The output layout will
//! // be the same as in the signal.
//! let output = convolution.compute(
//!     // `FeatureMap` wraps `ArrayView4` with information about
//!     // memory layout (which is "channels-last" / NHWC in this case).
//!     FeatureMap::nhwc(&signal),
//!     &filters,
//! )?;
//! assert_eq!(output.shape(), [1, 4, 4, 2]);
//!
//! // For increased efficiency, we may pin filter memory.
//! // This is especially useful when the same filters are convolved
//! // with multiple signals.
//! let convolution = convolution.with_filters(&filters)?;
//! let new_output = convolution.compute(FeatureMap::nhwc(&signal))?;
//! assert_eq!(output, new_output);
//! # Ok(())
//! # }
//! ```
//!
//! ## Quantized convolution
//!
//! ```
//! use ndarray::Array4;
//! use rand::Rng;
//! use ocl_convolution::{Convolution, I8Params, FeatureMap, Params};
//!
//! # fn main() -> Result<(), ocl::Error> {
//! const BIT_SHIFT: u8 = 16;
//! let params = I8Params {
//!     common: Params::default(),
//!     // These params are found by profiling; here, they are
//!     // chosen randomly.
//!     bit_shift: BIT_SHIFT,
//!     scale: I8Params::convert_scale(BIT_SHIFT, 0.1),
//!     output_bias: -10,
//!     signal_bias: 20,
//!     filter_bias: -5,
//! };
//! let convolution = Convolution::i8(3)?.build(params)?;
//!
//! // Generate random signal with 6x6 spatial dims and 3 channels.
//! let mut rng = rand::rng();
//! let signal = Array4::from_shape_fn([1, 6, 6, 3], |_| rng.random_range(-127..=127));
//! // Construct two 3x3 spatial filters.
//! let filters = Array4::from_shape_fn([2, 3, 3, 3], |_| rng.random_range(-127..=127));
//! // Perform the convolution. The output must have 4x4 spatial dims
//! // and contain 2 channels (1 per each filter).
//! let output = convolution.compute(
//!     FeatureMap::nhwc(&signal),
//!     &filters,
//! )?;
//! assert_eq!(output.shape(), [1, 4, 4, 2]);
//! # Ok(())
//! # }
//! ```

#![doc(html_root_url = "https://docs.rs/ocl-convolution/0.4.0")]
#![warn(missing_debug_implementations, missing_docs, bare_trait_objects)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::doc_markdown
)]

use ndarray::{Array4, ArrayView4};
use ocl::OclPrm;

use std::{fmt, marker::PhantomData};

mod base;
mod buffers;
mod params;

use crate::{
    base::Base,
    buffers::{Filters, Pinned},
};
pub use crate::{
    base::ConvolutionBuilder,
    buffers::{FeatureMap, FeatureMapShape, Layout},
    params::{I8Params, Params},
};

const SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/conv.cl"));

/// Supported element types for convolutions.
pub trait ConvElement: OclPrm + Copy + 'static {
    /// Type of the multiply-add accumulator.
    type Acc: OclPrm + Copy + 'static;
    /// Parameters of the convolution.
    type Params: Copy + Into<Params> + Into<Self::ClParams>;
    /// OpenCL-friendly version of parameters. This is considered an implementation detail.
    type ClParams: OclPrm;
}

impl ConvElement for f32 {
    type Acc = f32;
    type Params = Params;
    type ClParams = params::ClParams;
}

impl ConvElement for i8 {
    type Acc = i32;
    type Params = I8Params;
    type ClParams = params::ClI8Params;
}

impl ConvolutionBuilder<f32> {
    /// Creates a new floating-point convolution.
    pub fn build(&self, params: Params) -> ocl::Result<Convolution<f32>> {
        Base::new(self, params).map(Convolution)
    }
}

impl ConvolutionBuilder<i8> {
    /// Creates a new quantized convolution.
    pub fn build(&self, params: I8Params) -> ocl::Result<Convolution<i8>> {
        Base::new(self, params).map(Convolution)
    }
}

/// Convolution without pinned memory.
pub struct Convolution<T: ConvElement>(Base<PhantomData<T>>);

impl<T> fmt::Debug for Convolution<T>
where
    T: ConvElement,
    T::Params: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Convolution").field(&self.0).finish()
    }
}

impl Convolution<f32> {
    /// Creates a new floating-point convolution builder. `size` determines the filter size
    /// and must be odd (1, 3, 5, ...).
    ///
    /// # Panics
    ///
    /// Panics if the filter `size` is even.
    pub fn f32(size: u32) -> ocl::Result<ConvolutionBuilder<f32>> {
        ConvolutionBuilder::new(size, &[("KERNEL_TYPE", 32)], SOURCE)
    }
}

/// Quantized convolution over signed 8-bit integers.
///
/// Due to use of `i8` inputs, computations are performed much faster than on `f32` inputs
/// (the difference manifests most on the specialized hardware, but it is seen in this
/// OpenCL-powered implementation as well).
///
/// ## Connection to real-value convolution
///
/// Quantized convolution mirrors real-valued convolution in which `i8` elements
/// of the signal, filter and output tensors represent real-valued numbers with the
/// following mapping:
///
/// ```
/// let scale: f32 = // ...
/// # 1.0;
/// let bias: i32 = // ...
/// # 0; drop(
/// |x: i8| -> f32 { scale * (i32::from(x) - bias) as f32 }
/// # )
/// ```
///
/// `scale` and `bias` may differ for different tensors; these params are usually determined
/// by *profiling* the corresponding convolutional neural network (see e.g. [this paper]).
///
/// Denote these quantiation params for tensor `T` as `T.scale` and `T.bias`. Denote `S`
/// the signal, `F` the filter, `O` the output. Convolution parameters must be set as follows:
///
/// | `I8Params` field | Value     |
/// |------------------|-----------|
/// | `signal_bias`    | `-S.bias` |
/// | `filter_bias`    | `-F.bias` |
/// | `output_bias`    | `O.bias`  |
/// | `scale`          | `S.scale * F.scale / O.scale` |
///
/// `scale` is represented as a fixed-point number with [`bit_shift`] binary digits after
/// the point. Note that filter biases `B` are not transformed during the computation.
///
/// # Computing convolution
///
/// Suppose `S` is the signal and `F` is the filter tensor; both contain `i8` values.
/// The computation is performed as follows:
///
/// 1. Unbias the signal: `S := S + params.signal_bias`.
/// 2. Unbias the filters: `F := F + params.filter_bias`.
/// 3. Compute "standard" convolution output `O := S (*) F` using `i32` precision.
/// 4. Upscale each number in the output: `O := O * params.scale`.
/// 5. If there is filter bias `B` provided, apply bias to the output per each output channel:
///    `O[f, ..] := O[f, ..] + B[f]`.
/// 6. Downscale the output: `O := round(O / 2**self.bit_shift)`,
///    where `round()` works as floating-point rounding with the default mode
///    (round to nearest, ties to even).
/// 7. Apply output bias: `O := O + params.output_bias`.
/// 8. Saturate output to `i8` range.
///
/// [`bit_shift`]: I8Params::bit_shift
/// [this paper]: https://arxiv.org/abs/1805.00907
impl Convolution<i8> {
    /// Creates a new `i8` convolution builder. `size` determines the filter size
    /// and must be odd (1, 3, 5, ...).
    ///
    /// # Panics
    ///
    /// Panics if the filter `size` is even.
    pub fn i8(size: u32) -> ocl::Result<ConvolutionBuilder<i8>> {
        ConvolutionBuilder::new(size, &[("KERNEL_TYPE", 8)], SOURCE)
    }
}

impl<T: ConvElement> Convolution<T> {
    /// Spatial size of the convolution.
    pub fn size(&self) -> u32 {
        self.0.size()
    }

    /// Returns general parameters of the convolution.
    pub fn params(&self) -> T::Params {
        self.0.params()
    }

    /// Sets convolution parameters.
    pub fn set_params(&mut self, params: T::Params) -> ocl::Result<()> {
        self.0.set_params(params)
    }

    /// Returns the convolution with pinned filter memory.
    ///
    /// # Parameters
    ///
    /// - `filters` must have `MxK_HxK_WxC` layout, where `M` is the number of filters,
    ///   `K_H` and `K_W` are spatial dimensions of a filter, `C` is the number of input channels.
    pub fn with_filters<'a>(
        self,
        filters: impl Into<ArrayView4<'a, T>>,
    ) -> ocl::Result<FiltersConvolution<T>> {
        self.0
            .with_filters(&filters.into(), None)
            .map(FiltersConvolution)
    }

    /// Returns the convolution with pinned filter / filter bias memory.
    pub fn with_biased_filters<'a>(
        self,
        filters: impl Into<ArrayView4<'a, T>>,
        filter_biases: &[T::Acc],
    ) -> ocl::Result<FiltersConvolution<T>> {
        self.0
            .with_filters(&filters.into(), Some(filter_biases))
            .map(FiltersConvolution)
    }

    /// Performs convolution on the provided `signal` and `filters`.
    ///
    /// # Parameters
    ///
    /// - `filters` must have `MxK_HxK_WxC` layout, where `M` is the number of filters,
    ///   `K_H` and `K_W` are spatial dimensions of a filter, `C` is the number of input channels.
    ///
    /// # Return value
    ///
    /// The output will have the same layout as `signal`. An error means something wrong
    /// with OpenCL.
    ///
    /// # Panics
    ///
    /// - Panics if `filters` do not have expected spatial dimensions, i.e.,
    ///   `self.size() x self.size()`.
    /// - Panics if the number of input channels differs from number of channels in `filters`.
    pub fn compute<'a>(
        &self,
        signal: FeatureMap<'_, T>,
        filters: impl Into<ArrayView4<'a, T>>,
    ) -> ocl::Result<Array4<T>> {
        self.0.compute(signal, &filters.into(), None)
    }

    /// Performs convolution on the provided `signal` and `filters`, with the output offset
    /// by the provided per-filter biases.
    ///
    /// Parameters, return value and panics are the same as for [`Self::compute()`].
    pub fn compute_with_biases<'a>(
        &self,
        signal: FeatureMap<'_, T>,
        filters: impl Into<ArrayView4<'a, T>>,
        filter_biases: &[T::Acc],
    ) -> ocl::Result<Array4<T>> {
        self.0.compute(signal, &filters.into(), Some(filter_biases))
    }
}

/// Convolution with pinned filters memory. Pinning memory increases efficiency at the cost
/// of making the convolution less flexible.
///
/// `FiltersConvolution` can be created by calling [`with_filters()`](Convolution::with_filters())
/// or [`with_biased_filters()`](Convolution::with_biased_filters()) methods in `Convolution`.
pub struct FiltersConvolution<T: ConvElement>(Base<Filters<T>>);

impl<T> fmt::Debug for FiltersConvolution<T>
where
    T: ConvElement,
    T::Params: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("FiltersConvolution")
            .field(&self.0)
            .finish()
    }
}

impl<T: ConvElement> FiltersConvolution<T> {
    /// Spatial size of the convolution.
    pub fn size(&self) -> u32 {
        self.0.size()
    }

    /// Returns general parameters of the convolution.
    pub fn params(&self) -> T::Params {
        self.0.params()
    }

    /// Sets convolution parameters.
    pub fn set_params(&mut self, params: T::Params) -> ocl::Result<()> {
        self.0.set_params(params)
    }

    /// Pins signal and output memory for this convolution.
    pub fn pin(self, signal_shape: FeatureMapShape) -> ocl::Result<PinnedConvolution<T>> {
        self.0.pinned(signal_shape).map(PinnedConvolution)
    }

    /// Computes the convolution on the provided signal.
    pub fn compute(&self, signal: FeatureMap<'_, T>) -> ocl::Result<Array4<T>> {
        self.0.compute(signal)
    }
}

/// Convolution with pinned memory for filters, signal and output. Pinning memory increases
/// efficiency at the cost of making the convolution less flexible.
///
/// `PinnedConvolution` can be created from a [`FiltersConvolution`] by calling
/// [`pin()`](FiltersConvolution::pin()).
pub struct PinnedConvolution<T: ConvElement>(Base<Pinned<T>>);

impl<T> fmt::Debug for PinnedConvolution<T>
where
    T: ConvElement,
    T::Params: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("PinnedConvolution")
            .field(&self.0)
            .finish()
    }
}

impl<T: ConvElement> PinnedConvolution<T> {
    /// Spatial size of the convolution.
    pub fn size(&self) -> u32 {
        self.0.size()
    }

    /// Returns general parameters of the convolution.
    pub fn params(&self) -> T::Params {
        self.0.params()
    }

    /// Sets convolution parameters.
    pub fn set_params(&mut self, params: T::Params) -> ocl::Result<()> {
        self.0.set_params(params)
    }

    /// Computes the convolution on the provided signal.
    ///
    /// # Panics
    ///
    /// - Panics if signal dimensions do not agree with the ones provided
    ///   to the [`pin()` method](FiltersConvolution::pin()).
    pub fn compute(&self, signal: FeatureMap<'_, T>) -> ocl::Result<Array4<T>> {
        self.0.compute(signal)
    }
}

#[cfg(doctest)]
doc_comment::doctest!("../README.md");
