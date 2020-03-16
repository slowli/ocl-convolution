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
//! [`Convolution`]: struct.Convolution.html
//!
//! # Examples
//!
//! ## Floating-point convolution
//!
//! ```
//! use ndarray::Array4;
//! use rand::{Rng, thread_rng};
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
//! let mut rng = thread_rng();
//! let signal = Array4::from_shape_fn([1, 6, 6, 3], |_| rng.gen_range(-1.0, 1.0));
//! // Construct two 3x3 spatial filters.
//! let filters = Array4::from_shape_fn([2, 3, 3, 3], |_| rng.gen_range(-1.0, 1.0));
//! // Perform the convolution. The output should have 4x4 spatial dims
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
//! use rand::{Rng, thread_rng};
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
//! let mut rng = thread_rng();
//! let signal = Array4::from_shape_fn([1, 6, 6, 3], |_| rng.gen_range(-127, 127));
//! // Construct two 3x3 spatial filters.
//! let filters = Array4::from_shape_fn([2, 3, 3, 3], |_| rng.gen_range(-127, 127));
//! // Perform the convolution. The output should have 4x4 spatial dims
//! // and contain 2 channels (1 per each filter).
//! let output = convolution.compute(
//!     FeatureMap::nhwc(&signal),
//!     &filters,
//! )?;
//! assert_eq!(output.shape(), [1, 4, 4, 2]);
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs, missing_debug_implementations)]

use ndarray::{Array4, ArrayView4};
use ocl::OclPrm;

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
    params::{I8Params, Params, WithParams},
};

const SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/conv.cl"));

/// Supported element types for convolutions.
pub trait ConvElement: OclPrm + Copy + Default + WithParams + 'static {
    /// Type of the multiply-add accumulator.
    type Acc: OclPrm + Copy + Default + 'static;
}

impl ConvElement for f32 {
    type Acc = f32;
}

impl ConvElement for i8 {
    type Acc = i32;
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
#[derive(Debug)]
pub struct Convolution<T: ConvElement>(Base<T>);

impl Convolution<f32> {
    /// Creates a new floating-point convolution builder. `size` determines the filter size
    /// and should be odd (1, 3, 5, ...).
    ///
    /// # Panics
    ///
    /// Panics if the filter `size` is even.
    pub fn f32(size: usize) -> ocl::Result<ConvolutionBuilder<f32>> {
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
/// the signal, `F` the filter, `O` the output. Convolution parameters should be set as follows:
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
///   where `round()` works as floating-point rounding with the default mode
///   (round to nearest, ties to even).
/// 7. Apply output bias: `O := O + params.output_bias`.
/// 8. Saturate output to `i8` range.
///
/// [`bit_shift`]: struct.I8Params.html#field.bit_shift
/// [this paper]: https://arxiv.org/abs/1805.00907
impl Convolution<i8> {
    /// Creates a new `i8` convolution builder. `size` determines the filter size
    /// and should be odd (1, 3, 5, ...).
    ///
    /// # Panics
    ///
    /// Panics if the filter `size` is even.
    pub fn i8(size: usize) -> ocl::Result<ConvolutionBuilder<i8>> {
        ConvolutionBuilder::new(size, &[("KERNEL_TYPE", 8)], SOURCE)
    }
}

impl<T: ConvElement> Convolution<T> {
    /// Spatial size of the convolution.
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Returns general parameters of the convolution.
    pub fn params(&self) -> &T::Params {
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
    /// - `filters` should have `MxK_HxK_WxC` layout, where `M` is the number of filters,
    ///   `K_H` and `K_W` are spatial dimensions of a filter, `C` is the number of input channels.
    pub fn with_filters<'a>(
        self,
        filters: impl Into<ArrayView4<'a, T>>,
    ) -> ocl::Result<FiltersConvolution<T>> {
        self.0
            .with_filters(filters.into(), None)
            .map(FiltersConvolution)
    }

    /// Returns the convolution with pinned filter / filter bias memory.
    pub fn with_biased_filters<'a>(
        self,
        filters: impl Into<ArrayView4<'a, T>>,
        filter_biases: &[T::Acc],
    ) -> ocl::Result<FiltersConvolution<T>> {
        self.0
            .with_filters(filters.into(), Some(filter_biases))
            .map(FiltersConvolution)
    }

    /// Performs convolution on the provided `signal` and `filters`.
    ///
    /// # Parameters
    ///
    /// - `filters` should have `MxK_HxK_WxC` layout, where `M` is the number of filters,
    ///   `K_H` and `K_W` are spatial dimensions of a filter, `C` is the number of input channels.
    ///
    /// # Return value
    ///
    /// The output will have the same layout as `signal`. An error means something wrong
    /// with OpenCL.
    ///
    /// # Panics
    ///
    /// - The method will panic if `filters` do not have expected spatial dimensions, i.e.,
    ///   `self.size() x self.size()`.
    /// - Likewise, the method will panic if the number of input channels differs from number of
    ///   channels in `filters`.
    pub fn compute<'a>(
        &self,
        signal: FeatureMap<T>,
        filters: impl Into<ArrayView4<'a, T>>,
    ) -> ocl::Result<Array4<T>> {
        self.0.compute(signal, filters.into(), None)
    }

    /// Performs convolution on the provided `signal` and `filters`, with the output offset
    /// by the provided per-filter biases.
    ///
    /// Parameters, return value and panics are generally the same as for
    /// [`compute()`](#method.compute).
    pub fn compute_with_biases<'a>(
        &self,
        signal: FeatureMap<T>,
        filters: impl Into<ArrayView4<'a, T>>,
        filter_biases: &[T::Acc],
    ) -> ocl::Result<Array4<T>> {
        self.0.compute(signal, filters.into(), Some(filter_biases))
    }
}

/// Convolution with pinned filters memory.
#[derive(Debug)]
pub struct FiltersConvolution<T: ConvElement>(Base<Filters<T>>);

impl<T: ConvElement> FiltersConvolution<T> {
    /// Spatial size of the convolution.
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Returns general parameters of the convolution.
    pub fn params(&self) -> &T::Params {
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
    pub fn compute(&self, signal: FeatureMap<T>) -> ocl::Result<Array4<T>> {
        self.0.compute(signal)
    }
}

/// Convolution with pinned memory for filters, signal and output.
#[derive(Debug)]
pub struct PinnedConvolution<T: ConvElement>(Base<Pinned<T>>);

impl<T: ConvElement> PinnedConvolution<T> {
    /// Spatial size of the convolution.
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Returns general parameters of the convolution.
    pub fn params(&self) -> &T::Params {
        self.0.params()
    }

    /// Sets convolution parameters.
    pub fn set_params(&mut self, params: T::Params) -> ocl::Result<()> {
        self.0.set_params(params)
    }

    /// Computes the convolution on the provided signal. Signal dimensions must agree with
    /// the ones provided to the `pinned()` constructor.
    pub fn compute(&self, signal: FeatureMap<T>) -> ocl::Result<Array4<T>> {
        self.0.compute(signal)
    }
}

#[cfg(test)]
mod tests {
    use failure::Error;
    use ndarray::Axis;
    use rand::{thread_rng, Rng};
    use std::f32;

    use super::*;

    #[test]
    fn basics() -> Result<(), Error> {
        let convolution = Convolution::f32(3)?.build(Params::default())?;
        let signal = Array4::from_shape_vec(
            [1, 5, 5, 1],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24.,
            ],
        )?;
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.0; 9])?;

        let c = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
        assert_eq!(
            c,
            Array4::from_shape_vec(
                [1, 3, 3, 1],
                vec![54., 63., 72., 99., 108., 117., 144., 153., 162.],
            )?,
        );

        let signal = Array4::from_shape_vec(
            [1, 1, 5, 5],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24.,
            ],
        )?;
        let c = convolution.compute(FeatureMap::nchw(&signal), &filter)?;
        assert_eq!(
            c,
            Array4::from_shape_vec(
                [1, 1, 3, 3],
                vec![54., 63., 72., 99., 108., 117., 144., 153., 162.],
            )?,
        );

        Ok(())
    }

    #[test]
    fn f32_convolution_with_filters() -> Result<(), Error> {
        let filters = Array4::from_elem([1, 3, 3, 1], 1.0);
        let convolution = Convolution::f32(3)?
            .build(Params::default())?
            .with_filters(filters.view())?;

        let signal = Array4::from_shape_vec(
            [1, 5, 5, 1],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24.,
            ],
        )?;

        let c = convolution.compute(FeatureMap::nhwc(&signal))?;
        assert_eq!(
            c,
            Array4::from_shape_vec(
                [1, 3, 3, 1],
                vec![54., 63., 72., 99., 108., 117., 144., 153., 162.],
            )?,
        );

        for i in 1..=5 {
            let signal = Array4::from_elem([1, 5 + i, 5 + i, 1], i as f32);
            assert!(convolution.compute(FeatureMap::nhwc(&signal)).is_ok());
        }

        let pinned = convolution.pin(FeatureMapShape {
            batch_size: 1,
            width: 5,
            height: 5,
            channels: 1,
        })?;
        let c = pinned.compute(FeatureMap::nhwc(&signal))?;
        assert_eq!(
            c,
            Array4::from_shape_vec(
                [1, 3, 3, 1],
                vec![54., 63., 72., 99., 108., 117., 144., 153., 162.],
            )?,
        );
        for i in 1..=5 {
            let signal = Array4::from_elem([1, 5, 5, 1], i as f32);
            assert!(pinned.compute(FeatureMap::nhwc(&signal)).is_ok());
        }
        Ok(())
    }

    #[test]
    fn f32_convolution_with_filters_and_biases() -> Result<(), Error> {
        let filters = Array4::from_elem([1, 3, 3, 1], 1.0);
        let convolution = Convolution::f32(3)?
            .build(Params::default())?
            .with_biased_filters(filters.view(), &[-100.0])?;

        let signal = Array4::from_shape_vec(
            [1, 5, 5, 1],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24.,
            ],
        )?;

        let c = convolution.compute(FeatureMap::nhwc(&signal))?;
        assert_eq!(
            c,
            Array4::from_shape_vec(
                [1, 3, 3, 1],
                vec![-46., -37., -28., -1., 8., 17., 44., 53., 62.],
            )?,
        );
        Ok(())
    }

    #[test]
    fn grouped_convolution() -> Result<(), Error> {
        let convolution = Convolution::f32(3)?.build(Params {
            strides: [1, 1],
            pads: [0; 4],
            dilation: [1, 1],
            groups: 2,
        })?;

        // All elements on the `i`th channel have value `i`.
        let signal = Array4::from_shape_vec(
            [1, 3, 3, 4],
            vec![
                1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1.,
                2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
            ],
        )?;

        let filters = Array4::from_shape_vec(
            [2, 3, 3, 2],
            vec![
                // 1st filter (applied to channels 0..2)
                1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
                // 2nd filter (applied to channels 2..4)
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
        )?;
        let expected_output = Array4::from_shape_vec(
            [1, 1, 1, 2],
            vec![
                -9.0, // = (1 + 1 + ... + 1) * (1 - 2)
                63.0, // = (1 + 1 + ... + 1) * (3 + 4)
            ],
        )?;

        let output = convolution.compute(FeatureMap::nhwc(&signal), &filters)?;
        assert_eq!(output, expected_output);
        Ok(())
    }

    #[test]
    fn grouped_i8_convolution() -> Result<(), Error> {
        let convolution = Convolution::i8(3)?.build(I8Params {
            common: Params {
                strides: [1, 1],
                pads: [0; 4],
                dilation: [1, 1],
                groups: 4,
            },
            bit_shift: 12,
            scale: I8Params::convert_scale(12, 1.0),
            output_bias: 0,
            signal_bias: 0,
            filter_bias: 0,
        })?;

        // All elements on the `i`th channel have value `i`.
        let signal = Array4::from_shape_vec(
            [1, 3, 3, 4],
            vec![
                1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                1, 2, 3, 4, 1, 2, 3, 4,
            ],
        )?;

        let filters = Array4::from_shape_vec(
            [4, 3, 3, 1],
            vec![
                1, -1, 1, -1, 1, -1, 1, -1, 1, //
                1, 1, 1, 1, 1, 1, 1, 1, 1, //
                1, -1, 1, -1, 1, -1, 1, -1, 1, //
                1, 1, 1, 1, 1, 1, 1, 1, 1, //
            ],
        )?;
        let expected_output = Array4::from_shape_vec(
            [1, 1, 1, 4],
            vec![
                1,  // 1 * (1 - 1 + 1 - ... + 1)
                18, // 2 * 9
                3,  // 3 * (1 - 1 + 1 - ... + 1)
                36, // 4 * 9
            ],
        )?;

        let output = convolution.compute(FeatureMap::nhwc(&signal), &filters)?;
        assert_eq!(output, expected_output);
        Ok(())
    }

    #[test]
    fn with_padding() -> Result<(), Error> {
        let convolution = Convolution::f32(3)?.build(Params {
            strides: [1, 1],
            pads: [1; 4],
            dilation: [1, 1],
            groups: 1,
        })?;

        let signal = Array4::from_shape_vec(
            [1, 5, 5, 1],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24.,
            ],
        )?;
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.0; 9])?;

        let c = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
        assert_eq!(
            c,
            Array4::from_shape_vec(
                [1, 5, 5, 1],
                vec![
                    12., 21., 27., 33., 24., 33., 54., 63., 72., 51., 63., 99., 108., 117., 81.,
                    93., 144., 153., 162., 111., 72., 111., 117., 123., 84.,
                ],
            )?,
        );
        Ok(())
    }

    #[test]
    fn with_strides() -> Result<(), Error> {
        let convolution = Convolution::f32(3)?.build(Params {
            strides: [2, 2],
            pads: [0; 4],
            dilation: [1, 1],
            groups: 1,
        })?;

        let signal = Array4::from_shape_vec(
            [1, 7, 5, 1],
            vec![
                0., 1., 2., 3., 4., //
                5., 6., 7., 8., 9., //
                10., 11., 12., 13., 14., //
                15., 16., 17., 18., 19., //
                20., 21., 22., 23., 24., //
                25., 26., 27., 28., 29., //
                30., 31., 32., 33., 34., //
            ],
        )?;
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.; 9])?;
        let expected_output =
            Array4::from_shape_vec([1, 3, 2, 1], vec![54., 72., 144., 162., 234., 252.])?;

        assert_eq!(
            convolution.compute(FeatureMap::nhwc(&signal), &filter)?,
            expected_output
        );
        Ok(())
    }

    #[test]
    fn with_strides_and_padding() -> Result<(), Error> {
        let convolution = Convolution::f32(3)?.build(Params {
            strides: [2, 2],
            pads: [1; 4],
            dilation: [1, 1],
            groups: 1,
        })?;

        let signal = Array4::from_shape_vec(
            [1, 7, 5, 1],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
                34.,
            ],
        )?;
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.; 9])?;

        let expected_output = Array4::from_shape_vec(
            [1, 4, 3, 1],
            vec![
                12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.,
            ],
        )?;

        assert_eq!(
            convolution.compute(FeatureMap::nhwc(&signal), &filter)?,
            expected_output
        );
        Ok(())
    }

    #[test]
    fn with_several_input_channels() -> Result<(), Error> {
        let convolution = Convolution::f32(3)?.build(Params {
            strides: [1, 1],
            pads: [1; 4],
            dilation: [1, 1],
            groups: 1,
        })?;

        let mut signal = vec![0.0; 100];
        for (i, val) in signal.iter_mut().enumerate() {
            *val = (i / 4) as f32;
        }
        let signal = Array4::from_shape_vec([1, 5, 5, 4], signal)?;
        let filter = Array4::from_shape_vec([1, 3, 3, 4], vec![1.; 36])?;
        let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;

        assert!((output[[0, 0, 0, 0]] - 48.0).abs() < f32::EPSILON);
        // 48 = 4 * (0 + 1 + 5 + 6), numbers in the upper left corner of the image.
        Ok(())
    }

    #[test]
    fn with_dilation() -> Result<(), Error> {
        let mut convolution = Convolution::f32(3)?.build(Params {
            strides: [1, 1],
            pads: [0; 4],
            groups: 1,
            dilation: [2, 2],
        })?;

        let signal = Array4::from_shape_vec(
            [1, 5, 5, 1],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, //
                6.0, 7.0, 8.0, 9.0, 10.0, //
                11.0, 12.0, 13.0, 14.0, 15.0, //
                16.0, 17.0, 18.0, 19.0, 20.0, //
                21.0, 22.0, 23.0, 24.0, 25.0, //
            ],
        )?;
        let filters = Array4::from_elem([1, 3, 3, 1], 1.0);

        // 117.0 = 1.0 + 3.0 + ... + 25.0
        let expected_output = Array4::from_elem([1, 1, 1, 1], 117.0);
        assert_eq!(
            convolution.compute(FeatureMap::nhwc(&signal), &filters)?,
            expected_output
        );

        convolution.set_params(Params {
            strides: [1, 1],
            pads: [1; 4],
            groups: 1,
            dilation: [2, 2],
        })?;

        let expected_output = Array4::from_shape_vec(
            [1, 3, 3, 1],
            vec![
                52.0, 78.0, 52.0, //
                78.0, 117.0, 78.0, //
                52.0, 78.0, 52.0, //
            ],
        )?;
        assert_eq!(
            convolution.compute(FeatureMap::nhwc(&signal), &filters)?,
            expected_output
        );
        Ok(())
    }

    #[test]
    fn rounding_in_i8_convolution() -> Result<(), Error> {
        const BIT_SHIFT: u8 = 8;
        let params = I8Params {
            common: Params::default(),
            bit_shift: BIT_SHIFT,
            scale: I8Params::convert_scale(BIT_SHIFT, 0.5),
            output_bias: 0,
            signal_bias: 0,
            filter_bias: 0,
        };
        let convolution = Convolution::i8(1)?.build(params)?;
        let signal = Array4::from_shape_vec([1, 2, 3, 1], vec![-7, -6, -5, 5, 6, 7])?;
        let filter = Array4::from_shape_vec([1, 1, 1, 1], vec![1])?;

        let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
        let expected_output = Array4::from_shape_vec([1, 2, 3, 1], vec![-4, -3, -2, 2, 3, 4])?;
        assert_eq!(output, expected_output);
        Ok(())
    }

    #[test]
    fn i8_convolution() -> Result<(), Error> {
        const BIT_SHIFT: u8 = 8;
        let params = I8Params {
            common: Params::default(),
            bit_shift: BIT_SHIFT,
            scale: I8Params::convert_scale(BIT_SHIFT, 1.0),
            output_bias: 0,
            signal_bias: 0,
            filter_bias: 0,
        };
        let mut convolution = Convolution::i8(3)?.build(params)?;

        let signal = vec![
            0, 1, 2, 3, 4, //
            5, 6, 7, 8, 9, //
            10, 11, 12, 13, 14, //
            -5, -6, -7, -8, -9, //
            0, -1, -2, -3, -4, //
        ];
        let signal = Array4::from_shape_vec([1, 5, 5, 1], signal)?;
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1; 9])?;

        let expected_output = vec![
            54, 63, 72, //
            33, 36, 39, //
            12, 9, 6, //
        ];
        let expected_output = Array4::from_shape_vec([1, 3, 3, 1], expected_output)?;
        let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
        assert_eq!(output, expected_output);

        // Check the same convolution with different scale / bias params.
        // We use post-conv transform |x| { x / 3 - 12 }.
        let expected_output = vec![
            6, 9, 12, //
            -1, 0, 1, //
            -8, -9, -10, //
        ];
        let expected_output = Array4::from_shape_vec([1, 3, 3, 1], expected_output)?;

        convolution.set_params(I8Params {
            common: Params::default(),
            bit_shift: BIT_SHIFT,
            scale: I8Params::convert_scale(BIT_SHIFT, 1.0 / 3.0),
            output_bias: -12,
            signal_bias: 0,
            filter_bias: 0,
        })?;
        let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
        assert_eq!(output, expected_output);

        // Check `filter_bias` / `signal_bias`.
        let signal = vec![
            0, 1, 2, 3, 4, //
            5, 6, 7, 8, 9, //
            10, 11, 12, 13, 14, //
            -5, -6, -7, -8, -9, //
            0, -1, -2, -3, -4, //
        ];
        let signal = Array4::from_shape_vec([1, 5, 5, 1], signal)? - 7;
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![0; 9])?;

        convolution.set_params(I8Params {
            common: Params::default(),
            output_bias: -12,
            filter_bias: 1,
            signal_bias: 7,
            bit_shift: BIT_SHIFT,
            scale: I8Params::convert_scale(BIT_SHIFT, 1.0 / 3.0),
        })?;
        let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
        assert_eq!(output, expected_output);
        Ok(())
    }

    #[test]
    fn i8_convolution_with_filter_bias() -> Result<(), Error> {
        const BIT_SHIFT: u8 = 8;
        const MULTIPLIER: i32 = 1 << (BIT_SHIFT as i32);

        let params = I8Params {
            common: Params::default(),
            bit_shift: BIT_SHIFT,
            scale: I8Params::convert_scale(BIT_SHIFT, 1.0 / 3.0),
            output_bias: 0,
            signal_bias: 0,
            filter_bias: 0,
        };
        let convolution = Convolution::i8(3)?.build(params)?;

        let signal = vec![
            0, 1, 2, 3, 4, //
            5, 6, 7, 8, 9, //
            10, 11, 12, 13, 14, //
            -5, -6, -7, -8, -9, //
            0, -1, -2, -3, -4, //
        ];
        let signal = Array4::from_shape_vec([1, 5, 5, 1], signal)?;
        let signal = FeatureMap::nhwc(&signal);
        let filter = Array4::from_shape_vec([2, 3, 3, 1], vec![1; 18])?;

        let expected_output = vec![
            // First filter output
            6, 9, 12, //
            -1, 0, 1, //
            -8, -9, -10, //
            // Second filter output
            17, 20, 23, //
            10, 11, 12, //
            3, 2, 1, //
        ];
        let expected_output =
            Array4::from_shape_vec([1, 2, 3, 3], expected_output)?.permuted_axes([0, 2, 3, 1]);

        let biases = &[-12 * MULTIPLIER, -MULTIPLIER];
        let output = convolution
            .compute_with_biases(signal, &filter, biases)
            .unwrap();
        assert_eq!(output, expected_output);

        // Check filter pinning.
        let convolution = convolution.with_biased_filters(&filter, biases)?;
        let output = convolution.compute(signal)?;
        assert_eq!(output, expected_output);

        let convolution = convolution.pin(FeatureMapShape {
            batch_size: 1,
            width: 5,
            height: 5,
            channels: 1,
        })?;
        let output = convolution.compute(signal)?;
        assert_eq!(output, expected_output);
        Ok(())
    }

    #[test]
    #[allow(clippy::deref_addrof)] // the problem is in the `ndarray::s!` macro
    fn f32_batching() -> Result<(), Error> {
        use ndarray::{s, stack};

        let mut rng = thread_rng();
        let conv = Convolution::f32(3)?.build(Params::default())?;
        let filters = Array4::from_shape_fn([2, 3, 3, 4], |_| rng.gen_range(-1.0, 1.0));
        let conv = conv.with_filters(&filters)?;

        for batch_size in 2..8 {
            // Test both NHWC and NCHW layouts
            let signal_shape = if batch_size % 2 == 0 {
                [batch_size, 5, 5, 4]
            } else {
                [batch_size, 4, 5, 5]
            };
            let to_map = if batch_size % 2 == 0 {
                FeatureMap::nhwc
            } else {
                FeatureMap::nchw
            };

            let signal = Array4::from_shape_fn(signal_shape, |_| rng.gen_range(-1.0, 1.0));
            let batched_output = conv.compute(to_map(signal.view()))?;

            let sample_outputs: Vec<_> = (0..batch_size)
                .map(|i| {
                    let sample_signal = signal.slice(s![i..=i, .., .., ..]);
                    conv.compute(to_map(sample_signal))
                })
                .collect::<Result<_, _>>()?;
            let sample_outputs: Vec<_> = sample_outputs.iter().map(Array4::view).collect();
            let stitched_output = stack(Axis(0), &sample_outputs)?;

            let max_diff = (batched_output.clone() - stitched_output.clone())
                .mapv(f32::abs)
                .fold(0.0, |acc, &x| if x > acc { x } else { acc });
            assert!(
                max_diff < f32::EPSILON,
                "batched={}, stitched={}",
                batched_output,
                stitched_output
            );
        }
        Ok(())
    }
}
