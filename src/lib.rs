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
//! # Implementation details
//!
//! For now, the crate implements convolutions on two numerical formats:
//!
//! - Single-precision floats (`f32`) with [`Convolution`]
//! - Signed 8-bit integers with 32-bit multiply-add accumulator (this format is frequently denoted
//!   `int8/32` in deep learning literature), with [`I8Convolution`]
//!
//! In both cases, the convolution uses output-stationary workflow (see, e.g., [this paper] for
//! the definition); that is, each element of the output tensor is computed in a single run
//! of the OpenCL kernel. This minimizes memory overhead, but may not be the fastest algorithm.
//!
//! [Convolution]: https://en.wikipedia.org/wiki/Convolution
//! [deep learning]: https://en.wikipedia.org/wiki/Deep_learning
//! [cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
//! [OpenCL]: https://www.khronos.org/opencl/
//! [this paper]: https://dl.acm.org/citation.cfm?id=3001177
//! [`Convolution`]: struct.Convolution.html
//! [`I8Convolution`]: struct.I8Convolution.html

use ndarray::{Array3, ArrayView3, ArrayView4};
use ocl::{
    flags,
    prm::{Uint2, Uint3, Uint4},
    Buffer, ProQue,
};

use std::{borrow::Cow, convert::TryFrom};

const SOURCE: &str = include_str!("conv.cl");
const I8_SOURCE: &str = include_str!("i8_conv.cl");

/// Convolution of a specific filter size on single-precision floats.
///
/// # Examples
///
/// ```
/// use ndarray::{Array3, Array4};
/// use rand::{Rng, thread_rng};
/// use std::iter;
/// # use ocl_convolution::Convolution;
///
/// # fn main() -> Result<(), ocl::Error> {
/// let convolution = Convolution::new(3)?;
/// // Generate random signal with 6x6 spacial dims and 3 channels.
/// let mut rng = thread_rng();
/// let mut signal = Array3::zeros([6, 6, 3]);
/// signal.map_mut(|x| *x = rng.gen_range(-1.0, 1.0));
/// // Construct two 3x3 spacial filters.
/// let mut filters = Array4::zeros([2, 3, 3, 3]);
/// filters.map_mut(|x| *x = rng.gen_range(-1.0, 1.0));
/// // Perform the convolution. The output should have 4x4 spacial dims
/// // and contain 2 channels (1 per each filter).
/// let output = convolution.compute(
///     signal.view(),
///     filters.view(),
///     [1, 1],
///     [0; 4],
/// )?;
/// assert_eq!(output.shape(), [2, 4, 4]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Convolution {
    size: usize,
    program: ProQue,
}

impl Convolution {
    /// Creates a convolution with a specific spatial size.
    pub fn new(size: usize) -> ocl::Result<Self> {
        assert_eq!(size % 2, 1, "Even convolution sizes are not supported");

        let src = format!("#define FILTER_SIZE {}\n{}", size, SOURCE);
        let program = ProQue::builder().src(src).build()?;
        Ok(Self { size, program })
    }

    /// Spatial size of the convolution.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Performs convolution on the provided `signal` and `filters`.
    ///
    /// # Parameters
    ///
    /// - `signal` should have `HxWxC` layout (i.e., the channel dimension is the inner-most one).
    /// - `filters` should have `MxK_HxK_WxC` layout, where `M` is the number of filters,
    ///   `K_H` and `K_W` are spatial dimensions of a filter, `C` is the number of input channel.
    ///
    /// # Return value
    ///
    /// The output will have form `MxH'xW'`. An error means something wrong with OpenCL.
    ///
    /// # Panics
    ///
    /// - The method will panic if `filters` do not have expected spatial dimensions, i.e.,
    ///   `self.size() x self.size()`.
    /// - Likewise, the method will panic if the number of input channels differs from number of
    ///   channels in `filters`.
    pub fn compute(
        &self,
        signal: ArrayView3<f32>,
        filters: ArrayView4<f32>,
        strides: [usize; 2],
        pads: [usize; 4],
    ) -> ocl::Result<Array3<f32>> {
        let in_h = signal.shape()[0];
        let in_w = signal.shape()[1];
        let in_channels = signal.shape()[2];
        let filter_count = filters.shape()[0];

        let out_h = (in_h - self.size + pads[0] + pads[2] + strides[0]) / strides[0];
        let out_w = (in_w - self.size + pads[1] + pads[3] + strides[1]) / strides[1];

        assert!(
            filters.shape()[1] == self.size && filters.shape()[2] == self.size,
            "Invalid filter shape"
        );
        assert_eq!(
            in_channels,
            filters.shape()[3],
            "Channel dimensionality in signal and filters must agree"
        );

        let program = &self.program;

        let signal_slice = signal
            .as_slice()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(signal.iter().cloned().collect()));
        let signal_buffer = Buffer::builder()
            .queue(program.queue().clone())
            .len(<[usize; 3]>::try_from(signal.shape()).unwrap())
            .flags(flags::MEM_READ_ONLY)
            .copy_host_slice(signal_slice.as_ref())
            .build()?;

        let filters_slice = filters
            .as_slice()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(filters.iter().cloned().collect()));
        let filters_buffer = Buffer::builder()
            .queue(program.queue().clone())
            .len(filters.shape().iter().product::<usize>())
            .flags(flags::MEM_READ_ONLY)
            .copy_host_slice(filters_slice.as_ref())
            .build()?;

        let output_buffer = Buffer::builder()
            .queue(program.queue().clone())
            .len([filter_count, out_h, out_w])
            .flags(flags::MEM_HOST_READ_ONLY)
            .build()?;

        let kernel = program
            .kernel_builder("conv")
            .global_work_size([out_h * self.size, out_w * self.size, filter_count])
            .local_work_size([self.size, self.size])
            .arg_named("convolved", &output_buffer)
            .arg_named("signal", &signal_buffer)
            .arg_named(
                "signal_dims",
                Uint3::new(in_h as u32, in_w as u32, in_channels as u32),
            )
            .arg_named("filters", &filters_buffer)
            .arg_named("strides", Uint2::new(strides[0] as u32, strides[1] as u32))
            .arg_named(
                "pads",
                Uint4::new(
                    pads[0] as u32,
                    pads[1] as u32,
                    pads[2] as u32,
                    pads[3] as u32,
                ),
            )
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        let mut output_data = vec![0.0_f32; output_buffer.len()];
        output_buffer.read(&mut output_data).enq()?;
        let output = Array3::from_shape_vec([filter_count, out_h, out_w], output_data).unwrap();
        Ok(output)
    }
}

/// Params for `I8Convolution`.
///
/// See [`I8Convolution`] docs for details how to set these parameters.
///
/// [`I8Convolution`]: struct.I8Convolution.html
#[derive(Debug, Clone, Copy)]
pub struct I8Params {
    /// Strides (spacial distances between sequential application of filters).
    pub strides: [usize; 2],
    /// Pads for the signal.
    pub pads: [usize; 4],
    /// Fixed-point scale of the post-convolution transform.
    pub scale: i32,
    /// Bias for the post-convolution transform.
    pub output_bias: i32,
    /// Bias for the signal.
    pub signal_bias: i32,
    /// Bias for the filters.
    pub filter_bias: i32,
}

impl I8Params {
    /// Converts `scale` to fixed-point presentation. The resulting value can be used
    /// as the `scale` field.
    pub fn convert_scale(bit_shift: u8, scale: f32) -> i32 {
        (((1 << i32::from(bit_shift)) as f32) * scale).round() as i32
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(C, packed)]
struct ClI8Params {
    strides: Uint2,
    pads: Uint4,
    scale: i32,
    output_bias: i32,
    signal_bias: i32,
    filter_bias: i32,
}

impl From<I8Params> for ClI8Params {
    fn from(value: I8Params) -> Self {
        ClI8Params {
            strides: Uint2::new(value.strides[0] as u32, value.strides[1] as u32),
            pads: Uint4::new(
                value.pads[0] as u32,
                value.pads[1] as u32,
                value.pads[2] as u32,
                value.pads[3] as u32,
            ),
            scale: value.scale,
            output_bias: value.output_bias,
            signal_bias: value.signal_bias,
            filter_bias: value.filter_bias,
        }
    }
}

// Safety ensured by the same alignment here and in OCL code.
unsafe impl ocl::OclPrm for ClI8Params {}

/// Quantized convolution over signed 8-bit integers with the specified filter size.
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
/// Due to use of `i8` inputs, computations are performed much faster than on `f32` inputs
/// (the difference manifests most on the specialized hardware, but it is seen in this
/// OpenCL-powered implementation as well).
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
/// `scale` is represented as a fixed-point number with [`bit_shift()`] binary digits after
/// the point. Note that filter biases `B` are assumed to be unbiased and upscaled; they are
/// not transformed during the computation.
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
/// [`bit_shift()`]: #method.bit_shift
/// [this paper]: https://arxiv.org/abs/1805.00907
///
/// # Examples
///
/// ```
/// use ndarray::{Array3, Array4};
/// use rand::{Rng, thread_rng};
/// use std::iter;
/// # use ocl_convolution::{I8Convolution, I8Params};
///
/// # fn main() -> Result<(), ocl::Error> {
/// const BIT_SHIFT: u8 = 16;
/// let convolution = I8Convolution::new(3, BIT_SHIFT)?;
/// // Generate random signal with 6x6 spacial dims and 3 channels.
/// let mut rng = thread_rng();
/// let mut signal = Array3::zeros([6, 6, 3]);
/// signal.map_mut(|x| *x = rng.gen_range(-127, 127));
/// // Construct two 3x3 spacial filters.
/// let mut filters = Array4::zeros([2, 3, 3, 3]);
/// filters.map_mut(|x| *x = rng.gen_range(-127, 127));
/// // Perform the convolution. The output should have 4x4 spacial dims
/// // and contain 2 channels (1 per each filter).
/// let output = convolution.compute(
///     signal.view(),
///     filters.view(),
///     None, // no filter biases
///     I8Params {
///         strides: [1, 1],
///         pads: [0; 4],
///         // These params are found by profiling; here, they are
///         // chosen randomly.
///         scale: I8Params::convert_scale(BIT_SHIFT, 0.1),
///         output_bias: -10,
///         signal_bias: 20,
///         filter_bias: -5,
///     },
/// )?;
/// assert_eq!(output.shape(), [2, 4, 4]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct I8Convolution {
    size: usize,
    bit_shift: u8,
    program: ProQue,
}

impl I8Convolution {
    /// Creates a convolution with the specified size and bit shift.
    pub fn new(size: usize, bit_shift: u8) -> ocl::Result<Self> {
        assert_eq!(size % 2, 1, "Even convolution sizes are not supported");

        let src = format!(
            "#define FILTER_SIZE {}\n#define BIT_SHIFT {}\n{}",
            size, bit_shift, I8_SOURCE
        );
        let program = ProQue::builder().src(src).build()?;
        Ok(Self {
            size,
            bit_shift,
            program,
        })
    }

    /// Spatial size of the convolution.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Bit shift used in fixed-point number operations.
    pub fn bit_shift(&self) -> u8 {
        self.bit_shift
    }

    /// Performs convolution on the provided `signal` and `filters`.
    ///
    /// # Parameters
    ///
    /// - `signal` should have `HxWxC` layout (i.e., the channel dimension is the inner-most one).
    /// - `filters` should have `MxK_HxK_WxC` layout, where `M` is the number of filters,
    ///   `K_H` and `K_W` are spatial dimensions of a filter, `C` is the number of input channel.
    /// - `filter_biases`, if present, should have `M` elements.
    ///
    /// # Return value
    ///
    /// The output will have form `MxH'xW'`. An error means something wrong with OpenCL.
    ///
    /// # Panics
    ///
    /// - The method will panic if `filters` do not have expected spatial dimensions, i.e.,
    ///   `self.size() x self.size()`.
    /// - Likewise, the method will panic if the number of input channels differs from number of
    ///   channels in `filters`, or if `filter_biases` length does not agree
    ///   with the `filter` dimensions.
    pub fn compute(
        &self,
        signal: ArrayView3<i8>,
        filters: ArrayView4<i8>,
        filter_biases: Option<&[i32]>,
        params: I8Params,
    ) -> ocl::Result<Array3<i8>> {
        let in_h = signal.shape()[0];
        let in_w = signal.shape()[1];
        let in_channels = signal.shape()[2];
        let filter_count = filters.shape()[0];

        let pads = params.pads;
        let strides = params.strides;
        let out_h = (in_h - self.size + pads[0] + pads[2] + strides[0]) / strides[0];
        let out_w = (in_w - self.size + pads[1] + pads[3] + strides[1]) / strides[1];

        assert!(
            filters.shape()[1] == self.size && filters.shape()[2] == self.size,
            "Invalid filter shape"
        );
        assert_eq!(
            in_channels,
            filters.shape()[3],
            "Channel dimensionality in signal and filters must agree"
        );
        if let Some(ref biases) = filter_biases {
            assert_eq!(
                biases.len(),
                filter_count,
                "Channel biases should have the same number of channels as the output"
            );
        }

        let program = &self.program;

        let signal_slice = signal
            .as_slice()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(signal.iter().cloned().collect()));
        let signal_buffer = Buffer::builder()
            .queue(program.queue().clone())
            .len(<[usize; 3]>::try_from(signal.shape()).unwrap())
            .flags(flags::MEM_READ_ONLY)
            .copy_host_slice(signal_slice.as_ref())
            .build()?;

        let filters_slice = filters
            .as_slice()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(filters.iter().cloned().collect()));
        let filters_buffer = Buffer::builder()
            .queue(program.queue().clone())
            .len(filters.shape().iter().product::<usize>())
            .flags(flags::MEM_READ_ONLY)
            .copy_host_slice(filters_slice.as_ref())
            .build()?;

        let output_buffer = Buffer::builder()
            .queue(program.queue().clone())
            .len([filter_count, out_h, out_w])
            .flags(flags::MEM_HOST_READ_ONLY)
            .build()?;

        let kernel = program
            .kernel_builder("conv")
            .global_work_size([out_h * self.size, out_w * self.size, filter_count])
            .local_work_size([self.size, self.size])
            .arg_named("convolved", &output_buffer)
            .arg_named("signal", &signal_buffer)
            .arg_named(
                "signal_dims",
                Uint3::new(in_h as u32, in_w as u32, in_channels as u32),
            )
            .arg_named("filters", &filters_buffer)
            .arg_named("filter_biases", None::<&Buffer<i32>>)
            .arg_named("params", ClI8Params::from(params))
            .build()?;
        if let Some(biases) = filter_biases {
            let biases = Buffer::builder()
                .queue(program.queue().clone())
                .len(biases.len())
                .flags(flags::MEM_READ_ONLY)
                .copy_host_slice(biases)
                .build()?;
            kernel.set_arg("filter_biases", &biases)?;
        }

        unsafe {
            kernel.enq()?;
        }

        let mut output_data = vec![0; output_buffer.len()];
        output_buffer.read(&mut output_data).enq()?;
        let output = Array3::from_shape_vec([filter_count, out_h, out_w], output_data).unwrap();
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array4;
    use std::f32;

    use super::*;

    #[test]
    fn basics() {
        let convolution = Convolution::new(3).unwrap();
        let signal = Array3::from_shape_vec(
            [5, 5, 1],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24.,
            ],
        )
        .unwrap();
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.0; 9]).unwrap();

        let c = convolution
            .compute(signal.view(), filter.view(), [1, 1], [0; 4])
            .unwrap();
        assert_eq!(
            c,
            Array3::from_shape_vec(
                [1, 3, 3],
                vec![54., 63., 72., 99., 108., 117., 144., 153., 162.],
            )
            .unwrap(),
        );
    }

    #[test]
    fn with_padding() {
        let convolution = Convolution::new(3).unwrap();
        let signal = Array3::from_shape_vec(
            [5, 5, 1],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24.,
            ],
        )
        .unwrap();
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.0; 9]).unwrap();

        let c = convolution
            .compute(signal.view(), filter.view(), [1, 1], [1; 4])
            .unwrap();
        assert_eq!(
            c,
            Array3::from_shape_vec(
                [1, 5, 5],
                vec![
                    12., 21., 27., 33., 24., 33., 54., 63., 72., 51., 63., 99., 108., 117., 81.,
                    93., 144., 153., 162., 111., 72., 111., 117., 123., 84.,
                ],
            )
            .unwrap(),
        );
    }

    #[test]
    fn with_strides() {
        let convolution = Convolution::new(3).unwrap();

        let signal = Array3::from_shape_vec(
            [7, 5, 1],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
                34.,
            ],
        )
        .unwrap();

        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.; 9]).unwrap();

        let expected_output =
            Array3::from_shape_vec([1, 3, 2], vec![54., 72., 144., 162., 234., 252.]).unwrap();

        assert_eq!(
            convolution
                .compute(signal.view(), filter.view(), [2, 2], [0; 4])
                .unwrap(),
            expected_output
        );
    }

    #[test]
    fn with_strides_and_padding() {
        let convolution = Convolution::new(3).unwrap();

        let signal = Array3::from_shape_vec(
            [7, 5, 1],
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
                34.,
            ],
        )
        .unwrap();

        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.; 9]).unwrap();

        let expected_output = Array3::from_shape_vec(
            [1, 4, 3],
            vec![
                12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.,
            ],
        )
        .unwrap();

        assert_eq!(
            convolution
                .compute(signal.view(), filter.view(), [2, 2], [1; 4])
                .unwrap(),
            expected_output
        );
    }

    #[test]
    fn with_several_input_channels() {
        let convolution = Convolution::new(3).unwrap();

        let mut signal = vec![0.0; 100];
        for (i, val) in signal.iter_mut().enumerate() {
            *val = (i / 4) as f32;
        }
        let signal = Array3::from_shape_vec([5, 5, 4], signal).unwrap();

        let filter = Array4::from_shape_vec([1, 3, 3, 4], vec![1.; 36]).unwrap();
        let output = convolution
            .compute(signal.view(), filter.view(), [1, 1], [1; 4])
            .unwrap();

        assert!((output[[0, 0, 0]] - 48.0).abs() < f32::EPSILON);
        // 48 = 4 * (0 + 1 + 5 + 6), numbers in the upper left corner of the image.
    }

    #[test]
    fn rounding_in_i8_convolution() {
        const BIT_SHIFT: u8 = 8;
        let convolution = I8Convolution::new(1, BIT_SHIFT).unwrap();
        let signal = Array3::from_shape_vec([2, 3, 1], vec![-7, -6, -5, 5, 6, 7]).unwrap();
        let filter = Array4::from_shape_vec([1, 1, 1, 1], vec![1]).unwrap();
        let params = I8Params {
            strides: [1, 1],
            pads: [0; 4],
            scale: I8Params::convert_scale(BIT_SHIFT, 0.5),
            output_bias: 0,
            signal_bias: 0,
            filter_bias: 0,
        };

        let output = convolution
            .compute(signal.view(), filter.view(), None, params)
            .unwrap();
        let expected_output = Array3::from_shape_vec([1, 2, 3], vec![-4, -3, -2, 2, 3, 4]).unwrap();
        assert_eq!(output, expected_output);
    }

    #[test]
    fn i8_convolution() {
        const BIT_SHIFT: u8 = 8;
        let convolution = I8Convolution::new(3, BIT_SHIFT as u8).unwrap();

        let signal = vec![
            0, 1, 2, 3, 4, //
            5, 6, 7, 8, 9, //
            10, 11, 12, 13, 14, //
            -5, -6, -7, -8, -9, //
            0, -1, -2, -3, -4, //
        ];
        let signal = Array3::from_shape_vec([5, 5, 1], signal).unwrap();
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1; 9]).unwrap();

        let expected_output = vec![
            54, 63, 72, //
            33, 36, 39, //
            12, 9, 6, //
        ];
        let expected_output = Array3::from_shape_vec([1, 3, 3], expected_output).unwrap();

        let params = I8Params {
            strides: [1, 1],
            pads: [0; 4],
            scale: I8Params::convert_scale(BIT_SHIFT, 1.0),
            output_bias: 0,
            signal_bias: 0,
            filter_bias: 0,
        };
        let output = convolution
            .compute(signal.view(), filter.view(), None, params)
            .unwrap();
        assert_eq!(output, expected_output);

        // Check the same convolution with different scale / bias params.
        // We use post-conv transform |x| { x / 3 - 12 }.
        let expected_output = vec![
            6, 9, 12, //
            -1, 0, 1, //
            -8, -9, -10, //
        ];
        let expected_output = Array3::from_shape_vec([1, 3, 3], expected_output).unwrap();

        let params = I8Params {
            strides: [1, 1],
            pads: [0; 4],
            scale: I8Params::convert_scale(BIT_SHIFT, 1.0 / 3.0),
            output_bias: -12,
            signal_bias: 0,
            filter_bias: 0,
        };;
        let output = convolution
            .compute(signal.view(), filter.view(), None, params)
            .unwrap();
        assert_eq!(output, expected_output);

        // Check `filter_bias` / `signal_bias`.
        let signal = vec![
            0, 1, 2, 3, 4, //
            5, 6, 7, 8, 9, //
            10, 11, 12, 13, 14, //
            -5, -6, -7, -8, -9, //
            0, -1, -2, -3, -4, //
        ];
        let signal = Array3::from_shape_vec([5, 5, 1], signal).unwrap() - 7;
        let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![0; 9]).unwrap();

        let params = I8Params {
            strides: [1, 1],
            pads: [0; 4],
            output_bias: -12,
            filter_bias: 1,
            signal_bias: 7,
            scale: I8Params::convert_scale(BIT_SHIFT, 1.0 / 3.0),
        };
        let output = convolution
            .compute(signal.view(), filter.view(), None, params)
            .unwrap();
        assert_eq!(output, expected_output);
    }

    #[test]
    fn i8_convolution_with_filter_bias() {
        const BIT_SHIFT: u8 = 8;
        const MULTIPLIER: i32 = 1 << (BIT_SHIFT as i32);

        let convolution = I8Convolution::new(3, BIT_SHIFT).unwrap();

        let signal = vec![
            0, 1, 2, 3, 4, //
            5, 6, 7, 8, 9, //
            10, 11, 12, 13, 14, //
            -5, -6, -7, -8, -9, //
            0, -1, -2, -3, -4, //
        ];
        let signal = Array3::from_shape_vec([5, 5, 1], signal).unwrap();
        let filter = Array4::from_shape_vec([2, 3, 3, 1], vec![1; 18]).unwrap();

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
        let expected_output = Array3::from_shape_vec([2, 3, 3], expected_output).unwrap();

        let params = I8Params {
            strides: [1, 1],
            pads: [0; 4],
            scale: I8Params::convert_scale(BIT_SHIFT, 1.0 / 3.0),
            output_bias: 0,
            signal_bias: 0,
            filter_bias: 0,
        };
        let biases = &[-12 * MULTIPLIER, -MULTIPLIER];
        let output = convolution
            .compute(signal.view(), filter.view(), Some(biases), params)
            .unwrap();
        assert_eq!(output, expected_output);
    }
}
