//! Convolution parameters.

use ocl::{
    builders::KernelBuilder,
    prm::{Uint2, Uint4},
};

/// General convolution parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Params {
    /// Strides along spatial dimensions.
    pub strides: [usize; 2],
    /// Pads along spatial dimensions. The first 2 values denote pads at the beginning of
    /// rows / columns, the second 2 values - pads at the end.
    pub pads: [usize; 4],
    /// Number of groups in the convolution. Each group of filters will be applied to
    /// a subset of input channels.
    pub groups: usize,
    /// Signal dilation along spatial dimensions.
    pub dilation: [usize; 2],
}

impl Default for Params {
    fn default() -> Self {
        Self {
            strides: [1, 1],
            pads: [0; 4],
            groups: 1,
            dilation: [1, 1],
        }
    }
}

impl Params {
    pub(crate) fn pass_as_arguments(self, builder: &mut KernelBuilder) {
        builder.arg_named("params", ClParams::from(self));
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(C, packed)]
pub struct ClParams {
    strides: Uint2,
    pads: Uint4,
    groups: u32,
    dilation: Uint2,
}

impl From<Params> for ClParams {
    fn from(value: Params) -> Self {
        ClParams {
            strides: Uint2::new(value.strides[0] as u32, value.strides[1] as u32),
            pads: Uint4::new(
                value.pads[0] as u32,
                value.pads[1] as u32,
                value.pads[2] as u32,
                value.pads[3] as u32,
            ),
            groups: value.groups as u32,
            dilation: Uint2::new(value.dilation[0] as u32, value.dilation[1] as u32),
        }
    }
}

// Safety ensured by the same alignment here and in OCL code.
unsafe impl ocl::OclPrm for ClParams {}

/// Params for the quantized convolution.
///
/// See [`Convolution`] docs for details how to set these parameters.
///
/// [`Convolution`]: struct.Convolution.html#connection-to-real-value-convolution
#[derive(Debug, Clone, Copy)]
pub struct I8Params {
    /// Common parameters.
    pub common: Params,
    /// Upscale bit shift.
    pub bit_shift: u8,
    /// Fixed-point scale of the post-convolution transform.
    pub scale: i32,
    /// Bias for the post-convolution transform.
    pub output_bias: i32,
    /// Bias for the signal.
    pub signal_bias: i32,
    /// Bias for the filters.
    pub filter_bias: i32,
}

impl From<I8Params> for Params {
    fn from(value: I8Params) -> Self {
        value.common
    }
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
pub struct ClI8Params {
    strides: Uint2,
    pads: Uint4,
    scale: i32,
    output_bias: i32,
    signal_bias: i32,
    filter_bias: i32,
}

impl From<I8Params> for ClI8Params {
    fn from(value: I8Params) -> Self {
        let common_params = ClParams::from(value.common);
        ClI8Params {
            strides: common_params.strides,
            pads: common_params.pads,
            scale: value.scale,
            output_bias: value.output_bias,
            signal_bias: value.signal_bias,
            filter_bias: value.filter_bias,
        }
    }
}

// Safety ensured by the same alignment here and in OCL code.
unsafe impl ocl::OclPrm for ClI8Params {}
