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
}

impl Default for Params {
    fn default() -> Self {
        Self {
            strides: [1, 1],
            pads: [0; 4],
        }
    }
}

impl Params {
    pub(crate) fn pass_as_arguments(self, builder: &mut KernelBuilder) {
        builder
            .arg_named(
                "strides",
                Uint2::new(self.strides[0] as u32, self.strides[1] as u32),
            )
            .arg_named(
                "pads",
                Uint4::new(
                    self.pads[0] as u32,
                    self.pads[1] as u32,
                    self.pads[2] as u32,
                    self.pads[3] as u32,
                ),
            );
    }
}

/// Params for the quantized convolution.
///
/// See [`Convolution`] docs for details how to set these parameters.
///
/// [`Convolution`]: struct.Convolution.html#connection-to-real-value-convolution
#[derive(Debug, Clone, Copy)]
pub struct I8Params {
    /// Strides (spacial distances between sequential application of filters).
    pub strides: [usize; 2],
    /// Pads for the signal.
    pub pads: [usize; 4],
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
        Self {
            strides: value.strides,
            pads: value.pads,
        }
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
