//! Convolution parameters.

use ocl::{
    prm::{Uint2, Uint4},
    OclPrm,
};

use std::{convert::TryFrom, fmt};

use crate::{
    buffers::{Filters, Layout, Pinned},
    ConvElement,
};

/// General convolution parameters.
///
/// The parameters translate to the parameters of the [`Conv` ONNX operator][onnx-conv].
///
/// [onnx-conv]: https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Params {
    /// Strides along spatial dimensions.
    pub strides: [usize; 2],
    /// Pads along spatial dimensions. The first 2 values denote pads at the beginning of
    /// rows / columns, the second 2 values – pads at the end.
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
            strides: Uint2::new(
                u32::try_from(value.strides[0]).expect("Cannot convert stride to `u32`"),
                u32::try_from(value.strides[1]).expect("Cannot convert stride to `u32`"),
            ),
            pads: Uint4::new(
                u32::try_from(value.pads[0]).expect("Cannot convert pad to `u32`"),
                u32::try_from(value.pads[1]).expect("Cannot convert pad to `u32`"),
                u32::try_from(value.pads[2]).expect("Cannot convert pad to `u32`"),
                u32::try_from(value.pads[3]).expect("Cannot convert pad to `u32`"),
            ),
            groups: u32::try_from(value.groups).expect("Cannot convert groups to `u32`"),
            dilation: Uint2::new(
                u32::try_from(value.dilation[0]).expect("Cannot convert dilation to `u32`"),
                u32::try_from(value.dilation[1]).expect("Cannot convert dilation to `u32`"),
            ),
        }
    }
}

// Safety ensured by the same alignment here and in OCL code.
unsafe impl OclPrm for ClParams {}

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
    ///
    /// # Panics
    ///
    /// - Panics if the converted value is outside the `i32` bounds.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    // ^-- precision loss is OK in general, and we perform `i32` range check before
    // using `_ as i32` on an `f32` value.
    pub fn convert_scale(bit_shift: u8, scale: f32) -> i32 {
        let scale = (2.0_f32.powi(i32::from(bit_shift)) * scale).round();
        assert!(
            scale >= i32::min_value() as f32 && scale <= i32::max_value() as f32,
            "Scale is out of `i32` bounds"
        );
        scale as i32
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(C, packed)]
pub struct ClI8Params {
    strides: Uint2,
    pads: Uint4,
    group: u32,
    dilation: Uint2,
    bit_shift: i32,
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
            group: common_params.groups,
            dilation: common_params.dilation,
            bit_shift: i32::from(value.bit_shift),
            scale: value.scale,
            output_bias: value.output_bias,
            signal_bias: value.signal_bias,
            filter_bias: value.filter_bias,
        }
    }
}

// Safety ensured by the same alignment here and in OCL code.
unsafe impl OclPrm for ClI8Params {}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
#[repr(C, packed)]
pub struct OutputParams {
    pub batch_size: u32,
    pub layout: Layout,
}

unsafe impl OclPrm for OutputParams {}

impl Default for OutputParams {
    fn default() -> Self {
        Self {
            batch_size: 0,
            layout: Layout::ChannelsLast,
        }
    }
}

/// Type that can be associated with convolution parameters.
pub trait WithParams {
    /// Parameters of the convolution.
    type Params: Clone + fmt::Debug + Into<Self::ClParams>;
    /// OpenCL-friendly version of parameters.
    type ClParams: OclPrm;

    /// Extracts generic parameters.
    fn get_generic_params(params: &Self::Params) -> Params;
}

impl WithParams for f32 {
    type Params = Params;
    type ClParams = ClParams;

    fn get_generic_params(params: &Self::Params) -> Params {
        *params
    }
}

impl WithParams for i8 {
    type Params = I8Params;
    type ClParams = ClI8Params;

    fn get_generic_params(params: &Self::Params) -> Params {
        params.common
    }
}

impl<T: ConvElement> WithParams for Filters<T> {
    type Params = <T as WithParams>::Params;
    type ClParams = <T as WithParams>::ClParams;

    fn get_generic_params(params: &Self::Params) -> Params {
        T::get_generic_params(params)
    }
}

impl<T: ConvElement> WithParams for Pinned<T> {
    type Params = <T as WithParams>::Params;
    type ClParams = <T as WithParams>::ClParams;

    fn get_generic_params(params: &Self::Params) -> Params {
        T::get_generic_params(params)
    }
}
