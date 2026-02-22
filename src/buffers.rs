//! OpenCL buffer wrappers.

use std::{borrow::Cow, convert::TryFrom};

use ndarray::{Array4, ArrayView4};
use ocl::{Buffer, Kernel, flags, prm::Uint3};

use crate::{
    ConvElement, Params,
    base::Base,
    params::{OutputParams, WithParams},
};

/// Shape of a [`FeatureMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureMapShape {
    /// Number of samples constituting the map.
    pub batch_size: u32,
    /// Spatial width.
    pub width: u32,
    /// Spatial height.
    pub height: u32,
    /// Number of channels.
    pub channels: u32,
}

impl FeatureMapShape {
    fn from_nhwc_slice(shape: &[usize]) -> Self {
        assert_eq!(shape.len(), 4);
        FeatureMapShape {
            batch_size: u32::try_from(shape[0]).expect("Cannot convert batch size to `u32`"),
            height: u32::try_from(shape[1]).expect("Cannot convert height to `u32`"),
            width: u32::try_from(shape[2]).expect("Cannot convert width to `u32`"),
            channels: u32::try_from(shape[3]).expect("Cannot convert channel count to `u32`"),
        }
    }

    fn from_nchw_slice(shape: &[usize]) -> Self {
        assert_eq!(shape.len(), 4);
        FeatureMapShape {
            batch_size: u32::try_from(shape[0]).expect("Cannot convert batch size to `u32`"),
            height: u32::try_from(shape[2]).expect("Cannot convert height to `u32`"),
            width: u32::try_from(shape[3]).expect("Cannot convert width to `u32`"),
            channels: u32::try_from(shape[1]).expect("Cannot convert channel count to `u32`"),
        }
    }

    fn buffer_len(self) -> usize {
        self.batch_size as usize
            * self.width as usize
            * self.height as usize
            * self.channels as usize
    }

    fn as_array(self, layout: Layout) -> [usize; 4] {
        match layout {
            Layout::ChannelsFirst => [
                self.batch_size as usize,
                self.channels as usize,
                self.height as usize,
                self.width as usize,
            ],
            Layout::ChannelsLast => [
                self.batch_size as usize,
                self.height as usize,
                self.width as usize,
                self.channels as usize,
            ],
        }
    }
}

/// Memory layout of a [`FeatureMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Layout {
    /// `NCHW` / "channels-first" layout. In this layout, channels are an outer dimension compared
    /// to spatial width and height.
    ChannelsFirst = 0,
    /// `NHWC` / "channels-last" layout. In this layout, channels are the innermost dimension.
    ///
    /// This layout is preferred because it is used internally by the OpenCL code in order
    /// to efficiently vectorize multiply-add operations.
    ChannelsLast = 1,
}

/// Feature map, i.e., a signal or output of the convolution operation.
///
/// Internally, a `FeatureMap` is a thin wrapper around [`ArrayView`](ndarray::ArrayView)
/// that additionally indicates the memory layout of the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureMap<'a, T> {
    layout: Layout,
    inner: ArrayView4<'a, T>,
    shape: FeatureMapShape,
}

impl<'a, T: ConvElement> FeatureMap<'a, T> {
    /// Constructs a map from an NCHW-ordered tensor.
    ///
    /// # Panics
    ///
    /// - Panics if `array` dimensions do not fit into `u32` integers.
    pub fn nchw(array: impl Into<ArrayView4<'a, T>>) -> Self {
        let array = array.into();
        Self {
            layout: Layout::ChannelsFirst,
            shape: FeatureMapShape::from_nchw_slice(array.shape()),
            inner: array,
        }
    }

    /// Constructs a map from an NHWC-ordered tensor.
    ///
    /// # Panics
    ///
    /// - Panics if `array` dimensions do not fit into `u32` integers.
    pub fn nhwc(array: impl Into<ArrayView4<'a, T>>) -> Self {
        let array = array.into();
        Self {
            layout: Layout::ChannelsLast,
            shape: FeatureMapShape::from_nhwc_slice(array.shape()),
            inner: array,
        }
    }

    /// Gets the layout of this map.
    pub fn layout(self) -> Layout {
        self.layout
    }

    /// Gets the shape of this map.
    pub fn shape(self) -> FeatureMapShape {
        self.shape
    }

    fn to_nhwc(self) -> ArrayView4<'a, T> {
        match self.layout {
            Layout::ChannelsFirst => self.inner.permuted_axes([0, 2, 3, 1]),
            Layout::ChannelsLast => self.inner,
        }
    }
}

/// Container for convolution filters and optionally filter biases.
#[derive(Debug, Clone)]
pub(crate) struct Filters<T: ConvElement> {
    inner: Buffer<T>,
    biases: Option<Buffer<T::Acc>>,
    filter_count: u32,
    channel_count: u32,
}

impl<T: ConvElement> Filters<T> {
    pub fn filter_count(&self) -> u32 {
        self.filter_count
    }

    pub fn channel_count(&self) -> u32 {
        self.channel_count
    }

    pub fn new<U: WithParams>(
        filters: &ArrayView4<'_, T>,
        biases: Option<&[T::Acc]>,
        conv: &Base<U>,
    ) -> ocl::Result<Self> {
        assert!(
            filters.shape()[1] == conv.size() as usize
                && filters.shape()[2] == conv.size() as usize,
            "Invalid filter shape: expected {0}x{0}, got {1}x{2}",
            conv.size(),
            filters.shape()[1],
            filters.shape()[2]
        );
        if let Some(biases) = biases {
            assert_eq!(
                filters.shape()[0],
                biases.len(),
                "Number of filter biases does not agree with the number of filters"
            );
        }

        let filters_slice = filters.as_slice().map_or_else(
            || Cow::Owned(filters.iter().copied().collect()),
            Cow::Borrowed,
        );
        let filters_buffer = Buffer::builder()
            .queue(conv.queue().clone())
            .len(filters.shape().iter().product::<usize>())
            .flags(flags::MEM_READ_ONLY)
            .copy_host_slice(filters_slice.as_ref())
            .build()?;

        let filter_biases = biases
            .map(|biases| {
                Buffer::builder()
                    .queue(conv.queue().clone())
                    .len(biases.len())
                    .flags(flags::MEM_READ_ONLY)
                    .copy_host_slice(biases)
                    .build()
            })
            .transpose()?;

        conv.kernel().set_arg("filters", &filters_buffer)?;
        conv.kernel()
            .set_arg("filter_biases", filter_biases.as_ref())?;

        Ok(Self {
            inner: filters_buffer,
            biases: filter_biases,
            filter_count: u32::try_from(filters.shape()[0])
                .expect("Cannot convert filter count to `u32`"),
            channel_count: u32::try_from(filters.shape()[3])
                .expect("Cannot convert channel count to `u32`"),
        })
    }

    pub fn pass_as_arguments(&self, kernel: &Kernel) -> ocl::Result<()> {
        kernel.set_arg("filters", &self.inner)?;
        if let Some(ref biases) = self.biases {
            kernel.set_arg("filter_biases", biases)?;
        }
        Ok(())
    }
}

/// Container for convolution input and output.
#[derive(Debug, Clone)]
pub(crate) struct InputAndOutput<T: ConvElement> {
    signal_buffer: Buffer<T>,
    signal_dims: Uint3,
    output_buffer: Buffer<T>,
    output_shape: FeatureMapShape,
}

impl<T: ConvElement> InputAndOutput<T> {
    pub fn new<U: WithParams>(
        signal_shape: FeatureMapShape,
        filter_count: u32,
        conv: &Base<U>,
    ) -> ocl::Result<Self> {
        let Params {
            pads,
            strides,
            dilation,
            ..
        } = conv.params().into();
        let effective_kernel_h = conv.size() + (dilation[0] - 1) * (conv.size() - 1);
        let out_h = (signal_shape.height - effective_kernel_h + pads[0] + pads[2]) / strides[0] + 1;
        let effective_kernel_w = conv.size() + (dilation[1] - 1) * (conv.size() - 1);
        let out_w = (signal_shape.width - effective_kernel_w + pads[1] + pads[3]) / strides[1] + 1;
        let output_shape = FeatureMapShape {
            height: out_h,
            width: out_w,
            channels: filter_count,
            ..signal_shape
        };

        let signal_buffer = Buffer::builder()
            .queue(conv.queue().clone())
            .len(signal_shape.buffer_len())
            .flags(flags::MEM_READ_ONLY)
            .build()?;
        let output_buffer = Buffer::builder()
            .queue(conv.queue().clone())
            .len(output_shape.buffer_len())
            .flags(flags::MEM_HOST_READ_ONLY | flags::MEM_WRITE_ONLY)
            .build()?;

        let signal_dims = Uint3::new(
            signal_shape.height,
            signal_shape.width,
            signal_shape.channels,
        );
        Ok(InputAndOutput {
            signal_buffer,
            signal_dims,
            output_buffer,
            output_shape,
        })
    }

    pub fn write_signal(&self, signal: FeatureMap<'_, T>) -> ocl::Result<()> {
        let signal = signal.to_nhwc();
        let signal_slice = signal.as_slice().map_or_else(
            || Cow::Owned(signal.iter().copied().collect()),
            Cow::Borrowed,
        );
        self.signal_buffer.write(signal_slice.as_ref()).enq()
    }

    pub fn pass_as_arguments(&self, kernel: &Kernel) -> ocl::Result<()> {
        kernel.set_arg("signal_dims", self.signal_dims)
    }

    pub fn execute(&self, kernel: &Kernel, out_layout: Layout) -> ocl::Result<Array4<T>> {
        let s = self.output_shape;
        kernel.set_arg(
            "out_params",
            OutputParams {
                batch_size: s.batch_size,
                layout: out_layout,
            },
        )?;
        kernel.set_arg("output", &self.output_buffer)?;
        kernel.set_arg("signal", &self.signal_buffer)?;

        let command = kernel.cmd().global_work_size([
            s.height as usize * s.batch_size as usize,
            s.width as usize,
            s.channels as usize,
        ]);
        unsafe {
            command.enq()?;
        }

        let mut output_data = vec![T::default(); self.output_buffer.len()];
        self.output_buffer.read(&mut output_data).enq()?;
        let output =
            Array4::from_shape_vec(self.output_shape.as_array(out_layout), output_data).unwrap();
        Ok(output)
    }
}

/// Container for convolution filters (with optional filter biases), signal and output.
#[derive(Debug, Clone)]
pub(crate) struct Pinned<T: ConvElement> {
    pub io: InputAndOutput<T>,
    pub signal_shape: FeatureMapShape,
}
