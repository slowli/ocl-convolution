//! OpenCL buffer wrappers.

#![allow(missing_docs)]

use ndarray::{Array4, ArrayView4};
use ocl::{flags, prm::Uint3, Buffer, Kernel};

use std::borrow::Cow;

use crate::{ConvElement, Convolution, Params, WithParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureMapShape {
    pub batch_size: usize,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

impl FeatureMapShape {
    fn from_nhwc_slice(shape: &[usize]) -> Self {
        assert_eq!(shape.len(), 4);
        FeatureMapShape {
            batch_size: shape[0],
            height: shape[1],
            width: shape[2],
            channels: shape[3],
        }
    }

    fn from_nchw_slice(shape: &[usize]) -> Self {
        assert_eq!(shape.len(), 4);
        FeatureMapShape {
            batch_size: shape[0],
            height: shape[2],
            width: shape[3],
            channels: shape[1],
        }
    }

    fn buffer_len(self) -> usize {
        self.batch_size * self.width * self.height * self.channels
    }

    fn as_array(self, layout: Layout) -> [usize; 4] {
        match layout {
            Layout::ChannelsFirst => [self.batch_size, self.channels, self.height, self.width],
            Layout::ChannelsLast => [self.batch_size, self.height, self.width, self.channels],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Layout {
    /// `NCHW` order
    ChannelsFirst = 0,
    /// `NHWC` order
    ChannelsLast = 1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureMap<'a, T: ConvElement> {
    layout: Layout,
    inner: ArrayView4<'a, T>,
}

impl<'a, T: ConvElement> FeatureMap<'a, T> {
    pub fn nchw(array: impl Into<ArrayView4<'a, T>>) -> Self {
        Self {
            layout: Layout::ChannelsFirst,
            inner: array.into(),
        }
    }

    pub fn nhwc(array: impl Into<ArrayView4<'a, T>>) -> Self {
        Self {
            layout: Layout::ChannelsLast,
            inner: array.into(),
        }
    }

    pub fn layout(self) -> Layout {
        self.layout
    }

    pub fn shape(self) -> FeatureMapShape {
        match self.layout {
            Layout::ChannelsFirst => FeatureMapShape::from_nchw_slice(self.inner.shape()),
            Layout::ChannelsLast => FeatureMapShape::from_nhwc_slice(self.inner.shape()),
        }
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
pub struct Filters<T: ConvElement> {
    inner: Buffer<T>,
    biases: Option<Buffer<T::Acc>>,
    filter_count: usize,
    channel_count: usize,
}

impl<T: ConvElement> Filters<T> {
    pub(crate) fn filter_count(&self) -> usize {
        self.filter_count
    }

    pub(crate) fn channel_count(&self) -> usize {
        self.channel_count
    }

    pub(crate) fn new<U: WithParams>(
        filters: ArrayView4<T>,
        biases: Option<&[T::Acc]>,
        conv: &Convolution<U>,
    ) -> ocl::Result<Self> {
        assert!(
            filters.shape()[1] == conv.size && filters.shape()[2] == conv.size,
            "Invalid filter shape: expected {0}x{0}, got {1}x{2}",
            conv.size,
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

        let filters_slice = filters
            .as_slice()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(filters.iter().cloned().collect()));
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

        conv.kernel.set_arg("filters", &filters_buffer)?;
        if let Some(ref biases) = filter_biases {
            conv.kernel.set_arg("filter_biases", biases)?;
        }

        Ok(Self {
            inner: filters_buffer,
            biases: filter_biases,
            filter_count: filters.shape()[0],
            channel_count: filters.shape()[3],
        })
    }

    pub(crate) fn pass_as_arguments(&self, kernel: &Kernel) -> ocl::Result<()> {
        kernel.set_arg("filters", &self.inner)?;
        if let Some(ref biases) = self.biases {
            kernel.set_arg("filter_biases", biases)?;
        }
        Ok(())
    }
}

/// Container for convolution input and output.
#[derive(Debug, Clone)]
pub struct InputAndOutput<T: ConvElement> {
    signal_buffer: Buffer<T>,
    signal_dims: Uint3,
    batch_size: u32,
    output_buffer: Buffer<T>,
    output_shape: FeatureMapShape,
}

impl<T: ConvElement> InputAndOutput<T> {
    pub fn new<U: WithParams>(
        signal_shape: FeatureMapShape,
        filter_count: usize,
        conv: &Convolution<U>,
    ) -> ocl::Result<Self> {
        let Params {
            pads,
            strides,
            dilation,
            ..
        } = U::get_generic_params(&conv.params);
        let effective_kernel_h = conv.size + (dilation[0] - 1) * (conv.size - 1);
        let out_h = (signal_shape.height - effective_kernel_h + pads[0] + pads[2]) / strides[0] + 1;
        let effective_kernel_w = conv.size + (dilation[1] - 1) * (conv.size - 1);
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
            signal_shape.height as u32,
            signal_shape.width as u32,
            signal_shape.channels as u32,
        );
        Ok(InputAndOutput {
            signal_buffer,
            signal_dims,
            batch_size: signal_shape.batch_size as u32,
            output_buffer,
            output_shape,
        })
    }

    pub fn write_signal(&self, signal: FeatureMap<T>) -> ocl::Result<()> {
        let signal = signal.to_nhwc();
        let signal_slice = signal
            .as_slice()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(signal.iter().cloned().collect()));
        self.signal_buffer.write(signal_slice.as_ref()).enq()
    }

    pub fn pass_as_arguments(&self, kernel: &Kernel) -> ocl::Result<()> {
        kernel.set_arg("signal_dims", self.signal_dims)
    }

    pub fn execute(
        &self,
        kernel: &Kernel,
        filter_size: usize,
        out_layout: Layout,
    ) -> ocl::Result<Array4<T>> {
        let s = self.output_shape;
        kernel.set_arg("convolved_layout", out_layout as u8)?;

        for i in 0..s.batch_size {
            let output_slice = if s.batch_size > 1 {
                let out_sample_len = s.width * s.height * s.channels;
                Cow::Owned(self.output_buffer.create_sub_buffer(
                    Some(flags::MEM_HOST_READ_ONLY | flags::MEM_WRITE_ONLY),
                    i * out_sample_len,
                    out_sample_len,
                )?)
            } else {
                Cow::Borrowed(&self.output_buffer)
            };
            kernel.set_arg("convolved", output_slice.as_ref())?;

            let signal_slice = if s.batch_size > 1 {
                let signal_sample_len =
                    (self.signal_dims[0] * self.signal_dims[1] * self.signal_dims[2]) as usize;
                Cow::Owned(self.signal_buffer.create_sub_buffer(
                    Some(flags::MEM_READ_ONLY),
                    i * signal_sample_len,
                    signal_sample_len,
                )?)
            } else {
                Cow::Borrowed(&self.signal_buffer)
            };
            kernel.set_arg("signal", signal_slice.as_ref())?;

            let command = kernel
                .cmd()
                .global_work_size([s.height * filter_size, s.width * filter_size, s.channels])
                .local_work_size([filter_size, filter_size]);
            unsafe {
                command.enq()?;
            }
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
pub struct Pinned<T: ConvElement> {
    pub(crate) filters: Filters<T>,
    pub(crate) io: InputAndOutput<T>,
    pub(crate) signal_shape: FeatureMapShape,
}
