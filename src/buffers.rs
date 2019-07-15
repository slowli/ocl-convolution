//! OpenCL buffer wrappers.

use ndarray::{Array3, ArrayView3, ArrayView4};
use ocl::{flags, prm::Uint3, Buffer, Kernel};

use std::borrow::Cow;

use crate::{ConvElement, Convolution};

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

    pub(crate) fn new<U>(
        filters: ArrayView4<T>,
        biases: Option<&[T::Acc]>,
        conv: &Convolution<U>,
    ) -> ocl::Result<Self> {
        assert!(
            filters.shape()[1] == conv.size && filters.shape()[2] == conv.size,
            "Invalid filter shape"
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
    output_buffer: Buffer<T>,
    output_dims: [usize; 3],
}

impl<T: ConvElement> InputAndOutput<T> {
    pub fn new<U>(
        signal_dims: [usize; 3],
        filter_count: usize,
        conv: &Convolution<U>,
    ) -> ocl::Result<Self> {
        let [in_h, in_w, in_channels] = signal_dims;
        let pads = conv.params.pads;
        let strides = conv.params.strides;
        let dilation = conv.params.dilation;
        let effective_kernel_h = conv.size + (dilation[0] - 1) * (conv.size - 1);
        let out_h = (in_h - effective_kernel_h + pads[0] + pads[2]) / strides[0] + 1;
        let effective_kernel_w = conv.size + (dilation[1] - 1) * (conv.size - 1);
        let out_w = (in_w - effective_kernel_w + pads[1] + pads[3]) / strides[1] + 1;
        let output_dims = [filter_count, out_h, out_w];

        let signal_buffer = Buffer::builder()
            .queue(conv.queue().clone())
            .len(signal_dims)
            .flags(flags::MEM_READ_ONLY)
            .build()?;
        let output_buffer = Buffer::builder()
            .queue(conv.queue().clone())
            .len(output_dims)
            .flags(flags::MEM_HOST_READ_ONLY)
            .build()?;

        let signal_dims = Uint3::new(in_h as u32, in_w as u32, in_channels as u32);
        Ok(InputAndOutput {
            signal_buffer,
            signal_dims,
            output_buffer,
            output_dims,
        })
    }

    pub fn write_signal(&self, signal: ArrayView3<T>) -> ocl::Result<()> {
        let signal_slice = signal
            .as_slice()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(signal.iter().cloned().collect()));
        self.signal_buffer.write(signal_slice.as_ref()).enq()
    }

    pub fn pass_as_arguments(&self, kernel: &Kernel) -> ocl::Result<()> {
        kernel.set_arg("convolved", &self.output_buffer)?;
        kernel.set_arg("signal", &self.signal_buffer)?;
        kernel.set_arg("signal_dims", self.signal_dims)
    }

    pub fn execute(&self, kernel: &Kernel, filter_size: usize) -> ocl::Result<Array3<T>> {
        let [filter_count, out_h, out_w] = self.output_dims;
        let command = kernel
            .cmd()
            .global_work_size([out_h * filter_size, out_w * filter_size, filter_count])
            .local_work_size([filter_size, filter_size]);

        unsafe {
            command.enq()?;
        }

        let mut output_data = vec![T::default(); self.output_buffer.len()];
        self.output_buffer.read(&mut output_data).enq()?;
        let output = Array3::from_shape_vec(self.output_dims, output_data).unwrap();
        Ok(output)
    }
}

/// Container for convolution filters (with optional filter biases), signal and output.
#[derive(Debug, Clone)]
pub struct Pinned<T: ConvElement> {
    pub(crate) filters: Filters<T>,
    pub(crate) io: InputAndOutput<T>,
    pub(crate) signal_dims: [usize; 3],
}
