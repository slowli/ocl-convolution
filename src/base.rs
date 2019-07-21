use ndarray::{Array4, ArrayView4};
use ocl::{builders::KernelBuilder, prm::Uint3, Buffer, Context, Kernel, ProQue, Queue};

use std::marker::PhantomData;

use crate::{
    buffers::{FeatureMap, FeatureMapShape, Filters, InputAndOutput, Pinned},
    params::{OutputParams, WithParams},
    ConvElement,
};

/// Convolution builder. The same builder can be used to create multiple `Convolutions`
/// which share the same spatial size.
#[derive(Debug)]
pub struct ConvolutionBuilder<T> {
    program: ProQue,
    filter_size: usize,
    _t: PhantomData<T>,
}

impl<T: ConvElement> ConvolutionBuilder<T> {
    /// Initializes a builder with a specific filter size.
    pub(crate) fn new(
        filter_size: usize,
        defines: &[(&'static str, i32)],
        source: &str,
    ) -> ocl::Result<Self> {
        assert_eq!(
            filter_size % 2,
            1,
            "Even convolution sizes are not supported"
        );

        let mut program_builder = ocl::Program::builder();
        program_builder.cmplr_def("FILTER_SIZE", filter_size as i32);
        for &(name, value) in defines {
            program_builder.cmplr_def(name, value);
        }
        program_builder.source(source);

        let program = ProQue::builder().prog_bldr(program_builder).build()?;
        Ok(Self {
            program,
            filter_size,
            _t: PhantomData,
        })
    }

    fn kernel_builder(&self) -> KernelBuilder {
        self.program.kernel_builder("conv")
    }
}

fn create_io<T: ConvElement, U: WithParams>(
    signal_shape: FeatureMapShape,
    filters: &Filters<T>,
    conv: &Base<U>,
) -> ocl::Result<InputAndOutput<T>> {
    assert_eq!(
        signal_shape.channels,
        filters.channel_count() * U::get_generic_params(&conv.params).groups,
        "Channel dimensionality in signal and filters must agree"
    );
    let io = InputAndOutput::new(signal_shape, filters.filter_count(), conv)?;
    io.pass_as_arguments(&conv.kernel).map(|()| io)
}

#[derive(Debug)]
pub struct Base<T: WithParams> {
    size: usize,
    params: T::Params,
    kernel: Kernel,
    buffers: T,
    context: Context,
}

impl<T: WithParams> Base<T> {
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    pub fn queue(&self) -> &Queue {
        self.kernel
            .default_queue()
            .expect("kernel should come with a pre-configured queue")
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn params(&self) -> &T::Params {
        &self.params
    }

    pub fn set_params(&mut self, params: T::Params) -> ocl::Result<()> {
        self.params = params.clone();
        self.kernel.set_arg("params", params.into())
    }
}

impl<T: ConvElement> Base<T> {
    pub fn new(builder: &ConvolutionBuilder<T>, params: T::Params) -> ocl::Result<Self> {
        let kernel = builder
            .kernel_builder()
            .arg_named("output", None::<&Buffer<T>>)
            .arg_named("out_params", OutputParams::default())
            .arg_named("signal", None::<&Buffer<T>>)
            .arg_named("signal_dims", Uint3::new(0, 0, 0))
            .arg_named("filters", None::<&Buffer<T>>)
            .arg_named("filter_biases", None::<&Buffer<T::Acc>>)
            .arg_named("params", params.clone().into())
            .build()?;
        Ok(Base {
            size: builder.filter_size,
            params,
            kernel,
            buffers: T::default(),
            context: builder.program.context().clone(),
        })
    }

    pub fn with_filters(
        self,
        filters: ArrayView4<T>,
        filter_biases: Option<&[T::Acc]>,
    ) -> ocl::Result<Base<Filters<T>>> {
        let filters = Filters::new(filters, filter_biases, &self)?;
        Ok(Base {
            buffers: filters,
            size: self.size,
            params: self.params,
            kernel: self.kernel,
            context: self.context,
        })
    }

    pub fn compute(
        &self,
        signal: FeatureMap<T>,
        filters: ArrayView4<T>,
        filter_biases: Option<&[T::Acc]>,
    ) -> ocl::Result<Array4<T>> {
        assert_eq!(
            signal.shape().channels,
            filters.shape()[3] * T::get_generic_params(&self.params).groups,
            "Channel dimensionality in signal and filters must agree"
        );

        let filter_count = filters.shape()[0];
        let filters = Filters::new(filters, filter_biases, self)?;
        filters.pass_as_arguments(&self.kernel)?;
        let io = InputAndOutput::new(signal.shape(), filter_count, self)?;
        io.write_signal(signal)?;
        io.pass_as_arguments(&self.kernel)?;
        io.execute(&self.kernel, signal.layout())
    }
}

impl<T: ConvElement + WithParams> Base<Filters<T>> {
    pub fn pinned(self, signal_shape: FeatureMapShape) -> ocl::Result<Base<Pinned<T>>> {
        let io = create_io(signal_shape, &self.buffers, &self)?;
        Ok(Base {
            size: self.size,
            params: self.params,
            kernel: self.kernel,
            buffers: Pinned {
                filters: self.buffers,
                io,
                signal_shape,
            },
            context: self.context,
        })
    }

    pub fn compute(&self, signal: FeatureMap<T>) -> ocl::Result<Array4<T>> {
        let io = create_io(signal.shape(), &self.buffers, self)?;
        io.write_signal(signal)?;
        io.execute(&self.kernel, signal.layout())
    }
}

impl<T: ConvElement + WithParams> Base<Pinned<T>> {
    pub fn compute(&self, signal: FeatureMap<T>) -> ocl::Result<Array4<T>> {
        assert_eq!(
            signal.shape(),
            self.buffers.signal_shape,
            "Signal dimensions differ from the ones set by `with_pinned_memory()`"
        );
        self.buffers.io.write_signal(signal)?;
        self.buffers.io.execute(&self.kernel, signal.layout())
    }
}
