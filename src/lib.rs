use ndarray::{Array3, ArrayView3, ArrayView4};
use ocl::{
    flags,
    prm::{Uint2, Uint3, Uint4},
    Buffer, ProQue,
};

use std::{borrow::Cow, convert::TryFrom};

const SOURCE: &str = include_str!("conv.cl");

#[derive(Debug)]
pub struct Convolution {
    size: usize,
    program: ProQue,
}

impl Convolution {
    pub fn new(size: usize) -> ocl::Result<Self> {
        assert_eq!(size % 2, 1, "Even convolution sizes are not supported");
        let src = format!("#define FILTER_SIZE {}\n{}", size, SOURCE);
        let program = ProQue::builder().src(src).build()?;
        Ok(Self { size, program })
    }

    pub fn size(&self) -> usize {
        self.size
    }

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
            .arg(&output_buffer)
            .arg(&signal_buffer)
            .arg(Uint3::new(in_h as u32, in_w as u32, in_channels as u32))
            .arg(&filters_buffer)
            .arg(Uint2::new(strides[0] as u32, strides[1] as u32))
            .arg(Uint4::new(
                pads[0] as u32,
                pads[1] as u32,
                pads[2] as u32,
                pads[3] as u32,
            ))
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

#[cfg(test)]
mod tests {
    use ndarray::Array4;

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

        assert_eq!(output[[0, 0, 0]], 48.0);
        // 48 = 4 * (0 + 1 + 5 + 6), numbers in the upper left corner of the image.
    }
}
