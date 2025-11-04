//! Comparison with the CPU convolution implementation.

use ndarray::{Array1, Array3, Array4, ArrayView3, ArrayView4, Axis, LinalgScalar};
use rand::Rng;

use std::ops;

use ocl_convolution::{Convolution, FeatureMap, I8Params, Params};

fn slow_compute<T: LinalgScalar + ops::AddAssign>(
    signal: ArrayView3<'_, T>,
    filters: ArrayView4<'_, T>,
    bias: Option<&[T]>,
    params: Params,
) -> Array3<T> {
    let (stride_x, stride_y) = (params.strides[0] as usize, params.strides[1] as usize);
    let (dilation_x, dilation_y) = (params.dilation[0] as usize, params.dilation[1] as usize);
    let groups = params.groups as usize;

    let (input_h, input_w) = (signal.shape()[0], signal.shape()[1]);
    let kernel_size = filters.shape()[1];
    assert_eq!(kernel_size, filters.shape()[2]);

    let kernel_h = kernel_size + (dilation_x - 1) * (kernel_size - 1);
    let output_h =
        (input_h - kernel_h + params.pads[0] as usize + params.pads[2] as usize) / stride_x + 1;
    let kernel_w = kernel_size + (dilation_y - 1) * (kernel_size - 1);
    let output_y =
        (input_w - kernel_w + params.pads[1] as usize + params.pads[3] as usize) / stride_y + 1;
    let (input_ch, output_ch) = (filters.shape()[3] * groups, filters.shape()[0]);
    let group_input_ch = input_ch / groups;
    let group_output_ch = output_ch / groups;

    let output = bias
        .map(|bias| Array1::from(bias.to_vec()))
        .unwrap_or_else(|| Array1::zeros([output_ch]));
    let mut output = output
        .broadcast([output_h, output_y, output_ch])
        .unwrap()
        .to_owned();

    for o_x in 0..output_h {
        for k_x in 0..kernel_size {
            let i_x = match (o_x * stride_x + k_x * dilation_x).checked_sub(params.pads[0] as usize)
            {
                Some(i_x) if i_x < input_h => i_x,
                _ => continue,
            };

            for o_y in 0..output_y {
                for k_y in 0..kernel_size {
                    let i_y = match (o_y * stride_y + k_y * dilation_y)
                        .checked_sub(params.pads[1] as usize)
                    {
                        Some(i_y) if i_y < input_w => i_y,
                        _ => continue,
                    };

                    for g in 0..groups {
                        let output_range = (g * group_output_ch)..((g + 1) * group_output_ch);
                        for o_f in output_range {
                            for i_f in 0..group_input_ch {
                                output[[o_x, o_y, o_f]] += filters[[o_f, k_x, k_y, i_f]]
                                    * signal[[i_x, i_y, i_f + g * group_input_ch]];
                            }
                        }
                    }
                }
            }
        }
    }
    output
}

fn compare_f32_convolution(signal_dims: [usize; 3], filter_dims: [usize; 4], params: Params) {
    let mut rng = rand::rng();
    let signal = Array3::from_shape_fn(signal_dims, |_| rng.random_range(-1.0..=1.0));
    let filter = Array4::from_shape_fn(filter_dims, |_| rng.random_range(-1.0..=1.0));
    let cpu_output = slow_compute(signal.view(), filter.view(), None, params);

    let convolution = Convolution::f32(filter_dims[1] as u32)
        .unwrap()
        .build(params)
        .unwrap();
    let signal = FeatureMap::nhwc(signal.view().insert_axis(Axis(0)));
    let cl_output = convolution
        .compute(signal, &filter)
        .unwrap()
        .index_axis_move(Axis(0), 0);

    let diff = (cpu_output - cl_output).mapv(f32::abs);
    let max_diff = diff.fold(0.0, |acc, &val| if val > acc { val } else { acc });
    assert!(
        max_diff < 1e-4,
        "signal={signal:?}, filter={filter}, params={params:?}"
    );
}

fn test_f32_convolution(signal_dims: [usize; 3], filter_dims: [usize; 4]) {
    compare_f32_convolution(signal_dims, filter_dims, Params::default());

    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            pads: [1; 4],
            ..Params::default()
        },
    );
    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            pads: [2; 4],
            ..Params::default()
        },
    );

    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            strides: [2, 2],
            ..Params::default()
        },
    );
    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            strides: [2, 2],
            pads: [1; 4],
            ..Params::default()
        },
    );
}

#[test]
fn f32_convolution_small() {
    const SIGNAL_DIMS: [usize; 3] = [5, 5, 4];
    const FILTER_DIMS: [usize; 4] = [4, 3, 3, 4];
    test_f32_convolution(SIGNAL_DIMS, FILTER_DIMS);
}

#[test]
fn f32_convolution_medium() {
    const SIGNAL_DIMS: [usize; 3] = [17, 17, 32];
    const FILTER_DIMS: [usize; 4] = [10, 3, 3, 32];
    test_f32_convolution(SIGNAL_DIMS, FILTER_DIMS);
}

#[test]
fn f32_non_square_convolution() {
    const FILTER_DIMS: [usize; 4] = [4, 3, 3, 4];
    test_f32_convolution([7, 5, 4], FILTER_DIMS);
    test_f32_convolution([5, 7, 4], FILTER_DIMS);
}

#[test]
fn f32_convolution_on_small_number_of_channels() {
    for channels in (1..=8).chain(vec![10, 13, 16, 17]) {
        test_f32_convolution([5, 5, channels], [1, 3, 3, channels]);
    }
}

fn test_grouped_f32_convolution(signal_dims: [usize; 3], filter_dims: [usize; 4], groups: u32) {
    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            groups,
            ..Params::default()
        },
    );
    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            pads: [1; 4],
            groups,
            ..Params::default()
        },
    );
    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            strides: [2, 2],
            pads: [1; 4],
            groups,
            ..Params::default()
        },
    );
}

#[test]
fn f32_grouped_convolution_small() {
    const SIGNAL_DIMS: [usize; 3] = [5, 5, 8];
    const FILTER_DIMS: [usize; 4] = [4, 3, 3, 4];
    test_grouped_f32_convolution(SIGNAL_DIMS, FILTER_DIMS, 2);
}

#[test]
fn f32_grouped_convolution_medium() {
    const SIGNAL_DIMS: [usize; 3] = [28, 28, 32];
    const FILTER_DIMS: [usize; 4] = [32, 3, 3, 8];
    test_grouped_f32_convolution(SIGNAL_DIMS, FILTER_DIMS, 4);
}

fn test_dilated_f32_convolution(signal_dims: [usize; 3], filter_dims: [usize; 4], dilation: u32) {
    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            dilation: [dilation; 2],
            ..Params::default()
        },
    );
    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            pads: [1; 4],
            dilation: [dilation; 2],
            ..Params::default()
        },
    );
    compare_f32_convolution(
        signal_dims,
        filter_dims,
        Params {
            pads: [2; 4],
            dilation: [dilation; 2],
            ..Params::default()
        },
    );
}

#[test]
fn f32_dilated_convolution_small() {
    const SIGNAL_DIMS: [usize; 3] = [10, 10, 8];
    const FILTER_DIMS: [usize; 4] = [4, 3, 3, 8];
    test_dilated_f32_convolution(SIGNAL_DIMS, FILTER_DIMS, 2);
}

#[test]
fn f32_dilated_convolution_medium() {
    const SIGNAL_DIMS: [usize; 3] = [28, 28, 32];
    const FILTER_DIMS: [usize; 4] = [10, 3, 3, 32];
    test_dilated_f32_convolution(SIGNAL_DIMS, FILTER_DIMS, 2);
}

fn downscale(x: i32, shift: i32) -> i8 {
    let mask = (1 << shift) - 1;
    let threshold = 1 << (shift - 1);
    let mut downscaled = x >> shift;
    let remainder = x & mask;
    if remainder > threshold || (remainder == threshold && downscaled & 1 == 1) {
        downscaled += 1;
    }
    downscaled.clamp(-128, 127) as i8
}

fn compare_i8_convolution(signal_dims: [usize; 3], filter_dims: [usize; 4], params: Params) {
    let mut rng = rand::rng();
    let signal = Array3::from_shape_fn(signal_dims, |_| rng.random_range(-127_i8..=127));
    let filter = Array4::from_shape_fn(filter_dims, |_| rng.random_range(-127_i8..=127));
    let cpu_output = slow_compute(
        signal.mapv(i32::from).view(),
        filter.mapv(i32::from).view(),
        None,
        params,
    )
    .mapv(|x| downscale(x, 8));

    let convolution = Convolution::i8(filter_dims[1] as u32)
        .unwrap()
        .build(I8Params {
            common: params,
            bit_shift: 14,
            scale: 1 << 6, // corresponds to 1/256
            output_bias: 0,
            signal_bias: 0,
            filter_bias: 0,
        })
        .unwrap();
    let signal = FeatureMap::nhwc(signal.view().insert_axis(Axis(0)));
    let cl_output = convolution
        .compute(signal, &filter)
        .unwrap()
        .index_axis_move(Axis(0), 0);

    if cl_output != cpu_output {
        let index = (cl_output.clone() - cpu_output.clone())
            .indexed_iter()
            .filter_map(|(i, &x)| if x != 0 { Some(i) } else { None })
            .next()
            .unwrap();
        panic!("cl={cl_output}, cpu={cpu_output}, index={index:?}");
    }
}

fn test_i8_convolution(signal_dims: [usize; 3], filter_dims: [usize; 4]) {
    compare_i8_convolution(signal_dims, filter_dims, Params::default());

    compare_i8_convolution(
        signal_dims,
        filter_dims,
        Params {
            pads: [1; 4],
            ..Params::default()
        },
    );
    compare_i8_convolution(
        signal_dims,
        filter_dims,
        Params {
            pads: [2; 4],
            ..Params::default()
        },
    );

    compare_i8_convolution(
        signal_dims,
        filter_dims,
        Params {
            strides: [2, 2],
            ..Params::default()
        },
    );
    compare_i8_convolution(
        signal_dims,
        filter_dims,
        Params {
            strides: [2, 2],
            pads: [1; 4],
            ..Params::default()
        },
    );
}

#[test]
fn i8_convolution_small() {
    const SIGNAL_DIMS: [usize; 3] = [5, 5, 4];
    const FILTER_DIMS: [usize; 4] = [4, 3, 3, 4];
    test_i8_convolution(SIGNAL_DIMS, FILTER_DIMS);
}

#[test]
fn i8_convolution_medium() {
    const SIGNAL_DIMS: [usize; 3] = [17, 17, 32];
    const FILTER_DIMS: [usize; 4] = [10, 3, 3, 32];
    test_i8_convolution(SIGNAL_DIMS, FILTER_DIMS);
}

#[test]
fn i8_non_square_convolution() {
    const FILTER_DIMS: [usize; 4] = [4, 3, 3, 4];
    test_i8_convolution([7, 5, 4], FILTER_DIMS);
    test_i8_convolution([5, 7, 4], FILTER_DIMS);
}

#[test]
fn i8_convolution_on_small_number_of_channels() {
    for channels in (1..=8).chain(vec![10, 13, 16, 17]) {
        test_i8_convolution([5, 5, channels], [1, 3, 3, channels]);
    }
}

#[test]
fn i8_grouped_convolution_small() {
    compare_f32_convolution(
        [5, 5, 8],
        [2, 3, 3, 4],
        Params {
            groups: 2,
            ..Params::default()
        },
    );
    compare_f32_convolution(
        [5, 5, 8],
        [4, 3, 3, 2],
        Params {
            groups: 4,
            ..Params::default()
        },
    );
    compare_f32_convolution(
        [5, 5, 8],
        [8, 3, 3, 1],
        Params {
            groups: 8,
            ..Params::default()
        },
    );
}
