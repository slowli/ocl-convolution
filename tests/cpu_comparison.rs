//! Comparison with the CPU convolution implementation.

use ndarray::{Array1, Array3, Array4, ArrayView3, ArrayView4};
use rand::{thread_rng, Rng};

use ocl_convolution::{Convolution, Params};

fn slow_compute(
    signal: ArrayView3<f32>,
    filters: ArrayView4<f32>,
    bias: Option<&[f32]>,
    params: Params,
) -> Array3<f32> {
    let (input_h, input_w) = (signal.shape()[0], signal.shape()[1]);
    let kernel_size = filters.shape()[1];
    assert_eq!(kernel_size, filters.shape()[2]);

    let output_h =
        (input_h - kernel_size + params.pads[0] + params.pads[2]) / params.strides[0] + 1;
    let output_y =
        (input_w - kernel_size + params.pads[1] + params.pads[3]) / params.strides[1] + 1;
    let (input_ch, output_ch) = (filters.shape()[3] * params.groups, filters.shape()[0]);
    let group_input_ch = input_ch / params.groups;
    let group_output_ch = output_ch / params.groups;

    let output = bias
        .map(|bias| Array1::from_vec(bias.to_vec()))
        .unwrap_or_else(|| Array1::zeros([output_ch]));
    let output = output.broadcast([output_h, output_y, output_ch]).unwrap();
    let mut output = output.permuted_axes([2, 0, 1]).to_owned();

    for o_x in 0..output_h {
        for k_x in 0..kernel_size {
            if o_x * params.strides[0] + k_x < params.pads[0] {
                continue;
            }
            let i_x = o_x * params.strides[0] + k_x - params.pads[0];
            if i_x >= input_h {
                continue;
            }

            for o_y in 0..output_y {
                for k_y in 0..kernel_size {
                    if o_y * params.strides[0] + k_y < params.pads[1] {
                        continue;
                    }
                    let i_y = o_y * params.strides[1] + k_y - params.pads[1];
                    if i_y >= input_w {
                        continue;
                    }

                    for g in 0..params.groups {
                        let output_range = (g * group_output_ch)..((g + 1) * group_output_ch);
                        for o_f in output_range {
                            for i_f in 0..group_input_ch {
                                output[[o_f, o_x, o_y]] += filters[[o_f, k_x, k_y, i_f]]
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
    let mut rng = thread_rng();
    let signal = Array3::from_shape_fn(signal_dims, |_| rng.gen_range(-1.0, 1.0));
    let filter = Array4::from_shape_fn(filter_dims, |_| rng.gen_range(-1.0, 1.0));
    let cpu_output = slow_compute(signal.view(), filter.view(), None, params);

    let convolution = Convolution::new(filter_dims[1], params).unwrap();
    let cl_output = convolution.compute(signal.view(), filter.view()).unwrap();

    let diff = (cpu_output - cl_output).mapv(f32::abs);
    let max_diff = diff.fold(0.0, |acc, &val| if val > acc { val } else { acc });
    assert!(
        max_diff < 1e-4,
        "signal={}, filter={}, params={:?}",
        signal,
        filter,
        params
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

fn test_grouped_f32_convolution(signal_dims: [usize; 3], filter_dims: [usize; 4], groups: usize) {
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
