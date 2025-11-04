//! Tests for basic functionality provided by the crate.

use ndarray::{Array4, Axis};
use rand::Rng;

use std::fmt;

use ocl_convolution::{Convolution, FeatureMap, FeatureMapShape, I8Params, Params};

/// Simple wrapper for both error types used in tests.
#[derive(Debug)]
enum Error {
    Shape(ndarray::ShapeError),
    Ocl(ocl::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(e) => fmt::Display::fmt(e, formatter),
            Self::Ocl(e) => fmt::Display::fmt(e, formatter),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Shape(e) => Some(e),
            Self::Ocl(_) => None,
        }
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(error: ndarray::ShapeError) -> Self {
        Self::Shape(error)
    }
}

impl From<ocl::Error> for Error {
    fn from(error: ocl::Error) -> Self {
        Self::Ocl(error)
    }
}

#[allow(clippy::cast_precision_loss)] // There is no loss, since `i` values are small
fn create_signal(width: usize, height: usize) -> Array4<f32> {
    Array4::from_shape_vec(
        [1, width, height, 1],
        (0..(width * height)).map(|i| i as f32).collect(),
    )
    .unwrap()
}

#[test]
fn basics() -> Result<(), Error> {
    let convolution = Convolution::f32(3)?.build(Params::default())?;
    let signal = create_signal(5, 5);
    let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.0; 9])?;

    let c = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
    assert_eq!(
        c,
        Array4::from_shape_vec(
            [1, 3, 3, 1],
            vec![54., 63., 72., 99., 108., 117., 144., 153., 162.],
        )?,
    );

    let signal = Array4::from_shape_vec(
        [1, 1, 5, 5],
        vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            19., 20., 21., 22., 23., 24.,
        ],
    )?;
    let c = convolution.compute(FeatureMap::nchw(&signal), &filter)?;
    assert_eq!(
        c,
        Array4::from_shape_vec(
            [1, 1, 3, 3],
            vec![54., 63., 72., 99., 108., 117., 144., 153., 162.],
        )?,
    );

    Ok(())
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn f32_convolution_with_filters() -> Result<(), Error> {
    let filters = Array4::from_elem([1, 3, 3, 1], 1.0);
    let convolution = Convolution::f32(3)?
        .build(Params::default())?
        .with_filters(filters.view())?;

    let signal = create_signal(5, 5);

    let c = convolution.compute(FeatureMap::nhwc(&signal))?;
    assert_eq!(
        c,
        Array4::from_shape_vec(
            [1, 3, 3, 1],
            vec![54., 63., 72., 99., 108., 117., 144., 153., 162.],
        )?,
    );

    for i in 1..=5 {
        let signal = Array4::from_elem([1, 5 + i, 5 + i, 1], i as f32);
        // ^-- no loss since `i` values are small
        assert!(convolution.compute(FeatureMap::nhwc(&signal)).is_ok());
    }

    let pinned = convolution.pin(FeatureMapShape {
        batch_size: 1,
        width: 5,
        height: 5,
        channels: 1,
    })?;
    let c = pinned.compute(FeatureMap::nhwc(&signal))?;
    assert_eq!(
        c,
        Array4::from_shape_vec(
            [1, 3, 3, 1],
            vec![54., 63., 72., 99., 108., 117., 144., 153., 162.],
        )?,
    );

    for i in 1..=5 {
        let signal = Array4::from_elem([1, 5, 5, 1], i as f32);
        // ^-- no loss since `i` values are small
        assert!(pinned.compute(FeatureMap::nhwc(&signal)).is_ok());
    }
    Ok(())
}

#[test]
fn f32_convolution_with_filters_and_biases() -> Result<(), Error> {
    let filters = Array4::from_elem([1, 3, 3, 1], 1.0);
    let convolution = Convolution::f32(3)?
        .build(Params::default())?
        .with_biased_filters(filters.view(), &[-100.0])?;

    let signal = create_signal(5, 5);

    let c = convolution.compute(FeatureMap::nhwc(&signal))?;
    assert_eq!(
        c,
        Array4::from_shape_vec(
            [1, 3, 3, 1],
            vec![-46., -37., -28., -1., 8., 17., 44., 53., 62.],
        )?,
    );
    Ok(())
}

#[test]
fn grouped_convolution() -> Result<(), Error> {
    let convolution = Convolution::f32(3)?.build(Params {
        strides: [1, 1],
        pads: [0; 4],
        dilation: [1, 1],
        groups: 2,
    })?;

    // All elements on the `i`th channel have value `i`.
    let signal = Array4::from_shape_vec(
        [1, 3, 3, 4],
        vec![
            1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2.,
            3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
        ],
    )?;

    let filters = Array4::from_shape_vec(
        [2, 3, 3, 2],
        vec![
            // 1st filter (applied to channels 0..2)
            1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.,
            // 2nd filter (applied to channels 2..4)
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ],
    )?;
    let expected_output = Array4::from_shape_vec(
        [1, 1, 1, 2],
        vec![
            -9.0, // = (1 + 1 + ... + 1) * (1 - 2)
            63.0, // = (1 + 1 + ... + 1) * (3 + 4)
        ],
    )?;

    let output = convolution.compute(FeatureMap::nhwc(&signal), &filters)?;
    assert_eq!(output, expected_output);
    Ok(())
}

#[test]
fn grouped_i8_convolution() -> Result<(), Error> {
    let convolution = Convolution::i8(3)?.build(I8Params {
        common: Params {
            strides: [1, 1],
            pads: [0; 4],
            dilation: [1, 1],
            groups: 4,
        },
        bit_shift: 12,
        scale: I8Params::convert_scale(12, 1.0),
        output_bias: 0,
        signal_bias: 0,
        filter_bias: 0,
    })?;

    // All elements on the `i`th channel have value `i`.
    let signal = Array4::from_shape_vec(
        [1, 3, 3, 4],
        vec![
            1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
            2, 3, 4, 1, 2, 3, 4,
        ],
    )?;

    let filters = Array4::from_shape_vec(
        [4, 3, 3, 1],
        vec![
            1, -1, 1, -1, 1, -1, 1, -1, 1, //
            1, 1, 1, 1, 1, 1, 1, 1, 1, //
            1, -1, 1, -1, 1, -1, 1, -1, 1, //
            1, 1, 1, 1, 1, 1, 1, 1, 1, //
        ],
    )?;
    let expected_output = Array4::from_shape_vec(
        [1, 1, 1, 4],
        vec![
            1,  // 1 * (1 - 1 + 1 - ... + 1)
            18, // 2 * 9
            3,  // 3 * (1 - 1 + 1 - ... + 1)
            36, // 4 * 9
        ],
    )?;

    let output = convolution.compute(FeatureMap::nhwc(&signal), &filters)?;
    assert_eq!(output, expected_output);
    Ok(())
}

#[test]
fn with_padding() -> Result<(), Error> {
    let convolution = Convolution::f32(3)?.build(Params {
        strides: [1, 1],
        pads: [1; 4],
        dilation: [1, 1],
        groups: 1,
    })?;

    let signal = create_signal(5, 5);
    let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.0; 9])?;

    let c = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
    assert_eq!(
        c,
        Array4::from_shape_vec(
            [1, 5, 5, 1],
            vec![
                12., 21., 27., 33., 24., 33., 54., 63., 72., 51., 63., 99., 108., 117., 81., 93.,
                144., 153., 162., 111., 72., 111., 117., 123., 84.,
            ],
        )?,
    );
    Ok(())
}

#[test]
fn with_strides() -> Result<(), Error> {
    let convolution = Convolution::f32(3)?.build(Params {
        strides: [2, 2],
        pads: [0; 4],
        dilation: [1, 1],
        groups: 1,
    })?;

    let signal = create_signal(7, 5);
    let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.; 9])?;
    let expected_output =
        Array4::from_shape_vec([1, 3, 2, 1], vec![54., 72., 144., 162., 234., 252.])?;

    assert_eq!(
        convolution.compute(FeatureMap::nhwc(&signal), &filter)?,
        expected_output
    );
    Ok(())
}

#[test]
fn with_strides_and_padding() -> Result<(), Error> {
    let convolution = Convolution::f32(3)?.build(Params {
        strides: [2, 2],
        pads: [1; 4],
        dilation: [1, 1],
        groups: 1,
    })?;

    let signal = create_signal(7, 5);
    let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1.; 9])?;

    let expected_output = Array4::from_shape_vec(
        [1, 4, 3, 1],
        vec![
            12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.,
        ],
    )?;

    assert_eq!(
        convolution.compute(FeatureMap::nhwc(&signal), &filter)?,
        expected_output
    );
    Ok(())
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn with_several_input_channels() -> Result<(), Error> {
    let convolution = Convolution::f32(3)?.build(Params {
        strides: [1, 1],
        pads: [1; 4],
        dilation: [1, 1],
        groups: 1,
    })?;

    let mut signal = vec![0.0; 100];

    for (i, val) in signal.iter_mut().enumerate() {
        *val = (i / 4) as f32; // no loss since `i` values are small
    }
    let signal = Array4::from_shape_vec([1, 5, 5, 4], signal)?;
    let filter = Array4::from_shape_vec([1, 3, 3, 4], vec![1.; 36])?;
    let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;

    assert!((output[[0, 0, 0, 0]] - 48.0).abs() < f32::EPSILON);
    // 48 = 4 * (0 + 1 + 5 + 6), numbers in the upper left corner of the image.
    Ok(())
}

#[test]
fn with_dilation() -> Result<(), Error> {
    let mut convolution = Convolution::f32(3)?.build(Params {
        strides: [1, 1],
        pads: [0; 4],
        groups: 1,
        dilation: [2, 2],
    })?;

    let signal = Array4::from_shape_vec(
        [1, 5, 5, 1],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, //
            6.0, 7.0, 8.0, 9.0, 10.0, //
            11.0, 12.0, 13.0, 14.0, 15.0, //
            16.0, 17.0, 18.0, 19.0, 20.0, //
            21.0, 22.0, 23.0, 24.0, 25.0, //
        ],
    )?;
    let filters = Array4::from_elem([1, 3, 3, 1], 1.0);

    // 117.0 = 1.0 + 3.0 + ... + 25.0
    let expected_output = Array4::from_elem([1, 1, 1, 1], 117.0);
    assert_eq!(
        convolution.compute(FeatureMap::nhwc(&signal), &filters)?,
        expected_output
    );

    convolution.set_params(Params {
        strides: [1, 1],
        pads: [1; 4],
        groups: 1,
        dilation: [2, 2],
    })?;

    let expected_output = Array4::from_shape_vec(
        [1, 3, 3, 1],
        vec![
            52.0, 78.0, 52.0, //
            78.0, 117.0, 78.0, //
            52.0, 78.0, 52.0, //
        ],
    )?;
    assert_eq!(
        convolution.compute(FeatureMap::nhwc(&signal), &filters)?,
        expected_output
    );
    Ok(())
}

#[test]
fn rounding_in_i8_convolution() -> Result<(), Error> {
    const BIT_SHIFT: u8 = 8;
    let params = I8Params {
        common: Params::default(),
        bit_shift: BIT_SHIFT,
        scale: I8Params::convert_scale(BIT_SHIFT, 0.5),
        output_bias: 0,
        signal_bias: 0,
        filter_bias: 0,
    };
    let convolution = Convolution::i8(1)?.build(params)?;
    let signal = Array4::from_shape_vec([1, 2, 3, 1], vec![-7, -6, -5, 5, 6, 7])?;
    let filter = Array4::from_shape_vec([1, 1, 1, 1], vec![1])?;

    let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
    let expected_output = Array4::from_shape_vec([1, 2, 3, 1], vec![-4, -3, -2, 2, 3, 4])?;
    assert_eq!(output, expected_output);
    Ok(())
}

#[test]
fn i8_convolution() -> Result<(), Error> {
    const BIT_SHIFT: u8 = 8;
    let params = I8Params {
        common: Params::default(),
        bit_shift: BIT_SHIFT,
        scale: I8Params::convert_scale(BIT_SHIFT, 1.0),
        output_bias: 0,
        signal_bias: 0,
        filter_bias: 0,
    };
    let mut convolution = Convolution::i8(3)?.build(params)?;

    let signal = vec![
        0, 1, 2, 3, 4, //
        5, 6, 7, 8, 9, //
        10, 11, 12, 13, 14, //
        -5, -6, -7, -8, -9, //
        0, -1, -2, -3, -4, //
    ];
    let signal = Array4::from_shape_vec([1, 5, 5, 1], signal)?;
    let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![1; 9])?;

    let expected_output = vec![
        54, 63, 72, //
        33, 36, 39, //
        12, 9, 6, //
    ];
    let expected_output = Array4::from_shape_vec([1, 3, 3, 1], expected_output)?;
    let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
    assert_eq!(output, expected_output);

    // Check the same convolution with different scale / bias params.
    // We use post-conv transform |x| { x / 3 - 12 }.
    let expected_output = vec![
        6, 9, 12, //
        -1, 0, 1, //
        -8, -9, -10, //
    ];
    let expected_output = Array4::from_shape_vec([1, 3, 3, 1], expected_output)?;

    convolution.set_params(I8Params {
        common: Params::default(),
        bit_shift: BIT_SHIFT,
        scale: I8Params::convert_scale(BIT_SHIFT, 1.0 / 3.0),
        output_bias: -12,
        signal_bias: 0,
        filter_bias: 0,
    })?;
    let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
    assert_eq!(output, expected_output);

    // Check `filter_bias` / `signal_bias`.
    let signal = vec![
        0, 1, 2, 3, 4, //
        5, 6, 7, 8, 9, //
        10, 11, 12, 13, 14, //
        -5, -6, -7, -8, -9, //
        0, -1, -2, -3, -4, //
    ];
    let signal = Array4::from_shape_vec([1, 5, 5, 1], signal)? - 7;
    let filter = Array4::from_shape_vec([1, 3, 3, 1], vec![0; 9])?;

    convolution.set_params(I8Params {
        common: Params::default(),
        output_bias: -12,
        filter_bias: 1,
        signal_bias: 7,
        bit_shift: BIT_SHIFT,
        scale: I8Params::convert_scale(BIT_SHIFT, 1.0 / 3.0),
    })?;
    let output = convolution.compute(FeatureMap::nhwc(&signal), &filter)?;
    assert_eq!(output, expected_output);
    Ok(())
}

#[test]
fn i8_convolution_with_filter_bias() -> Result<(), Error> {
    const BIT_SHIFT: u8 = 8;
    const MULTIPLIER: i32 = 1 << (BIT_SHIFT as i32);

    let params = I8Params {
        common: Params::default(),
        bit_shift: BIT_SHIFT,
        scale: I8Params::convert_scale(BIT_SHIFT, 1.0 / 3.0),
        output_bias: 0,
        signal_bias: 0,
        filter_bias: 0,
    };
    let convolution = Convolution::i8(3)?.build(params)?;

    let signal = vec![
        0, 1, 2, 3, 4, //
        5, 6, 7, 8, 9, //
        10, 11, 12, 13, 14, //
        -5, -6, -7, -8, -9, //
        0, -1, -2, -3, -4, //
    ];
    let signal = Array4::from_shape_vec([1, 5, 5, 1], signal)?;
    let signal = FeatureMap::nhwc(&signal);
    let filter = Array4::from_shape_vec([2, 3, 3, 1], vec![1; 18])?;

    let expected_output = vec![
        // First filter output
        6, 9, 12, //
        -1, 0, 1, //
        -8, -9, -10, //
        // Second filter output
        17, 20, 23, //
        10, 11, 12, //
        3, 2, 1, //
    ];
    let expected_output =
        Array4::from_shape_vec([1, 2, 3, 3], expected_output)?.permuted_axes([0, 2, 3, 1]);

    let biases = &[-12 * MULTIPLIER, -MULTIPLIER];
    let output = convolution
        .compute_with_biases(signal, &filter, biases)
        .unwrap();
    assert_eq!(output, expected_output);

    // Check filter pinning.
    let convolution = convolution.with_biased_filters(&filter, biases)?;
    let output = convolution.compute(signal)?;
    assert_eq!(output, expected_output);

    let convolution = convolution.pin(FeatureMapShape {
        batch_size: 1,
        width: 5,
        height: 5,
        channels: 1,
    })?;
    let output = convolution.compute(signal)?;
    assert_eq!(output, expected_output);
    Ok(())
}

#[test]
#[allow(clippy::deref_addrof)] // the problem is in the `ndarray::s!` macro
fn f32_batching() -> Result<(), Error> {
    use ndarray::{concatenate, s};

    let mut rng = rand::rng();
    let conv = Convolution::f32(3)?.build(Params::default())?;
    let filters = Array4::from_shape_fn([2, 3, 3, 4], |_| rng.random_range(-1.0..=1.0));
    let conv = conv.with_filters(&filters)?;

    for batch_size in 2..8 {
        // Test both NHWC and NCHW layouts
        let signal_shape = if batch_size % 2 == 0 {
            [batch_size, 5, 5, 4]
        } else {
            [batch_size, 4, 5, 5]
        };
        let to_map = if batch_size % 2 == 0 {
            FeatureMap::nhwc
        } else {
            FeatureMap::nchw
        };

        let signal = Array4::from_shape_fn(signal_shape, |_| rng.random_range(-1.0..=1.0));
        let batched_output = conv.compute(to_map(signal.view()))?;

        let sample_outputs: Vec<_> = (0..batch_size)
            .map(|i| {
                let sample_signal = signal.slice(s![i..=i, .., .., ..]);
                conv.compute(to_map(sample_signal))
            })
            .collect::<Result<_, _>>()?;
        let sample_outputs: Vec<_> = sample_outputs.iter().map(|array| array.view()).collect();
        let stitched_output = concatenate(Axis(0), &sample_outputs)?;

        let max_diff = (batched_output.clone() - stitched_output.clone())
            .mapv(f32::abs)
            .fold(0.0, |acc, &x| if x > acc { x } else { acc });
        assert!(
            max_diff < f32::EPSILON,
            "batched={batched_output}, stitched={stitched_output}"
        );
    }
    Ok(())
}
