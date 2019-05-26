use criterion::{criterion_group, criterion_main, Bencher, Criterion, ParameterizedBenchmark};
use ndarray::{Array3, Array4};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use ocl_convolution::{Convolution, I8Convolution, I8Params};

const INPUT_SIZES: &[usize] = &[8, 16, 32, 64, 128, 256];
const CHANNELS: usize = 32;
const SAMPLE_SIZE: usize = 20;

fn run_convolution(bencher: &mut Bencher, input_size: usize) {
    let convolution = Convolution::new(3).unwrap();
    let mut rng = XorShiftRng::from_seed(*b"!seed seed seed!");

    let mut signal = Array3::zeros([input_size, input_size, CHANNELS]);
    signal
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-1.0, 1.0));

    let mut filters = Array4::zeros([CHANNELS, 3, 3, CHANNELS]);
    filters
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-1.0, 1.0));

    bencher.iter(|| {
        convolution
            .compute(signal.view(), filters.view(), [1, 1], [0; 4])
            .unwrap()
    });
}

fn run_i8_convolution(bencher: &mut Bencher, input_size: usize) {
    const BIT_SHIFT: u8 = 8;

    let convolution = I8Convolution::new(3, BIT_SHIFT).unwrap();
    let mut rng = XorShiftRng::from_seed(*b"_seed_seed_seed_");

    let mut signal = Array3::zeros([input_size, input_size, CHANNELS]);
    signal
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-127, 127));

    let mut filters = Array4::zeros([CHANNELS, 3, 3, CHANNELS]);
    filters
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-127, 127));

    let scale = I8Params::convert_scale(BIT_SHIFT, (CHANNELS as f32).sqrt().recip());
    let params = I8Params {
        strides: [1, 1],
        pads: [0; 4],
        scale,
        output_bias: 0,
        signal_bias: 0,
        filter_bias: 0,
    };

    bencher.iter(|| {
        convolution
            .compute(signal.view(), filters.view(), None, params)
            .unwrap()
    });
}

fn basic_benches(criterion: &mut Criterion) {
    criterion.bench(
        "vectorized_conv",
        ParameterizedBenchmark::new(
            "input_size",
            |bencher, &&size| {
                run_convolution(bencher, size);
            },
            INPUT_SIZES,
        )
        .sample_size(SAMPLE_SIZE),
    );

    criterion.bench(
        "i8_conv",
        ParameterizedBenchmark::new(
            "input_size",
            |bencher, &&size| {
                run_i8_convolution(bencher, size);
            },
            INPUT_SIZES,
        )
        .sample_size(SAMPLE_SIZE),
    );
}

criterion_group!(benches, basic_benches);
criterion_main!(benches);
