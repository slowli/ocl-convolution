use criterion::{criterion_group, criterion_main, Bencher, Criterion, ParameterizedBenchmark};
use ndarray::{Array3, Array4};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use ocl_convolution::{Convolution, I8Params, Params};

const INPUT_SIZE: usize = 28;
const CHANNELS: &[usize] = &[8, 16, 32, 64, 128, 256, 512];
const SAMPLE_SIZE: usize = 20;

#[derive(Clone, Copy)]
enum Memory {
    Simple,
    Filters,
    Pinned,
}

fn run_convolution(bencher: &mut Bencher, channels: usize, memory: Memory) {
    let convolution = Convolution::new(3, Params::default()).unwrap();
    let mut rng = XorShiftRng::from_seed(*b"!seed seed seed!");

    let mut signal = Array3::zeros([INPUT_SIZE, INPUT_SIZE, channels]);
    signal
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-1.0, 1.0));

    let mut filters = Array4::zeros([channels, 3, 3, channels]);
    filters
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-1.0, 1.0));

    match memory {
        Memory::Simple => {
            bencher.iter(|| convolution.compute(signal.view(), filters.view()).unwrap())
        }
        Memory::Filters => {
            let convolution = convolution.with_filters(filters.view()).unwrap();
            bencher.iter(|| convolution.compute(signal.view()).unwrap())
        }
        Memory::Pinned => {
            let convolution = convolution
                .with_filters(filters.view())
                .unwrap()
                .pinned([INPUT_SIZE, INPUT_SIZE, channels])
                .unwrap();
            bencher.iter(|| convolution.compute(signal.view()).unwrap())
        }
    }
}

fn run_i8_convolution(bencher: &mut Bencher, channels: usize, memory: Memory) {
    const BIT_SHIFT: u8 = 8;

    let scale = I8Params::convert_scale(BIT_SHIFT, (channels as f32).sqrt().recip());
    let params = I8Params {
        common: Params::default(),
        bit_shift: BIT_SHIFT,
        scale,
        output_bias: 0,
        signal_bias: 0,
        filter_bias: 0,
    };
    let convolution = Convolution::quantized(3, params).unwrap();
    let mut rng = XorShiftRng::from_seed(*b"_seed_seed_seed_");

    let mut signal = Array3::zeros([INPUT_SIZE, INPUT_SIZE, channels]);
    signal
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-127, 127));
    let mut filters = Array4::zeros([channels, 3, 3, channels]);
    filters
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-127, 127));

    match memory {
        Memory::Simple => {
            bencher.iter(|| convolution.compute(signal.view(), filters.view()).unwrap())
        }
        Memory::Filters => {
            let convolution = convolution.with_filters(filters.view()).unwrap();
            bencher.iter(|| convolution.compute(signal.view()).unwrap())
        }
        Memory::Pinned => {
            let convolution = convolution
                .with_filters(filters.view())
                .unwrap()
                .pinned([INPUT_SIZE, INPUT_SIZE, channels])
                .unwrap();
            bencher.iter(|| convolution.compute(signal.view()).unwrap())
        }
    }
}

fn basic_benches(criterion: &mut Criterion) {
    criterion.bench(
        "vectorized_conv",
        ParameterizedBenchmark::new(
            "input_size",
            |bencher, &&channels| run_convolution(bencher, channels, Memory::Simple),
            CHANNELS,
        )
        .sample_size(SAMPLE_SIZE)
        .with_function("with_filters", |bencher, &&channels| {
            run_convolution(bencher, channels, Memory::Filters)
        })
        .with_function("pinned", |bencher, &&channels| {
            run_convolution(bencher, channels, Memory::Pinned)
        }),
    );

    criterion.bench(
        "i8_conv",
        ParameterizedBenchmark::new(
            "input_size",
            |bencher, &&channels| {
                run_i8_convolution(bencher, channels, Memory::Simple);
            },
            CHANNELS,
        )
        .sample_size(SAMPLE_SIZE)
        .with_function("with_filters", |bencher, &&channels| {
            run_i8_convolution(bencher, channels, Memory::Filters)
        })
        .with_function("pinned", |bencher, &&channels| {
            run_i8_convolution(bencher, channels, Memory::Pinned)
        }),
    );
}

criterion_group!(benches, basic_benches);
criterion_main!(benches);
