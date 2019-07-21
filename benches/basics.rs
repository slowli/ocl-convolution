use criterion::{criterion_group, criterion_main, Bencher, Criterion, ParameterizedBenchmark};
use ndarray::{Array4, Axis};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use ocl_convolution::{Convolution, FeatureMap, FeatureMapShape, I8Params, Params};

const INPUT_SIZE: usize = 28;
const CHANNELS: &[usize] = &[8, 16, 32, 64, 128, 256];
const SAMPLE_SIZE: usize = 20;

#[derive(Clone, Copy)]
enum Memory {
    Simple,
    Filters,
    Pinned,
}

fn run_convolution(bencher: &mut Bencher, channels: usize, memory: Memory) {
    let convolution = Convolution::f32(3)
        .unwrap()
        .build(Params::default())
        .unwrap();
    let mut rng = XorShiftRng::from_seed(*b"!seed seed seed!");

    let mut signal = Array4::zeros([1, INPUT_SIZE, INPUT_SIZE, channels]);
    signal
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-1.0, 1.0));
    let signal = FeatureMap::nhwc(&signal);

    let mut filters = Array4::zeros([channels, 3, 3, channels]);
    filters
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-1.0, 1.0));

    match memory {
        Memory::Simple => {
            bencher.iter(|| convolution.compute(signal, &filters).unwrap());
        }
        Memory::Filters => {
            let convolution = convolution.with_filters(&filters).unwrap();
            bencher.iter(|| convolution.compute(signal).unwrap());
        }
        Memory::Pinned => {
            let convolution = convolution
                .with_filters(&filters)
                .unwrap()
                .pin(FeatureMapShape {
                    batch_size: 1,
                    width: INPUT_SIZE,
                    height: INPUT_SIZE,
                    channels,
                })
                .unwrap();
            bencher.iter(|| convolution.compute(signal).unwrap());
        }
    }
}

fn run_batched_convolution(bencher: &mut Bencher, channels: usize, sequential: bool) {
    const BATCH_SIZE: usize = 8;

    let convolution = Convolution::f32(3)
        .unwrap()
        .build(Params::default())
        .unwrap();
    let mut rng = XorShiftRng::from_seed(*b"!seed seed seed!");

    let mut signal = Array4::zeros([BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, channels]);
    signal
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-1.0, 1.0));

    let mut filters = Array4::zeros([channels, 3, 3, channels]);
    filters
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-1.0, 1.0));

    let convolution = convolution.with_filters(filters.view()).unwrap();
    if sequential {
        bencher.iter(|| {
            (0..BATCH_SIZE)
                .map(|i| {
                    let sample_signal = signal.index_axis(Axis(0), i).insert_axis(Axis(0));
                    convolution
                        .compute(FeatureMap::nhwc(&sample_signal))
                        .unwrap()
                })
                .collect::<Vec<_>>()
        });
    } else {
        bencher.iter(|| convolution.compute(FeatureMap::nhwc(&signal)).unwrap());
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
    let convolution = Convolution::i8(3).unwrap().build(params).unwrap();
    let mut rng = XorShiftRng::from_seed(*b"_seed_seed_seed_");

    let mut signal = Array4::zeros([1, INPUT_SIZE, INPUT_SIZE, channels]);
    signal
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-127, 127));
    let signal = FeatureMap::nhwc(&signal);
    let mut filters = Array4::zeros([channels, 3, 3, channels]);
    filters
        .iter_mut()
        .for_each(|v| *v = rng.gen_range(-127, 127));

    match memory {
        Memory::Simple => bencher.iter(|| convolution.compute(signal, &filters).unwrap()),
        Memory::Filters => {
            let convolution = convolution.with_filters(&filters).unwrap();
            bencher.iter(|| convolution.compute(signal).unwrap())
        }
        Memory::Pinned => {
            let convolution = convolution
                .with_filters(&filters)
                .unwrap()
                .pin(FeatureMapShape {
                    batch_size: 1,
                    width: INPUT_SIZE,
                    height: INPUT_SIZE,
                    channels,
                })
                .unwrap();
            bencher.iter(|| convolution.compute(signal).unwrap())
        }
    }
}

fn basic_benches(criterion: &mut Criterion) {
    criterion.bench(
        "f32_conv",
        ParameterizedBenchmark::new(
            "channels",
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
        "batched_f32_conv",
        ParameterizedBenchmark::new(
            "channels",
            |bencher, &&channels| run_batched_convolution(bencher, channels, false),
            &CHANNELS[1..4],
        )
        .sample_size(SAMPLE_SIZE)
        .with_function("seq", |bencher, &&channels| {
            run_batched_convolution(bencher, channels, true)
        }),
    );

    criterion.bench(
        "i8_conv",
        ParameterizedBenchmark::new(
            "channels",
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
