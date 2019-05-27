/// Filter size.
// #define FILTER_SIZE 3

/// Bit shift used in scaling operations.
// #define BIT_SHIFT 8

#define MAX_POINT ((uint2)(-1, -1))
#ifndef NULL
#define NULL 0
#endif

/// Maps the output coordinate to an input one.
uint2 map_to_input(
    uint2 output_pt,
    uint2 offsets,
    uint2 strides,
    uint4 pads,
    uint2 signal_dims
) {
    uint2 without_pads = output_pt * strides + offsets;

    if ((without_pads.x < pads.x) || (without_pads.y < pads.y)) {
        return MAX_POINT;
    }

    uint2 input_pt = without_pads - pads.xy;
    if ((input_pt.x >= signal_dims.x) || (input_pt.y >= signal_dims.y)) {
        return MAX_POINT;
    }

    return input_pt;
}

struct __attribute__((packed)) Params {
    uint2 strides;
    uint4 pads;
    int scale;
    int output_bias;
    int signal_bias;
    int filter_bias;
};

// Round to nearest, ties to even integer bit shift to the right.
int rounded_rshift(int number) {
    const int MASK = (1 << BIT_SHIFT) - 1;
    const int THRESHOLD = 1 << (BIT_SHIFT - 1);

    int lower_bits = number & MASK;
    int result = number >> BIT_SHIFT;
    // We assume `lower_bits >= 0`.

    if ((lower_bits > THRESHOLD) || ((lower_bits == THRESHOLD) && ((result & 1) == 1))) {
        result += 1;
    }
    return result;
}

/// Computes convolution of two `i8` (or `char`, in OpenCL C terms) tensors.
///
/// - `signal` should have `HxWxC` layout (i.e., the channel dimension is the inner-most one).
/// - `filters` should have `MxK_HxK_WxC` layout, where `M` is the number of filters,
///   `K_H` and `K_W` are spatial dimensions of a filter, `C` is the number of input channels.
///
/// The output will have form `MxH'xW'`.
///
/// The kernel should be launched with `[K_H * H', K_W * W', C]` workgroups,
/// with `[K_H, K_W]` tasks per group. Each group will compute a single output point.
__kernel void conv(
    __global char *convolved,
    __constant char *signal,
    uint3 signal_dims,
    __constant char *filters,
    __constant int *filter_biases,
    struct Params params
) {
    size_t x = get_group_id(0);
    size_t y = get_group_id(1);
    size_t filter = get_group_id(2);
    size_t convolved_h = get_num_groups(0);
    size_t convolved_w = get_num_groups(1);

    uint2 offset = (uint2)(get_local_id(0), get_local_id(1));

    int sum = 0;
    uint2 input_pt = map_to_input(
        (uint2)(x, y),
        offset,
        params.strides,
        params.pads,
        signal_dims.xy
    );
    if ((input_pt.x != -1) && (input_pt.y != -1)) {
        size_t signal_offset =
            input_pt.x * signal_dims.y * signal_dims.z +
            input_pt.y * signal_dims.z;
        size_t filter_offset =
            filter * FILTER_SIZE * FILTER_SIZE * signal_dims.z +
            offset.x * FILTER_SIZE * signal_dims.z +
            offset.y * signal_dims.z;

        int8 vec_sum = (int8)(0);
        __constant char8 *signal_channels = (__constant char8*) &signal[signal_offset];
        __constant char8 *filter_channels = (__constant char8*) &filters[filter_offset];
        for (size_t i = 0; i < signal_dims.z / 8; i++) {
            int8 signal_val = convert_int8(signal_channels[i]) + params.signal_bias;
            int8 filter_val = convert_int8(filter_channels[i]) + params.filter_bias;
            vec_sum = mad24(signal_val, filter_val, vec_sum);
        }
        sum = vec_sum.s0 + vec_sum.s1 + vec_sum.s2 + vec_sum.s3
            + vec_sum.s4 + vec_sum.s5 + vec_sum.s6 + vec_sum.s7;

        // Add remaining elements without vectorization.
        __constant char *signal_channels_ = &signal[signal_offset];
        __constant char *filter_channels_ = &filters[filter_offset];
        for (size_t i = signal_dims.z & ~7; i < signal_dims.z; i++) {
            sum = mad24(
                params.signal_bias + signal_channels_[i],
                params.filter_bias + filter_channels_[i],
                sum
            );
        }
    }

    __local float results[FILTER_SIZE][FILTER_SIZE];
    results[offset.x][offset.y] = sum;

    // Wait for all workers in the group to submit their results.
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((offset.x == 0) && (offset.y == 0)) {
        int final_sum = 0;

        for (size_t o_x = 0; o_x < FILTER_SIZE; o_x++) {
            for (size_t o_y = 0; o_y < FILTER_SIZE; o_y++) {
                final_sum += results[o_x][o_y];
            }
        }

        // Perform a linear transform.
        final_sum *= params.scale;
        if (filter_biases != NULL) {
            final_sum += filter_biases[filter];
        }
        final_sum = rounded_rshift(final_sum) + params.output_bias;

        size_t output_offset =
            filter * convolved_h * convolved_w +
            x * convolved_w +
            y;
        convolved[output_offset] = convert_char_sat(final_sum);
    }
}
