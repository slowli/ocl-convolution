#define LAYOUT_NCHW 0
#define LAYOUT_NHWC 1
#define MAX_POINT ((uint2)(-1, -1))
#ifndef NULL
#define NULL 0
#endif

/// Maps the output coordinate to an input one.
uint2 map_to_input(
    uint2 output_pt,
    uint2 offsets,
    uint2 strides,
    uint2 dilation,
    uint4 pads,
    uint2 signal_dims
) {
    uint2 without_pads = output_pt * strides + offsets * dilation;

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
    uint groups;
    uint2 dilation;
    int bit_shift;
    int scale;
    int output_bias;
    int signal_bias;
    int filter_bias;
};

// Round to nearest, ties to even integer bit shift to the right.
int rounded_rshift(int number, int bit_shift) {
    int mask = (1 << bit_shift) - 1;
    int threshold = 1 << (bit_shift - 1);
    int lower_bits = number & mask;
    int result = number >> bit_shift;
    // We assume `lower_bits >= 0`.

    if ((lower_bits > threshold) || ((lower_bits == threshold) && ((result & 1) == 1))) {
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
    uchar convolved_layout,
    __constant char *signal,
    uint3 signal_dims,
    __constant char *filters,
    __constant int *filter_biases,
    struct Params params
) {
    size_t convolved_h = get_num_groups(0) * get_local_size(0);
    size_t convolved_w = get_num_groups(1) * get_local_size(1);
    size_t convolved_ch = get_num_groups(2) * get_local_size(2);

    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t filter = get_global_id(2);
    size_t channels_per_group = signal_dims.z / params.groups;
    size_t group = filter * params.groups / convolved_ch;

    int sum = 0;

    for (uint o_x = 0; o_x < FILTER_SIZE; o_x++) {
        for (uint o_y = 0; o_y < FILTER_SIZE; o_y++) {
            uint2 input_pt = map_to_input(
                (uint2)(x, y),
                (uint2)(o_x, o_y),
                params.strides,
                params.dilation,
                params.pads,
                signal_dims.xy
            );

            if ((input_pt.x != -1) && (input_pt.y != -1)) {
                size_t signal_offset =
                    input_pt.x * signal_dims.y * signal_dims.z +
                    input_pt.y * signal_dims.z +
                    channels_per_group * group;
                size_t filter_offset =
                    filter * FILTER_SIZE * FILTER_SIZE * channels_per_group +
                    o_x * FILTER_SIZE * channels_per_group +
                    o_y * channels_per_group;

                __constant char *signal_channels = &signal[signal_offset];
                __constant char *filter_channels = &filters[filter_offset];

                int16 sum_v = (int16) 0;
                size_t i = 0;
                for (; i + 16 <= signal_dims.z; i += 16) {
                    char16 signal_lane = vload16(i >> 4, signal_channels);
                    char16 filter_lane = vload16(i >> 4, filter_channels);
                    int16 signal_val = convert_int16(signal_lane) + params.signal_bias;
                    int16 filter_val = convert_int16(filter_lane) + params.filter_bias;
                    sum_v = mad24(signal_val, filter_val, sum_v);
                }
                int4 reduced_sum_v = sum_v.lo.lo + sum_v.lo.hi + sum_v.hi.lo + sum_v.hi.hi;
                sum += reduced_sum_v.x + reduced_sum_v.y + reduced_sum_v.z + reduced_sum_v.w;

                for (; i < channels_per_group; i++) {
                    sum = mad24(
                        signal_channels[i] + params.signal_bias,
                        filter_channels[i] + params.filter_bias,
                        sum
                    );
                }
            }
        }
    }

    // Perform a linear transform.
    sum *= params.scale;
    if (filter_biases != NULL) {
        sum += filter_biases[filter];
    }
    sum = rounded_rshift(sum, params.bit_shift) + params.output_bias;

    size_t output_offset = 0;
    if (convolved_layout == LAYOUT_NCHW) {
        output_offset = filter * convolved_h * convolved_w +
            x * convolved_w +
            y;
    } else if (convolved_layout == LAYOUT_NHWC) {
        output_offset = x * convolved_w * convolved_ch +
            y * convolved_ch +
            filter;
    }
    convolved[output_offset] = convert_char_sat(sum);
}
