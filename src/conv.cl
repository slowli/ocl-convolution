#define MAX_POINT ((uint2)(-1, -1))

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

/// Computes convolution of two tensors.
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
    __global float *convolved,
    __constant float *signal,
    uint3 signal_dims,
    __constant float *filters,
    __constant float *filter_biases,
    uint2 strides,
    uint4 pads
) {
    size_t x = get_group_id(0);
    size_t y = get_group_id(1);
    size_t filter = get_group_id(2);
    size_t convolved_h = get_num_groups(0);
    size_t convolved_w = get_num_groups(1);

    uint2 offset = (uint2)(get_local_id(0), get_local_id(1));

    float sum = 0.0;
    uint2 input_pt = map_to_input((uint2)(x, y), offset, strides, pads, signal_dims.xy);
    if ((input_pt.x != -1) && (input_pt.y != -1)) {
        // First, we vectorize `float` channels into 4-value vectors to calculate
        // their dot product ~4 times as fast.
        size_t signal_offset =
            input_pt.x * signal_dims.y * signal_dims.z +
            input_pt.y * signal_dims.z;
        __constant float4 *signal_channels = (__constant float4*) &signal[signal_offset];

        size_t filter_offset =
            filter * FILTER_SIZE * FILTER_SIZE * signal_dims.z +
            offset.x * FILTER_SIZE * signal_dims.z +
            offset.y * signal_dims.z;
        __constant float4 *filter_channels = (__constant float4*) &filters[filter_offset];
        for (size_t i = 0; i < signal_dims.z / 4; i++) {
            sum += dot(signal_channels[i], filter_channels[i]);
        }

        // Add remaining elements without vectorization.
        __constant float *signal_channels_ = &signal[signal_offset];
        __constant float *filter_channels_ = &filters[filter_offset];
        for (size_t i = signal_dims.z & ~3; i < signal_dims.z; i++) {
            sum = fma(signal_channels_[i], filter_channels_[i], sum);
        }
    }

    __local float results[FILTER_SIZE][FILTER_SIZE];
    results[offset.x][offset.y] = sum;

    // Wait for all workers in the group to submit their results.
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((offset.x == 0) && (offset.y == 0)) {
        float final_sum = 0.0;
        for (size_t o_x = 0; o_x < FILTER_SIZE; o_x++) {
            for (size_t o_y = 0; o_y < FILTER_SIZE; o_y++) {
                final_sum += results[o_x][o_y];
            }
        }
        if (filter_biases != NULL) {
            final_sum += filter_biases[filter];
        }

        size_t output_offset =
            filter * convolved_h * convolved_w +
            x * convolved_w +
            y;
        convolved[output_offset] = final_sum;
    }
}
