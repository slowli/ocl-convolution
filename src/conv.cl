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
};

struct __attribute__((packed)) OutputParams {
    uint batch_size;
    uchar layout;
};

/// Computes convolution of two tensors.
///
/// - `signal` should have `HxWxC` layout (i.e., the channel dimension is the inner-most one).
/// - `filters` should have `MxK_HxK_WxC` layout, where `M` is the number of filters,
///   `K_H` and `K_W` are spatial dimensions of a filter, `C` is the number of input channels.
///
/// The output will have layout `MxH'xW'` or `H'xW'xM`, depending on the `convolved_layout`
/// param.
///
/// The kernel should be launched with `[K_H * H', K_W * W', C]` workgroups,
/// with `[K_H, K_W]` items per group. Each group will compute a single output point.
__kernel void conv(
    __global float *output,
    struct OutputParams out_params,
    __constant float *signal,
    uint3 signal_dims,
    __constant float *filters,
    __constant float *filter_biases,
    struct Params params
) {
    size_t output_h = get_num_groups(0) * get_local_size(0) / out_params.batch_size;
    size_t output_w = get_num_groups(1) * get_local_size(1);
    size_t output_ch = get_num_groups(2) * get_local_size(2);

    size_t batch_idx = get_global_id(0) / output_h;
    size_t x = get_global_id(0) % output_h;
    size_t y = get_global_id(1);
    size_t filter = get_global_id(2);
    size_t channels_per_group = signal_dims.z / params.groups;
    size_t group = filter * params.groups / output_ch;

    float sum = (filter_biases == NULL) ? 0.0 : filter_biases[filter];

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

            if ((input_pt.x != -1) || (input_pt.y != -1)) {
                size_t signal_offset =
                    batch_idx * signal_dims.x * signal_dims.y * signal_dims.z +
                    input_pt.x * signal_dims.y * signal_dims.z +
                    input_pt.y * signal_dims.z +
                    channels_per_group * group;
                size_t filter_offset =
                    filter * FILTER_SIZE * FILTER_SIZE * channels_per_group +
                    o_x * FILTER_SIZE * channels_per_group +
                    o_y * channels_per_group;

                __constant float *signal_channels = &signal[signal_offset];
                __constant float *filter_channels = &filters[filter_offset];

                // Use vectorized multiply-adds: first 16-element ones, then 4-element ones,
                // and finally scalars.
                size_t i = 0;
                float16 sum_v = (float16) 0.0;
                for (; i + 16 <= channels_per_group; i += 16) {
                    float16 signal_v = vload16(i >> 4, signal_channels);
                    float16 filter_v = vload16(i >> 4, filter_channels);
                    sum_v = fma(signal_v, filter_v, sum_v);
                }
                sum += dot((float4) 1.0, sum_v.lo.lo + sum_v.lo.hi + sum_v.hi.lo + sum_v.hi.hi);

                for (; i + 4 <= channels_per_group; i += 4) {
                    sum += dot(vload4(i >> 2, signal_channels), vload4(i >> 2, filter_channels));
                }
                for (; i < channels_per_group; i++) {
                    sum = fma(signal_channels[i], filter_channels[i], sum);
                }
            }
        }
    }

    size_t out_offset = output_h * output_w * output_ch * batch_idx;
    switch (out_params.layout) {
        case LAYOUT_NCHW:
            out_offset += filter * output_h * output_w +
                x * output_w +
                y;
            break;
        case LAYOUT_NHWC:
            out_offset += x * output_w * output_ch +
                y * output_ch +
                filter;
            break;
    }
    output[out_offset] = sum;
}
