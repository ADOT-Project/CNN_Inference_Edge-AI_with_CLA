/**
 * @file convolution.c
 * @brief CPU kernels: legacy layers and the fused CLA twins.
 */
#include "convolution.h"

/** @brief Legacy float 3x3 convolution, one input channel. */
void convolution(const float* input, float* output, const float* filters, const float* bias, int in_h, int in_w)
{
    uint16_t f, r, c, fi, fj;

    int out_h = in_h - FILTER_SIZE + 1;
    int out_w = in_w - FILTER_SIZE + 1;

    for (f = 0; f < NUM_FILTERS; f++) {

        uint32_t filter_offset = f * (FILTER_SIZE * FILTER_SIZE);
        uint32_t output_offset = f * (out_h * out_w);

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {
                float sum = 0.0f;

                for (fi = 0; fi < FILTER_SIZE; fi++) {
                    uint32_t input_row_ptr = (r + fi) * in_w;

                    for (fj = 0; fj < FILTER_SIZE; fj++) {
                        sum += input[input_row_ptr + (c + fj)] * filters[filter_offset + (fi * FILTER_SIZE) + fj];
                    }
                }

                output[output_offset + (r * out_w) + c] = sum + bias[f];
            }
        }
    }
}

/** @brief Legacy float ReLU, in place. */
void relu_activation(float* data, uint32_t size)
{
    uint32_t i;

    for (i = 0; i < size; i++) {

        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

/** @brief Legacy float 2x2 max pooling, in place. */
void max_pooling(float* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters) {
    uint16_t f, r, c, i, j;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        uint32_t filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < in_h - 1; r += 2) {
            for (c = 0; c < in_w - 1; c += 2) {
                float max_val = -1e37f;

                for (i = 0; i < 2; i++) {
                    uint32_t row_ptr = filter_offset + (r + i) * in_w;
                    for (j = 0; j < 2; j++) {
                        float val = data[row_ptr + (c + j)];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }

                data[write_idx++] = max_val;
            }
        }
    }
}

/** @brief Legacy float 3x3 convolution, NUM_FILTERS input channels. */
void convolution_2(const float* input, float* output, const float* filters, const float* bias, uint16_t in_h, uint16_t in_w)
{
    uint16_t out_f, in_c, r, c, fi, fj;

    uint16_t out_h = in_h - 3 + 1;
    uint16_t out_w = in_w - 3 + 1;

    for (out_f = 0; out_f < NUM_FILTERS; out_f++) {
        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {
                float sum = 0.0f;

                for (in_c = 0; in_c < NUM_FILTERS; in_c++) {
                    uint32_t in_offset = (uint32_t)in_c * (in_h * in_w);

                    uint32_t filter_offset = (out_f * NUM_FILTERS * 9) + (in_c * 9);

                    for (fi = 0; fi < 3; fi++) {
                        uint32_t input_row_ptr = in_offset + (r + fi) * in_w;

                        for (fj = 0; fj < 3; fj++) {
                            sum += input[input_row_ptr + (c + fj)] * filters[filter_offset + fi * 3 + fj];
                        }
                    }
                }

                output[out_f * (out_h * out_w) + r * out_w + c] = sum + bias[out_f];
            }
        }
    }
}

/** @brief Legacy float 2x2 max pooling. */
void max_pooling_2(float* input, float* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters)
{
    uint16_t f, r, c, i, j;

    uint16_t out_h = in_h / 2;
    uint16_t out_w = in_w / 2;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        uint32_t input_filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {
                float max_val = -1e37f;

                for (i = 0; i < 2; i++) {
                    uint32_t row_offset = input_filter_offset + ((r * 2) + i) * in_w;

                    for (j = 0; j < 2; j++) {
                        float val = input[row_offset + (c * 2) + j];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                output[write_idx++] = max_val;
            }
        }
    }
}

/** @brief Legacy int16 convolution, single scale. */
void convolution_int16(const int16_t* input, int16_t* output, const int16_t* filters, const int16_t* bias, int in_h, int in_w, float scale_factor)
{
    uint16_t f, r, c, fi, fj;

    int out_h = in_h - FILTER_SIZE + 1;
    int out_w = in_w - FILTER_SIZE + 1;

    for (f = 0; f < NUM_FILTERS; f++) {
        uint32_t filter_offset = f * (FILTER_SIZE * FILTER_SIZE);
        uint32_t output_offset = f * (out_h * out_w);

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {

                float sum = 0;

                for (fi = 0; fi < FILTER_SIZE; fi++) {
                    uint32_t input_row_ptr = (r + fi) * in_w;

                    for (fj = 0; fj < FILTER_SIZE; fj++) {
                        int32_t in_val = input[input_row_ptr + (c + fj)];
                        int32_t weight_val = filters[filter_offset + (fi * FILTER_SIZE) + fj];

                        sum += (in_val * weight_val);
                    }
                }

                sum += bias[f];

                output[output_offset + (r * out_w) + c] = (int16_t)(sum / scale_factor);
            }
        }
    }
}

/** @brief Legacy int16 ReLU, in place. */
void relu_activation_int16(int16_t* data, uint32_t size)
{
    uint32_t i;
    for (i = 0; i < size; i++) {
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}

/** @brief Legacy int16 2x2 max pooling, in place. */
void max_pooling_int16(int16_t* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters) {
    uint16_t f, r, c, i, j;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        uint32_t filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < in_h - 1; r += 2) {
            for (c = 0; c < in_w - 1; c += 2) {

                int16_t max_val = -32768;

                for (i = 0; i < 2; i++) {
                    uint32_t row_ptr = filter_offset + (r + i) * in_w;
                    for (j = 0; j < 2; j++) {
                        int16_t val = data[row_ptr + (c + j)];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                data[write_idx++] = max_val;
            }
        }
    }
}

/** @brief Legacy int16 2x1 max pooling (rows only). */
void max_pooling_2x1_int16(const int16_t* input, int16_t* output,
                                 uint16_t in_h, uint16_t in_w, uint16_t num_filters)
{
    uint16_t f, r, c;
    uint16_t out_h = in_h / 2;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        uint32_t off = (uint32_t)f * in_h * in_w;
        for (r = 0; r < out_h; r++) {
            const int16_t* row0 = &input[off + (uint32_t)(r * 2)     * in_w];
            const int16_t* row1 = &input[off + (uint32_t)(r * 2 + 1) * in_w];
            for (c = 0; c < in_w; c++) {
                int16_t a = row0[c], b = row1[c];
                output[write_idx++] = (a > b) ? a : b;
            }
        }
    }
}

/** @brief Legacy int16 multi-channel convolution, single scale. */
void convolution_2_int16(const int16_t* input, int16_t* output, const int16_t* filters, const int16_t* bias, uint16_t in_h, uint16_t in_w, float scale_factor)
{
    uint16_t out_f, in_c, r, c, fi, fj;

    uint16_t out_h = in_h - 3 + 1;
    uint16_t out_w = in_w - 3 + 1;

    for (out_f = 0; out_f < NUM_FILTERS; out_f++) {
        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {

                float sum = 0;

                for (in_c = 0; in_c < NUM_FILTERS; in_c++) {
                    uint32_t in_offset = (uint32_t)in_c * (in_h * in_w);
                    uint32_t filter_offset = (out_f * NUM_FILTERS * 9) + (in_c * 9);

                    for (fi = 0; fi < 3; fi++) {
                        uint32_t input_row_ptr = in_offset + (r + fi) * in_w;

                        for (fj = 0; fj < 3; fj++) {
                            int32_t in_val = input[input_row_ptr + (c + fj)];
                            int32_t w_val = filters[filter_offset + fi * 3 + fj];
                            sum += (in_val * w_val);
                        }
                    }
                }

                sum += bias[out_f];
                output[out_f * (out_h * out_w) + r * out_w + c] = (int16_t)(sum / scale_factor);
            }
        }
    }
}

/** @brief Legacy int16 2x2 max pooling. */
void max_pooling_2_int16(int16_t* input, int16_t* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters)
{
    uint16_t f, r, c, i, j;

    uint16_t out_h = in_h / 2;
    uint16_t out_w = in_w / 2;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        uint32_t input_filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {

                int16_t max_val = -32768;

                for (i = 0; i < 2; i++) {
                    uint32_t row_offset = input_filter_offset + ((r * 2) + i) * in_w;

                    for (j = 0; j < 2; j++) {
                        int16_t val = input[row_offset + (c * 2) + j];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                output[write_idx++] = max_val;
            }
        }
    }
}

/** @brief Emit the fused CPU kernels from fused_kernels_opt.h. */
#define KFN_CONV1  conv1_pool_fused_cpu
#define KFN_CONV2  conv2_pool_fused_cpu
#define KFN_DENSE  dense_fc_cpu
#include "fused_kernels_opt.h"
