/**
 * @file fused_kernels_opt.h
 * @brief Pointer-walking fused conv/pool/dense kernels.
 * @details Define KFN_CONV1, KFN_CONV2 or KFN_DENSE to the function name
 * before including to emit that kernel.
 */
#ifndef FUSED_KERNELS_OPT_H_
#define FUSED_KERNELS_OPT_H_

#include <stdint.h>
#include "definitions.h"

#ifndef KERNEL_ADDR_WALK
#define KERNEL_ADDR_WALK 1 /**< 1: walk pointers, 0: constant offsets. */
#endif
#ifndef KERNEL_USE_RECIPROCAL
#define KERNEL_USE_RECIPROCAL 0 /**< 1: multiply by 1/scale (not bit-exact). */
#endif
#ifndef KERNEL_POOL_FLOAT
#define KERNEL_POOL_FLOAT 1 /**< 1: pool on float sums, 0: on int16. */
#endif

#if (FILTER_SIZE != 3)
#error "fused_kernels_opt.h is written for 3x3 filters"
#endif

/** @brief Removes a quantization scale. */
#if KERNEL_USE_RECIPROCAL
#define K_DESCALE(x, scale)   ((x) * (1.0f / (scale)))
#else
#define K_DESCALE(x, scale)   ((x) / (scale))
#endif

/** @brief One multiply-accumulate term. */
#define K_MAC(x, y)           ((float)(x) * (float)(y))

/** @brief ReLU + 2x1 max-pool helpers. */
#if KERNEL_POOL_FLOAT
#define K_POOL_DECL           float k_m;
#define K_POOL_INIT           k_m = 0.0f;
#define K_POOL_ACC(sum, S)    if ((sum) > k_m) k_m = (sum);
#define K_POOL_OUT(S)         ((int16_t)K_DESCALE(k_m, S))
#else
#define K_POOL_DECL           int16_t k_m; int16_t k_v;
#define K_POOL_INIT           k_m = -32768;
#define K_POOL_ACC(sum, S)    if ((sum) < 0.0f) (sum) = 0.0f;              \
                              k_v = (int16_t)K_DESCALE(sum, S);            \
                              if (k_v > k_m) k_m = k_v;
#define K_POOL_OUT(S)         (k_m)
#endif

/** @brief 3x3 window MAC and step to the next input plane. */
#if KERNEL_ADDR_WALK

#define K_TAPS9(sum, a, w, ROW_W)                               \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a,   *w++);   a += (ROW_W) - 2;               \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a,   *w++);   a += (ROW_W) - 2;               \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a,   *w++);

#define K_NEXT_PLANE(a, ROW_W, PLANE)                           \
    a += (PLANE) - 2 * (ROW_W) - 2;

#else

#define K_TAPS9(sum, a, w, ROW_W)                               \
    sum += K_MAC(a[0],               w[0]);                     \
    sum += K_MAC(a[1],               w[1]);                     \
    sum += K_MAC(a[2],               w[2]);                     \
    sum += K_MAC(a[(ROW_W)],         w[3]);                     \
    sum += K_MAC(a[(ROW_W) + 1],     w[4]);                     \
    sum += K_MAC(a[(ROW_W) + 2],     w[5]);                     \
    sum += K_MAC(a[2 * (ROW_W)],     w[6]);                     \
    sum += K_MAC(a[2 * (ROW_W) + 1], w[7]);                     \
    sum += K_MAC(a[2 * (ROW_W) + 2], w[8]);                     \
    w += 9;

#define K_NEXT_PLANE(a, ROW_W, PLANE)                           \
    a += (PLANE);

#endif

/** @brief Dense layer body: NUM_CLASSES dot products plus bias. */
#define K_DENSE_BODY(IN, W, B, OUT)                                         \
    {                                                                       \
        register const int16_t *kd_x;                                       \
        register const int16_t *kd_w = (W);                                 \
        register float kd_sum;                                              \
        const int16_t *kd_b = (B);                                          \
        float *kd_o = (OUT);                                                \
        uint16_t kd_k, kd_i;                                                \
        for (kd_k = NUM_CLASSES; kd_k != 0; kd_k--) {                       \
            kd_x = (IN);                                                    \
            kd_sum = 0.0f;                                                  \
            for (kd_i = DENSE_LAYER_INPUT; kd_i != 0; kd_i--) {             \
                kd_sum += K_MAC(*kd_x++, *kd_w++);                          \
            }                                                               \
            *kd_o++ = K_DESCALE(kd_sum, FC_WEIGHT_SCALE)                    \
                    + (float)(*kd_b++) * (INPUT_SCALE / FC_BIAS_SCALE);     \
        }                                                                   \
    }

#endif

#ifdef KFN_CONV1

/**
 * @brief Fused conv1 + ReLU + 2x1 pool.
 * @param input      IMAGE_H x IMAGE_W features.
 * @param pooled_out Pool1 output.
 * @param filters    conv1 weights.
 * @param bias       conv1 bias.
 */
void KFN_CONV1(const int16_t* input, int16_t* pooled_out,
               const int16_t* filters, const int16_t* bias)
{
    register const int16_t *a;
    register const int16_t *w;
    register float sum;
    const int16_t *wf = filters;
    const int16_t *row;
    const int16_t *col;
    const int16_t *q;
    float b;
    K_POOL_DECL
    uint16_t f, pr, c, sub;

    for (f = NUM_FILTERS; f != 0; f--) {
        b = (float)(*bias++) * ((CONV1_WEIGHT_SCALE / CONV1_BIAS_SCALE) * INPUT_SCALE);
        row = input;
        for (pr = POOL1_OUT_H; pr != 0; pr--) {
            col = row;
            for (c = CONV1_OUT_W; c != 0; c--) {
                K_POOL_INIT
                q = col;
                for (sub = 2; sub != 0; sub--) {
                    a = q;
                    w = wf;
                    sum = 0.0f;
                    K_TAPS9(sum, a, w, IMAGE_W)
                    sum += b;
                    K_POOL_ACC(sum, CONV1_WEIGHT_SCALE)
                    q += IMAGE_W;
                }
                *pooled_out++ = K_POOL_OUT(CONV1_WEIGHT_SCALE);
                col++;
            }
            row += 2 * IMAGE_W;
        }
        wf += FILTER_SIZE * FILTER_SIZE;
    }
}
#undef KFN_CONV1
#endif

#ifdef KFN_CONV2

/**
 * @brief Fused conv2 + ReLU + 2x1 pool.
 * @param input      Pool1 output.
 * @param pooled_out Pool2 output (dense input).
 * @param filters    conv2 weights.
 * @param bias       conv2 bias.
 */
void KFN_CONV2(const int16_t* input, int16_t* pooled_out,
               const int16_t* filters, const int16_t* bias)
{
    register const int16_t *a;
    register const int16_t *w;
    register float sum;
    const int16_t *wf = filters;
    const int16_t *row;
    const int16_t *col;
    const int16_t *q;
    float b;
    K_POOL_DECL
    uint16_t f, pr, c, sub, ic;

    for (f = NUM_FILTERS; f != 0; f--) {
        b = (float)(*bias++) * ((CONV2_WEIGHT_SCALE / CONV2_BIAS_SCALE) * INPUT_SCALE);
        row = input;
        for (pr = POOL2_OUT_H; pr != 0; pr--) {
            col = row;
            for (c = CONV2_OUT_W; c != 0; c--) {
                K_POOL_INIT
                q = col;
                for (sub = 2; sub != 0; sub--) {
                    a = q;
                    w = wf;
                    sum = 0.0f;
                    for (ic = NUM_FILTERS; ic != 0; ic--) {
                        K_TAPS9(sum, a, w, POOL1_OUT_W)
                        K_NEXT_PLANE(a, POOL1_OUT_W, POOL1_OUT_H * POOL1_OUT_W)
                    }
                    sum += b;
                    K_POOL_ACC(sum, CONV2_WEIGHT_SCALE)
                    q += POOL1_OUT_W;
                }
                *pooled_out++ = K_POOL_OUT(CONV2_WEIGHT_SCALE);
                col++;
            }
            row += 2 * POOL1_OUT_W;
        }
        wf += NUM_FILTERS * FILTER_SIZE * FILTER_SIZE;
    }
}
#undef KFN_CONV2
#endif

#ifdef KFN_DENSE
/**
 * @brief Dense layer.
 * @param input   DENSE_LAYER_INPUT values.
 * @param weights One row per class.
 * @param bias    NUM_CLASSES values.
 * @param output  NUM_CLASSES float logits.
 */
void KFN_DENSE(const int16_t* input, const int16_t* weights,
               const int16_t* bias, float* output)
{
    K_DENSE_BODY(input, weights, bias, output)
}
#undef KFN_DENSE
#endif
