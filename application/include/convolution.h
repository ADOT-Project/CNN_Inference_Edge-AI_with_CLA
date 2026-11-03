/**
 * @file convolution.h
 * @brief CPU convolution, pooling and dense kernels.
 */
#include <stdlib.h>
#include <stdint.h>
#include "definitions.h"

#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

/** @brief Legacy float 3x3 convolution (unused). */
void convolution(const float* input, float* output, const float* filters, const float* bias, int in_h, int in_w);

/** @brief Legacy float ReLU (unused). */
void relu_activation(float* data, uint32_t size);

/** @brief Legacy float max pooling (unused). */
void max_pooling(float* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters);

/** @brief Legacy float second convolution (unused). */
void convolution_2(const float* input, float* output, const float* filters, const float* bias, uint16_t in_h, uint16_t in_w);

/** @brief Legacy float second pooling (unused). */
void max_pooling_2(float* input, float* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters);

/** @brief Legacy int16 convolution, single scale (unused). */
void convolution_int16(const int16_t* input, int16_t* output, const int16_t* filters, const int16_t* bias, int in_h, int in_w, float scale_factor);

/** @brief Legacy int16 ReLU (unused). */
void relu_activation_int16(int16_t* data, uint32_t size);

/** @brief Legacy int16 max pooling (unused). */
void max_pooling_int16(int16_t* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters);

/** @brief Legacy int16 2x1 max pooling (unused). */
void max_pooling_2x1_int16(const int16_t* input, int16_t* output,
                                 uint16_t in_h, uint16_t in_w, uint16_t num_filters);

/** @brief Legacy int16 second convolution (unused). */
void convolution_2_int16(const int16_t* input, int16_t* output, const int16_t* filters, const int16_t* bias, uint16_t in_h, uint16_t in_w, float scale_factor);

/** @brief Legacy int16 second pooling (unused). */
void max_pooling_2_int16(int16_t* input, int16_t* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters);

/**
 * @brief Fused conv1 + ReLU + 2x1 pool on the CPU (CLA twin).
 * @param input      IMAGE_H x IMAGE_W features.
 * @param pooled_out NUM_FILTERS x POOL1_OUT_H x POOL1_OUT_W.
 * @param filters    conv1 weights.
 * @param bias       conv1 bias.
 */
void conv1_pool_fused_cpu(const int16_t* input, int16_t* pooled_out,
                           const int16_t* filters, const int16_t* bias);

/**
 * @brief Fused conv2 + ReLU + 2x1 pool on the CPU (CLA twin).
 * @param input      pool1 output.
 * @param pooled_out NUM_FILTERS x POOL2_OUT_H x POOL2_OUT_W.
 * @param filters    conv2 weights.
 * @param bias       conv2 bias.
 */
void conv2_pool_fused_cpu(const int16_t* input, int16_t* pooled_out,
                           const int16_t* filters, const int16_t* bias);

/**
 * @brief Dense layer on the CPU (CLA twin).
 * @param input   DENSE_LAYER_INPUT values.
 * @param weights Row-major, one row per class.
 * @param bias    NUM_CLASSES values.
 * @param output  NUM_CLASSES float logits.
 */
void dense_fc_cpu(const int16_t* input, const int16_t* weights,
                   const int16_t* bias, float* output);

#endif
