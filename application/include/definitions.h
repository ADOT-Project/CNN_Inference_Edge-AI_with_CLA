/**
 * @file definitions.h
 * @brief Network sizes and quantization scales.
 * @note ADOT export updates the values; comments are kept.
 */
#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#define IMAGE_H              65 /**< Input frames. */
#define IMAGE_W              12 /**< Features per frame. */
#define NUM_FILTERS          12 /**< Filters per conv layer. */
#define NUM_CLASSES          2 /**< Output classes. */

#define FILTER_SIZE          3 /**< 3x3 kernels. */

#define CONV1_OUT_H          (IMAGE_H - FILTER_SIZE + 1)
#define CONV1_OUT_W          (IMAGE_W - FILTER_SIZE + 1)
#define POOL1_OUT_H          (CONV1_OUT_H / 2) /**< 2x1 pool halves H only. */
#define POOL1_OUT_W          (CONV1_OUT_W)

#define CONV2_OUT_H          (POOL1_OUT_H - FILTER_SIZE + 1)
#define CONV2_OUT_W          (POOL1_OUT_W - FILTER_SIZE + 1)
#define POOL2_OUT_H          (CONV2_OUT_H / 2)
#define POOL2_OUT_W          (CONV2_OUT_W)

#define CONV1_SIZE           (NUM_FILTERS * CONV1_OUT_H * CONV1_OUT_W)
#define POOL1_SIZE           (NUM_FILTERS * POOL1_OUT_H * POOL1_OUT_W)
#define CONV2_SIZE           (NUM_FILTERS * CONV2_OUT_H * CONV2_OUT_W)

#define DENSE_LAYER_INPUT    (NUM_FILTERS * POOL2_OUT_H * POOL2_OUT_W) /**< Flattened pool2. */
#define DENSE_LAYER_WEIGHTS  (DENSE_LAYER_INPUT * NUM_CLASSES)

#define MAX_ELEMENTS         (POOL1_SIZE + DENSE_LAYER_INPUT) /**< Workspace size. */

#define INPUT_SCALE          256.0f /**< Float feature to int16 scale. */

#define CONV1_WEIGHT_SCALE   64371.8982661337f
#define CONV1_BIAS_SCALE     98602.9378561312f

#define CONV2_WEIGHT_SCALE   81078.8662259782f
#define CONV2_BIAS_SCALE     110958.3382079614f

#define FC_WEIGHT_SCALE      119116.5152133079f
#define FC_BIAS_SCALE        1106641.0844232002f

#endif
