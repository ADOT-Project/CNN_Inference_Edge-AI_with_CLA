/**
 * @file cla_dense_layer_shared.h
 * @brief Data and tasks shared between the C28x CPU and the CLA.
 *
 * @copyright
 * Copyright (C) 2013-2024 Texas Instruments Incorporated - http://www.ti.com/
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the
 *   distribution.
 *
 *   Neither the name of Texas Instruments Incorporated nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef _CLA_SQRT_SHARED_H_
#define _CLA_SQRT_SHARED_H_

#include "F2837xD_Cla_defines.h"
#include <stdint.h>
#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

extern uint16_t cla_count_param;                  /**< Task input count. */
extern uint16_t cla_offset_param;                 /**< Task input offset. */
extern float fResult[NUM_CLASSES];                /**< Output logits. */
extern const int16_t weight [DENSE_LAYER_WEIGHTS]; /**< Dense weights (CLA copy). */
extern const int16_t fc_bias [NUM_CLASSES];       /**< Dense bias. */
extern float weight_1 [200];
extern float weight_2 [200];

extern int16_t input_image[];        /**< Network input (MFCC features). */
extern const int16_t conv1_weights[];
extern const int16_t conv1_bias[];
extern const int16_t conv2_weights[];
extern const int16_t conv2_bias[];
extern int16_t workspace[];          /**< Pool1/pool2 scratch buffer. */

/** @brief CLA task 1: the CLA stages, or only conv1+pool1 when split. */
__interrupt void Cla1Task1();
#include "cla_pipeline_config.h"
#if !CLA_SINGLE_TASK && CLA_STAGE_CONV2_POOL2
/** @brief CLA task 2: conv2+pool2. */
__interrupt void Cla1Task2();
#endif
#if !CLA_SINGLE_TASK && CLA_STAGE_DENSE
/** @brief CLA task 3: dense layer. */
__interrupt void Cla1Task3();
#endif

#ifdef __cplusplus
}
#endif

#endif
