/**
 * @file adc.h
 * @brief ADC, ePWM trigger and DMA setup for audio capture.
 */
#ifndef FIRMWARE_INCLUDE_ADC_H_
#define FIRMWARE_INCLUDE_ADC_H_

#include "F28x_Project.h"

#define RESULTS_BUFFER_SIZE 8448 /**< Samples per buffer (multiple of 16). */

extern Uint16 adcData0[RESULTS_BUFFER_SIZE]; /**< DMA buffer, first half. */
extern Uint16 adcData1[RESULTS_BUFFER_SIZE]; /**< DMA buffer, second half. */
extern volatile Uint16 done;                 /**< Set when a capture is complete. */

/** @brief ADCA1 interrupt handler. */
__interrupt void adca1_isr(void);
/** @brief DMA channel 1 interrupt handler. */
__interrupt void dmach1_isr(void);
/** @brief Configures ePWM to trigger ADC conversions. */
void ConfigureEPWM(void);
/** @brief Powers up and configures ADCA. */
void ConfigureADC(void);
/**
 * @brief Sets up continuous conversion on one ADC channel.
 * @param adcRegs ADC register block.
 * @param channel Input channel.
 */
void SetupADCContinuous(volatile struct ADC_REGS * adcRegs, Uint16 channel);
/** @brief Configures DMA to move ADC results into the buffers. */
void DMAInit(void);

#endif
