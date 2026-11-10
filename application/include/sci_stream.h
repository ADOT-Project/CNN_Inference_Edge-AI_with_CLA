/**
 * @file sci_stream.h
 * @brief Test mode: ADC windows arrive over SCIA instead of the ADC, and the
 *        logits go back, so a PC can score the whole test set on the board.
 *
 * Set TEST_SCI_STREAM to 1 here (or as a predefined symbol) to build it; at 0
 * nothing of it is compiled and the firmware is unchanged. On the LaunchPad,
 * SCIA (GPIO42 TX, GPIO43 RX) is the XDS100's virtual COM port, so the USB
 * cable that programs the board also carries the data.
 *
 * Protocol (all fields little-endian, 8-bit bytes):
 *   board -> PC  READY   C3 3C 'R'  u16 FEAT_SLICE_LEN  u16 NUM_CLASSES
 *   PC -> board  WINDOW  A5 5A      u16 index  u16 label  FEAT_SLICE_LEN x u16 code
 *   board -> PC  RESULT  C3 3C 'D'  u16 index  u16 predicted  u16 code sum
 *                                   NUM_CLASSES x f32 logit (fResult, x INPUT_SCALE)
 * The board sends READY whenever it waits for a window; the code sum lets the
 * PC see a lost or corrupted byte.
 */
#ifndef SCI_STREAM_H
#define SCI_STREAM_H

#include <stdint.h>

#ifndef TEST_SCI_STREAM
#define TEST_SCI_STREAM 1 /**< 1: windows come over SCIA, not the ADC. */
#endif

/** BRR = LSPCLK / (8 * baud) - 1, LSPCLK = 50 MHz (SYSCLK / 4):
 *  26 -> 231 481 baud (+0.5%, default), 53 -> 115 741 baud (+0.5%). */
#ifndef SCI_STREAM_BRR
#define SCI_STREAM_BRR 26U
#endif

#if TEST_SCI_STREAM
/** @brief SCIA on GPIO42/43, 8N1, FIFOs on. Call once before the loop. */
void sci_stream_init(void);

/**
 * @brief Sends READY, then waits for one window.
 * @param adc Receives FEAT_SLICE_LEN ADC codes.
 * @return The label the PC sent with the window.
 */
uint16_t sci_stream_receive(unsigned int *adc);

/**
 * @brief Sends the result of the window last received.
 * @param logits NUM_CLASSES logits, as in fResult.
 */
void sci_stream_send_result(const float *logits);
#endif

#endif /* SCI_STREAM_H */
