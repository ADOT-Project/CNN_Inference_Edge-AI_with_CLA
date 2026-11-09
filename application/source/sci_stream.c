/**
 * @file sci_stream.c
 * @brief SCIA transport for the TEST_SCI_STREAM mode (see sci_stream.h).
 * @note Polled, blocking: while a window is received the CPU does nothing
 *       else, so no byte is lost to an interrupt that is late.
 */
#include "F28x_Project.h"
#include "definitions.h"
#include "feature_tables.h"
#include "sci_stream.h"

#if TEST_SCI_STREAM

static uint16_t last_index = 0; /**< Index of the window last received. */
static uint16_t last_sum = 0;   /**< Sum of its codes, modulo 2^16. */

/** @brief Clears an overrun/framing error by resetting the SCI state. */
static void sci_recover(void)
{
    SciaRegs.SCICTL1.bit.SWRESET = 0;
    SciaRegs.SCICTL1.bit.SWRESET = 1;
    SciaRegs.SCIFFRX.bit.RXFIFORESET = 0;
    SciaRegs.SCIFFRX.bit.RXFIFORESET = 1;
}

/** @brief Blocks until one byte has arrived. */
static uint16_t sci_get(void)
{
    for (;;) {
        if (SciaRegs.SCIRXST.bit.RXERROR) {
            sci_recover();
        }
        if (SciaRegs.SCIFFRX.bit.RXFFST != 0) {
            return SciaRegs.SCIRXBUF.all & 0x00FFU;
        }
    }
}

/** @brief Blocks until there is room in the TX FIFO, then queues one byte. */
static void sci_put(uint16_t b)
{
    while (SciaRegs.SCIFFTX.bit.TXFFST >= 15U) {}
    SciaRegs.SCITXBUF.all = b & 0x00FFU;
}

static uint16_t sci_get16(void)
{
    uint16_t lo = sci_get();
    return lo | (sci_get() << 8);
}

static void sci_put16(uint16_t v)
{
    sci_put(v);
    sci_put(v >> 8);
}

void sci_stream_init(void)
{
    GPIO_SetupPinMux(43, GPIO_MUX_CPU1, 15);
    GPIO_SetupPinOptions(43, GPIO_INPUT, GPIO_PUSHPULL);
    GPIO_SetupPinMux(42, GPIO_MUX_CPU1, 15);
    GPIO_SetupPinOptions(42, GPIO_OUTPUT, GPIO_ASYNC);

    SciaRegs.SCICCR.all = 0x0007;    /* 1 stop bit, no parity, 8 bits */
    SciaRegs.SCICTL1.all = 0x0003;   /* TX, RX on, held in reset */
    SciaRegs.SCICTL2.all = 0x0000;   /* no interrupts: polled */
    SciaRegs.SCIHBAUD.all = (SCI_STREAM_BRR >> 8) & 0x00FFU;
    SciaRegs.SCILBAUD.all = SCI_STREAM_BRR & 0x00FFU;
    SciaRegs.SCIFFTX.all = 0xE040;   /* FIFO on, TX FIFO reset released */
    SciaRegs.SCIFFRX.all = 0x2044;
    SciaRegs.SCIFFCT.all = 0x0000;
    SciaRegs.SCICTL1.all = 0x0023;   /* out of reset */
    SciaRegs.SCIFFTX.bit.TXFIFORESET = 1;
    SciaRegs.SCIFFRX.bit.RXFIFORESET = 1;
}

uint16_t sci_stream_receive(unsigned int *adc)
{
    uint16_t label, n, prev = 0, b;

    sci_put(0xC3); sci_put(0x3C); sci_put('R');
    sci_put16(FEAT_SLICE_LEN);
    sci_put16(NUM_CLASSES);

    for (;;) {                       /* sync on A5 5A */
        b = sci_get();
        if (prev == 0xA5U && b == 0x5AU) break;
        prev = b;
    }
    last_index = sci_get16();
    label = sci_get16();
    last_sum = 0;
    for (n = 0; n < FEAT_SLICE_LEN; n++) {
        uint16_t code = sci_get16();
        adc[n] = code;
        last_sum += code;
    }
    return label;
}

void sci_stream_send_result(const float *logits)
{
    uint16_t c, best = 0;
    for (c = 1; c < NUM_CLASSES; c++) {
        if (logits[c] > logits[best]) best = c;
    }
    sci_put(0xC3); sci_put(0x3C); sci_put('D');
    sci_put16(last_index);
    sci_put16(best);
    sci_put16(last_sum);
    for (c = 0; c < NUM_CLASSES; c++) {
        union { float f; uint32_t u; } v;   /* IEEE-754, low word first */
        v.f = logits[c];
        sci_put16((uint16_t)(v.u & 0xFFFFUL));
        sci_put16((uint16_t)(v.u >> 16));
    }
    while (SciaRegs.SCIFFTX.bit.TXFFST != 0U) {}
}

#endif /* TEST_SCI_STREAM */
