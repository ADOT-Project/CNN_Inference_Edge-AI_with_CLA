/**
 * @file main.c
 * @brief CPU1 entry: MFCC on the CPU, CNN on the CLA/CPU split.
 * @note ADOT export updates weight[] only; comments are kept.
 */
#include "F28x_Project.h"
#include "cla_dense_layer_shared.h"
#include "cla_pipeline_config.h"
#include "weights.h"
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "convolution.h"
#include "definitions.h"
#include "feature_tables.h"
#include "test_inputs.h"
#if TEST_N_CLASSES != NUM_CLASSES
#error "test_inputs.h belongs to another model (TEST_N_CLASSES != NUM_CLASSES) -- export again with ADOT"
#endif
#if (TEST_INPUT_H != IMAGE_H) || (TEST_INPUT_W != IMAGE_W)
#error "test_inputs.h belongs to another input size -- export again with ADOT"
#endif
#if (FEAT_N_FRAMES != IMAGE_H) || (FEAT_N_OUT != IMAGE_W)
#error "feature_tables.h does not match definitions.h: MFCC output size differs from the network input"
#endif
#if TEST_GOLDEN_ADC == 2
  #include "golden_vectors.h"
  #if GOLDEN_SLICE_LEN != FEAT_SLICE_LEN
  #error "golden_vectors.h belongs to another window length -- export again with ADOT"
  #endif
  #if GOLDEN_N != NUM_CLASSES
  #error "golden_vectors.h belongs to another model (GOLDEN_N != NUM_CLASSES) -- export again with ADOT"
  #endif
#elif TEST_GOLDEN_ADC
  #if NUM_CLASSES == 2
  #include "golden_vector_normal.h"
  #include "golden_vector_abnormal.h"
  #elif NUM_CLASSES == 3
  #include "golden_vector_healthy.h"
  #include "golden_vector_bearing.h"
  #include "golden_vector_propeller.h"
  #else
  #error "TEST_GOLDEN_ADC 1 but no golden vector set for this class count (2 = MIMII, 3 = BLDC)"
  #endif
#endif
#include "adc.h"
#include "sci_stream.h"

#define INPUT_SIZE 2704 /**< Unused legacy size. */

#ifdef __cplusplus
#pragma DATA_SECTION("CpuToCla1MsgRAM");
float fVal;
#pragma DATA_SECTION("Cla1ToCpuMsgRAM");
float fResult;
#else
#pragma DATA_SECTION(cla_count_param,"CpuToCla1MsgRAM");
uint16_t cla_count_param;

#pragma DATA_SECTION(cla_offset_param,"CpuToCla1MsgRAM");
uint16_t cla_offset_param;

#pragma DATA_SECTION(fResult,"CLADataLS0");
float fResult[NUM_CLASSES]; /**< Output logits. */

#pragma DATA_SECTION(workspace, "CLADataLS0");
#pragma DATA_ALIGN(workspace, 8)
 int16_t workspace[MAX_ELEMENTS] = {0.0f}; /**< Pool1 + pool2 outputs. */

/** @brief Dense weights read by the CLA (written by ADOT export). */
#if CLA_STAGE_DENSE
#pragma DATA_SECTION(weight,"CLADataLS1");
#endif
const int16_t weight [DENSE_LAYER_WEIGHTS] = {
    7133, 1790, -4055, 1772, 1803, 13139, -841, -7589, 17168, 9375,
    -3940, 11785, 12948, -1060, -5323, -1724, 4698, -1061, 703, -15323,
    11096, 10879, 2911, 14293, 11348, -3321, -7359, 6283, 1616, 4789,
    -5896, -9545, 21936, 17919, 2566, 18088, 2560, -10136, -6366, -4081,
    -5611, 3108, 1386, -2097, 21047, 9180, 3727, 13200, 4167, -8639,
    -5647, -11964, -7141, -3632, -226, -2619, 6460, 7055, 4230, 6083,
    8800, -3101, -9219, -10129, -16119, -5540, -12308, -2872, 10389, 6155,
    -5908, 5467, 3265, -9719, -5407, -7621, 355, 4486, 4556, -12617,
    10412, 9820, -2954, 10263, 9123, -2887, -8244, -4731, -8657, 6437,
    6096, -10009, 11908, 4721, -4573, 7930, 3420, -7727, -13568, 547,
    -1056, 5574, 2827, -15649, 8707, 11862, 1258, 12885, -8976, 116,
    -10417, -10819, -1159, 7177, 13666, -13829, -25160, 5390, 2926, -2856,
    -7391, 6630, 30, -263, 5599, 8108, 8608, -9340, -18120, 3593,
    5186, -4243, -5975, 4716, -10786, -10613, 1531, 9890, 15918, -20462,
    -19479, 10678, 13544, -4570, -3264, -2916, -16810, -8948, -10549, -2556,
    9988, -7163, -29081, -223, 51, -111, 1495, 3825, -11706, -19420,
    -14100, -455, 13542, -9594, -20919, 6569, -2826, -5151, -1990, 8526,
    -9581, -18671, -19763, -1655, 3698, -16823, -30617, 8194, 5403, -5354,
    708, 52, -10748, -10961, -11333, -8623, 5363, 2028, -19238, -11008,
    -1488, -5020, 283, 2008, -4243, -11072, -5699, -3006, 1187, -829,
    -19011, -8558, -6877, 617, 1797, 8601, -2328, -15284, -1081, -3083,
    7182, 4080, -12287, -4242, -1685, -3191, 13880, 204, -10941, -9348,
    -18920, 3941, 8324, -18778, 13578, 3892, -5160, 6646, 7997, 3146,
    -9378, -14444, -14766, -2008, 6339, -9171, 7062, 2877, -3747, 6025,
    12941, 1978, -11290, -8958, -18721, -1483, 6611, -12358, -915, 9828,
    -12999, 11244, 4536, -3664, -11045, -2886, 11081, 12454, -13685, -18137,
    8303, 9130, -135, -4730, 8142, -3654, -4815, 4585, 7523, 7951,
    -7534, -18069, 4924, 4062, -3391, -6150, 3680, -10681, -6052, -3616,
    2825, 4464, -17511, -14602, 1283, 5467, -2914, 4395, 7796, -4383,
    11302, 9470, 13508, 127, -29986, 6456, 21962, 8307, -2114, 8150,
    898, 2121, 10455, 10019, 16038, -9070, -24794, -2658, 17809, 462,
    -2350, 3942, 9924, -6232, 10751, 9881, 13175, -7625, -28073, 5601,
    21239, 7897, -1968, 13269, 8270, 6896, 8721, -1484, -4453, -10882,
    2831, 9375, -13630, 4726, -1112, -8387, 7667, 3423, -1240, 3413,
    -157, -1343, 612, 9773, -8071, 10171, -233, -7314, 11911, -5678,
    2712, 4141, -4511, -10023, 5081, 8758, -9949, 557, -127, -2222,
    -16248, -20040, -2699, -6963, -21063, -12308, -5378, -12997, -2963, 6644,
    3621, -937, -16309, -15057, 712, 4744, -10164, -6622, -7779, -1982,
    -3371, 2480, -470, -1409, -15737, -9971, -5742, -1097, -15060, -9891,
    -8419, -9655, 231, 4272, -905, -3183, 9079, 6918, 12431, 2049,
    -1485, -5926, -438, 16240, -8736, 3022, -5341, -7190, 10070, 8221,
    5026, -2756, 4483, -7164, -324, 5372, -5758, 1241, -3720, -998,
    15245, 12991, 14055, -1366, 2385, -4875, 1392, 6090, -1544, -4459,
    -14, -3716, 11849, 13157, 9235, 9959, -984, -8987, -3606, -7746,
    -2773, -13065, -4307, 952, 4043, 8040, 8294, 11408, 2398, -5253,
    -4213, -10352, -2619, -16541, -4683, 592, 15748, 13759, 7234, 9852,
    -132, -9185, -6637, -11138, 832, -13812, -4069, -5329, 148, 9558,
    5853, 8394, 8932, 5602, -10359, -8683, -393, -10515, -3987, -7353,
    2874, 8725, 6970, 7002, -985, 4223, -11416, -10419, -4055, -17134,
    -12377, -4943, 6295, 12836, 11932, 13410, 8097, 5773, -13181, -10592,
    -4887, -16477, -10812, -8949, 4863, 4564, 7182, 7327, 2329, 2132,
    -7483, -438, 8294, -2746, -7229, -5959, 611, 2893, 2545, 10866,
    -101, 2240, -6434, -2026, 3707, -7996, -6350, -2625, 3667, 9535,
    5863, 11427, 4749, 4591, 300, -3542, 6229, -2284, -4665, 53,
    2522, -20789, -1563, -2777, -6903, -16646, -21440, 17535, -5246, -2092,
    -1050, -3456, 3455, -21218, 4827, 7660, -830, -13280, -22219, 9568,
    -7737, 1800, -3132, -8563, 2650, -27541, 10314, 4388, -3331, -21762,
    -17472, 16043, -9117, 208, 4029, -336, 14436, 10825, 7636, 7196,
    -421, -8328, 2643, -521, -7314, -6000, -3443, -2706, 9538, 6585,
    1654, 5522, -2696, -3080, -3659, -4915, -9750, -6920, -582, -4658,
    13120, 8505, 4798, 5907, 3715, -8103, 1746, 3102, -4117, -6779,
    -6861, 398, 10523, 10604, -1526, 3110, -12368, 7699, 16658, -12202,
    418, 5606, -6602, -2547, 4065, 4462, 1578, 648, -10837, 8070,
    16842, -16666, 6688, 6368, -2147, -5363, 12483, 10031, 3887, -5519,
    -15958, 6352, 18030, -11086, 2054, 9629, 2521, -9416, -696, 1812,
    5155, -1602, -2763, 834, 2592, 9296, 246, -11337, -11164, -7765,
    -3431, 605, 5109, -2773, -1001, 2575, 4216, 6733, 1616, -1833,
    -10203, -3527, 1906, 4109, 5860, 1296, 1761, 1589, -2345, 8192,
    8558, -6786, -6314, -5599, 1106, 7135, 1238, -768, -566, -1827,
    7458, 9253, -2938, -13268, -5440, -6836, 40, -404, 3595, 3330,
    832, 8479, 3454, 5136, 1322, -11782, -2612, -8076, -30, -723,
    -108, 4029, 9651, 8279, 3281, 7716, 2470, -4157, -7091, -2975,
    -2126, -4486, 4294, 1131, 4475, 389, 6835, -2401, 6734, -1888,
    -9162, -6087, 2383, 5146, -4730, -5235, -1805, 2544, -602, -1150,
    2051, -1833, -6652, -5180, 2317, 3021, -5378, 2429, 3446, -410,
    -5054, 7854, 5956, -6200, -7034, -7722, -8169, -11864, -12543, -5984,
    -7914, -8306, -16016, -694, 2074, 2775, -1701, -3123, -5652, -6237,
    -3234, 2538, -9550, -15441, -19335, 2163, 7356, 11427, -9747, -4269,
    -14148, -9318, -9133, -1091, -5200, -9076, -16998, -790, 4332, 4101,
    -6643, -6, -9133, -14618, -12854, -8025, -7303, -7762, -17554, -8206,
    879, 6376, -7776, -8594, -5063, -16033, -3342, 593, -3453, -3241,
    -14191, -12239, 4522, 2374, -6664, -1663, -15641, -4960, -5793, -4450,
    -9310, -7966, -18820, -6901, 2115, 930, -7877, -3658, -4151, -3030,
    -3922, -8020, -1478, -3972, -5275, -7759, -3702, 7599, 3014, -874,
    -4123, -11207, -4686, -1529, -4048, -10883, -9937, -9042, -5719, 3558,
    4456, -8003, -6769, -11983, -7581, -4990, -2172, -3183, -7197, -12808,
    -8476, 3844, -6296, -4097, -2477, 236, 1298, 6247, -335, 9439,
    -6620, -10063, 3789, -10852, -925, -5415, -1542, -3882, -3219, 7330,
    10700, 5558, -4997, -727, 1714, -9761, -4548, -11911, -3287, -188,
    6997, 2180, 6703, 2003, -10110, -2449, 9016, -8605, -9185, -9473,
    -11878, -5876, -9659, -12515, -12980, -8391, -1293, 281, 6421, 2349,
    -6118, 738, -14072, -10652, -875, -12009, -11241, -8527, -4065, 203,
    7878, -202, -13558, -2968, -11862, -5087, -6853, -4113, -10270, -4628,
    -1122, 10144, 8268, -6990, -12462, 1110, -7896, -3989, -2093, -4391,
    -3300, -12535, 2713, 10114, 1567, -13503, -4582, 684, -4685, -4107,
    -660, -3577, -11870, -13774, -17, 6945, 8714, -13140, -757, -3898,
    -1767, -2269, -7128, -9670, -14476, -5604, 6454, 12583, 4313, -3092,
    -7408, 4360, -21635, -238, -8997, -10238, -1251, -12255, -6156, -653,
    -17337, -6391, 6540, 5561, -17243, 4450, -2214, -6441, -6612, -2377,
    -7145, 1045, -17613, -12509, 3809, 958, -22567, -196, -3158, -8817,
    -3751, -4728, -10983, 5106, -21805, -12313, 7483, -397, 1005, 15578,
    5345, 4335, 9207, 10007, 4395, -3530, -15938, -3483, 10686, -3962,
    11144, 10720, 6972, 833, 7908, 6353, 5452, -860, -9339, -958,
    4801, -2260, -273, 8607, 9862, 9285, 16641, 10470, -817, -3105,
    -10488, 1622, 2371, -39, -27163, -4115, -4306, 2425, -3445, -4138,
    -823, 3152, -4034, -1249, 10860, 425, -11654, 139, -2269, -4112,
    -4362, 2729, -3036, -3375, -7814, 125, 7766, -7617, -15306, -7982,
    -8678, -1941, -4394, 3263, -5194, -4868, -8202, -3809, 14512, 5543,
    -9021, -18242, -8351, -5577, 3158, 4774, -919, 5492, 6834, -6004,
    1081, 8123, -8865, -15533, -4811, 1563, 3244, 5664, -4175, 4137,
    7241, -5658, -762, 2969, -7803, -21067, -10360, -4722, 4015, 1536,
    -1318, 2768, 4005, -4635, 6536, -301, 5956, -8857, -6911, -1418,
    9419, 2109, 5738, 11954, 8743, 1427, 6995, 4690, 4792, -7724,
    2098, -4234, 421, 1416, 3378, 13586, 7495, 466, 10908, 9802,
    3780, -11524, 1476, -3858, 6057, 9734, 4401, 11089, 11046, 4311,
    8417, 11325, 8352, -10478, -5751, -3788, 2267, 3689, 4275, 14170,
    5355, 2766, 10952, 11845, 2107, -3005, 521, -787, 3888, 2439,
    4312, 9302, 5071, 3720, 5356, 6216, 2440, -4414, 1268, 1244,
    -1569, 9769, 5674, 5547, 8791, 1730, 8480, 9697, -16594, 14495,
    -1074, 3681, 8583, 2940, 15466, 10933, -1120, 3180, 14880, 8415,
    -12439, 12576, -389, 4862, 6504, 8412, 16360, 7630, -474, -2761,
    13853, -306, -15459, 24867, 5639, -954, 12302, 952, 9248, 9353,
    -3585, 2462, 12996, -1584, -13971, -11668, -5734, 202, 6901, 12979,
    12599, 4162, -793, 4995, 6702, -1138, -6855, -6242, -5779, -5838,
    -2561, 6447, 10676, -5571, -5434, -2160, 8399, 4970, -10807, -13917,
    -8812, -5702, -842, 10232, 11227, 4985, -2956, 5276, 6870, 2861,
    -11635, -510, -10600, -6044, -3105, -6839, -2184, -1059, -31665, 1031,
    2145, -121, -13793, -2930, -7306, -7689, -188, -5002, -6416, -4684,
    -31627, -5565, 3776, 1786, -11162, -396, -8241, -3193, 7008, 2740,
    -2948, -3686, -32767, 1677, 10332, 8165
};

#pragma DATA_SECTION(adcData0, "ramgs0");
Uint16 adcData0[RESULTS_BUFFER_SIZE]; /**< ADC DMA buffer. */
volatile Uint16 done; /**< Set by the DMA ISR. */
volatile uint16_t cla_done = 0; /**< Set when inference finished. */
#endif

void CLA_configClaMemory(void);
void CLA_initCpu1Cla1(void);

static float features[FEAT_N_FRAMES][FEAT_N_OUT]; /**< MFCC output. */
__interrupt void cla1Isr1();

/** @brief Test input source, chosen by TEST_GOLDEN_ADC in test_inputs.h. */
#if TEST_SCI_STREAM
static unsigned int sci_adc[FEAT_SLICE_LEN]; /**< Window received over SCIA. */
#endif
#if TEST_GOLDEN_ADC == 2
#define TEST_CYCLE_LEN        GOLDEN_N
#define TEST_EXPECTED(i)      golden_labels[i]
#define TEST_CLASS_NAME(c)    test_class_names[c]
#define TEST_GOLDEN_SLICE(i)  ((const unsigned int*)golden_adc[i])
#elif TEST_GOLDEN_ADC
  #if NUM_CLASSES == 2
static const uint16_t *const golden_inputs[NUM_CLASSES] = {
    golden_adc_normal, golden_adc_abnormal
};
static const char *const golden_names[NUM_CLASSES] = { "Normal", "Abnormal" };
  #else
static const uint16_t *const golden_inputs[NUM_CLASSES] = {
    golden_adc_healthy, golden_adc_bearing, golden_adc_propeller
};
static const char *const golden_names[NUM_CLASSES] = { "Healthy", "Bearing", "Propeller" };
  #endif
#define TEST_CYCLE_LEN        NUM_CLASSES
#define TEST_EXPECTED(i)      (i)
#define TEST_CLASS_NAME(c)    golden_names[c]
#define TEST_GOLDEN_SLICE(i)  ((const unsigned int*)golden_inputs[i])
#else
#if !TEST_SCI_STREAM  /* one window buffer is enough: SCI mode has its own */
static unsigned int test_adc[FEAT_SLICE_LEN];
#endif
#define TEST_CYCLE_LEN        TEST_N_INPUTS
#define TEST_EXPECTED(i)      test_labels[i]
#define TEST_CLASS_NAME(c)    test_class_names[c]
#endif
volatile uint16_t g_expected_class = 0; /**< Label of the current input. */
volatile uint16_t g_predicted_class = 0; /**< Argmax of the logits. */
volatile float    g_logits[NUM_CLASSES]; /**< Copy of fResult. */
volatile uint32_t g_correct_count = 0; /**< Correct predictions. */
volatile uint32_t g_total_count = 0; /**< Total predictions. */
const char *volatile g_predicted_name = 0; /**< Predicted class name. */

void run_inference(uint16_t expected_class);

void mfcc_extract(const unsigned int* adc_buf,
                        float out[FEAT_N_FRAMES][FEAT_N_OUT]);

/**
 * @brief Sets up the device, then loops: features, inference, score.
 * @return Never returns.
 */
int main(void)
{
    Uint16 resultsIndex;

    InitSysCtrl();
    InitGpio();

    EALLOW;
    GpioCtrlRegs.GPCGMUX1.bit.GPIO67 = 0;
    GpioCtrlRegs.GPCDIR.bit.GPIO67 = 1;
    GpioDataRegs.GPCSET.bit.GPIO67 = 1;
    GpioCtrlRegs.GPCPUD.bit.GPIO67 = 0;
    GpioCtrlRegs.GPCCSEL1.bit.GPIO67 = 1;
    EDIS;

    GPIO_SetupPinOptions(18, GPIO_OUTPUT, GPIO_PUSHPULL);
    GPIO_SetupPinMux(18, GPIO_MUX_CPU1, 0);
    GPIO_SetupPinOptions(19, GPIO_OUTPUT, GPIO_PUSHPULL);
    GPIO_SetupPinMux(19, GPIO_MUX_CPU1, 0);

    DINT;
    InitPieCtrl();

    IER = 0x0000;
    IFR = 0x0000;

    InitPieVectTable();

        EALLOW;
        PieVectTable.ADCA1_INT = &adca1_isr;
        PieVectTable.DMA_CH1_INT = &dmach1_isr;
        EDIS;

        IER |= M_INT1;
        IER |= M_INT7;

        PieCtrlRegs.PIEIER1.bit.INTx1 = 1;
        PieCtrlRegs.PIEIER7.bit.INTx1 = 1;

            EALLOW;
            CpuSysRegs.PCLKCR0.bit.TBCLKSYNC = 0;
            EDIS;

            ConfigureEPWM();

            EALLOW;
            CpuSysRegs.PCLKCR0.bit.TBCLKSYNC = 1;
            EDIS;

            ConfigureADC();

            SetupADCContinuous(&AdcaRegs, 3);
            SetupADCContinuous(&AdcbRegs, 3);

            DMAInit();

    CLA_configClaMemory();
    CLA_initCpu1Cla1();

    EINT;
    ERTM;
        for(resultsIndex = 0; resultsIndex < RESULTS_BUFFER_SIZE; resultsIndex++)
        {
            adcData0[resultsIndex] = 0;

        }

        EALLOW;

        DmaRegs.CH1.CONTROL.bit.PERINTCLR = 1;
        DmaRegs.CH2.CONTROL.bit.PERINTCLR = 1;
        AdcaRegs.ADCINTFLGCLR.all = 0x3;
        AdcbRegs.ADCINTFLGCLR.all = 0x3;
        EPwm2Regs.ETCNTINITCTL.bit.SOCAINITFRC = 1;
        EPwm2Regs.ETCLR.bit.SOCA = 1;

        AdcaRegs.ADCINTSOCSEL1.bit.SOC0 = 2;
        AdcbRegs.ADCINTSOCSEL1.bit.SOC0 = 2;

        EDIS;

        done = 0;
        StartDMACH1();
        StartDMACH2();

        EPwm2Regs.ETSEL.bit.SOCAEN = 1;

    static uint16_t golden_cycle_idx = 0;

#if TEST_SCI_STREAM
    sci_stream_init();
#endif
#if !TEST_GOLDEN_ADC && !TEST_SCI_STREAM
    {
        uint32_t lcg = 12345UL;
        uint32_t n;
        for (n = 0; n < FEAT_SLICE_LEN; n++) {
            lcg = lcg * 1664525UL + 1013904223UL;
            test_adc[n] = (unsigned int)(1024U + (unsigned int)((lcg >> 20) & 0x7FFUL));
        }
    }
#endif

    for(;;)
     {
        GPIO_WritePin(18, 0);
        static uint32_t f,o;
#if TEST_SCI_STREAM
        g_expected_class = sci_stream_receive(sci_adc);   /* blocks */
        mfcc_extract(sci_adc, features);
#elif TEST_GOLDEN_ADC
        g_expected_class = TEST_EXPECTED(golden_cycle_idx);
        mfcc_extract(TEST_GOLDEN_SLICE(golden_cycle_idx), features);
#else
        g_expected_class = TEST_EXPECTED(golden_cycle_idx);
        mfcc_extract(test_adc, features);
#endif

        int idx = 0;
                for ( f = 0; f < FEAT_N_FRAMES; f++) {
                    for ( o = 0; o < FEAT_N_OUT; o++) {
                        float scaled = features[f][o] * INPUT_SCALE;
                        if (scaled > 32767.0f) scaled = 32767.0f;
                        if (scaled < -32768.0f) scaled = -32768.0f;
                        input_image[idx++] = (int16_t)scaled;
                    }
                }
        GPIO_WritePin(18, 1);

#if !TEST_GOLDEN_ADC && !TEST_SCI_STREAM
        {
            const int16_t *src = test_inputs[golden_cycle_idx];
            uint16_t n;
            for (n = 0; n < (IMAGE_H * IMAGE_W); n++) input_image[n] = src[n];
        }
#endif

        GPIO_WritePin(19, 0);

        run_inference(g_expected_class);
        GPIO_WritePin(19, 1);

#if TEST_SCI_STREAM
  #if CLA_SINGLE_TASK
        while (!cla_done) {}   /* single task: the result is not in yet */
  #endif
        sci_stream_send_result(fResult);
#endif

        golden_cycle_idx++;
        if (golden_cycle_idx >= TEST_CYCLE_LEN) golden_cycle_idx = 0;

     }

}

/**
 * @brief Stores the logits and updates the score.
 * @param expected_class Label of the input.
 */
static void record_result(uint16_t expected_class)
{
    uint16_t c, best = 0;
    for (c = 0; c < NUM_CLASSES; c++) {
        g_logits[c] = fResult[c];
        if (fResult[c] > fResult[best]) best = c;
    }
    g_predicted_class = best;
    g_predicted_name = TEST_CLASS_NAME(best);
    g_total_count++;
    if (best == expected_class) g_correct_count++;
}

/**
 * @brief Runs the three stages on the CLA or CPU per cla_pipeline_config.h.
 * @param expected_class Label used for scoring.
 * @note In single-task mode the result is scored on the next call.
 */
void run_inference(uint16_t expected_class)
{
#if CLA_SINGLE_TASK
    static uint16_t pending = 0, pending_class = 0;
    if (pending && cla_done) {
        record_result(pending_class);
    }
    cla_done = 0;
    Cla1ForceTask1();
    pending = 1;
    pending_class = expected_class;
#else
    cla_done = 0;

  #if CLA_STAGE_CONV1_POOL1
    Cla1ForceTask1andWait();
  #else
    conv1_pool_fused_cpu(input_image, workspace, conv1_weights, conv1_bias);
  #endif

  #if CLA_STAGE_CONV2_POOL2
    Cla1ForceTask2andWait();
  #else
    conv2_pool_fused_cpu(workspace, &workspace[POOL1_SIZE], conv2_weights, conv2_bias);
  #endif

  #if CLA_STAGE_DENSE
    Cla1ForceTask3andWait();
  #else
    dense_fc_cpu(&workspace[POOL1_SIZE], weight, fc_bias, fResult);
  #endif

    cla_done = 1;
    record_result(expected_class);
#endif
}
/** @brief Copies CLA code/data to RAM and assigns LS RAM to the CLA. */
void CLA_configClaMemory(void)
{
    extern uint32_t Cla1funcsRunStart, Cla1funcsLoadStart, Cla1funcsLoadSize;
    extern uint32_t Cla1DataRunStart, Cla1DataLoadStart, Cla1DataLoadSize;
    EALLOW;

#ifdef _FLASH
    memcpy((uint32_t *)&Cla1funcsRunStart, (uint32_t *)&Cla1funcsLoadStart,
           (uint32_t)&Cla1funcsLoadSize);

    memcpy((uint32_t *)&Cla1DataRunStart, (uint32_t *)&Cla1DataLoadStart,
           (uint32_t)&Cla1DataLoadSize);
#endif

    MemCfgRegs.MSGxINIT.bit.INIT_CLA1TOCPU = 1;
    while(MemCfgRegs.MSGxINITDONE.bit.INITDONE_CLA1TOCPU != 1){};

    MemCfgRegs.MSGxINIT.bit.INIT_CPUTOCLA1 = 1;
    while(MemCfgRegs.MSGxINITDONE.bit.INITDONE_CPUTOCLA1 != 1){};

    MemCfgRegs.LSxMSEL.bit.MSEL_LS5 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS5 = 1;

    MemCfgRegs.LSxMSEL.bit.MSEL_LS0 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS0 = 0;

    MemCfgRegs.LSxMSEL.bit.MSEL_LS1 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS1 = 0;

    MemCfgRegs.LSxMSEL.bit.MSEL_LS2 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS2 = 0;

    MemCfgRegs.LSxMSEL.bit.MSEL_LS3 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS3 = 0;

    MemCfgRegs.LSxMSEL.bit.MSEL_LS4 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS4 = 0;
    EDIS;
}

/** @brief Sets the CLA task vectors and enables the CLA interrupt. */
void CLA_initCpu1Cla1(void)
{
    EALLOW;
#if CLA_SINGLE_TASK
    Cla1Regs.MVECT1 = (uint16_t)(&Cla1Task1);
#else
  #if CLA_STAGE_CONV1_POOL1
    Cla1Regs.MVECT1 = (uint16_t)(&Cla1Task1);
  #endif
  #if CLA_STAGE_CONV2_POOL2
    Cla1Regs.MVECT2 = (uint16_t)(&Cla1Task2);
  #endif
  #if CLA_STAGE_DENSE
    Cla1Regs.MVECT3 = (uint16_t)(&Cla1Task3);
  #endif
#endif

    Cla1Regs.MCTL.bit.IACKE = 1;
    Cla1Regs.MIER.all = 0x00FF;

    PieVectTable.CLA1_1_INT  = &cla1Isr1;

    PieCtrlRegs.PIEIER11.bit.INTx1 = 1;
    IER |= (M_INT11 );
    EDIS;
}

/** @brief Marks inference done and acknowledges the interrupt. */
__interrupt void cla1Isr1 ()
{
    cla_done = 1;

    PieCtrlRegs.PIEACK.all = M_INT11;

}
