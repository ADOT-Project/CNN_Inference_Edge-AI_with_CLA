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
    -11523, -10630, -8988, 1004, -5138, -14528, -1529, 19115, -5305, 1320,
    557, 3492, 3780, 11016, 377, 9787, -12915, -6688, 7387, 6594,
    7342, -7223, -2911, 6184, -6009, -7177, 204, -4722, -4898, -1483,
    3268, 323, -4775, -14952, -4061, -10721, -2304, -2702, 7295, 8033,
    -8895, -20647, -329, 8250, -2774, -8516, 8118, 13188, -4433, -2459,
    910, 5997, -10166, 397, 2205, 16328, -12477, -17581, -4796, 6286,
    8573, 690, 10516, 8696, -9865, -10067, 373, 6377, 2077, -8253,
    -9635, 14417, -10417, -890, -3008, 10590, -12887, -2684, 6442, 11375,
    -15755, -3445, 2227, -17, 87, 253, 2220, -3702, -1776, -8670,
    -6314, -9047, 2848, -3954, -4566, -3140, -2852, -14603, 669, -1151,
    -7000, 518, 4423, 18310, -15108, -12482, -558, 8129, 1433, -8472,
    9026, 15724, 289, 4372, 8819, 15974, 4182, 3474, -2299, 19355,
    -2300, -5981, 10659, 15124, -169, 8122, 13515, 21207, 514, -2114,
    6006, 7508, -4182, 1885, 2136, 2873, -3871, 4386, 4413, 14418,
    5036, -1039, 5793, 15107, -1316, 2421, -9177, 10204, -1463, -586,
    3160, 13565, 2826, -733, 673, 5262, 9803, -12083, 648, 14894,
    6448, 1177, 5496, 10419, 10991, -8545, 3016, 9107, 4270, 3562,
    3539, 3357, 4819, 1867, 3957, 7320, 757, -2460, 1719, 8297,
    4718, 2263, 892, 14284, -738, 3354, 4390, 8819, 5496, -2261,
    7158, 29215, -3068, -6473, -3183, 8014, -6177, 4323, 3380, 20169,
    -6977, -5262, 3646, 16658, 2910, -2202, -1298, 15369, 3053, -674,
    -6112, 15742, 2308, -2175, 7186, 13512, -1017, 87, -5538, 11619,
    5382, -7095, -2493, 15721, 4613, 3591, -22515, 8473, -2036, 14093,
    -915, -1954, 7908, 3608, -22410, 1584, 235, 4731, -13620, -1376,
    206, -4114, -19577, 476, -3945, 4456, -13558, -3277, -873, 1514,
    -17637, 5569, -4371, 3203, -9411, -2234, 2665, 8085, -19529, 1730,
    -8280, 1392, -15201, -840, 2263, -1103, -11919, 17509, -3442, -3474,
    -9083, -4275, 11433, -1664, -10645, 2238, -4152, 9439, -7326, 1210,
    5428, 7052, -9649, 9388, -9230, 4452, -5916, -2208, 1862, 3593,
    -19382, 6096, -2401, 12629, -3862, 36, 163, -475, -18381, 9298,
    3367, 5655, -2670, -12383, -1106, -1553, -18037, 9022, -1738, 5316,
    -12151, -612, -4787, -965, -22846, 8300, -157, 13151, -10153, -1270,
    1064, 4890, -16090, 8582, -8891, 2716, -13043, -1835, 952, 3398,
    -14871, 14151, -234, 1440, -5725, -1055, -4575, -8671, -8245, -1633,
    16197, -9421, 4407, -2309, -6604, 1656, -7674, 5943, 15901, -820,
    3978, -5716, -4192, 2548, -9265, -11927, 14256, 532, 932, -2369,
    -5890, 4078, -7792, -9001, 8603, -5817, 5261, -11814, -3211, -371,
    -10708, 2700, 14763, -5589, 1236, -8861, -7548, -13305, -3090, 1694,
    21765, -5725, 2313, -977, -4060, -2758, -948, 473, 11825, -3192,
    7165, -3941, -1250, -6992, -9529, -7943, 14259, -5474, 7630, -1370,
    -6957, -4905, -6173, -6877, 15697, -103, 1062, 6183, -4640, 514,
    -8216, -7778, 10101, -5304, -1583, -6762, -3897, -443, -5681, -8523,
    18524, 4689, 434, -5785, -5488, 11134, -2496, -6622, 7421, -279,
    3761, -7957, -3789, -5951, -9041, 1022, 12919, -8439, 2509, 808,
    -4410, -11677, -3640, 503, 23578, -1748, 9333, 708, 14, -64,
    -20780, -15826, 13459, -6768, 20955, -3835, 1228, 11587, -22698, -3902,
    8425, -15328, 6233, 5273, 806, 13414, -8481, -18967, 15284, -1923,
    8014, 8144, 2119, 11321, -15028, -10408, 7894, -4835, 15823, 14878,
    8721, 8654, -32767, -15110, -555, 3380, 3114, -3712, 11722, -1924,
    -18947, -12473, 18407, -11355, 11064, -8617, 2677, 9857, -16706, -11266,
    12063, 9434, 26265, 3586, 3788, 1799, -13212, -5628, 11004, -810,
    10617, 4834, -2713, 4611, -16545, -16585, 14010, -3818, 5695, 1333,
    -4192, 3251, -23313, -4108, 8524, -10341, 10647, -14486, 12796, 6949,
    -14720, -20607, 15109, 4261, 10022, 803, 4935, 14526, -15918, -6425,
    24054, -2519, 28047, -12160, 10379, 16720, -15260, -8182, 8984, -147,
    14532, 2195, 11536, 7575, -24824, -1378, 15489, -11204, 14575, -5079,
    -9065, -28338, -4509, -15567, -838, 2453, -4966, 4764, -2868, -21991,
    -10212, 2663, 10289, 1809, -213, 5913, 1987, -29221, 974, 1468,
    -3224, 4830, 1959, 2822, 5636, -4162, 4870, 2353, 7163, 4269,
    -739, -5904, -14634, -20418, 3381, -3044, -1731, 1981, 3151, 2539,
    -2585, -24929, 9220, 1490, 706, -107, -7685, 12444, -1093, -18404,
    5912, -22021, 4696, 2292, 8531, 701, -415, -20935, 1514, -8534,
    8578, 2166, 3280, 14725, -8807, -24999, -1744, -18584, 6362, 1981,
    -2771, 6869, 6503, -20013, -9500, -4792, 6623, 3700, -9514, 7548,
    4139, -30853, 8858, 3317, 2830, -1187, -1226, 3175, -3299, -18379,
    3002, -7003, 2068, 5571, -317, 1038, -8200, -20942, -1340, 4143,
    4049, 1789, 6516, -1918, -7917, -26183, 25212, -1754, 8017, -1884,
    4397, 9693, -7000, -9515, 6469, 1405, 26930, 4885, 3568, 20049,
    -4123, 3634, -10040, 2650, -1333, -8017, 10374, 17948, -5997, -253,
    3763, 1256, 6461, 2396, 7199, 22517, -3724, -957, -4108, 2462,
    -3809, 690, 7129, 30771, -6524, 9479, 2328, 5629, 7881, 764,
    634, 9087, -3174, 3021, 10458, 1608, 8351, -6675, 19596, 9945,
    -11227, -8830, -442, 637, 7087, -1543, 5337, 27307, -6853, -14776,
    296, 4372, 11435, 3178, 1317, 19183, -7078, -3329, 4076, 3925,
    20192, 7229, -201, 25026, -4844, 835, -2061, -15102, 12550, -5325,
    6730, 22214, -2509, 345, 1665, 7958, -4636, -14548, -1361, 28973,
    -4728, 4713, -6008, -12000, 4161, 4763, 1669, 22958, -11584, 4396,
    -483, 5622, -4903, -618, 5577, 9127, -7171, -5560, 7447, 7501,
    7699, 4442, 4551, -3667, -2670, -26211, 7047, 2878, -6980, -115,
    562, -21262, -2242, -3269, 2357, -5749, 194, -3146, -11908, -8541,
    3102, -17060, 17630, 3557, -5675, 2969, 3109, -8716, 4030, -24161,
    3979, -1293, -1224, -3653, -8250, -18351, -6695, -19238, 15058, 5493,
    375, 10542, 7535, -11904, 3333, -11769, 8365, 4270, -5216, 2652,
    -452, -4977, 2465, -12477, 12316, -8551, -4910, -4741, -4237, -11813,
    -1011, -9546, 10712, 8338, 1376, -9686, 2725, -11278, -6319, -16078,
    13090, 7152, 144, -2373, 6280, -10827, 9391, -7798, 9455, -4893,
    -4207, -4537, -4090, -12291, 11286, -8555, 17444, -1516, 3989, 175,
    1256, -8170, -251, -26597, 2428, 101, -2198, 373, -12724, -4655,
    3043, -19253, 17532, 4477, -1604, 9617, 6688, -17403, -4905, -25847,
    13158, -5141, 5126, -8356, 3998, -18815, -22151, -11716, -11215, -4912,
    21786, 12505, -11087, 13238, -20012, 1958, -1432, -5502, 808, -6003,
    -12421, 5778, -26637, 6309, 429, -5313, 5005, 1787, -9187, 8442,
    -20337, -2884, -14098, -593, -840, -10580, 3722, 1882, -5887, 9430,
    8603, 516, 11438, 201, 17510, -11429, -13321, -7785, -2329, -1683,
    10904, -1639, 6600, 6743, -7942, 1967, -8053, -4579, 4344, 537,
    15205, 1551, -19483, -3267, -5246, 4029, -634, 600, 9402, 6008,
    -19110, -13699, -5668, 322, 16427, 2966, 2981, -354, -29174, -6151,
    3523, -2529, 10195, -2171, -7448, 8824, -15811, 4456, 1780, -2570,
    8548, 6862, -7393, -8479, -19491, -336, -8489, -5555, -4413, -6409,
    -1833, 6096, -4764, 5419, 5790, 383, 6576, 4308, -5593, -10055,
    -20354, -7453, -127, -12300, 4452, -7943, 12463, 6638, -5816, -9404,
    23227, 1798, 11906, -1442, 12868, 18741, -4316, 8277, 14421, 18916,
    -1845, -11913, -2731, 6364, -4540, 4780, 12154, 13697, -1642, -10940,
    -3430, 5409, 279, 6380, 10450, 8711, -674, 596, 5875, 7259,
    -5697, -1875, 23369, 15618, -5133, -8560, -5, 11277, 1975, -8948,
    19052, 14233, -5687, -2823, -42, 3962, -2622, -5775, 10670, 8814,
    3339, -5146, 11374, 17009, -7738, -9903, 8898, 8243, -2550, -7111,
    16195, 7857, -3918, -2595, 10422, -2690, -32, 2884, 9306, 14273,
    -1538, 3881, 2844, 17390, -7190, -3736, 6315, 18020, -5398, 2392,
    13572, 10701, -10641, -7241, 9185, 19049, -1146, 7347, 5605, 4414,
    -6829, -648, 8631, 4423, -2245, -3791, 18390, 15455, 1035, -6927,
    2285, 20845, -4828, -10915, 12174, 14355, -4558, -2381, 10226, 3628,
    -12252, -4606, 956, -4845, -9042, -2402, -3776, 7080, -4955, -4151,
    -5146, 4704, -16804, 6495, 10110, 8892, -8821, 1053, -5224, 5830,
    -7898, 5173, 1251, -87, -3386, 1805, 235, 3985, -4530, -3696,
    1431, 10362, -12049, -5280, 10100, 4973, 3696, 3224, 4615, 6167,
    -14039, -1459, 3311, 12326, -5998, 2782, 4647, 3189, -15495, -3484,
    -8282, 10039, -8544, 35, 2461, 5983, -18600, -716, -2231, 3460,
    -11880, -1492, 4629, 6078, -13212, -2717, 1398, 1126, -4036, 176,
    -465, -1622, -6694, -3349, -1761, 11946, -8922, -1771, 10173, 8045,
    -4744, 998, -11818, 6273, -3871, 8106, 3501, 3254, -11874, 3323,
    -6113, 4349, -5500, -1619, -6477, 4927, -1327, -4909, -4323, 8833,
    -8271, -982, 148, 922, -13616, -8470, 6167, 25361, -6385, -1328,
    5063, 11721, -14596, 3852, 7745, -17447, 10741, -18336, 3990, -4146,
    -4249, 15986, -15774, -27785, 3787, -11004, -5141, 7147, -8250, 14012,
    4050, -9350, 10340, -2226, -5825, -9314, 3836, -5106, 17655, -10132,
    7276, -8172, 6853, -5964, 1045, -4812, 3697, -22658, 6178, -18309,
    9144, -5509, -8962, -2763, 9081, -9164, 7730, -14982, -6034, -14502,
    -5089, 5606, 11694, -3174, 3985, -13454, -3226, -5975, -13500, 5032,
    10041, -14336, 10103, -13986, 1250, -5867, -11022, -666, 10300, -3303,
    6411, -16879, 5021, -8624, 8320, 8783, -1785, -24750, 1756, -8626,
    -3950, -2709, -2941, 7714, 340, -24064, 9274, -6095, -115, -7064,
    -5775, -137, 7822, -10099, 3621, -6680, 1655, -5498, -5328, -1714,
    3224, -19791, 6035, -14213, 14614, -10554, -6507, -2534, -364, -17822,
    15293, -4882, 140, -13764, 15957, 8468, 5862, -583, 4333, 15003,
    3753, -15310, 4866, -3116, 217, -223, -5589, -7541, -2381, -9216,
    9776, 7587, -10986, -5933, -9965, 6546, -721, -4213, 4011, 9087,
    1502, 3114, 4855, -354, -1251, 1464, 6868, 15841, 1169, 13341,
    1870, 2839, -10354, -9016, 8596, 21684, 183, -8202, 3256, 8307,
    -8802, -13252, 4513, 1214, -41, -9009, 6056, -1673, -3505, -10948,
    9946, 16651, -777, -6689, -7621, 2330, -8056, -13016, 11234, 9255,
    1763, -6255, -6345, 8075, 10733, -12099, 12947, 327, 4898, -10573,
    11888, 4780, -8644, -9881, 18462, 5314, -2803, 3256, 1614, 5193,
    -2200, 3954, -2952, 3183, 8572, 8022, -1605, 1917, 6769, -98,
    4478, 12957, 2684, 2502, 3327, -1444, -5661, -19106, 17401, 12462,
    686, -9668, 63, 8106, -7509, -11607, 1059, -3204, -6332, -15625,
    -2051, 1127, 1305, -18251, 1793, 1468, -5697, -12112, 3732, -9544,
    -9066, -21895, 2789, 4809, -5902, -12098, 2879, 3364, -2582, -8237,
    2021, -4913, -6445, -15630, -4287, 767, -4359, -13402, 2925, -1657,
    3237, -10713, 2237, 608, -8120, -13107, -2850, -2238, -3799, -4523,
    -8731, 9843, -5591, -17356, -2457, 2347, -3233, -11807, -11714, 6381,
    623, -8038, -3392, -426, -2324, -3449, -3580, -2176, -6234, -5527,
    612, 2288, -2868, -5533, -8580, 780, -5204, -15365, 174, 1811,
    -7036, -10530, -7171, 2800, -5876, -25603, 2328, 3842, -436, -9330,
    7303, -7286, -3409, -17053, 3006, 2076, -5331, -20256, 1232, -2395,
    2917, -13663, 1065, 1490, 5548, -14532, 264, -3218, -8107, -10113,
    523, 2733, 7023, -11551, -3427, 8269, -3505, -17154, -4273, -6519,
    22925, -6485, 3052, -15581, 3359, 2114, -3642, -1612, 25035, -2677,
    174, -910, 17936, 7145, -1528, 2090, 20468, 3526, 7702, -7920,
    11495, 3422, -1779, -3242, 20581, -5585, 3630, -8489, 6435, 1135,
    396, -4433, 19252, -6217, 7889, -934, 15218, 2179, -6717, -446,
    10469, -17651, 3141, 5549, 10578, 4426, -11491, -1190, 12540, 1176,
    5023, -12228, 4734, 3459, -1796, -5060, 7399, -8888, 5154, -7200,
    5933, 688, -4089, -3317, 23944, -3632, 1719, -14431, 1960, 938,
    1730, -1871, 14174, -7948, -4067, -3897, 2307, 11261, -705, 2195,
    20015, -9797, -34, -6142, 11056, -688, 3193, -2475, 27997, -11053,
    1010, -13498, 12327, -3370, -1465, -7001, 10652, -9666, 6730, -252,
    14283, 4444, -2766, -334, 18763, -8555, -809, -7110, 5268, 3668,
    5347, 5653, 12286, 4618, -13664, 7074, -3682, 1639, 5145, -6535,
    5380, -3264, -17885, -2456, -914, 10100, 4230, -3925, 8412, 13726,
    -8186, -4709, 1352, -2847, 7844, -4129, 7918, 9051, -10406, 9385,
    -2688, 15660, 2167, 2033, 9687, 2326, -14576, 7343, -4020, 6755,
    7135, 10113, 761, 530, -22204, 10955, -2544, 764, 3000, 7585,
    1929, -1880, -13851, 6738, -8192, 5404, 656, 4823, 10051, 12118,
    -17219, 3279, -5060, 2907, 3678, 5550, 5606, 9660, -17205, -953,
    3011, -8400, 5470, 1880, 7077, 7467, -10579, 2831, 284, 6454,
    4653, 555, 7684, 5670, -15437, -3237, 3044, 2958, 5700, -9350,
    5995, 9146, -4845, -1222, -6566, 3405, 1514, 3499, 10381, 687,
    -12088, 8836, -4205, -3805, 3796, 16774, 4514, -3183, -24400, 5236,
    -7721, 1039, -3665, -3486, 22742, 17150, -9950, 6988, -23387, 4407,
    -825, -6829, 23321, 2525, -7286, 18791, -9674, -4451, -5490, -14330,
    6568, 16919, -14282, -303, -9653, -9511, -5877, -8636, 14187, 12132,
    -7533, 7360, -15826, -10762, -10779, -8573, 28813, 13819, 550, 581,
    -3816, 6118, -8312, -1063, 15344, 15621, -17042, 8549, -15037, 10160,
    -987, -7346, 16220, 14139, -8165, -10797, -31750, -2246, -200, -4854,
    12487, 2053, -6745, 374, -16655, -4810, -496, -2793, 21449, 18084,
    -12276, 3628, -5259, -1650, 9080, -2108, 22625, 2607, -9690, 10360,
    -13669, 12506, -9767, -6637, 19847, 17330, -14180, -2676, -12798, 2244,
    -5634, -17034, 15611, 9253, -18816, 3914, -22627, 12627, -9724, -14933,
    12858, 8572, -3886, -517, -12182, 409, -11744, -9181, 21268, 1930,
    -14005, 12649, -18052, 4551, 9308, 28234, 5601, 14857, 927, -2985,
    8474, -3163, 4266, 20439, 5978, -1288, -12083, 1464, -2322, -6347,
    -3531, 30416, 275, 2858, 199, -4567, -3555, -5077, -6650, 4574,
    -2335, -1158, -9792, -7176, 195, 5242, 17128, 19421, -3520, 5065,
    2736, -3259, -1435, -685, 1381, 25522, -10352, -1439, -2021, 2950,
    8607, -12270, 1327, 13873, -1522, 20654, -5145, 901, -2506, 524,
    1955, 21042, -2206, 6742, -10234, -966, -3732, -13155, 8065, 26314,
    -79, 18223, -1092, -2323, 2907, -5022, -6201, 19276, 8029, 212,
    -6169, -145, 10231, -12885, -5355, 32313, -6184, -8187, -3361, -4115,
    1113, -630, 2570, 17210, -4849, 2290, -2757, -2110, 5260, 2675,
    7486, 24749, 3408, -2227, -3142, -1191, -10036, -1119, 9273, 26637,
    -22310, 686, -10594, 1786, -5041, -9148, 3843, 10942, -5222, 588,
    -24417, -10417, -8628, -23167, -549, -7355, 10770, 434, 1032, 11859,
    -11441, -17127, 10183, -133, -5905, 30, -5414, 1225, -3943, -17853,
    9165, 2201, 5070, 1087, 6007, -2072, -10728, -28392, 2933, -5835,
    -2920, -6683, -8098, -3684, -1335, -11106, 960, 2990, -10219, -2821,
    -8457, 7703, -21161, -12585, 14626, 12015, 6146, 1458, -9457, 855,
    -5792, -23165, 9003, 14300, -2572, -9418, -6648, -2697, -3528, -18004,
    11091, 4065, -6074, -8074, -17753, -5625, 4075, -26489, 7441, 588,
    -1325, 12123, -12581, 3851, -5796, -24556, 1521, -4550, -1491, -11055,
    5900, 14401, 113, -28762, 2914, -1196, 7147, 8106, -6429, -7424,
    -3659, -24688, 6348, -5503, -1989, -8041, 4881, 570, -5559, -11879,
    5078, 1968, -8056, -11382, -11352, -379, -3772, 4131, 2652, 30644,
    -5599, -3100, 5698, 1745, -942, 23438, 1392, 2652, -6392, 9389,
    1581, 6303, 8441, 11206, 209, 20331, -14157, -2251, 367, -1916,
    -5738, 10859, -191, 21463, -7834, 86, 598, 5863, 8154, 18734,
    7592, 22495, -14537, -5903, -1860, -7183, -9498, 9332, 1507, 12753,
    -6819, -4129, 256, -3663, 6072, 2014, -4502, 7833, -12811, 7505,
    1155, 3714, 2198, 9420, -4287, 5210, -8712, -10358, -3999, 4019,
    -770, 12253, 2752, 18135, -13086, -2547, 3333, 952, -4885, 10460,
    -8117, 4125, -7275, 4812, 4035, 3019, 3846, 11051, -10495, 6852,
    -17837, -2020, -5062, -2023, -3384, 4598, -209, 23988, -1678, -1448,
    1270, -2334, 9834, 5562, -3825, 16806, -14113, -3489, 377, -7028,
    -7086, 15865, 5954, 22319, -11569, 7266, -5148, 2908, -1413, 14300,
    22451, 11042, 6011, -1058, -17045, -10259, 11792, -13091, 24406, 234,
    3145, 3184, -3665, 1481, 14589, -9942, 23739, -5227, -267, 6075,
    -4629, -1337, 8565, -7165, 20248, 4440, 13836, -450, -1447, 8231,
    -2187, -4378, 7707, -7922, -9632, -1102, -10635, -1514, -18058, 8230,
    10722, 9916, 3014, 1162, -7457, -1172, -5068, -6086, 6028, -247,
    5143, 9091, -7301, -1480, -14806, -1902, 20456, 888, 7626, -8503,
    4803, -1765, -7192, -6121, 16721, 7708, 10049, -885, -16488, -1127,
    -1155, 2563, 28215, 6930, 206, 4794, -13836, 1067, 9376, -8724,
    10035, -1876, -6430, 5239, -8740, -5449, 1987, 4782, 17282, 1049,
    8219, 2955, 5582, 7706, 2527, -6721, 10284, -5806, -10636, -5615,
    -9647, -7001, 898, 14071, 20229, 2501, -2112, 12503, -2942, 4068,
    -8744, -6059, 4398, 6897, -24085, 676, -12468, 3106, -11011, -16890,
    3137, -4664, -10777, -19201, 2934, 14194, -58, -8706, 4868, -5957,
    -9198, -16652, 2645, 9332, -2216, -9967, 5413, -1678, -11310, -9246,
    866, 2776, -3184, -6745, 906, 2139, -19424, -15396, 2453, 7278,
    2577, -9543, 3217, 14220, -17354, -19175, 6756, 539, 4667, -5502,
    6181, 8716, -9329, -8162, -4100, 3865, -7552, -12723, 2086, 8067,
    -10357, -7338, 598, 8821, -14523, -2377, 6268, 3684, -15713, -2311,
    -2131, -794, -13150, -15878, 1820, -6461, -5933, -18748, 8539, 1477,
    -2647, -19093, 2978, -37, -18568, -11244, 12257, 6722, -8644, -15054,
    -2033, -6253, -7836, -3074, 4134, 4782, -8666, -4148, 5340, 4535,
    -17292, -14074, -1640, 7159, 96, -21905, 1675, 12212, -15248, -9090,
    4963, 2469, -10264, -3713, 9694, 6540, 1517, 5288, 7907, -1988,
    4542, -8657, 5925, 4811, 4238, -7401, 13799, -4373, -12110, -8845,
    4777, -2752, 2953, -7472, 9355, 590, -737, -4195, 7785, -4187,
    3740, -4153, 1521, 3166, -2990, -14272, 11381, 4822, -11830, -1911,
    -228, -525, -1239, -4631, 9276, -1330, -5408, -11917, 4509, -2529,
    -3702, -2741, 15930, -309, 10115, -10694, 5075, -1404, -3625, -8209,
    13077, -730, 5657, -3509, 15605, -823, -9275, -11119, 16289, 941,
    -2930, 762, 1920, -1807, 415, -1291, 1076, 973, 1888, -12751,
    7797, 760, -7751, -8185, 5541, 768, 9610, -10723, -1621, -10055,
    -6900, -3999, 10708, 1218, 814, -5546, 5797, -1033, 1623, -9892,
    4369, 6940, -1139, -4404, 5498, -1460, -1610, -817, 13118, 9142,
    -8626, -20668, 5517, 3812, -4973, -11365, 11519, 281, -11485, 16713,
    -10847, 18081, -1907, 5768, 8107, -13397, 16218, 30184, -4690, 7103,
    95, -8253, 10792, -12951, -8203, 13112, -6822, 5923, 3885, 8205,
    -6300, 4774, -20009, 13376, -7346, 12329, -6864, 3514, 187, 8715,
    -9223, 26867, -6259, 19845, -12795, 8672, 4363, 7181, -7391, 10049,
    -10763, 10097, 6509, 14239, 6775, -5544, -11972, 1290, -764, 15555,
    3971, 1427, 15129, -5607, -11600, 15957, -8903, 14706, -76, 6866,
    10202, 1609, -8256, 1294, -3588, 17102, -8310, 8067, -3271, -5191,
    -457, 25388, 2785, 7514, 1185, 2369, -352, -9689, 1806, 24904,
    -3454, 7987, 4183, 6637, 5117, -2123, -7372, 11149, -8440, 5311,
    -1886, 3799, 1303, 5869, -5903, 19081, -1972, 15407, -14095, 11969,
    4762, 4525, -2998, 18728, -13540, 9237, -1799, 13084
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
