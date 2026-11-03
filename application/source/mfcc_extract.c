/**
 * @file mfcc_extract.c
 * @brief MFCC / log-mel feature extraction on the CPU.
 * @details Real-input radix-4 FFT; tables come from feature_tables.h.
 */
#include <math.h>
#include <stdint.h>
#include "feature_tables.h"

#define FFT_N   FEAT_N_FFT /**< FFT length. */
#define FFT_M   (FEAT_N_FFT / 2) /**< Complex FFT length. */

#if (FFT_N < 32) || (FFT_N & (FFT_N - 1))
#error "FEAT_N_FFT must be a power of two, 32 or larger"
#endif

#ifndef MFCC_FAST_LOG
#define MFCC_FAST_LOG 1 /**< 1: polynomial log, 0: log10f(). */
#endif

#ifndef FEAT_USE_DCT
#define FEAT_USE_DCT 1 /**< 1: MFCC, 0: log-mel bands. */
#endif
#if !FEAT_USE_DCT && (FEAT_N_OUT != FEAT_N_MELS)
#error "FEAT_USE_DCT 0 needs one output per band: FEAT_N_OUT == FEAT_N_MELS"
#endif

static float log_mel[FEAT_N_FRAMES][FEAT_N_MELS]; /**< Log-mel energies (dB). */

static float z[2 * FFT_M + 2]; /**< FFT buffer, then power spectrum. */

static float tw[2 * FFT_M]; /**< Twiddles for the last stage and split. */

static float tw4[2 * FFT_M]; /**< Radix-4 stage twiddles. */

static uint16_t rev4[FFT_M / 4]; /**< Bit-reversed input offsets. */

static int tables_ready = 0; /**< Set once build_tables() ran. */

/** @brief Fills the twiddle and bit-reversal tables. */
static void build_tables(void)
{
    const float two_pi = 2.0f * 3.14159265358979f;
    float *t;
    uint16_t k, n, bit, r, h;

    for (k = 0; k < FFT_M; k++) {
        float ang = two_pi * (float)k / (float)FFT_N;
        tw[2 * k]     = cosf(ang);
        tw[2 * k + 1] = sinf(ang);
    }

    t = tw4;
    for (h = 4; 4 * h <= FFT_M; h <<= 2) {
        for (k = 0; k < h; k++) {
            float ang = two_pi * (float)k / (float)(4 * h);
            *t++ = cosf(ang);         *t++ = sinf(ang);
            *t++ = cosf(2.0f * ang);  *t++ = sinf(2.0f * ang);
            *t++ = cosf(3.0f * ang);  *t++ = sinf(3.0f * ang);
        }
    }

    for (n = 0; n < FFT_M / 4; n++) {
        r = 0;
        k = 4 * n;
        for (bit = FFT_M >> 1; bit != 0; bit >>= 1) {
            if (k & 1) r |= bit;
            k >>= 1;
        }
        rev4[n] = 2 * r;
    }
    tables_ready = 1;
}

#if MFCC_FAST_LOG

/**
 * @brief Fast 10*log10(x).
 * @param x Positive power value.
 * @return Value in dB.
 */
static float power_db(float x)
{
    union { float f; uint32_t u; } b;
    float m, t, t2, ln_m;
    int16_t e;

    b.f = x;
    e = (int16_t)((b.u >> 23) & 0xFFu) - 127;
    b.u = (b.u & 0x007FFFFFuL) | 0x3F800000uL;
    m = b.f;
    if (m > 1.41421356f) {
        m *= 0.5f;
        e += 1;
    }
    t  = (m - 1.0f) / (m + 1.0f);
    t2 = t * t;
    ln_m = 2.0f * t * (1.0f + t2 * (0.333333333f + t2 * (0.2f + t2 * 0.142857143f)));
    return 3.01029995664f * (float)e + 4.34294481903f * ln_m;
}
#define POWER_DB(x)   power_db(x)
#else
#define POWER_DB(x)   (10.0f * log10f(x))
#endif

/**
 * @brief Windowed power spectrum of one frame into z[].
 * @param s    First ADC sample of the frame.
 * @param mean DC offset to subtract.
 */
static void frame_power(const unsigned int *s, float mean)
{
    register float *p;
    register float *q;
    register const float *t;
    float *p2, *p3;
    const unsigned int *sp;
    const float *hp;
    const uint16_t *r = rev4;
    float ar, ai, br, bi, cr, ci, dr, di;
    float ur, ui, vr, vi, wr, wi;
    uint16_t n, g, k, h, h2;

#define WS(i)   (((float)sp[i] - mean) * hp[i])

    p = z;
    for (n = FFT_M / 4; n != 0; n--) {
        sp = s + *r;
        hp = feat_hann_window + *r;
        r++;
        ar = WS(0);                 ai = WS(1);
        br = WS(FFT_M);             bi = WS(FFT_M + 1);
        cr = WS(FFT_M / 2);         ci = WS(FFT_M / 2 + 1);
        dr = WS(3 * FFT_M / 2);     di = WS(3 * FFT_M / 2 + 1);
        ur = ar + br;  ui = ai + bi;
        vr = cr + dr;  vi = ci + di;
        p[0] = ur + vr;  p[1] = ui + vi;
        p[4] = ur - vr;  p[5] = ui - vi;
        ur = ar - br;  ui = ai - bi;
        vr = cr - dr;  vi = ci - di;
        p[2] = ur + vi;  p[3] = ui - vr;
        p[6] = ur - vi;  p[7] = ui + vr;
        p += 8;
    }
#undef WS

    t = tw4;
    for (h = 4; 4 * h <= FFT_M; h <<= 2) {
        const float *t0 = t;
        h2 = 2 * h;
        p = z;
        for (g = FFT_M / (4 * h); g != 0; g--) {
            q  = p + h2;
            p2 = q + h2;
            p3 = p2 + h2;
            t = t0;
            for (k = h; k != 0; k--) {
                ar = q[0]  * t[2] + q[1]  * t[3];
                ai = q[1]  * t[2] - q[0]  * t[3];
                br = p2[0] * t[0] + p2[1] * t[1];
                bi = p2[1] * t[0] - p2[0] * t[1];
                cr = p3[0] * t[4] + p3[1] * t[5];
                ci = p3[1] * t[4] - p3[0] * t[5];
                t += 6;
                ur = p[0] + ar;  ui = p[1] + ai;
                vr = br + cr;    vi = bi + ci;
                wr = p[0] - ar;  wi = p[1] - ai;
                p[0]  = ur + vr;  p[1]  = ui + vi;
                p2[0] = ur - vr;  p2[1] = ui - vi;
                vr = br - cr;    vi = bi - ci;
                q[0]  = wr + vi;  q[1]  = wi - vr;
                p3[0] = wr - vi;  p3[1] = wi + vr;
                p += 2;  q += 2;  p2 += 2;  p3 += 2;
            }
            p = p3;
        }
    }

    if (h < FFT_M) {
        p = z;
        q = z + 2 * h;
        t = tw;
        for (k = h; k != 0; k--) {
            vr = q[0] * t[0] + q[1] * t[1];
            vi = q[1] * t[0] - q[0] * t[1];
            t += 4;
            ur = p[0]; ui = p[1];
            p[0] = ur + vr;  p[1] = ui + vi;
            q[0] = ur - vr;  q[1] = ui - vi;
            p += 2;
            q += 2;
        }
    }

    ur = z[0]; ui = z[1];
    vr = ur + ui;
    vi = ur - ui;
    z[0]         = 4.0f * vr * vr;
    z[2 * FFT_M] = 4.0f * vi * vi;

    p = z + 2;
    q = z + 2 * FFT_M - 2;
    t = tw + 2;
    for (k = FFT_M / 2 - 1; k != 0; k--) {
        float er = p[0] + q[0];
        float ei = p[1] - q[1];
        float dr = p[0] - q[0];
        float di = p[1] + q[1];
        wr = t[0]; wi = t[1];
        t += 2;
        vr = wr * di - wi * dr;
        vi = wr * dr + wi * di;
        ur = er + vr;  ui = ei - vi;
        p[0] = ur * ur + ui * ui;
        ur = er - vr;  ui = ei + vi;
        q[0] = ur * ur + ui * ui;
        p += 2;
        q -= 2;
    }
    ur = p[0]; ui = p[1];
    p[0] = 4.0f * (ur * ur + ui * ui);
}

/**
 * @brief Turns one ADC window into normalized network features.
 * @param adc_buf FEAT_SLICE_LEN raw ADC codes.
 * @param out     Z-scored features, clipped to FEAT_CLIP.
 */
void mfcc_extract(const unsigned int *adc_buf, float out[FEAT_N_FRAMES][FEAT_N_OUT])
{
    register const float *a;
    register const float *w;
    register float acc;
    const unsigned int *s;
    const uint16_t *ms, *ml;
    const float *mu, *sg;
    float *lm, *o;
    float mean, gmax, floor_v, v;
    uint32_t sum;
    uint16_t i, f, m, k;

    if (!tables_ready) build_tables();

    sum = 0;
    s = adc_buf;
    for (i = FEAT_SLICE_LEN; i != 0; i--) sum += *s++;
    mean = (float)sum / (float)FEAT_SLICE_LEN;

    lm = &log_mel[0][0];
    gmax = -1.0e30f;
    s = adc_buf;
    for (f = FEAT_N_FRAMES; f != 0; f--) {
        frame_power(s, mean);
        s += FEAT_HOP;

        w  = feat_mel_weights;
        ms = feat_mel_start;
        ml = feat_mel_len;
        for (m = FEAT_N_MELS; m != 0; m--) {
            a = z + 2 * (*ms++);
            acc = 0.0f;
            for (k = *ml++; k != 0; k--) {
                acc += (*a) * (*w++);
                a += 2;
            }
            acc *= 0.25f / (FEAT_ADC_SCALE * FEAT_ADC_SCALE);
            if (acc < FEAT_AMIN) acc = FEAT_AMIN;
            v = POWER_DB(acc);
            *lm++ = v;
            if (v > gmax) gmax = v;
        }
    }

    floor_v = gmax - FEAT_TOP_DB;
    lm = &log_mel[0][0];
    for (i = FEAT_N_FRAMES * FEAT_N_MELS; i != 0; i--) {
        if (*lm < floor_v) *lm = floor_v;
        lm++;
    }

    lm = &log_mel[0][0];
    o  = &out[0][0];
    for (f = FEAT_N_FRAMES; f != 0; f--) {
        mu = feat_mu;
        sg = feat_sigma;
#if FEAT_USE_DCT
        w  = &feat_dct_matrix[0][0];
        for (i = FEAT_N_OUT; i != 0; i--) {
            a = lm;
            acc = 0.0f;
            for (m = FEAT_N_MELS; m != 0; m--) acc += (*w++) * (*a++);
            v = (acc - *mu++) / *sg++;
            if (v >  FEAT_CLIP) v =  FEAT_CLIP;
            if (v < -FEAT_CLIP) v = -FEAT_CLIP;
            *o++ = v;
        }
        lm += FEAT_N_MELS;
#else
        for (i = FEAT_N_OUT; i != 0; i--) {
            v = (*lm++ - *mu++) / *sg++;
            if (v >  FEAT_CLIP) v =  FEAT_CLIP;
            if (v < -FEAT_CLIP) v = -FEAT_CLIP;
            *o++ = v;
        }
#endif
    }
}
