/**
 * @file adc.c
 * @brief ADC, ePWM trigger and DMA setup for audio capture.
 */
#include "adc.h"

/** @brief Stops the ePWM trigger after the first conversion burst. */
#pragma CODE_SECTION(adca1_isr, ".TI.ramfunc");
__interrupt void adca1_isr(void)
{
    EPwm2Regs.ETSEL.bit.SOCAEN = 0;
    PieCtrlRegs.PIEIER1.bit.INTx1 = 0;
    PieCtrlRegs.PIEACK.all = PIEACK_GROUP1;
}

/** @brief Marks the capture complete. */
#pragma CODE_SECTION(dmach1_isr, ".TI.ramfunc");
__interrupt void dmach1_isr(void)
{
    done = 1;

    PieCtrlRegs.PIEACK.all = PIEACK_GROUP7;
}

/** @brief ePWM2 SOCA on counter zero, period 4000. */
void ConfigureEPWM(void)
{
    EPwm2Regs.TBCTL.all = 0x0000;
    EPwm2Regs.TBPRD = 4000;
    EPwm2Regs.AQCTLA.bit.ZRO = AQ_SET;
    EPwm2Regs.AQCTLA.bit.CAU = AQ_CLEAR;
    EPwm2Regs.CMPA.bit.CMPA = 2000;
    EPwm2Regs.ETSEL.bit.SOCASEL = ET_CTR_ZERO;
    EPwm2Regs.ETPS.bit.SOCAPRD =  ET_1ST;
    EPwm2Regs.ETCNTINITCTL.bit.SOCAINITEN = 1;
}

/** @brief ADCA: 12-bit, single-ended, powered up. */
void ConfigureADC(void)
{
    EALLOW;
    AdcaRegs.ADCCTL2.bit.PRESCALE = 6;

    AdcSetMode(ADC_ADCA, ADC_RESOLUTION_12BIT, ADC_SIGNALMODE_SINGLE);

    AdcaRegs.ADCCTL1.bit.INTPULSEPOS = 1;
    AdcaRegs.ADCCTL1.bit.ADCPWDNZ = 1;

    DELAY_US(1000);
    EDIS;
}

/**
 * @brief SOC0-15 on one channel, chained for continuous sampling.
 * @param adcRegs ADC register block.
 * @param channel Input channel.
 */
void SetupADCContinuous(volatile struct ADC_REGS * adcRegs, Uint16 channel)
{
    Uint16 acqps = 14;

    EALLOW;
    adcRegs->ADCSOC0CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC1CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC2CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC3CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC4CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC5CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC6CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC7CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC8CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC9CTL.bit.CHSEL  = channel;
    adcRegs->ADCSOC10CTL.bit.CHSEL = channel;
    adcRegs->ADCSOC11CTL.bit.CHSEL = channel;
    adcRegs->ADCSOC12CTL.bit.CHSEL = channel;
    adcRegs->ADCSOC13CTL.bit.CHSEL = channel;
    adcRegs->ADCSOC14CTL.bit.CHSEL = channel;
    adcRegs->ADCSOC15CTL.bit.CHSEL = channel;

    adcRegs->ADCSOC0CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC1CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC2CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC3CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC4CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC5CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC6CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC7CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC8CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC9CTL.bit.ACQPS  = acqps;
    adcRegs->ADCSOC10CTL.bit.ACQPS = acqps;
    adcRegs->ADCSOC11CTL.bit.ACQPS = acqps;
    adcRegs->ADCSOC12CTL.bit.ACQPS = acqps;
    adcRegs->ADCSOC13CTL.bit.ACQPS = acqps;
    adcRegs->ADCSOC14CTL.bit.ACQPS = acqps;
    adcRegs->ADCSOC15CTL.bit.ACQPS = acqps;

    adcRegs->ADCSOC0CTL.bit.TRIGSEL = 7;

    adcRegs->ADCINTSOCSEL1.bit.SOC1 = 1;
    adcRegs->ADCINTSOCSEL1.bit.SOC2 = 1;
    adcRegs->ADCINTSOCSEL1.bit.SOC3 = 1;
    adcRegs->ADCINTSOCSEL1.bit.SOC4 = 1;
    adcRegs->ADCINTSOCSEL1.bit.SOC5 = 1;
    adcRegs->ADCINTSOCSEL1.bit.SOC6 = 1;
    adcRegs->ADCINTSOCSEL1.bit.SOC7 = 1;
    adcRegs->ADCINTSOCSEL2.bit.SOC8 = 1;
    adcRegs->ADCINTSOCSEL2.bit.SOC9 = 1;
    adcRegs->ADCINTSOCSEL2.bit.SOC10 = 1;
    adcRegs->ADCINTSOCSEL2.bit.SOC11 = 1;
    adcRegs->ADCINTSOCSEL2.bit.SOC12 = 1;
    adcRegs->ADCINTSOCSEL2.bit.SOC13 = 1;
    adcRegs->ADCINTSOCSEL2.bit.SOC14 = 1;
    adcRegs->ADCINTSOCSEL2.bit.SOC15 = 1;

    adcRegs->ADCINTSEL1N2.bit.INT1E = 1;
    adcRegs->ADCINTSEL1N2.bit.INT2E = 1;
    adcRegs->ADCINTSEL3N4.bit.INT3E = 0;
    adcRegs->ADCINTSEL3N4.bit.INT4E = 0;

    adcRegs->ADCINTSEL1N2.bit.INT1CONT = 1;
    adcRegs->ADCINTSEL1N2.bit.INT2CONT = 1;

    adcRegs->ADCINTSEL1N2.bit.INT1SEL = 0;
    adcRegs->ADCINTSEL1N2.bit.INT2SEL = 15;

    EDIS;
}

/** @brief DMA CH1: 16-word bursts from ADC results into adcData0. */
void DMAInit(void)
{
    DMAInitialize();
    DMACH1AddrConfig(adcData0, &AdcaResultRegs.ADCRESULT0);

    DMACH1BurstConfig(15, 1, 1);
    DMACH1TransferConfig((RESULTS_BUFFER_SIZE / 16) - 1, -15, 1);

    DMACH1ModeConfig(
                        DMA_ADCAINT2,
                        PERINT_ENABLE,
                        ONESHOT_DISABLE,
                        CONT_ENABLE,
                        SYNC_DISABLE,
                        SYNC_SRC,
                        OVRFLOW_DISABLE,
                        SIXTEEN_BIT,
                        CHINT_END,
                        CHINT_ENABLE
                    );
}
