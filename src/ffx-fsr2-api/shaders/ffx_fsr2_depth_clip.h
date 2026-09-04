// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFX_FSR2_DEPTH_CLIP_H
#define FFX_FSR2_DEPTH_CLIP_H

FFX_STATIC const FfxFloat32 DepthClipBaseScale = 4.0f;

FfxFloat32 ComputeDepthClip(FfxFloat32x2 fUvSample, FfxFloat32 fCurrentDepthSample)
{
    FfxFloat32 fCurrentDepthViewSpace = GetViewSpaceDepth(fCurrentDepthSample);
    BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fUvSample, RenderSize());

    FfxFloat32 fDilatedSum = 0.0f;
    FfxFloat32 fDepth = 0.0f;
    FfxFloat32 fWeightSum = 0.0f;

    // Keep the indices literal. NVIDIA's OpenGL compiler otherwise places these
    // private arrays in local memory even though every loop bound is constant.
#if FFX_FSR2_OPTION_INVERTED_DEPTH
#define FFX_FSR2_DEPTH_PLANE(PREV_DEPTH, CURRENT_DEPTH) ffxMin((PREV_DEPTH), (CURRENT_DEPTH))
#else
#define FFX_FSR2_DEPTH_PLANE(PREV_DEPTH, CURRENT_DEPTH) ffxMax((PREV_DEPTH), (CURRENT_DEPTH))
#endif

#define FFX_FSR2_ACCUMULATE_DEPTH_SAMPLE(SAMPLE_INDEX)                                             \
    {                                                                                              \
        const FfxInt32x2 iOffset = bilinearInfo.iOffsets[(SAMPLE_INDEX)];                           \
        const FfxInt32x2 iSamplePos = bilinearInfo.iBasePos + iOffset;                              \
        if (IsOnScreen(iSamplePos, RenderSize())) {                                                 \
            const FfxFloat32 fWeight = bilinearInfo.fWeights[(SAMPLE_INDEX)];                        \
            if (fWeight > fReconstructedDepthBilinearWeightThreshold) {                             \
                const FfxFloat32 fPrevDepthSample = LoadReconstructedPrevDepth(iSamplePos);          \
                const FfxFloat32 fPrevNearestDepthViewSpace = GetViewSpaceDepth(fPrevDepthSample);   \
                const FfxFloat32 fDepthDiff = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;   \
                if (fDepthDiff > 0.0f) {                                                             \
                    const FfxFloat32 fPlaneDepth =                                                   \
                        FFX_FSR2_DEPTH_PLANE(fPrevDepthSample, fCurrentDepthSample);                  \
                    const FfxFloat32x3 fCenter = GetViewSpacePosition(                               \
                        FfxInt32x2(RenderSize() * 0.5f), RenderSize(), fPlaneDepth);                  \
                    const FfxFloat32x3 fCorner =                                                     \
                        GetViewSpacePosition(FfxInt32x2(0, 0), RenderSize(), fPlaneDepth);           \
                    const FfxFloat32 fHalfViewportWidth = length(FfxFloat32x2(RenderSize()));        \
                    const FfxFloat32 fDepthThreshold =                                               \
                        ffxMax(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);                   \
                    const FfxFloat32 Ksep = 1.37e-05f;                                               \
                    const FfxFloat32 Kfov = length(fCorner) / length(fCenter);                        \
                    const FfxFloat32 fRequiredDepthSeparation =                                      \
                        Ksep * Kfov * fHalfViewportWidth * fDepthThreshold;                           \
                    const FfxFloat32 fResolutionFactor = ffxSaturate(                                \
                        length(FfxFloat32x2(RenderSize())) /                                        \
                        length(FfxFloat32x2(1920.0f, 1080.0f)));                                    \
                    const FfxFloat32 fPower = ffxLerp(1.0f, 3.0f, fResolutionFactor);                \
                    fDepth += ffxPow(                                                               \
                        ffxSaturate(FfxFloat32(fRequiredDepthSeparation / fDepthDiff)),              \
                        fPower) * fWeight;                                                           \
                    fWeightSum += fWeight;                                                           \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
    }

    FFX_FSR2_ACCUMULATE_DEPTH_SAMPLE(0)
    FFX_FSR2_ACCUMULATE_DEPTH_SAMPLE(1)
    FFX_FSR2_ACCUMULATE_DEPTH_SAMPLE(2)
    FFX_FSR2_ACCUMULATE_DEPTH_SAMPLE(3)

#undef FFX_FSR2_ACCUMULATE_DEPTH_SAMPLE
#undef FFX_FSR2_DEPTH_PLANE

    return (fWeightSum > 0) ? ffxSaturate(1.0f - fDepth / fWeightSum) : 0.0f;
}

FfxFloat32 ComputeMotionDivergence(FfxInt32x2 iPxPos, FfxInt32x2 iPxInputMotionVectorSize)
{
    FfxFloat32 minconvergence = 1.0f;

    FfxFloat32x2 fMotionVectorNucleus = LoadInputMotionVector(iPxPos);
    FfxFloat32 fNucleusVelocityLr = length(fMotionVectorNucleus * RenderSize());
    FfxFloat32 fMaxVelocityUv = length(fMotionVectorNucleus);

    const FfxFloat32 MotionVectorVelocityEpsilon = 1e-02f;

    if (fNucleusVelocityLr > MotionVectorVelocityEpsilon) {
        for (FfxInt32 y = -1; y <= 1; ++y) {
            for (FfxInt32 x = -1; x <= 1; ++x) {

                FfxInt32x2 sp = ClampLoad(iPxPos, FfxInt32x2(x, y), iPxInputMotionVectorSize);

                FfxFloat32x2 fMotionVector = LoadInputMotionVector(sp);
                FfxFloat32 fVelocityUv = length(fMotionVector);

                fMaxVelocityUv = ffxMax(fVelocityUv, fMaxVelocityUv);
                fVelocityUv = ffxMax(fVelocityUv, fMaxVelocityUv);
                minconvergence = ffxMin(minconvergence, dot(fMotionVector / fVelocityUv, fMotionVectorNucleus / fVelocityUv));
            }
        }
    }

    return ffxSaturate(1.0f - minconvergence) * ffxSaturate(fMaxVelocityUv / 0.01f);
}

FfxFloat32 ComputeDepthDivergence(FfxInt32x2 iPxPos)
{
    const FfxFloat32 fMaxDistInMeters = GetMaxDistanceInMeters();
    FfxFloat32 fDepthMax = 0.0f;
    FfxFloat32 fDepthMin = fMaxDistInMeters;

    FfxInt32 iMaxDistFound = 0;

    for (FfxInt32 y = -1; y < 2; y++) {
        for (FfxInt32 x = -1; x < 2; x++) {

            const FfxInt32x2 iOffset = FfxInt32x2(x, y);
            const FfxInt32x2 iSamplePos = iPxPos + iOffset;

            const FfxFloat32 fOnScreenFactor = IsOnScreen(iSamplePos, RenderSize()) ? 1.0f : 0.0f;
            FfxFloat32 fDepth = GetViewSpaceDepthInMeters(LoadDilatedDepth(iSamplePos)) * fOnScreenFactor;

            iMaxDistFound |= FfxInt32(fMaxDistInMeters == fDepth);

            fDepthMin = ffxMin(fDepthMin, fDepth);
            fDepthMax = ffxMax(fDepthMax, fDepth);
        }
    }

    return (1.0f - fDepthMin / fDepthMax) * (FfxBoolean(iMaxDistFound) ? 0.0f : 1.0f);
}

FfxFloat32 ComputeTemporalMotionDivergence(FfxInt32x2 iPxPos)
{
    const FfxFloat32x2 fUv = FfxFloat32x2(iPxPos + 0.5f) / RenderSize();

    FfxFloat32x2 fMotionVector = LoadDilatedMotionVector(iPxPos);
    FfxFloat32x2 fReprojectedUv = fUv + fMotionVector;
    fReprojectedUv = ClampUv(fReprojectedUv, RenderSize(), MaxRenderSize());
    FfxFloat32x2 fPrevMotionVector = SamplePreviousDilatedMotionVector(fReprojectedUv);

    float fPxDistance = length(fMotionVector * DisplaySize());
    return fPxDistance > 1.0f ? ffxLerp(0.0f, 1.0f - ffxSaturate(length(fPrevMotionVector) / length(fMotionVector)), ffxSaturate(ffxPow(fPxDistance / 20.0f, 3.0f))) : 0;
}

void PreProcessReactiveMasks(FfxInt32x2 iPxLrPos, FfxFloat32 fMotionDivergence)
{
    // Compensate for bilinear sampling in accumulation pass

    FfxFloat32x3 fReferenceColor = LoadInputColor(iPxLrPos).xyz;
    FfxFloat32x2 fReactiveFactor = FfxFloat32x2(0.0f, fMotionDivergence);

    float fMasksSum = 0.0f;

    FfxFloat32x3 fColorSamples[9];
    FfxFloat32 fReactiveSamples[9];
    FfxFloat32 fTransparencyAndCompositionSamples[9];

    // FFX_UNROLL is empty for GLSL, and dynamic private-array indexing spills on
    // NVIDIA OpenGL. Explicit expansion lets glslang scalarize all three arrays.
#define FFX_FSR2_LOAD_REACTIVE_SAMPLE(X, Y)                                                              \
    {                                                                                                    \
        const FfxInt32x2 sampleCoord =                                                                   \
            ClampLoad(iPxLrPos, FfxInt32x2((X), (Y)), FfxInt32x2(RenderSize()));                         \
        const FfxInt32 sampleIdx = ((Y) + 1) * 3 + (X) + 1;                                             \
        const FfxFloat32x3 fColorSample = LoadInputColor(sampleCoord).xyz;                              \
        const FfxFloat32 fReactiveSample = LoadReactiveMask(sampleCoord);                               \
        const FfxFloat32 fTransparencyAndCompositionSample =                                            \
            LoadTransparencyAndCompositionMask(sampleCoord);                                             \
        fColorSamples[sampleIdx] = fColorSample;                                                         \
        fReactiveSamples[sampleIdx] = fReactiveSample;                                                   \
        fTransparencyAndCompositionSamples[sampleIdx] = fTransparencyAndCompositionSample;              \
        fMasksSum += (fReactiveSample + fTransparencyAndCompositionSample);                              \
    }

    FFX_FSR2_LOAD_REACTIVE_SAMPLE(-1, -1)
    FFX_FSR2_LOAD_REACTIVE_SAMPLE( 0, -1)
    FFX_FSR2_LOAD_REACTIVE_SAMPLE( 1, -1)
    FFX_FSR2_LOAD_REACTIVE_SAMPLE(-1,  0)
    FFX_FSR2_LOAD_REACTIVE_SAMPLE( 0,  0)
    FFX_FSR2_LOAD_REACTIVE_SAMPLE( 1,  0)
    FFX_FSR2_LOAD_REACTIVE_SAMPLE(-1,  1)
    FFX_FSR2_LOAD_REACTIVE_SAMPLE( 0,  1)
    FFX_FSR2_LOAD_REACTIVE_SAMPLE( 1,  1)

#undef FFX_FSR2_LOAD_REACTIVE_SAMPLE

    if (fMasksSum > 0)
    {
#define FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(SAMPLE_IDX)                                              \
    {                                                                                                \
        const FfxFloat32x3 fColorSample = fColorSamples[(SAMPLE_IDX)];                               \
        const FfxFloat32 fReactiveSample = fReactiveSamples[(SAMPLE_IDX)];                           \
        const FfxFloat32 fTransparencyAndCompositionSample =                                        \
            fTransparencyAndCompositionSamples[(SAMPLE_IDX)];                                        \
        const FfxFloat32 fMaxLenSq =                                                                 \
            ffxMax(dot(fReferenceColor, fReferenceColor), dot(fColorSample, fColorSample));          \
        const FfxFloat32 fSimilarity = dot(fReferenceColor, fColorSample) / fMaxLenSq;               \
        const FfxFloat32 fPowerBiasMax = 6.0f;                                                       \
        const FfxFloat32 fSimilarityPower =                                                         \
            1.0f + (fPowerBiasMax - fSimilarity * fPowerBiasMax);                                    \
        const FfxFloat32 fWeightedReactiveSample = ffxPow(fReactiveSample, fSimilarityPower);        \
        const FfxFloat32 fWeightedTransparencyAndCompositionSample =                                \
            ffxPow(fTransparencyAndCompositionSample, fSimilarityPower);                             \
        fReactiveFactor = ffxMax(                                                                    \
            fReactiveFactor,                                                                         \
            FfxFloat32x2(fWeightedReactiveSample, fWeightedTransparencyAndCompositionSample));       \
    }

        FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(0)
        FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(1)
        FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(2)
        FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(3)
        FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(4)
        FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(5)
        FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(6)
        FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(7)
        FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE(8)

#undef FFX_FSR2_ACCUMULATE_REACTIVE_SAMPLE
    }

    StoreDilatedReactiveMasks(iPxLrPos, fReactiveFactor);
}

FfxFloat32x3 ComputePreparedInputColor(FfxInt32x2 iPxLrPos)
{
    //We assume linear data. if non-linear input (sRGB, ...),
    //then we should convert to linear first and back to sRGB on output.
    FfxFloat32x3 fRgb = ffxMax(FfxFloat32x3(0, 0, 0), LoadInputColor(iPxLrPos));

    fRgb = PrepareRgb(fRgb, Exposure(), PreExposure());

    const FfxFloat32x3 fPreparedYCoCg = RGBToYCoCg(fRgb);

    return fPreparedYCoCg;
}

FfxFloat32 EvaluateSurface(FfxInt32x2 iPxPos, FfxFloat32x2 fMotionVector)
{
    FfxFloat32 d0 = GetViewSpaceDepth(LoadReconstructedPrevDepth(iPxPos + FfxInt32x2(0, -1)));
    FfxFloat32 d1 = GetViewSpaceDepth(LoadReconstructedPrevDepth(iPxPos + FfxInt32x2(0, 0)));
    FfxFloat32 d2 = GetViewSpaceDepth(LoadReconstructedPrevDepth(iPxPos + FfxInt32x2(0, 1)));

    return 1.0f - FfxFloat32(((d0 - d1) > (d1 * 0.01f)) && ((d1 - d2) > (d2 * 0.01f)));
}

void DepthClip(FfxInt32x2 iPxPos)
{
    FfxFloat32x2 fDepthUv = (iPxPos + 0.5f) / RenderSize();
    FfxFloat32x2 fMotionVector = LoadDilatedMotionVector(iPxPos);

    // Discard tiny mvs
    fMotionVector *= FfxFloat32(length(fMotionVector * DisplaySize()) > 0.01f);

    const FfxFloat32x2 fDilatedUv = fDepthUv + fMotionVector;
    const FfxFloat32 fDilatedDepth = LoadDilatedDepth(iPxPos);
    const FfxFloat32 fCurrentDepthViewSpace = GetViewSpaceDepth(LoadInputDepth(iPxPos));

    // Compute prepared input color and depth clip
    FfxFloat32 fDepthClip = ComputeDepthClip(fDilatedUv, fDilatedDepth) * EvaluateSurface(iPxPos, fMotionVector);
    FfxFloat32x3 fPreparedYCoCg = ComputePreparedInputColor(iPxPos);
    StorePreparedInputColor(iPxPos, FfxFloat32x4(fPreparedYCoCg, fDepthClip));

    // Compute dilated reactive mask
#if FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS
    FfxInt32x2 iSamplePos = iPxPos;
#else
    FfxInt32x2 iSamplePos = ComputeHrPosFromLrPos(iPxPos);
#endif

    FfxFloat32 fMotionDivergence = ComputeMotionDivergence(iSamplePos, RenderSize());
    FfxFloat32 fTemporalMotionDifference = ffxSaturate(ComputeTemporalMotionDivergence(iPxPos) - ComputeDepthDivergence(iPxPos));

    PreProcessReactiveMasks(iPxPos, ffxMax(fTemporalMotionDifference, fMotionDivergence));
}

#endif //!defined( FFX_FSR2_DEPTH_CLIPH )
