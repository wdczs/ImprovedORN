#include <stdio.h>
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/RotationInvariantEncoding.cu"
#else

int cuorn_(RIE_AlignFeature)(
    THCTensor *feature,
    THCudaByteTensor *mainDirection,
    THCTensor *aligned,
    const uint8 nOrientation)
{
    uint8 ndim = feature->nDimension;
    THArgCheck(ndim == 4, 1, "only supports batch mode.");

    const uint16 nBatch = feature->size[0];
    const uint16 nChannel = feature->size[1];
    const uint16 nFeature = nChannel / nOrientation;

    THCUNN_assertSameGPU(state, 3, feature, mainDirection, aligned);

    THCudaByteTensor_resize2d(state, mainDirection, nBatch, nFeature);
    THCTensor_(resizeAs)(state, aligned, feature);

    real *feature_data = THCTensor_(data)(state, feature);
    uint8 *mainDirection_data = THCudaByteTensor_data(state, mainDirection);
    real *aligned_data = THCTensor_(data)(state, aligned);

    const uint32 count = nBatch * nFeature;

    kernel_(AlignFeature)(
        THCState_getCurrentStream(state), 
        count, 
        feature_data, 
        nBatch, 
        nFeature, 
        nOrientation, 
        mainDirection_data, 
        aligned_data);
    THCudaCheck(cudaGetLastError());

    return 1;
}

int cuorn_(RIE_UnAlignFeature)(
    THCTensor *feature,
    THCudaByteTensor *mainDirection,
    THCTensor *aligned,
    const uint8 nOrientation)
{
    const uint16 nBatch = mainDirection->size[0];
    const uint16 nFeature = mainDirection->size[1];

    THCUNN_assertSameGPU(state, 3, feature, mainDirection, aligned);

    THCTensor_(resizeAs)(state, feature, aligned);

    real *feature_data = THCTensor_(data)(state, feature);
    uint8 *mainDirection_data = THCudaByteTensor_data(state, mainDirection);
    real *aligned_data = THCTensor_(data)(state, aligned);

    const uint32 count = nBatch * nFeature;

    kernel_(UnAlignFeature)(
        THCState_getCurrentStream(state), 
        count, 
        aligned_data, 
        mainDirection_data, 
        nBatch, 
        nFeature, 
        nOrientation, 
        feature_data);
    THCudaCheck(cudaGetLastError());
    
    return 1;
}

//////////////////////////////////////////////////////////////////////
int cuorn_(RIE_AlignFeature2d)(
    THCTensor *feature,
    THCudaByteTensor *mainDirection,
    THCTensor *aligned,
    const uint8 nOrientation)
{
    uint8 ndim = feature->nDimension;
    THArgCheck(ndim == 4, 1, "only supports batch mode.");

    const uint16 nBatch = feature->size[0];
    const uint16 nChannel = feature->size[1];
    const uint16 feature_h = feature->size[2];
    const uint16 feature_w = feature->size[3];
    const uint16 nFeature = nChannel / nOrientation;

    THCUNN_assertSameGPU(state, 3, feature, mainDirection, aligned);

    THCudaByteTensor_resize2d(state, mainDirection, nBatch, nFeature);
    THCTensor_(resizeAs)(state, aligned, feature);

    real *feature_data = THCTensor_(data)(state, feature);
    uint8 *mainDirection_data = THCudaByteTensor_data(state, mainDirection);
    real *aligned_data = THCTensor_(data)(state, aligned);

    const uint32 count = nBatch * nFeature;

    kernel_(AlignFeature2d)(
        THCState_getCurrentStream(state), 
        count, // count = nBatch * nFeature
        feature_data, // THCTensor_(data)(state, feature) feature: input
        nBatch, // nBatch = feature->size[0]
        nFeature, // nFeature = nChannel / nOrientation
        nOrientation, // 8
        feature_h,
        feature_w,
        mainDirection_data, //THCudaByteTensor_data(state, mainDirection)
        aligned_data); //THCTensor_(data)(state, aligned)
    THCudaCheck(cudaGetLastError());

    return 1;
}

int cuorn_(RIE_UnAlignFeature2d)(
    THCTensor *feature,
    THCudaByteTensor *mainDirection,
    THCTensor *aligned,
    const uint8 nOrientation)
{
    const uint16 nBatch = aligned->size[0];// 128
    const uint16 nChannel = aligned->size[1];// 640
    const uint16 nFeature = nChannel / nOrientation;// 80
    const uint16 feature_h = aligned->size[2];
    const uint16 feature_w = aligned->size[3];   
    THCUNN_assertSameGPU(state, 3, feature, mainDirection, aligned);

    THCTensor_(resizeAs)(state, feature, aligned);
    // printf("2");

    real *feature_data = THCTensor_(data)(state, feature);
    uint8 *mainDirection_data = THCudaByteTensor_data(state, mainDirection);
    real *aligned_data = THCTensor_(data)(state, aligned);

    const uint32 count = nBatch * nFeature;
    // printf("3");

    kernel_(UnAlignFeature2d)(
        THCState_getCurrentStream(state), 
        count, 
        aligned_data, 
        mainDirection_data, 
        nBatch, 
        nFeature, 
        nOrientation,
        feature_h,
        feature_w, 
        feature_data);
    THCudaCheck(cudaGetLastError());
    
    return 1;
}
////////////////////////////////////////////
#endif