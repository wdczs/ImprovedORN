#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RotationInvariantEncoding.c"
#else

int orn_(RIE_AlignFeature)(
    THTensor *feature, //128 x 640 x 1 x 1
    THByteTensor *mainDirection, // None -> 128 x 80
    THTensor *aligned, // 128 x 640 x 1 x 1
    const uint8 nOrientation) // 8
{
    THArgCheck(feature->nDimension == 4, 1, "only supports batch mode.");

    const uint16 nBatch = feature->size[0];// 128
    const uint16 nChannel = feature->size[1];// 640
    const uint16 nFeature = nChannel / nOrientation;// 80
    THArgCheck(feature->size[2] == 1 && feature->size[3] == 1, 1, "mH x mW should be 1x1.");

    THByteTensor_resize2d(mainDirection, nBatch, nFeature);
    THTensor_(resizeAs)(aligned, feature);

    real *feature_data = THTensor_(data)(feature);
    uint8 *mainDirection_data = THByteTensor_data(mainDirection);
    real *aligned_data = THTensor_(data)(aligned);

    uint16 i;
    uint16 j;
    uint8 l;

    #pragma omp parallel for private(i, j, l)
    for (i = 0; i < nBatch; i++) {// 128
        for (j = 0; j < nFeature; j++) { // nChannel / nOrientation = 640 / 8 = 80
            uint8 *direction = mainDirection_data + i * nFeature + j;
            real maxVal = -THInf;
            for (l = 0; l < nOrientation; l++) {
                real val = *(feature_data + i * (nFeature * nOrientation)
                                          + j * (nOrientation)
                                          + l);
                if (val > maxVal) {
                    maxVal = val;
                    *direction = l;
                }
            }
            for (l = 0; l < nOrientation; l++) {
                real src = *(feature_data + i * (nFeature * nOrientation)
                                          + j * (nOrientation)
                                          + l);
                uint8 alignedIndex = (l - (uint8)*direction + nOrientation) % nOrientation;
                real *target = aligned_data + i * (nFeature * nOrientation)
                                            + j * (nOrientation)
                                            + alignedIndex;
                *target = src;
            }
        }
    }

    return 1;
}

int orn_(RIE_UnAlignFeature)(
    THTensor *feature,
    THByteTensor *mainDirection,
    THTensor *aligned,
    const uint8 nOrientation)
{
    const uint16 nBatch = mainDirection->size[0];
    const uint16 nFeature = mainDirection->size[1];

    THTensor_(resizeAs)(feature, aligned);

    real *feature_data = THTensor_(data)(feature);
    uint8 *mainDirection_data = THByteTensor_data(mainDirection);
    real *aligned_data = THTensor_(data)(aligned);

    uint16 i;
    uint16 j;
    uint8 l;

    #pragma omp parallel for private(i, j, l)
    for (i = 0; i < nBatch; i++) { // 128
        for (j = 0; j < nFeature; j++) {  // 80
            uint8 direction = *(mainDirection_data + i * nFeature + j);
            for (l = 0; l < nOrientation; l++) {
                real src = *(aligned_data + i * (nFeature * nOrientation)
                                          + j * (nOrientation)
                                          + l);
                uint8 alignedIndex = (l + direction) % nOrientation;
                real *target = feature_data + i * (nFeature * nOrientation)
                                            + j * (nOrientation)
                                            + alignedIndex;
                *target = src;
            }
        }
    }
    
    return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
int orn_(RIE_AlignFeature2d)(
    THTensor *feature, //128 x 640 x 1 x 1
    THByteTensor *mainDirection, // None -> 128 x 80
    THTensor *aligned, // 128 x 640 x 1 x 1
    const uint8 nOrientation) // 8
{
    THArgCheck(feature->nDimension == 4, 1, "only supports batch mode.");

    const uint16 nBatch = feature->size[0];// 128
    const uint16 nChannel = feature->size[1];// 640
    const uint16 nFeature = nChannel / nOrientation;// 80
    const uint16 feature_h = feature->size[2];
    const uint16 feature_w = feature->size[3];
    // THArgCheck(feature->size[2] == 1 && feature->size[3] == 1, 1, "mH x mW should be 1x1.");

    THByteTensor_resize2d(mainDirection, nBatch, nFeature);
    THTensor_(resizeAs)(aligned, feature);

    real *feature_data = THTensor_(data)(feature);
    uint8 *mainDirection_data = THByteTensor_data(mainDirection);
    real *aligned_data = THTensor_(data)(aligned);

    uint16 i;
    uint16 j;
    uint16 p;
    uint8 l;
    real* target;

    #pragma omp parallel for private(i, j, l)
    for (i = 0; i < nBatch; i++) {// 128
        for (j = 0; j < nFeature; j++) { // nChannel / nOrientation = 640 / 8 = 80
            uint8 *direction = mainDirection_data + i * nFeature + j;
            real maxVal = -THInf;
            for (l = 0; l < nOrientation; l++) {
                real val = 0;
                for (p = 0; p < feature_h * feature_w; p++){
                    val = val + *((feature_data + i * (nFeature * nOrientation * feature_h * feature_w)
                                           + j * (nOrientation * feature_h * feature_w)
                                           + l * feature_h * feature_w) + p);
                }
                if (val > maxVal) {
                    maxVal = val;
                    *direction = l;
                }
            }
            for (l = 0; l < nOrientation; l++) {
                uint8 alignedIndex = ((l - (uint8)*direction) + nOrientation) % nOrientation;
                target = aligned_data + i * (nFeature * nOrientation * feature_h * feature_w)
                                             + j * (nOrientation * feature_h * feature_w)
                                             + alignedIndex * feature_h * feature_w;
                for (p = 0; p < feature_h * feature_w; p++){
                    *(target+p) = *((feature_data + i * (nFeature * nOrientation * feature_h * feature_w)
                                           + j * (nOrientation * feature_h * feature_w)
                                           + l * feature_h * feature_w)+p);
                }
            }
        }
    }

    return 1;
}

int orn_(RIE_UnAlignFeature2d)(
    THTensor *feature, //128 x 640 x 1 x 1
    THByteTensor *mainDirection, // None -> 128 x 80
    THTensor *aligned, // 128 x 640 x 1 x 1
    const uint8 nOrientation) // 8
{
    const uint16 nBatch = aligned->size[0];// 128
    const uint16 nChannel = aligned->size[1];// 640
    const uint16 nFeature = nChannel / nOrientation;// 80
    const uint16 feature_h = aligned->size[2];
    const uint16 feature_w = aligned->size[3];
    // THArgCheck(feature->size[2] == 1 && feature->size[3] == 1, 1, "mH x mW should be 1x1.");

    // THByteTensor_resize2d(mainDirection, nBatch, nFeature);
    THTensor_(resizeAs)(feature, aligned);

    real *feature_data = THTensor_(data)(feature);
    uint8 *mainDirection_data = THByteTensor_data(mainDirection);
    real *aligned_data = THTensor_(data)(aligned);

    uint16 i;
    uint16 j;
    uint16 p;
    uint8 l;

    #pragma omp parallel for private(i, j, l)
    for (i = 0; i < nBatch; i++) { // 128
        for (j = 0; j < nFeature; j++) {  // 80
            uint8 direction = *(mainDirection_data + i * nFeature + j);
            for (l = 0; l < nOrientation; l++) {
                for (p = 0; p < feature_h * feature_w; p++){
                    real src = *(aligned_data + i * (nFeature * nOrientation * feature_h * feature_w)
                                              + j * (nOrientation * feature_h * feature_w)
                                              + l * feature_h * feature_w
                                              + p);
                    uint8 alignedIndex = (l + direction) % nOrientation;
                    real *target = feature_data + i * (nFeature * nOrientation * feature_h * feature_w)
                                                + j * (nOrientation * feature_h * feature_w)
                                                + alignedIndex * feature_h * feature_w
                                                + p;
                    *target = src;                   
                }
            }
        }
    }
    return 1;
}
// int orn_(RIE_UnAlignFeature2d)(
//     THTensor *feature,
//     THByteTensor *mainDirection,
//     THTensor *aligned,
//     const uint8 nOrientation)
// {
//     const uint16 nBatch = mainDirection->size[0];
//     const uint16 nFeature = mainDirection->size[1];
//     const uint16 feature_h = feature->size[2];
//     const uint16 feature_w = feature->size[3];

//     THTensor_(resizeAs)(feature, aligned);

//     real *feature_data = THTensor_(data)(feature);
//     uint8 *mainDirection_data = THByteTensor_data(mainDirection);
//     real *aligned_data = THTensor_(data)(aligned);

//     uint16 i;
//     uint16 j;
//     uint16 p;
//     uint8 l;
//     real* target;

//     #pragma omp parallel for private(i, j, l)
//     for (i = 0; i < nBatch; i++) { // 128
//         for (j = 0; j < nFeature; j++) {  // 80
//             uint8 direction = *(mainDirection_data + i * nFeature + j);
//             for (l = 0; l < nOrientation; l++) {
//                 uint8 alignedIndex = (l + direction) % nOrientation;
//                 target = feature_data + i * (nFeature * nOrientation * feature_h * feature_w)
//                                            + j * (nOrientation * feature_h * feature_w)
//                                            + alignedIndex * feature_h * feature_w;
//                 for (p = 0; p < feature_h * feature_w; p++)
//                     *(target+p) = *((aligned_data + i * (nFeature * nOrientation * feature_h * feature_w)
//                                              + j * (nOrientation * feature_h * feature_w)
//                                              + l * feature_h * feature_w)+p);
//             }
//         }
//     }
    
//     return 1;
// }

#endif
