#pragma once
#include <cstdint>

__global__ void appearance_cost_matrix_kernel(
    const uint8_t* trackDescriptors,
    const int* trackCounts,

    const uint8_t* detectionDescriptors,
    const int* detectionCounts,

    int numTracks,
    int numDetections,

    float* appearanceCost);