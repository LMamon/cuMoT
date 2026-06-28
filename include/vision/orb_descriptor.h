#pragma once

#include <cstdint>

#include "association/appearance_model.h"

struct AppearanceCudaScratch {
    uint8_t* d_trackDescriptors = nullptr;
    uint8_t* d_detectionDescriptors = nullptr;

    int* d_trackCounts = nullptr;
    int* d_detectionCounts = nullptr;

    float* d_appearanceCost = nullptr;

    int maxTracks = 0;
    int maxDetections = 0;
};

void initializeAppearanceCudaScratch(AppearanceCudaScratch& scratch,
                                    int maxTracks,
                                    int maxDetections);

void destroyAppearanceCudaScratch(AppearanceCudaScratch& scratch);

// generate an appearance cost matrix.
void computeAppearanceCostMatrixCUDA(AppearanceCudaScratch& scratch,
                                    const AppearanceModel* tracks,
                                    int numTracks,

                                    const AppearanceModel* detections,
                                    int numDetections,

                                    float* h_appearanceCost);

float computeAppearanceSimilarity(AppearanceCudaScratch& scratch, 
                                    const AppearanceModel& a,
                                    const AppearanceModel& b);

float computeAppearanceSimilarityCUDA(AppearanceCudaScratch& scratch, 
                                    const AppearanceModel& a,
                                    const AppearanceModel& b);