#pragma once

#include <vector>
#include "detection/detector_engine.h"
#include "tracking/tracker.h"

float computeIoU(const Box& a, const Box& b);

float computeIoU(const Track& track, const Detection& detection);

std::vector<float> computeIoUMatrix(const std::vector<Box>& tracks, 
                                    const std::vector<Box>& detections);

std::vector<float> computeIoUMatrix(const std::vector<Track>& tracks, 
                                    const std::vector<Detection>& detections);


std::vector<float> computeIoUMatrixGPU(const std::vector<Box>& tracks,
                                        const std::vector<Box>& detections);