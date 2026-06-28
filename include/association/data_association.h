#pragma once

#include <vector>
#include "vision/orb_descriptor.h"
#include "tracking/tracker.h"
#include "detection/detector_engine.h"

// some renaming should be done during refactor
struct Match {
    int trackIndex; // maybe rename to lhsIndex
    int detectionIndex; // maybe rename to rhsIndex
    float score;
};

struct CandidateMatch {
    size_t trackIndex;
    size_t detectionIndex;
    float score;
};

struct AssociationResult {
    std::vector<Match> matches;
    std::vector<int> unmatchedTracks;
    std::vector<int> unmatchedDetections;
};

AssociationResult associate(
    const std::vector<Track>& tracks,
    const std::vector<Detection>& detections,
    AppearanceCudaScratch& appearanceScratch);

AssociationResult associate(
    const std::vector<Box>& objBoxes,
    const std::vector<Box>& motBoxes);
