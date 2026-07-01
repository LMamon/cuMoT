#include <vector>
#include <cstddef>
#include <algorithm>

#include <iostream>
#include "association/data_association.h"
#include "vision/orb_descriptor.h"
#include "association/iou.h"

AssociationResult associate(const std::vector<Track>& tracks, const std::vector<Detection>& detections,
                                                                AppearanceCudaScratch& appearanceScratch) {
    
    AssociationResult result;
    
    if (tracks.empty() || detections.empty()) {
        for (size_t i = 0; i < tracks.size(); i++)
            result.unmatchedTracks.push_back(i);

        for (size_t i = 0; i < detections.size(); i++)
            result.unmatchedDetections.push_back(i);

        return result;
    }
    constexpr float ASSOCIATION_THRESHOLD = 0.30f;
    constexpr float IOU_WEIGHT = 0.65f;
    constexpr float APPEARANCE_WEIGHT = 0.40f;

    size_t numTracks = tracks.size();
    size_t numDetections = detections.size();

    std::vector<Box> trackBoxes;
    std::vector<Box> detectionBoxes;
    
    trackBoxes.reserve(numTracks);
    for (const auto& track : tracks) trackBoxes.push_back(track.lastDetection.box);

    detectionBoxes.reserve(numDetections);
    for (const auto& detection : detections) detectionBoxes.push_back(detection.box);
    
    // compute IoU matrix using cuda
    std::vector<float> IoUMatrix = computeIoUMatrixGPU(trackBoxes, detectionBoxes);
    
    // build appearance vectors
    std::vector<AppearanceModel> trackAppearances;
    trackAppearances.reserve(numTracks);

    for (const auto& track : tracks) trackAppearances.push_back(track.appearance);

    std::vector<AppearanceModel> detectionAppearances;
    detectionAppearances.reserve(numDetections);

    for (const auto& detection : detections) detectionAppearances.push_back(detection.appearance);

    std::vector<float> appearanceCost(numTracks * numDetections);
    computeAppearanceCostMatrixCUDA(appearanceScratch,
                                    trackAppearances.data(),
                                    static_cast<int>(numTracks),

                                    detectionAppearances.data(),
                                    static_cast<int>(numDetections),

                                    appearanceCost.data());

    // build every valid track-detection candidate
    std::vector<CandidateMatch> candidates;
    candidates.reserve(numTracks * numDetections);

    for (size_t t = 0; t < numTracks; t++) {
        for (size_t d = 0; d < numDetections; d++) {
            float iouScore = IoUMatrix[t * numDetections + d];
            float appearanceSimilarity = 1.0f - appearanceCost[t * numDetections + d];

            float score = IOU_WEIGHT * iouScore + APPEARANCE_WEIGHT * appearanceSimilarity;

            // ignore weak associations
            if (score >= ASSOCIATION_THRESHOLD) {
                candidates.push_back({t, d, score});
            }
        }
    }

    // highest confidence matches first
    std::sort(candidates.begin(), candidates.end(), [](const CandidateMatch& a, const CandidateMatch& b) {
        return a.score > b.score;
    });

    // track and detection already assigned
    std::vector<bool> trackUsed(numTracks, false);
    std::vector<bool> detectionUsed(numDetections, false);

    for (const auto& candidate : candidates) {
        // skip already claimed tracks and detections
        if (trackUsed[candidate.trackIndex]) continue;
        if (detectionUsed[candidate.detectionIndex]) continue;

        Match match;

        match.trackIndex = candidate.trackIndex;
        match.detectionIndex = candidate.detectionIndex;
        match.score = candidate.score;

        result.matches.push_back(match);

        trackUsed[candidate.trackIndex] = true;
        detectionUsed[candidate.detectionIndex] = true;
    }
    
    // fill result.unmatchedTracks
    // these tracks were not assigned a detection during
    // the current frame and may need their lost-frame counter incremented by the tracker
    for (size_t t = 0; t <numTracks; t++) {
        if (!trackUsed[t]) result.unmatchedTracks.push_back(t);
    }

    // fill result.unmatchedDetections
    // detections were not assigned to an existing track and may become new tracks
    for (size_t d = 0; d < numDetections; d++) {
        if (!detectionUsed[d]) result.unmatchedDetections.push_back(d);
    }
    return result;
}

AssociationResult associate(const std::vector<Box>& objBoxes, const std::vector<Box>& motBoxes) {
    AssociationResult result;

    if (objBoxes.empty() || motBoxes.empty()) {
        for (size_t i = 0; i < objBoxes.size(); i++)
            result.unmatchedTracks.push_back(i);

        for (size_t i = 0; i < motBoxes.size(); i++)
            result.unmatchedDetections.push_back(i);

        return result;
    }

    const float IOU_THRESHOLD = 0.6f;

    size_t numObjects = objBoxes.size();
    size_t numMotion = motBoxes.size();

    // compute IoU matrix using CUDA
    std::vector<float> scoreMatrix = computeIoUMatrixGPU(objBoxes, motBoxes);

    // find best matches
    // build every valid object-motion candidate
    std::vector<CandidateMatch> candidates;
    candidates.reserve(numObjects * numMotion);

    for (size_t o = 0; o < numObjects; o++) {
        for (size_t m = 0; m < numMotion; m++) {
            float score = scoreMatrix[o * numMotion + m];

            // ignore weak associations
            if (score >= IOU_THRESHOLD) candidates.push_back({o, m, score});
        }
    }

    // highest confidence matches first
    std::sort(candidates.begin(), candidates.end(), [](const CandidateMatch& a, const CandidateMatch& b) {
            return a.score > b.score;
        });

    // objects and motion already assigned
    std::vector<bool> objectUsed(numObjects, false);
    std::vector<bool> motionUsed(numMotion, false);

    // assign highest scoring pairs first + skip already assigned objects and motion
    for (const auto& candidate : candidates) {
        if (objectUsed[candidate.trackIndex]) continue;
        if (motionUsed[candidate.detectionIndex]) continue;

        Match match;
        match.trackIndex = candidate.trackIndex;
        match.detectionIndex = candidate.detectionIndex;
        match.score = candidate.score;

        result.matches.push_back(match);

        // reserve object and motion box
        objectUsed[candidate.trackIndex] = true;
        motionUsed[candidate.detectionIndex] = true;
    }

    // unmatched detector boxes
    for (size_t o = 0; o < numObjects; o++) {
        if (!objectUsed[o]) result.unmatchedTracks.push_back(o);
    }

    // unmatched motion boxes
    for (size_t m = 0; m < numMotion; m++) {
        if (!motionUsed[m]) result.unmatchedDetections.push_back(m);
    }
    return result;
}
