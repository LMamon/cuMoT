// 
// redesign will mostly refactor this
// 
#include <vector>
#include <algorithm>

#include "association/data_association.h"
#include "detection/detection_engine.h"
#include "association/iou.h"

std::vector<Detection> DetectionFuser::merge(const std::vector<Detection>& objDetections,
                                            const std::vector<Detection>& motDetections) {
    
    std::vector<Detection> fused;
    std::vector<Box> objBoxes;
    std::vector<Box> motBoxes;

    objBoxes.reserve(objDetections.size());
    for (const auto& det : objDetections) objBoxes.push_back(det.box);

    motBoxes.reserve(motDetections.size());
    for (const auto& det : motDetections) motBoxes.push_back(det.box);

    AssociationResult assoc = associate(objBoxes, motBoxes);
    
    // matched detector + motion
    for (const auto& match : assoc.matches) {
        Detection det = objDetections[match.trackIndex];
        det.source = DetectionSource::FUSED;

        det.confidence = std::min(1.0f,
                                objDetections[match.trackIndex].confidence +
                                motDetections[match.detectionIndex].confidence);

        fused.push_back(det);
    }

    // detector only
    for (int idx : assoc.unmatchedTracks) fused.push_back(objDetections[idx]);

    // motion only
    for (int idx : assoc.unmatchedDetections) fused.push_back(motDetections[idx]);

    return fused;
};