#include "association/iou.h"
#include "tracking/tracker.h"

#include <algorithm>

float computeIoU(const Box& a, const Box& b) {
    const float interX1 = std::max(left(a), left(b));
    const float interY1 = std::max(top(a), top(b));
    const float interX2 = std::min(right(a), right(b));
    const float interY2 = std::min(bottom(a), bottom(b));

    const float interW = std::max(0.0f, interX2 - interX1);
    const float interH = std::max(0.0f, interY2 - interY1);

    const float intersection = interW * interH;
    float unionArea = (a.w * a.h) + (b.w * b.h) - intersection;

    if (unionArea <= 0.0f) return 0.0f;

    return intersection / unionArea;
}

float computeIoU(const Track& track, const Detection& detection) {
    return computeIoU(track.lastDetection.box, detection.box);
}

std::vector<float> computeIoUMatrix(const std::vector<Track>& tracks, 
                                    const std::vector<Detection>& detections) {
    
    std::vector<float> matrix(tracks.size() * detections.size());

    for (size_t t = 0; t < tracks.size(); ++t) {
        for (size_t d = 0; d < detections.size(); ++d) {
            matrix[t * detections.size() + d] = computeIoU(tracks[t], detections[d]);
        }
    }

    return matrix;
}