#include "tracking/tracker.h"
#include "association/data_association.h"
#include "association/appearance_model.h"
#include "tracking/kalman_filter.h"
#include "association/iou.h"

#include <opencv2/core.hpp>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

const std::vector<Track>& Tracker::getTracks() const { return tracks; }

int Tracker::getConfirmedTrackCount() const {
    return std::count_if(tracks.begin(), tracks.end(), [](const Track& track) {
                    
            return track.state == TrackState::Confirmed;
        });
}

Tracker::Tracker(ORBContext& orbCtx) : orbCtx_(orbCtx) {
    initializeAppearanceCudaScratch(appearanceScratch_,
                                    1024,   // max tracks
                                    512);  // max detections
}

void Tracker::createTrack(const Detection& detection, const cv::Mat& frame) {
    Track track;

    track.trackId = nextTrackId++;
    track.lastDetection = detection;
    track.age = 1;
    track.confidence = detection.confidence;
    track.state = TrackState::Tentative;
    track.framesSinceUpdate = 0;
    track.hits = 1;
    track.kf.init(detection.box);

    track.appearance = detection.appearance;

    tracks.push_back(track);
}

void Tracker::update(const std::vector<Detection>& detections, const cv::Mat& frame) {
    if (tracks.empty() && lostTracks.empty()) {
        for (const auto& d : detections) {
            createTrack(d, frame);
        }
        return;
    }

    predictTracks();
    AssociationResult result = associate(tracks, detections, appearanceScratch_);

    // update matched tracks
    updateMatchedTracks(result, detections);

    // update unmatched tracks
    updateUnmatchedTracks(result);
    
    for (int detectionIndex : result.unmatchedDetections) {
        const Detection& d = detections[detectionIndex];
        if (!recoverLostTrack(d)) createTrack(d, frame);
    }

    moveLostTracks();
    removeDeadTracks();
}

void Tracker::predictTracks() {
    for (auto& track : tracks) {
        track.age++;
        if (track.framesSinceUpdate <= 5) {
            track.lastDetection.box = track.kf.predict();
        }
    }

    for(auto& track : lostTracks) {
        track.framesSinceUpdate++;
        if (track.framesSinceUpdate <= 10) {
            track.lastDetection.box = track.kf.predict();
        }
    }
}

void Tracker::updateTrackFromDetection(Track& track, const Detection& detection) {
    
    if (!isMeasurementValid(track, detection)) return;

    track.framesSinceUpdate = 0;
    track.hits++;
    track.kf.update(detection.box);
    track.lastDetection = detection;
    track.confidence = std::min(1.0f, track.confidence + detection.confidence);
    track.appearance = detection.appearance;
    
    if (track.state == TrackState::Tentative && track.hits >= ConfirmationHits) {
        track.state = TrackState::Confirmed;
    }
}

void Tracker::updateMatchedTracks(const AssociationResult& result, const std::vector<Detection>& detections) {
    // matched track receives a new detection
    for (const auto& match : result.matches) {
        Track& track = tracks[match.trackIndex];

        updateTrackFromDetection(track, detections[match.detectionIndex]);
        if (track.state == TrackState::Tentative && track.hits > ConfirmationHits) {
            track.state = TrackState::Confirmed;
        }
    }
}

void Tracker::updateUnmatchedTracks(const AssociationResult& result) {
    // track was not assigned a detection
    for (int trackIndex : result.unmatchedTracks) {
        Track& track = tracks[trackIndex];

        track.framesSinceUpdate++;

        track.confidence = std::max(0.0f, track.confidence - 0.15f);

        if (track.state == TrackState::Tentative && track.framesSinceUpdate > TentativeMaxMisses) {
            track.state = TrackState::Removed;
        }
        
        if (track.state == TrackState::Confirmed && track.framesSinceUpdate > ConfirmedMaxMisses) {
            track.state = TrackState::Lost;
        }
    }
}

void Tracker::moveLostTracks() {
    auto it = tracks.begin();

    while (it != tracks.end()) {
        if (it->state == TrackState::Lost) {
            it->framesSinceUpdate = 0;
            lostTracks.push_back(std::move(*it));

            it = tracks.erase(it);
        } else {
            ++it;
        }
    }
}

bool Tracker::recoverLostTrack(const Detection& detection) {
    int bestIndex = -1;
    float bestScore = 0.0f;

    for (size_t i = 0; i < lostTracks.size(); ++i) { // search wont scale so 
        Track& track = lostTracks[i];

        float appearance = computeAppearanceSimilarityCUDA(appearanceScratch_,
                                                        track.appearance,
                                                        detection.appearance);
        
        float spatial = computeIoU(track.lastDetection.box, detection.box);
        float score = RecoveryAppearanceWeight * appearance +
                     RecoveryMotionWeight * spatial;

        if (score > bestScore) {
            bestScore = score;
            bestIndex = static_cast<int>(i);
        }
    }

    if (bestIndex == -1 || bestScore < RecoveryThreshold) return false;
    Track recovered = std::move(lostTracks[bestIndex]);

    updateTrackFromDetection(recovered, detection);
    
    recovered.state = TrackState::Confirmed;

    tracks.push_back(std::move(recovered));
    lostTracks.erase(lostTracks.begin() + bestIndex);

    return true;
}

// this gotta go
bool Tracker::isMeasurementValid(const Track& track, const Detection& detection) {
    float iou = computeIoU(track.lastDetection.box, detection.box);
    float previousArea = track.lastDetection.box.w * track.lastDetection.box.h;
    float detectionArea = detection.box.w * detection.box.h;

    float areaRatio = detectionArea / std::max(previousArea, 1.0f);

    float dx = detection.box.cx - track.lastDetection.box.cx;
    float dy = detection.box.cy - track.lastDetection.box.cy;
    
    float centerDistance = std::sqrt(dx * dx + dy * dy);

    return iou > 0.10f && areaRatio > 0.4f && areaRatio < 2.5f && centerDistance < 100.0f;
}

// CAN DELETE
void Tracker::updateTrackAppearance(Track& track, const Detection& detection, const cv::Mat& frame) {
    const Box& b = detection.box;

    int x = std::max(0, static_cast<int>(left(b)));
    int y = std::max(0, static_cast<int>(top(b)));
    int w = static_cast<int>(b.w);
    int h = static_cast<int>(b.h);

    if (x >= frame.cols || y >= frame.rows || w <= 0 || h <= 0)
        return;

    w = std::min(w, frame.cols - x);
    h = std::min(h, frame.rows - y);

    if (w <= 4 || h <= 4)
        return;

    // Crop the detection region from the current frame.
    // This gives VPI ORB actual pixel data, not just the bbox.
    cv::Rect roiRect(x, y, w, h);
    cv::Mat roi = frame(roiRect);

    // Populate the track's appearance descriptor cache.
    buildAppearanceModelFromROI(orbCtx_, roi, track.appearance);
}


void Tracker::removeDeadTracks() {
    lostTracks.erase(std::remove_if(lostTracks.begin(), lostTracks.end(),
                                    // select tracks that should be removed
                                    [this](const Track& t) {

            return t.framesSinceUpdate > LostTrackBufferFrames;
        }),
    lostTracks.end());
}

void Tracker::drawTracks(cv::Mat& outputFrame) const {
    for (const auto& track : tracks) {
        if (track.state != TrackState::Confirmed) //showing only confirmed tracks
            continue;
        cv::Scalar color = cv::Scalar(0, 255, 0);

        const Detection& d = track.lastDetection;

        int cx1 = static_cast<int>(d.box.cx - d.box.w * 0.5f);
        int cy1 = static_cast<int>(d.box.cy - d.box.h * 0.5f);

        int cx2 = static_cast<int>(d.box.cx + d.box.w * 0.5f);
        int cy2 = static_cast<int>(d.box.cy + d.box.h * 0.5f);

        cv::rectangle(outputFrame,
            cv::Point(cx1, cy1),
            cv::Point(cx2, cy2),
            color,
            2);

        std::ostringstream labelStream;
        labelStream << track.trackId;
                    // << "::"
                    // << track.appearance.descriptorCount
                    // << " "
                    // << std::fixed
                    // << std::setprecision(2)
                    // << d.confidence
                    // << " ORB:"
                    // << track.appearance.descriptorCount;

        cv::putText(outputFrame,
            labelStream.str(),
            cv::Point(cx1, cy1 - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2);
    }
}
