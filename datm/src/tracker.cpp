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
#include <unordered_map>

const std::vector<Track>& Tracker::getTracks() const { return tracks; }

int Tracker::getConfirmedTrackCount() const {
    return std::count_if(tracks.begin(), tracks.end(), [](const Track& track) {
                    
            return track.state == TrackState::Confirmed;
        });
}

Tracker::Tracker(ORBContext& orbCtx, std::unique_ptr<ILocalTracker> localTracker) : orbCtx_(orbCtx), 
                                                                                    localTracker_(std::move(localTracker)) {

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

    updateTrackAppearance(track, detection, frame);

    tracks.push_back(track);
    if (localTrackerInitialized_) {
        tracks.back().localTrackerInitialized = localTracker_->addTrack(tracks.back().trackId,
                                                                        tracks.back().lastDetection.box);
    }
}

void Tracker::detectorUpdate(const std::vector<Detection>& detections, const cv::Mat& frame) {
    // TODO
    AssociationResult result = associate(tracks, detections, appearanceScratch_);

    // update matched tracks
    updateMatchedTracks(result, detections, frame);

    // update unmatched tracks
    updateUnmatchedTracks(result);
    
    for (int detectionIndex : result.unmatchedDetections) {
        const Detection& d = detections[detectionIndex];
        if (!recoverLostTrack(d, frame)) createTrack(d, frame);
    }
}

void Tracker::localTrackerUpdate(const cv::Mat& frame) {
    std::vector<LocalTrackInput> inputs;

    for (const auto& track : tracks) {
        if (track.state == TrackState::Removed) continue;
        if (!track.localTrackerInitialized) continue;

        inputs.push_back(LocalTrackInput{track.trackId, track.lastDetection.box});
    }

    auto results = localTracker_->update(inputs, frame);

    std::unordered_map<int, LocalTrackResult> byId;

    for (const auto& result : results) {
        byId[result.trackId] = result;
    }

    for (auto& track : tracks) {
        if (!track.localTrackerInitialized) continue;
        auto it = byId.find(track.trackId);

        if (it == byId.end() || !it->second.valid) {
            track.framesSinceUpdate++;
            track.localTrackMisses++;
            continue;
        }

        const Box& measuredBox = it->second.box;

        if (!isLocalTrackerMeasurementValid(track, measuredBox, frame.size())) {
            track.framesSinceUpdate++;
            track.localTrackMisses++;
            continue;
        }
            
        track.kf.update(measuredBox);    
        Detection localDetection = track.lastDetection;
        localDetection.box = measuredBox;
        localDetection.source = DetectionSource::LOCAL_TRACKER;
        localDetection.confidence = it->second.confidence;
        
        track.lastDetection = localDetection;
        track.framesSinceUpdate = 0;
        track.localTrackMisses = 0;
    }
}

void Tracker::update(const std::vector<Detection>& detections, const cv::Mat& frame, bool detectorRan) {
    if (!localTrackerInitialized_) {
        localTrackerInitialized_ = localTracker_->initialize(frame.size());
    }

    if (tracks.empty() && lostTracks.empty()) {
        for (const auto& d : detections) {
            createTrack(d, frame);
        }
        return;
    }

    predictTracks();
    if (detectorRan) {
        // detector update
//     2. If detectorRan:
//       - Associate detector detections to tracks.
//       - Correct matched tracks with detector boxes.
//       - Update ORB appearance only for reliable detector matches.
//       - Create new tracks from unmatched detector detections.
//       - Reinitialize or refresh KLT templates for matched/new tracks.
//       - Attempt recovery for lost tracks using detector detections and appearance.
        detectorUpdate(detections, frame);

    } else {
        // klt update
//     3. If detector did not run:
//       - Use VPI KLT to update existing confirmed/tentative tracks.
//       - Validate KLT results against Kalman prediction.
//       - Correct Kalman with valid KLT boxes.
//       - Mark invalid KLT results as missed updates.
//       - Do not create new tracks.
//       - Do not update long-term ORB appearance.
        localTrackerUpdate(frame);
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
    
    if (track.state == TrackState::Tentative && track.hits >= ConfirmationHits) {
        track.state = TrackState::Confirmed;
    }
}

void Tracker::updateMatchedTracks(const AssociationResult& result, const std::vector<Detection>& detections,
                                                                    const cv::Mat& frame) {
    // matched track receives a new detection
    for (const auto& match : result.matches) {
        Track& track = tracks[match.trackIndex];
        const Detection& detection = detections[match.detectionIndex];

        updateTrackFromDetection(track, detection);
        updateTrackAppearance(track, detection, frame);

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
            localTracker_->removeTrack(it->trackId);

            it->framesSinceUpdate = 0;
            lostTracks.push_back(std::move(*it));

            it = tracks.erase(it);
        } else {
            ++it;
        }
    }
}

bool Tracker::recoverLostTrack(const Detection& detection, const cv::Mat& frame) {
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
    Track& track = tracks.back();

    // track.localTrackerInitialized = localTracker_->initialize(track.trackId, track.lastDetection.box);
    track.localTrackerInitialized = localTracker_->addTrack(track.trackId, track.lastDetection.box);
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

// keep, will actually be crutial for klt update
void Tracker::updateTrackAppearance(Track& track, const Detection& detection, const cv::Mat& frame) {
    const Box& b = detection.box;

    int x = std::max(0, static_cast<int>(left(b)));
    int y = std::max(0, static_cast<int>(top(b)));
    int w = static_cast<int>(b.w);
    int h = static_cast<int>(b.h);

    if (x >= frame.cols || y >= frame.rows || w <= 0 || h <= 0) return;

    w = std::min(w, frame.cols - x);
    h = std::min(h, frame.rows - y);

    if (w <= 4 || h <= 4) return;

    // Crop the detection region from the current frame.
    // This gives VPI ORB actual pixel data, not just the bbox.
    cv::Rect roiRect(x, y, w, h);
    cv::Mat roi = frame(roiRect);

    // Populate the track's appearance descriptor cache.
    buildAppearanceModelFromROI(orbCtx_, roi, track.appearance);
}

bool Tracker::isLocalTrackerMeasurementValid(const Track& track, const Box& measuredBox, const cv::Size& frameSize) {
    
    if (measuredBox.w <= 4 || measuredBox.h <= 4) return false;
    if (measuredBox.cx < 0 || measuredBox.cy < 0) return false;
    if (measuredBox.cx >= frameSize.width || measuredBox.cy >= frameSize.height) return false;

    float prevArea = track.lastDetection.box.w * track.lastDetection.box.h;
    float newArea = measuredBox.w * measuredBox.h;
    float areaRatio = newArea / std::max(prevArea, 1.0f);

    float dx = measuredBox.cx - track.lastDetection.box.cx;
    float dy = measuredBox.cy - track.lastDetection.box.cy;
    float centerDistance = std::sqrt(dx * dx + dy * dy);

    return areaRatio > 0.35f && areaRatio < 3.0f && centerDistance < 150.0f;
}

void Tracker::removeDeadTracks() {
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [this](const Track& t) {
        if (t.state != TrackState::Removed) return false;
        if (t.localTrackerInitialized) localTracker_->removeTrack(t.trackId);
        
        return true;
    }),
        tracks.end());

    lostTracks.erase(std::remove_if(lostTracks.begin(), lostTracks.end(), [this](const Track& t) {
        if (t.state != TrackState::Removed) return false;
        if (t.localTrackerInitialized) localTracker_->removeTrack(t.trackId);

        return t.framesSinceUpdate > LostTrackBufferFrames;
    }),
        lostTracks.end());
}

void Tracker::drawTracks(cv::Mat& outputFrame) const {
    for (const auto& track : tracks) {
        if (track.state != TrackState::Confirmed) //showing only confirmed tracks
            continue;
        // if (track.state == TrackState::Removed)
        //     continue;
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
            1);

        std::string state = track.state == TrackState::Tentative ? "T" : track.state == TrackState::Confirmed ? "C" :
                            track.state == TrackState::Lost ? "L" : "R";

        std::string source = d.source == DetectionSource::DETECTOR ? "DET" : d.source == DetectionSource::LOCAL_TRACKER ? "KLT" : "UNK";

        std::ostringstream labelStream;
        labelStream << track.trackId;
                    // << " miss:" << track.framesSinceUpdate
                    // << " hits:" << track.hits;
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