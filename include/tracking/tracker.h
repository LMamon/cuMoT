#pragma once

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "association/appearance_model.h"
#include "detection/detector_engine.h"
#include "tracking/kalman_filter.h"
#include "tracking/local_tracker.h"
#include "tracking/track.h"
#include "vision/orb_descriptor.h"

struct AssociationResult;

class Tracker {
    public:
        Tracker(ORBContext& orbCtx, std::unique_ptr<ILocalTracker> localTracker);
        ~Tracker() { destroyAppearanceCudaScratch(appearanceScratch_); }

        void update(const std::vector<Detection>& detections, const cv::Mat& frame, bool detectorRan);
        const std::vector<Track>& getTracks() const;
        int getConfirmedTrackCount() const;
        void drawTracks(cv::Mat& outputFrame) const;

    private:
        static constexpr int TentativeMaxMisses = 80;
        static constexpr int ConfirmedMaxMisses = 60;
        static constexpr int LostTrackBufferFrames = 120;

        static constexpr int ConfirmationHits = 7;
        static constexpr float RecoveryAppearanceWeight = 0.7f;
        static constexpr float RecoveryMotionWeight = 0.3f;
        static constexpr float RecoveryThreshold = 0.2f;

        int nextTrackId = 0;

        AppearanceCudaScratch appearanceScratch_;
        ORBContext& orbCtx_;
        std::unique_ptr<ILocalTracker> localTracker_;
        bool localTrackerInitialized_ = false;

        std::vector<Track> tracks;
        std::vector<Track> lostTracks;

    private:
        void detectorUpdate(const std::vector<Detection>& detections, const cv::Mat& frame);

        void localTrackerUpdate(const cv::Mat& frame);

        void predictTracks();

        void createTrack(const Detection& detection, const cv::Mat& frame);
        void updateTrackAppearance(Track& track, const Detection& detection, const cv::Mat& frame);

        void updateMatchedTracks(const AssociationResult& result, const std::vector<Detection>& detections, const cv::Mat& frame);
        void updateUnmatchedTracks(const AssociationResult& result);
        void updateTrackFromDetection(Track& track, const Detection& detection);

        void moveLostTracks();
        void removeDeadTracks();

        bool recoverLostTrack(const Detection& detection, const cv::Mat& frame);
        bool isMeasurementValid(const Track& track, const Detection& detection);
        bool isLocalTrackerMeasurementValid(const Track& track, const Box& measuredBox, const cv::Size& frameSize);
};
