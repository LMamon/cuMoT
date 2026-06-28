#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "association/data_association.h"
#include "association/appearance_model.h"
#include "tracking/tracker.h"
#include "detection/detector_engine.h"
#include "tracking/kalman_filter.h"
#include "vision/orb_descriptor.h"


enum class TrackState {
    Tentative,
    Confirmed,
    Lost,
    Removed
};

struct Track {
    TrackState state;
    float confidence = 0.0f;

    Detection lastDetection;
    KalmanFilterCV kf;

    AppearanceModel appearance;

    int age = 0;
    int framesSinceUpdate = 0;
    int hits = 1;
    int trackId;
};


class Tracker {
    public:
        explicit Tracker(ORBContext& orbCtx);
        ~Tracker() { destroyAppearanceCudaScratch(appearanceScratch_); }

        void update(const std::vector<Detection>& detections, const cv::Mat& frame);
        const std::vector<Track>& getTracks() const;
        int getConfirmedTrackCount() const;
        void drawTracks(cv::Mat& outputFrame) const;

    private:
        void createTrack(const Detection& detection, const cv::Mat& frame);
        void updateTrackAppearance(Track& track, const Detection& detection, const cv::Mat& frame);

    private:
        static constexpr int TentativeMaxMisses = 6;
        static constexpr int ConfirmedMaxMisses = 50;
        static constexpr int LostTrackBufferFrames = 250;

        static constexpr int ConfirmationHits = 15;
        static constexpr float RecoveryAppearanceWeight = 0.8f;
        static constexpr float RecoveryMotionWeight = 0.3f;
        static constexpr float RecoveryThreshold = 0.4f;

        int nextTrackId = 0;

        AppearanceCudaScratch appearanceScratch_;
        ORBContext& orbCtx_;

        std::vector<Track> tracks;
        std::vector<Track> lostTracks;

    private:
        void predictTracks();
        
        void updateUnmatchedTracks(const AssociationResult& result);
        void updateMatchedTracks(const AssociationResult& result, const std::vector<Detection>& detections);
        void updateTrackFromDetection(Track& track, const Detection& detection);

        void moveLostTracks();
        bool recoverLostTrack(const Detection& detection);
        bool isMeasurementValid(const Track& track, const Detection& detection);
        void removeDeadTracks();
};
