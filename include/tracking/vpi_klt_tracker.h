#pragma once

#include <unordered_map>
#include <vector>
#include <vpi/VPI.h>
#include <vpi/algo/KLTFeatureTracker.h>
#include <opencv2/opencv.hpp>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/OpenCVInterop.hpp>

#include "tracking/local_tracker.h"

class VPIKLTLocalTracker : public ILocalTracker {
    public:
        VPIKLTLocalTracker();
        ~VPIKLTLocalTracker() override;

        bool initialize(const cv::Size& frameSize) override;
        bool addTrack(int trackId, const Box& initialBox) override;
        void initializeIdentityPrediction(VPIHomographyTransform2D& prediction);

        void removeTrack(int trackId) override;

        std::vector<LocalTrackResult> update(const std::vector<LocalTrackInput>& tracks, 
                                            const cv::Mat&frame) override;

        void reset() override;
        
    private:
        bool ensureInitialized(const cv::Mat& frame);
        void rebuildInputArrays(const std::vector<LocalTrackInput>& tracks);
        
    private:
        VPIBackend backend_ = VPI_BACKEND_CUDA;
        VPIStream stream_ = nullptr;
        VPIPayload payload_ = nullptr;
        
        VPIImage previousFrame_ = nullptr;
        VPIImage currentFrame_ = nullptr;
        
        VPIArray inputBoxes_ = nullptr;
        VPIArray inputPredictions_ = nullptr;
        
        VPIArray outputBoxes_ = nullptr;
        VPIArray outputEstimations_ = nullptr;
        
        VPIKLTFeatureTrackerParams params_{};
        
        int width_ = 0;
        int height_ = 0;
        bool initialized_ = false;
        static constexpr int MAX_TRACKS = 256;
        
        std::vector<VPIKLTTrackedBoundingBox> inputBoxData_;
        std::vector<VPIHomographyTransform2D> inputPredictionData_;

        std::unordered_map<int, int> trackIdToSlot_;
};
