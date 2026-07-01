#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "detection/detector_engine.h"

struct LocalTrackInput {
    int trackId = -1;
    Box predictedBox{};
};

struct LocalTrackResult {
    int trackId = -1;
    bool valid = false;
    Box box{};
    float confidence = 0.0f;
};


class ILocalTracker {
    public:
        virtual ~ILocalTracker() = default;

        virtual bool initialize(const cv::Size& frameSize) = 0;

        virtual bool addTrack(int trackId, const Box& initialBox)=0;
        virtual std::vector<LocalTrackResult> update(const std::vector<LocalTrackInput>& tracks,
                                                        const cv::Mat& frame)=0;

        virtual void removeTrack(int trackId) = 0;
        virtual void reset() = 0;
};