#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

class MotionDetector {
    public:
        MotionDetector();
        std::vector<Detection> detections(const cv::Mat& inputFrame);
        const cv::Mat& foregroundMask() const;
        const cv::Mat& cleanedMask() const;

    private:
        cv::Ptr<cv::BackgroundSubtractorMOG2> mog2_;
        int frameCount_ = 0;
        int warmupFrames_ = 100;

        cv::Mat foregroundMask_;
        cv::Mat cleanedMask_;
};