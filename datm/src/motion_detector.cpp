// 
// only enable for datasets/feed with static cameras due to background subtracting
// 
#include <vector>

#include "detection/detector_engine.h"
#include "vision/motion_detector.h"


MotionDetector::MotionDetector() : mog2_(cv::createBackgroundSubtractorMOG2(70, 16.0, false)) {
}

const cv::Mat& MotionDetector::foregroundMask() const { return foregroundMask_; }
const cv::Mat& MotionDetector::cleanedMask() const { return cleanedMask_; }

std::vector<Detection> MotionDetector::detections(const cv::Mat& inputFrame) {
    using Clock = std::chrono::high_resolution_clock;

    auto mog2Start = Clock::now();
    auto mog2End = mog2Start;

    auto morphStart = Clock::now();
    auto morphEnd = morphStart;

    auto CCStart = Clock::now();
    auto CCEnd = CCStart;

    auto createDetectionsStart = Clock::now();
    auto createDetectionsEnd = createDetectionsStart;


    frameCount_++;
    std::vector<Detection> detections;
    foregroundMask_.release();
    cleanedMask_.release();

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    mog2Start = Clock::now();
    mog2_->apply(inputFrame, foregroundMask_);
    mog2End = Clock::now();
    
    if (frameCount_ < warmupFrames_) return {};

    // cleaned foreground mask
    cv::Mat cleanedMask;

    // 3x3 kernel

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    morphStart = Clock::now();
    // remove small foreground noise and holes in blobs
    cv::morphologyEx(foregroundMask_, cleanedMask_, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(cleanedMask_, cleanedMask_, cv::MORPH_CLOSE, kernel);
    morphEnd = Clock::now();

    CCStart = Clock::now();
    // connected components
    int num_components = cv::connectedComponentsWithStats(
                            cleanedMask_, 
                            labels, 
                            stats, 
                            centroids, 
                            4, 
                            CV_32S);
    CCEnd = Clock::now();
    // TODO: Phase 5b:
    // Replace CPU CC w stats with CUDA CC w stats
    // int num_components = connectedComponentsWtihStats();
    
    createDetectionsStart = Clock::now();
    for (int i = 1; i < num_components; ++i) {
        int cx = stats.at<int>(i, cv::CC_STAT_LEFT);
        int cy = stats.at<int>(i, cv::CC_STAT_TOP);

        int width  = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area < 500) continue;

        Detection det;

        det.box.cx = static_cast<float>(cx) / inputFrame.cols;
        det.box.cy = static_cast<float>(cy) / inputFrame.rows;

        det.box.w = static_cast<float>(width) / inputFrame.cols;
        det.box.h = static_cast<float>(height) / inputFrame.rows;

        det.confidence = 0.25f;
        det.classId = -1;
        det.source = DetectionSource::MOTION;

        detections.push_back(det);
    }
    createDetectionsEnd = Clock::now();

    std::cout
            << "Mog2            : "
            << std::chrono::duration<double, std::milli>(mog2End - mog2Start).count()
            << " ms\n"
            << "morph           : "
            << std::chrono::duration<double, std::milli>(morphEnd - morphStart).count()
            << " ms\n"
            << "CC              : "
            << std::chrono::duration<double, std::milli>(CCEnd - CCStart).count()
            << " ms\n"
            << "create motDet   : "
            << std::chrono::duration<double, std::milli>(createDetectionsEnd - createDetectionsStart).count()
            << " ms\n";

    return detections;
}