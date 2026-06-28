#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// --- metrics displayed on debug output
struct DebugMetrics {
    int frameNumber = 0;

    int detectorCount = 0;
    int motionCount = 0;
    int fusedCount = 0;
    int activeTracks = 0;
    
    int foregroundPixels = 0;
    double foregroundPercent = 0.0;
    
    double inferenceMs = 0.0;
    double trackerMs = 0.0;
    double fps = 0;
};

class Visualizer {
    public:
        // --- build 3-panel debug view
        cv::Mat createDebugView(
            cv::Mat originalFrame,
            cv::Mat foregroundMask,
            cv::Mat trackerOutput,
            const DebugMetrics& metrics) const;

    private:
        // --- panel labels
        void drawPanelTitles(
            cv::Mat& originalFrame,
            cv::Mat& foregroundMask,
            cv::Mat& trackerOutput) const;

        // --- tracker/debug statistics
        void drawMetrics(cv::Mat& trackerOutput, const DebugMetrics& metrics) const;
};
