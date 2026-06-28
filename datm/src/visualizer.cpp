#include "visualizer.h"

void Visualizer::drawPanelTitles(cv::Mat& originalFrame, cv::Mat& foregroundMask,
                                    cv::Mat& trackerOutput) const {
    // --- original image panel
    cv::putText(originalFrame,
        "Original Frame",
        {20, 40},
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        {0, 255, 0},
        2);

    // --- foreground mask panel
    cv::putText(foregroundMask,
        "Foreground Mask",
        {20, 40},
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        {0, 255, 0},
        2);

    // --- final tracker output panel
    cv::putText(trackerOutput,
        "cuMoT Tracker Output",
        {20, 40},
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        {0, 255, 0},
        2);
}

void Visualizer::drawMetrics(cv::Mat& trackerOutput, const DebugMetrics& metrics) const {

    const cv::Scalar color(0,255,255);
    constexpr double scale = 1.0;
    constexpr int thick = 2;


    cv::putText(trackerOutput,
                "Frame: " + std::to_string(metrics.frameNumber),
                {20,80},
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick);

    cv::putText(trackerOutput,
                "FPS: " + std::to_string(metrics.fps).substr(0,5),
                {20,120},
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick);

    cv::putText(trackerOutput,
                "Infer: " + std::to_string(metrics.inferenceMs).substr(0,5) + " ms",
                {20,160},
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick);

    cv::putText(trackerOutput,
                "Tracker: " + std::to_string(metrics.trackerMs).substr(0,5) + " ms",
                {20,200},
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick);

    cv::putText(trackerOutput,
                "Detector: " + std::to_string(metrics.detectorCount),
                {20,240},
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick);

    cv::putText(trackerOutput,
                "Motion: " + std::to_string(metrics.motionCount),
                {20,280},
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick);

    cv::putText(trackerOutput,
                "Fused: " + std::to_string(metrics.fusedCount),
                {20,320},
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick);

    cv::putText(trackerOutput,
                "Tracks: " + std::to_string(metrics.activeTracks),
                {20,360},
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick);

    cv::putText(trackerOutput,
                "FG %: " + std::to_string(metrics.foregroundPercent).substr(0,5),
                {20,400},
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick);
}

cv::Mat Visualizer::createDebugView(cv::Mat originalFrame, cv::Mat foregroundMask,
                                        cv::Mat trackerOutput,
                                        const DebugMetrics& metrics) const {
    // --- convert mask to BGR for side-by-side display
    if (foregroundMask.channels() == 1) {
        cv::cvtColor(foregroundMask,
            foregroundMask,
            cv::COLOR_GRAY2BGR);
    }

    // --- ensure dimensions match
    cv::resize(foregroundMask, foregroundMask, originalFrame.size());
    cv::resize(trackerOutput, trackerOutput, originalFrame.size());

    // --- add labels and metrics
    drawPanelTitles(originalFrame, foregroundMask, trackerOutput);
    drawMetrics(trackerOutput, metrics);

    // --- combine panels
    cv::Mat leftPair;
    cv::Mat combined;

    cv::hconcat(originalFrame, foregroundMask,leftPair);
    cv::hconcat(leftPair, trackerOutput, combined);

    return combined;
}