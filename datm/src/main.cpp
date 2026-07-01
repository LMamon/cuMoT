#include <opencv2/opencv.hpp>

#include "detection/detector_engine.h"
#include "tracking/tracker.h"
#include "visualizer.h"
#include "association/appearance_model.h"
#include "tracking/vpi_klt_tracker.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// balloon-3 449
// insect-1 200
// stock-0 374 +
// ball-1 114
// stock-3 312 +
// ball-2 198
// boat-2 197

const int DETECTOR_INTERVAL = 3;
const int DETECTOR_WARMUP_FRAMES = 20;
constexpr bool USE_MOG2 = false; // add cli args for this stuff


static void convertDetectorBoxesToPixels(std::vector<Detection>& detections, const cv::Size& frameSize) {
    for (auto& d : detections) {
        d.box.cx *= frameSize.width;
        d.box.cy *= frameSize.height;
        d.box.w  *= frameSize.width;
        d.box.h  *= frameSize.height;
    }
}

int main() {
    std::string modelPath = "...";
    std::string inputFolder = "...";
    std::string outputFolder = "...";

    std::filesystem::create_directories(outputFolder);

    // --- video writer for 3-panel output
    cv::VideoWriter writer;
    bool writerInitialized = false;
    
    double inferenceMs = 0.0;

    DebugMetrics metrics;
    DetectorEngine objDetector;
    ORBContext orbCtx;
    
    if (!objDetector.loadEngine(modelPath)) return -1;

    if (!initializeORBContext(orbCtx)) {
        std::cerr << "Failed to initialize ORB context\n";
        return -1;
    }

    auto localTracker = std::make_unique<VPIKLTLocalTracker>();
    // create tracker
    Tracker tracker(orbCtx, std::move(localTracker));
    Visualizer visualizer;

    for (int i = 0; i < 374; ++i) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        using Clock = std::chrono::high_resolution_clock;

        auto detectorStart = Clock::now();
        auto detectorEnd = detectorStart;

        auto motionStart = Clock::now();
        auto motionEnd = motionStart;

        auto orbStart = Clock::now();
        auto orbEnd = orbStart;

        auto trackerStart = Clock::now();
        auto trackerEnd = trackerStart;

        auto writerStart = Clock::now();
        auto writerEnd = writerStart;

        // make filepath
        std::stringstream iss;
        iss << inputFolder << "/" << std::setw(6) << std::setfill('0') << i << ".jpg";
        std::string inputFilename = iss.str();

        cv::Mat inputFrame = cv::imread(inputFilename);

        if (inputFrame.empty()) {
            std::cerr
            << "end of sequence or failed to load "
            << inputFilename
            << std::endl;
            break;
        }

        // --- run detector every frame during startup,
        // then every DETECTOR_INTERVAL frames afterwards.
        bool detectorRan = (i < DETECTOR_WARMUP_FRAMES) || (i % DETECTOR_INTERVAL == 0);
        std::vector<Detection> detectorDetections;

        if (detectorRan) {
            detectorStart = Clock::now();
            detectorDetections = objDetector.runInference(inputFrame);
            detectorEnd = Clock::now();
            inferenceMs = std::chrono::duration<double, std::milli>(detectorEnd - detectorStart).count();

            convertDetectorBoxesToPixels(detectorDetections, inputFrame.size());
        }

        // const auto& objDetections = detectorDetections;
        // std::vector<Detection> localDetections = localSearch.search(tracker.getTracks(), inputFrame);

        // motionStart = Clock::now();
        // std::vector<Detection> motDetections = motDetector.detections(inputFrame);
        // motionEnd = Clock::now();
        // std::vector<Detection> fusedDetections = DetectionFuser::merge(objDetections, motDetections);
        // std::vector<Detection> trackerInputs;

        cv::Mat detectionPanel = inputFrame.clone();

        trackerStart = Clock::now();
        tracker.update(detectorDetections, inputFrame, detectorRan);
        trackerEnd = Clock::now();

        tracker.drawTracks(detectionPanel);

        // --- metrics
        // int foregroundPixels = cv::countNonZero(fgMask);
        // double foregroundPercent = 100.0 * foregroundPixels / (fgMask.rows * fgMask.cols);
        
        metrics.activeTracks = tracker.getConfirmedTrackCount();
        metrics.frameNumber = i;
        metrics.trackerMs = std::chrono::duration<double,std::milli>(trackerEnd - trackerStart).count();
        metrics.detectorCount = static_cast<int>(detectorDetections.size());
        metrics.motionCount = 0;
        metrics.fusedCount = static_cast<int>(detectorDetections.size());
        metrics.foregroundPixels = 0;
        metrics.foregroundPercent = 0.0;
        metrics.inferenceMs = inferenceMs;

        // cv::Mat combined = visualizer.createDebugView(inputFrame, fgMask, detectionPanel, metrics);
        cv::Mat combined = detectionPanel;
        // --- create video once
        if (!writerInitialized) {
            writer.open(outputFolder + "/motion_analysis.mp4",
                        cv::VideoWriter::fourcc('m','p','4','v'),
                        30,
                        combined.size());

            if (!writer.isOpened()) { 
                std::cerr<< "failed to create video" << std::endl;
                return -1;
            }

            writerInitialized = true;
        }

        // cv::putText(detectionPanel,
        //             detectorRan ? "DETECTOR" : "KLT",
        //             {20, 40},
        //             cv::FONT_HERSHEY_SIMPLEX,
        //             1.0,
        //             detectorRan ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 255, 0),
        //             2);
        // --- write frame to mp4
        writerStart = Clock::now();
        writer.write(combined);
        writerEnd = Clock::now();
        
        auto frameEnd = std::chrono::high_resolution_clock::now();
        double frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

        metrics.fps = 1000.0 / frameMs;

        std::cout
            << "Detector : " << inferenceMs << " ms\n"
            << "Motion   : "
            << std::chrono::duration<double, std::milli>(motionEnd - motionStart).count()
            << " ms\n"
            << "ORB      : "
            << std::chrono::duration<double, std::milli>(orbEnd - orbStart).count()
            << " ms\n"
            << "Tracker  : "
            << std::chrono::duration<double, std::milli>(trackerEnd - trackerStart).count()
            << " ms\n"
            << "Writer   : "
            << std::chrono::duration<double, std::milli>(writerEnd - writerStart).count()
            << " ms\n"
            << "Frame    : " << frameMs << " ms\n"
            << "FPS      : " << metrics.fps << "\n\n";
        
        if (cv::waitKey(1) == 27) break;
    }

    destroyORBContext(orbCtx);
    writer.release();
    cv::destroyAllWindows();

    return 0;
}
