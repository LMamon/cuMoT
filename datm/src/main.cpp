#include <opencv2/opencv.hpp>

#include "detection/detector_engine.h"
#include "vision/motion_detector.h"
#include "tracking/tracker.h"
#include "visualizer.h"
#include "association/appearance_model.h"

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

const int DETECTOR_INTERVAL = 6;
const int DETECTOR_WARMUP_FRAMES = 70;

int main() {
    std::string modelPath = "...";
    std::string inputFolder = "...";
    std::string outputFolder = "...";

    std::filesystem::create_directories(outputFolder);

    // --- video writer for 3-panel output
    cv::VideoWriter writer;
    bool writerInitialized = false;
    
    std::vector<Detection> cachedDetections;
    double inferenceMs = 0.0;

    DebugMetrics metrics;
    DetectorEngine objDetector;
    MotionDetector motDetector;
    ORBContext orbCtx;
    
    if (!objDetector.loadEngine(modelPath)) return -1;

    if (!initializeORBContext(orbCtx)) {
        std::cerr << "Failed to initialize ORB context\n";
        return -1;
    }

    // create multi objecy tracker
    Tracker tracker(orbCtx);
    Visualizer visualizer;

    for (int i = 0; i < 197; ++i) {
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
    
        if (detectorRan) {
            detectorStart = Clock::now();
            cachedDetections = objDetector.runInference(inputFrame);
            detectorEnd = Clock::now();
            inferenceMs = std::chrono::duration<double, std::milli>(detectorEnd - detectorStart).count();
        }

        const auto& objDetections = cachedDetections;
        motionStart = Clock::now();
        std::vector<Detection> motDetections = motDetector.detections(inputFrame);
        motionEnd = Clock::now();
        std::vector<Detection> fusedDetections = DetectionFuser::merge(objDetections, motDetections);
        
        cv::Mat fgMask = motDetector.foregroundMask();
        if (fgMask.empty()) continue;

        // --- convert mask to color for display
        cv::Mat maskBgr;
        cv::cvtColor(fgMask, maskBgr, cv::COLOR_GRAY2BGR);

        // --- ensure same size for hconcat
        cv::resize(maskBgr, maskBgr, inputFrame.size());

        // --- panel 3: final tracker output
        cv::Mat detectionPanel = inputFrame.clone();

        orbStart = Clock::now();
        // convert normalized coordinates
        if (detectorRan) {
            for (auto& d : fusedDetections) {
                d.box.cx *= inputFrame.cols;
                d.box.cy *= inputFrame.rows;

                d.box.w *= inputFrame.cols;
                d.box.h *= inputFrame.rows;
            
                // build appearance descriptors for every detection
                int x = std::max(0, static_cast<int>(left(d.box)));
                int y = std::max(0, static_cast<int>(top(d.box)));

                int w = static_cast<int>(d.box.w);
                int h = static_cast<int>(d.box.h);

                if (x >= inputFrame.cols || y >= inputFrame.rows || w <= 4 || h <= 4) {
                    continue;
                }

                w = std::min(w, inputFrame.cols - x);
                h = std::min(h, inputFrame.rows - y);

                cv::Rect roiRect(x, y, w, h);
                cv::Mat roi = inputFrame(roiRect);

                buildAppearanceModelFromROI(orbCtx, roi, d.appearance);
            }
        } else {
            for (auto& d : fusedDetections) {
                d.box.cx *= inputFrame.cols;
                d.box.cy *= inputFrame.rows;
                d.box.w *= inputFrame.cols;
                d.box.h *= inputFrame.rows;
            }
        }
        orbEnd = Clock::now();

        // update MoT
        trackerStart = Clock::now();
        tracker.update(fusedDetections, inputFrame);
        trackerEnd = Clock::now();

        tracker.drawTracks(detectionPanel);

        // --- metrics
        int foregroundPixels = cv::countNonZero(fgMask);
        double foregroundPercent = 100.0 * foregroundPixels / (fgMask.rows * fgMask.cols);
        
        metrics.activeTracks = tracker.getConfirmedTrackCount();
        metrics.frameNumber = i;
        metrics.trackerMs = std::chrono::duration<double,std::milli>(trackerEnd-trackerStart).count();
        metrics.detectorCount = static_cast<int>(objDetections.size());
        metrics.motionCount = static_cast<int>(motDetections.size());
        metrics.fusedCount = static_cast<int>(fusedDetections.size());
        metrics.foregroundPixels = foregroundPixels;
        metrics.foregroundPercent = foregroundPercent;
        metrics.inferenceMs = inferenceMs;

        cv::Mat combined = visualizer.createDebugView(inputFrame, fgMask,
                                                        detectionPanel,
                                                        metrics);

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