#include "tracking/vpi_klt_tracker.h"
#include "cuda/cuda_check.h"

#include <algorithm>
#include <cstring>

VPIKLTLocalTracker::VPIKLTLocalTracker() = default;
VPIKLTLocalTracker::~VPIKLTLocalTracker() { reset(); }

bool VPIKLTLocalTracker::initialize(const cv::Size& frameSize) {
    // * init params
    width_ = frameSize.width;
    height_ = frameSize.height;
    // * reserve vectors
    inputBoxData_.reserve(MAX_TRACKS);
    inputPredictionData_.reserve(MAX_TRACKS);

    VPI_CHECK(vpiStreamCreate(VPI_BACKEND_CUDA, &stream_));
    VPI_CHECK(vpiInitKLTFeatureTrackerParams(&params_));
    
    // * create arrays
    VPI_CHECK(vpiArrayCreate(MAX_TRACKS, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &inputBoxes_));
    VPI_CHECK(vpiArrayCreate(MAX_TRACKS, VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D, 0, &inputPredictions_));
    VPI_CHECK(vpiArrayCreate(MAX_TRACKS, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &outputBoxes_));
    VPI_CHECK(vpiArrayCreate(MAX_TRACKS, VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D, 0, &outputEstimations_));

    // * done
    initialized_ = true;
    return true;
}

void VPIKLTLocalTracker::initializeIdentityPrediction(VPIHomographyTransform2D& prediction) {
    prediction = {};

    prediction.mat3[0][0] = 1.f;
    prediction.mat3[1][1] = 1.f;
    prediction.mat3[2][2] = 1.f;
}

bool VPIKLTLocalTracker::addTrack(int trackId, const Box& initialBox) {
    // trackId
    if (!initialized_) return false;
    if (inputBoxData_.size() >= MAX_TRACKS) return false;
    if (trackIdToSlot_.find(trackId) != trackIdToSlot_.end()) return false;

    // slot
    int slot = static_cast<int>(inputBoxData_.size());
    trackIdToSlot_[trackId] = slot;
    // ↓

    // trackedBoxes_
    VPIKLTTrackedBoundingBox track = {};
    track.bbox.xform.mat3[0][0] = 1.f;
    track.bbox.xform.mat3[1][1] = 1.f;
    track.bbox.xform.mat3[2][2] = 1.f;

    track.bbox.xform.mat3[0][2] = initialBox.cx - initialBox.w * 0.5f;
    track.bbox.xform.mat3[1][2] = initialBox.cy - initialBox.h * 0.5f;

    track.bbox.width = initialBox.w;
    track.bbox.height = initialBox.h;

    track.trackingStatus = 0;
    track.templateStatus = 1;

    inputBoxData_.push_back(track);
    // identity prediction
    VPIHomographyTransform2D prediction;
    initializeIdentityPrediction(prediction);

    inputPredictionData_.push_back(prediction);
    return true;
}

std::vector<LocalTrackResult> VPIKLTLocalTracker::update(const std::vector<LocalTrackInput>& tracks, 
                                                            const cv::Mat& frame) {

    std::vector<LocalTrackResult> results;
    results.reserve(tracks.size());

    if (!initialized_ || frame.empty() || inputBoxData_.empty()) return results;
    if (frame.cols != width_ || frame.rows != height_) return results;

    for (const auto& input : tracks) {
        auto it = trackIdToSlot_.find(input.trackId);
        if (it == trackIdToSlot_.end()) continue;

        int slot = it->second;
        auto& box = inputBoxData_[slot];

        box.bbox.xform.mat3[0][2] = input.predictedBox.cx - input.predictedBox.w * 0.5f;
        box.bbox.xform.mat3[1][2] = input.predictedBox.cy - input.predictedBox.h * 0.5f;

        box.bbox.width = input.predictedBox.w;
        box.bbox.height = input.predictedBox.h;
    }

    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    // TODO:
    // Wrap current frame.
    if (!currentFrame_) {
        VPI_CHECK(vpiImageCreateWrapperOpenCVMat(gray, 0, &currentFrame_));
    } else {
        VPI_CHECK(vpiImageSetWrappedOpenCVMat(currentFrame_, gray));
    }

    if (!payload_) {
        VPIImageFormat format;
        VPI_CHECK(vpiImageGetFormat(currentFrame_, &format));
        VPI_CHECK(vpiCreateKLTFeatureTracker(backend_, width_, height_, format, nullptr, &payload_));
    }

    if (!previousFrame_) {
        VPI_CHECK(vpiImageCreateWrapperOpenCVMat(gray, 0, &previousFrame_));
        for (const auto& input : tracks) {
            auto it = trackIdToSlot_.find(input.trackId);
            if (it == trackIdToSlot_.end()) continue;

            const auto& b = inputBoxData_[it->second];

            float x = b.bbox.xform.mat3[0][2];
            float y = b.bbox.xform.mat3[1][2];
            float w = b.bbox.width;
            float h = b.bbox.height;

            results.push_back(LocalTrackResult{input.trackId,
                                                true,
                                                Box{x + w * 0.5f, y + h * 0.5f, w, h},
                                                0.5f});
        }
        return results;
    }

    VPI_CHECK(vpiArraySetSize(inputBoxes_, static_cast<int32_t>(inputBoxData_.size())));
    VPI_CHECK(vpiArraySetSize(inputPredictions_, static_cast<int32_t>(inputPredictionData_.size())));

    {
        VPIArrayData boxData;
        VPI_CHECK(vpiArrayLockData(inputBoxes_, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &boxData));

        auto* boxes = reinterpret_cast<VPIKLTTrackedBoundingBox*>(boxData.buffer.aos.data);
        std::copy(inputBoxData_.begin(), inputBoxData_.end(), boxes);

        VPI_CHECK(vpiArrayUnlock(inputBoxes_));
    }
    {
        VPIArrayData predData;
        VPI_CHECK(vpiArrayLockData(inputPredictions_, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &predData));

        auto* preds = reinterpret_cast<VPIHomographyTransform2D*>(predData.buffer.aos.data);
        std::copy(inputPredictionData_.begin(), inputPredictionData_.end(), preds);

        VPI_CHECK(vpiArrayUnlock(inputPredictions_));
    }

    VPI_CHECK(vpiSubmitKLTFeatureTracker(stream_, backend_, payload_, previousFrame_, inputBoxes_, inputPredictions_,
                                                                                                    currentFrame_,
                                                                                                    outputBoxes_,
                                                                                                    outputEstimations_,
                                                                                                    &params_));

    VPI_CHECK(vpiStreamSync(stream_));

    VPIArrayData outputBoxData;
    VPI_CHECK(vpiArrayLockData(outputBoxes_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outputBoxData));

    VPIArrayData outputEstimBoxData;
    VPI_CHECK(vpiArrayLockData(outputEstimations_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outputEstimBoxData));

    auto* updatedBoxes = reinterpret_cast<VPIKLTTrackedBoundingBox*>(outputBoxData.buffer.aos.data);
    auto* estimations = reinterpret_cast<VPIHomographyTransform2D*>(outputEstimBoxData.buffer.aos.data);

    // Real VPI submit comes next. For now return current box state
    // so Tracker integration can compile/run.
    for (const auto& input : tracks) {
        auto it = trackIdToSlot_.find(input.trackId);
        if (it == trackIdToSlot_.end()) continue;

        int slot = it->second;
        auto& box = inputBoxData_[slot];
        VPIKLTTrackedBoundingBox& updated = updatedBoxes[slot];

        LocalTrackResult result;
        result.trackId = input.trackId;

        if (updated.trackingStatus != 0) {
            result.valid = false;
            results.push_back(result);
            continue;
        }

        if (updated.templateStatus != 0) {
            inputBoxData_[slot] = updated;
            inputBoxData_[slot].templateStatus = 1;
            initializeIdentityPrediction(inputPredictionData_[slot]);
        } else {
            inputBoxData_[slot].templateStatus = 0;
            inputPredictionData_[slot] = estimations[slot];
        }

        float x = inputBoxData_[slot].bbox.xform.mat3[0][2] + inputPredictionData_[slot].mat3[0][2];
        float y = inputBoxData_[slot].bbox.xform.mat3[1][2] + inputPredictionData_[slot].mat3[1][2];
        float w = inputBoxData_[slot].bbox.width * inputBoxData_[slot].bbox.xform.mat3[0][0] *
                                                 inputPredictionData_[slot].mat3[0][0];

        float h = inputBoxData_[slot].bbox.height * inputBoxData_[slot].bbox.xform.mat3[1][1] *
                                                 inputPredictionData_[slot].mat3[1][1];

        result.valid = true;
        result.box = Box{x + w * 0.5f, y + h * 0.5f, w, h};
        result.confidence = 0.75f;
        results.push_back(result);
    }

    VPI_CHECK(vpiArrayUnlock(outputEstimations_));
    VPI_CHECK(vpiArrayUnlock(outputBoxes_));

    std::swap(previousFrame_, currentFrame_);

    return results;
}

void VPIKLTLocalTracker::removeTrack(int trackId) {
    // Remove the slot.
    auto it = trackIdToSlot_.find(trackId);
    if (it == trackIdToSlot_.end()) return;

    int slot = it->second;
    int last = static_cast<int>(inputBoxData_.size()) - 1;

    // compact the vectors.
    if (slot != last) {
        inputBoxData_[slot] = inputBoxData_[last];
        inputPredictionData_[slot] = inputPredictionData_[last];

        for (auto& kv : trackIdToSlot_) {
            if (kv.second == last) {
                kv.second = slot;
                break;
            }
        }
    }

    // Update
    // trackIdtoSlow_
    // done
    inputBoxData_.pop_back();
    inputPredictionData_.pop_back();
    trackIdToSlot_.erase(it);
}

void VPIKLTLocalTracker::reset() {
    if (stream_) vpiStreamSync(stream_);

    if (payload_) {
        vpiPayloadDestroy(payload_);
        payload_ = nullptr;
    }

    if (inputBoxes_) {
        vpiArrayDestroy(inputBoxes_);
        inputBoxes_ = nullptr;
    }

    if (inputPredictions_) {
        vpiArrayDestroy(inputPredictions_);
        inputPredictions_ = nullptr;
    }

    if (outputBoxes_) {
        vpiArrayDestroy(outputBoxes_);
        outputBoxes_ = nullptr;
    }

    if (outputEstimations_) {
        vpiArrayDestroy(outputEstimations_);
        outputEstimations_ = nullptr;
    }

    if (previousFrame_) {
        vpiImageDestroy(previousFrame_);
        previousFrame_ = nullptr;
    }

    if (currentFrame_) {
        vpiImageDestroy(currentFrame_);
        currentFrame_ = nullptr;
    }

    if (stream_) {
        vpiStreamDestroy(stream_);
        stream_ = nullptr;
    }

    inputBoxData_.clear();
    inputPredictionData_.clear();
    trackIdToSlot_.clear();

    initialized_ = false;
}