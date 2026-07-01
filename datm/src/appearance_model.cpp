#include "association/appearance_model.h"

#include <vpi/VPI.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/algo/GaussianPyramid.h>

#include <algorithm>
#include <cstring>

bool initializeORBContext(ORBContext& ctx) {
    vpiInitORBParams(&ctx.params);

    constexpr int ORB_DESCRIPTOR_BITS = 256;
    ctx.params.maxFeaturesPerLevel = MAX_ORB_FEATURES;
    ctx.params.maxPyramidLevels = 1;
    ctx.params.fastParams.intensityThreshold = 30;

    int bufferCapacity = ctx.params.maxFeaturesPerLevel * 20;
    int outputCapacity = ctx.params.maxFeaturesPerLevel * ctx.params.maxPyramidLevels;

    if (vpiStreamCreate(0, &ctx.stream) != VPI_SUCCESS) return false;

    if (vpiCreateORBFeatureDetector(VPI_BACKEND_CUDA, bufferCapacity, &ctx.payload) != VPI_SUCCESS) {
        destroyORBContext(ctx);
        return false;
    }
    
    if (vpiArrayCreate(outputCapacity,
                        VPI_ARRAY_TYPE_KEYPOINT_F32,
                        0,
                        &ctx.corners) != VPI_SUCCESS) {
        destroyORBContext(ctx);
        return false;
    }
    
    if(vpiArrayCreate(outputCapacity,
                        VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR,
                        0,
                        &ctx.descriptors) != VPI_SUCCESS) {
        destroyORBContext(ctx);
        return false;
    }
    return true;
}

void destroyORBContext(ORBContext& ctx) {
    if (ctx.corners) vpiArrayDestroy(ctx.corners);
    if (ctx.descriptors) vpiArrayDestroy(ctx.descriptors);
    if (ctx.payload) vpiPayloadDestroy(ctx.payload);
    if (ctx.stream) vpiStreamDestroy(ctx.stream);
    
    ctx = ORBContext{};
}

bool buildAppearanceModelFromROI(ORBContext& ctx, const cv::Mat& roi, AppearanceModel& model) {
    model.descriptorCount = 0;
    model.descriptors.fill(0);

    if (roi.empty()) return false;

    cv::Mat gray;
    if (roi.channels() == 1)
        gray = roi;
    else
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

    VPIImage input = nullptr;
    VPIPyramid pyramid = nullptr;

    if (vpiImageCreateWrapperOpenCVMat(gray, 0, &input) != VPI_SUCCESS) return false;

    if(vpiPyramidCreate(gray.cols, 
                        gray.rows,
                        VPI_IMAGE_FORMAT_U8,
                        ctx.params.maxPyramidLevels,
                        0.5,
                        VPI_BACKEND_CUDA,
                        &pyramid) != VPI_SUCCESS) {
        vpiImageDestroy(input);
        return false;
    }

    bool ok = true;

    ok &= vpiSubmitGaussianPyramidGenerator(ctx.stream,
                                            VPI_BACKEND_CUDA,
                                            input,
                                            pyramid,
                                            VPI_BORDER_CLAMP) == VPI_SUCCESS;

                                        
    ok &= vpiSubmitORBFeatureDetector(ctx.stream,
                                        VPI_BACKEND_CUDA,
                                        ctx.payload,
                                        pyramid,
                                        ctx.corners,
                                        ctx.descriptors,
                                        &ctx.params,
                                        VPI_BORDER_LIMITED) == VPI_SUCCESS;

    ok &= vpiStreamSync(ctx.stream) == VPI_SUCCESS;
    
    if (ok) {
        VPIArrayData descData;
        if (vpiArrayLockData(ctx.descriptors, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &descData) == VPI_SUCCESS) {
                                            // use pointer
            const int count = std::min<int>(*descData.buffer.aos.sizePointer, MAX_ORB_FEATURES);
            const auto* src = static_cast<const uint8_t*>(descData.buffer.aos.data);

            for (int i = 0; i < count; ++i) {
                std::memcpy(model.descriptors.data() + i * ORB_DESCRIPTOR_SIZE,
                            src + i * descData.buffer.aos.strideBytes, ORB_DESCRIPTOR_SIZE);
            }

            model.descriptorCount = count;
            vpiArrayUnlock(ctx.descriptors);
        }
    }
    vpiPyramidDestroy(pyramid);
    vpiImageDestroy(input);

    return ok && model.descriptorCount > 0;
}
