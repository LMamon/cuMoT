#pragma once

#include <vpi/VPI.h>
#include <vpi/algo/ORB.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstdint>

constexpr int MAX_ORB_FEATURES = 32;
constexpr int ORB_DESCRIPTOR_SIZE = 32;

struct ORBContext {
    VPIStream stream = nullptr;
    VPIPayload payload = nullptr;

    VPIArray corners = nullptr;
    VPIArray descriptors = nullptr;

    VPIORBParams params;
};

struct AppearanceModel {
    int descriptorCount = 0;

    std::array<uint8_t, MAX_ORB_FEATURES * ORB_DESCRIPTOR_SIZE> descriptors;
};

bool initializeORBContext(ORBContext& ctx);
void destroyORBContext(ORBContext& ctx);

bool buildAppearanceModelFromROI(ORBContext& ctx, const cv::Mat& roi, AppearanceModel& model);
                                