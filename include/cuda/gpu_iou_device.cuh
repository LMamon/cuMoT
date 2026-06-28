#pragma once

#include <vector>
#include "detection/detector_engine.h"
#include "tracking/tracker.h"

__device__ __forceinline__ float boxIoU(const Box& a, const Box& b) {
    float acx1 = a.cx - a.w * 0.5f;
    float acy1 = a.cy - a.h * 0.5f;
    float ax2 = a.cx + a.w * 0.5f;
    float ay2 = a.cy + a.h * 0.5f;

    float bx1 = b.cx - b.w * 0.5f;
    float bcy1 = b.cy - b.h * 0.5f;
    float bx2 = b.cx + b.w * 0.5f;
    float by2 = b.cy + b.h * 0.5f;

    float icx1 = fmaxf(acx1, bx1);
    float icy1 = fmaxf(acy1, bcy1);
    float icx2 = fminf(ax2, bx2);
    float icy2 = fminf(ay2, by2);

    float iw = fmaxf(0.0f, icx2 - icx1);
    float ih = fmaxf(0.0f, icy2 - icy1);

    float inter = iw * ih;
    float areaA = a.w * a.h;
    float areaB = b.w * b.h;

    float uni = areaA + areaB - inter;

    return uni > 0.0f ? inter / uni : 0.0f;
}
