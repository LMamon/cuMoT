#pragma once

#include <vector>
#include "detection/detector_engine.h"

std::vector<Detection> cudaNMS(const std::vector<Detection>& detections);