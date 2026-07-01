#pragma once

#include "detection/detector_engine.h"
#include "tracking/kalman_filter.h"
#include "association/appearance_model.h"

enum class TrackState {
    Tentative,
    Confirmed,
    Lost,
    Removed
};

struct Track {
    TrackState state;
    float confidence = 0.0f;

    Detection lastDetection;
    KalmanFilterCV kf;
    AppearanceModel appearance;

    int age = 0;
    int framesSinceUpdate = 0;
    int hits = 1;
    int trackId = -1;

    bool localTrackerInitialized = false;
    int localTrackMisses = 0;
};
