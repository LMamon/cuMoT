#pragma once

#include <array>
#include "detection/detector_engine.h"

class KalmanFilterCV {
    public:
        KalmanFilterCV();

        void init(const Box& z);
        Box predict(float dt = 1.0f);
        Box update(const Box& z);

        Box state_box() const;
        bool initialized() const { return initialized_; }

    private:
        bool initialized_ = false;

        // State: [cx, cy, w, h, vx, vy, vw, vh]
        std::array<float, 8> x_{};

        // Covariance matrix P, row-major 8x8
        std::array<float, 64> P_{};

        float process_noise_pos_ = 1.0f;
        float process_noise_vel_ = 10.0f;
        float measurement_noise_ = 5.0f;

        void set_identity_P(float value);
};