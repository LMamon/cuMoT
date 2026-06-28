#include "tracking/kalman_filter.h"

#include <algorithm>
#include <cmath>

// State:
// [x, y, w, h, vx, vy, vw, vh]
//
// x,y = top-left corner of bounding box

KalmanFilterCV::KalmanFilterCV() { set_identity_P(100.0f); }

void KalmanFilterCV::set_identity_P(float value) {
    P_.fill(0.0f);
    for (int i = 0; i < 8; ++i) {
        P_[i * 8 + i] = value;
    }
}

void KalmanFilterCV::init(const Box& z) {
    x_ = {z.cx, z.cy, z.w, z.h,
        0.0f, 0.0f, 0.0f, 0.0f
    };

    set_identity_P(10.0f);
    initialized_ = true;
}

Box KalmanFilterCV::predict(float dt) {
    if (!initialized_) return {};

    // x = F x
    x_[0] += dt * x_[4];
    x_[1] += dt * x_[5];
    x_[2] += dt * x_[6];
    x_[3] += dt * x_[7];

    // P = F P F^T + Q
    std::array<float, 64> FP{}, newP{};

    // F is identity with F[i][i+4] = dt for i=0..3
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            FP[r * 8 + c] = P_[r * 8 + c];

            if (r < 4) {
                FP[r * 8 + c] += dt * P_[(r + 4) * 8 + c];
            }
        }
    }

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            newP[r * 8 + c] = FP[r * 8 + c];

            if (c < 4) {
                newP[r * 8 + c] += dt * FP[r * 8 + (c + 4)];
            }
        }
    }

    for (int i = 0; i < 4; ++i) {
        newP[i * 8 + i] += process_noise_pos_;
        newP[(i + 4) * 8 + (i + 4)] += process_noise_vel_;
    }

    P_ = newP;
    return state_box();
}

Box KalmanFilterCV::update(const Box& z) {
    if (!initialized_) {
        init(z);
        return z;
    }

    float measurement[4] = { z.cx, z.cy, z.w, z.h };

    // H selects first 4 state components.
    // Innovation y = z - Hx
    float y[4];
    for (int i = 0; i < 4; ++i) {
        y[i] = measurement[i] - x_[i];
    }

    // S = HPH^T + R => top-left 4x4 of P + R
    float S[4][4]{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            S[r][c] = P_[r * 8 + c];
        }
        S[r][r] += measurement_noise_;
    }

    // Invert S using simple Gauss-Jordan for 4x4
    float A[4][8]{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            A[r][c] = S[r][c];
        }
        A[r][r + 4] = 1.0f;
    }

    for (int i = 0; i < 4; ++i) {
        float pivot = A[i][i];
        if (std::fabs(pivot) < 1e-6f) {
            pivot = 1e-6f;
        }

        for (int c = 0; c < 8; ++c) {
            A[i][c] /= pivot;
        }

        for (int r = 0; r < 4; ++r) {
            if (r == i) continue;

            float factor = A[r][i];
            for (int c = 0; c < 8; ++c) {
                A[r][c] -= factor * A[i][c];
            }
        }
    }

    float S_inv[4][4]{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            S_inv[r][c] = A[r][c + 4];
        }
    }

    // K = P H^T S^-1
    float K[8][4]{};
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 4; ++c) {
            for (int k = 0; k < 4; ++k) {
                K[r][c] += P_[r * 8 + k] * S_inv[k][c];
            }
        }
    }

    // x = x + K y
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 4; ++c) {
            x_[r] += K[r][c] * y[c];
        }
    }

    // P = (I - KH)P
    std::array<float, 64> newP{};
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            newP[r * 8 + c] = P_[r * 8 + c];

            for (int k = 0; k < 4; ++k) {
                newP[r * 8 + c] -= K[r][k] * P_[k * 8 + c];
            }
        }
    }

    P_ = newP;

    return state_box();
}

Box KalmanFilterCV::state_box() const {
    return {x_[0], x_[1], std::max(1.0f, x_[2]), std::max(1.0f, x_[3])};
}