#pragma once

#include <vector>
#include "tracking/local_search.h"

class VPIDCFLocalTracker : public ILocalTracker {
    public:
        VPIDCFLocalTracker();
        ~VPIDCFLocalTracker() override;

        void initialize(const std::vector<LocalTrackInput>& tracks,
                        const cv::Mat& frame) override;

        std::vector<LocalTrackResult> update(const std::vector<LocalTrackInput>& tracks,
                                                cont cv::Mat& frame) override;

        void remove(const std::vector<int>& trackIds) override;
        void reset() override;
}