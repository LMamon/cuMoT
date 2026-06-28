// Reference: https://github.com/komrad36/CUDAHammingMean.git
//
// adapted for cuMoT appearance association.
// generates a track × detection appearance cost matrix.
//
// each block evaluates one:
//
//     TrackAppearanceModel
//             vs
//     DetectionAppearanceModel
//
// shared memory caches detection descriptors
// warp-level reductions find best descriptor matches.

#include "cuda/orb_descriptor_kernels.cuh"
#include "association/data_association.h"
#include "vision/orb_descriptor.h"
#include "cuda/cuda_check.h"

#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <cstdint>

// Compute Hamming distance between two ORB descriptors.

// ORB descriptor:
//     32 bytes
//     256 bits
//
// packed as 4 x uint64_t

__device__ __forceinline__ int descriptorDistance256(const uint8_t* a, const uint8_t* b) {
    const auto* a64 = reinterpret_cast<const uint64_t*>(a);
    const auto* b64 = reinterpret_cast<const uint64_t*>(b);

    int dist = 0;

    dist += __popcll(a64[0] ^ b64[0]);
    dist += __popcll(a64[1] ^ b64[1]);
    dist += __popcll(a64[2] ^ b64[2]);
    dist += __popcll(a64[3] ^ b64[3]);

    return dist;
}

__global__ void appearance_cost_matrix_kernel(const uint8_t* trackDescriptors,
                                                const int* trackCounts,
                                                const uint8_t* detectionDescriptors,
                                                const int* detectionCounts,
                                                int numTracks,
                                                int numDetections,
                                                float* appearanceCost) {

    const int detectionIdx = blockIdx.x;
    const int trackIdx = blockIdx.y;

    if (trackIdx >= numTracks || detectionIdx >= numDetections) return;

    const int trackCount = trackCounts[trackIdx];
    const int detectionCount = detectionCounts[detectionIdx];

    // no descriptors available.
    if (trackCount <= 0 || detectionCount <= 0) {
        if (threadIdx.x == 0) {
            appearanceCost[trackIdx * numDetections + detectionIdx] = 1.0f;
        }

        return;
    }

    const uint8_t* trackBase = trackDescriptors + trackIdx * MAX_ORB_FEATURES * ORB_DESCRIPTOR_SIZE;
    const uint8_t* detectionBase = detectionDescriptors + detectionIdx * MAX_ORB_FEATURES * ORB_DESCRIPTOR_SIZE;

    // cache detection descriptors in shared memory.
    __shared__ uint8_t sharedDetection[MAX_ORB_FEATURES * ORB_DESCRIPTOR_SIZE];

    for (int i = threadIdx.x; i < detectionCount * ORB_DESCRIPTOR_SIZE; i += blockDim.x) {
        sharedDetection[i] = detectionBase[i];
    }

    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;

    constexpr int WARPS_PER_BLOCK = 256 / 32;

    // best distance for the descriptor currently
    // assigned to this warp.
    int bestDist = 256;

    // store one best match per track descriptor.
    __shared__ int descriptorBest[MAX_ORB_FEATURES];

    // a warp may process multiple track descriptors.
    //
    // with:
    //     32 descriptors
    //     8 warps
    //
    // warp 0 handles:
    //     0,8,16,24
    //
    // warp 1 handles:
    //     1,9,17,25
    // etc.

    for (int trackDescIdx = warpId; trackDescIdx < trackCount; trackDescIdx += WARPS_PER_BLOCK) {
        bestDist = 256;
        const uint8_t* trackDesc =trackBase + trackDescIdx * ORB_DESCRIPTOR_SIZE;

        // Lanes scan detection descriptors.

        for (int d = lane; d < detectionCount; d += 32) {
            const uint8_t* detDesc = sharedDetection + d * ORB_DESCRIPTOR_SIZE;
            const int dist = descriptorDistance256(trackDesc, detDesc);

            bestDist = min(bestDist, dist);
        }

        // warp reduction:
        // find best detection descriptor match
        // for this track descriptor.
        for (int offset = 16; offset > 0; offset >>= 1) {
            bestDist = min(bestDist, __shfl_down_sync(0xffffffff, bestDist, offset));
        }

        if (lane == 0) {
            descriptorBest[trackDescIdx] = bestDist;
        }
    }

    __syncthreads();

    // first warp reduces descriptor matches
    // into a single appearance score.

    if (warpId == 0) {
        int sum = 0;

        if (lane < trackCount) {
            sum = descriptorBest[lane];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            const float meanDist = static_cast<float>(sum) / static_cast<float>(trackCount);
            const float similarity = 1.0f - (meanDist / 256.0f);

            // cost convention:
            //
            // 0.0 = strong match
            // 1.0 = weak match
            appearanceCost[trackIdx * numDetections + detectionIdx] = 1.0f - similarity;
        }
    }
}

void initializeAppearanceCudaScratch(AppearanceCudaScratch& scratch, int maxTracks, int maxDetections) {
    scratch.maxTracks = maxTracks;
    scratch.maxDetections = maxDetections;

    const size_t descriptorBytes = MAX_ORB_FEATURES * ORB_DESCRIPTOR_SIZE;

    CUDA_CHECK(cudaMalloc(&scratch.d_trackDescriptors, maxTracks * descriptorBytes));
    CUDA_CHECK(cudaMalloc(&scratch.d_detectionDescriptors, maxDetections * descriptorBytes));
    CUDA_CHECK(cudaMalloc(&scratch.d_trackCounts, maxTracks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scratch.d_detectionCounts, maxDetections * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&scratch.d_appearanceCost, maxTracks * maxDetections * sizeof(float)));
}

void destroyAppearanceCudaScratch(AppearanceCudaScratch& scratch) {
    if (scratch.d_trackDescriptors) cudaFree(scratch.d_trackDescriptors);
    if (scratch.d_detectionDescriptors) cudaFree(scratch.d_detectionDescriptors);
    if (scratch.d_trackCounts) cudaFree(scratch.d_trackCounts);
    if (scratch.d_detectionCounts) cudaFree(scratch.d_detectionCounts);
    if (scratch.d_appearanceCost) cudaFree(scratch.d_appearanceCost);

    scratch = AppearanceCudaScratch{};
}

void computeAppearanceCostMatrixCUDA(AppearanceCudaScratch& scratch,
                                        const AppearanceModel* tracks,
                                        int numTracks,

                                        const AppearanceModel* detections,
                                        int numDetections,

                                        float* h_appearanceCost) {

    const size_t descriptorBytes = MAX_ORB_FEATURES * ORB_DESCRIPTOR_SIZE;

    std::vector<uint8_t> trackDescriptors(numTracks * descriptorBytes);

    std::vector<uint8_t> detectionDescriptors(numDetections * descriptorBytes);

    std::vector<int> trackCounts(numTracks);
    std::vector<int> detectionCounts(numDetections);

    for (int i = 0; i < numTracks; ++i) {
        std::memcpy(trackDescriptors.data() + i * descriptorBytes,
                    tracks[i].descriptors.data(),
                    descriptorBytes);

        trackCounts[i] = tracks[i].descriptorCount;
    }    

        for (int i = 0; i < numDetections; ++i) {
        std::memcpy(detectionDescriptors.data() + i * descriptorBytes,
                    detections[i].descriptors.data(),
                    descriptorBytes);

        detectionCounts[i] = detections[i].descriptorCount;
    }

    CUDA_CHECK(cudaMemcpy(scratch.d_trackDescriptors,
                            trackDescriptors.data(),
                            numTracks * descriptorBytes,
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(scratch.d_detectionDescriptors,
                            detectionDescriptors.data(),
                            numDetections * descriptorBytes,
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(scratch.d_trackCounts,
                            trackCounts.data(),
                            numTracks * sizeof(int),
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(scratch.d_detectionCounts,
                            detectionCounts.data(),
                            numDetections * sizeof(int),
                            cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(numDetections, numTracks);

    appearance_cost_matrix_kernel<<<grid, block>>>(
        scratch.d_trackDescriptors,
        scratch.d_trackCounts,
        scratch.d_detectionDescriptors,
        scratch.d_detectionCounts,
        numTracks,
        numDetections,
        scratch.d_appearanceCost);

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_appearanceCost,
                            scratch.d_appearanceCost,
                            numTracks * numDetections * sizeof(float),
                            cudaMemcpyDeviceToHost));
}

float computeAppearanceSimilarityCUDA(AppearanceCudaScratch& scratch, const AppearanceModel& a,
                                                                    const AppearanceModel& b) {
    float cost = 1.0f;

    computeAppearanceCostMatrixCUDA(scratch, &a, 1, &b, 1, &cost);

    return 1.0f - cost;
}