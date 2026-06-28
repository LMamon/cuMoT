#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "detection/detector_engine.h"
#include "cuda/gpu_iou_device.cuh"
#include <cmath>

#define CUDA_EXPRESSION_CHECKER
#include "cuda/cuda_check.h"

__global__ void computeMatrixKernel(const Box* __restrict__ tracks, const Box* __restrict__ detections,
                                    float* __restrict__ matrix, int numTracks, int numDetections) {

        int d = blockIdx.x * blockDim.x + threadIdx.x;
        int t = blockIdx.y * blockDim.y + threadIdx.y;

        if (t >= numTracks || d >= numDetections) return;

        matrix[t * numDetections + d] = boxIoU(tracks[t], detections[d]);
}

std::vector<float> computeIoUMatrixGPU(const std::vector<Box>& tracks, const std::vector<Box>& detections) {
        size_t matrixSize = tracks.size() * detections.size();
        size_t matrixBytes = matrixSize * sizeof(float);

        std::vector<float> matrix(matrixSize);
        // host output buffers
        Box* d_tracks = nullptr;
        Box* d_detections = nullptr;
        float* d_matrix = nullptr;
        
        CUDA_CHECK(cudaMalloc(&d_tracks, tracks.size() * sizeof(Box)));
        CUDA_CHECK(cudaMalloc(&d_detections, detections.size() * sizeof(Box)));
        CUDA_CHECK(cudaMalloc(&d_matrix, matrixBytes));
        
        CUDA_CHECK(cudaMemcpy(d_tracks, tracks.data(), tracks.size() * sizeof(Box), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_detections, detections.data(), detections.size() * sizeof(Box), cudaMemcpyHostToDevice));
        
        dim3 block(16, 16);
        dim3 grid((detections.size() + block.x - 1) / block.x,
                    (tracks.size() + block.y - 1) / block.y);
        
        
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        computeMatrixKernel<<<grid, block>>>(
            d_tracks,
            d_detections,
            d_matrix,
            tracks.size(),
            detections.size()
        );

        CUDA_KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(matrix.data(), d_matrix, matrixBytes, cudaMemcpyDeviceToHost));

        cudaFree(d_tracks);
        cudaFree(d_detections);
        cudaFree(d_matrix);

        return matrix;
    }