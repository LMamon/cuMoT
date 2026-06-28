#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "cuda/gpu_iou_device.cuh"

#include <cmath>
#include <algorithm>

#define CUDA_EXPRESSION_CHECKER
#include "cuda/cuda_check.h"

constexpr int TILE_SIZE = 32;
constexpr float NMS_THRESHOLD = 0.5f;

__global__ void nmsKernel(const Box* boxes, uint32_t* keep, int numDetections, int numTiles) {
    // detection handled by this thread
    int localX = threadIdx.x;
    int localY = threadIdx.y;
    int localIdx = localY * blockDim.x + localX;

    int det = blockIdx.x * TILE_SIZE + localIdx;

    // tile being compared against
    int loadedTile = blockIdx.y;

    // exit if detection is outside valid range
    if (loadedTile >= numTiles || det >= numDetections) return;

    __shared__ Box tileBoxes[4][8];
    
    int tileStart = loadedTile * TILE_SIZE;
    int tileBoxIdx = tileStart + localIdx;

    if (tileBoxIdx < numDetections) {
        tileBoxes[localY][localX] = boxes[tileBoxIdx];
    }

    // synchronizing with syncwarp instead of syncthreads since using tilesize 32 
    __syncwarp();

    uint32_t mask = 0;

    // compare detection box against boxes in tile
    // for (all boxes in tileBoxes)
    // keeping 2D for intuition
    // higher detections supress lower condifence ones
    for (int t = 0; t < TILE_SIZE; t++) {
        int tileGlobalIdx = tileStart + t;
        // skip invalid tile entries
        if (tileGlobalIdx >= numDetections) continue;

        if (tileGlobalIdx <= det) continue;

        // get box t from tileBoxes
        int r = t / 8;
        int c = t % 8;
        float iou = boxIoU(boxes[det], tileBoxes[r][c]);

        if (iou > NMS_THRESHOLD) {
            // set bit t
            mask |= (1u << t);
        }
    }
    // write mask to global memory
    keep[det * numTiles + loadedTile] = mask;    
}


std::vector<Detection> cudaNMS(const std::vector<Detection>& detections) {
    // cudaEvent_t start;
    // cudaEvent_t stop;
    
    // pre sort detections by confidence for box comparison
    std::vector<Detection> sortedDetections = detections;
    std::sort(sortedDetections.begin(), sortedDetections.end(), 
                [](const Detection& a, const Detection& b) {
                    return a.confidence > b.confidence;
        });

    size_t numDetections = sortedDetections.size();
    if (detections.empty()) return {};
    int numTiles = (numDetections + TILE_SIZE - 1) / TILE_SIZE;
    if (numTiles == 0) return sortedDetections;
    size_t keepBytes = numDetections * numTiles * sizeof(uint32_t);
    std::vector<uint32_t> keepMasks(numDetections * numTiles);

    // extract box data
    std::vector<Box> boxes;
    boxes.reserve(numDetections);

    for (const auto& d : sortedDetections) {
        boxes.push_back(d.box);
    }

    // allocate device buffers
    Box* d_boxes = nullptr;
    uint32_t* d_keep = nullptr;

    CUDA_CHECK(cudaMalloc(&d_boxes, boxes.size() * sizeof(Box)));
    CUDA_CHECK(cudaMalloc(&d_keep, keepBytes));

    // copy detections to device
    CUDA_CHECK(cudaMemcpy(d_boxes, boxes.data(), boxes.size() * sizeof(Box), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_keep, 0, keepBytes));

    // launch NMS kernel
    dim3 block(8, 4);
    dim3 grid(numTiles, numTiles);
    // warmup, not timed
    // nmsKernel<<<grid, block>>>(d_boxes,
    //     d_keep,
    //     numDetections,
    //     numTiles
    // );

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
    // start benchmark
    // CUDA_CHECK(cudaEventCreate(&start));
    // CUDA_CHECK(cudaEventCreate(&stop));

    // CUDA_CHECK(cudaEventRecord(start));
    nmsKernel<<<grid, block>>>(d_boxes,
        d_keep,
        numDetections,
        numTiles
    );
    
    // CUDA_CHECK(cudaEventRecord(stop));
    CUDA_KERNEL_CHECK();

    // CUDA_CHECK(cudaEventSynchronize(stop));

    // float ms = 0.0f;
    // CUDA_CHECK(cudaEventElapsedTime(
    //     &ms,
    //     start,
    //     stop));

    // std::cout << block.x << "x" << block.y
    //     << " nms kernel: " << ms << " ms\n";

    // CUDA_CHECK(cudaEventDestroy(start));
    // CUDA_CHECK(cudaEventDestroy(stop));

    // copy suppression mask back
    CUDA_CHECK(cudaMemcpy(keepMasks.data(), d_keep, keepBytes, cudaMemcpyDeviceToHost));

    // build filtered detections
    std::vector<Detection> filteredDetections;
    std::vector<bool> suppressed(numDetections, false);

    // walk detection × tile × bitmask tensor
    for (int x = 0; x < numDetections; x++) {
        // skip high conf detections
        if (suppressed[x]) continue;

        for (int y = 0; y < numTiles; y++) {
            uint32_t mask = keepMasks[x * numTiles + y];

            for (int z = 0; z < TILE_SIZE; z++) {
                if (mask & (1u << z)) {
                    // convert (y, z) back into global x index
                    int suppressedDetection = y * TILE_SIZE + z;
                    if (suppressedDetection == x) continue;
                    
                    if (suppressedDetection < numDetections) {
                        suppressed[suppressedDetection] = true;
                    }
                }
            }
        }
    }

    for (int i = 0; i < numDetections; i++) {
        if (!suppressed[i]) {
            filteredDetections.push_back(sortedDetections[i]);
        }
    }

    // cleanup
    cudaFree(d_boxes);
    cudaFree(d_keep);

    // std::cout << "filtered detections: " << filteredDetections.size() << "\n";
    return filteredDetections;
}
