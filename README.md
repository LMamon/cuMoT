# cuMoT

CUDA-accelerated multi-object tracking system targeting NVIDIA Jetson edge devices.

cuMoT integrates object detection, motion-based discovery, data association, and state estimation into a modular perception pipeline for real-time edge deployment.

---

## Current Features

- TensorRT object detection
- Motion-based object discovery
- Detection fusion
- Kalman filter state estimation
- CUDA-accelerated IoU computation
- ORB descriptor matching
- Multi-object tracking

---

## Project Structure

```
include/
    association/
    cuda/
    detection/
    tracking/
    vision/

datm/
    Detector implementations
    Tracker management/pipeline
    Video processing

kernels/
    CUDA kernels
```

---

## Technologies

- CUDA
- TensorRT
- CMake
- OpenCV
- VPI

---

## Current Pipeline

```
Video
   │
   ▼
Object Detector
   │
   ├─────────────────┐
   ▼                 │
Motion segmentation  │
   │                 │
   └─────Fusion──────┘
         │
         ▼
CUDA IoU Association
         │
         ▼
Kalman Tracker
         │
         ▼
Visualization
```

---

## Design Motivation

cuMoT investigates the tradeoff between algorithmic complexity and deployment efficiency in modern multi-object tracking.

Rather than relying exclusively on learned models throughout the tracking pipeline, cuMoT combines CUDA-accelerated vision algorithms with lightweight neural network components where they provide clear value.

The objective is to develop a modular tracking system optimized for real-time execution on edge platforms.

---

## Repository Status

This branch captures the initial implementation of cuMoT.

Future development focuses on restructuring the tracking pipeline around motion-based object discovery, improving system modularity, and expanding benchmarking and evaluation.

---
## Corrent Limitations

- Evaluation benchmarks in progress
- Architecture undergoing redesign

---
## License

MIT