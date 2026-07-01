# cuMoT

CUDA-accelerated multi-object tracking prototype targeting NVIDIA Jetson edge devices.

cuMoT combines TensorRT object detection, Kalman motion prediction, data association, appearance matching, and local visual tracking into a modular real-time perception pipeline. The project focuses on practical edge-AI tracking architecture: detector integration, track lifecycle management, GPU-accelerated association, and local tracker backends for maintaining object tracks between detector updates.

---

## Demo

[![cuMoT demo](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID)

---

## Current Features

- TensorRT object detection
- Detector-driven multi-object tracking
- Kalman filter state prediction and correction
- Track lifecycle management
- CUDA-accelerated IoU computation
- ORB-based appearance descriptor matching
- Detection fusion pipeline
- Local tracker interface for short-term propagation
- VPI KLT local tracker backend
- VPI DCF tracker interface/stub
- Real-time visualization of confirmed tracks

---

## Technologies

- C++
- CUDA
- TensorRT
- OpenCV
- NVIDIA VPI
- CMake

---

## Tracking Pipeline

cuMoT processes video frames through a detector-driven tracking pipeline. TensorRT object detection provides the primary object observations, which are filtered and associated with existing tracks using motion consistency, IoU, and appearance cues. Kalman filtering predicts track state between measurements, while the tracker lifecycle manages tentative, confirmed, lost, and removed tracks. Confirmed tracks can optionally be propagated by a local visual tracker between detector updates, allowing the system to separate expensive detector inference from cheaper short-term motion tracking.

---

## Performance Snapshot incomplete

Runtime throughput was measured on a GMOT-40 sequence using detector refresh intervals of 3, 5, and 10 frames. GMOT-40 is a generic multiple-object tracking benchmark with 40 annotated sequences across 10 object categories, including dense scenes with frequent occlusion, blur, and target entry/exit events. The benchmark sequence used here is from the livestock category.  [oai_citation:0‡spritea.github.io](https://spritea.github.io/GMOT40/download.html)

| Detector Interval | Runtime Throughput |
|---:|---:|
| 3rd frame | 21–24 FPS |
| 5th frame | 28–33 FPS |
| 10th frame | 35–40 FPS |

Higher detector intervals reduce detector load and increase throughput by relying more heavily on local tracking between detector updates.

## Local Tracking Redesign

The current branch introduces a local tracker abstraction so short-term visual tracking can be swapped without rewriting the main tracker lifecycle.

Current local tracker work includes:

- `ILocalTracker` interface
- VPI KLT tracker backend
- Tracker lifecycle integration
- Local tracker initialization for confirmed tracks
- Local measurement validation before Kalman correction

The local tracker is intended to maintain confirmed tracks between detector updates. It is not responsible for creating new object tracks or replacing detector-based association.

---

## Design Motivation

cuMoT investigates the engineering tradeoffs involved in real-time multi-object tracking on edge devices.

Rather than treating tracking as a single algorithm, cuMoT separates the system into detection, association, motion prediction, appearance matching, lifecycle management, local propagation, GPU acceleration, and edge deployment. This structure makes it easier to compare tracking strategies, benchmark performance, and replace individual components without redesigning the full pipeline.

The current implementation prioritizes a stable detector-driven tracking core. KLT and DCF local tracking are being integrated as short-term propagation tools, while ORB appearance matching serves as a lightweight baseline for identity consistency.

---

## Current Limitations

- Evaluation benchmarks are still in progress
- VPI DCF backend is not fully implemented
- ORB appearance matching is a lightweight baseline, not ReID
- Occlusion-heavy identity recovery remains a future improvement
- Motion-based discovery is no longer the primary tracking path

---

## Roadmap

- Stabilize detector-scheduled tracking
- Evaluate DCF as an alternate local tracker backend
- evaulate ORB baseline with learned ReID embeddings
- Jetson performance profiling
- Document latency, throughput, and tracking-quality tradeoffs
- document full benchmarks for GMOT-40

---

## Repository Status

This branch captures the transition from an initial detector/motion-fusion tracker toward a modular detector-driven MOT system with local tracking support.

---

## License

MIT