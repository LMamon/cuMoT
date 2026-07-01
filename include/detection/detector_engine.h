#pragma once

#include <string>
#include <vector>

#include "association/appearance_model.h"
#include "NvInfer.h"


#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

enum class DetectionSource {
    DETECTOR,
    MOTION,
    LOCAL_TRACKER,
    FUSED
};

struct Box {
    float cx, cy, w, h;
};

struct Detection {
    AppearanceModel appearance;
    Box box;
    DetectionSource source;
    
    float confidence;
    int classId;
};

inline float left(const Box& b) { return b.cx - b.w * 0.5f; }
inline float top(const Box& b) { return b.cy - b.h * 0.5f; }
inline float right(const Box& b) { return b.cx + b.w * 0.5f; }
inline float bottom(const Box& b) { return b.cy + b.h * 0.5f; }

class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
};

class DetectionFuser {
    public:
        static std::vector<Detection> merge(
            const std::vector<Detection>& objDetections,
            const std::vector<Detection>& motDetections);
    
    private:
};

class DetectorEngine {
    public:
        bool loadEngine(const std::string& modelPath);
        std::vector<Detection> runInference(const cv::Mat& inputFrame);
        
        ~DetectorEngine();
        
    private:
        void* getTensorDevicePtr(nvcv::Tensor& tensor);

    private:
        Logger logger;

        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;

        std::string inputName;
        std::string outputName;

        cudaStream_t stream = nullptr;

        // output buffer +  input points directly to CV-CUDA tensor memory
        void* dOutput = nullptr;
        void* dInput = nullptr;

        // CV-CUDA source tensor
        nvcv::Tensor srcTensor;

        // TensorRT input tensor
        nvcv::Tensor trtInputTensor;

        void* dSrcFrame = nullptr;

        int srcTensorW = 0;
        int srcTensorH = 0;

        bool srcTensorAllocated = false;

        size_t outputBytes = 300 * 84 * sizeof(float);

        std::vector<float> hostOutput;

        // fused preprocessing operator
        cvcuda::ResizeCropConvertReformat preprocessOp;
};
