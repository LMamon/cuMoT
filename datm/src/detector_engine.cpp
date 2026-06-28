#include "cuda/cuda_check.h"
#include "detection/detector_engine.h"
#include "detection/nms.h"

#include <cuda_runtime_api.h>
#include <fstream>

#include <iterator>
#include <cstdlib>
#include <random>

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

bool DetectorEngine::loadEngine(const std::string& modelPath) {
    runtime = nvinfer1::createInferRuntime(logger);
    TRT_CHECK(runtime);

    // open engine file
    std::ifstream file(modelPath, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open engine\n";
        return false;
    }

    // read engine file into memory
    std::vector<char> modelData{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

    engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
    TRT_CHECK(engine);

    inputName = engine->getIOTensorName(0);
    outputName = engine->getIOTensorName(1);
    context = engine->createExecutionContext();
    TRT_CHECK(context);

    // one stream for upload + CV-CUDA preprocessing + TensorRT inference
    CUDA_CHECK(cudaStreamCreate(&stream));

    // output buffer stays persistent
    CUDA_CHECK(cudaMalloc(&dOutput, outputBytes));

    // CPU output buffer reused every frame
    hostOutput.resize(300 * 84);

    // input tensor:
    // [1, 3, 640, 640] float32 NCHW
    //
    // CV-CUDA owns this GPU memory.
    // TensorRT will read directly from this tensor's device pointer.
    trtInputTensor = nvcv::Tensor(
        nvcv::TensorShape({1, 3, 640, 640}, "NCHW"),
        nvcv::TYPE_F32);

    dInput = getTensorDevicePtr(trtInputTensor);

    return true;
}

std::vector<Detection> DetectorEngine::runInference(const cv::Mat& inputFrame) {
    std::vector<Detection> detections;

    constexpr int INPUT_W = 640;
    constexpr int INPUT_H = 640;

    constexpr int OUTPUT_ROWS = 300;
    constexpr int OUTPUT_COLS = 84;
    constexpr int NUM_CLASSES = 80;

    const int srcW = inputFrame.cols;
    const int srcH = inputFrame.rows;

    // allocate/reallocate CV-CUDA source tensor if frame size changes
    if (!srcTensorAllocated || srcW != srcTensorW || srcH != srcTensorH) {
        srcTensor = nvcv::Tensor(
            nvcv::TensorShape({1, srcH, srcW, 3}, "NHWC"),
            nvcv::TYPE_U8);

        dSrcFrame = getTensorDevicePtr(srcTensor);

        srcTensorW = srcW;
        srcTensorH = srcH;
        srcTensorAllocated = true;
    }

    // upload OpenCV CPU frame to CV-CUDA GPU tensor
    CUDA_CHECK(cudaMemcpyAsync(dSrcFrame,
                                inputFrame.data,
                                srcW * srcH * 3 * sizeof(uint8_t),
                                cudaMemcpyHostToDevice,
                                stream));

    // fused CV-CUDA preprocessing
    preprocessOp(stream,
                srcTensor,
                trtInputTensor,
                NVCVSize2D{INPUT_W, INPUT_H},
                NVCV_INTERP_LINEAR,
                int2{0, 0},
                NVCV_CHANNEL_REVERSE,
                1.0f / 255.0f,
                0.0f,
                true);

    // bind TensorRT input/output memory
    context->setTensorAddress(inputName.c_str(), dInput);
    context->setTensorAddress(outputName.c_str(), dOutput);

    // run TensorRT inference
    TRT_CHECK(context->enqueueV3(stream));

    // copy model output back to CPU for parsing
    CUDA_CHECK(cudaMemcpyAsync(hostOutput.data(),
                                dOutput,
                                outputBytes,
                                cudaMemcpyDeviceToHost,
                                stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int row = 0; row < OUTPUT_ROWS; row++) {
        float bestScore = 0.0f;
        int bestClass = -1;

        // find highest scoring class for this detection row
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = hostOutput[row * OUTPUT_COLS + 4 + c];

            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        // skip low-confidence detections
        if (bestScore < 0.24f)
            continue;

        Detection det;

        det.box.cx = hostOutput[row * OUTPUT_COLS + 0];
        det.box.cy = hostOutput[row * OUTPUT_COLS + 1];
        det.box.w = hostOutput[row * OUTPUT_COLS + 2];
        det.box.h = hostOutput[row * OUTPUT_COLS + 3];

        det.confidence = bestScore;
        det.classId = bestClass;
        det.source = DetectionSource::DETECTOR;

        detections.push_back(det);
    }

    // std::cout << "raw detections: " << detections.size() << '\n';

    return cudaNMS(detections);
}

DetectorEngine::~DetectorEngine() {
    if (dOutput) cudaFree(dOutput);
    if (stream) cudaStreamDestroy(stream);

    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
}

void* DetectorEngine::getTensorDevicePtr(nvcv::Tensor& tensor) {
    auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    TRT_CHECK(data);
    return data->basePtr();
}
