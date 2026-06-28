cmake_minimum_required(VERSION 3.18)

project(cuMoT LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(datm
    datm/src/appearance_model.cpp
    datm/src/data_association.cpp
    datm/src/detection_fuser.cpp
    datm/src/detector_engine.cpp
    datm/src/iou.cpp
    datm/src/kalman_filter.cpp
    datm/src/main.cpp
    datm/src/motion_detector.cpp
    datm/src/tracker.cpp
    datm/src/visualizer.cpp
    kernels/src/gpu_iou.cu
    kernels/src/nms.cu
    kernels/src/orb_descriptor.cu
)

target_include_directories(datm PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    /opt/nvidia/cvcuda0/include
    /opt/nvidia/vpi3/include
)

target_link_directories(datm PRIVATE
    /opt/nvidia/cvcuda0/lib
    /opt/nvidia/vpi3/lib/aarch64-linux-gnu
)

target_link_libraries(datm PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer
    cvcuda
    nvcv_types
    nvvpi
)