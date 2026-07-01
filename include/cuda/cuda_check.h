#include <cuda_runtime_api.h>
#include <npp.h>
#include <cstdio>
#include <cstdlib>
#include <vpi/VPI.h>
#include <iostream>

#define CUDA_CHECK(expr) do {                           \
    cudaError_t cuda_result = (expr);                   \
    if(cuda_result != cudaSuccess) {                    \
        fprintf(stderr,                                 \
                "CUDA Error: %s:%i:%d = %s\n",  \
                __FILE__,                               \
                __LINE__,                               \
                cuda_result,                            \
                cudaGetErrorString(cuda_result));       \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

#define CUDA_KERNEL_CHECK() do {                        \
    CUDA_CHECK(cudaGetLastError());                     \
    CUDA_CHECK(cudaDeviceSynchronize());                \
} while(0)

#define NPP_CHECK(expr) do {                            \
    NppStatus npp_result = (expr);                      \
    if (npp_result != NPP_SUCCESS) {                    \
        fprintf(stderr,                                 \
                "NPP Error: %s:%i:%d\n",                \
                __FILE__,                               \
                __LINE__,                               \
                npp_result);                            \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

#define TRT_CHECK(expr) do {                            \
    if (!(expr)) {                                      \
        fprintf(stderr,                                 \
                "TensorRT Error: %s:%i\n",              \
                __FILE__,                               \
                __LINE__);                              \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

#define VPI_CHECK(STMT) do {                             \
    VPIStatus status = (STMT);                           \
    if (status != VPI_SUCCESS) {                         \
        char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];      \
        vpiGetLastStatusMessage(buffer, sizeof(buffer)); \
        std::cerr << vpiStatusGetName(status)            \
                    << ": " << buffer << std::endl;      \
        std::abort();                                    \
    }                                                    \
} while (0)
