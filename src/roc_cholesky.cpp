#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Args {
    int n = 1024;
    int iters = 3;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            args.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            args.iters = std::atoi(argv[++i]);
        }
    }
    return args;
}

void check_hip(hipError_t status, const char* msg) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(status));
    }
}

void check_rocblas(rocblas_status status, const char* msg) {
    if (status != rocblas_status_success) {
        throw std::runtime_error(std::string(msg) + ": rocblas error");
    }
}
}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const int n = args.n;
    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);

    std::vector<double> hA(elems);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col <= row; ++col) {
            double val = dist(rng);
            hA[row * n + col] = val;
            hA[col * n + row] = val;
        }
        hA[row * n + row] += static_cast<double>(n);
    }

    rocblas_handle handle;
    check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle");

    hipStream_t stream;
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    check_rocblas(rocblas_set_stream(handle, stream), "rocblas_set_stream");

    double* dA = nullptr;
    check_hip(hipMalloc(&dA, elems * sizeof(double)), "hipMalloc dA");
    rocblas_int* dInfo = nullptr;
    check_hip(hipMalloc(&dInfo, sizeof(rocblas_int)), "hipMalloc dInfo");

    hipEvent_t start, stop;
    check_hip(hipEventCreate(&start), "hipEventCreate start");
    check_hip(hipEventCreate(&stop), "hipEventCreate stop");

    double total_ms = 0.0;
    for (int iter = 0; iter < args.iters; ++iter) {
        check_hip(hipMemcpy(dA, hA.data(), elems * sizeof(double), hipMemcpyHostToDevice),
                  "hipMemcpy H2D");
        check_hip(hipEventRecord(start, stream), "hipEventRecord start");
        check_rocblas(rocsolver_dpotrf(handle, rocblas_fill_lower, n, dA, n, dInfo),
                      "rocsolver_dpotrf");
        check_hip(hipEventRecord(stop, stream), "hipEventRecord stop");
        check_hip(hipEventSynchronize(stop), "hipEventSynchronize stop");
        float elapsed = 0.0f;
        check_hip(hipEventElapsedTime(&elapsed, start, stop), "hipEventElapsedTime");
        total_ms += static_cast<double>(elapsed);
    }

    double avg_ms = total_ms / static_cast<double>(args.iters);
    std::printf("{\"method\":\"rocsolver\",\"n\":%d,\"iters\":%d,\"time_ms\":%.6f}\n",
                n, args.iters, avg_ms);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(dInfo);
    hipFree(dA);
    hipStreamDestroy(stream);
    rocblas_destroy_handle(handle);
    return 0;
}
