#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t ret = (expr);                                                  \
    if (ret != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(ret),   \
              __FILE__, __LINE__, cudaGetErrorString(ret));                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

__global__ void saxpy(const float *x, float *y, int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += gridDim.x * blockDim.x) {
    y[idx] = 2 * x[idx] + y[idx];
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("usage: %s n\n", argv[0]);
    return 1;
  }

  int n = std::atoi(argv[1]);

  if (n == 0) {
    printf("SUM=0\n");
    return 0;
  }

  size_t bytes = n * sizeof(float);

  float *h_x = (float *)malloc(bytes);
  float *h_y = (float *)malloc(bytes);

  for (int i = 0; i < n; i++) {
    h_x[i] = ((i % 2048) - 1024) * 0.5f;
    h_y[i] = (i % 1024) - 512;
  }

  auto t0 = std::chrono::steady_clock::now();

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

  saxpy<<<blocks, threads>>>(d_x, d_y, n);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

  auto t1 = std::chrono::steady_clock::now();
  auto cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += h_y[i];
  }
  printf("SUM=%.0f\nmemcpy + kernel + memcpy time: %.2lfms", sum, cpu_ms);

  return 0;
}
