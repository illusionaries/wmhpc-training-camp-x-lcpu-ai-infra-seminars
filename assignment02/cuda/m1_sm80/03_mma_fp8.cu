#include "../common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp8.h>

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = 32;
constexpr int RANGE = 40;

#define __const__ __attribute__((const))

__device__ __const__ inline static int a_row_of(int lane, int i) {
  // (lane // 4) + 8 * (4 <= i < 8 || 12 <= i < 16)
  return (lane >> 2) + 8 * ((i >> 2) & 0b1);
}
__device__ __const__ inline static int a_col_of(int lane, int i) {
  // 4 * (lane % 4) + i % 4 + 16 * (i >= 8)
  return 4 * (lane & 0b11) + (i & 0b11) + 16 * (i >> 3);
}
__device__ __const__ inline static int a_idx_of(int lane, int i) {
  return a_row_of(lane, i) * K + a_col_of(lane, i);
}

__device__ __const__ inline static int b_row_of(int lane, int i) {
  // 4 * (lane % 4) + i % 4 + 16 * (i >= 4)
  return 4 * (lane & 0b11) + (i & 0b11) + 16 * (i >> 2);
} // k
__device__ __const__ inline static int b_col_of(int lane, int i) {
  // lane // 4
  return lane >> 2;
} // n
__device__ __const__ inline static int b_idx_of(int lane, int i) {
  return b_col_of(lane, i) * K + b_row_of(lane, i);
}

__global__ void mma_fp8(const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B,
                        float *D) {
  int lane = threadIdx.x;
  int gid = lane >> 2;
  int tig = lane & 0b11;

  uint32_t rA[4], rB[2];
  float rC[4] = {0}, rD[4];

#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      reinterpret_cast<__nv_fp8_e4m3 *>(&rA[i])[j] =
          A[a_idx_of(lane, 4 * i + j)];
    }
  }

#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      reinterpret_cast<__nv_fp8_e4m3 *>(&rB[i])[j] =
          B[b_idx_of(lane, 4 * i + j)];
    }
  }

  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(rD[0]), "=f"(rD[1]), "=f"(rD[2]), "=f"(rD[3])
      : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]), "r"(rB[0]), "r"(rB[1]),
        "f"(rC[0]), "f"(rC[1]), "f"(rC[2]), "f"(rC[3]));

  D[gid * 8 + 2 * tig] = rD[0];
  D[gid * 8 + 2 * tig + 1] = rD[1];
  D[(gid + 8) * 8 + 2 * tig] = rD[2];
  D[(gid + 8) * 8 + 2 * tig + 1] = rD[3];
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("usage: %s <seed>\n", argv[0]);
  }
  int seed = atoi(argv[1]);
  srand(seed);

  __nv_fp8_e4m3 hA[M * K], hB[N * K];
  float hD[M * N];
  float ref[M * N] = {};

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      hA[i * K + j] = __nv_fp8_e4m3(rand() % RANGE);
    }
  }

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      hB[j * K + i] = __nv_fp8_e4m3(rand() % RANGE);
    }
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        ref[i * N + j] += float(hA[i * K + k]) * float(hB[j * K + k]);
      }
    }
  }

  __nv_fp8_e4m3 *dA, *dB;
  float *dD;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
  CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
  CUDA_CHECK(cudaMalloc(&dD, sizeof(hD)));
  CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  mma_fp8<<<1, 32>>>(dA, dB, dD);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));

  int diff = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ?= %f\n", hD[i * N + j], ref[i * N + j]);
      if (hD[i * N + j] != ref[i * N + j]) {
        diff += 1;
        printf("mismatch at (%d, %d), expected %f, got %f\n", i, j,
               ref[i * N + j], hD[i * N + j]);
      }
    }
  }

  if (diff) {
    printf("FAIL: %d/%d mismatches\n", diff, M * N);
  } else {
    printf("PASS\n");
  }
}