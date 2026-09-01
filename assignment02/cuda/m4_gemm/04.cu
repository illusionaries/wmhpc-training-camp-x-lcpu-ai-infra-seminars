#include "../common.h"
#include <cooperative_groups.h>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_bf16.h>
#include <random>
#include <vector>

#ifndef STAGES
#define STAGES 3
#endif

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int BROWS = BN / 2;
constexpr int NSTAGE = STAGES;

// 每个 CTA 自己搬运的 A + B 数据量:
// A: BM x BK, B: (BN/2) x BK
constexpr int STAGE_CP_BYTES = (BM + BROWS) * BK * 2;

// [FIX 1] 2SM/2CTA TMA 的 leader full barrier 统计两个 CTA 的总 transaction
// bytes.
constexpr int PAIR_STAGE_CP_BYTES = 2 * STAGE_CP_BYTES;

namespace ptx = cuda::ptx;
namespace cg = cooperative_groups;

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)layout << 61;
  return d;
}

__device__ inline uint32_t make_idesc(uint8_t dtype, uint8_t atype,
                                      uint8_t btype, uint16_t n, uint16_t m) {
  uint32_t d = 0;
  d |= dtype << 4;
  d |= atype << 7;
  d |= btype << 10;
  d |= (n >> 3) << 17;
  d |= (m >> 4) << 24;
  return d;
}

__device__ inline void mbar_wait_blocking(uint64_t *mbar, uint32_t phase) {
  bool done = false;
  while (!done) {
    done = ptx::mbarrier_try_wait_parity(mbar, phase);
  }
}

__device__ inline bool is_elected() {
  auto warp_id = threadIdx.x >> 5;
  uint32_t elected;
  asm volatile("{\n"
               ".reg .pred p;\n"
               "elect.sync _|p, 0xFFFFFFFF;\n"
               "selp.b32 %0, 1, 0, p;\n"
               "}"
               : "=r"(elected)::"memory");
  return warp_id == 0 && elected;
}

_CCCL_DEVICE static inline void
cp_async_bulk_tensor(ptx::space_cluster_t, ptx::space_global_t,
                     ptx::cta_group_2_t, void *__dstMem,
                     const void *__tensorMap,
                     const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
                     _CUDA_VSTD::uint64_t *__smem_bar) {
  // .shared::cta -> .shared::cluster
  asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes.cta_group::2 [%0], [%1, {%2, "
      "%3}], [%4];"
      :
      : "r"((uint32_t)__cvta_generic_to_shared(__dstMem)), "l"(__tensorMap),
        "r"(__tensorCoords[0]), "r"(__tensorCoords[1]),
        "r"((uint32_t)__cvta_generic_to_shared(__smem_bar))
      : "memory");
}

__global__ void gemm_pipeline(const __nv_bfloat16 *gA, const __nv_bfloat16 *gB,
                              float *gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
  auto cluster = cg::this_cluster();
  const auto block_rank_in_cluster = cluster.block_rank();
  const auto leader_cta = (block_rank_in_cluster == 0);

  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem = (uint8_t *)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

  uint8_t *sA_base = smem;
  uint8_t *sB_base = smem + BM * BK * 2 * NSTAGE;
  uint32_t sA_base_shared = __cvta_generic_to_shared(sA_base);
  uint32_t sB_base_shared = __cvta_generic_to_shared(sB_base);

  const auto warp_id = threadIdx.x >> 5;
  const auto lane_id = threadIdx.x & 31;

  // only mbar_full of leader CTA is used
  __shared__ __align__(8) uint64_t mbar_full[NSTAGE];
  __shared__ __align__(8) uint64_t mbar_empty[NSTAGE];
  __shared__ uint32_t tmem_base;

  if (warp_id == 0) {
    if (lane_id == 0) {
#pragma unroll
      for (int i = 0; i < NSTAGE; i++) {
        ptx::mbarrier_init(&mbar_full[i], 1);
        ptx::mbarrier_init(&mbar_empty[i], 1);
        ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster);
      }
    }
    ptx::tcgen05_alloc(ptx::cta_group_2, &tmem_base, 64);
    ptx::tcgen05_relinquish_alloc_permit(ptx::cta_group_2);
  }

  // peer CTA launched, TMEM and mbarrier ready.
  cluster.sync();

  uint64_t *pair_full =
      leader_cta ? mbar_full : cluster.map_shared_rank(mbar_full, 0);

  const auto tileM = blockIdx.x * BM;
  const auto tileN = blockIdx.y * BN;
  const auto iters = K / BK;

  const int warmup_iters = min(iters, NSTAGE);
  const bool elected = is_elected();
  if (elected) {
    for (int it = 0; it < warmup_iters; it++) {
      if (leader_cta) {
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                       ptx::space_shared, &mbar_full[it],
                                       PAIR_STAGE_CP_BYTES);
      }
      int32_t a_coords[2] = {it * BK, (int32_t)tileM};
      int32_t b_coords[2] = {it * BK,
                             (int32_t)(tileN + block_rank_in_cluster * BROWS)};

      cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                           ptx::cta_group_2, sA_base + BM * BK * 2 * it, &tmapA,
                           a_coords, &pair_full[it]);

      cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                           ptx::cta_group_2, sB_base + BROWS * BK * 2 * it,
                           &tmapB, b_coords, &pair_full[it]);
    }
  }

  auto next_stage = warmup_iters;

  for (int it = 0; it < iters; it++) {
    auto idx = it % NSTAGE;
    auto round = it / NSTAGE;

    if (elected) {
      if (it == next_stage) {
        mbar_wait_blocking(&mbar_empty[idx], (round & 1) ^ 1);

        if (leader_cta) {
          ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                         ptx::space_shared, &mbar_full[idx],
                                         PAIR_STAGE_CP_BYTES);
        }

        int32_t a_coords[2] = {it * BK, (int32_t)tileM};
        int32_t b_coords[2] = {
            it * BK, (int32_t)(tileN + block_rank_in_cluster * BROWS)};

        cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                             ptx::cta_group_2, sA_base + BM * BK * 2 * idx,
                             &tmapA, a_coords, &pair_full[idx]);

        cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                             ptx::cta_group_2, sB_base + BROWS * BK * 2 * idx,
                             &tmapB, b_coords, &pair_full[idx]);

        next_stage += 1;
      } // forced TMA

      if (next_stage < iters &&
          ptx::mbarrier_try_wait_parity(&mbar_empty[next_stage % NSTAGE],
                                        ((next_stage / NSTAGE) & 1) ^ 1)) {
        auto next_stage_idx = next_stage % NSTAGE;

        if (leader_cta) {
          ptx::mbarrier_arrive_expect_tx(
              ptx::sem_release, ptx::scope_cta, ptx::space_shared,
              &mbar_full[next_stage_idx], PAIR_STAGE_CP_BYTES);
        }

        int32_t a_coords[2] = {next_stage * BK, (int32_t)tileM};
        int32_t b_coords[2] = {
            next_stage * BK, (int32_t)(tileN + block_rank_in_cluster * BROWS)};

        cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                             ptx::cta_group_2,
                             sA_base + BM * BK * 2 * next_stage_idx, &tmapA,
                             a_coords, &pair_full[next_stage_idx]);

        cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                             ptx::cta_group_2,
                             sB_base + BROWS * BK * 2 * next_stage_idx, &tmapB,
                             b_coords, &pair_full[next_stage_idx]);

        next_stage += 1;
      } // prefetch

      if (leader_cta) {
        mbar_wait_blocking(&mbar_full[idx], round & 1);
        ptx::tcgen05_fence_after_thread_sync();

#pragma unroll
        for (int k = 0; k < BK; k += 16) {
          auto a_desc = make_desc_sm100(
              sA_base_shared + BM * BK * 2 * idx + k * 2, 1, 1024, 2);
          auto b_desc = make_desc_sm100(
              sB_base_shared + BROWS * BK * 2 * idx + k * 2, 1, 1024, 2);
          auto idesc = make_idesc(1, 1, 1, BN, BM * 2);
          ptx::tcgen05_mma(ptx::kind_f16, ptx::cta_group_2, tmem_base, a_desc,
                           b_desc, idesc, it + k > 0);
        }
        ptx::tcgen05_commit_multicast(ptx::cta_group_2, &mbar_empty[idx], 0b11);
      }
    } // MMA
  } // it = 0 .. (K / BK)

  __syncthreads();
  for (int it = iters - warmup_iters; it < iters; it++) {
    const auto idx = it % NSTAGE;
    const auto round = it / NSTAGE;
    mbar_wait_blocking(&mbar_empty[idx], round & 1);
  } // drain

  __syncthreads();
  ptx::tcgen05_fence_after_thread_sync();
  float r[8];
  for (int n = 0; n < BN; n += 8) {
    uint32_t tmem_ld_base = tmem_base + ((warp_id * 32) << 16) + n;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]),
                   "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
                 : "r"(tmem_ld_base));
    ptx::tcgen05_wait_ld();

#pragma unroll
    for (int i = 0; i < 8; i++) {
      gD[(tileM + threadIdx.x) * N + tileN + n + i] = r[i];
    }
  }

  // make sure peer CTA has finished TMEM usage
  cluster.sync();
  if (warp_id == 0) {
    ptx::tcgen05_dealloc(ptx::cta_group_2, tmem_base, 64);
  }
}

int main(int argc, char **argv) {
  int M = argc > 3 ? atoi(argv[1]) : 4096;
  int N = argc > 3 ? atoi(argv[2]) : 4096;
  int K = argc > 3 ? atoi(argv[3]) : 4096;
  if (M % (BM * 2) || N % BN || K % BK) {
    printf("形状需按 %dx%dx%d 对齐\n", BM * 2, BN, BK);
    return 1;
  }
  size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(-3, 3);
  std::vector<__nv_bfloat16> hA(nA), hB(nB);
  for (auto &v : hA)
    v = __float2bfloat16((float)dist(rng));
  for (auto &v : hB)
    v = __float2bfloat16((float)dist(rng));
  __nv_bfloat16 *dA, *dB;
  float *dD, *dRef;
  CUDA_CHECK(cudaMalloc(&dA, nA * 2));
  CUDA_CHECK(cudaMalloc(&dB, nB * 2));
  CUDA_CHECK(cudaMalloc(&dD, nD * 4));
  CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

  // TODO:tensor map 从你的 4.2 原样复制。
  CUtensorMap tmapA = {}, tmapB = {};
  uint64_t globalDimA[2] = {(uint64_t)K,
                            (uint64_t)M}; // fastest moving dim comes first
  uint64_t globalStridesA[1] = {(uint64_t)K * sizeof(__nv_bfloat16)};
  uint32_t boxDimA[2] = {BK, BM};
  uint32_t elementStridesDense[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &tmapA, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dA,
      globalDimA, globalStridesA, boxDimA, elementStridesDense,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  uint64_t globalDimB[2] = {(uint64_t)K, (uint64_t)N};
  uint64_t globalStridesB[1] = {(uint64_t)K * sizeof(__nv_bfloat16)};
  uint32_t boxDimB[2] = {BK, BROWS};
  CU_CHECK(cuTensorMapEncodeTiled(
      &tmapB, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dB,
      globalDimB, globalStridesB, boxDimB, elementStridesDense,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  dim3 grid(M / BM, N / BN);
  // NSTAGE=3 时 72KB+对齐余量,超 48KB 静态上限,动态 smem 必须。
  size_t smemBytes = (size_t)NSTAGE * STAGE_CP_BYTES + 1024;
  CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)smemBytes));

  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {2, 1, 1};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smemBytes;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;

  auto launch = [&] {
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, gemm_pipeline, dA, dB, dD, M, N, K,
                                  tmapA, tmapB));
  };
  launch();
  CUDA_CHECK_KERNEL();

  cublasHandle_t h;
  cublasCreate(&h);
  float alpha = 1.f, beta = 0.f;
  cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16BF, K,
               dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(nD), ref(nD);
  CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
  long bad = 0;
  for (size_t i = 0; i < nD; i++)
    bad += got[i] != ref[i];

  int iters = (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
  float ms = time_avg_ms(launch, iters);
  double tflops = 2.0 * M * N * K / (ms * 1e9);
  float cub_ms = time_avg_ms(
      [&] {
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                     CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                     CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      },
      iters);
  double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
  printf("[4.4 2-CTA pipeline S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
         "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
         NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
         100.0 * tflops / cub_tflops);
  cublasDestroy(h);
  return bad != 0;
}

// clang-format off
// == 形状 4096 4096 4096 ==
// [4.4 2-CTA pipeline S=2] M=4096 N=4096 K=4096  PASS(bad=0)  0.24 ms  577.0 TFLOPS  (cuBLAS 1766.9, 达成率 33%)
// [4.4 2-CTA pipeline S=3] M=4096 N=4096 K=4096  PASS(bad=0)  0.24 ms  577.5 TFLOPS  (cuBLAS 1764.6, 达成率 33%)
// [4.4 2-CTA pipeline S=4] M=4096 N=4096 K=4096  PASS(bad=0)  0.23 ms  590.7 TFLOPS  (cuBLAS 1764.8, 达成率 33%)
// [4.4 2-CTA pipeline S=6] M=4096 N=4096 K=4096  PASS(bad=0)  0.34 ms  403.8 TFLOPS  (cuBLAS 1762.1, 达成率 23%)
// == 形状 256 4096 16384 ==
// [4.4 2-CTA pipeline S=2] M=256 N=4096 K=16384  PASS(bad=0)  0.15 ms  224.6 TFLOPS  (cuBLAS 924.3, 达成率 24%)
// [4.4 2-CTA pipeline S=3] M=256 N=4096 K=16384  PASS(bad=0)  0.14 ms  238.5 TFLOPS  (cuBLAS 925.1, 达成率 26%)
// [4.4 2-CTA pipeline S=4] M=256 N=4096 K=16384  PASS(bad=0)  0.10 ms  342.3 TFLOPS  (cuBLAS 925.1, 达成率 37%)
// [4.4 2-CTA pipeline S=6] M=256 N=4096 K=16384  PASS(bad=0)  0.07 ms  465.8 TFLOPS  (cuBLAS 926.4, 达成率 50%)