// 问题 4.2(MODIFY):把 4.1 的 staging 换成 TMA,其余不动(仍单缓冲)。
//
// 从你自己的 01_tiled.cu 出发:mma 发射、epilogue、判测口径全部不变,
// 改动集中在两处——host 侧建 tensor map,kernel 侧把 st.shared staging
// 换成 cp.async.bulk.tensor + mbarrier。
//
// 直接告知的事实(工具链与布局配对,不属于考核点):
//   - tensor map 用驱动 API cuTensorMapEncodeTiled 建(Makefile 已链
//     -lcuda);kernel 参数按 const __grid_constant__ CUtensorMap 传
//   - 维度次序:dim0 是最内维(这里是 K,单位为元素数);globalStrides
//     只填外维的字节跨度 {K*2};box 是一次搬运的块 {BK, BM}(B 矩阵
//     {BK, BN});elementStrides 全 1
//   - swizzle 选 CU_TENSOR_MAP_SWIZZLE_128B:TMA 硬件落进 smem 的布局
//     与你 4.1 手工 swz128 摆出来的完全相同,descriptor 一个字段都
//     不用改;interleave/L2 promotion/oob fill 都取 NONE
//   - fence 口径(2.1(b) 在这里兑现):TMA 写 smem 与 tcgen05 读 smem
//     都走 async proxy,fence.proxy.async 不再需要;mbar_wait 之后的
//     tcgen05.fence::after_thread_sync 仍然要
//
// 交付:PASS + 梯子表第二行;回答 handout 4.2 的问题(相对 4.1 的提升
// 为什么这么大——4.1 的 staging 成本由什么构成,用 ncu 佐证)。
//
// 运行:make run/m4_gemm/02_tma;自定形状 ./bin/m4_gemm/02_tma M N K
#include "../common.h"
#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda/ptx>
#include <cuda_bf16.h>
#include <random>
#include <vector>

namespace ptx = cuda::ptx;

constexpr int BM = 128, BN = 64, BK = 64;

// SM100 smem descriptor(与 4.1 相同;swz128 已经不需要了)。
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
                                      uint8_t btype, uint8_t n, uint8_t m) {
  uint32_t d = 0;
  d |= dtype << 4;
  d |= atype << 7;
  d |= btype << 10;
  d |= (n >> 3) << 17;
  d |= (m >> 4) << 24;
  return d;
}

__device__ inline void mbar_wait_parity(uint64_t *mbar, uint32_t phase) {
  uint32_t done = 0;
  while (!done)
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.b32 %0, 1, 0, p;\n}"
                 : "=r"(done)
                 : "l"(mbar), "r"(phase));
}

__device__ inline bool is_elected() {
  auto warp_id = threadIdx.x >> 5;
  uint32_t elected;
  asm volatile("{\n"
               ".reg .pred p;\n"
               "elect.sync _|p, 0xFFFFFFFF;\n"
               "selp.b32 %0, 1, 0, p;\n"
               "}\n"
               : "=r"(elected));
  return (warp_id == 0 && elected);
}

__global__ void gemm_tma(const __nv_bfloat16 *gA, const __nv_bfloat16 *gB,
                         float *gD, int M, int N, int K,
                         const __grid_constant__ CUtensorMap tmapA,
                         const __grid_constant__ CUtensorMap tmapB) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem = (uint8_t *)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

  // TODO:把你 4.1 的 kernel 搬进来,K 循环的 staging 部分改为:
  // (1) 多初始化一组 mbarrier:full(TMA 到达)。4.1 里等 mma 消费
  //     完成的那个继续当 empty 用
  // (2) 每轮:除首轮外先等 empty(smem 可覆写)→ 单线程发 TMA →
  //     等 full → mma(与 4.1 相同)→ commit
  //     发 TMA = 一条 mbarrier.arrive.expect_tx(字节数一次报满
  //     (BM+BN)*BK*2,A、B 两条拷贝共用一个 mbar)+ 两条
  //     cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::
  //     complete_tx::bytes,坐标次序与 tensor map 的维度次序一致:
  //     A 是 {it*BK, tileM},B 是 {it*BK, tileN}
  // (3) 删掉 st.shared staging、swz128、fence.proxy.async(见文件头)
  // full/empty 的 parity 都随轮次翻转,想清楚各自翻转的节奏。

  uint8_t *sA = smem;
  uint8_t *sB = smem + BM * BK * 2;

  auto warp_id = threadIdx.x >> 5;
  auto lane_id = threadIdx.x & 31;

  __shared__ __align__(8) uint64_t mbar_empty;
  __shared__ __align__(8) uint64_t mbar_full;
  __shared__ uint32_t tmem_addr;
  auto mbar_empty_shared_addr =
      (uint64_t *)__cvta_generic_to_shared(&mbar_empty);
  auto mbar_full_shared_addr = (uint64_t *)__cvta_generic_to_shared(&mbar_full);
  auto tmem_addr_shared_addr = (uint32_t *)__cvta_generic_to_shared(&tmem_addr);

  if (warp_id == 0) {
    if (lane_id == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"l"(
                       mbar_empty_shared_addr),
                   "r"(1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"l"(
                       mbar_full_shared_addr),
                   "r"(1));
      asm volatile("fence.mbarrier_init.release."
                   "cluster;"); // mem access ordering
    }

    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "l"(tmem_addr_shared_addr),
        "r"(BN));
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); // no
                                                                       // further
                                                                       // allocations
  }

  auto tileM = blockIdx.x * BM;
  auto tileN = blockIdx.y * BN;
  auto phase = 0;
  for (int it = 0; it < K / BK; it++) {
    if (is_elected()) {
      int32_t tensor_coords_a[2] = {it * BK, (int32_t)tileM};
      int32_t tensor_coords_b[2] = {it * BK, (int32_t)tileN};
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                     ptx::space_shared, &mbar_full,
                                     (BM + BN) * BK * sizeof(__nv_bfloat16));
      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, sA,
                                &tmapA, tensor_coords_a, &mbar_full);
      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, sB,
                                &tmapB, tensor_coords_b, &mbar_full);
    }

    // (b) fence.proxy.async + __syncthreads
    // asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();
    mbar_wait_parity(mbar_full_shared_addr, phase);

    auto a_base = __cvta_generic_to_shared(sA);
    auto b_base = __cvta_generic_to_shared(sB);

    if (is_elected()) {
      asm volatile("tcgen05.fence::after_thread_sync;");
      for (int k = 0; k < BK; k += 16) {
        auto a_desc = make_desc_sm100(a_base + k * 2, 1, 1024, 2);
        auto b_desc = make_desc_sm100(b_base + k * 2, 1, 1024, 2);
        auto i_desc = make_idesc(1, 1, 1, BN, BM);
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
                     "}\n" ::"r"(tmem_addr),
                     "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(k + it));
      }
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                   "cluster.b64 [%0];" ::"l"(mbar_empty_shared_addr)
                   : "memory");
    }
    mbar_wait_parity(mbar_empty_shared_addr, phase);
    phase ^= 1;
  }

  asm volatile("tcgen05.fence::after_thread_sync;");
  float r[8];
  for (int n = 0; n < BN; n += 8) {
    auto tmem_src = tmem_addr + (uint32_t)((warp_id * 32) << 16) + n;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]),
                   "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
                 : "r"(tmem_src));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    for (int i = 0; i < 8; i++) {
      auto row = tileM + threadIdx.x;
      auto col = tileN + n + i;
      gD[row * N + col] = r[i];
    }
  }

  __syncthreads();
  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(
                     tmem_addr),
                 "r"(BN));
  }
}

int main(int argc, char **argv) {
  int M = argc > 3 ? atoi(argv[1]) : 4096;
  int N = argc > 3 ? atoi(argv[2]) : 4096;
  int K = argc > 3 ? atoi(argv[3]) : 4096;
  if (M % BM || N % BN || K % BK) {
    printf("形状需按 %dx%dx%d 对齐\n", BM, BN, BK);
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

  // TODO:cuTensorMapEncodeTiled 建 tmapA/tmapB(参数要点见文件头;
  // 返回值要检查,CUDA_SUCCESS 之外一律报错退出——tensor map 参数错
  // 的典型症状是 kernel 静默读到 0 或越界,而不是启动失败)。
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
  uint32_t boxDimB[2] = {BK, BN};
  CU_CHECK(cuTensorMapEncodeTiled(
      &tmapB, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dB,
      globalDimB, globalStridesB, boxDimB, elementStridesDense,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  dim3 grid(M / BM, N / BN);
  size_t smemBytes = (size_t)(BM + BN) * BK * 2 + 1024;
  CUDA_CHECK(cudaFuncSetAttribute(
      gemm_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes));
  auto launch = [&] {
    gemm_tma<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA, tmapB);
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
  printf("[4.2 tma] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f TFLOPS  "
         "(cuBLAS %.1f, 达成率 %.0f%%)\n",
         M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
         100.0 * tflops / cub_tflops);
  cublasDestroy(h);
  return bad != 0;
}

// [4.2 tma] M=4096 N=4096 K=4096  PASS(bad=0)  0.26 ms  520.1 TFLOPS  (cuBLAS 1758.9, 达成率 30%)