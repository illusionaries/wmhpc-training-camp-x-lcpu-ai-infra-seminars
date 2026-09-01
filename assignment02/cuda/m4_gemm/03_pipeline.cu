// 问题 4.3(FROM-SCRATCH,模块压轴):多级缓冲流水。
//
// 从你自己的 02_tma.cu 出发,把单缓冲扩成 STAGES 级循环缓冲:TMA 往
// 前预取后续 K 段,mma 消费当前段,装载与计算重叠。STAGES 是编译参数:
//   STAGES=4 make -B run/m4_gemm/03_pipeline
// (-B 不能省:只改 -D 不改文件,make 会认为无需重编。)
//
// 明确不要求:warp specialization、persistent kernel、epilogue 融合。
// 不设达成率门槛,评分看实验与归因质量。
//
// 两个已知事实,直接告知:
//   1. smem 用量 = STAGES*(BM+BN)*BK*2,STAGES>=3 起超过 48KB 静态
//      上限,必须动态 smem + cudaFuncSetAttribute(main 已配好)。
//   2. 一条真实的流水线 hazard(我们开发答案时踩到的,写出来让你避开):
//      "机会式预取"(try_wait 非阻塞,空了就发)不能替代"强制发射"。
//      若本轮要消费的那段 TMA 在早先检查时 stage 未空而被跳过,后面
//      wait full 等的就是一条从未发出的拷贝——死锁。症状签名很典型:
//      1024^3 侥幸全过,4096^3 必挂(13 万次机会必中一次)。正确结构:
//      本轮要消费的 TMA 用阻塞等 empty 保证发出,机会式 try_wait 只
//      用于更深的预取。另外 empty mbarrier 必须每 stage 一个:单个
//      mbar 的 parity 区分不了相隔 2 轮的完成,STAGES>=2 必然歧义。
//
// 交付:
//   - 梯子表第三行(4096^3,默认 STAGES=3)
//   - stages 扫描表:S ∈ {2,3,4,6},在两个形状上各扫一遍——4096^3 与
//     M=256 N=4096 K=16384(小 grid、长 K)。两张表的 S 敏感度不一样,
//     解释差异来自什么(提示方向:每 SM 常驻 block 数怎么随 smem 用量
//     变、块间并发本身能隐藏多少延迟)。./sweep_stages.sh 会跑全表
//   - 流水时空图:任选一个 S,画出稳态下 TMA/mma 在各 stage 上的重叠
//   - handout 4.3 的三问:瓶颈移动;梯子表逐级归因(含 assignment01
//     的 naive matmul 同口径对照);smem 与 TMEM 谁先顶住扩 stage/tile
//
// 运行:make run/m4_gemm/03_pipeline;./bin/m4_gemm/03_pipeline M N K
#include "../common.h"
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
constexpr int NSTAGE = STAGES;
constexpr int STAGE_CP_BYTES = (BM + BN) * BK * 2;

namespace ptx = cuda::ptx;

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

__device__ inline void mbar_wait_blocking(uint64_t *mbar, uint32_t phase) {
  //   uint32_t done = 0;
  //   while (!done)
  //     asm volatile("{\n.reg .pred p;\n"
  //                  "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
  //                  "selp.b32 %0, 1, 0, p;\n}"
  //                  : "=r"(done)
  //                  : "r"(mbar), "r"(phase));
  bool done = 0;
  while (!done) {
    done = ptx::mbarrier_try_wait_parity(mbar, phase);
  }
}

// 非阻塞版:成功返回 true。机会式深预取用它。
__device__ inline bool mbar_try(uint32_t mbar, uint32_t phase) {
  uint32_t done;
  asm volatile("{\n.reg .pred p;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "selp.b32 %0, 1, 0, p;\n}"
               : "=r"(done)
               : "r"(mbar), "r"(phase));
  return done;
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

__global__ void gemm_pipeline(const __nv_bfloat16 *gA, const __nv_bfloat16 *gB,
                              float *gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem = (uint8_t *)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);

  // TODO:把你 4.2 的 kernel 扩成 NSTAGE 级流水。参考结构:
  // (1) smem 划成 NSTAGE 段,stage s 的 A/B 起点自己排;mbarrier 每
  //     stage 两个:full[s](TMA 到达)、empty[s](mma 消费完成)
  // (2) 预热:先发 min(NSTAGE, iters) 轮 TMA(发第 it 轮 = 对 stage
  //     it%NSTAGE 做 arrive.expect_tx + 两条 cp.async.bulk.tensor)
  // (3) 主循环 it:
  //     - 强制发射:若第 it 轮 TMA 还没发,阻塞等 empty[it%NSTAGE]
  //       后补发(见文件头 hazard;empty 的 parity 按该 stage 被复用
  //       的轮次算,第一次复用等的是上一轮使用的完成)
  //     - 机会式深预取:try_wait 下一个待发 stage 的 empty,成功就
  //       继续发,失败立刻停,不许阻塞
  //     - 等 full[it%NSTAGE](parity = (it/NSTAGE)&1)→ tcgen05.fence
  //       → mma(与 4.2 相同,累加位口径不变)→ commit 到
  //       empty[it%NSTAGE]
  // (4) drain:等最后一轮 mma 的 empty 到达,再进 epilogue
  uint8_t *sA_base = smem;
  uint8_t *sB_base = smem + BM * BK * 2 * NSTAGE;
  uint32_t sA_base_shared = __cvta_generic_to_shared(sA_base);
  uint32_t sB_base_shared = __cvta_generic_to_shared(sB_base);

  auto warp_id = threadIdx.x >> 5;
  auto lane_id = threadIdx.x & 31;

  __shared__ __align__(8) uint64_t mbar_full[NSTAGE];
  __shared__ __align__(8) uint64_t mbar_empty[NSTAGE];
  __shared__ uint32_t tmem_base;

  if (warp_id == 0) {
    if (lane_id == 0) {
      for (int i = 0; i < NSTAGE; i++) {
        ptx::mbarrier_init(&mbar_full[i], 1);
        ptx::mbarrier_init(&mbar_empty[i], 1);
        ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster);
      }
    }
    ptx::tcgen05_alloc(ptx::cta_group_1, &tmem_base, 64);
    ptx::tcgen05_relinquish_alloc_permit(ptx::cta_group_1);
  }

  auto tileM = blockIdx.x * BM;
  auto tileN = blockIdx.y * BN;
  auto iters = K / BK;

  auto warmup_iters = min(iters, NSTAGE);
  auto elected = is_elected();
  if (elected) {
    for (int it = 0; it < warmup_iters; it++) {
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                     ptx::space_shared, &mbar_full[it],
                                     STAGE_CP_BYTES);
      int32_t a_coords[2] = {it * BK, (int32_t)tileM};
      int32_t b_coords[2] = {it * BK, (int32_t)tileN};
      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                sA_base + BM * BK * 2 * it, &tmapA, a_coords,
                                &mbar_full[it]);
      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                sB_base + BN * BK * 2 * it, &tmapB, b_coords,
                                &mbar_full[it]);
    }
  }
  __syncthreads();
  auto next_stage = warmup_iters;

  for (int it = 0; it < iters; it++) {
    auto idx = it % NSTAGE;
    auto round = it / NSTAGE;
    if (it == next_stage) {
      mbar_wait_blocking(&mbar_empty[idx], (round & 1) ^ 1);
      if (elected) {
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                       ptx::space_shared, &mbar_full[idx],
                                       STAGE_CP_BYTES);
        int32_t a_coords[2] = {it * BK, (int32_t)tileM};
        int32_t b_coords[2] = {it * BK, (int32_t)tileN};
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                  sA_base + BM * BK * 2 * idx, &tmapA, a_coords,
                                  &mbar_full[idx]);
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                  sB_base + BN * BK * 2 * idx, &tmapB, b_coords,
                                  &mbar_full[idx]);
      }
      next_stage += 1;
    } // forced TMA

    __syncthreads();
    if (next_stage < iters &&
        ptx::mbarrier_try_wait_parity(&mbar_empty[next_stage % NSTAGE],
                                      ((next_stage / NSTAGE) & 1) ^ 1)) {
      if (elected) {
        auto next_stage_idx = next_stage % NSTAGE;
        ptx::mbarrier_arrive_expect_tx(
            ptx::sem_release, ptx::scope_cta, ptx::space_shared,
            &mbar_full[next_stage_idx], STAGE_CP_BYTES);
        int32_t a_coords[2] = {next_stage * BK, (int32_t)tileM};
        int32_t b_coords[2] = {next_stage * BK, (int32_t)tileN};
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                  sA_base + BM * BK * 2 * next_stage_idx,
                                  &tmapA, a_coords, &mbar_full[next_stage_idx]);
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                  sB_base + BN * BK * 2 * next_stage_idx,
                                  &tmapB, b_coords, &mbar_full[next_stage_idx]);
      }
      next_stage += 1;
    } // prefetch

    __syncthreads();
    mbar_wait_blocking(&mbar_full[idx], round & 1);
    if (elected) {
      ptx::tcgen05_fence_after_thread_sync();
      for (int k = 0; k < BK; k += 16) {
        auto a_desc = make_desc_sm100(
            sA_base_shared + BM * BK * 2 * idx + k * 2, 1, 1024, 2);
        auto b_desc = make_desc_sm100(
            sB_base_shared + BN * BK * 2 * idx + k * 2, 1, 1024, 2);
        auto idesc = make_idesc(1, 1, 1, BN, BM);
        ptx::tcgen05_mma(ptx::kind_f16, ptx::cta_group_1, tmem_base, a_desc,
                         b_desc, idesc, it + k > 0);
      }
      ptx::tcgen05_commit(ptx::cta_group_1, &mbar_empty[idx]);
    } // MMA
  } // it = 0 .. (K / BK)

  __syncthreads();
  for (int it = iters - warmup_iters; it < iters; it++) {
    auto idx = it % NSTAGE;
    auto round = it / NSTAGE;
    mbar_wait_blocking(&mbar_empty[idx], round & 1);
  }

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
    for (int i = 0; i < 8; i++) {
      gD[(tileM + threadIdx.x) * N + tileN + n + i] = r[i];
    }
  }

  __syncthreads();
  if (warp_id == 0) {
    ptx::tcgen05_dealloc(ptx::cta_group_1, tmem_base, 64);
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
  uint32_t boxDimB[2] = {BK, BN};
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
  auto launch = [&] {
    gemm_pipeline<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA, tmapB);
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
  printf("[4.3 pipeline S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
         "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
         NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
         100.0 * tflops / cub_tflops);
  cublasDestroy(h);
  return bad != 0;
}

// clang-format off
// == 形状 4096 4096 4096 ==
// [4.3 pipeline S=2] M=4096 N=4096 K=4096  PASS(bad=0)  0.25 ms  548.3 TFLOPS  (cuBLAS 1767.0, 达成率 31%)
// [4.3 pipeline S=3] M=4096 N=4096 K=4096  PASS(bad=0)  0.24 ms  566.8 TFLOPS  (cuBLAS 1768.1, 达成率 32%)
// [4.3 pipeline S=4] M=4096 N=4096 K=4096  PASS(bad=0)  0.26 ms  528.4 TFLOPS  (cuBLAS 1768.2, 达成率 30%)
// [4.3 pipeline S=6] M=4096 N=4096 K=4096  PASS(bad=0)  0.41 ms  332.9 TFLOPS  (cuBLAS 1769.1, 达成率 19%)
// == 形状 256 4096 16384 ==
// [4.3 pipeline S=2] M=256 N=4096 K=16384  PASS(bad=0)  0.15 ms  233.1 TFLOPS  (cuBLAS 926.9, 达成率 25%)
// [4.3 pipeline S=3] M=256 N=4096 K=16384  PASS(bad=0)  0.10 ms  329.0 TFLOPS  (cuBLAS 926.6, 达成率 36%)
// [4.3 pipeline S=4] M=256 N=4096 K=16384  PASS(bad=0)  0.09 ms  381.3 TFLOPS  (cuBLAS 925.1, 达成率 41%)
// [4.3 pipeline S=6] M=256 N=4096 K=16384  PASS(bad=0)  0.09 ms  380.9 TFLOPS  (cuBLAS 924.9, 达成率 41%)