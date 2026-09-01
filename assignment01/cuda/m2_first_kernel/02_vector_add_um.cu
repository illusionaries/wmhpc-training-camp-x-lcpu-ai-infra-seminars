// 问题 2.3：把显式内存管理改成 Unified Memory（ MODIFY ）。
// 下面是一份完整可运行的显式管理版本。任务：
//   0. 先按原样跑一次，记下耗时——这一版会被你的改动覆盖掉，// 搬运 + kernel + 读回: 29.6 ms
//      第 4 步的对比要拿它做基准；
//   1. 用 cudaMallocManaged 替换 cudaMalloc + malloc；
//   2. 删掉所有 cudaMemcpy，kernel 直接读写同一组指针，CPU 也直接读；
//   3. 想清楚哪里需要 cudaDeviceSynchronize；
//   4. 对比两版的耗时。两版的计时窗口要保持一致：分配和填数据都在窗口
//      外，窗口从"数据已经在内存里备好"开始，到 CPU 把结果全部读完为止
//      （下面用一个累加校验和的循环代表"CPU 读完全部结果"，别把它删了）。
// 改完仍要 PASS。// 33.7 ms
#include <chrono>
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1 << 24;  // 16M 元素
    size_t bytes = (size_t)n * sizeof(float);

    // 先把 CUDA context 建起来。首次调用 CUDA API 要花几百毫秒初始化，
    // 放进计时窗口会把要观察的差距完全淹掉。
    CUDA_CHECK(cudaFree(0));

    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));

    fill_random(a, n, 1);
    fill_random(b, n, 2);

    // 期望的校验和，host 上先算好，同样不计入计时。
    double want = 0;
    for (int i = 0; i < n; i++) want += (double)(a[i] + b[i]);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // ================= 计时窗口开始 =================
    auto t0 = std::chrono::steady_clock::now();

    vectorAdd<<<blocks, threads>>>(a, b, c, n);
    CUDA_CHECK_KERNEL(); // already contains cudaDeviceSynchronize

    // CPU 读完全部结果。unified memory 版里，这一步才会把结果页搬回 host。
    double got = 0;
    for (int i = 0; i < n; i++) got += (double)c[i];

    auto t1 = std::chrono::steady_clock::now();
    // ================= 计时窗口结束 =================

    printf("搬运 + kernel + 读回: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    REPORT(fabs(got - want) <= 1e-3 * (1.0 + fabs(want)));
    return 0;
}
