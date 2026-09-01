"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import torch
import tilelang
import tilelang.language as T
import functools

@functools.cache
def make_softmax(m: int, n: int, dtype="float32"):
    p = 1 << ((n - 1).bit_length())
    @T.prim_func
    def main(x: T.Tensor((m, n), dtype), out: T.Tensor((m, n), dtype)):
        with T.Kernel(m, threads=128) as b:
            row_shared = T.alloc_fragment((1, p), dtype)
            max_shared = T.alloc_fragment((1,), dtype)
            sum_shared = T.alloc_fragment((1,), dtype)
            for i in T.Parallel(p):
                row_shared[0, i] = T.if_then_else(i < n, x[b, i], -T.infinity(dtype))
            T.reduce_max(row_shared, max_shared)
            for i in T.Parallel(p):
                row_shared[0, i] = T.exp(row_shared[0, i] - max_shared[0])
            T.reduce_sum(row_shared, sum_shared)
            for i in T.Parallel(p):
                row_shared[0, i] /= sum_shared[0]
            for i in T.Parallel(p):
                if i < n:
                    out[b, i] = row_shared[0, i]
    
    return main
            

def softmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    func = make_softmax(m, n)
    kernel = tilelang.compile(func, out_idx=1)
    return kernel(x)

def main():
    import torch
    x = torch.randn((32, 255), device="cuda", dtype=torch.float32)
    y = softmax(x)
    y_exp = torch.softmax(x, dim=-1)
    torch.testing.assert_close(y, y_exp)

if __name__ == "__main__":
    main()