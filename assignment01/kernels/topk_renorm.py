import tilelang
import tilelang.language as T
from tilelang import jit

real_kernel = None

@jit
def topk_renorm(*args):
    @T.prim_func
    def kernel(*args):
        ...
    return kernel

def run_kernel(
    probs,  # Tensor[float32]
    top_k,  # Tensor[int32]
    renorm_probs,  # Tensor[float32]
    batch_size,  # int64
    num_classes,  # int64
):
    global real_kernel
    if real_kernel is None:
        real_kernel = topk_renorm(...)
    real_kernel(probs, top_k, renorm_probs, batch_size, num_classes)