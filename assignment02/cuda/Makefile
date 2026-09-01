# 用法(在 cuda/ 目录下执行):
#   make bin/m5_lowprec/03a_encode_check    编译单个练习
#   make run/m5_lowprec/03a_encode_check    编译并运行
#   ARCH=89 make ...                        指定目标架构(默认 100f)
#
# ARCH 取值举例:100f(B200/B300 家族,默认)、103a(B300 专属指令)、
# 90a(Hopper)、89(Ada)、120a(5090)。
#
# 本作业统一用显式 -gencode 而不用 -arch=sm_XXXa 简写:CUDA 13.0 下简写
# 会把用户代码的 PTX target 展开成不带后缀的 compute_XXX,arch-specific
# 指令(如 e2m1 的 cvt)会被 ptxas 拒绝,报错还误导为"指令不支持"。

NVCC  ?= nvcc
ARCH  ?= 100f
FLAGS ?= -O2 -std=c++17 -I. --expt-relaxed-constexpr -g -lineinfo -D__CUDA_ARCH_FEAT_SM101_ALL

# 问题 4.3 的流水级数:STAGES=4 make -B bin/m4_gemm/03_pipeline
# -B 不能省——只改 -D 不改源文件,make 会认为目标是新鲜的。
ifdef STAGES
FLAGS += -DSTAGES=$(STAGES)
endif

GENCODE = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

bin/%: %.cu common.h
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAGS) $(GENCODE) -o $@ $<

# 消费端验证(问题 5.3)要链 cublasLt
bin/m5_lowprec/test_fp4_gemm: m5_lowprec/test_fp4_gemm.cu common.h
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAGS) $(GENCODE) -lcublasLt -o $@ $<

# M4(m4_gemm/)要链 cublas(判测参考)与 cuda 驱动 API(tensor map)。
# 运行时若报 libcublas.so 找不到,把 CUDA 安装目录下的 lib 路径加进
# LD_LIBRARY_PATH。
bin/m4_gemm/%: m4_gemm/%.cu common.h
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAGS) $(GENCODE) -lcublas -lcuda -o $@ $<

run/%: bin/%
	./$<

ptx/%: %.cu common.h
	$(NVCC) $(FLAGS) $(GENCODE) -ptx -o $*.ptx $<
	@echo "已生成 $*.ptx"

clean:
	rm -rf bin */*.ptx

.PHONY: run/% ptx/% clean
