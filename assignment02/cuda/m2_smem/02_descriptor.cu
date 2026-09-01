// 问题 2.2:SM100 的 64 位 smem matrix descriptor。
//
// 场景都是 64x64 bf16 的 B tile(供 tcgen05 消费),三种 smem 布局:
//   场景 1:K-major,无 swizzle,tile 起始 smem 地址 0x1000。
//           canonical 布局的 atom 是 8 行 x 16B 紧密打包(128B 连续),
//           atom 沿 K、沿 MN 各按什么步距排,自己从布局推 LBO/SBO。
//   场景 2:K-major,128B swizzle,起始地址 0x2000。
//   场景 3:MN-major,128B swizzle,起始地址 0x3000。
//
// 两层任务:
//   (a) make_desc:按位域把参数编进 64 位。SM100 版位域(注意与课上
//       sm90 wgmma 版的差异):start_address bit[0,14) 右移 4;
//       LBO bit[16,30) 右移 4;SBO bit[32,46) 右移 4;
//       version bit[46,48) 固定 1;layout_type bit[61,64) 3 bit:
//       NONE=0、128B=2、64B=4、32B=6(sm90 是 2 bit 且 128B=1)。
//   (b) 三个场景各自的 LBO / SBO / layout 填进 SCEN 表。
// 提示:swizzle 模式下 LBO 被硬件忽略,按 0 填;场景 2 和场景 3 的
// 描述符会相同——报告里回答:MN-major 与 K-major 的区别去了哪里?
//
// 判测真值来自在 B300 上实际发 tcgen05 验证过的描述符(3.2 的程序用的
// 就是这三组)。运行:make run/m2_smem/02_descriptor(无卡可判)。
#include <cstdint>
#include <cstdio>

// TODO (a):实现位域编码。

union GmmaDescriptor {
  uint64_t desc;
  struct {
    uint16_t start_address : 14, : 2;       // 16
    uint16_t leading_byte_offset : 14, : 2; // 32
    uint16_t stride_byte_offset : 14;       // 46
    uint8_t version : 2;                    // 48
    uint8_t : 1, base_offset : 3, : 4;      // 56
    uint8_t : 5, layout_type : 3;           // 64
  } bitfield;
};

static uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo,
                          uint32_t layout) {
  (void)saddr;
  (void)lbo;
  (void)sbo;
  (void)layout;
  GmmaDescriptor desc{};
  desc.bitfield.start_address = saddr >> 4;
  desc.bitfield.leading_byte_offset = lbo >> 4;
  desc.bitfield.stride_byte_offset = sbo >> 4;
  desc.bitfield.version = 1;
  desc.bitfield.layout_type = layout;
  return desc.desc;
}

// TODO (b):三个场景的 {LBO 字节, SBO 字节, layout 编码}。
static const uint32_t SCEN[3][3] = {
    {128, 1024, 0}, // 场景 1
    {0, 1024, 2},   // 场景 2
    {0, 1024, 2},   // 场景 3
};

// 以下为判测,不需要修改。不匹配时按字段报差异,不打印期望值。
static const uint32_t ADDR[3] = {0x1000, 0x2000, 0x3000};
static const uint64_t TRUTH[3] = {0x0000404000080100ull, 0x4000404000000200ull,
                                  0x4000404000000300ull};

static void field_diff(uint64_t got, uint64_t want) {
  struct {
    const char *name;
    int lo, w;
  } f[] = {{"start_address", 0, 14}, {"LBO", 16, 14},
           {"SBO", 32, 14},          {"version", 46, 2},
           {"base_offset", 49, 3},   {"layout_type", 61, 3}};
  for (auto &x : f) {
    uint64_t g = (got >> x.lo) & ((1ull << x.w) - 1);
    uint64_t w = (want >> x.lo) & ((1ull << x.w) - 1);
    if (g != w)
      printf("    字段 %s 不一致(yours=0x%llx)\n", x.name,
             (unsigned long long)g);
  }
}

int main() {
  int bad = 0;
  for (int i = 0; i < 3; i++) {
    uint64_t got = make_desc(ADDR[i], SCEN[i][0], SCEN[i][1], SCEN[i][2]);
    if (got != TRUTH[i]) {
      printf("场景 %d FAIL:\n", i + 1);
      field_diff(got, TRUTH[i]);
      bad++;
    } else {
      printf("场景 %d PASS\n", i + 1);
    }
  }
  return bad != 0;
}
