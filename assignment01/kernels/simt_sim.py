"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""

LANES = 32

def run(program, init_regs=[i for i in range(LANES)]):
    regs = [r for r in init_regs]
    cycles = 0
    for instruction in program:
        if instruction[0] == 'add':
            cycles += 1
            _, k = instruction
            regs = [r + k for r in regs]
        elif instruction[0] == 'mul':
            cycles += 1
            _, k = instruction
            regs = [r * k for r in regs]
        elif instruction[0] == 'if_lt':
            _, t, then_prog, else_prog = instruction
            then_mask = [r < t for r in regs]
            else_mask = [not m for m in then_mask]
            if any(then_mask):
                then_regs, then_cycles = run(then_prog, regs)
                cycles += then_cycles
            else:
                then_regs = [0] * LANES
            if any(else_mask):
                else_regs, else_cycles = run(else_prog, regs)
                cycles += else_cycles
            else:
                else_regs = [0] * LANES
            regs = [tr * tm + er * em for tr, tm, er, em in zip(then_regs, then_mask, else_regs, else_mask)]
        else:
            raise NotImplementedError("No such instruction.")
    return regs, cycles
