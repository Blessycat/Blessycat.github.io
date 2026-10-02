---
title: "GPU 算子与 FlashAttention：从 Roofline 理解数据搬运成本"
date: 2026-10-02
permalink: /posts/ai-infra/gpu-kernels-flash-attention/
excerpt: "用算术强度、融合和在线 softmax 推导 FlashAttention 的核心思想，并区分精确注意力与近似算法。"
categories: [AI Infra]
tags: [GPU Kernel, Roofline, FlashAttention, 算子融合]
series: AI Infra 与后训练
series_order: 28
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：理解矩阵乘法、softmax 与注意力计算，能够区分设备显存和片上存储；建议先阅读 [性能时间线](/posts/ai-infra/profiling-training-bottlenecks/)。

* 目录
{:toc}

## FLOPs 少，不一定运行得快

一个算子可能执行很少的浮点运算，却反复从高带宽显存读取大量数据。若设备多数时间在等待数据，增加计算单元并不能同比降低耗时。Roofline 用一个简化上界把计算能力与内存带宽联系起来：

$$
P_{achievable}\leq\min(P_{peak}, B_{memory}\times I),
\qquad I=\frac{FLOPs}{Bytes}.
$$

I 是算术强度，必须说明 Bytes 对应哪一级存储传输。假设一台虚构设备的相关精度峰值为 100 TFLOP/s，显存带宽为 1 TB/s，转折点为 100 FLOP/Byte。算术强度只有 2 时，带宽上界约为 2 TFLOP/s，即便设备有很高的理论算力也难以发挥。

这是纸面上界，真实算子还受访问连续性、同步、寄存器占用、分支和启动延迟影响。精度不同，计算峰值也不同；稀疏峰值不能直接代入稠密工作负载。NVIDIA 的 [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)强调了带宽、访存组织和测量对优化的重要性。

## 融合减少中间结果进出显存

考虑逐元素计算 `y = relu(a*x+b)`。分开执行乘法、加法、激活，可能把两个临时数组写回显存再读出。融合后，每个元素读取一次 x，完成全部算术，再写出 y。数学表达式没有减少多少 FLOPs，数据流却明显缩短。

假设 x 和 y 都是 FP32，a、b 是标量且忽略其重复读取。三个独立算子大致需要 24 字节的逐元素读写，融合理想情况下约为 8 字节。这是忽略缓存与编译器已有优化的流量估算，不能声称融合必然提速三倍。融合可能增加寄存器需求，降低可驻留线程数量；若资源压力过大，还可能发生溢出，把中间值重新写到更慢的存储。

因此“kernel 数量减少”只能作为线索。真正需要检查的是端到端耗时、实际访存量和资源占用，以及数值是否仍满足容差。融合边界应围绕数据复用机会选择，而不是越大越好。

## 标准注意力为何会产生庞大的中间矩阵

对单个注意力头，Q、K、V 的序列长度为 N，头维度为 d。先计算 QKᵀ，再做 softmax，最后乘 V：

$$
O=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)V.
$$

如果显式保存 N×N 分数或概率矩阵，空间随 N 的平方增长。N=8192 时有 67108864 个元素，单个 BF16 矩阵约为 128 MiB；32 个头仅这一类矩阵就达到 4 GiB，还不包括 batch、其他激活与反向中间状态。这是矩阵大小计算，不是某一框架实际分配的完整显存峰值。

FlashAttention 的关键是分块计算并减少高带宽显存与片上存储之间的往返，避免完整注意力矩阵落入显存。它并没有把普通稠密注意力的理论算术复杂度改成线性，也不是通过丢掉部分注意力连接来获得速度。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)将这一设计归纳为对 IO 的优化。

## 在线 softmax 让分块计算保持同一个数学结果

对一行分数，维护目前见到的最大值 m、指数和 l，以及加权值和向量 a。新的分数块到来后，最大值更新为 m′。旧统计量原本以 m 为基准，必须乘上 exp(m−m′) 才能与新块合并：

$$
l'=e^{m-m'}l+\sum_j e^{s_j-m'},
$$

$$
a'=e^{m-m'}a+\sum_j e^{s_j-m'}v_j,
\qquad O=a'/l'.
$$

每一行只需携带这些统计量，就能逐块处理 K 和 V。减去最大值还抑制了指数溢出。反向可以利用保存的统计量并重算部分中间值，以额外计算换取减少显存访问；具体实现还要处理因果 mask、dropout、布局和并行线程划分。

“精确注意力”指计算目标没有变成低秩或稀疏近似，不意味着浮点结果逐位相同。分块归约顺序改变会引入舍入差异；在低精度下比较结果，需要相对误差与绝对误差容限，同时测试长序列和极端分数。

## 个人主机实验：实现一行分块注意力

以下标准库代码只验证在线归一化。每个 value 是标量以便手算，真实注意力把 a 换成向量即可。它没有 GPU kernel，也不用于性能比较。

```python
import math

scores = [1000.0, 999.0, 1001.0, 998.0]
values = [1.0, 2.0, 4.0, 8.0]
maximum = max(scores)
weights = [math.exp(s - maximum) for s in scores]
reference = sum(w*v for w, v in zip(weights, values)) / sum(weights)

running_max = -math.inf
denominator = numerator = 0.0
for start in range(0, len(scores), 2):
    block_s = scores[start:start+2]
    block_v = values[start:start+2]
    new_max = max(running_max, max(block_s))
    correction = math.exp(running_max - new_max)
    exp_scores = [math.exp(s - new_max) for s in block_s]
    denominator = denominator * correction + sum(exp_scores)
    numerator = numerator * correction + sum(
        e*v for e, v in zip(exp_scores, block_v)
    )
    running_max = new_max

assert abs(reference - numerator / denominator) < 1e-12
```

验收时把分块大小改为 1、2、4，检查结果保持一致；将所有分数同时加上一个较大常数，检查输出不变；故意删除旧统计量的 correction，确认遇到新的最大值时结果出错。这能直接解释为什么“分别对每块做 softmax，再平均输出”不等价于完整注意力。

进一步可用列表计数比较显式矩阵和分块统计量的空间增长，而不实际申请巨大内存。验收标准是知道哪些量随 N² 增长，哪些随 Nd 或块大小增长；不能把 Python 对象开销拿来声称 GPU 节省了相同字节数。

## 调用高层接口时仍需确认实际路径

PyTorch 的 [scaled_dot_product_attention 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)描述了融合实现与数学实现等路径，实际选择受设备、类型、形状和支持条件影响。调用一个函数名并不能证明使用了 FlashAttention 内核，也不能保证所有 mask 和 dropout 配置都走同一路径。

真实测试应先验证输出和梯度，再在相同精度、序列长度、batch 与 mask 下测量，预热后用正确设备计时。短序列可能更受启动成本影响，长序列可能更能体现 IO 优势；不同区间都应单独记录。本文提供的是数学与数据流推导，没有提供任何未经测量的加速比。
