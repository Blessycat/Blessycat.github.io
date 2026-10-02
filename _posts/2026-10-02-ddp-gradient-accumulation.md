---
title: "DDP 与梯度累积：把有效批量、采样和同步语义算清楚"
date: 2026-10-02
permalink: /posts/ai-infra/ddp-gradient-accumulation/
excerpt: "从损失归一化推导 DDP 的有效批量，解释 no_sync、尾批和分布式采样器为何影响训练正确性。"
categories: [AI Infra]
tags: [DDP, 梯度累积, PyTorch, 数据并行]
series: AI Infra 与后训练
series_order: 23
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：理解梯度下降、损失的 mean 与 sum，以及 [集合通信](/posts/ai-infra/distributed-collectives/) 中 AllReduce 的语义。

* 目录
{:toc}

## DDP 解决副本同步，数据切分仍由训练程序负责

DistributedDataParallel 为不同进程保存模型副本，并在反向传播中同步梯度。只要初始参数、梯度和优化器更新规则相同，副本就能继续保持一致。它不会自动把一份输入 batch 分给各进程，也不会把错误的数据采样逻辑变正确。当前接口的职责边界可查阅 [DDP 官方 API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。

设有四个 rank，每个 rank 每次处理两条样本。若每个进程都读取相同的前两条，名义上运行了八次样本计算，实际只有两条独立数据。梯度同步可以完全成功，损失曲线也可能下降，但有效数据覆盖远小于预期。这类错误很难通过 GPU 利用率发现，应记录样本 ID 来验证。

DistributedSampler 按进程划分索引，并通过种子和 epoch 控制打乱顺序。每个 epoch 创建迭代器前调用 `sampler.set_epoch(epoch)`，才能改变跨 epoch 的排列。不足以整除进程数时，采样器可能补充索引或丢弃尾部；DataLoader 自己的 `drop_last` 又处理局部 batch 尾部，两者是不同层级。[数据加载文档](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)给出了这些边界。

## 有效批量公式背后是一次优化器更新

令 R 为数据并行进程数，b 为每个 rank 的微批样本数，K 为累积次数。各微批大小相同、没有样本重复且一次更新涵盖 K 个微批时：

$$
B_{effective}=R\,b\,K.
$$

例如 R=4、b=2、K=8，有效批量是 64 条样本。这里的“一步”必须指一次优化器更新；若日志中的 step 每个微批递增，学习率调度与样本数统计就可能相差八倍。累积只是延后更新，并没有让模型同时持有 64 条样本的全部激活。

设每个微批损失先对 b 个样本取平均。在通常的 DDP 梯度平均语义下，每个微批损失再除以 K，最终梯度就是这 RbK 条样本损失的平均梯度：

$$
g=\frac{1}{R}\sum_{r=1}^{R}\sum_{k=1}^{K}\frac{1}{K}\nabla L_{r,k}.
$$

漏除 K，会把梯度整体放大 K 倍；在 DDP 已经平均的基础上再除 R，则会把梯度缩小 R 倍。不要靠“学习率看起来更合适了”掩盖这种错误，因为 Adam、梯度裁剪与权重衰减未必与统一缩放等价。

## no_sync 的作用范围必须包含前向

累积的前 K−1 个微批无需马上同步，最后一个微批再执行正常的前向和反向。`no_sync()` 只改变梯度通信时机，不会自动改变损失归一化，也不会替调用方决定何时更新参数。官方 API 特别要求前向也包含在上下文内部。

下面是训练循环片段，假设已经初始化进程组，`ddp` 是包装完成的模型，`microbatches` 恰好包含 K 个等大小微批，`loss_fn` 返回微批平均损失。它不包含设备初始化，也不是可独立启动的分布式脚本。

```python
from contextlib import nullcontext

optimizer.zero_grad(set_to_none=True)
K = len(microbatches)
for i, (x, y) in enumerate(microbatches):
    context = ddp.no_sync() if i < K - 1 else nullcontext()
    with context:
        loss = loss_fn(ddp(x), y) / K
        loss.backward()
optimizer.step()
```

在循环内部每次 `zero_grad()` 会抹掉前面的累积；每个微批调用 `step()` 则不再是同一次更新。梯度裁剪应作用于最终累积并同步后的梯度。使用混合精度时，还需要让缩放、反缩放和溢出判断与更新边界保持一致；不要把独立微批的范数分别裁剪后，认为等价于裁剪完整梯度。

## 尾批和变长文本会破坏简单除法

若最后只有三个微批，却仍除以固定的八，梯度会偏小。更隐蔽的是两个微批分别有 20 与 80 个有效 token：把两个 token 平均损失简单各乘一半，会让前者每个 token 获得后者四倍的权重。

若目标确实是窗口内所有有效 token 的平均损失，应该累计损失之和，并按总有效 token 数归一化。对不等长的不同 rank，令全局有效 token 数为 N，而 DDP 对 R 个 rank 的梯度取平均，那么每个 rank 的本地损失和可乘 R/N，最终得到全局 token 平均梯度。这个 N 必须覆盖整个累积窗口，而不是仅当前微批。若 N 需要额外集合通信，应提前以所有 rank 一致的顺序计算。

并非所有任务都应该使用 token 平均：偏好学习可能按样本对平均，序列级奖励可能按轨迹平均。正确的分母由训练目标定义，不能为了让代码统一而偷偷改变长短样本权重。与大 batch 的等价性也有条件：BatchNorm 统计、依赖批内其他样本的损失和随机操作，都可能让微批累积与一次大前向产生差异。

## 把分母与采样覆盖放进同一个具体例子

考虑两个 rank：第一个有十个有效 token，其损失之和为二十；第二个有三十个有效 token，其损失之和为三十。两边各自的平均损失分别是二和一，直接对这两个平均值再平均，得到一点五；全局目标若是逐 token 平均，则应该是五十除以四十，即一点二五。问题不是通信做错了平均，而是通信之前的两个数已经丢失了数量信息。

把两边的损失和都乘以二除以四十，再由两进程平均，才能得到正确结果。训练时对梯度使用同样权重，逻辑完全一致。这个例子也解释了为什么应把有效 token 数写入日志：仅记录每个 rank 的平均 loss，事后往往无法恢复正确的全局统计。

再看采样覆盖。十条样本分给三个 rank，若要求每个 rank 拥有相同数量，又不丢弃尾部，就需要补齐到十二个索引，其中两次访问是重复。若选择丢弃尾部，则本轮只有九条不同样本参与。两种策略都可以有用途，但评估集尤其要避免把补齐的重复样本直接计入指标分母，否则少量样本会得到更高权重。

一个可操作的排查方式是在很小的数据集上打印本轮所有样本编号，并离线统计编号频次与缺失集合。训练数据集可以允许有意重复，但应能说明原因；评估时则可按真实样本编号聚合，或使用不补齐的分配方式并处理不等长参与者。此时不能随意让某些 rank 提前跳过仍由其他 rank 执行的集合通信，否则正确的数据统计又会变成通信挂起问题。

## 个人主机实验：用标量回归验证梯度

先不启动多进程，用标准库显式写出平方损失梯度。令预测为 wx，单样本损失为预测误差平方的一半，梯度即 `(w*x-y)*x`。以下示例把八条数据分给两个 rank，每个 rank 累积两个等大小微批。

```python
data = [(float(x), 2.0 * x + 1.0) for x in range(1, 9)]
w = 0.3
def grad(batch):
    return sum((w*x-y)*x for x, y in batch) / len(batch)

full = grad(data)
rank_batches = [[data[:2], data[2:4]], [data[4:6], data[6:]]]
accumulated = sum(
    sum(grad(batch) / 2 for batch in batches)
    for batches in rank_batches
) / 2
assert abs(full - accumulated) < 1e-12

a, b = data[:1], data[1:]
wrong = (grad(a) + grad(b)) / 2
weighted = (grad(a)*len(a) + grad(b)*len(b)) / len(data)
assert abs(weighted - full) < 1e-12
assert abs(wrong - full) > 1e-6
```

验收要包含故意制造错误：删除除以累积次数的操作，确认梯度放大；把不等大小微批简单平均，确认与全量梯度不一致；再用数量加权恢复一致。把这些结果记作数学语义验证，不能将 Python 循环时间解释为 DDP 性能。

若主机已有可用的 CPU 版 PyTorch，可进一步用 Gloo 启动两个进程，对不含随机层的小线性模型比较参数更新。固定初始参数与样本顺序，比较一次大 batch 更新和累积更新，记录浮点误差容限，而不是要求所有平台逐位相同。[DDP 设计说明](https://docs.pytorch.org/docs/stable/notes/ddp.html)解释了梯度桶与通信重叠，但其内部实现描述带有历史版本范围，具体调用行为应优先看当前 API。

最后同时验收三件事：各 rank 处理的数据符合预期；参数更新与目标损失的梯度一致；优化器步数、学习率步数与有效样本数相互对应。只有第一项和第二项正确，减少同步次数带来的性能收益才值得讨论。
