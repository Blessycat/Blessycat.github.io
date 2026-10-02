---
title: "训练显存怎么估：混合精度、AdamW 状态与峰值测量"
date: 2026-10-02
permalink: /posts/ai-infra/precision-optimizers-memory/
excerpt: "把权重、梯度、优化器、激活和缓存分别记账，解释为什么半精度训练不一定让显存减半。"
categories: [AI Infra]
tags: [混合精度, AdamW, CUDA, 显存分析]
series: AI Infra 与后训练
series_order: 9
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：训练循环、张量 dtype 与基本显存概念。字节数推导无需 GPU；后半段测量代码需要已有 CUDA PyTorch 环境，不包含安装步骤。

模型文件只有 2 GB，训练为什么可能需要十几 GB？因为文件主要描述持久参数，而训练还需要梯度、优化器状态和中间激活。混合精度改变了其中一部分张量的类型，却未必改变全部状态。显存估算的关键不是记一个通用倍数，而是把假设展开。

* 目录
{:toc}

## FP16 与 BF16 解决不同的数值取舍

两者都使用 16 bit，但指数和尾数的分配不同。FP16 的指数范围较窄，能表示的最大有限值约为 65504；BF16 的指数位数与 FP32 相同，范围更宽，但尾数更短。因此 BF16 通常更不容易发生范围溢出，却不意味着每个数都比 FP16 更精确。

混合精度通常按算子选择 dtype：适合的矩阵运算使用较低精度，对数值更敏感的部分保持较高精度。`autocast` 是这种运算策略，不是把模型所有参数永久转换成同一种类型。PyTorch 的 AMP 文档区分 autocast 与梯度缩放两个机制。[Automatic Mixed Precision](https://docs.pytorch.org/docs/2.14/amp.html)

例如 `1.0 + 0.0001` 在有限精度下可能舍入回 1.0。优化器如果直接在较低精度参数上累计很小更新，可能丢失信息；一些训练方案保留 FP32 主参数或状态来缓解这个问题。不过具体存储方式取决于实现，不能看到“bf16=True”就自动认定存在或不存在主参数副本。

## AdamW 为什么比 SGD 多占内存

除了当前梯度，Adam 类算法维护一阶矩和二阶矩估计，分别反映历史梯度与平方梯度的指数平均。更新还包含偏置修正。AdamW 将权重衰减与梯度自适应部分分开处理，其 API 与算法说明见官方文档。[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)

若有 P 个可训练参数，一种常见账本是 FP32 参数 4P 字节、FP32 梯度 4P、一阶与二阶状态合计 8P，总计 16P。这里尚未计算激活、临时缓冲和状态标量。另一种假设是低精度参数 2P、低精度梯度 2P、FP32 主参数 4P、FP32 两个矩状态 8P，同样是 16P。

这两个相同结果来自不同实现，不能反推出所有 AdamW 训练都固定每参数 16 字节。例如直接创建低精度参数时，某些实现的状态 dtype 可能跟随参数；优化器量化、分片与卸载又会改变驻留内存。最可靠的方式是检查参数、梯度和 optimizer state 中每个 tensor 的 `numel()*element_size()`。

取十亿可训练参数，16P 为 160 亿字节，约 14.90 GiB。若显卡标称容量与工具显示单位不同，还应区分十进制 GB 与二进制 GiB。把 16 GB 直接写成 16 GiB，会在紧张配置中高估可用空间。

## 激活使静态账本无法预测全部峰值

反向传播需要前向中的信息，框架会保存若干激活。它们的大小随层数、序列长度、隐藏维度和微批次改变；注意力实现还影响是否完整保留某些大矩阵。词表 logits 也可能成为大项：若 B=2、T=2048、V=50000，以 FP32 物化完整 logits 就约需 0.763 GiB。

梯度 checkpoint 通过反向时重新计算一部分前向结果来降低保存激活的需求，代价是额外计算。梯度累积则允许减小同时驻留的微批次，但权重和优化器状态不会随累积步数减少。这些方法作用于不同项，不能互相当作替代品。

优化器状态常在第一次 step 时才创建，因此只测模型加载后的显存会低估训练需求。评估、保存、生成或合并适配器也可能产生独立峰值；能够完成一个训练步，不代表整个训练生命周期都能完成。

## 梯度缩放与裁剪必须按正确顺序

FP16 训练中，较小梯度可能下溢。梯度缩放先把 loss 放大再反向，更新前把梯度还原。裁剪若发生在还原之前，裁剪阈值对应的是放大后的数值，实际含义就变了。下面是完整的一步教学代码，使用随机数据与小线性层，不代表语言模型训练配置。

```python
import torch

assert torch.cuda.is_available()
model = torch.nn.Linear(512, 32).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")
x = torch.randn(16, 512, device="cuda")
y = torch.randn(16, 32, device="cuda")

optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

若检测到非有限梯度，scaler 可能跳过实际参数更新并调整缩放因子，因此数据迭代次数不总等于有效更新次数。梯度累积时也不能在每个微批次中途随意改变缩放比例。官方 AMP 示例给出了裁剪、累积和多优化器等场景的顺序要求。[AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)

## allocated、reserved 与外部工具分别在看什么

PyTorch 的 allocated 关注活跃 tensor 占用，reserved 关注缓存分配器管理的内存。释放一个 tensor 后，allocated 可以下降，但内存块可能留在缓存中供下次复用，因此 reserved 不一定立即下降。外部工具还可能看到 CUDA 上下文和其他库的分配。[CUDA memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

`empty_cache()` 释放的是未被活跃 tensor 使用的缓存，并不能删除仍被计算图或 Python 容器引用的张量。训练循环中不断调用它可能引入额外开销，也不能修复保留整张计算图的内存泄漏。

测峰值前先同步设备并重置峰值统计，在一个完整测量区间结束后再次同步，然后记录 `max_memory_allocated()` 与 `max_memory_reserved()`。初始化峰值与稳定迭代峰值可以分开报告；不要为了得到更低数字而无意排除第一次优化器状态创建。计时同样需要处理 CUDA 异步执行，否则 CPU 计时可能只测到任务提交。

## 一次 OOM 应该留下什么证据

报错发生阶段十分重要：加载模型时失败，多半应先看持久权重；前向随长度增加失败，先看激活与 logits；第一次 step 失败，先核对优化器状态和临时缓冲。它们只是优先检查方向，不能替代实际快照。记录最后成功的形状、精度、微批次与阶段，下一次实验才能有针对性。

同时区分持续增长与稳定峰值。若相同形状每步都增加占用，检查是否把带图张量留在列表、回调或闭包里；若只是第一轮较高、以后稳定，可能与初始化或缓存有关。不要把不同阶段的数字拼成一张看似精确的账本。

为缓存和临时分配留出余量，也比把理论值恰好塞满显卡更稳妥。余量大小应由实际测量决定，不应作为固定百分比经验法则写死。模型能以某组输入训练，不等于长度分布稍变后仍能保持同样峰值。

## 主机实验与验收

在 CPU 上先统计一个小模型的参数字节数，再执行一次 AdamW 更新，枚举状态 tensor，观察前后差异。即使没有 GPU，这一步也能验证“状态在什么时候出现”和“状态是什么 dtype”。

具备 GPU 时，把微批次分别设为 1、2、4，在相同模型和长度下记录完整训练步峰值；再固定微批次改变序列长度。每次仅改变一个变量，保留相同精度与内核设置。预期激活相关部分会变化，但本文没有给出实测曲线或承诺线性比例。

验收结果应包含一张可解释的账本：理论参数状态字节数、实际 tensor 状态字节数、完整步骤峰值，以及两者差额的可能来源。最终目标是能回答“本次 OOM 最可能是哪一项增长”，并选择针对该项的调整，而不是把所有省显存开关同时打开后失去比较基准。
