---
title: "训练检查点恢复：从保存权重走向状态一致性"
date: 2026-10-02
permalink: /posts/ai-infra/checkpoint-recovery/
excerpt: "把检查点视为一次训练事务的提交，设计可验证的状态清单、原子发布和故障恢复实验。"
categories: [AI Infra]
tags: [Checkpoint, 故障恢复, 可复现性, 分布式训练]
series: AI Infra 与后训练
series_order: 26
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：理解优化器更新与梯度累积的区别，知道参数分片如何把一个逻辑模型分布到多个 rank。

* 目录
{:toc}

## 权重能加载，不代表训练已经恢复

推理通常只需要模型结构、权重和匹配的分词处理；继续训练还依赖优化器、学习率日程、随机数状态以及数据位置。若只加载参数，Adam 的动量重新变为初始值，下一步更新就可能与故障前计划执行的更新不同。[PyTorch 保存加载教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)区分了模型权重保存与一般训练检查点。

设动量更新为 mₜ=0.9mₜ₋₁+0.1gₜ。故障前 mₜ₋₁=2，下一梯度 gₜ=1，正确的新动量是 1.9；丢失状态后则变成 0.1。模型能顺利前向并不暴露这个差异，但整个训练轨迹已经变化。恢复验收因此应比较恢复后的下一次更新，而非只确认文件没有报错。

## 用一份状态清单定义恢复契约

一个实用检查点至少描述三个层面。算法状态包括参数、优化器、学习率调度器、混合精度缩放器以及需要保留的辅助模型；进度状态包括全局更新步、数据 epoch、采样顺序或游标、已消费样本数；环境身份包括配置、数据版本、分词器、代码版本和依赖版本。

随机状态不是一个 seed 字段就能替代。seed 描述随机序列起点，而训练中途需要当前状态；Python、数据处理库、框架 CPU 与各设备可能各有生成器。多进程数据加载的预取还可能让“已经取出”与“已经用于一次已提交更新”不是同一批数据。必须定义游标指向哪一个边界，才能避免恢复后跳过或重复样本。

检查点格式还应有 schema 版本和必填字段。加载器可以区分字段缺失、格式迁移与可接受默认值，不能静默把未知配置当成当前默认配置。重启后日志应明确指出从哪个检查点、哪个更新步开始，而不是只显示一个新 run ID。

## 在更新边界保存，减少需要恢复的中间状态

最容易解释的边界是：完成梯度同步和优化器更新，完成调度器等对应更新，清理梯度后，保存下一次更新需要的状态。此时通常不必保存半个累积窗口的梯度与微批进度。

若必须在梯度累积中途保存，就还要保留累积梯度、已完成的微批数、窗口分母和相关随机状态。仅把 global step 设回原值会把新数据加到旧梯度上，或者丢掉已计算的梯度。保存成本是否值得下降，应与恢复逻辑复杂度一起评估。

分布式系统还要保证各分片属于同一个逻辑更新。某些 rank 保存第 100 步，另一些保存第 101 步，文件再完整也不是一个合法检查点。集体保存通常要求相应 rank 共同参与。不要从“只有 rank 0 打印日志”推断“只有 rank 0 调用保存 API”。[PyTorch Distributed Checkpoint 文档](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)是检查分片状态保存、加载与参与约束的入口。

## 文件写完与检查点可用是两件事

可靠发布可以采用“写临时内容、验证分片、最后发布清单”的协议。清单记录检查点 ID、更新步、各分片的路径、大小与校验和。读者只把拥有完整提交标记的检查点当作可恢复对象。损坏的最新检查点不应阻止回退到更早的完整版本。

单机同文件系统中，写临时文件再替换最终路径，是一种便于教学的原子可见性方案；它不自动保证断电持久性，更不适用于所有远程对象存储。文件刷新、目录持久化、对象存储语义和多文件提交协议，需要结合实际存储确定。异步保存也必须在快照捕获后避免底层张量被训练继续修改，否则磁盘写出的是变化过程中的混合状态。

改变 world size 时，能否重分片加载由格式与框架能力决定，即使参数可以正确恢复，也不能自动保证逐位复现。规约顺序和数据分配变化会改变浮点舍入与样本组合。应把“继续优化有效”“相同环境数值接近”和“逐位一致”设为不同验收等级。[PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)也明确讨论了跨版本与跨平台的限制。

## 保存频率可以用损失成本估算

设一次保存额外阻塞 C 秒，每隔 I 秒保存一次，平均故障间隔为 M 秒。粗略认为故障时平均损失 I/2 秒工作，则单位时间开销近似为：

$$
f(I)\approx\frac{C}{I}+\frac{I}{2M}.
$$

求导得理想间隔约为：

$$
I^*\approx\sqrt{2CM}.
$$

若 C=30 秒、M=24 小时，得到约 2277 秒，即 38 分钟。这是忽略重启耗时、异步保存和相关故障的纸面模型，不能直接当作生产配置。它提示一个方向：保存越贵可适当延长间隔，故障越频繁则应缩短。实际还要考虑对象存储限流、容量预算和重要训练阶段。

## 个人主机实验：给一个有动量的训练过程注入故障

下面用标准库创建完全确定的标量优化过程，在第五步保存并恢复。它验证状态组合与提交边界，不模拟分布式文件系统。

```python
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

def advance(state):
    gradient = state["w"] - 3.0
    velocity = 0.9 * state["velocity"] + gradient
    return {"w": state["w"] - 0.1 * velocity,
            "velocity": velocity, "step": state["step"] + 1}

initial = {"w": 0.0, "velocity": 0.0, "step": 0}
baseline = initial
for _ in range(10):
    baseline = advance(baseline)

with TemporaryDirectory() as directory:
    state = initial
    for _ in range(5):
        state = advance(state)
    temporary = Path(directory) / "state.tmp"
    committed = Path(directory) / "state.json"
    temporary.write_text(json.dumps(state), encoding="utf-8")
    os.replace(temporary, committed)
    resumed = json.loads(committed.read_text(encoding="utf-8"))
    for _ in range(5):
        resumed = advance(resumed)
    assert resumed == baseline
```

第一项验收是完整恢复结果与连续训练一致。第二项是主动删除 velocity，再以零补回，观察结果不一致，证明优化器状态确实必要。第三项是在替换文件之前模拟异常，确认未发布临时文件不会被当作最新检查点。扩展为两个分片时，再故意缺失一个分片，要求加载器拒绝该提交，而不是部分恢复。

真实训练的故障演练还应覆盖数据游标和随机层：记录恢复前后的样本 ID、学习率、损失与参数摘要，在相同环境用合理容差比较。做完这些，检查点才从“磁盘上的一个文件”变为有明确恢复语义、能承担故障的系统接口。
