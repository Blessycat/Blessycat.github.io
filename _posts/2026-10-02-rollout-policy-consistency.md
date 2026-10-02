---
title: "Rollout 与训练必须对齐什么：概率口径、策略版本和离策略校正"
date: 2026-10-02
permalink: /posts/ai-infra/rollout-policy-consistency/
excerpt: "区分行为策略、旧策略、当前策略与参考策略，使用两动作实验解释重要性采样，并建立可审计的 rollout 元数据。"
categories: [AI Infra]
tags: [后训练, Rollout, 离策略学习, 训练推理一致性]
series: AI Infra 与后训练
series_order: 18
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：条件概率、PPO 概率比和基本采样概念。本文的“训练推理一致性”专指生成数据和训练计算的口径，所有性能数字均由离散例子推导，不是分布式系统实测。

* 目录
{:toc}

## 同一份权重不保证同一个行为策略

训练系统可能用一个推理引擎生成回答，再用另一个训练引擎重新计算 log probability。即使权重文件相同，温度、top-p、词表掩码、chat template、位置编码、精度和结束规则不同，也可能导致两个引擎表示不同的分布。

因此第一件事不是检查模型名称是否相同，而是问：这条 token 到底从哪个分布采出来，保存的概率对应哪个分布，训练目标希望对哪个分布取期望？

更快的采样只有在生成轨迹仍可解释时才有价值。若日志中的概率无法对应实际采样行为，后续重要性采样和 clipping 就可能在修正一个不存在的差异。

## 四个策略对象需要明确命名

行为策略 μ 是真正生成样本的分布，可能包含温度或截断采样；旧策略 π_old 是本轮训练冻结的比较基准；当前策略 π_θ 随梯度更新；参考策略 π_ref 常用于长期 KL 正则化。

理想的同步设置中，μ 与 π_old 按约定一致。异步设置里，采样 worker 可能仍使用前几个版本的权重；即使版本相同，也可能因为推理与训练计算差异导致不一致。

PPO 比值 π_θ/π_old 控制当前更新相对旧策略的变化。若样本其实来自 μ，还可能需要 π_old/μ 的校正。两者数学上可相乘为 π_θ/μ，但在 clipping、截断权重和停止梯度之后，具体计算图未必等价。不要随意把两个修正因子合并后假设算法性质保持不变。

## 重要性采样的最小例子

行为策略对 A、B 的概率为 (0.8,0.2)，目标策略为 (0.5,0.5)，奖励为 (0,1)。行为策略采样的平均奖励为 0.2，目标策略的真实平均奖励为 0.5。

利用概率比可以换测度：

$$
\mathbb E_{a\sim p}[f(a)]
=\mathbb E_{a\sim\mu}
\left[\frac{p(a)}{\mu(a)}f(a)\right].
$$

A 的权重为 0.625，B 为 2.5，故校正后的期望是 0.8×0.625×0+0.2×2.5×1=0.5。若把权重截断在 2，结果变成 0.4：方差可能更小，但已经有偏。把这件事写清楚，比笼统说“做了 off-policy correction”更有意义。

该等式要求目标分布有概率的动作，在行为分布中也有非零概率。若 top-k 采样永久排除某个动作，单靠现有样本无法估计该动作的贡献。再精巧的权重也不能从零覆盖中创造信息。

## 温度与截断需要进入记录

对于 logits z，温度采样通常使用 softmax(z/τ)。τ 不为 1 时，采样分布与原始 softmax 不同。top-p 或 top-k 还会删去一些 token，再对剩余概率重新归一化。

因此要区分“原模型对所选 token 的 log probability”和“采样处理后行为策略的 log probability”。训练算法可以选择不同约定，但必须保证分母、支持集与目标推导相符。不能简单规定永远使用某一个，而忽略算法是否已经对采样变换作出假设。

chat template 更基础：若推理端多一个换行，训练端少一个 assistant 起始标记，条件前缀已经改变。此时重新算出的 log probability 差异不是纯数值误差，而是在不同状态上计算概率。

位置与 mask 同样重要。左 padding、右 padding、截断方向、EOS、工具返回文本的插入方式都可能改变条件序列。只有先证明 token IDs 和有效位置一致，比较 log probability 才有解释力。

## 长轨迹的比值为什么容易爆炸

完整序列的重要性权重是各步比值的乘积：

$$
w(y)=\prod_t\frac{p(y_t\mid y_{<t},x)}
{\mu(y_t\mid y_{<t},x)}
=\exp\left(\sum_t(\log p_t-\log\mu_t)\right).
$$

即使每一步只偏离一点，长序列的乘积也可能很大或很小。实际算法会使用截断、分段、逐 token 近似或拒绝异常样本等方式控制方差，但这些选择对偏差和目标都有影响。

[IMPALA](https://arxiv.org/abs/1802.01561)用 V-trace 处理解耦 actor 与 learner 的策略差异。它提供了重要的离策略校正思路，但不能把名称直接贴到语言模型 PPO 上就视为完成了理论迁移。多轮工具环境中，状态分布和环境转移也进入轨迹计算，单个动作比值并不自动修正所有差异。

## 一个可以精确验收的 CPU 实验

```python
behavior = [0.8, 0.2]
target = [0.5, 0.5]
reward = [0.0, 1.0]
weights = [p/q for p, q in zip(target, behavior)]

naive = sum(q*r for q, r in zip(behavior, reward))
corrected = sum(q*w*r for q, w, r in zip(behavior, weights, reward))
clipped = sum(q*min(w, 2.0)*r
              for q, w, r in zip(behavior, weights, reward))
print(naive, corrected, clipped)

sample_weights = [weights[0]]*80 + [weights[1]]*20
ess = sum(sample_weights)**2 / sum(w*w for w in sample_weights)
print(ess)
```

预期的解析结果分别为 0.2、0.5、0.4，有效样本量 ESS 为 64。这里人为构造了 80 个 A 与 20 个 B，不是随机采样结果。进一步用随机抽样重复一千次，报告估计均值与方差，才能观察校正和截断的统计权衡。

ESS 反映权重集中程度，但不是“等价真实独立样本数”的万能保证。轨迹相关、奖励相关和状态覆盖问题仍会影响学习；它应作为诊断信号，与任务表现和权重分位数一起解释。

## 元数据比排查时的猜测便宜

策略版本必须能追溯到真实权重，而不仅是一个随手递增的计数器。若 worker 在一条回答中途加载新权重，这条序列可能由多个行为策略片段产生。最简单的设计是让完整回答固定版本；若系统有意支持中断与续写，就应记录每段对应的权重与概率，不能只给整条序列贴最后一个版本号。

同步也需要一个明确完成点。训练器发出新权重之后、采样器真正加载完成之前，队列里可能同时存在多个版本。检查版本滞后时应使用实际生成版本与开始训练版本，而非仅比较发送时间。丢弃过旧数据会损失生成成本，保留则增加分布差异，这是一项需要记录的系统与统计权衡。

每条 rollout 至少关联权重版本、tokenizer 与 template 版本、采样配置、提示及回答 token IDs、行为 log probabilities、生成时间、结束原因、奖励和验证器版本。异步系统还应保存入队与出队时间，计算版本滞后分布。

一次只容许一个变量变化的对照最容易定位问题：先在短固定前缀上比较两个引擎的 token IDs 与 logits，再比较采样后概率，最后才引入长序列、低精度与异步更新。验收阈值应按精度和任务建立，不要宣称所有引擎必须逐位相同。

[verl 官方校正文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)把生成与训练分布差异、重要性权重和样本筛选作为独立配置问题处理。实际复现要保存文档对应的代码版本；“开启校正”不能替代检查概率定义。

## 参考资料

- [IMPALA](https://arxiv.org/abs/1802.01561)：异步 actor–learner 与 V-trace。
- [PPO 原论文](https://arxiv.org/abs/1707.06347)：旧策略比值和有限数据复用。
- [verl：Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)：生成训练差异的工程处理。
- [verl：Mathematical Formulations](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)：校正目标与计算口径。
