---
title: "DPO 的目标函数：概率比、偏好间隔与长度归一化边界"
date: 2026-10-02
permalink: /posts/ai-infra/dpo-objective/
excerpt: "从 KL 正则化最优策略推导 DPO，用数值例子解释更新方向，并说明序列概率、长度归一化与离线偏好覆盖范围。"
categories: [AI Infra]
tags: [后训练, DPO, 偏好优化, 训练目标]
series: AI Infra 与后训练
series_order: 14
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：交叉熵、序列对数概率，以及 [KL 正则化](/posts/ai-infra/kl-regularization/)中的最优分布推导。本文讨论原始 DPO 的基本形式，所有概率均为教学构造。

* 目录
{:toc}

## 省去显式奖励模型，保留偏好假设

DPO 接收同一提示下的赢家和输家，通过策略与参考策略的概率比学习偏好。它不需要先拟合一个独立奖励网络，再用在线 rollout 进行 PPO 更新。这减少了一条训练链路，但没有消除偏好数据的噪声、偏差和覆盖不足。

[原论文](https://arxiv.org/abs/2305.18290)从特定 KL 正则化奖励最大化问题出发，结合 Bradley–Terry 偏好模型得到目标函数。理解这些假设很重要：所谓“直接”指优化形式更直接，不意味着任意偏好数据都准确，也不意味着它与所有 RLHF 实现逐步等价。

## 奖励怎样被概率比替换

对固定提示 x，KL 正则化问题的最优策略满足：

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta).
$$

反过来表达奖励：

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x).
$$

在同一个提示的两个回答之间相减时，难以计算的 log Z(x) 抵消。于是用待训练策略 π_θ 参数化隐式奖励，可以定义：

$$
z=\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right],
\qquad \mathcal L_{\mathrm{DPO}}=-\log\sigma(z).
$$

这个抵消要求两个回答共享同一个提示。把来自不同问题的“好回答”和“坏回答”随机拼成偏好对，就破坏了推导中的比较结构。一个多轮对话的提示包括此前历史，因此历史有差异也不能不加说明地当成同一个 x。

## 四个对数概率足以做一次手算

假设赢家在当前策略下的 log probability 为 -4，在参考策略下为 -5；输家分别为 -6 与 -5.5。两个 log-ratio 是 1 和 -0.5，差为 1.5。取 β=0.2，则 z=0.3，预测偏好概率约为 0.574，损失约为 0.554。

为什么赢家当前的绝对概率很低，依然可能获得正向更新？因为 DPO 关心的是相对参考策略的变化，以及赢家与输家之间的差，不是要求某条完整长序列的概率接近 1。

损失对 z 的导数为 σ(z)−1。当模型尚未很好地区分一对样本，梯度推动赢家的 log probability 上升、输家的下降。但在共享参数的语言模型里，一次更新影响大量回答，不能保证每对样本的赢家概率都逐步上升。偏好间隔变大，也可能主要来自输家概率下降，所以日志最好分别报告 chosen 与 rejected 的概率变化。

β 的影响也不是“越大越保守”一句话能概括的。在理论最优策略中，它控制奖励与 KL 的权衡；在固定偏好数据、固定参数化的 DPO 损失里，它同时缩放 logit 与梯度，并改变饱和速度。比较 β 时，需要固定数据、参考模型、训练步数和学习率口径。

## 序列求和不是无关紧要的实现细节

原始形式使用完整回答的序列 log probability，也就是各有效回答 token 的 log probability 之和。prompt token、padding 和不属于模型输出的工具观察不能无差别加入回答似然。

假设赢家有四个 token，每 token 相对参考的 log-ratio 都为 0.2，合计 0.8；输家有两个 token，每 token 为 0.3，合计 0.6。序列求和认为赢家相对提升更大；改为长度平均后，两者变成 0.2 与 0.3，比较方向反转。

这说明“为了消除长度偏差，把 log probability 除以长度”会改变目标。它可能是合理实验，但不能继续把所得分数当成原始序列概率，也不能不做额外论证地沿用原来的闭式映射。不同偏好算法对长度的处理不同，应引用具体变体，而不是把所有两条回答的分类损失都叫同一种 DPO。

回答截断同样会改变标签含义。如果原本获胜的答案在关键结论前被截断，数据中的 chosen 标签仍然指向它，训练的却已是另一段内容。优先统计截断率与两侧截断差异，检查完整问答边界，再决定最大长度。

## 一个稳定的 CPU 计算器

```python
import math

def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))

def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    margin = (chosen-ref_chosen) - (rejected-ref_rejected)
    return softplus(-beta * margin)

print(dpo_loss(-4, -6, -5, -5.5, 0.2))
assert abs(dpo_loss(-5, -5.5, -5, -5.5, 0.2)
           - math.log(2)) < 1e-12
```

稳定 softplus 避免对极大的指数直接求值。验收时要覆盖：策略等于参考时损失为 log 2；增大赢家 log probability 时，固定其他数值，损失下降；交换两侧会反转 margin；极端正负 margin 都得到有限结果。

随后生成二十对不同长度的人工 token log-ratio，分别用求和与均值计算，输出比较方向不一致的样本。这个实验可以直接揭示目标差异，不需要下载模型，也不应据此宣称某种长度策略在真实任务上更好。

## 离线数据的边界与诊断

DPO 学习的是数据里出现过的偏好关系。如果数据只有“流畅回答胜过明显乱码”，训练很可能改善表达，但未必学会识别两个都流畅、只有事实不同的答案。增加训练轮数不能补足缺失的比较类型。

一个实用的数据审计表应记录任务类别、赢家和输家长度、候选来源、平局比例、标签分歧、重复提示和截断情况。候选来源尤其重要：若赢家总来自某个模型，学习器可能记住该模型的标点、开头或模板，而非内容质量。

验收应该包括未参与偏好构造的独立提示集，报告人工或可验证评价、长度变化与基本能力回归。DPO loss 下降只说明训练偏好被更好拟合。训练隐式奖励的准确率上涨，也不能替代任务完成率。

在个人主机上，可训练一个只有几个回答选项的 softmax 表格策略：先固定参考分布，再提供有限偏好对，观察未比较选项的概率如何因归一化而变化。把一部分偏好故意反转，可观察损失与真实合成效用的分离。记录实际运行结果后，才能写下该实验的结论。

[IPO 所在论文](https://arxiv.org/abs/2310.12036)提供了另一种理解偏好学习与正则化的理论路径。它提醒读者，算法选择不只是更换训练器名称，还涉及对偏好概率、过拟合和目标间隔的不同建模。

## 参考资料

- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)：原始推导、假设与损失形式。
- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)：偏好优化的理论框架与 IPO。
- [TRL DPO Trainer 官方文档](https://huggingface.co/docs/trl/dpo_trainer)：当前实现支持的数据与损失选项；复现实验应锁定库版本，不把文档默认值视为算法定义。
