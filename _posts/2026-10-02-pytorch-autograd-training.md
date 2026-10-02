---
title: "训练循环到底更新了什么：张量、自动微分与梯度累积"
date: 2026-10-02
permalink: /posts/ai-infra/pytorch-autograd-training/
excerpt: "用一个可手算的线性模型连接张量形状、计算图、反向传播和优化器更新。"
categories: [AI Infra]
tags: [PyTorch, 自动微分, 训练循环, 梯度累积]
series: AI Infra 与后训练
series_order: 3
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：Python 函数、矩阵乘法和一元函数求导。代码运行前提是已有可用的 PyTorch；全部示例可在 CPU 上运行，不需要下载模型。

很多训练错误不会抛出异常。模型能前向、loss 能打印、进度条能结束，却可能根本没有更新想训练的参数。理解训练循环，最好从一个能用纸笔核对的例子开始：每个张量代表什么，梯度如何进入参数，哪一行真正改变了参数值。

* 目录
{:toc}

## 张量的形状就是接口契约

设输入批次有 B 个样本，每个样本有 D 个特征，线性层输出 K 个数。输入 X 的形状是 `[B, D]`，权重 W 是 `[D, K]`，偏置 b 是 `[K]`，输出 `X @ W + b` 是 `[B, K]`。偏置能够自动扩展到各个样本，这是广播；广播也会让不正确的标签形状悄悄通过。

例如回归预测是 `[8, 1]`，标签却是 `[8]`，直接相减可能得到 `[8, 8]`。程序比较了每个预测与每个标签，优化了完全不同的目标。检查 loss 前的形状，比检查 loss 数值是否“看起来正常”更可靠。实践中可写 `assert pred.shape == target.shape`，把隐含假设变成机器可检查的约束。

形状之外还要看 dtype 和 device。分类标签通常是整数索引，参数与连续输入通常是浮点数；CPU 张量不能无条件与 CUDA 张量做同一个矩阵运算。形状、类型、设备三者构成了张量的基本身份证明。

## 从一个梯度推导看懂 backward

取两个样本 x=[1,2]，目标 y=[2,4]，模型只有一个参数 w，预测为 wx。定义平均平方损失：

$$
L(w)=\frac{(w-2)^2+(2w-4)^2}{2}.
$$

对 w 求导得到：

$$
\frac{dL}{dw}=(w-2)+2(2w-4)=5w-10.
$$

当 w=0 时，loss 为 10，梯度为 -10。学习率取 0.1，普通 SGD 更新后 w=1，新的 loss 为 2.5。这里每个数都是公式推演的结果，不是性能测量。

```python
import torch

x = torch.tensor([1.0, 2.0])
y = torch.tensor([2.0, 4.0])
w = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.SGD([w], lr=0.1)

optimizer.zero_grad(set_to_none=True)
loss = ((w * x - y) ** 2).mean()
loss.backward()
print(float(loss.detach()), float(w.grad))
optimizer.step()
print(float(w.detach()))
```

前向运算构建计算图，`backward()` 按链式法则传播导数，将结果累积到叶子参数的 `.grad`。它没有执行 SGD 更新；改变参数的是 `optimizer.step()`。PyTorch 官方自动微分教程解释了计算图与叶子张量梯度的关系。[Automatic Differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

这里不能把 loss 先转换为 Python 数值再反向传播。`loss.item()` 适合日志，但返回的普通数值已经不再携带计算图。类似地，把中间结果 `detach()` 后用于后续损失，会切断对应路径上的梯度。

## 梯度为什么默认累加

一次训练迭代通常依次执行清梯度、前向、算 loss、反向和更新。清梯度不是语法仪式。假如第一次反向得到 -10，既不清梯度也不更新参数，又重新前向和反向，`.grad` 就会累积成 -20。

这种设计支持梯度累积：把放不进显存的大批次拆成多个微批次，计算各自的梯度，最后更新一次。`set_to_none=True` 会将梯度设为 None；这与填零存在语义差异，没有收到梯度的参数可能被优化器跳过。需要定位未参与计算的参数时，这个差别很有用。[Optimizer.zero_grad](https://docs.pytorch.org/docs/main/generated/torch.optim.Optimizer.zero_grad.html)

若四个微批次都有相同样本数，并且 loss 都按样本取平均，可将每个 loss 除以 4 再反向，得到合并大批次的平均梯度。然而语言模型通常按有效目标 token 平均。两个微批次分别有 10 和 90 个目标 token，简单平均它们的均值，会给前 10 个 token 一半权重，而正确的全局 token 平均只应给它们十分之一。

更一般的目标是所有损失项求和，再除以所有有效项数量：

$$
L=\frac{\sum_j S_j}{\sum_j N_j}.
$$

其中 S 是微批次损失总和，N 是有效样本数或 token 数。单进程中可先累计 sum-reduction 的梯度，窗口结束后用总 N 缩放参数梯度。最后一个不完整窗口也用实际数量。混合精度下应先完成梯度反缩放，再做归一化和裁剪；多卡还需处理归约语义，不能机械复制单卡代码。

## 训练模式与梯度模式是两回事

`model.train()` 和 `model.eval()` 主要改变 Dropout、BatchNorm 等模块的行为；它们不负责开启或关闭自动微分。验证通常既需要 `model.eval()`，也需要 `torch.no_grad()` 或适用的 inference mode。仅调用 eval，计算图仍可能占用内存。

另一个隐蔽问题是把带图的 loss 张量一直追加到 Python 列表。引用留在列表中，会让图中的对象无法及时释放。记录数值时用 `loss.detach().item()`；需要保存预测做后续分析时，通常先 detach，再按需求转到 CPU。不要把所有中间激活当作日志长期保留。

优化器应该收到真正需要训练的参数。只打印模型总参数量无法证明 LoRA 参数或某个新加的层已进入 optimizer。可以检查 `requires_grad`，再核对优化器参数组中的对象标识。官方训练教程提供了基本的 loss、optimizer 与训练/验证循环结构。[Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

## 个人主机上的三段验收实验

第一段运行上面的标量模型，用手算值核对 loss、梯度和更新后的 w。再用中心差分近似梯度：计算 `L(w+epsilon)` 与 `L(w-epsilon)` 的差，除以 `2*epsilon`。使用 float64 和约 1e-5 的 epsilon 比较合适；epsilon 太小会放大舍入误差，太大又增加近似误差。

第二段对同一批数据比较两种方法：一次完整批次更新，与两次微批次累积后更新。关闭随机层，使用相同初始权重和普通 SGD，并确保损失归一化一致。预期参数应在浮点容差内接近，而不是要求不同计算顺序下每个二进制位完全相同。

第三段故意制造一个错误：把预测 detach 后构造损失，或把目标 `[B,1]` 改成 `[B]`。记录是异常直接暴露了问题，还是程序继续执行却改变了目标。验收不以“跑完”为标准，而以能解释每个参数梯度是否存在、大小为何如此、两种累积方式为何一致为标准。

若连两个样本都无法过拟合，先查目标、mask、梯度链和学习率，再考虑更复杂的模型。这个小实验建立的是之后理解 SFT 的基本单位：训练步骤不是黑盒，它只是对一组明确损失项求导，再按明确规则修改参数。
