---
title: "SFT 的核心是目标定义：右移标签、回答掩码与有效 token 归一化"
date: 2026-10-02
permalink: /posts/ai-infra/supervised-finetuning/
excerpt: "逐位置检查监督微调的损失，把答案边界、标签右移和梯度权重变成可验证的约束。"
categories: [AI Infra]
tags: [SFT, Loss Masking, 交叉熵, 后训练]
series: AI Infra 与后训练
series_order: 7
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：自回归注意力、聊天模板与训练循环。代码前提是已有 PyTorch，示例使用人工 logits，可在 CPU 上检查，不需要预训练模型。

监督微调的基本动作，是提高模型在给定上下文下生成示范答案的概率。它并不会因为样本叫“指令数据”就自动知道哪些 token 是问题、哪些是答案。模板决定输入协议，标签与 mask 决定优化目标。许多 SFT 问题最终都能追到这两层中的一个错位。

* 目录
{:toc}

## 从条件概率写出训练目标

把提示记为 x，答案记为 y。自回归模型将答案概率拆成逐 token 条件概率：

$$
p_\theta(y\mid x)=\prod_{t=1}^{|y|}p_\theta(y_t\mid x,y_{<t}).
$$

取负对数后，连乘变为求和。若只监督助手回答，平均损失写为：

$$
L=-\frac{\sum_t m_t\log p_\theta(z_t\mid z_{<t})}{\sum_t m_t},
$$

其中 z 是完整模板序列，m 指明哪些目标位置参与训练。有效 token 总数必须大于零，否则这个平均值没有定义。SFT 是指令后训练中的常见步骤，原始 InstructGPT 工作将示范数据监督训练与之后的偏好学习阶段区分开来。[Training language models to follow instructions](https://arxiv.org/abs/2203.02155)

这也解释了 SFT 的边界：它学习示范分布中的行为，不等于保证事实正确，也不等于直接优化某个偏好评测器。答案有错误、格式不一致或难度失衡，训练目标仍会奖励对这些样本的拟合。

## 预测位置与目标位置相差一格

设模板后的序列为 `[BOS, U, A, X, Y, EOS]`，其中 U 代表用户内容，A 是助手起始标记，X、Y 是答案。位置 A 输出的 logits 应预测 X；位置 X 预测 Y；位置 Y 预测 EOS。若只训练答案及其结束，可以把同长 labels 写成 `[-100,-100,-100,X,Y,EOS]`，然后使用前一个位置的 logits 与后一个位置的 label 配对。

```python
import torch
import torch.nn.functional as F

# 六个输入位置，十个词表项；仅为损失检查构造随机 logits。
torch.manual_seed(1)
logits = torch.randn(1, 6, 10, requires_grad=True)
labels = torch.tensor([[-100, -100, -100, 4, 5, 2]])
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
valid = shift_labels.ne(-100)
count = valid.sum()
assert count.item() == 3
loss_sum = F.cross_entropy(
    shift_logits.reshape(-1, 10), shift_labels.reshape(-1),
    ignore_index=-100, reduction="sum",
)
loss = loss_sum / count
loss.backward()
assert torch.all(logits.grad[:, :2, :] == 0)
assert torch.all(logits.grad[:, -1, :] == 0)
```

`-100` 不是词表中的特殊 token，而是此处交叉熵 API 约定忽略的目标索引。它只改变损失计算，不删除输入，也不自动构造 attention mask。PyTorch 文档说明了 `ignore_index` 与 reduction 的行为。[CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

很多 causal language model 在接收 `labels` 后，会在内部完成这一格对齐。若外部先移动，模型内部又移动一次，就会训练错误目标。接入框架时必须确认“谁负责 shift”，不能把手写损失示例不加修改地塞进现成训练器。

## 不监督提示，并不等于不学习利用提示

用户 token 的预测损失为零，但它们仍参与前向计算，并影响答案位置的隐藏状态。答案损失通过注意力等路径反传，因此模型仍能学习如何使用提示信息。需要屏蔽的是目标项，而不是把用户输入从上下文中隐藏。

前面代码中，未监督位置的独立 logits 梯度为零，这是局部检查；真实模型中更早位置的隐藏状态或共享参数仍可能收到梯度。混淆这两层，会导致错误地以为“只训练答案就可以不计算 prompt”。

多轮数据还需决定监督范围。只训练最后一轮，适合某些明确任务；训练所有助手轮次，提供更多监督，但也提高较长对话的权重。是否监督助手起始标记、轮次结束标记，应与推理格式和任务目标一致。TRL 的 `assistant_only_loss` 依赖适合的对话格式与模板能力，不能仅凭配置项名称判断实际标签。[TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)

## 平均损失暗含样本权重

假设短答案有 10 个 token，长答案有 90 个 token。全局 token 平均使长答案贡献约九倍损失项；先对每条答案平均、再对样本平均，则给两条样本相同总权重。这两种目标都可以定义，但必须知道自己实现的是哪一种。

梯度累积进一步放大这个问题。如果每个微批次的有效回答长度差异很大，直接把各批均值除以累积步数，不等于全局 token 平均。应按有效 token 数加权，或采用明确支持这种归一化的训练实现。验证 loss 也应该累计损失总和与有效 token 总数，避免最后一个小 batch 被错误赋予相同权重。

对教学中的三个目标，若正确 token 的预测概率分别为 0.5、0.25、0.5，平均负对数似然为约 0.924，自然指数得到的 perplexity 约为 2.52。这只是手算例子。比较 perplexity 时必须保持 tokenizer、数据、mask 与截断策略一致；不同目标集合上的数字没有直接可比性。

## 为什么先做小样本过拟合

在完整训练前，固定十几条短而清晰的样本，多次重复训练。预期是损失显著下降，模型能够复现这些有限答案。这只能证明基本优化链路可工作，不能证明泛化或产品质量。若此步骤失败，继续增加数据往往只会掩盖问题。

排错时先打印一条样本的原文、模板文本、token ID 和逐位置 label，再确认可训练参数确实在更新。如果有效目标数为零，查截断与答案边界；如果输出只重复提示，查模板和 mask；如果训练 loss 下降但验证生成异常，查训练/推理前缀、结束条件和评测输入是否意外包含答案。

“loss 越低越好”也有前提。重复样本可以让训练 loss 很低，过长模板的监督可能稀释真正答案的信号，过拟合会让验证指标停滞。应同时观察回答 token 的 loss、目标任务得分，以及固定提示下的真实生成。

## 个人主机验收方案

先运行人工 logits 示例，把 labels 中一个有效位置改为 -100。预期有效计数从 3 降到 2，对应的 logits 梯度归零；其他目标仍有贡献。再手动改变被忽略位置的 logits，确认总损失不变。这样验证的是损失函数的确切语义。

第二阶段在本地小模型上做十几条样本的过拟合试验，运行前保存参数摘要，结束后比较可训练参数变化，并重新加载 checkpoint 生成答案。此阶段需要读者自行准备模型与环境，本篇没有执行或报告训练结果。

验收应同时满足：每条样本有有效目标、标签只移动一次、padding 不参与损失、批次归一化符合定义、训练和推理模板一致、保存后加载的输出可核对。把这些条件写清楚，之后比较学习率、LoRA rank 或不同数据配方才有意义。
