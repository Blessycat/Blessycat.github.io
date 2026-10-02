---
permalink: /
title: ""
author_profile: true
lang: zh
locale: zh-CN
description: "Julie · AI Infra · 清华大学硕士"
redirect_from:
  - /about/
  - /about.html
---

## AI Infra 与后训练学习笔记

从训练数据和目标函数出发，理解模型如何学习，再追踪显存、通信、生成和评测中的工程问题。这里整理原理推导、论文阅读和个人主机上的实践方法。

**首次阅读：** [学习路线、课程资源与实战项目](/posts/ai-infra/learning-roadmap/)

[系列目录](/ai-infra/) · [全部博文](/year-archive/) · [CPU 实验室](/ai-infra/labs/) · [RSS 订阅](/feed.xml)

### 按问题阅读

| 想弄清的问题 | 推荐入口 |
| --- | --- |
| 如何开始一次正确的微调？ | [训练循环](/posts/ai-infra/pytorch-autograd-training/) → [SFT](/posts/ai-infra/supervised-finetuning/) → [单卡实验](/posts/ai-infra/single-gpu-sft-lab/) |
| 偏好数据如何改变模型？ | [奖励建模](/posts/ai-infra/preference-reward-modeling/) → [DPO](/posts/ai-infra/dpo-objective/) → [GRPO](/posts/ai-infra/grpo-group-advantages/) |
| 为什么训练慢、显存不够？ | [显存与精度](/posts/ai-infra/precision-optimizers-memory/) → [分片训练](/posts/ai-infra/fsdp-zero-sharding/) → [性能分析](/posts/ai-infra/profiling-training-bottlenecks/) |
| 如何验证模型和系统的改进？ | [评测设计](/posts/ai-infra/evaluation-design/) → [推理压测](/posts/ai-infra/inference-serving-benchmarks/) → [系统设计](/posts/ai-infra/post-training-system-design/) |

### 内容约定

文章区分原理、教学实验和性能测量；参考资料优先链接官方文档与原论文。硬件预算说明适用条件，代码示例说明运行前提。实验从小规模正确性检查开始，再逐步增加模型、数据和并发规模。
