---
title: "AI Infra 与后训练：学习路线、资源和个人主机实验"
date: 2026-10-02
permalink: /posts/ai-infra/learning-roadmap/
excerpt: "把模型训练、偏好优化、分布式系统和推理服务连成一条可实践的学习路线，附课程、论文与分硬件等级的项目清单。"
categories: [AI Infra]
tags: [学习路线, 后训练, 实战项目]
series: AI Infra 与后训练
series_order: 1
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [全部文章](/year-archive/) · [实验脚本](/ai-infra/labs/)

后训练的难点往往出现在算法和系统的交界处：为什么 loss 在下降，回答却没有改善？为什么模型权重能够放进显存，训练还是 OOM？为什么提高生成吞吐后，整轮强化学习反而变慢？要回答这些问题，需要同时理解目标函数、数据路径、内存生命周期和评测方式。

这份路线以“能够解释、复现、定位问题”为目标。先用 CPU 上的小实验建立直觉，再用小模型完成单卡闭环，最后学习多卡与多机系统。这里的项目是供读者选择的通用教学练习，不代表已经完成的实测；显存档位是起点建议，实际需求必须通过具体配置验证。

* 目录
{:toc}

## 先建立一张任务地图

预训练通过大量文本学习通用预测能力；监督微调（SFT）用示范数据塑造回答形式与任务行为；偏好优化利用回答之间的比较；基于可验证奖励的强化学习则让模型从可计算的反馈中调整策略。它们会共享一些组件，却不共享同一套数据假设和损失函数。

一次典型后训练迭代可以画成如下数据流：

```text
任务与评测协议 → 数据清洗/拆分 → 基线模型 → SFT
                                        ↓
固定测试集 ← checkpoint ← 策略更新 ← 奖励/验证器 ← rollout
     ↓                        ↑                       ↑
误差分类与人工审阅          reference/critic        权重同步
     ↓
服务部署 → 延迟、成本、稳定性 → 下一轮实验假设
```

SFT 主要反复读取固定数据；在线 RL 还要不断生成新样本、计算奖励、同步权重。因此，相同规模的模型，完整 RL 流水线所需资源可能远高于一次 LoRA 微调。学习时先识别系统里同时存活的模型副本，再讨论选哪一个框架。

## 六个阶段与可检查的学习成果

| 阶段 | 要回答的问题 | 阅读入口 | 完成标志 |
| --- | --- | --- | --- |
| 工程基础 | 代码、数据、环境如何保持一致？ | [环境与复现](/posts/ai-infra/linux-python-reproducibility/)、[自动求导](/posts/ai-infra/pytorch-autograd-training/) | 从空目录重建小实验，重启后恢复训练状态 |
| 模型与数据 | 哪些 token 被预测，哪些被计算损失？ | [注意力](/posts/ai-infra/transformer-attention/)、[聊天模板](/posts/ai-infra/tokenization-chat-templates/)、[数据管线](/posts/ai-infra/post-training-data-pipeline/) | 手算一个训练样本的 shift、mask 和有效 token 数 |
| 单卡后训练 | 微调到底改变了什么？ | [SFT](/posts/ai-infra/supervised-finetuning/)、[LoRA](/posts/ai-infra/lora-qlora-memory/)、[单卡实验](/posts/ai-infra/single-gpu-sft-lab/) | 在固定留出集上比较基线与微调结果，并解释失败样例 |
| 偏好与强化学习 | 模型为何偏向一个回答？ | [奖励建模](/posts/ai-infra/preference-reward-modeling/)、[DPO](/posts/ai-infra/dpo-objective/)、[PPO](/posts/ai-infra/ppo-rlhf/)、[GRPO](/posts/ai-infra/grpo-group-advantages/) | 从日志概率重算一条样本的 loss，验证梯度方向 |
| 训练与推理系统 | 时间和显存消耗在哪里？ | [通信](/posts/ai-infra/distributed-collectives/)、[分片](/posts/ai-infra/fsdp-zero-sharding/)、[KV cache](/posts/ai-infra/kv-cache-paged-attention/) | 给出理论预算、实际测量和两者差异的解释 |
| 工程闭环 | 改进是否可靠，能否恢复与服务？ | [评测](/posts/ai-infra/evaluation-design/)、[恢复](/posts/ai-infra/checkpoint-recovery/)、[系统设计](/posts/ai-infra/post-training-system-design/) | 交付可复现实验报告、回归评测和故障恢复记录 |

不必按时间表匆忙前进。一个可调整的安排是：基础与模型占约四周，单卡实验与评测三周，偏好/RL 四周，系统与综合项目五周；每周投入六到十小时只是规划示例。已经熟悉 PyTorch 的读者可以从 SFT 的 mask 检查开始；偏工程的读者也应至少手算一次 DPO 与策略梯度，再优化训练系统。

## 网络资源：按用途选择，而不是全部收藏

以下优先选择作者维护的课程、项目文档和原始论文。文档通常滚动更新；运行实验时应另外记录版本或 commit，而不是把网页中的 `latest` 当成可复现环境。

| 资源 | 适合解决什么问题 | 建议使用方式 |
| --- | --- | --- |
| [Python 官方教程](https://docs.python.org/3/tutorial/) | 模块、迭代器、异常、文件处理 | 配合一个 JSONL 清洗程序学习；没有编程基础时先补基本控制流 |
| [MIT Missing Semester](https://missing.csail.mit.edu/) | Shell、Git、调试与工具使用 | 把命令行、版本管理和性能分析练习迁移到训练脚本 |
| [PyTorch 基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | 张量、自动求导和训练循环 | 先运行 CPU 小网络，再逐项解释每个张量的 shape |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Transformers、数据集、tokenizer、微调 | 学一节便保留一个最小可运行例子，不一次安装所有扩展 |
| [Stanford CS336（2025）](https://cs336.stanford.edu/spring2025/) | 从模型原理走向训练系统 | 将课程作业拆小；完整作业可能超过个人硬件能力 |
| [Spinning Up：RL 核心概念](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) | 状态、动作、回报、策略和值函数 | 用 bandit 或小型环境理解公式，再映射到 token 生成 |
| [TRL 文档](https://huggingface.co/docs/trl/en/index) | SFT、DPO、GRPO 等训练接口 | 每次只选一种 Trainer，阅读对应版本的数据格式和损失定义 |
| [PEFT 文档](https://huggingface.co/docs/peft/en/index) | 适配器、LoRA 配置与模型保存 | 检查可训练参数及 target modules，验证重新加载后的输出 |
| [PyTorch 分布式文档](https://docs.pytorch.org/docs/stable/distributed.html) | 进程组、集合通信和故障定位 | 先在 CPU 上验证 collective 顺序，再看 GPU 通信性能 |
| [vLLM 文档](https://docs.vllm.ai/en/latest/) | 推理调度与服务参数 | 固定输入/输出长度分布，比较延迟与吞吐曲线 |
| [verl 官方仓库](https://github.com/verl-project/verl) | 后训练组件如何组合 | 追踪 actor、rollout、reward 的数据契约，不直接照搬大规模配置 |
| [llama.cpp 官方仓库](https://github.com/ggml-org/llama.cpp) | CPU、量化和本地推理 | 无独立 GPU 时建立推理与量化误差的直觉 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 可配置的模型评测 | 固定 task、版本、prompt、few-shot、解码参数，保存逐样本结果 |
| [Machine Learning Systems](https://mlsysbook.ai/) | 系统约束、可靠性与全生命周期 | 用章节中的系统问题审视自己的实验设计 |
| [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/) | 数据、部署和实验管理 | 重点学习工程方法；历史课程中的具体 API 需用当前文档核验 |
| [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) | 工具调用与代理执行循环 | 先做离线轨迹和假工具，再加入受限执行环境 |

论文阅读可以从 [LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314)、[InstructGPT](https://arxiv.org/abs/2203.02155)、[DPO](https://arxiv.org/abs/2305.18290)、[DeepSeekMath](https://arxiv.org/abs/2402.03300)、[ZeRO](https://arxiv.org/abs/1910.02054)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[PagedAttention](https://arxiv.org/abs/2309.06180) 依次展开。每篇只先做四件事：写出它要解决的限制，找出关键假设，重算一个公式，解释一个消融实验。论文里的最高指标不是个人实验必须达到的目标。

## 个人主机的能力边界

不要从“别人用几张卡”倒推自己的学习路线。先盘点操作系统、RAM、可用磁盘、GPU 型号/显存及支持的算子。下面的档位只用于挑选实验规模，不保证某个参数规模必然可以运行。

| 硬件起点 | 优先完成的任务 | 应避免的默认假设 |
| --- | --- | --- |
| CPU，8–16 GB RAM | 数据审计、损失函数、排队模拟、微型网络、量化小模型推理 | CPU 上能运行不等于具有 GPU 吞吐代表性 |
| 约 8 GB GPU | 数千万到数亿参数模型的短序列训练、LoRA、算子 profile | 不能把量化权重大小当成训练峰值 |
| 约 12–16 GB GPU | 先从 0.5B–1.5B 模型的小 batch/短序列 PEFT 探索 | DPO 可能需要额外 reference 计算；在线 RL 资源更复杂 |
| 约 24 GB GPU | 较大 adapter 实验、受控推理服务、小规模 rollout | “7B QLoRA 可放下”依赖序列长、实现、精度和优化器 |
| 两张 GPU 或更多 | DDP、分片、通信 overlap 的测量 | 不同 PCIe/NVLink 拓扑的结果不可直接比较 |

例如，7B 模型纯 BF16 权重按每参数 2 字节计算约为 14 GB（十进制）；若采用一个每参数总计 16 字节的混合精度 Adam 状态模型，静态训练状态就约为 112 GB，还没有计入激活和临时缓冲。4-bit 权重的理想下界约 3.5 GB，也不包含分组量化元数据和未量化层。计算预算之后，再用缩小的序列长度做探测，记录峰值，而不是先启动一整夜的任务。

Windows 用户可先完成所有纯 Python 实验。CUDA、分布式和高性能推理栈通常更适合在受支持的 Linux/WSL2 环境中学习；macOS 可以选择 CPU、受支持的 MPS 或 llama.cpp 路径，但不能假设 CUDA kernel 和 NCCL 能在这些后端等价运行。依赖安装以各项目的当前平台支持说明为准。

## 二十四个可以逐步完成的实战项目

每个项目都应有输入、输出和验收标准。下面的“可完成”指可以按个人资源缩小范围开展实验，不表示能复现论文规模或线上服务质量。

| 项目 | 硬件起点 | 交付物与验收标准 |
| --- | --- | --- |
| 1. JSONL 数据审计器 | CPU | 检查 schema、空回答、重复样本和分组泄漏；报告每类数量 |
| 2. 聊天模板对照实验 | CPU | 同一对话的 token/角色边界对照表，验证训练与推理模板相同 |
| 3. Loss mask 可视化 | CPU | 打印每个 token 的 label 和 mask；padding 与 prompt 不误入指定损失 |
| 4. 微型语言模型训练 | CPU/小 GPU | 小词表模型在人工任务上过拟合，保存并恢复后输出一致 |
| 5. 显存预算计算器 | CPU | 分开权重、梯度、优化器、KV；标注哪些激活项只能实测 |
| 6. LoRA 参数量计算与注入检查 | CPU/小 GPU | 核对 target modules、可训练参数比例和冻结参数梯度 |
| 7. SFT 格式转换器 | 8 GB 起探索 | 对人工 JSON 格式任务训练 adapter，报告留出集格式通过率 |
| 8. 学习率与数据量消融 | 单 GPU | 固定其他变量，比较三档学习率或数据量，保存失败样例 |
| 9. DPO 二分类策略玩具实验 | CPU | 手算和程序 loss 一致；有限差分检查梯度方向 |
| 10. 小模型偏好优化 | 单 GPU | 构造同提示偏好对，与 SFT 基线比较，记录长度分布变化 |
| 11. Bradley–Terry 奖励拟合 | CPU | 用可控噪声偏好对拟合分数，展示排序与校准的区别 |
| 12. GRPO 组优势计算器 | CPU | 比较全同奖励、混合奖励和异常值，说明零梯度组 |
| 13. 算术验证器对抗测试 | CPU | 覆盖正确、错误、伪格式、重复答案和解析失败 |
| 14. 小规模 RLVR 回路 | 单 GPU，严格缩小模型 | 保存每条 rollout 的版本、reward、logprob；核对更新前后一致性 |
| 15. 拒绝采样与蒸馏 | CPU 可做数据部分 | 对固定生成池筛选，训练时只用训练分区，检查覆盖率下降 |
| 16. 工具调用离线轨迹集 | CPU | 用假计算器/假检索工具验证参数 schema 与终止条件 |
| 17. CPU 多进程 collective | CPU，受支持的后端 | 在两个进程中验证 all-reduce 结果，加入超时和错误诊断 |
| 18. 梯度累积一致性实验 | CPU/单 GPU | 无 dropout、同归一化条件下，对比大 batch 与分批梯度 |
| 19. 激活重算实验 | 单 GPU | 在同一 workload 下记录峰值显存与 step 时间，比较 Pareto 取舍 |
| 20. Checkpoint 故障恢复演练 | CPU/单 GPU | 中途终止后恢复，核对 global step、优化器、采样和 RNG 状态 |
| 21. Attention kernel 对照 | 支持对应 kernel 的 GPU | 小输入校验数值误差，大输入测峰值与耗时，包含预热 |
| 22. KV cache 与并发预算 | CPU 估算/GPU 实测 | 比较 GQA/MHA、不同上下文长度；解释吞吐拐点 |
| 23. 本地推理服务压测 | CPU 或单 GPU | 按负载曲线报告 TTFT、ITL、端到端延迟、错误率和 goodput |
| 24. 后训练实验仪表盘 | CPU | 从日志汇总质量、tokens/s、显存、版本、故障，能追溯每条结论 |

本系列提供六个不下载模型、只依赖 Python 标准库的[起步脚本](/ai-infra/labs/)，对应数据检查、显存/KV 估算、DPO、组优势、验证器和排队模拟。它们验证局部原理；后续文章会说明如何把同一问题迁移到真实训练与服务中。

## 把项目组织成三个闭环

**数据到 SFT。** 选择一个人工生成的格式转换任务，例如把有固定字段的短记录转成 JSON。先写独立验证器，按生成模板分割数据，再保存未训练基线输出。训练时先过拟合 16 条样本，确认 label shift、mask 和保存加载正确；最后比较新模板上的格式准确率。这样能避免把“模型记住了几个例子”误判成泛化。

**偏好到策略更新。** 从两个候选回答的可解释偏好开始，在 CPU 上重算 DPO loss，再扩展到小模型 adapter。为模型偏好、reference 偏好和回答长度分别画图。若平均奖励上升但真实验证器通过率下降，先检查奖励定义、长度偏差和数据分布，不立即增加训练步数。

**训练到服务。** 对一个固定 checkpoint 建立推理基线，改变并发、输入长度和输出长度，记录端到端延迟与错误率。随后启用一个优化，例如更合适的 batching 或量化；重新跑相同任务评测。任何加速都要同时回答两个问题：质量是否保持，测量是否包含排队和失败请求。

## 一份合格的实验记录

每个实验目录至少应包含以下信息，且配置与结果分开保存：

```text
experiment/
  README.md            # 假设、运行入口、硬件、局限
  config.json          # 模型revision、数据hash、超参、seed、模板
  environment.txt      # Python/框架/驱动/依赖版本
  metrics.jsonl        # step、tokens、loss、时间、显存、质量指标
  evaluation.jsonl     # 每条输入、输出、得分与失败类型
  checkpoint/          # 按需保留，并说明可否精确恢复
```

先写成功和失败的判据，再开始实验。例如：“在保持同一测试集格式通过率的前提下，提高满足既定延迟阈值的完成请求数”，比“让推理更快”明确得多。结果不显著也有价值：记录样本量、差异和不确定性，保留反例，下一次实验才知道应该改变哪个变量。

## 如何继续阅读

首次学习可以按[系列目录](/ai-infra/)顺序推进。需要快速建立动手反馈时，先运行 CPU 实验，再读 [SFT 单卡闭环](/posts/ai-infra/single-gpu-sft-lab/)；已经接触过训练系统时，可以从 [rollout 一致性](/posts/ai-infra/rollout-policy-consistency/)与[系统设计](/posts/ai-infra/post-training-system-design/)倒查薄弱环节。

技术框架会变化，判断方法应保持稳定：先定义任务与数据契约，算清资源下界，做最小正确性检查，再测性能，最后用固定评测验证改进。阅读量和模型规模都不能替代这一闭环。
