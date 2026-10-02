---
title: "一个可审计的单卡 SFT 实验：从合成任务到独立评测"
date: 2026-10-02
permalink: /posts/ai-infra/single-gpu-sft-lab/
excerpt: "设计一套规模受控的单卡监督微调实验，用固定数据、LoRA、严格输出校验和重载检查完成闭环。"
categories: [AI Infra]
tags: [单卡训练, SFT, LoRA, 实验设计]
series: AI Infra 与后训练
series_order: 11
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：本系列的模板、数据、SFT、LoRA 和评测篇。数据生成只需 Python；训练方案面向已配置 CUDA、PyTorch、Transformers、Datasets、TRL 和 PEFT 的主机，并要求事先准备本地小模型。以下是一份实验设计，未执行 GPU 训练，不包含任何声称已测得的提升。

第一场单卡实验适合回答一个小问题：在固定计算预算下，适配器微调能否提高某个格式转换任务的准确率，同时保留基本通用行为？不要同时比较五个模型、三种量化和多个任务，否则失败后难以定位原因。本实验选择普通 LoRA 与短上下文，先把整个证据链做完整。

* 目录
{:toc}

## 用能确定评分的任务排除歧义

任务输入是一行虚构记录，例如“条目 item-0042，颜色红，数量 7”，要求只输出 `{"color":"red","count":7}`。颜色映射固定为红、蓝、绿对应 red、blue、green，数量保留整数。答案由规则生成，因此能检查格式与内容，不需要外部裁判。

这不是现实业务能力证明，也不是检验复杂推理的基准。它的价值是让数据、模板、loss 和保存加载中的问题有清晰症状。确定性规则程序也是一个应保留的强基线：如果目标任务本来可以被简单规则完整解决，微调实验只是学习训练流程，不能因此宣称模型方案更划算。

每个虚构条目生成两种措辞，整组进入同一个集合。下面生成 1200 条训练、200 条验证、200 条测试记录；条目 ID 不承担真实身份含义。

```python
import json
import random
from pathlib import Path

rng = random.Random(42)
colors = [("红", "red"), ("蓝", "blue"), ("绿", "green")]
out = Path("data")
out.mkdir(exist_ok=True)
buckets = {"train": [], "validation": [], "test": []}
for i in range(800):
    split = "train" if i < 600 else "validation" if i < 700 else "test"
    zh, en = rng.choice(colors)
    n = rng.randrange(100)
    item = f"item-{i:04d}"
    target = json.dumps({"color": en, "count": n}, separators=(",", ":"))
    inputs = [f"条目 {item}，颜色{zh}，数量 {n}。",
              f"{item} 有 {n} 件，标记颜色为{zh}。"]
    for text in inputs:
        buckets[split].append({
            "group_id": item,
            "prompt": [{"role": "user", "content":
                "将记录转换为 JSON，仅含 color 和 count，颜色用英文：" + text}],
            "completion": [{"role": "assistant", "content": target}],
        })
for split, rows in buckets.items():
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    (out / f"{split}.jsonl").write_text(payload + "\n", encoding="utf-8")
```

两种模板在各集合都存在，因此主评测测的是同类分布中的新条目，而不是全新任务格式。可以额外准备措辞不同的挑战集，但应单独报告。相同颜色数量组合跨集合出现是任务本身的有限标签空间，不要把标签重复误判为整条输入泄漏；仍需检查完整输入和来源组不交叉。

## 在训练前冻结实验条件

选择已经在本地、约 0.5B 到 1.5B 量级的解码器指令模型作为候选范围，并核对许可证与架构。这个范围不是任何显卡都能训练的保证。记录模型、tokenizer、依赖版本与文件摘要；确定聊天模板和 EOS，不在训练途中更换它们。

先用底座在完整验证集生成输出，保存输入、原始文本和解析结果。固定为 greedy decoding，最大新增 token 取 64；若答案被截断，应在正式比较前调整两者共同预算。保留十几条与训练任务无关的简单回归提示，用来观察格式学习是否侵入其他回答。

随后检查十条训练样本的实际 labels，确保 prompt 被忽略、completion 含有效监督、padding 被忽略，且没有重复移动标签。TRL 对 prompt-completion 数据支持 completion-only loss，实际行为应以所固定版本和打印出的 batch 为准。[TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)

## 一个可执行配方需要明确前提

以下训练脚本要求 `./model` 已包含本地模型和 tokenizer，模型真实存在 `q_proj` 与 `v_proj` 层，模板已通过前述验收。若架构名称不同，应先列出模块并修改目标层，不能盲目照抄。此脚本不自动获取模型，也不启用 QLoRA。

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

assert torch.cuda.is_available()
bf16 = torch.cuda.is_bf16_supported()
tok = AutoTokenizer.from_pretrained("./model", local_files_only=True)
assert tok.chat_template and tok.eos_token_id is not None
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "./model", local_files_only=True,
    dtype=torch.bfloat16 if bf16 else torch.float16,
)
model.config.use_cache = False
data = load_dataset("json", data_files={
    "train": "data/train.jsonl", "validation": "data/validation.jsonl"
})
args = SFTConfig(
    output_dir="runs/sft", max_length=256, packing=False,
    completion_only_loss=True,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=8, max_steps=150,
    learning_rate=1e-4, warmup_steps=10,
    bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
    logging_steps=10, eval_strategy="steps", eval_steps=50,
    save_steps=50, save_total_limit=2, report_to="none", seed=42,
)
trainer = SFTTrainer(
    model=model, processing_class=tok, args=args,
    train_dataset=data["train"], eval_dataset=data["validation"],
    peft_config=LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none",
        task_type="CAUSAL_LM",
    ),
)
trainer.train()
trainer.save_model("runs/final-adapter")
tok.save_pretrained("runs/final-adapter")
```

配置 API 会随版本变化，复建时应使用实验锁定的版本，而不是在旧环境里猜参数名。Trainer 管理训练流程，PEFT 管理适配器参数化与保存；两者职责需要分清。[Transformers Trainer](https://huggingface.co/docs/transformers/trainer) · [PEFT Quicktour](https://huggingface.co/docs/peft/quicktour)

## 按阶段推进，而不是一次跑到底

第一阶段只取 16 条训练记录，进行过拟合检查；不把其结果用于最终性能结论。第二阶段做 10 个完整 optimizer step 的资源试跑，观察是否出现非有限 loss、梯度异常或 OOM，并测量第一次更新和稳定步骤的峰值。显存不足先缩短长度或减小模型，再评估是否需要量化；记录每次改变。

正式阶段固定一个配方。微批次 1、累积 8 时，每次更新名义上处理 8 条样本，150 步约为 1200 条，即这个训练集的一轮；具体仍受数据加载与末批处理影响。256 长度上限对应最多约 307,200 个输入位置，但动态 padding 下实际处理量不同，回答监督 token 又是另一项。日志应分别统计它们。

不要用验证 loss 选完 checkpoint 后又用测试集选一次。若需要比较学习率，可先在验证集上比较少量候选，再冻结最终方案。单卡优化手段会改变吞吐与内存，HF 官方训练优化概览适合查找对应机制，但不能直接把文档中的性能数字移植到本机。[Training optimization overview](https://huggingface.co/docs/transformers/v5.17.0/optimization_overview)

## 把失败与恢复也纳入实验

正式运行前，应区分“用于部署的适配器导出”和“用于继续训练的 checkpoint”。前者只需恢复模型行为，后者还需要优化器、调度器、随机数状态和进度信息。仅加载适配器继续训练，并不等于无缝恢复原来的优化轨迹。可在小规模试跑中中断一次，再按训练器支持的恢复方式检查步数与学习率是否连续。

如果 loss 很快下降到接近零，也不应立即延长实验。这个任务规模小、规则简单，重复训练可能只是记住格式。先检查验证结果与失败类别，再决定是否增加难度，例如改变字段顺序或加入无关描述；新增测试应作为后续任务版本，不得混入已冻结的结论。

耗时预算也应事先定义。用试跑测得的稳定每步时间估算剩余运行时长，另计评测和保存时间。这里没有预设某张显卡的速度，也没有把短序列试跑的吞吐外推为任意输入长度的吞吐。实验结束后的资源记录，应使读者能知道结论花了多少实际计算代价。

## 保存后重载，再做独立评测

重新启动进程，以完全相同的底座加载适配器和 tokenizer。先比较固定提示上的训练结束输出与重载输出，再生成测试集答案。解析时只接受一个完整 JSON 对象，键集合必须恰为 color、count，count 必须是整数而不是布尔值，字段与规则答案一致。额外解释文字应计入格式失败，而不是人工替模型修补。

分别报告 JSON 有效率、字段完全正确率、各颜色与数量范围切片，并按同一条目组做成对比较。两种措辞属于同一组，不能假装是两个完全独立来源。额外回归提示的变化用实例说明，不把十几条样本包装成广泛能力评测。

成功运行的验收是：数据组无交叉、mask 正确、参数发生更新、checkpoint 可重载、资源日志完整。证明改进还需要：在冻结测试集上优于底座，差异足以支持结论，且退化在可接受范围。若底座已经接近满分，或者微调并未改善，就如实记录，这同样完成了实验。

最终保留一份可审计实验包：配置与依赖、输入摘要、底座输出、训练日志、适配器、tokenizer、候选输出和评分脚本。它使下一次实验能够只改变一个因素，而不是重新猜测上一次到底运行了什么。
