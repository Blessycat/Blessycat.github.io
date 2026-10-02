---
title: "模型实际读到的是什么：分词、特殊 token 与聊天模板"
date: 2026-10-02
permalink: /posts/ai-infra/tokenization-chat-templates/
excerpt: "从文本到 token ID 检查训练与推理的输入契约，避免模板、边界和特殊 token 不一致。"
categories: [AI Infra]
tags: [Tokenizer, 聊天模板, 数据处理, SFT]
series: AI Infra 与后训练
series_order: 5
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：理解序列、整数索引和自回归预测。可选代码需要已经安装 Transformers，并准备一份本地 tokenizer 文件；不加载模型权重，也不要求 GPU。

对人而言，一段聊天包含角色、内容和轮次。对语言模型而言，最终输入是一串整数。训练时的整数序列与推理时的整数序列如果不遵循同一种格式，即使自然语言内容完全相同，也可能形成显著分布差异。本文围绕一个工程问题展开：怎样证明训练器和生成器实际使用了同一套输入协议？

* 目录
{:toc}

## token 不等于字，也不等于词

分词器通常由规范化、预分词、子词模型和后处理等阶段组成。某些步骤会处理大小写或 Unicode 形式，有些会在空白和标点处分段，最后再映射到词表 ID。并非所有 tokenizer 都启用同样的步骤。[Hugging Face Tokenizers pipeline](https://huggingface.co/docs/tokenizers/pipeline)

子词方法在词表大小和序列长度之间折中。词表只收录整词会遇到生僻词，逐字符又常使序列很长。BPE 可以从较小单位开始，把训练语料中常见的相邻单位逐步合并。一个教学例子是把 `l o w` 合为 `lo w`，再合为 `low`；但实际实现可能以字节为基础，包含特殊空白标记，不能据此猜测任何模型的真实 ID。[Subword Units 原论文](https://arxiv.org/abs/1508.07909)

相同的一百个汉字，在不同词表下可能得到不同 token 数。英文、代码、数字串、罕见符号的分词效率也可能不同。所以训练预算应统计实际 tokenizer 的结果，不能统一使用“一个汉字等于一个 token”的换算。

词表大小还影响 embedding 和输出投影。若隐藏维度为 2048，新增 1000 个词表项，单独一份 embedding 就增加约 204.8 万个参数；是否还新增独立输出参数，取决于是否共享权重。修改 tokenizer 不是单纯的文本预处理操作，可能改变模型参数接口。

## 聊天模板把结构变成序列

设一条记录包含 system 指令、user 问题和 assistant 答案。模板负责把角色标记、换行、轮次结束标记和内容连接起来。不同模型训练时使用的标记未必相同；把一个模型的“看起来很合理”的格式套到另一个模型上，未必有效。

下面是概念示意，不是某个真实模型的模板，也不能直接用它训练现有模型：

```text
[开始]system
回答保持简洁。[轮次结束]
[开始]user
2 加 3 是多少？[轮次结束]
[开始]assistant
5。[轮次结束]
```

训练样本通常含完整答案。推理输入则通常截止到用户消息，再由模板添加助手回答的起始部分。`add_generation_prompt` 的意义是请求这种起始提示；具体添加哪些 token、是否有作用，取决于模板。继续补全最后一条已有消息是另一种模式，不应与新开一条助手消息混用。[Transformers Chat templates](https://huggingface.co/docs/transformers/chat_templating)

EOS、BOS、轮次结束 token 与 padding token 的作用也要分别确认。轮次结束不一定等同于整段会话结束。训练若没有教会模型合适的结束标记，生成可能不停续写；终止条件配置错误，则可能过早停下或不停止。

## 在本地检查两条处理路径

以下示例假设 `./tokenizer` 中已存在与目标模型同 revision 的 tokenizer 文件，且内含聊天模板。使用 `local_files_only=True` 避免隐式下载。若没有模板，程序应当报错并让实验者解决输入协议，而不是随手填一个通用格式。

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "./tokenizer", local_files_only=True
)
messages = [
    {"role": "user", "content": "把 2 和 3 相加，只给数字。"},
    {"role": "assistant", "content": "5"},
]
text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
ids_a = tok.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    return_dict=False,
)
ids_b = tok(text, add_special_tokens=False)["input_ids"]
assert ids_a == ids_b
print(repr(text))
print(len(ids_a), ids_a)
print(tok.convert_ids_to_tokens(ids_a))
```

关键是第二条路径使用 `add_special_tokens=False`。模板通常已经负责加入必要标记，再要求 tokenizer 自动补一次，可能出现重复 BOS 或 EOS。打印 `repr(text)` 而不是仅打印 text，能看见换行、制表符和尾部空格。

解码回文本适合人眼检查，但不应作为唯一断言。规范化可能不可逆，`skip_special_tokens=True` 还会主动隐藏关键边界。应保存 token ID 序列、模板摘要和 tokenizer revision，才能发现“肉眼一样、实际输入不同”的情况。

## 回答边界不能靠字符串猜

SFT 常需要只训练助手回答，因此要知道答案对应的 token 区间。一个危险办法是分别分词 prompt 和完整文本，再把 prompt 的 token 数当作精确边界。子词合并可能跨越拼接边界，模板末尾的空白也可能变化，两次分词并不保证前缀完全一致。

更稳妥的做法是使用模板提供的助手区域掩码，或利用能够可靠映射字符位置的 tokenizer offset，并以真实样本验证。框架支持助手专属损失时，也可能要求模板具备对应的生成区域标记；只设置一个布尔参数并不能证明 mask 正确。[TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)

多轮对话尤其要说明训练策略：所有助手轮次都监督，还是只监督最后一次回答？系统指令和用户消息通常仍是上下文，只是不作为监督目标。工具调用和工具结果还可能有额外角色与格式，不宜用两轮聊天的简单拼接逻辑直接处理。

## 截断是一种数据选择

若最大长度为 512，模板后的序列长达 700，简单保留前 512 个 token 可能把答案全部丢掉。模型仍有输入，batch 也能构造，但有效监督 token 为零。保留最后 512 个又可能删掉问题和系统约束。

正确策略取决于任务：按完整轮次删除较早历史、拒绝超长样本、单独设置 prompt 与 answer 预算，都是可以明确记录的选择。无论选哪种，都应统计截断前后长度、答案保留比例和零监督样本数。长度过滤之后，训练集的任务分布可能改变，例如复杂问题更容易被删除；这个变化需要在数据报告里出现。

## 主机实验与验收

准备六条自拟样本，覆盖中文、英文、代码、数字、空白与多轮对话。分别记录原始字符数、模板后 token 数、特殊 token 位置和助手监督位置。故意添加一次重复特殊 token，观察 ID 序列在哪里发生变化；再将长度上限压到很小，检查答案是否被截断。

验收标准不是“decode 看起来顺眼”，而是直接模板分词与分两步处理得到相同 ID；每条训练样本至少有一个有效目标 token；推理前缀符合训练模板；结束 token 与生成终止设置一致。把这六条样本保存为小型输入契约测试，以后升级依赖或更换模型时重新运行。

本文中的数字是教学估算，代码是可执行的检查方案，没有报告具体 tokenizer 的实测压缩率。实际词表和模板不同，结果也会不同；实验的价值正是用可观察的 ID 序列取代格式猜测。
