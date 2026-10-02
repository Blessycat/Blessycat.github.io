---
title: "把一次训练变成可重建的实验：Linux 与 Python 环境基础"
date: 2026-10-02
permalink: /posts/ai-infra/linux-python-reproducibility/
excerpt: "从解释器、依赖、数据指纹和随机性四个层面，建立能重建、能排错的训练环境。"
categories: [AI Infra]
tags: [Linux, Python, 可复现性, 实验工程]
series: AI Infra 与后训练
series_order: 2
lang: zh
toc: true
read_time: false
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

先修要求：会运行 Python 脚本，理解文件、目录和命令行参数。本文讨论个人主机上的实验组织；命令以 Linux 或 WSL 中的 Bash 为例，不要求 GPU。

训练结果首先是一个程序的产物。程序除了读取代码，还读取包版本、输入文件、环境变量和随机数状态。只留下 `train.py`，相当于只留下菜谱中的烹饪步骤，却没有记录食材和分量。本篇要解决的问题是：几天后换一个空目录，能否重新建立同样的实验条件，并解释结果为什么相同或不同？

* 目录
{:toc}

## 先确定究竟在运行哪个 Python

终端里出现 `(venv)` 不是充分证据。编辑器、定时任务与终端可能使用三个不同的解释器。Python 的虚拟环境隔离的是包安装位置，通常仍依赖创建它的基础解释器；环境目录不适合直接复制到另一台机器。官方将它视作应该能够重新创建的对象。[Python venv 文档](https://docs.python.org/3/library/venv.html)

下面的命令只创建环境和检查路径，不安装训练框架：

```bash
mkdir -p sft-lab/{src,data,configs,runs}
cd sft-lab
python3 -m venv .venv
.venv/bin/python -c 'import sys; print(sys.executable); print(sys.version)'
.venv/bin/python -m pip --version
```

始终通过 `python -m pip` 安装包，能把安装操作与解释器绑在一起。用裸 `pip` 时，PATH 中更靠前的可执行文件可能指向另一套环境。Windows 原生 PowerShell 对应路径是 `.venv\Scripts\python.exe`，不必为了运行它修改全局执行策略。

学习 Linux 最有价值的部分是理解进程边界。当前工作目录影响相对路径，环境变量影响子进程，退出码说明执行是否成功，标准输出与标准错误承载不同信息。把一条命令放进脚本后失败，先比较 `pwd`、解释器路径和实际参数，往往比重装包有效。文件名大小写、换行符和路径分隔符，也可能让 Windows 上的脚本在 Linux 中暴露问题。

## 依赖清单需要记录选择，也需要记录结果

直接依赖是主动选择的库，解析后的依赖是安装器最终得到的完整包集合。两者各有用途：前者便于维护，后者便于复建。`pip freeze` 是当前环境快照，并不能表达最初为何选择这些包，也不能自动记录操作系统、GPU 驱动或包下载来源。

对于已经测试过的实验，可以保存版本固定的依赖文件；进一步要求严格重建时，记录哈希并保留适合目标平台的 wheel。哈希验证能帮助发现包文件发生变化，但不会让 Linux 的 wheel 自动适用于 Windows。pip 官方的可重复安装指南区分了固定版本、哈希校验和安装包归档这些层次。[pip Repeatable Installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/)

合理的目录可以这样分工：`src` 放代码，`configs` 放参数，`data` 放输入或输入索引，`runs` 放每次运行的证据。虚拟环境、模型权重和生成的大文件不要混入源码版本管理。即使使用容器，也要记录镜像摘要；只有浮动标签时，下一次拉取未必得到同一份内容。

## 数据版本比随机种子更容易被忽略

假设训练集原先有 10,000 条记录，清洗脚本更新后只剩 9,730 条。如果仍把两次运行都标成“seed=42”，结果当然不能直接比较。数据指纹应对应实际输入字节或规范化后的记录，而不是仅记录文件名。

下面的标准库脚本适合保存为 `src/manifest.py`。运行前需在 `data/train.jsonl` 放置教学数据。它只读取文件，不上传任何内容。

```python
import hashlib
import json
import platform
import sys
from pathlib import Path

data_path = Path("data/train.jsonl")
digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
manifest = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "data_sha256": digest,
    "seed": 42,
    "config": {"batch_size": 4, "learning_rate": 0.0001},
}
Path("runs").mkdir(exist_ok=True)
Path("runs/manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

真实训练还应增加代码提交号、未提交改动说明、模型和 tokenizer 的固定 revision、依赖清单、硬件型号与驱动版本。路径中如果含有私人信息，公开日志前要移除；数据摘要通常比公开原始样本更合适。摘要只能证明内容是否一致，不能证明来源授权或质量正确。

## “可复现”至少有三种含义

第一层是可重建：新环境能加载同样的代码和输入。第二层是数值复现：在明确的软件与硬件条件下，输出在指定容差内一致。第三层是结论复现：换种子或合理改变样本后，主要结论仍成立。这三层不能互相替代。

设置种子控制的是随机序列的起点。Python、NumPy 和 PyTorch 可能各有随机数生成器；数据加载 worker 又引入进程级状态。即便种子完全相同，浮点数加法顺序不同也可能引起微小误差，因为有限精度下加法不严格满足结合律。小误差进入长训练后可能逐步放大。

PyTorch 提供确定性算法开关，但官方明确不保证跨版本、平台或 CPU/GPU 的完全一致。确定性策略也可能降低性能，或遇到没有确定性实现的算子时直接报错。[PyTorch Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

因此，日志不要只写“已固定随机种子”。更明确的说法是：在相同环境、相同数据顺序与同一设备上，比较两次运行的前若干步 loss；差异容差事先定义。若目标是比较两个训练配方，则要运行多个种子，观察改进是否大于种子波动。

## 一个不需要 GPU 的主机实验

准备三条自拟 JSONL 记录，用上述脚本保存指纹。复制输入到另一个新目录，重建虚拟环境，再生成一份 manifest。预期现象是解释器绝对路径可能改变，而数据摘要保持一致。这是教学预期，本篇没有声称完成过这次运行。

接着只把一条记录末尾增加一个空格，再运行指纹脚本。字节级摘要应改变。由此可以讨论两个问题：字节完全一致是不是任务真正需要的标准？若希望忽略无意义空白，应在哪一步规范化，并怎样记录规范化代码的版本？对于代码数据，空白可能有语义，不能直接套用自然语言清洗规则。

最后写一个固定种子的抽样脚本，连续生成五个随机数。比较“每次启动进程时设置种子”和“每次抽样前都重设同一种子”的差别。前者得到可重复的序列，后者反复得到序列开头，可能意外消除训练数据的随机性。

验收标准是能交付一份简短运行说明：从空目录执行哪些命令、输入指纹是什么、输出保存在哪里、哪些差异允许存在。另一位读者不必猜测工作目录，也不必手工修改源码中的机器路径。若结果变化，能先定位到环境、数据、参数或随机性中的某一层，实验基础才算建立起来。

## 常见的排错顺序

导入失败先查解释器与包位置；能导入但 CUDA 不可用，再查框架构建类型、驱动与设备，而不是一口气升级全部组件。训练能跑但结果异常，先确认数据条数、首批样本和标签，再怀疑数值内核。所谓“清理环境”不应成为没有证据的默认操作。

一个好的实验目录不要求工具繁多。它要求每次运行有独立标识，参数不会被默默覆盖，失败日志仍被保留，最终产物能追溯到输入。这些习惯会在后面的分词、SFT 与显存排查中持续减少歧义。
