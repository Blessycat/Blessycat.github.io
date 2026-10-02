---
title: "CPU 实验室：六个可运行的后训练小实验"
permalink: /ai-infra/labs/
excerpt: "使用 Python 标准库验证数据拆分、显存预算、DPO、组优势、奖励验证器和排队现象。"
lang: zh
locale: zh-CN
author_profile: true
---

[系列导航](/ai-infra/) · [学习路线](/posts/ai-infra/learning-roadmap/)

这些脚本只依赖 Python 3.10 或更新版本的标准库，不需要下载模型、注册服务或配置 GPU。每个脚本自带正确性断言；正常运行会打印示例结果，断言失败会抛出异常。它们用于验证局部原理，不替代真实模型训练和硬件性能测量。

## 下载与运行

保存下表中的 `.py` 文件，在保存目录打开终端执行。例如：

```bash
python dpo_toy.py
python memory_budget.py --parameters 7000000000 --kv-heads 8 --tokens 4096
```

也可以克隆[本站仓库](https://github.com/Blessycat/Blessycat.github.io)，在仓库根目录执行 `python files/ai-infra-labs/dpo_toy.py`。Linux/macOS 环境若命令名为 `python3`，相应替换即可。

| 实验 | 下载 | 应观察什么 | 对应文章 |
| --- | --- | --- | --- |
| 数据审计与分组拆分 | [data_audit.py](/files/ai-infra-labs/data_audit.py) | 40 条有效样本，1 条重复、1 条无效；训练/测试 group 不相交 | [数据管线](/posts/ai-infra/post-training-data-pipeline/) |
| 静态显存与 KV 预算 | [memory_budget.py](/files/ai-infra-labs/memory_budget.py) | 默认配置 KV 为 0.5 GiB；4-bit 项只是权重理想下界 | [KV cache](/posts/ai-infra/kv-cache-paged-attention/) |
| 两动作 DPO | [dpo_toy.py](/files/ai-infra-labs/dpo_toy.py) | 初始 loss 为 ln(2)，有限差分吻合，偏好动作概率上升 | [DPO](/posts/ai-infra/dpo-objective/) |
| 同提示组内优势 | [group_advantages.py](/files/ai-infra-labs/group_advantages.py) | 全同奖励得到零优势，混合奖励产生正负信号 | [GRPO](/posts/ai-infra/grpo-group-advantages/) |
| 严格整数奖励验证器 | [reward_verifier.py](/files/ai-infra-labs/reward_verifier.py) | 八个测试覆盖正确、错误、伪格式、重复答案、超长输入 | [可验证奖励](/posts/ai-infra/verifiable-rewards/) |
| 单服务台排队模拟 | [queue_simulation.py](/files/ai-infra-labs/queue_simulation.py) | 到达速率超过服务能力后，尾延迟快速增长 | [服务压测](/posts/ai-infra/inference-serving-benchmarks/) |

## 六个值得修改的变量

1. 给数据审计器增加同一 prompt 的矛盾答案检查，思考“重复”与“冲突”为什么不能混为一谈。
2. 将 KV heads 从 8 改为 32，解释相同上下文长度下 KV 内存为何变化；再将 batch 翻倍。
3. 修改 DPO 的 beta，用同样训练步数比较结果；不要把 toy policy 的行为当成大模型的泛化结论。
4. 比较组内奖励 `[0,0,0,1]` 与 `[0,1,2,100]`；思考异常值如何改变信号分配。
5. 给验证器增加负数、前导零和合法但不同格式的答案测试；先写清楚任务是否允许这些形式。
6. 在排队模拟中加入长短混合请求，比较均值与 P95。注意这里没有 GPU batching 和 token streaming，不能把输出称为 TTFT。

## 从小实验走向真实训练

保留同一组正确性判据，再替换底层实现。例如，把 toy DPO 的两个动作换成同一 prompt 下的两条回答时，应先核对 response-only log probability 的求和与 mask；把显存估算换成 CUDA 实测时，应保留估算项，新增峰值、激活与临时缓冲的测量；把假验证器换成代码任务时，需要独立受限执行环境，而非直接执行生成内容。

脚本输出都是人工数据或数学模拟。真实模型实验应另外记录环境、模型 revision、数据指纹、随机种子、评测协议和失败样例。
