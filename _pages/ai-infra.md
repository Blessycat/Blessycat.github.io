---
layout: single
permalink: /ai-infra/
title: "AI Infra 与后训练"
excerpt: "按依赖顺序阅读模型训练、偏好优化、分布式系统与推理工程。"
author_profile: true
lang: zh
locale: zh-CN
---

这组笔记把“模型为什么这样更新”与“系统怎样支撑更新”放在一起讨论。每篇都可以独立阅读，也可以按下面的顺序推进。先修知识、推导、具体例子和练习写在各篇正文中。

**起点：** [学习路线、资源与二十四个实战项目](/posts/ai-infra/learning-roadmap/)

[CPU 实验室](/ai-infra/labs/) · [分类](/categories/) · [标签](/tags/) · [RSS](/feed.xml)

## 基础、数据与单卡训练

先建立张量、token、损失和显存的统一认识。完成单卡 SFT 闭环后，再进入偏好优化。

{% assign notes = site.posts | where: "series", "AI Infra 与后训练" | sort: "series_order" %}
{% for post in notes %}
{% if post.series_order >= 2 and post.series_order <= 11 %}
- **[{{ post.title }}]({{ post.url | relative_url }})** — {{ post.excerpt }}
{% endif %}
{% endfor %}

## 偏好优化与强化学习

重点检查概率、优势、奖励与数据分布之间的关系。先计算 CPU 玩具例子，再阅读框架实现。

{% for post in notes %}
{% if post.series_order >= 12 and post.series_order <= 21 %}
- **[{{ post.title }}]({{ post.url | relative_url }})** — {{ post.excerpt }}
{% endif %}
{% endfor %}

## 分布式训练、推理与工程闭环

从通信与内存生命周期出发，理解性能优化的条件，并把检查点、压测和可观测性纳入实验。

{% for post in notes %}
{% if post.series_order >= 22 and post.series_order <= 31 %}
- **[{{ post.title }}]({{ post.url | relative_url }})** — {{ post.excerpt }}
{% endif %}
{% endfor %}

## 选择一条实践路线

- **只有 CPU：** 环境与复现 → 数据管线 → DPO 玩具实验 → 奖励验证器 → 排队模拟。
- **有单张 GPU：** SFT → LoRA → 固定评测 → adapter 保存加载 → 推理性能对照。
- **已有分布式基础：** rollout 一致性 → FSDP/ZeRO → 检查点 → 后训练系统设计。

完整的学习成果应包括一份能够重新运行的实验、一份解释失败样例的报告，以及一个明确写出适用范围的结论。
