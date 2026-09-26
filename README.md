# AIVC 研究地图与可执行选题库

本仓库交付一份面向研究者的、以 **AIVC = AI/学习式视频压缩（learned/neural video coding）** 为工作定义的研究尽调。主报告为 [`research/AIVC_RESEARCH_MAP_ZH.md`](research/AIVC_RESEARCH_MAP_ZH.md)，并附有机器可读的论文、代码、数据集和竞赛线索索引 [`research/aivc_landscape.csv`](research/aivc_landscape.csv)。

> **时效与边界。** 报告按公开资料整理到 **2024-06 的可核验知识边界**；当前执行环境拒绝访问 Web（HTTP 401/403），所以绝不把“2026 年所有工作”伪装为已经逐一联网核验的事实。报告给出了可复跑的增量检索、去重、代码审计与复现实验协议，正是把这份高密度地图持续更新到提交日的方法。

## 快速入口

1. [范围、术语、证据等级与检索协议](research/AIVC_RESEARCH_MAP_ZH.md#0-范围术语与证据规则)
2. [端到端方法谱系、公式和工程取舍](research/AIVC_RESEARCH_MAP_ZH.md#2-方法谱系从预测编码到生成式码流)
3. [论文/代码/数据/标准/竞赛清单](research/AIVC_RESEARCH_MAP_ZH.md#3-公开生态论文代码数据标准与竞赛)
4. [严谨评测与复现计划](research/AIVC_RESEARCH_MAP_ZH.md#4-评测复现实验与审计)
5. [研究空白、排序后的 ideas 与完整实验设计](research/AIVC_RESEARCH_MAP_ZH.md#5-未解问题与可落地的研究-ideas)

## 使用提醒

不要把不同数据集、不同颜色空间、不同帧数、不同 chroma 格式、不同锚点或仅报 MS-SSIM 的数字放进同一张“排名表”。该报告要求随每个结果发布：码流、解码器 commit、命令、每序列 rate/quality、失败案例和硬件/时间/峰值显存。
