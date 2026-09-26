# AIVC（AI/学习式视频压缩）研究地图、证据台账与选题设计

**版本：** 2026-09-26（知识核验边界：2024-06）
**读者：** 打算做论文、开源系统、标准化或产品原型的研究者。
**本报告回答：** 到底压什么、有哪些真正不同的方法、哪些结果可复现、尚缺什么，以及怎样把“想法”变成可证伪的研究计划。

---

## 0. 范围、术语与证据规则

### 0.1 工作定义和不混淆的边界

本文的 **AIVC** 指以可训练神经网络取代或增强视频编码器中的预测、变换、量化、熵模型、环路滤波或端到端率失真（R-D）优化的技术。它包含：

* **learning-based video codec / neural video codec (NVC)**：训练一次、对任意视频编码，输出可传输码流；这是本文主线。
* **AI-enhanced conventional codec**：AV1/HEVC/VVC 的工具选择、帧内/帧间预测、滤波、超分、GOP/RDO 被 AI 加速或替换；仍可能以标准码流互通。
* **neural/implicit video representation (INR/NVR)**：每条视频训练网络并传输权重或网络差分；极低码率、离线和视频特定压缩很有价值，但不能和通用实时 codec 混为一谈。
* **generative/perceptual codec**：解码端以生成先验补全细节，优化感知或下游任务而非逐像素保真；必须报告真实性与失真风险。

不把普通“视频生成”“视频理解模型”“只对单帧图像压缩”计入 AIVC。也不将“论文声称优于 VVC”自动解释为通用胜利：指标、配置、色度、锚点和速度可能完全不同。

### 0.2 最小数学对象

视频 (x_{1:T}) 被分为帧内锚点（I）和预测帧（P/B）。第 (t) 帧解码为

\[
\hat x_t = g_s(\hat y_t, c_t),\quad \hat y_t=Q(y_t),\quad y_t=g_a(x_t,c_t),
\]

其中 (c_t) 是从已解码参考帧、运动、时序状态或特征得到的因果上下文。学习式熵编码以

\[
R_t\simeq -\log_2 p_{\theta}(\hat y_t\mid c_t) - \log_2p_{\psi}(\hat z_t)
\]

估计比特，训练常最小化

\[
\mathcal L=\mathbb E\sum_t(R_t+\lambda D(x_t,\hat x_t)) + \alpha L_{lat}+
\beta L_{mem}+\gamma L_{robust}.
\]

(D) 可为 MSE/PSNR、MS-SSIM、VMAF/LPIPS 或任务损失。训练时以均匀噪声、STE 或软量化近似不可导的 (Q)；部署时必须对整数符号做确定性 range/ANS/arithmetic coding。**估计的 likelihood bpp 不是实际码流 bpp。**

### 0.3 证据等级、时间限制与增量检索

表中 `P` 为论文/预印本，`C` 为可检查源码，`D` 为数据页，`S` 为标准组织。链接是审计入口而非“已在本环境下载并跑通”的宣称。当前环境的外网请求返回 HTTP 401/403；因此本版不声称覆盖 2024-06 之后的“全部”工作。要将地图升级到某个提交日，执行：

```bash
# 1) 文献：导出 Crossref/OpenAlex/Semantic Scholar/arXiv 的 CSV，关键词不能只用 AIVC
for q in 'learned video compression' 'neural video coding' \
         'deep video compression' 'implicit neural video representation' \
         'perceptual video compression' 'MPEG AI video coding'; do
  # 通过机构允许的 API/数据库检索，记录 query、日期、返回总数、DOI/arXiv ID
  printf '%s\n' "$q"
done
# 2) 代码：固定到 tag/commit，而不是默认分支
# git clone --recurse-submodules URL && git rev-parse HEAD && git submodule status
# 3) 对每条条目填写：发布日期、任务、训练数据许可、decoder license、真实码流、复现命令
```

去重键优先 DOI/arXiv ID，其次题名+第一作者+年份；一个家族的 conference 版和扩展版均保留并标注关系。把“有 GitHub”与“可复现”分开：后者至少需要 checkpoint、依赖锁定、压缩和解压命令、可解码 bitstream、精确评测脚本。

---

## 1. 研究问题的坐标系

| 轴 | 可选值 | 为什么会改变结论 |
|---|---|---|
| 互通性 | 标准码流增强 / 私有神经码流 / 权重码流 | 决定部署门槛与标准化价值 |
| 时延 | 离线双向 / low-delay P / all-intra / streaming | B 帧和未来上下文不可用于直播 |
| 工作点 | 极低码率 / 中高保真 / 可变码率 | 同一模型通常不能各点最优 |
| 目标 | 像素、感知、语义/任务、可信重建 | PSNR 高不等于人眼或下游好 |
| 输入 | 4:2:0 8-bit SDR / 10-bit HDR / RGB / 360° | RGB 训练和 YUV 交付不可直接比 |
| 资源 | GPU server / mobile NPU / CPU decoder | FLOPs 不等于熵编码、访存与功耗 |
| 鲁棒性 | 无损网络 / 丢包 / 截断 / bit error | 神经上下文错误会跨帧漂移 |

真正的论文问题应落在一个坐标，例如：“在 1080p 4:2:0 10-bit low-delay、CPU 可解码、1% packet loss 下，以 VMAF 与检测 mAP 的 Pareto 曲线超过 VVC anchor”，而非笼统地“用 Transformer 做视频压缩”。

---

## 2. 方法谱系：从预测编码到生成式码流

### 2.1 传统混合编码是不可忽略的强基线

HEVC/VVC/AV1 的共同骨架是分块运动补偿预测、残差变换/量化、上下文熵编码、帧内预测和环路滤波。它们已把几十年经验装进高度优化的 C/C++/SIMD/硬件。AI 方法的公平比较必须说明：x265/x266/VTM/SVT-AV1 版本、preset、tune、GOP、QP、threads、色度和命令行。仅与过时 HM 或仅用单一码率比，会夸大收益。

### 2.2 第一代：显式光流 + 残差（DVC、HLVC、RLVC、SSF）

DVC 的典型管线：FlowNet 类网络估计光流 (v_t)，压缩 (v_t)；将参考帧 warp 为 \(\bar x_t\)；通过补偿网络获得预测 \(\tilde x_t\)；压缩残差 \(r_t=x_t-\tilde x_t\)。优点是解释直观、可替换模块；瓶颈是流本身昂贵、遮挡/新显露区域难以处理、warp 误差造成残差熵高、逐帧误差累积。HLVC 用层级/双向结构，RLVC 用循环状态；SSF 将运动表示改为尺度空间 flow。它们建立了学习式视频 R-D 的范式，但今天不应以其数字充当 SOTA。

**可研究的诊断：** 在每帧记录 motion bits、residual bits、预测 PSNR、occlusion mask 和 scene-cut 标签；否则无法知道改进来自“更强图像压缩器”还是时域建模。

### 2.3 端到端运动表征与多参考预测（FVC/MMVC 等）

不是传输光流，而是让分析变换传输可压缩的 motion latent；解码端从 latent、参考帧和可变形卷积/补偿网络产生预测。多参考和多尺度上下文提升困难运动和遮挡，但使 causal buffer、内存和随机访问复杂化。关键比较包括：仅上一帧、固定多帧、隐状态、显式光流、P/B 的未来参考。必须计算“参考帧数 × 分辨率 × feature channels”的真实显存，而非只报参数量。

### 2.4 上下文条件熵模型（DCVC 家族的核心贡献）

DCVC 将时间上下文直接用于当前 latent 的高斯/混合分布参数预测：

\[
p(\hat y_i\mid \hat y_{<i},\hat z,c)=\mathcal N(\mu_i(\hat z,c),\sigma_i^2(\hat z,c))*\mathcal U(-\tfrac12,\tfrac12).
\]

这把“预测得好”与“预测得可编码”联结起来：(c) 同时帮助合成和熵模型。后续 hybrid entropy、temporal-context mining、feature modulation、real-time 及 diverse contexts 分别处理并行性、长时上下文、码率控制、延迟及多类条件信息。代价是上下文网络和自回归熵模型可显著拖慢 decode；模型若只给 likelihood 而没有 bit-exact coder，就不能称完整编解码器。

### 2.5 纯特征域和任务导向编码（ELF-VC 等）

ELF-VC 类工作在特征空间预测和编码，再还原像素；另一支是把检测、分割、跟踪、行动识别的中间特征或任务 loss 纳入训练。它适合“机器看视频”，但必须回答：谁定义任务？任务模型换代后码流是否仍有用？在关键少数类、长尾、夜间和攻击下能否安全退化？评测至少要有 pixel RD、task RD 与跨模型 transfer 三张曲线。

### 2.6 Transformer、状态空间与长上下文

注意力可在大运动、重复纹理、长依赖上比局部卷积更灵活；也引入 (O(N^2)) token 成本、缓存、分辨率变化与时延问题。可行路线包括窗口/金字塔注意力、cross-attention 到参考 feature、可学习 memory bank、状态空间模型（线性扫描）及块稀疏化。新颖性不应只是“替换 backbone”：需证明同等参数、训练数据、真实 bpp 和 decode latency 下，长上下文确实降低 entropy 或减少 drift。

### 2.7 可变码率、可伸缩和多目标

常见做法有：多个 (lambda) 模型；将 quality level 注入 FiLM/feature modulation；条件量化步长；渐进 latent bit-plane；base+enhancement layers；ROI/semantic mask 分配。挑战是一个网络在码率两端失真模式不同，熵表必须随条件一致，且“目标 bpp”会因内容波动。需要报告 per-sequence bpp 偏差、切换质量的状态处理、码率切换后 I-frame spike 以及层间可独立解码性。

### 2.8 神经表示/权重压缩（NeRV、HNeRV、HiNeRV）

INR 用坐标（时间、空间）输入网络输出帧或特征；视频被网络权重、量化权重和可选 latent 表示。它避免传统码流结构，可自然支持随机时刻合成和一次一视频的拟合；但训练时间、每视频优化、模型传输开销和 decoder 加速器决定其适用场景。HNeRV/HiNeRV 以混合/分层表示改善速度和保真。评价应另列“encoder 训练 GPU 小时、模型 bytes、首帧可用时间、任意帧访问时间”，不能同 generic codec 的 fps 直接相加。

### 2.9 生成式、语义式与扩散式压缩

极低码率下，发送结构、运动、语义 token 或低频 latent，解码器以 GAN/diffusion/video prior 合成纹理；优点是 LPIPS/FVD/主观感知可能好，风险是 hallucination、身份/文字/医学/取证细节被虚构，且采样耗时及随机性破坏确定性 decode。适合创意预览、远程呈现或明确可容忍的场景；不宜无说明地用于证据、监控、医疗和驾驶。研究必须有真实性约束（文本/OCR、身份、几何、时序一致性）、多随机种子、提示/先验泄漏控制和人类主观实验。

### 2.10 AI 工具化标准编码与神经增强

另一条高转化路径不改码流：用网络做 partition/mode/QP/滤波决策、环路滤波、super-resolution 或 ROI 控制，输出仍为 VVC/AV1。它可用现有硬件和互通生态，但训练标签常来自昂贵 RDO，必须防止将编码器搜索预算转移到离线数据生成而不披露。若 decoder 有神经后处理，则需定义模型版本、随码流传输方式、确定性数值和专利/许可。

---

## 3. 公开生态：论文、代码、数据、标准与竞赛

### 3.1 代表论文/代码台账（不是穷尽声明）

| 家族/条目 | 主贡献 | 可审计入口 | 复现前必须核查 |
|---|---|---|---|
| DVC (Hu et al., 2019) | 压缩 flow 与 residual 的端到端原型 | [P](https://arxiv.org/abs/1812.00101) · [C](https://github.com/ZhihaoHu/PyTorchVideoCompression) | PyTorch/CUDA 版本、光流实现、真实 arithmetic coder |
| HLVC (Yang et al., 2020) | 层级双向视频压缩 | [P](https://arxiv.org/abs/2003.01966) · [C](https://github.com/RenYang-home/HLVC) | 是否对应论文配置、B-frame delay |
| RLVC (Yang et al., 2020) | recurrent 编码和时序状态 | [P](https://arxiv.org/abs/2006.15864) · [C](https://github.com/RenYang-home/RLVC) | 长序列 drift、state reset |
| SSF (Agustsson et al., 2020) | scale-space flow motion 表示 | [P](https://arxiv.org/abs/2003.11912) | 运动码率与 warp 操作成本 |
| FVC (Hu et al., 2021) | feature-space motion/残差压缩 | [P](https://arxiv.org/abs/2105.02649) · [C](https://github.com/BruceChen7/FVC) | feature 对齐与熵编码实现 |
| ELF-VC (Rippel et al., 2021) | 端到端特征空间 video coding | [P](https://arxiv.org/abs/2104.14335) · [C](https://github.com/menushen/ELF-VC) | feature buffer、目标延迟 |
| DCVC (Li et al., 2021) | temporal context 参与熵模型和重建 | [P](https://arxiv.org/abs/2109.15009) · [C](https://github.com/microsoft/DCVC) | release/commit 与 bitstream 实测 |
| DCVC-HEM (2022) | hybrid entropy modeling | [P](https://arxiv.org/abs/2207.08388) · [C](https://github.com/microsoft/DCVC) | 自回归顺序、decode 时间 |
| DCVC-TCM (2022) | temporal context mining | [P](https://arxiv.org/abs/2211.12117) · [C](https://github.com/microsoft/DCVC) | context 消融和长 GOP |
| DCVC-RT (2023) | 面向实时的深度 codec | [P](https://arxiv.org/abs/2305.02797) · [C](https://github.com/microsoft/DCVC) | 分辨率、batch=1、端到端 latency |
| DCVC-FM (2023) | feature modulation 可变码率 | [P](https://arxiv.org/abs/2308.03131) · [C](https://github.com/microsoft/DCVC) | quality 控制误差、切点质量 |
| DCVC-DC (2024) | diverse contexts | [P](https://arxiv.org/abs/2404.19111) · [C](https://github.com/microsoft/DCVC) | 代码发布日期/许可/模型权重 |
| NeRV (2021) | per-video implicit neural representation | [P](https://arxiv.org/abs/2110.13903) · [C](https://github.com/haochen-rye/NeRV) | per-video optimization 成本 |
| HNeRV (2023) | hybrid neural representation | [P](https://arxiv.org/abs/2304.02633) · [C](https://github.com/haochen-rye/HNeRV) | 权重熵编码和任意帧访问 |
| HiNeRV (2023) | 高保真分层神经表示 | [P](https://arxiv.org/abs/2306.09818) · [C](https://github.com/PKU-YuanGroup/HiNeRV) | bitrate 是否含全部 side information |
| DiffVC (2023) | 扩散式低码率视频压缩方向 | [P](https://arxiv.org/abs/2308.08445) | 随机性、采样步数、真实性评测 |

该表和 CSV 是“seed set”。扩展时应纳入 citation graph 的前向/后向引用、IEEE TCSVT/TIP、DCC/PCS/MMSP/ICIP、CVPR/ICCV/ECCV、MPEG/JVET 文档、厂商技术报告与专利，而不是只搜 arXiv 标题。

### 3.2 数据集：训练、测试和它们缺失的内容

| 数据 | 常见用途 | 价值 | 不能替代 |
|---|---|---|---|
| [Vimeo-90K](http://toflow.csail.mit.edu/) | triplet/septuplet 训练 | 场景多、易获得的短片段 | 长时漂移、HDR、专业内容 |
| [REDS](https://seungjunnah.github.io/Datasets/reds.html) | 高分辨率序列训练/恢复 | 运动与退化研究 | 真实压缩源分布 |
| [BVI-DVC](https://www.bristol.ac.uk/engineering/research/vision/datasets/) | video coding 训练/测试 | 压缩取向集合 | 大规模多域数据 |
| [UVG](https://ultravideo.fi/#testsequences) | 4K 客观测试 | 社区常用、便于横比 | 只有 7 条，不足以泛化 |
| HEVC CTC Class B–E | 传统 anchor 对比 | 兼容历史 RD 结果 | 现代竖屏、HDR、屏幕内容 |
| [MCL-JCV](https://www.epfl.ch/labs/mmspg/downloads/mcl-jcv-dataset/) | 主观/客观研究 | 内容与评分研究价值 | 训练规模 |
| UGC/直播/屏幕内容/360°/HDR 自建集 | 域外验证 | 发现 domain shift | 需明确授权、隐私和分发限制 |

建议建立 `manifest.csv`：`sequence_id, source_url, sha256, license, width,height,fps,frames,pixel_format,bit_depth,transfer,primaries,scene_type,split`。所有编码输入先由 FFmpeg 固定成 yuv4mpegpipe 或 raw YUV，并保存命令和哈希；否则不同 RGB↔YUV 矩阵/范围会产生伪 RD 差异。

### 3.3 标准与互操作线索

* [ITU-T H.266/VVC](https://www.itu.int/rec/T-REC-H.266) 是强传统编码锚点；研究报告应给出所用参考实现或生产实现版本。
* [MPEG Neural Network Compression](https://mpeg.chiariglione.org/standards/mpeg-7/neural-network-compression) 关注网络模型压缩，是“传 decoder/INR 权重”的相邻技术，不等于视频码流标准。
* [MPEG standards portal](https://www.mpeg.org/standards/) 是跟踪 AI/神经工具、测试条件和 call for evidence 的入口。每月查一次会议输出，不要凭新闻稿推断标准已完成。
* 标准化研究的关键不是某网络 PSNR，而是 normative bitstream syntax、模型标识/更新、integer determinism、错误恢复、profile/level、专利池和参考软件。

### 3.4 竞赛与基准：把“比赛成绩”放对位置

[CLIC](https://compression.cc/) 曾长期组织学习式图像压缩并在某些年份/赛道涉及视频或相关任务；赛道、指标和许可随年份变动，必须逐届读取 rules、提交格式、test server 和允许训练数据。其他压缩/多媒体挑战常由 CVPR workshop、PCS、MMSP、ICIP 或实验室组织，不能把“存在挑战”误写成固定 AIVC 世界锦标赛。

对任何竞赛条目保存：年份、track、deadline、训练数据限制、是否可用外部预训练、指标（PSNR/MS-SSIM/VMAF/主观）、runtime/hardware 限制、hidden-test 规模、获奖方案报告、代码和模型许可。隐藏测试分数适合比较，不代替跨版本可复现实验。

---

## 4. 评测、复现实验与审计

### 4.1 最低报告卡（每一个模型）

1. **码流：** 每序列 `.bin`、大小、SHA-256、能否独立 decode；字节数包含 motion、hyperprior、header、model id、I-frame、padding 和权重。
2. **协议：** intra period/GOP、open/closed GOP、参考列表、low-delay 或 random-access、scene-cut reset、码率控制和 quality level。
3. **像素：** 4:2:0/4:4:4、bit depth、full/limited range、BT.601/709/2020、HDR transfer；PSNR-Y、PSNR-YUV 的权重和 crop 规则。
4. **质量：** PSNR、MS-SSIM、VMAF（模型版本）和 LPIPS；生成式再报 temporal metrics、OCR/identity/geometry/任务指标与主观测试。
5. **效率：** batch=1 下 encode/decode 的 p50/p95 ms/frame、fps、端到端 latency、峰值 VRAM/RAM、参数、MACs、能耗；GPU/CPU/driver/CUDA/torch 版本。
6. **可靠性：** 码流截断、单包丢失、随机 access、跨 300–1000 帧漂移、重复运行 bit-exact 性。
7. **统计：** 每序列点而非只有平均值、bootstrap 置信区间、内容分层结果和失败示例。

### 4.2 BD-rate 不是魔法

Bjøntegaard delta 在相同质量范围、单调合理曲线、至少四个可靠 rate points 时才可解释。报告原始 `(bpp, metric)`，说明用 log-rate 对 quality 拟合的脚本；遇到曲线不重叠/非单调，报告共同区间或 Pareto 图，不外推一个漂亮百分比。PSNR 与 MS-SSIM 应分开训练、分开比较；VMAF 需说明模型和分辨率。对视频尤需画随时间的质量/码率，平均数会藏起 I-frame spikes 和 drift。

### 4.3 基线矩阵与实验预算

| 层级 | 必做基线 | 用途 |
|---|---|---|
| 图像模块 | CompressAI/同等 learned image codec、同参数量 ablation | 分离 image prior 的贡献 |
| 传统 | x265/HEVC、VTM/x266/VVC、SVT-AV1（固定版本/preset） | 显示真实工程差距 |
| 学习视频 | DVC 类、DCVC 官方稳定版本、同代码库的消融 | 控制训练/代码差异 |
| 任务/感知 | 原图、传统编码、相同 bpp 的像素 codec | 防止只在有利指标胜出 |
| 鲁棒/部署 | packet-loss、长 GOP、CPU/mobile/no-GPU decoder | 排除实验室特例 |

建议三阶段：`P0`（48–72 GPU-hours，小数据/4 quality 验证码流和消融）；`P1`（200–500 GPU-hours，完整训练与 UVG/HEVC）；`P2`（500+ GPU-hours，跨域、长序列、用户/下游、性能优化）。每阶段设 kill criterion：若在 P0 没有可解释的 per-frame entropy 下降或不满足目标时延，停止扩大模型。

### 4.4 可复现实验骨架

```text
configs/          # yaml: data hash, GOP, lambda, seed, model commit
src/codec/        # encode/decode 返回真实 bytes；禁止只返回 likelihood
scripts/          # download/prepare/encode/decode/metrics/bd_rate
artifacts/        # manifest, bitstreams, decoded frames, env lock
reports/          # per-sequence CSV, plots, failure gallery
```

单元测试至少包括 `(decode(encode(x)) shape/dtype 正确)`、range coder round-trip、CPU/GPU 一致容差、帧序状态 reset、损坏 header 拒绝、同版本生成 bitstream hash 相同。CI 可跑 16 帧 tiny clip；完整 RD 用带版本的实验任务，而非 CI 偶然环境。

---

## 5. 未解问题与可落地的研究 ideas

### 5.1 空白的优先级评分

以下按 **影响(1–5) × 可证伪性(1–5) × 与既有主线差异(1–5) ÷ 资源风险(1–5)** 粗排；高分不表示保证发表，表示值得先做 P0。

| ID | 问题 | 分数 | 为什么现在值得做 |
|---|---|---:|---|
| I1 | 不确定性驱动的鲁棒/可恢复 neural bitstream | 20.0 | 私有码流在丢包和状态漂移上仍缺统一答案 |
| I2 | 有真实性约束的生成式低码率视频编码 | 18.8 | 生成质量进展快，但关键事实保真评测很弱 |
| I3 | 硬件感知的并行熵模型与 decoder co-design | 18.0 | entropy decode 常是实际瓶颈而非 MAC |
| I4 | 跨域/HDR/屏幕内容的 codec calibration | 16.0 | 小型旧测试集不能代表现实输入 |
| I5 | 可迁移任务导向编码与隐私约束 | 15.0 | 机器视觉部署需要比 mAP 曲线更多保证 |
| I6 | 低成本长上下文与错误漂移理论/诊断 | 14.4 | 长视频、实时、memory 三者尚未统一 |

### I1：R3VC — 可恢复、风险受控的上下文神经视频编码

**假设。** 若 encoder 估计每个时空 latent 对未来预测的敏感度，并把少量高敏感信息放入独立可验证的 base layer，同时对 enhancement context 做可丢弃编码，则在相同平均码率和 0–5% Gilbert–Elliott 丢包下，可显著降低 outage duration 和错误扩散，而 clean-channel RD 损失有限。

**方法。** (a) 从当前/历史 feature 用 ensemble variance、entropy residual 与 Jacobian 近似得到风险图 (u_t)；(b) `base` 传低频 motion/anchor/context checkpoint，`enhancement` 传其余 latent；(c) latent 分片包含 frame id、layer id、依赖 id、CRC；(d) 缺片时用确定性 concealment 和最近 checkpoint reset，而非静默错误传播；(e) 损失为 RD + 预测的未来 K 帧敏感度 + 模拟网络损失的 quality/outage 项。

\[
L=R_b+R_e+\lambda D+\eta\sum_{k=0}^{K}w_kD(x_{t+k},\hat x_{t+k}^{loss})+\rho R_{base}.
\]

**最小实验。** 以固定 DCVC-style P codec 为骨架；UVG/HEVC + 一个 UGC holdout；4 rate points、GOP 32/96；i.i.d. loss 与 burst loss；对比无保护 codec、均匀 FEC、周期性 I-frame、同 byte budget 的 extra reference。指标：clean BD-rate、loss BD-rate、95% temporal PSNR/VMAF、恢复到原质量阈值的帧数、base overhead、p95 latency。

**反证与风险。** 若风险图仅是纹理复杂度代理，均匀 FEC 同样有效；若 base layer 过大，RD 优势消失。消融 `no-risk / no-checkpoint / no-layering / oracle-risk`，并在 clean 情况报告完整代价。发表价值来自**显式 bitstream 依赖图+公共 loss simulator+恢复指标**，而不是又一个 concealment 网络。

### I2：TruthVC — 具可测事实保真的生成式压缩

**假设。** 将生成 decoder 限制为“以传输的结构 token 为条件的细节生成”，并对文字、脸部/身份、关键点、深度/光流施加事实锚定，可在极低码率改善主观感知而不比像素 codec 更频繁地篡改关键内容。

**方法。** Encoder 传 (z_s)（低频结构/边缘/深度/运动）、(z_c)（内容锚点，如 OCR 字符区域）和可选 (z_d)（纹理）；latent diffusion/flow decoder 接受 deterministic seed。训练用 RD、LPIPS、temporal warp、OCR consistency、face embedding/landmark、segmentation boundary 和“生成差异 mask”约束。输出同时带 `fidelity profile`，禁止将该模式用于不允许生成修复的使用场景。

**评测。** 分开收集字幕/路牌、脸部、手势、快运动、低光五类测试集（授权与隐私处理）；与 VVC、像素 neural codec、无锚生成 codec 同码率比较。报告 PSNR/LPIPS/VMAF、tLPIPS/warp error、CER/WER、identity verification、keypoint error、双盲 MOS 和 hallucination rate（预注册错误定义）。每片段至少 3 seeds，码率含 seed/side data。

**失败边界。** 生成模型的训练数据可能记忆敏感样本或对罕见文字/人群失效；即便感知指标胜出，也不可称“无失真”。若事实指标无改进，结论应是生成式仅适合娱乐性视觉质量，而不是压缩通用方案。

### I3：ParaEntropy-VC — 并行、硬件友好的熵解码协同设计

**假设。** 将 latent 划成少量 checkerboard/group slices，使用已经解码的时间 context 与 group-level hyperprior，联合训练一个受限依赖图，可获得接近自回归 RD 而把符号 decode critical path 从 (O(HW)) 降到固定 (G) 轮。

**方法。** 明确 DAG：每一 group 的条件只能读先前 groups 和 causal reference features；每组在 GPU 并行预测 CDF、CPU/ASIC 并行 rANS lanes 解码。训练中把 CDF quantization、integer CDF 精度、buffer copy 和 group barrier 成本纳入 proxy；导出固定点 CDF 并 bit-exact 验证。与 pure factorized、checkerboard、fully autoregressive 和现有 temporal-context entropy model 比较。

**指标。** 除 RD 外，`symbols/s`、p50/p95 decoder ms、CPU 单线程/多线程、GPU、GPU↔CPU transfer、峰值 memory、功耗/帧和实际码流。禁止以“网络前向时间”替代熵编码时间。若 G=4 在 RD 仅小损失而 decode 明显加速，是清晰贡献；若只是换更大网络换 RD，则淘汰。

### I4：DomainCal-VC — 面向 HDR、屏幕和 UGC 的可验证域适配

**问题。** 训练于 SDR RGB 短片段、测于 UVG 的模型往往在 PQ/HLG、10-bit、动画、屏幕文字、竖屏 UGC 上色彩漂移或码率失控。当前论文通常把它当附录，而这是部署障碍。

**设计。** 建立有许可的分层 benchmark：内容域×动态范围×分辨率×运动×文本；训练 shared codec + 小型 domain adapter/entropy calibration，adapter id 显式进入码流 header。优化 (R+lambda D_{linear-light}+kappa D_{color}+\nu D_{temporal})。控制 adapter bytes、无标签测试、未知域 fallback 和 metadata 缺失。报告 domain-wise BD-rate、色域裁剪率、亮度误差、OCR、主观 HDR 观看协议和 calibration latency。

### I5：TaskSafe-VC — 任务可迁移、隐私可控的编码

不要只优化单一冻结 detector。令 (f_j) 是若干不同架构/版本/任务的 black-box 或可微代理，优化最坏/分位数任务损失，并发送最小像素残差以支持人类审计：

\[
L=R+\lambda D_{human}+\tau\operatorname{CVaR}_{j,domain}L_{task}(f_j(\hat x),f_j(x)) + \xi L_{privacy}.
\]

评估 train/test task model 不重合、类别长尾、域迁移、攻击/遮挡；privacy 用成员推断、属性泄露和可视化重建攻击，而非一句“latent 天然隐私”。明确不可接受的应用、审计码流保留策略和人类复核接口。

### I6：DriftBench — 将长时错误传播变成共同基准

**贡献可小但很基础：** 发布固定 decoder checkpoint、1000+ 帧内容、scene-cut/occlusion/camera motion 标签、每帧 bits/PSNR/VMAF、reference-age、motion/residual entropy、随机 access 与扰动脚本。定义 `drift area`（相对 intra/reset oracle 的累计质量损失）、`recovery frames` 和 `tail quality`。这会让“长上下文有效”变成可证伪命题，也可服务 I1/I3/I4。数据许可、raw hash 和运行容器是研究本体的一部分。

---

## 6. 推荐的首个项目：I1 + DriftBench（12 周）

| 周 | 交付物 | 决策门 |
|---:|---|---|
| 1 | 文献/代码 commit 台账，数据 manifest，传统+DCVC-like baseline 跑通 | 能真实 encode/decode，非 likelihood-only |
| 2 | 统一 YUV pipeline、4 rate RD、每帧 telemetry | 结果与官方量级一致才继续 |
| 3 | loss simulator、packet 格式、CRC、drift dashboard | 单包丢失可稳定复现 |
| 4–5 | risk estimator 与 oracle-risk 上界 | oracle 也无收益则改题 |
| 6–7 | base/enhancement 与 checkpoint ablation | 比均匀保护更好才扩大训练 |
| 8 | burst/随机 loss、长 GOP、跨域集 | 不只在单一 loss 模型成功 |
| 9 | 速度/内存/实际 byte 审计 | 保护开销可解释 |
| 10 | 失败案例、统计 CI、复现容器 | 没有负例不投稿 |
| 11 | 撰写：问题、协议、结果、局限 | 所有图可由脚本重建 |
| 12 | 开源最小码流/decoder/benchmark | 许可和安全说明齐全 |

建议的目录/API：

```python
packet = codec.encode(frame, refs, quality=q)
# packet = {header, base_bytes, enhancement_bytes, crc, dependencies}
recon, state = codec.decode(packet_or_missing, refs, state)
metrics.log(frame_id=t, bits=len(packet)*8, risk=risk, recovered=state.recovered)
```

这里 `packet_or_missing` 是一等输入；若系统 API 根本不能表达丢片，便无法研究网络鲁棒性。

---

## 7. 常见失败模式（审稿/产品前检查）

* 用估计 entropy 代替真实比特、漏计 hyperprior/header/I-frame/模型权重。
* RGB 训练结果与 YUV 4:2:0 anchor 直接 BD-rate；或未给 conversion 命令。
* 只测 7 条 UVG 短视频；不测长序列、scene-cut、UGC/HDR/屏幕文本。
* 报 GPU batch throughput 而不报 batch=1、熵解码、传输、p95 和能耗。
* 只测 clean channel；loss 后 decoder silently propagates 伪影。
* 生成式模型只报 LPIPS，不报 OCR/身份/时序与误造案例。
* 用更强预训练、更多数据或更慢 preset 却把收益归因于一个新模块。
* 引用 GitHub 默认分支而没有 commit、license、权重 checksum、环境锁定。
* 将没有公开规则的挑战或没有可下载的代码写成“已验证 SOTA”。

---

## 8. 执行清单与维护模板

每月维护动作：

1. 从四类关键词和 citation graph 导入新条目，保存原始查询快照。
2. 对每篇论文填 `scope, causality, color, dataset, anchor, metric, actual_bitstream, runtime, code_commit, license, reproduction_status`。
3. 对每个 repository 做许可证、release、issues、checkpoint、码流和解码测试的审计；不要以 stars 排序。
4. 对每个 benchmark 记录版权、下载 hash、预处理、split，删除不能再合法分发的副本。
5. 更新标准/竞赛时只引用当届规则 PDF/官方网页，标明访问日期。
6. 每个新 idea 先写假设、最强反例、最小消融、kill criterion 和资源上限；没有这些就只是主题。

### 参考入口

* DVC: Hu et al., *Deep Video Compression*, CVPR 2019，[论文](https://arxiv.org/abs/1812.00101)。
* DCVC 系列：[Microsoft 官方仓库](https://github.com/microsoft/DCVC)（其中 README/release 才是具体实现版本的证据）。
* NeRV: Chen et al., *NeRV: Neural Representations for Videos*, NeurIPS 2021，[论文](https://arxiv.org/abs/2110.13903)。
* VVC: [ITU-T Recommendation H.266](https://www.itu.int/rec/T-REC-H.266)。
* CLIC: [官方主页与历届入口](https://compression.cc/)。

本报告有意将“有据可核的 2024-06 地图”和“2026 以后待联网更新的缺口”分开。科研尽调的质量不取决于把不确定条目塞进表格，而取决于任何读者能否沿着链接、commit、码流和脚本重建或推翻每一个结论。
