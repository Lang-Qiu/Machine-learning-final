# ⚠️ READ-FIRST — AGID 项目流程保护

> 本文件由用户授权写入，作为跨会话的 pipeline 状态锁。任何 Codex 会话在对本仓库做出任何动作之前**必须**先阅读本文件 + `AGID_Project/docs/experiment_handoff.md`。

**最近一次更新**：2026-06-02 — **Stage 2 全部章节 plan 已锁定（RW/Method/Experiments/Discussion/Limitations/Conclusion/Abstract），进入逐章 drafting（从 Introduction 重建）；本会话新增 ForenSynths 实验(Route B 73.65%) → 定调 competitive(非SOTA) detector + audit instrument**。背景：Gate #5 于 2026-06-01 解锁进入 Stage 2，Socratic 规划中因 Grommelt scoop 暂停并插入 E-Delta 重训 excursion（见下方 2026-06-01 块）。E-Delta（Q96-debiased，25k/类 × 4 生成器，10 ep，seed 42）结论 = **`jpeg_quant` 概念在 Q96 去偏后依然主导**：linear-head 权重 −13.53 → −9.11（仅缩 1.49×，仍 rank #1），**debiased 模型上 zero-out 仍 −48.18pp**；与 Route B 形成干净 **double dissociation**（Route B 赢 raw-OOD 99.7%，E-Delta 赢 Q96-val 99.78%）。smoke 阶段的"reliance 迁移"叙事已证实为训练集 overfit 假象（已弃用）。论文 novelty = **concept bottleneck as audit instrument**（不宣称 6 个可解释概念；详见 memory `project-agid-audit-pivot` / `project-agid-grommelt-scoop`）。**证据 spine 已锁定**于 `AGID_Project/Stage2_Writing/00_Evidence_Spine_Lock.md`（+ `01_Decision_Log.md` / `02_Chapter_Plan.md`）。Stage 2 入口命令 = `/ars-plan`（= academic-paper skill 的 plan/Socratic 模式，ARS v3.9.3）。

> **⚠️ 2026-06-01 后续（Stage 2 规划中发现，必读）**：lit search 发现 **Grommelt et al. 2024「Fake or JPEG?」(arXiv 2403.17608)** 已先发表"GenImage 有 JPEG/尺寸 confound"这一 audit 核心发现 → 原 audit-discovery novelty **被 scoop**。用户改选 **RE-POSITION 3 = delta-over-Grommelt**：在 Grommelt-style debiased GenImage 上重训 CBNet，报告 jpeg_quant 被数据层中和后**概念依赖如何迁移**。**【2026-06-02 更新】E-Delta 实验 excursion 已完成 → Outcome C（`jpeg_quant` 在 Q96 去偏后持续主导，非迁移）；Stage 2 写作已恢复，从 Related Work 章继续。RE-POSITION 1 零算力 fallback 已不再需要。** 详见 memory `project-agid-grommelt-scoop` + `AGID_Project/Stage2_Writing/00_Evidence_Spine_Lock.md`。Grommelt 必须在 Intro + Related Work 显著引用（定位 = "we extend"，非 "we discover"）。

---

## Pipeline 状态锁（Authoritative）

```
Skill           : academic-research-skills v3.9.3 → `/ars-plan` (= academic-paper skill, plan mode)
Position        : Stage 4 REVISE COMPLETE — R-2 multi-seed (3 seeds) + R-1 baseline + B1 confound 全部完成 ✓ · manuscript revised + recompiled (15pp, D-41) · next = Stage 3' re-review to verify M1/M2/M3 resolved · thesis = competitive(非SOTA) detector + audit instrument
Stage 2 mode    : REVISE (Stage 4 experiments + revision COMPLETE 2026-06-05; entering Stage 3' re-review) · very-high oversight
Evidence policy : Route A + Route B + Batch 1-4 全部完成 ✓
Stage 2 entry gate : Route B 多生成器实验完成 ✓ + 用户解锁 Gate #5 ✓ (2026-06-01)
Paper deadline  : 2026-06-24 23:59
Word target     : 30,000 (compact tier)
Novelty target  : A+C grade (4-5/5)
```

进入 Stage 2 的前置条件（**已于 2026-06-01 全部满足并解锁 Gate #5**）：
1. ✅ `AGID_Project/Data/GenImage/BigGAN/train/{ai,nature}/` 存在且每类 ≥ 25k 张（已通过：各 162k）
2. ✅ `AGID_Project/Data/GenImage/ADM/train/{ai,nature}/` 存在且每类 ≥ 25k 张（已通过：ai 162k / nature 157k）
3. ✅ `AGID_Project/Data/GenImage/Midjourney/train/{ai,nature}/` 存在且每类 ≥ 25k 张（已通过：ai 162k / nature 162k）
4. ✅ Route B 训练完成，`Code/Results/cbnet_multigen_cbnet_multigen_main_25k_s42.json` 已存在（2026-05-27）
   - SD-1.4 val: acc=99.88% AUC=1.000；BigGAN held-out: 99.90%；ADM held-out: 99.95%；MJ held-out: 99.85%
   - held-out eval: `Code/Results/cbnet_multigen_heldout_eval.json`
   - OOD eval（3 个未见生成器）: `Code/Results/ood_eval_full.json`（2026-05-27）
     GLIDE: 99.80% / Wukong: 99.45% / VQDM: 99.75% → mean acc=99.67% AUC=1.000
5. ✅ 用户显式确认进入 Stage 2（2026-06-01：用户「解锁 Gate #5」）

**Stage 1.5 — Pre-Stage 2 实验补强（2026-05-27 / 28）**：
- ✅ Batch 1 — 推理 dump + 概念统计 / 校准 / Route A vs B 对比表（`Code/Results/batch1_analysis/`）
- ✅ Batch 2 — 混淆量化：B1 (PNG/JPEG)、B2 (res128)、B5 (independent_real) → `Code/Results/confound/` + `batch1_analysis/B_*`
- ✅ Batch 3 — 概念干预：A3 zero-out / cumulative ablation / keep-only-one、A4 counterfactual swap → `batch1_analysis/A3_*, A4_*, A_intervention_summary.md`

**关键证据**（驱动 Stage 2 写作策略）：
- `jpeg_quant` 单概念支撑 ~50pp 的鉴别力（A3 zero-out）
- `freq_radial` 与 `jpeg_quant` 相关 -0.80，编码同一压缩信号（A5）
- 其余 4 个概念干预后 prediction flip <2%（A4），被线性头忽略
- JPEG-q95 统一重编码 → mean Δacc -10.16pp（B1），证实 PNG/JPEG 容器泄漏
- res128 高频破坏 → real_acc 99.7% → 8.1%（B2），分辨率混淆灾难性
- OOD 真图池非伪影（B5：disjoint sampling +0.12pp 不变）

**Route B 启动前已修复的方法论缺陷**（2026-05-26）：
- Fix #1：`train.py --disable_destructive_aug` flag 关闭 JPEG/Blur，避免 30% 概念标签噪声
- Fix #2：`scripts/build_shared_concept_norm.py` 提供跨生成器共享 P2/P98 归一化
- Fix #3：`precompute_concept_labels.py` 加 `--center_crop` / `--color_prior_in/out` / `--extra_real_dirs`；修复 `--norm_stats_path` 路径下 color prior 不被构建的潜伏 bug

**Route B 已知的数据集级风险**（不可在代码层修复，论文 Limitation 必写）：
- 4 路生成器原生分辨率跨度 **8 倍**：BigGAN 128² / ADM 256² / SD-1.4 512² / Midjourney 1024²。center_crop 256 后，BigGAN 上采样 2.25×、ADM 不变、SD 下采样 0.5×、MJ 下采样 0.25×（丢 75% 高频）。`freq_radial` / `bitplane_lsb` / `hf_noise` 三个高频概念在跨生成器对比时系统性退化
- PNG (ai) / JPEG (nature) 的标签泄漏在 GenImage 全数据集存在，shortcut 不可去除
- Diffusion:GAN 训练样本比 = 3:1（SD/ADM/MJ vs BigGAN），分类器可能偏 diffusion 模式

如果 deadline 前 14 天（即 2026-06-10）Route B 仍未完成 → 触发 **Fallback**：用 Route A 单证据进入 Stage 2，把 Route B 改写为 future work。Fallback 切换需用户显式批准。

---

## DO NOT 清单（绝对禁止）

1. ❌ 不要继续在 SD-1.4 单生成器数据上训练（Route A 已封盘，再训没有信息增量）
2. ❌ 不要修改模型架构：`Code/cbnet_agid/models/cbnet.py`、`models/backbone.py`、`concepts/base.py`
3. ❌ 不要删除/覆盖 `Code/Logs/`、`Results/`、`Checkpoints/` 下任何文件
4. ❌ 不要在 BigGAN + ADM 数据落盘验证前启动 Route B 训练
5. ❌ 不要修改概念集合 K=6（除非用户解锁）
6. ❌ 不要在 `--max_train_samples` 配合 `shuffle=True` 时跳过 concept label 对齐校验
7. ❌ 不要绕过 academic-pipeline 的 MANDATORY checkpoint（Stage 2.5 / 4.5 完整性检查不可跳过）
8. ❌ 不要自动启动 `/experiment-agent` 或 `/academic-pipeline` 的实质性 stage——必须用户显式批准

---

## 必读文档（按顺序）

任何一个新 Codex 会话开始工作前，按这个顺序读：

| # | 文件 | 用途 |
|---|------|------|
| 1 | `AGID_Project/docs/experiment_handoff.md` | Route A 全部结果 + Route B 计划 + Gate check 协议 |
| 2 | `AGID_Project/Stage1_Research/07_Experiment_Plan.md` | 15 行实验矩阵 + 4 个假设 + 决策规则 |
| 3 | `AGID_Project/Stage1_Research/01_RQ_Brief.md` | 研究问题锁定 |
| 4 | `AGID_Project/Stage1_Research/02_Methodology_Blueprint.md` | 方法论蓝图 |
| 5 | `C:\Users\LQiu\.Codex\projects\E--LQiu-lab-folder-Machine-learning\memory\MEMORY.md` | 项目 memory 索引 |

---

## 环境提示

- Conda env：`E:\LQiu\conda_envs\agid`（PyTorch 2.1.2+cu121，numpy 1.26.4）
- 激活脚本：`AGID_Project\activate_agid.ps1`
- Shell：PowerShell（用 `$null`、`$env:VAR`、反引号续行；不可用 `&&` / `2>/dev/null`）
- 长跑 numpy 任务前必须设：`$env:PYTHONNOUSERSITE=1; $env:PYTHONMALLOC='malloc'`
- GPU：RTX 4060 Laptop 8GB（不许超 batch 32 + bf16 AMP）

---

## 困惑时的恢复路径

如果你（未来的 Codex）读到这里仍不确定下一步：

1. 不要写代码、不要训练、不要修改任何文件
2. 跑 handoff 文档 §6 的 **Mandatory Gate**（数据/Results/Logs 清单核对）
3. 把 gate 结果呈给用户
4. 等用户决定是 (a) 继续 Route B、(b) 触发 Fallback 进入 Stage 2、还是 (c) 其他动作

---

*本文件由 academic-pipeline orchestrator 在 Stage 1.5 → Stage 2 转场时写入。除非用户授权，不要修改本文件的 Pipeline 状态锁部分。*
