# Stage 4 → 提交：剩余工作时间表

**截止日期：2026-06-24 23:59**
**当前日期：2026-06-05**
**剩余天数：19 天**

---

## 总览

| 阶段 | 内容 | 预计耗时 | 状态 |
|------|------|---------|------|
| R-1 基线训练 | no-bottleneck BaselinePlain, 20 ep | ~11 h | 🔄 进行中 |
| R-1 审计 | OOD eval + B1/B2 + post-hoc attribution | 2–3 h | ⬜ 待开始 |
| Stage 4 手稿修订 | 应用 R-3~R-8 全部修订 | 1 天 (6–8 h) | ⬜ 待开始 |
| Stage 3' 复审 | 5-reviewer 修订验证 | 2–3 h | ⬜ 待开始 |
| Stage 4' 二次修订 | 如复审仍有 Major (可选) | 0–1 天 | ⬜ 视复审而定 |
| Stage 4.5 终审 | 100% 引用/数据/声明验证 | 2–3 h | ⬜ 待开始 |
| Stage 5 定稿 | LaTeX → PDF, 格式确认 | 1–2 h | ⬜ 待开始 |
| 人工终审 | 作者栏 + 通读 + 最终确认 | 1–2 h | ⬜ 待开始 |

**总计：约 4–5 个工作日**（含 buffer），充裕地在截止日期前完成。

---

## 逐日时间表

### Day 1 — 2026-06-05（周四）· 今天

| 时段 | 任务 | 产出 |
|------|------|------|
| 全天（后台） | R-1 baseline 训练运行中 | ⏳ ETA ~11h |
| 晚间 | 训练完成后：运行 OOD eval + 审计流水线 | `Results/baseline_analysis/` |

**依赖：** laptop 保持唤醒 + 接通电源。

---

### Day 2 — 2026-06-06（周五）

| 时段 | 任务 | 产出 |
|------|------|------|
| 上午 | 完成 R-1 审计分析：baseline vs CBNet 对比 | `04_Stage4_Evidence.md` 更新 |
| 下午 | 开始 Stage 4 手稿修订：R-3 贡献锐化 + R-5 reliance vs faithfulness | `draft/` 各章更新 |

**关键产出：**
- R-1 结论：bottleneck 带来了什么（vs post-hoc attribution）
- R-3：Outcome C + "intervention > weights" 贡献表述锐化
- R-5：functional reliance ≠ semantic faithfulness 明确声明

---

### Day 3 — 2026-06-06~07（周五~周六）

| 时段 | 任务 | 产出 |
|------|------|------|
| Day 3 全天 | Stage 4 手稿修订剩余项：R-6 trace-vs-history / R-7 scope hedge / R-8 ECE+措辞修正 + 多 seed 证据植入 | `draft/` 全部更新 |

**修订清单（R-3 ~ R-8）：**

| # | 修订项 | 性质 | 涉及章节 |
|---|--------|------|---------|
| R-3 | 贡献锐化：foreground Outcome C + "intervention > weights" | 措辞 | Intro, Abstract, Discussion |
| R-4 | 一个 P4 baseline（可选，如 NPR 同协议重训） | 实验（可跳过） | Experiments Table 1 |
| R-5 | "functional reliance ≠ semantic faithfulness" 明确声明 | 措辞 | Discussion, Limitations |
| R-6 | trace-vs-history 作为核心开放问题 | 措辞 | Discussion, Limitations |
| R-7 | 范围围栏（compression pair 而非 jpeg_quant 单概念） | 措辞 | 全文 |
| R-8 | ECE split 说明 / res128 措辞 / resolution confound | 修正 | Experiments, Limitations |

**多 seed 修订要点：**
- 全文 `jpeg_quant` 单概念表述 → 改为 **compression pair `{jpeg_quant, freq_radial}`**
- E-Delta Outcome C 标注为 **seed-42 单 seed**，交叉引用多 seed caveat
- 新增 Table：3-seed concept attribution（seed-42 / seed-1 / seed-2）

---

### Day 4 — 2026-06-08（周日）

| 时段 | 任务 | 产出 |
|------|------|------|
| 上午 | `_build_body.py` 重新生成 body.tex → `latexmk -pdf` 重编译 | `manuscript.pdf` 更新版 |
| 下午 | 自查：确保无未定义引用 / 数字一致 / 结构完整 | 编译通过报告 |

**退出标准：**
- `latexmk` exit 0
- 0 undefined citations
- 所有新数字来自 `04_Stage4_Evidence.md` 磁盘工件
- 无新增 prohibited claims

---

### Day 5 — 2026-06-09（周一）

| 时段 | 任务 | 产出 |
|------|------|------|
| 全天 | **Stage 3' 复审**：5-reviewer 验证修订响应 | `Stage3_Review/Stage3_ReReview_Report.md` |

**复审重点：**
- R-1（baseline）是否充分回答了 M2？
- R-2（multi-seed）是否充分回答了 M1？
- R-3~R-8 措辞修订是否到位？
- 是否出现新问题？

**退出标准：**
- 决策 = Accept 或 Minor Revision
- 如 Major → 进入 Day 6 Stage 4'

---

### Day 6 — 2026-06-10（周二）· 可选

| 时段 | 任务 | 产出 |
|------|------|------|
| 全天 | **Stage 4' 二次修订**（仅当 Stage 3' 为 Major）| 修订后手稿 |

**注意：** pipeline 限制 Stage 4' 最多 1 轮。如仍有问题 → 标记为 Acknowledged Limitations，直接推进 Stage 4.5。

---

### Day 7 — 2026-06-11（周三）

| 时段 | 任务 | 产出 |
|------|------|------|
| 上午 | **Stage 4.5 终审**：100% 引用/数据/声明验证 + 7-mode failure checklist | `05_Stage4.5_Integrity_Report.md` |
| 下午 | 修复终审发现的问题（如有） | 修订后手稿 |

**⚠ IRON RULE：Stage 4.5 必须 PASS（0 issues）才能进入 Stage 5。**

---

### Day 8 — 2026-06-12（周四）

| 时段 | 任务 | 产出 |
|------|------|------|
| 上午 | **Stage 5 定稿**：最终 PDF 编译 + 格式确认 | `manuscript.pdf` 最终版 |
| 下午 | 人工终审：填写作者栏 + 通读全文 + 最终确认 | **提交就绪** |

---

### Buffer — 2026-06-13 ~ 2026-06-24（12 天）

- 如需额外修订 / 格式调整 / 通读
- 课程提交系统操作
- 意外情况处理

---

## 关键风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Baseline 训练崩溃（laptop 睡眠/断电） | 中 | +11h 延迟 | 保持唤醒；Day 1 结束前检查 |
| Stage 3' 复审仍为 Major | 低 | +1 天 | Stage 4' 仅修订措辞，不再跑实验 |
| Stage 4.5 终审发现新问题 | 低 | +半天 | 最多 3 轮修复 |
| 数字不一致需要重跑审计 | 极低 | +2–3h | 所有审计走同一条代码路径 |

---

## 实验依赖图

```
seed-42 ✅ ─┐
seed-1  ✅ ─┼─→ 3-seed synthesis ✅ ─→ R-2 解决 ✓
seed-2  ✅ ─┘
                                       ┌─→ Stage 4 手稿修订 (R-3~R-8)
R-1 baseline 🔄 ─→ baseline audit ────┘
                  (B1/B2 + attribution)
```

---

## 每日检查点

- **Day 1 结束：** baseline 训练是否完成？
- **Day 3 结束：** 所有 R-3~R-8 修订是否落地？
- **Day 4 结束：** PDF 是否编译通过？数字是否一致？
- **Day 5 结束：** Stage 3' 决策是什么？
- **Day 7 结束：** Stage 4.5 是否 PASS？
- **Day 8 结束：** **提交就绪？**

---

*最后更新：2026-06-05 · 基于 pipeline state + Stage 3 review roadmap*
