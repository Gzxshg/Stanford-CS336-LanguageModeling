# CS336 后续作业（3/4/5）学习计划

> 前提：每天投入 1-2 小时，本机有 GPU。总计约 6 周（37 个工作日）。
> 创建于 2026-08-28，按实际进度灵活调整。

## 概览

| 作业 | 主题 | 预计天数 | 关键点 |
|---|---|---|---|
| Assignment 3 | Scaling Laws（缩放定律） | ~5 天 | 研究型任务，无代码骨架；50 分排行榜需斯坦福 VPN+学号，非在校生做本地部分 |
| Assignment 4 | Data（数据过滤流水线） | ~14 天 | 从零实现 ~11 个函数，测试可完全离线 |
| Assignment 5 | Alignment（GRPO 推理 RL） | ~18 天 | 最难：~12 个 adapter 函数 + 推导题 + 大量 GPU 实验 |

---

## 第 1 周：Assignment 3 — Scaling Laws（Day 1-5）

- [ ] **Day 1**：通读 `assignment3-scaling/cs336_assignment3_scaling.pdf`，搞懂 IsoFLOPs 方法和幂律拟合流程（C = 6ND）
- [ ] **Day 2-3**：写 `chinchilla_isoflops` 脚本：用 `data/isoflops_curves.json` 拟合各 IsoFLOPs 曲线最低点，产出两张图
- [ ] **Day 4**：拟合幂律 N_opt ∝ C^a、D_opt ∝ C^b，外推到 10²³/10²⁴ FLOPs，预测最优模型/数据规模
- [ ] **Day 5**：缓冲 + 写分析。非在校生到此收尾；有余力可翻 `cs336_scaling/`（服务端完整实现）了解实验调度设计

> 注：若有斯坦福校园网 VPN + 学号，可追加 4-6 天做 50 分排行榜实验（设计实验矩阵 → 提交 API → 拟合 scaling law → final_submission + writeup）。

## 第 2-4 周：Assignment 4 — Data（Day 6-19）

- [ ] **Day 6**：跑 `uv run scripts/download_data.py --offline-only` 下载数据，肉眼检查 WARC/WET 样例
- [ ] **Day 7-8**：`extract_text`：HTML→纯文本（Resiliparse + 编码检测）+ 对应书面小题
- [ ] **Day 9**：`identify_language`（fastText lid.176）+ PII 掩码（`mask_emails/phones/ips`，正则）
- [ ] **Day 10**：`classify_nsfw` / `classify_toxic_speech`（Dolma fastText 分类器）+ `gopher_quality_filter` 启发式规则
- [ ] **Day 11-13**：`classify_quality`（最重编程题）：Wikipedia 外链正例 + 随机 CC 负例，训练 fastText 分类器，调阈值，报告错误率
- [ ] **Day 14-15**：`exact_line_deduplication`（两遍扫描+哈希）+ `minhash_deduplication`（MinHash+LSH，算法细节多，易错）
- [ ] **Day 16-17**：组装并行过滤流水线，跑 2500 个 WET 文件，统计各过滤器保留比例
- [ ] **Day 18-19**：tokenize + 本地 GPU 训练（官方为 8×B200，单机缩小步数/数据量，曲线趋势对即可）+ writeup 收尾

## 第 4-6 周：Assignment 5 — Alignment（Day 20-37）

- [ ] **Day 20**：下载 OLMo-2-0425-1B（HF），跑 vLLM prompting 基线（zero-shot / few-shot / CoT）
- [ ] **Day 21-22**：`tokenize_prompt_and_output` + `get_response_log_probs`（含 entropy）
- [ ] **Day 23-24**：`compute_rollout_rewards` + `compute_group_normalized_rewards`（3 种 normalizer）
- [ ] **Day 25-27**：`compute_policy_gradient_loss`（none/noclip/grpo/gspo）+ `aggregate_loss_across_microbatch` + `grpo_train_step`，跑通全部测试
- [ ] **Day 28-29**：书面推导题：baseline 方差、长度归一化、RFT vs Dr.GRPO、难度重加权、重要性加权偏差-方差
- [ ] **Day 30-32**：标准 GRPO 训练（目标 ≥25% val 准确率）+ LR sweep；实验挂着跑，人做别的
- [ ] **Day 33-34**：on-policy 变体实验（Dr. GRPO / RFT / MaxRL）+ 画图分析
- [ ] **Day 35-36**：off-policy 实现（PPO token 级 clip / GSPO 序列级 clip）+ 实验
- [ ] **Day 37**：自研 policy gradient 变体 + writeup 收尾
- [ ] *（可选）补充包*：Llama-3.1-8B SFT + DPO + 安全评测，约 +5-6 天

---

## 注意事项

- **GPU 墙钟时间是最大变量**：A5 官方按 2×B200 估 24+ 小时实验。单卡建议 seeds 4→2、off-policy 实验规模减半，先跑通流程再补实验。
- **A4 数据未下载**：第一步必须先跑 `download_data.py --offline-only`（约几百 MB，存 `local-shared-data/`）。
- **A4 的 `cs336_basics/` 是官方训练栈，禁止修改**；实现代码放 `cs336_data/`，适配层在 `tests/adapters.py`。
- **A3/A4/A5 的 AGENTS.md 约定**：AI 助手默认只做教学引导、不代写代码；如需直接协助实现，明确告知即可。
- 三个作业之间无代码依赖，A3 卡住可直接跳到 A4。
