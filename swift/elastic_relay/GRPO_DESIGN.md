# GRPO 弹性接力训练设计文档（草案）

> 状态：设计阶段（未实现）
> 分支：`feat/grpo-elastic`
> 关联：SFT 版弹性接力见 `swift/elastic_relay/`（`ELASTIC_RELAY_VERSION = 1.0.0`）

## 1. 背景与目标

SFT 版弹性接力已实现并封版：用「训练计划游标 `committed_cursor`」与框架 `global_step` 解耦，
支持变卡数（change of GPU/NPU count）下的数据级断点续训，保证数据**不重不漏**。

目标：把这套能力扩展到 GRPO（`swift rlhf --rlhf_type grpo`）。

**核心难点**：GRPO 不是"样本过一遍"，而是"prompt → rollout 生成 → 训练复用"，
消费进度与 `global_step` 不是线性关系，且强依赖 rollout 引擎（vLLM/SGLang）。

## 2. 为什么不能直接复用 SFT 方案

| | SFT | GRPO |
|---|---|---|
| 消费单位 | 样本（sample） | prompt + 其 rollout |
| 一个 optimizer step 消耗 | `global_batch` 个样本 | 见第 3 节公式，非简单整批 |
| 顺序来源 | plan 顺序 | plan 顺序（prompt），但生成会打散/复用 |
| 额外状态 | 无 | rollout 引擎、ref 模型、KL/advantage、generation 缓冲 |
| 变卡数影响 | 只改 `gb` | 还改生成并行度、`max_num_seqs`、生成批次切分 |

因此 SFT 的 `committed = start + step × gb` **直接失效**，需要新的记账。

## 3. 关键参数与记账公式

来自 `swift/rlhf_trainers/`：

- `args_mixin.py:154-230`：
  - `global_batch_size = per_device_train_batch_size × world_size`
  - `generation_batch_size = global_batch_size × steps_per_generation`
  - 且 `generation_batch_size % num_generations == 0`（否则报错）
- `rollout_mixin.py:180`：`batch_size = generation_batch_size // num_generations`
  → **每轮生成消耗的 unique prompt 数 = `generation_batch_size // num_generations`**
- `grpo_trainer.py:197-206`：一轮生成结果被**复用** `steps_per_generation × num_iterations` 个训练 step
- `grpo_trainer.py:2118`：`num_iterations`（论文中的 μ，multi-step）

于是：

```
每轮生成消耗 prompt 数      P = generation_batch_size // num_generations
每轮覆盖的优化步数          S = steps_per_generation × num_iterations
每个优化步消耗 prompt 数    = P / S
                              = per_device_train_batch_size × world_size
                                / (num_generations × num_iterations)
```

**结论：GRPO 的游标应以「生成轮次（generation round）」为原子单位推进**：

```
generation_rounds = step // (steps_per_generation × num_iterations)
committed_prompts = start_prompt_cursor + generation_rounds × P
```

- 轮次边界上 `committed_prompts` 精确；
- **轮次中途**的 checkpoint 不能精确落到 prompt 边界（有 `num_iterations` 复用与生成缓冲），
  需要策略：要么只允许在轮次边界保存，要么中途停止时**回滚到上一轮次边界**（推荐后者，语义简单、不重不漏）。

## 4. 必须锁定的 recipe（跨段不可变）

以下任一变化都会让记账公式失效，必须写入 recipe 并在每段启动时校验：

- `num_generations`
- `steps_per_generation` 与 `generation_batch_size`（二者只能设其一，另一个推导；跨段需保持同一组合）
- `num_iterations`
- `per_device_train_batch_size`、`world_size`、`gradient_accumulation_steps`（用于推导 `steps_per_generation`）
- `max_completion_length`、temperature 等影响 rollout 的配置（影响生成但一般不改记账；仍需锁定以复现）

> 注意：SFT 允许跨段改 `gb`（因为记账是 `step × gb`）。GRPO 若要变卡数，必须重新推导
> `generation_batch_size / steps_per_generation` 与 `P`，并保证新旧 `P` 的切换只发生在轮次边界。

## 5. Rollout 引擎的跨段重建（最大的硬骨头）

- GRPO 依赖 vLLM/SGLang 生成（`rollout_mixin.py:464-466`）：
  `max_num_seqs = per_rollout_batch_size × steps_per_generation`；
- 变卡数会改变 TP/PP/DP 并行配置与 `max_num_seqs`，**必须整体重建 rollout 引擎**；
- 若 rollout 与训练 colocate（同卡），重建还要处理显存/权重同步；
- `swift` 现有的 NPU vLLM-Ascend 与 transformers 5.x 版本兼容问题（见 SFT 排查记录）同样会放大这里。

**推论**：GRPO 的"变卡数"不仅是数据记账问题，更是**生成引擎重建**问题。

## 6. 恢复语义需验证

- SFT 走 `resume_only_model + ignore_data_skip` 重置 optimizer；但在 `rlhf_trainers/` 中
  **未发现使用这两个 flag**（grep 无结果），其续训行为需重新确认；
- GRPO checkpoint 涉及：policy 权重、ref 模型、optimizer、KL/advantage、可能的生成缓冲；
- 我们方案的取向：**只加载 policy 权重，重置 optimizer 与 advantage 状态**；
  ref 模型为初始策略（LoRA 时即 base），需要保证与首段一致；
- `gkd_trainer.py:350-354` 的 `_buffered_inputs`（按 `steps_per_generation` 缓冲）说明"消费到哪"还受缓冲影响；
- callback 机制：GRPO trainer 继承 HF Trainer，理论支持 `TrainerCallback`，但 `on_step_end`
  在"生成步 vs 训练步"的语义不同，`elastic_stop`/`elastic_state` 需要针对 GRPO 校验或新增回调。

## 7. 数据层可复用的部分

以下 SFT 版组件**基本可复用**，只把"消费单位"从 sample 换成 prompt / generation round：

- `plan_builder.py`：prompt 序列的确定性重放（seed/epochs 语义可沿用）；
- `slicer.py`：切片 + dummy 补齐（GRPO 的 batch 对齐单位需按生成配置重算）；
- `state.py`：`platform_state.json` 原子写、committed 发现；
- `runner` / `scheduler`：控制面编排；
- STOP / 信号软停：可复用，但保存点要对齐 generation round 边界。

## 8. 分阶段落地

### 阶段 1：同卡数严格续训（可行性最高）
- 数据层复用 plan+cursor，游标以 generation round 推进；
- rollout 配置、卡数不变；
- 验证：`resume_only_model` 对 GRPO 是否安全、ref/optimizer/advantage 影响；
- 只在 generation round 边界保存 checkpoint。

### 阶段 2：变卡数，但固定 generation 配置
- 通过调整 `per_device_train_batch_size` / `steps_per_generation` 保持
  `generation_batch_size` 与 `P` 不变，使 `committed_prompts` 公式继续成立；
- 只在 generation round 边界切换；
- 处理 rollout 引擎重建。

### 阶段 3：完全弹性
- 允许 generation 配置变化；
- 需要重新推导记账、处理 mid-round 回滚、rollout 重建与版本兼容；
- 属于独立课题。

## 9. 风险与开放问题

1. GRPO 的中途 checkpoint 能否对齐轮次边界？若不能，回滚粒度是否可接受？
2. `num_iterations > 1` 时，同一 prompt 被训练多次属正常，如何向用户解释"不重不漏"的语义（针对 prompt 而非训练次数）？
3. rollout 引擎重建的正确性与稳定性（尤其 NPU vLLM-Ascend）。
4. `resume_only_model` 重置 optimizer 后，GRPO 的 KL/advantage 是否需要额外校正？
5. 动态过滤/丢弃样本（`grpo_trainer.py:672-689`）是否影响"消耗了哪些 prompt"的记账？
6. callback 与生成循环的交互（`on_step_end` 的触发时机）。

## 10. 验收标准（草案）

- prompt 级不重不漏：多段消费的 prompt 集合 + 顺序 == plan 的 prompt 序列（生成 round 为单位）；
- 停止/续训后 `committed_prompts` 精确、可从最新 committed 恢复；
- 同卡数续训：最终 eval/reward 与不中断基线差异在噪声范围；
- 变卡数（阶段 2）：切换点 reward 无爆炸，步数/时长符合预期；
- rollout 引擎重建无挂死、无权重错乱。

---

## 附：SFT 版可直接参照的实现

- 入口/编排：`swift/elastic_relay/runner.py`、`scheduler.py`、`launcher.py`
- 计划与切片：`plan_builder.py`、`slicer.py`
- 状态与回调：`state.py`、`callbacks.py`
- 自测：`tests/elastic_relay_selftest.py`
