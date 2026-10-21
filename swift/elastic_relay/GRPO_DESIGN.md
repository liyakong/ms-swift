# GRPO 弹性接力训练设计文档（草案）

> 状态：**已封版（v1.1.0）** — Phase 1（同卡数）与 Phase 2（变卡数）在 GPU 与 NPU 上均验证通过（见文末验证记录）
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
每轮覆盖的优化步数          S = steps_per_generation × num_iterations // gradient_accumulation_steps
每个优化步消耗 prompt 数    = P / S
                              = per_device_train_batch_size × world_size × gradient_accumulation_steps
                                / (num_generations × num_iterations)
```

**结论：GRPO 的游标应以「生成轮次（generation round）」为原子单位推进**：

```
optimizer_steps_per_round = steps_per_generation × num_iterations // gradient_accumulation_steps
completed_rounds          = global_step // optimizer_steps_per_round
committed_prompts         = start_prompt_cursor + completed_rounds × P
```

- 轮次边界上 `committed_prompts` 精确；
- **轮次中途**的 checkpoint 不能精确落到 prompt 边界（有 `num_iterations` 复用与生成缓冲），
  需要策略：要么只允许在轮次边界保存，要么中途停止时**回滚到上一轮次边界**（推荐后者，语义简单、不重不漏）。
- 关键前提 **`dataset_shuffle=false`**：此时 sampler 顺序取数据集，unique prompt 顺序 == plan 顺序，
  与 `P`（即卡数/`generation_batch_size`）无关，因此 **`P` 可以随卡数变化**而不破坏不重不漏。

## 4. 必须锁定的 recipe（跨段不可变）

规则：**只锁"决定顺序/语义"的参数；world 派生量不锁**（因为 `shuffle=false` 时 prompt 顺序
与 P 无关，P 每段按实际卡数重算）。

锁定：

- `seed` / `num_train_epochs` / `dataset 指纹`（决定 plan 顺序）
- `num_generations` / `num_iterations`（分组与复用语义）
- `per_device_train_batch_size` / `gradient_accumulation_steps`（batch/优化语义）
- `tuner_type` / `beta` / `ref_model` / `ref_adapters` / `sync_ref_model`（KL 基准与目标函数）

**不锁**（随卡数变化，每段重算并写入该段 config）：

- `world_size`、`steps_per_generation`、`generation_batch_size`、`prompts_per_round`、
  `optimizer_steps_per_round`

> 前提仍是 `dataset_shuffle=false`：否则 P 变化会改变 prompt 顺序，记账失效。
> `max_completion_length`/`temperature` 等影响生成但一般不改记账；为可复现仍建议保持不变。

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

---

## Phase 1 实现说明（已落地）

Phase 1 = **同卡数**下的 GRPO 数据级续训，colocate / server 两种 rollout 都兼容。
（变卡数属于 Phase 2，本阶段不放开。）

### 新增/改动

| 文件 | 内容 |
|---|---|
| `swift/elastic_relay/grpo.py` | 纯函数记账：`resolve_generation_params`（复刻 `_init_generation_batch_params`）、`round_aligned_end`、`completed_rounds`、`committed_prompts`、`is_round_boundary` |
| `swift/elastic_relay/runner.py` | 支持路由 `sft`/`rlhf`/`pt`；`route=rlhf` 且 `--rlhf_type grpo` 时走 GRPO 分支：整轮切片、按 `save_steps` 对齐轮边界、`committed_prompts` 配置注入、recipe 校验、GRPO 约束校验 |
| `swift/elastic_relay/callbacks.py` | 回调按 `mode` 区分：GRPO 只在生成轮边界提交 `platform_state.json`（写 `cursor_unit=prompt`），STOP 延迟到轮边界；SFT 逻辑不变 |
| `tests/elastic_relay_selftest.py` | 新增 GRPO 记账与 runner 分支单测 |

### Phase 1 的强约束（runner 会校验）

- `--dataset_shuffle false`（runner 强制写入）；
- `--truncation_strategy left`（`delete`/`raise` 会走 resample，脱离 plan）；
- `--dynamic_sample false`（默认）；
- 整轮切片、**不使用 dummy 补齐**（GRPO 补样会被真的拿去生成）；
- `save_steps` 自动对齐到 `optimizer_steps_per_round`；STOP 延迟到轮边界；
- recipe 锁定：`seed / num_train_epochs / dataset 指纹 / num_generations / num_iterations /
  per_device_batch / gradient_accumulation_steps / beta / ref_model / ref_adapters /
  sync_ref_model / tuner_type` 跨段不可变。**world 派生量（`steps_per_generation`、
  `generation_batch_size`、`prompts_per_round`）不锁**，所以卡数可改；
- **全量 GRPO 且 `beta != 0` 的 KL 基准固定**：未显式给 `--ref_model` 时，runner 会自动用
  `--model`（固定 base）补上并写入 recipe；否则框架 `ref_model = self.model` 在续训时会变成
  "上一段 checkpoint"，KL 基准每段漂移。LoRA 的 ref 由 base 派生，无需指定。
- 追加锁定 rollout 后端与采样：`use_vllm / vllm_mode / temperature / top_p / top_k /
  max_completion_length`（不影响游标，但改变生成分布，混用会造成语义漂移）；
- colocate 启动期校验 `vllm_tensor_parallel_size | world_size`；
- GRPO + DeepSpeed 的 `offload`（`--offload_model/-offload_optimizer` 或 `*_offload` preset）
  在**部分 DeepSpeed 版本**上会触发 ZeRO-3 `parameter_offload` 的 `pop from an empty deque`
  （GRPO 有额外本地 forward）。属版本相关：**实测 `deepspeed==0.19.7` 可用**（`0.18.9` 会报）；
  退路是 `zero2` 或 zero3 不带 offload。runner 在检测到 offload 时会告警。

### 恢复语义

沿用 SFT：`resume_only_model + ignore_data_skip`（只加载 policy 权重、`global_step=0`、重置
optimizer/RNG）；框架会自动重置 `_step`/`_buffered_inputs`，从该段切片第一轮重新生成；
ref 模型每段从 base 重建。封版验证中：变卡切换点 reward 无爆炸、步数/时长符合预期。

### 使用

```bash
python -m swift.elastic_relay.runner rlhf \
  --rlhf_type grpo \
  --model ... --dataset <prompt 目录> \
  --truncation_strategy left --dataset_shuffle false \
  --per_device_train_batch_size ... --num_generations ... \
  --output_dir /abs/out \
  ... （其余 GRPO 参数照常）
```

续训：加 `--resume_from_checkpoint /abs/out`（或上一段 checkpoint 目录）。

---

## 验证记录（回归基线）

代码：`swift/elastic_relay`（`ELASTIC_RELAY_VERSION = 1.1.0`，**已封版**）。
环境：
- NPU（Ascend）+ transformers 5.x + `deepspeed==0.19.7`；colocate 走 vllm-ascend。
- GPU + transformers 5.x + `deepspeed==0.19.7`。

| 平台 | rollout | 卡数 | DeepSpeed | 结果 |
|---|---|---|---|---|
| GPU | transformers（`--use_vllm false`） | 固定 | 无 / 有 | ✅ 通过 |
| GPU | transformers | **变卡（2→4）** | 无 / 有 | ✅ 通过 |
| GPU | colocate（`--vllm_mode colocate`） | 固定 | 无 / 有 | ✅ 通过 |
| GPU | colocate | **变卡（2→4）** | 无 / 有 | ✅ 通过 |
| NPU | transformers | 固定 | 无 / 有 | ✅ 通过 |
| NPU | transformers | **变卡（2→4）** | 无 / 有 | ✅ 通过 |
| NPU | colocate | 固定 | 无 / 有 | ✅ 通过 |
| NPU | colocate | **变卡（2→4）** | 无 / 有 | ✅ 通过 |
| — | server | — | — | 平台未适配，不验证 |

已验证的关键点：

- 游标公式 `committed = start + (step // OSR) × P` 精确，跨段严丝合缝；
- 变卡时 `gb` / `P` 随卡数变化（如 2 卡 `gb=4,P=2` → 4 卡 `gb=8,P=4`），recipe 不冲突、顺序不乱；
- 整轮切片（无 dummy）、`save_steps` 对齐轮边界、STOP 延迟到轮边界；
- `use_vllm / vllm_mode / temperature / top_p / top_k / max_completion_length / ref* / beta` 跨段锁定；
- colocate 的 `TP | world_size` 校验生效；
- `resume_only_model + ignore_data_skip`（runner 自动注入）在有无 DeepSpeed 下均正确恢复权重、游标接续。

已知限制 / 注意：

- server 模式未适配（平台侧），如需支持要先设计"续训后 server 权重重载/重启"；
- `zero3 + offload` 需 `deepspeed>=0.19.7`（`0.18.9` 会触发 `parameter_offload` 报错）；
- 全量 GRPO 且 `beta != 0`：未给 `--ref_model` 时自动用 `--model` 固定 KL 基准（否则 ref 会随 checkpoint 漂移）；
- 只支持"只变卡数"的弹性；跨段改 generation 配置属 Phase 3（未做）。

---

## 封版结论（v1.1.0）

- 代码冻结于 tag **`elastic-relay-v1.1.0`**；后续任何改动需另起版本号。
- 支持范围：**SFT + GRPO 的数据级弹性接力（只变卡数）**；已在 GPU 与 NPU 上回归通过。
- 不支持：server 模式、Megatron、streaming/packing、跨段改 generation 配置。


