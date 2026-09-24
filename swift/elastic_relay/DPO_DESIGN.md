# DPO 弹性接力训练说明

> 分支：`feat/dpo-elastic`
> 关联：SFT / GRPO 见 `GRPO_DESIGN.md`；公共机制（游标、state、回调、切片）同一套代码。

## 1. 背景与目标

把"数据级弹性接力"（跨卡数变更的断点续训，保证数据**不重不漏**）扩展到 DPO
（`swift rlhf --rlhf_type dpo`，以及共用同一 trainer 的 `orpo` / `cpo`；swift 会把
`simpo` 归一化为 `cpo`）。

**结论：DPO 属"SFT 级难度"**——它没有 rollout、没有 generation round，消费是**样本级线性**的：

- 一个样本 = 一对偏好（chosen + rejected）；
- 每个 optimizer step 消耗 `gb = num_gpus × per_device_batch × grad_accum` 个样本。

因此 SFT 的记账公式**直接适用**，不需要 GRPO 那套轮次/整轮切片。

## 2. 与 SFT / GRPO 的对比

| | SFT | DPO | GRPO |
|---|---|---|---|
| 消费单位 | 样本 | 偏好对 | prompt + rollout |
| 每步消耗 | `gb` 个样本 | `gb` 个偏好对 | 见 GRPO 文档公式 |
| 有 rollout/轮次 | 否 | 否 | 是 |
| 记账 | `start + step×gb` | `start + step×gb` | 整轮对齐 |
| 补齐行 | 全 `-100` → 零损失 | **需要显式屏蔽** | 不补齐（整轮） |

记账（`callbacks.py`，`mode='sft'`）：

```
committed_cursor = min(segment_start_cursor + global_step × segment_global_batch, total_units)
```

## 3. 核心问题：补齐行对 DPO 不是零损失

### 3.1 为什么会有补齐行

`calculator.plan_segment` 在每段末尾把不足一批的尾巴补齐到 `gb` 的整数倍（`end_cursor = total_units`），
否则：不满一批会被 `dataloader_drop_last` 丢掉却仍被计为"已消费" → **漏**；或跑半批导致
`step×gb` 记账对不上 → **重**；或尾巴永远吃不掉 → **卡死**。详见 `GRPO_DESIGN.md` 相关小节。

### 3.2 为什么 SFT 可以、DPO 不行

- **SFT**：补齐行是 user-only 对话 → 模板渲染后 label 全 `-100` → 交叉熵无监督 token → **贡献 0**。
- **DPO**：损失是 `-log σ(β · margin)`，每个样本**必然**产生一项：
  - 复制真实对 → 真实非零；
  - 清空 chosen、留 rejected → 大损失且方向错；
  - chosen == rejected → `-log σ(0) = log2 ≈ 0.693`。
  
  **没有任何有限输入能让 DPO 损失等于 0**，所以补齐行必须**显式排除**。

### 3.3 采用的方案：从 `loss_mask` 反推 `pad_mask`

现有 `slicer.make_dummy_sample` 已经把补齐行的 **chosen 侧清成 user-only**（`messages` →
`[{'role':'user','content':<pad>}]`），即 chosen 侧全 `-100`。于是：

```python
# dpo_trainer.concatenated_forward（非 padding-free 分支）
chosen_mask   = loss_mask[:num_examples]
rejected_mask = loss_mask[num_examples:]
output['pad_mask'] = chosen_mask.sum(dim=-1) == 0
```

`get_batch_loss_metrics` 据此排除：

```python
keep = (~pad_mask).to(losses.dtype)
loss = losses.mul(keep).sum() / keep.sum().clamp(min=1)   # 全 pad 的 rank → 有限 0，不 NaN
```

**为什么不用"数据集列标记"**：DPO 默认 `remove_unused_columns=True`（只有 GRPO 关掉），
加自定义列会在 dataset 加载阶段被丢弃；若改成 `False`，其它杂质列会顺 `_extra_kwargs`
一路进模型 forward，有回归风险。用 `loss_mask` 无需加列、无需改 flag。

同时把 `mean_chosen_logits` / `mean_rejected_logits` / `nll_loss` 的
`x[mask].mean()` 改成 mask-safe 求和均值——否则全补齐行会得到**空选择 → NaN**。

## 4. 实现位置

| 文件 | 改动 |
|---|---|
| `swift/rlhf_trainers/dpo_trainer.py` | `concatenated_forward` 计算 `pad_mask` 并返回；三个均值改为 mask-safe；`get_batch_loss_metrics` 按 `pad_mask` 掩码损失与指标，最终按 `keep` 归约 |
| `swift/elastic_relay/runner.py` | 识别 DPO 家族（`DPO_TYPES = {dpo, orpo, cpo}`）：记账走 `mode='sft'`；**跳过** SFT 的 `--loss_type elastic_cross_entropy` 注入；新增 `_dpo_recipe`（recipe 锁）与 `_check_dpo_constraints`（拒绝 `padding_free`）；`resolved['rlhf_type']` 记录 DPO 类型 |
| `tests/elastic_relay_selftest.py` | DPO 相关单测（recipe 锁、dummy 语义、runner 不注入 elastic loss_type） |
| `scripts/dpo_elastic_smoke.sh` | 一键 smoke：同卡续训 / 变卡(2→4)，校验游标公式 |

## 5. 强约束（runner 会校验）

- `--padding_free false`：`pad_mask` 从二维 `loss_mask` 推导，padding-free 会摊平成 `cu_seqlens`；
- `--packing / --streaming / --interleave_prob` 关闭（沿用 SFT 约束，会改变顺序/长度）；
- `--dataset_shuffle / --train_dataloader_shuffle / --group_by_length false`（runner 强制）；
- **recipe 跨段锁定**：`dpo_rlhf_type / beta / loss_type / loss_weights / rpo_alpha / ld_alpha /
  label_smoothing / reference_free / ref_model / ref_adapter_name / sync_ref_model /
  precompute_ref_log_probs`（改变其一都会改变目标函数或参考模型语义）；
- 世界派生量（`gb`、`num_gpus`、`per_device_batch`）**不锁** → 卡数可变。

## 6. 参考模型（`ref_model`）

DPO 的 ref 应是**基座初始策略**。swift 对非 GRPO 默认 `ref_model = self.model`
（`rlhf_args.py:294`）。续训时若 `resume_only_model` 把 checkpoint 权重载入 `model`，
而 ref 又不在此前建好，ref 会变成"上一段 checkpoint"→ KL 基准每段漂移。

- **LoRA**：ref 由 base 派生，无需指定；
- **全参 + `beta != 0`**：建议显式 `--ref_model <基座路径>`；
- 待真机确认 ref 创建时机后再决定是否像 GRPO 一样**强制**要求。

## 7. 使用

```bash
python -m swift.elastic_relay.runner rlhf \
  --rlhf_type dpo \
  --model <base> \
  --dataset <偏好数据目录或文件> \
  --per_device_train_batch_size <pdb> --gradient_accumulation_steps <ga> \
  --output_dir /abs/out \
  ... （其余 DPO 参数照常，如 --beta）
```

续训：加 `--resume_from_checkpoint /abs/out`（或上一段 checkpoint 目录）。
runner 会自动注入 `--resume_only_model true --ignore_data_skip true`。

## 8. 验证

- 逻辑自测：`python tests/elastic_relay_selftest.py`（含 DPO 用例）；
- 端到端 smoke：
  ```bash
  MODEL=<base> DSET=<偏好数据> NGPU1=2 NGPU2=4 bash scripts/dpo_elastic_smoke.sh
  ```
  校验每个 `platform_state.json`：`mode=sft`、`cursor_unit=sample`、
  `committed = min(start + step×gb, total)`。

## 9. 已知限制

- **只支持"只变卡数"**；跨段改 generation 配置对 DPO 无意义（无 rollout）；
- `bco_pair` / `aot` / `aot_unpaired` 等**在 batch 内做排序/统计**的 `loss_type`，掩码会改变
  其 batch 级语义；常规 `sigmoid / hinge / ipo / robust / exo_pair / sppo_hard / nca_pair /
  apo_* / sft` 无问题；
- 真实样本若本身 chosen 侧无任何监督 token，会被误判为补齐行（属退化数据）；
- server 模式不适用（DPO 不涉及 rollout 引擎）；
- `precompute_ref_log_probs` 被纳入 recipe 锁定；如需开启，请确认切片后重算。

## 10. checkpoint 可移植性（export / merge）

runner 会注入 `--external_plugins <repo>/swift/elastic_relay/plugin.py`，而 swift 的 `save_args`
会把**所有参数**（含该路径）写进 `args.json`，并复制进每个 checkpoint。这会让 checkpoint 变成
"**镜像相关**"：在**不含该路径的镜像**里 `swift export/merge-lora` 时，`load_args` 会去 import
它，`import_external_file` 的 `assert os.path.isdir(py_dir)` 直接失败。

处理：

- **新训练**：runner 在每段结束后调用 `_strip_external_plugins`，把 `external_plugins` /
  `custom_register_path` 从 `output_dir/**/args.json` 清空 → checkpoint 变得**可移植**。
  这对续训零影响：runner 每段都强制 `--load_args false` 且从 argv 重新注入插件；
- **历史 checkpoint**（清理前生成的）：合并时加 `--load_args false` 跳过读取，或手动把
  `args.json` 里的 `external_plugins` 置空，或用含 `swift/elastic_relay` 的镜像合并。
