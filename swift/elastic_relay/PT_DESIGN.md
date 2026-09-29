# 预训练（PT）弹性接力训练说明

> 分支：`feat/dpo-elastic`（PT/DPO 改动所在分支）
> 关联：SFT / GRPO 见 `GRPO_DESIGN.md`，DPO 见 `DPO_DESIGN.md`；公共机制（游标、state、回调、切片）同一套代码。

## 1. 背景与目标

把"数据级弹性接力"（跨卡数变更的断点续训，保证数据**不重不漏**）扩展到预训练
（`swift pt`）。目标与 SFT/DPO 一致：多段消费整份计划、每段可停在步边界、换卡数后从
`committed_cursor` 精确接续。

## 2. 为什么 PT ≈ SFT（记账层）

`swift` 内部：

- `SwiftPretrain(SwiftSft)`：**同一套 pipeline / trainer**；
- `PretrainArguments(SftArguments)` 只改了**两个开关**：

  ```python
  use_chat_template = False    # 不套 chat 模板，纯文本
  loss_scale = 'all'           # 所有 token 都算损失
  ```

因此 PT 的消费是**样本级线性**的（无 rollout、无轮次），记账与 SFT 完全相同：

```
committed_cursor = min(segment_start_cursor + global_step × segment_global_batch, total_units)
cursor_unit = sample
```

路由也无需新增：`ROUTES = {'sft', 'rlhf', 'pt'}`，`_prepare` 走 SFT 分支，
`_dispatch(argv, 'pt')` → `swift pt`。

## 3. 与 SFT 的唯一实质差异：监督范围

| | SFT | PT |
|---|---|---|
| 损失函数 | 交叉熵 | **同一个交叉熵** |
| `use_chat_template` | True | **False** |
| `loss_scale` | `default`（通常只算 assistant 回复） | **`all`（所有 token）** |
| 数据形态 | 对话 | 原始文本（`.txt` / `messages`） |

## 4. 核心问题：PT 的补齐行不是"天然零损失"

弹性接力为了"每步恰好 `gb` 条"（见 `GRPO_DESIGN.md`/`DPO_DESIGN.md` 的尾部处理说明），
会把每段末尾不足一批的尾巴**补满**。补齐行必须零损失，否则污染训练。

- **SFT**：补齐行是 user-only 对话 → 模板把非 assistant 标 `label=-100` → CE 天然跳过；`elastic_cross_entropy` 再对"全补齐 micro-batch"做分母 `clamp(min=1)` 防 NaN。
- **PT**：`use_chat_template=False` + `loss_scale='all'`，**没有"上下文"概念**，补齐行的文本是**真实 token**、label 就是它的下一个 token → CE 非零；只能靠**逐 token 权重**把它关掉。

官方文档（`docs/.../Custom-dataset.md:85`）：assistant 消息的 `"loss": false` 让该 content
**不计算损失**，且**优先级高于 `--loss_scale`**（含 `all`）。所以 PT 的补齐行写成：

```json
{"messages": [{"role": "assistant", "content": "<elastic-relay-pad>", "loss": false}]}
```

### 4.1 关键约束：补齐行与真实行的 **message 键集必须一致**

HF `datasets`（`Json` builder）**先用第一块数据推断 `features`，再对每一行 `cast`**。
若补齐行多了 `loss` 键、而真实行没有，加载时会直接失败：

```
TypeError: Couldn't cast array of type
struct<role: string, content: string, loss: bool>
to
{'role': Value('string'), 'content': Value('string')}
```

因此：

- **`make_dummy_sample` 镜像真实行的 message 键**（拷贝首条 message 的键集，再覆盖
  `role`/`content`/`loss`），不再硬编码键；
- **runner 在 `route='pt'` 时给真实 message 显式补 `"loss": true`**
  （`runner._ensure_message_loss`）——`loss: true` 本就是默认（受监督），语义不变；
- `.txt` 读取器直接产出带 `loss: true` 的 message（`plan_builder._read_file_samples`）。

> 教训：不能只靠 `pyarrow.json.read_json` 做本地验证——它会把跨块 struct 统一，
> 而 HF `datasets` 的"先推断再 cast"才会暴露该问题。

## 5. 关键修复：自定义 loss 必须应用 `loss_scale`

设了 `--loss_type` 后，trainer 走 `compute_loss_func`（`mixin.py:1072` →
`seq2seq_trainer.py:189`），**绕过**了它默认的 `per_token_loss_func(...) * loss_scale`
（`seq2seq_trainer.py:165-169`），并把 `loss_scale` 作为参数交给自定义 loss。

而原有的 `ElasticCrossEntropyLoss`（同 swift 自带的 `CustomCrossEntropyLoss`）**忽略了
`loss_scale`** → `"loss": false` 会**失效** → PT 补齐行照样产生损失。

修复（`swift/elastic_relay/loss.py`）：

```python
token_loss = per_token_loss_func(outputs, labels)   # 逐 token CE，忽略位为 0
if loss_scale is not None:
    token_loss = token_loss * loss_scale            # trainer 已 roll/flatten 到 [B*S]
loss = token_loss.sum() / denom.clamp(min=1)        # 分母下限 1，全补齐批 → 有限 0
```

> 说明：`clamp(min=1)` 防 NaN；`* loss_scale` 管监督范围/加权。两者独立。

## 6. 数据格式与任意列名

### 6.1 支持的格式

| 格式 | 说明 |
|---|---|
| `.txt` | **每行一条文档**（空行跳过）；包成标准 `messages` schema |
| `.jsonl` | 每行一个对象 |
| `.json` | 顶层数组 / 单个对象 / 包装键 `{"data"\|"train": [...]}` |
| `.parquet` | 需 pyarrow |

> `.json` 的"包装键"不是 HF 自动识别的（HF 需 `field=`，swift 未传），只是常见导出写法的便利支持。

### 6.2 任意列名

补齐行必须与真实行**列结构一致**，并清空"响应列"。为此 `slicer.make_dummy_sample`：

- 覆盖 swift `ResponsePreprocessor` 的**全部自动映射列名**：
  `response/answer/output/targets/target/answer_key/answers/solution/text/completion/content`，
  以及 `query/prompt/input/instruction/question/problem`；
- 识别 `messages` 结构；
- 兼容 `--columns`（**框架原生参数**，列名映射 `{"你的列": "swift标准列"}`）：runner 解析后把
  "映射到 `response` 的自定义列名"也交给补齐行清空。

### 6.3 多数据集与数据归一化

- **支持多个 `--dataset`**（`--dataset a b`）：按**给定顺序**合并成一份计划（顺序参与指纹，
  跨段必须一致）。
- 构建计划前，对**每个数据集文件**用 swift 自己的 `AutoPreprocessor` **归一化到标准 `messages`**
  （与框架同一套映射：`messages` / `conversation` / `query-response` / `alpaca` / …），再
  `remove_useless_columns` 只保留标准列。因此**不同原始 schema 的数据集可以合并**，
  不会再因列不一致而失败。
- **图片不会变成 PIL**：预处理器的 `_cast_pil_image` 关闭解码（`decode=False`）；路径/base64 被
  `_cast_mm_data` 规整成**可序列化**的 `{'bytes': None, 'path': '...'}`。
- 可选列（`images/videos/audios/objects/tools`）**允许缺失**：缺失的行补**同类型**空值
  （list → `[]`，string → `''`）；其它列不一致仍会**明确报错**。
- 归一化后所有行都是标准 `messages`，所以"**单文件 + 全局打乱**"的 plan 设计不变。

## 7. 实现位置

| 文件 | 改动 |
|---|---|
| `swift/elastic_relay/plan_builder.py` | 新增 `.txt`（每行一条）；`.json` 支持数组/单对象/包装键；**多数据集**（`list_data_files`/`dataset_fingerprint`/`load_raw_samples` 接受路径列表）；**`normalize_dataset`**（`AutoPreprocessor` 归一化到标准 schema）；可选列缺失补空值 |
| `swift/elastic_relay/slicer.py` | `make_dummy_sample(response_keys=, no_chat_template=)`：PT → `assistant + loss:false`，并**镜像真实 message 的键集**；列名对齐 swift + `--columns` |
| `swift/elastic_relay/loss.py` | `ElasticCrossEntropyLoss` 应用 `loss_scale`（`loss:false` 生效） |
| `swift/elastic_relay/runner.py` | `route='pt'`：`no_chat_template=True` + `_response_keys(--columns)`；`_ensure_message_loss` 给真实 message 补 `loss:true`；`_pt_recipe` 锁 `use_chat_template/loss_scale`；`resolved['route']` |
| `swift/elastic_relay/plan_builder.py` | `.txt` 行产出带 `loss:true` 的 assistant message |
| `swift/elastic_relay/state.py` | `scrub_external_plugins`：清空 `args.json` 里的插件路径（checkpoint 可移植） |
| `swift/elastic_relay/callbacks.py` | 状态日志标注 `route=pt`；每次存盘 `scrub_external_plugins` |
| `tests/elastic_relay_selftest.py` | `.txt` / `.json` 形态 / PT 补齐行 / PT runner / recipe 单测 |
| `scripts/pt_elastic_smoke.sh` | 同卡续训 / 变卡(2→4) 冒烟 |

## 8. 强约束

- `--packing false`：预训练常用 packing 提升吞吐，但 elastic relay **不支持 packing**（会重排样本，破坏计划顺序）；
- `--streaming / --interleave_prob` 关闭（无固定长度/固定顺序）；
- `--dataset_shuffle / --group_by_length false`（runner 强制）；
- **recipe 跨段锁定**：`seed / num_train_epochs / dataset 指纹 / use_chat_template / loss_scale`；
- 世界派生量（`gb / num_gpus / per_device_batch`）**不锁** → 卡数可变。

## 9. 使用

```bash
python -m swift.elastic_relay.runner pt \
  --model <base> \
  --dataset <corpus.txt 或 目录> \
  --per_device_train_batch_size <pdb> --gradient_accumulation_steps <ga> \
  --output_dir /abs/out/pt
```

续训：加 `--resume_from_checkpoint /abs/out/pt`（runner 会自动注入
`--resume_only_model true --ignore_data_skip true`）。

自定义列名示例：

```bash
--columns '{"my_text": "response"}'
```

## 10. 验证

- 逻辑自测：`python tests/elastic_relay_selftest.py`（含 PT 用例）；
- 端到端 smoke：
  ```bash
  MODEL=<base> DSET=<corpus.txt> NGPU1=2 NGPU2=4 bash scripts/pt_elastic_smoke.sh
  ```
  校验每个 `platform_state.json`：`mode=sft`、`cursor_unit=sample`、
  `committed = min(start + step×gb, total)`。

### 10.1 多数据集 / 归一化 验证点（发布前必测）

1. **异构 schema 混训**：`query-response` + `messages`（+ `alpaca`）多个 `--dataset` 一起训练，能正常加载；
2. **多模态图片**：切片里 `images` 已规整为 `[{'bytes': null, 'path': '...'}]`，训练正常；
3. **PT（`.txt`）回归**：`.txt` 经归一化后仍正常；
4. **SFT 补齐行零损失**：归一化后 messages 可能自带 `loss/loss_scale` 键，确认补齐行仍不产生损失；
5. **续训衔接**：`committed = start + step×gb` 跨段成立、loss 连续。

> **真机验证状态：待执行**（代码已完成适配与自测）。

## 11. 已知限制

- **packing 必须关闭**（弹性接力的硬约束）；
- `.txt` 为**每行一条**；要"整篇文档一条"需自行分块或改用 jsonl；
- `.txt` 与 jsonl **可以混在同一目录/多个 `--dataset`**（构建计划前会按 6.3 归一化到标准 `messages`）；
- 补齐行用 `loss:false`（scale=0），其 token 的 label **不是 -100**，仍会计入我们的分母
  `(labels!=-100).sum()` → 对"含补齐行的那一步"有**轻微稀释**（无 NaN 风险）。如需精确排除，
  可把分母改为 `((labels!=-100) & (loss_scale>0)).sum()`（需兼容 `loss_scale>1` 的加权语义）。
