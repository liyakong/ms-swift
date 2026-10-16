# 自定义奖励函数 (Custom Reward) 接入规范

面向平台/用户:如何给 GRPO 训练挂一个自定义奖励函数。模板见同目录
`custom_reward_template.py`。

## 框架支持的奖励类型与模板对应

| 类型 | 基类 / 入口 | 模板示例 |
|---|---|---|
| 同步奖励函数 | `ORM` + `orms` | `MyReward` |
| 异步奖励函数 | `AsyncORM` | `AsyncAPIExample` |
| 多任务(官方) | 多函数 + 非本任务返回 `None` | `MathTaskReward` / `CodeTaskReward` |
| 多任务(替代) | 单函数内部按列分发 | `MultiTaskReward` |
| 奖励模型 | `rm_plugins`(配 `--reward_model`) | `MyRMPlugin` |

## 1. 契约

- 继承 `swift.rewards.ORM`(不要用 3.x 的 `swift.plugin` / `swift.llm`)。
- 实现 `__call__(self, completions, **kwargs) -> List[float]`:
  - `completions`: `List[str]`,模型生成结果;
  - `kwargs`: 数据集列,按**列名**批量传入(如 `solution`、`verification_info`);
  - 返回:与 `completions` **等长、同序**的 `float` 列表。
- 注册: `orms['<name>'] = <Class>`。

## 2. 挂载方式

```bash
swift rlhf --rlhf_type grpo --model <model> \
    --external_plugins /abs/path/your_plugin.py \
    --reward_funcs <name> \
    [--reward_funcs a b --reward_weights 1.0 0.5] \
    --truncation_strategy left \
    ...
```

- 一个插件文件可注册多个奖励,用 `--reward_funcs a b` 组合,`--reward_weights` 配权重。
- 走 elastic_relay 时必须带 `--truncation_strategy left`。

## 3. 强制纪律

| 项 | 要求 |
|---|---|
| 版本 | 只对 4.5.2 负责;禁用 3.x 的 `swift.llm` / `swift.plugin` / `PtEngine` |
| 依赖 | 烤进镜像,禁止运行时 `pip install` |
| 导入 | 顶层只 import 轻量依赖;重库/网络客户端惰性导入 |
| 失败 | 抛异常或返回真实分数,**禁止静默返回全 0** |
| 命名 | 注册名加前缀(如 `user_`),避免与内置 `accuracy`/`format` 等冲突 |
| 路径 | 插件路径按进程 CWD(`/`)解析,平台应传**绝对路径**或统一挂到 `/opt/custom_rewards/` |

## 3.1 多任务奖励(按 task 字段路由)

框架开发者指南已覆盖此主题,**优先参考**:
- `docs/source/Instruction/GRPO/DeveloperGuide/reward_function.md`
- `docs/source/Instruction/GRPO/DeveloperGuide/multi_task.md`

做法:数据集加一个标识任务的列(官方示例用 `task`,你的数据里是 `task_type`),
该列会作为 kwargs 传入奖励函数。

**官方推荐**:写多个奖励函数,各自只算自己负责的任务,对**非本任务返回 `None`**。
框架把 `None` 记为 NaN,并在加权求和时用 `nansum` 忽略(`grpo_algorithm.py:66,77`、
`grpo_trainer.py:467`),所以非本任务既不计分也不拖低奖励。

```python
class MathReward(ORM):
    def __call__(self, completions, task_type, **kwargs):
        return [score_math(c) if t == 'math' else None
                for c, t in zip(completions, task_type)]

class CodeReward(ORM):
    def __call__(self, completions, task_type, **kwargs):
        return [score_code(c, v) if t == 'verifiable_code' else None
                for c, t, v in zip(completions, task_type, kwargs['verification_info'])]

orms['user_math'] = MathReward
orms['user_code'] = CodeReward
```

启动: `--reward_funcs user_math user_code`(可配 `--reward_weights`)。

**替代方案**:也可在**单个奖励内按 task 分发**(见模板 `MultiTaskReward`),适合想集中
管理路由的场景。

要点:
- 非本任务务必返回 `None`(**不要返回 0.0**,否则会拖低加权奖励)。
- kwarg 名必须与数据集**列名一致**(列叫 `task_type` 就不能写成 `task`)。
- 异步分支写成 `async def` 即可,框架自动识别并 `asyncio.gather`
  (`grpo_algorithm.py:12-14,72-78`)。
- 注意:`query`/`response` 列会被并成 `messages`,数据集里原有的 assistant 响应会被
  **丢弃**;需要保留的信息请放到额外列(见 `reward_function.md` 顶部说明)。

## 3.2 异步奖励函数

涉及 I/O(API / 数据库 / 沙箱)时用 `AsyncORM`;框架自动识别协程并 `asyncio.gather` 并行
执行,无需手动起事件循环(`grpo_algorithm.py:12-14,72-78`)。见模板 `AsyncAPIExample`。

```python
import asyncio
from swift.rewards import AsyncORM, orms

class MyAsyncReward(AsyncORM):
    async def __call__(self, completions, **kwargs):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            rewards = await asyncio.gather(*[_score(session, c) for c in completions])
        return list(rewards)

orms['user_async_api'] = MyAsyncReward
```

## 3.3 奖励模型(rm_plugins)

奖励模型(生成式 / 打分模型)是**另一条入口**,用 `--reward_model` 而非 `--reward_funcs`:
- 插件类实现 `__init__(self, model, template)` 与 `__call__(self, inputs, **kwargs)`:
  `inputs` 为 reward 行列表(含 `messages` 与数据集列),返回每条一个分数(`torch.Tensor`/list);
- 注册到 `rm_plugins['<name>']`,启动用
  `--reward_model <模型路径> --reward_model_plugin <name>`;
- 现成实现直接继承即可:`DefaultRMPlugin`(带 value head 的分类模型)、
  `GenRMPlugin`(生成式,LLM-as-judge)。

见模板 `MyRMPlugin`。

## 4. 平台侧需要提供

- 两个输入:插件文件路径 + `--reward_funcs` 名字(对应前端 `--external_plugins` / `--reward_funcs`)。
- 把插件文件放进容器:挂载或烤进镜像。
- 校验:启动前确认该奖励需要的库/密钥/服务已在镜像与环境中(缺环境要么补镜像,要么别启用)。

## 5. 常见奖励与依赖

| 场景 | 依赖 | 备注 |
|---|---|---|
| 数学判分 | `math_verify`(已在 `requirements/framework.txt`) | 参考 `math_accuracy_plugin.py` |
| 文本相似 | `rouge`(已在 framework) | |
| LLM 裁判 | `openai`(已在 framework) | 需网络/密钥 |
| 异步 HTTP | `aiohttp`(已在 framework) | 可对接自建 Judge0 |
| 代码执行(E2B) | `e2b-code-interpreter` | 需 `E2B_API_KEY` + 外网,离线集群不可用 |
| 代码执行(Judge0) | 仅 `aiohttp` | 自建 Judge0 服务即可,离线推荐 |

> 已在 `requirements/framework.txt` 的库无需重复安装;不在其中的库请在 `Dockerfile`
> 构建阶段补齐(带版本约束,勿动 NPU 栈)。
