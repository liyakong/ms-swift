# Custom reward plugin template for ms-swift 4.5.2 GRPO.
#
# 覆盖的奖励类型(框架支持的全部形式):
#   - 同步奖励函数  : ORM              -> MyReward
#   - 异步奖励函数  : AsyncORM         -> AsyncAPIExample
#   - 多任务(官方)  : 多函数各返回 None -> MathTaskReward / CodeTaskReward
#   - 多任务(替代)  : 单函数内部分发     -> MultiTaskReward
#   - 奖励模型      : rm_plugins       -> MyRMPlugin
#
# 用法:
#   1) 复制本文件为你的插件, 改类名与逻辑
#   2) 注册: orms['<name>'] = <Class>
#   3) 启动:
#        swift rlhf --rlhf_type grpo \
#            --external_plugins /abs/path/your_plugin.py \
#            --reward_funcs <name> ...
#
# 契约:
#   - 继承 swift.rewards.ORM
#   - 实现 __call__(self, completions, **kwargs) -> List[float]
#       completions: List[str], 模型生成结果
#       kwargs:      数据集列, 按列名批量传入 (e.g. solution / verification_info)
#       返回:        与 completions 等长、同序的 float 列表
#   - 注册: orms['<name>'] = <Class>
#
# 纪律:
#   - 只对 swift 4.5.2 负责, 不要用 3.x 的 swift.llm / swift.plugin / PtEngine
#   - 重库/网络客户端惰性导入(放 __init__ / __call__ 内), 顶层只 import 轻量依赖
#   - 依赖一律烤进镜像, 禁止运行时 pip install (离线 pod 会失败)
#   - 失败要么抛异常要么返回真实分数, 不要静默返回全 0 (会没梯度且难以察觉)
#   - 注册名建议加前缀(如 user_), 避免与内置 accuracy/format/cosine 等冲突

import asyncio
from typing import List

from swift.rewards import ORM, AsyncORM, orms, rm_plugins
from swift.rewards.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()


class MyReward(ORM):
    """示例: completion 非空给 1.0, 否则 0.0。"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        # 惰性导入重库 / 初始化客户端放这里, 例如:
        # import some_heavy_lib
        # self._client = some_heavy_lib.Client(...)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        # 数据集列可直接按名取用, 例如: gold = kwargs.get('solution')
        rewards: List[float] = []
        for content in completions:
            rewards.append(1.0 if content else 0.0)
        return rewards


class MultiTaskReward(ORM):
    """示例(替代方案): 在单个奖励内按数据集列 task_type 分发到不同子奖励。

    官方开发者指南的推荐做法是"多个奖励函数, 各自对非本任务返回 None"
    (见 docs/source/Instruction/GRPO/DeveloperGuide/multi_task.md):
    None 会被框架记为 NaN 并在 nansum 时忽略, 非本任务不计分也不拖低奖励。
    本类展示的是替代方案——在一个奖励里集中路由, 适合想统一管理分发的场景。

    所有数据集列都会作为 kwargs 传入, 因此能直接拿到 task_type / solution 等。
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        # 惰性初始化各子评分器(重库 / 客户端放这里), 例如:
        # import math_verify
        # self._code_client = ...

    def _score_math(self, completion: str, solution: str) -> float:
        # 示例逻辑: 真正使用时建议复用 math_verify 或 math_accuracy_plugin.py
        if not completion or solution is None:
            return 0.0
        return 1.0 if completion.strip() in solution else 0.0

    def _score_code(self, completion: str, verification_info) -> float:
        # 示例逻辑: 真正使用时跑 verification_info['test_cases']。
        # 若走网络/沙箱(Judge0/E2B), 把本方法改成 async def, 并在 __call__ 内 await。
        if not completion or not verification_info:
            return 0.0
        return 0.0

    def __call__(self, completions: List[str], task_type=None, **kwargs) -> List[float]:
        tasks = task_type if task_type is not None else ['default'] * len(completions)
        solutions = kwargs.get('solution') or [None] * len(completions)
        verifications = kwargs.get('verification_info') or [None] * len(completions)
        rewards: List[float] = []
        for i, task in enumerate(tasks):
            if task == 'math':
                rewards.append(self._score_math(completions[i], solutions[i]))
            elif task == 'verifiable_code':
                rewards.append(self._score_code(completions[i], verifications[i]))
            else:
                rewards.append(0.0)
        return rewards


class AsyncAPIExample(AsyncORM):
    """示例: 异步奖励函数(涉及 I/O, 如 API/数据库/沙箱调用)。

    框架会自动识别协程(__call__ 是 async)并用 asyncio.gather 并行执行
    (swift/rl_core/grpo_algorithm.py:12-14,72-78)。
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        # 惰性初始化重库/客户端放这里

    async def __call__(self, completions: List[str], **kwargs) -> List[float]:
        import aiohttp  # 重库惰性导入

        async with aiohttp.ClientSession() as session:
            rewards = await asyncio.gather(*[self._score_one(session, c) for c in completions])
        return list(rewards)

    async def _score_one(self, session, text: str) -> float:
        # 示例: 调用外部打分服务
        # async with session.post(url, json={'text': text}) as resp:
        #     return float((await resp.json())['score'])
        return 1.0 if text else 0.0


class MathTaskReward(ORM):
    """官方多任务写法: 只算自己负责的任务, 对非本任务返回 None。

    见 docs/source/Instruction/GRPO/DeveloperGuide/multi_task.md。
    None 会被框架记为 NaN 并在 nansum 时忽略, 非本任务不计分也不拖低奖励。
    """

    def __call__(self, completions: List[str], task=None, **kwargs) -> List[float]:
        tasks = task if task is not None else [None] * len(completions)
        return [self._score(c) if t == 'math' else None for c, t in zip(completions, tasks)]

    def _score(self, completion: str) -> float:
        return 1.0 if completion and r'\boxed' in completion else 0.0


class CodeTaskReward(ORM):
    """官方多任务写法: 代码任务分支(非本任务返回 None)。"""

    def __call__(self, completions: List[str], task=None, **kwargs) -> List[float]:
        tasks = task if task is not None else [None] * len(completions)
        return [self._score(c) if t == 'verifiable_code' else None for c, t in zip(completions, tasks)]

    def _score(self, completion: str) -> float:
        return 1.0 if completion and '```' in completion else 0.0


class MyRMPlugin(DefaultRMPlugin):
    """示例: 自定义奖励模型插件(生成式/打分模型)。

    若用现成的 DefaultRMPlugin / GenRMPlugin 则无需自定义; 需要定制时继承并覆盖
    __call__, 再注册到 rm_plugins。使用方式(注意不是 --reward_funcs):
        --reward_model <rm 路径> --reward_model_plugin user_my_rm
    """

    def __call__(self, inputs, **kwargs):
        # inputs: reward 行列表(含 messages 与数据集列)
        return super().__call__(inputs, **kwargs)


orms['user_my_reward'] = MyReward
orms['user_multi_task_reward'] = MultiTaskReward
orms['user_async_api'] = AsyncAPIExample
orms['user_math_task'] = MathTaskReward
orms['user_code_task'] = CodeTaskReward
rm_plugins['user_my_rm'] = MyRMPlugin
