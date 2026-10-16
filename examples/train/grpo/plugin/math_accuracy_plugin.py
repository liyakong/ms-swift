# Custom reward plugin for a math GRPO dataset.
#
# Dataset schema (one row):
#   messages : [{"role": "user", ...}, {"role": "assistant", ...}]
#   solution : reference reasoning ending with "\\boxed{<answer>}"
#
# Usage:
#   swift rlhf --rlhf_type grpo \
#       --external_plugins examples/train/grpo/plugin/math_accuracy_plugin.py \
#       --reward_funcs math_accuracy math_format \
#       --dataset <.../query-response.no_problem.jsonl> \
#       ...
#
# `math_accuracy` compares the completion's final answer against `solution`.
# It uses `math_verify` when installed, otherwise falls back to a dependency-free
# normalized comparison (so it works in a minimal image with a real reward signal
# instead of silently returning zeros).

import importlib.util
import re
from fractions import Fraction
from typing import List, Optional

from swift.rewards import ORM, orms
from swift.utils import get_logger

logger = get_logger()

_HAVE_MATH_VERIFY = importlib.util.find_spec('math_verify') is not None


def _last_boxed(text: str) -> Optional[str]:
    """Return the content of the last ``\\boxed{...}`` (brace-matched)."""
    idx = text.rfind(r'\boxed')
    while idx != -1:
        brace = text.find('{', idx)
        if brace != -1:
            depth = 0
            for i in range(brace, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[brace + 1:i]
        idx = text.rfind(r'\boxed', 0, idx)
    return None


def _extract_answer(text: str) -> str:
    """Extract the final answer from a completion / solution string."""
    if not text:
        return ''
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if numbers:
        return numbers[-1]
    return text.strip()


def _normalize(ans: str) -> str:
    """Normalize a LaTeX-ish answer for string comparison."""
    if not ans:
        return ''
    s = ans.strip()
    boxed = _last_boxed(s)
    if boxed is not None:
        s = boxed
    s = s.replace(r'\left', '').replace(r'\right', '')
    s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\dfrac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\tfrac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
    s = s.replace('$', '').replace(r'\%', '%').replace(r'\,', '')
    s = s.replace('\\!', '').replace(' ', '')
    s = s.replace('{', '').replace('}', '')
    s = s.rstrip(' .,;')
    return s


def _to_fraction(s: str) -> Optional[Fraction]:
    """Best-effort conversion of a numeric string to a Fraction."""
    s = s.strip()
    if not s:
        return None
    for candidate in (s, f'({s})'):
        try:
            return Fraction(candidate)
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return Fraction(str(float(s)))
    except (ValueError, TypeError):
        pass
    try:
        value = eval(s, {'__builtins__': {}}, {})  # noqa: S307 - restricted namespace
        return Fraction(str(value))
    except Exception:
        return None


def _answers_equal(pred: str, gold: str) -> bool:
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    pf, gf = _to_fraction(p), _to_fraction(g)
    if pf is not None and gf is not None and pf == gf:
        return True
    return False


def _math_verify_equal(pred: str, gold: str) -> Optional[bool]:
    """Use math_verify if available; return None when unavailable/undecidable."""
    if not _HAVE_MATH_VERIFY:
        return None
    try:
        from math_verify import parse, verify

        gold_parsed = parse(gold, extraction_mode='first_match')
        if not gold_parsed:
            return False
        pred_parsed = parse(pred, extraction_mode='first_match')
        return bool(verify(gold_parsed, pred_parsed))
    except Exception as e:  # noqa: BLE001
        logger.warning('math_verify failed, falling back to string compare: %s', e)
        return None


class MathAccuracy(ORM):
    """Reward 1.0 when the completion's final answer matches the reference."""

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for content, gold in zip(completions, solution):
            pred = _extract_answer(content or '')
            gold_ans = _extract_answer(gold or '')
            result = _math_verify_equal(pred, gold_ans)
            if result is None:
                result = _answers_equal(pred, gold_ans)
            rewards.append(1.0 if result else 0.0)
        return rewards


class MathFormat(ORM):
    """Reward 1.0 when the completion ends with a ``\\boxed{...}`` answer."""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for content in completions:
            content = content or ''
            has_boxed = re.search(r'\\boxed\{.*?\}(?![\s\S]*\\boxed)', content, re.DOTALL) is not None
            rewards.append(1.0 if has_boxed else 0.0)
        return rewards


orms['math_accuracy'] = MathAccuracy
orms['math_format'] = MathFormat
