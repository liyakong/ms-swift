# Copyright (c) ModelScope Contributors. All rights reserved.
"""Physical slicing of the training plan, with dummy padding for the tail.

The slice for segment *N* is ``plan[start_cursor:end_cursor]`` plus ``pad_count`` dummy
rows so that the file length is an exact multiple of the global batch. Dummy rows are
built from a real record with a user-only conversation, so the template emits an all
``-100`` label row (no supervised tokens).
"""
import copy
import itertools
import json
import os
from typing import Any, Callable, Dict, Iterable, Optional, Union

from swift.utils import get_logger

from .constants import DUMMY_CONTENT, SLICE_META_FILENAME

logger = get_logger()

# Mirror swift's `ResponsePreprocessor` auto column mapping
# (swift/dataset/preprocessor/core.py) so the padding row neutralizes the same fields the
# training pipeline treats as the response.
_RESPONSE_KEYS = ('response', 'answer', 'output', 'targets', 'target', 'answer_key', 'answers', 'solution',
                  'text', 'completion', 'content')
_QUERY_KEYS = ('query', 'prompt', 'input', 'instruction', 'question', 'problem')


def _first_record(plan_path: str) -> Dict[str, Any]:
    with open(plan_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def make_dummy_sample(template_record: Optional[Dict[str, Any]] = None,
                      *,
                      response_keys: Optional[Iterable[str]] = None,
                      no_chat_template: bool = False) -> Dict[str, Any]:
    """A padding row that keeps the *same columns* as a real record and produces ~no loss.

    Schema compatibility matters: the slice must be loadable by HF ``datasets``, whose
    feature inference rejects a padding row that introduces/omits columns. So we copy a
    real record and only neutralize its response.

    * ``messages``-style (the pad message mirrors the real message's columns, so HF
      ``datasets`` can cast the whole slice to one schema):
        - chat template (SFT): a user-only conversation (all labels ``-100``);
        - no chat template (PT, ``loss_scale='all'``): an assistant message with
          ``"loss": false`` — the per-message ``loss`` field overrides ``--loss_scale``, so
          the row contributes no loss (a user-only row would be fully supervised here).
      The PT ``loss: false`` key must also exist on real messages; the runner adds an
      explicit ``loss: true`` (see ``runner._ensure_message_loss``) so the key sets match.
    * response-style: empty the response field. ``response_keys`` lets the caller add the
      dataset's own column names (swift auto-mapping + ``--columns``); the built-in list
      mirrors swift's ``ResponsePreprocessor``.
    * unknown: keep the record as-is (a bounded duplicate, better than a schema clash).
    """
    record = copy.deepcopy(template_record) if template_record else {}
    if not record:
        pad = {'role': 'assistant' if no_chat_template else 'user', 'content': DUMMY_CONTENT}
        if no_chat_template:
            pad['loss'] = False
        return {'messages': [pad]}
    if 'messages' in record:
        # Mirror the real message's columns: HF `datasets` infers features from the first block
        # and then casts every row, so a padding row that adds/omits a key breaks loading.
        template = next((m for m in (record.get('messages') or []) if isinstance(m, dict)), {})
        pad = dict(template)
        pad['role'] = 'assistant' if no_chat_template else 'user'
        pad['content'] = DUMMY_CONTENT
        if 'loss' in pad or no_chat_template:
            pad['loss'] = False
        if 'loss_scale' in pad:
            pad['loss_scale'] = 0.
        record['messages'] = [pad]
        # Multimodal rows keep their resources in top-level columns; the pad message has no
        # <image>/<video>/<audio>/<bbox> placeholder, so empty those columns (keys are kept so
        # the schema still matches). Otherwise the image count would not match the placeholders.
        for key in ('images', 'videos', 'audios', 'objects'):
            if key in record and isinstance(record[key], list):
                record[key] = []
        return record
    keys = list(response_keys or []) + [k for k in _RESPONSE_KEYS if k not in (response_keys or [])]
    for key in keys:
        if key in record and isinstance(record[key], str):
            record[key] = ''
            return record
    for key in _QUERY_KEYS:
        if key in record and isinstance(record[key], str):
            record[key] = ''
            return record
    return record


def _serialize(record: Union[str, Dict[str, Any]]) -> str:
    if isinstance(record, str):
        return record if record.endswith('\n') else record + '\n'
    return json.dumps(record, ensure_ascii=False, default=str) + '\n'


def write_slice(records: Iterable[Union[str, Dict[str, Any]]],
                start_cursor: int,
                end_cursor: int,
                pad_count: int,
                out_path: str,
                template_record: Optional[Dict[str, Any]] = None,
                dummy_factory: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                source: Optional[str] = None) -> Dict[str, Any]:
    """Write ``records`` (already in plan order) + ``pad_count`` dummy rows to ``out_path``."""
    if end_cursor < start_cursor:
        raise ValueError(f'end_cursor ({end_cursor}) < start_cursor ({start_cursor})')
    if pad_count < 0:
        raise ValueError(f'pad_count must be >= 0, got {pad_count}')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy_line = None
    if pad_count:
        dummy = (dummy_factory or make_dummy_sample)(template_record or {})
        dummy_line = _serialize(dummy)

    written = 0
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as fout:
        for record in records:
            fout.write(_serialize(record))
            written += 1
        for _ in range(pad_count):
            fout.write(dummy_line)
    os.replace(tmp_path, out_path)

    expected = end_cursor - start_cursor
    if written != expected:
        raise RuntimeError(f'Plan slice is short: wrote {written} rows, expected {expected}. '
                           f'Is the plan truncated? ({source})')

    meta = {
        'source': source,
        'slice_path': os.path.abspath(out_path),
        'start_cursor': start_cursor,
        'end_cursor': end_cursor,
        'written': written,
        'pad_count': pad_count,
        'total_rows': written + pad_count,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(out_path)), SLICE_META_FILENAME), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f'elastic_relay: sliced plan[{start_cursor}:{end_cursor}] -> {out_path} '
                f'(rows={written}, pad={pad_count})')
    return meta


def slice_remaining(plan_path: str,
                    start_cursor: int,
                    end_cursor: int,
                    pad_count: int,
                    out_path: str,
                    dummy_factory: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Write ``plan[start:end]`` + ``pad_count`` dummy rows to ``out_path``."""
    template_record = _first_record(plan_path) if pad_count > 0 else {}

    def _iter_lines():
        with open(plan_path, encoding='utf-8') as fin:
            for line in itertools.islice(fin, start_cursor, end_cursor):
                yield line

    return write_slice(
        _iter_lines(),
        start_cursor,
        end_cursor,
        pad_count,
        out_path,
        template_record=template_record,
        dummy_factory=dummy_factory,
        source=os.path.abspath(plan_path))


def count_rows(path: str) -> int:
    with open(path, encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())
