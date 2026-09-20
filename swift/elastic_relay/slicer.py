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


def _first_record(plan_path: str) -> Dict[str, Any]:
    with open(plan_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def make_dummy_sample(template_record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A padding row that keeps the *same columns* as a real record and produces ~no loss.

    Schema compatibility matters: the slice must be loadable by HF ``datasets``, whose
    feature inference rejects a padding row that introduces/omits columns. So we copy a
    real record and only neutralize its response:

    * messages-style: replace with a user-only conversation (all labels ``-100``);
    * response-style (``output``/``response``/...): empty the response;
    * unknown: keep the record as-is (a bounded duplicate, better than a schema clash).
    """
    record = copy.deepcopy(template_record) if template_record else {}
    if not record:
        return {'messages': [{'role': 'user', 'content': DUMMY_CONTENT}]}
    if 'messages' in record:
        record['messages'] = [{'role': 'user', 'content': DUMMY_CONTENT}]
        return record
    for key in ('output', 'response', 'answer', 'completion', 'target', 'targets', 'solution'):
        if key in record and isinstance(record[key], str):
            record[key] = ''
            return record
    return record


def _serialize(record: Union[str, Dict[str, Any]]) -> str:
    if isinstance(record, str):
        return record if record.endswith('\n') else record + '\n'
    return json.dumps(record, ensure_ascii=False) + '\n'


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
