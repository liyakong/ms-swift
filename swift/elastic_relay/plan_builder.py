# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training-plan freezing for elastic relay training.

At task-creation time the raw dataset is materialized into a fixed, indexable
sequence (``train_plan.jsonl``). Every later training segment only reads a slice
of that plan, so the *consumption cursor* is decoupled from the trainer's
``global_step``.
"""
import hashlib
import json
import os
import random
from typing import Any, Dict, Optional

from swift.utils import get_logger

from .constants import PLAN_FILENAME, PLAN_META_FILENAME

logger = get_logger()

_DATA_SUFFIXES = ('.jsonl', '.json', '.parquet', '.txt')
# Metadata files that frequently live next to the data in HF/ModelScope dataset folders,
# plus artifacts written by elastic relay itself (never training data). Matched by stem so
# both `dataset_info.json` and `dataset_info.jsonl` are skipped.
_METADATA_STEMS = {
    'dataset_info', 'dataset_infos', 'elastic_segment', 'platform_state', 'slice_meta', 'plan_meta'
}
# Directories never scanned for data (previous-run artifacts / vcs / caches).
_SKIP_DIRNAMES = {'output', 'elastic_relay', '__pycache__', '.git'}


def _is_metadata_file(name: str) -> bool:
    stem, _ = os.path.splitext(name)
    return stem in _METADATA_STEMS


def file_fingerprint(path: str, chunk: int = 1 << 20) -> str:
    """Return a stable content fingerprint for a file (used for plan integrity)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return 'sha256:' + h.hexdigest()[:16]


def list_data_files(raw_path: str, recursive: bool = False) -> list:
    """List data files for a single data file or a dataset directory, in a stable order.

    Files directly under the directory are used first. If none are found there, the scan
    falls back to a recursive search (skipping artifact directories such as ``output/`` and
    ``elastic_relay/``), so a parent directory that only contains a ``jsons/`` subtree still
    works.

    The order is defined by us (sorted relative path) and must never change, because the
    training plan and therefore ``committed_cursor`` are indexed against it.
    """
    if os.path.isfile(raw_path):
        return [raw_path]
    if not os.path.isdir(raw_path):
        raise FileNotFoundError(f'Dataset path does not exist: {raw_path}')

    # Guard against pointing --dataset at the code repository by mistake; the scan would
    # otherwise try to parse example/config json files as training data.
    repo_markers = ('setup.py', 'pyproject.toml', 'setup.cfg', '.git')
    if any(os.path.exists(os.path.join(raw_path, marker)) for marker in repo_markers):
        raise ValueError(f'--dataset points at a code repository, not a dataset directory: {raw_path}. '
                         'Pass the dataset directory/file itself (e.g. datasets/<name>/).')

    files = []

    def _collect(dir_path, recurse):
        for name in sorted(os.listdir(dir_path)):
            full = os.path.join(dir_path, name)
            if os.path.isdir(full):
                if recurse and name not in _SKIP_DIRNAMES:
                    _collect(full, recurse)
                continue
            if _is_metadata_file(name):
                continue
            if name.endswith(_DATA_SUFFIXES):
                files.append(full)

    _collect(raw_path, recursive)
    if not files and not recursive:
        _collect(raw_path, True)  # fallback: data may live in a subdirectory
    if not files:
        raise ValueError(f'No data files ({", ".join(_DATA_SUFFIXES)}) found under: {raw_path}')
    return files


def dataset_fingerprint(raw_path: str) -> str:
    """Fingerprint of a file or a directory (ordered by relative path)."""
    if os.path.isfile(raw_path):
        return file_fingerprint(raw_path)
    h = hashlib.sha256()
    for path in list_data_files(raw_path):
        rel = os.path.relpath(path, raw_path).replace(os.sep, '/')
        h.update(rel.encode('utf-8'))
        h.update(file_fingerprint(path).encode('utf-8'))
    return 'sha256:' + h.hexdigest()[:16]


def _read_file_samples(path: str) -> list:
    """Read one data file: jsonl (one object per line), a JSON array, parquet, or plain text.

    ``.txt`` is one document per line (blank lines skipped), wrapped into the standard
    pretraining ``messages`` schema (``{"messages": [{"role": "assistant", "content": ...}]}``)
    so the padding row can be made loss-free for both SFT and PT.
    """
    if path.endswith('.parquet'):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(f'Reading {path} requires pyarrow (usually installed with `datasets`).') from e
        return pq.read_table(path).to_pylist()

    if path.endswith('.txt'):
        with open(path, encoding='utf-8') as f:
            return [{'messages': [{'role': 'assistant', 'content': line.rstrip('\r\n')}]}
                    for line in f if line.strip()]

    with open(path, encoding='utf-8') as f:
        content = f.read()
    if path.endswith('.json') and content.lstrip().startswith('['):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f'Expected a JSON array of samples in {path}')
        return data

    samples = []
    # Split on '\n' only: str.splitlines() also breaks on U+2028/U+2029/U+0085, which are
    # valid characters inside JSON strings and would corrupt otherwise-valid jsonl rows.
    for i, line in enumerate(content.split('\n')):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid jsonl at line {i + 1} of {path}: {e}') from e
    return samples


def _read_raw_samples(raw_path: str) -> list:
    """Load all samples of a file/directory in plan order, validating schemas.

    Two levels of validation (see :func:`check_homogeneous`): every individual file must
    have consistent columns, and all files must share the same columns, because the runner
    merges them into a single slice file that HF ``datasets`` must be able to load.
    """
    files = list_data_files(raw_path)
    per_file = []
    for path in files:
        samples = _read_file_samples(path)
        check_homogeneous(samples, source=path)
        per_file.append((path, samples))
    if not per_file or not any(samples for _, samples in per_file):
        raise ValueError(f'No valid samples found in {raw_path}')

    if len(per_file) > 1:
        base_path, base_samples = next((p, s) for p, s in per_file if s)
        base_keys = frozenset(base_samples[0].keys())
        for path, samples in per_file:
            if not samples:
                continue
            keys = frozenset(samples[0].keys())
            if keys != base_keys:
                raise ValueError(
                    f'Dataset files have different columns: {base_path} -> {sorted(base_keys)}, '
                    f'{path} -> {sorted(keys)}. All files must share the same schema because the slice '
                    'merges them into one jsonl; unify the columns upstream.')

    return [sample for _, samples in per_file for sample in samples]


def load_raw_samples(raw_path: str) -> list:
    """Public alias: load all samples of a file/directory in plan order."""
    return _read_raw_samples(raw_path)


def check_homogeneous(samples: list, source: Optional[str] = None) -> None:
    """Fail loudly on mixed columns: HF `datasets` cannot load such a json/jsonl file."""
    if not samples:
        return
    where = f'File {source}' if source else 'Dataset'
    first = frozenset(samples[0].keys())
    for i, sample in enumerate(samples[1:], start=1):
        keys = frozenset(sample.keys())
        if keys != first:
            raise ValueError(
                f'{where} has inconsistent columns: row 0 -> {sorted(first)}, '
                f'row {i} -> {sorted(keys)} (missing={sorted(first - keys)}, extra={sorted(keys - first)}). '
                'Every json/jsonl file must have consistent columns (you may split the data across files, '
                'but a single file cannot mix rows with different columns).')


def build_train_plan(raw_path: str,
                     plan_dir: str,
                     num_train_epochs: int = 1,
                     seed: int = 42,
                     plan_filename: str = PLAN_FILENAME) -> Dict[str, Any]:
    """Materialize ``num_train_epochs`` independently-shuffled passes into one plan file.

    The plan is the single source of truth for "which unit is consumed when". It must
    never change after the first segment starts, otherwise ``committed_cursor`` becomes
    meaningless; the fingerprint in ``plan_meta.json`` guards against that.
    """
    if num_train_epochs < 1:
        raise ValueError(f'num_train_epochs must be >= 1, got {num_train_epochs}')
    os.makedirs(plan_dir, exist_ok=True)

    raw_fp = dataset_fingerprint(raw_path)
    samples = _read_raw_samples(raw_path)
    n = len(samples)

    rng = random.Random(seed)
    plan_path = os.path.join(plan_dir, plan_filename)
    tmp_path = plan_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as out:
        for _ in range(num_train_epochs):
            order = list(range(n))
            rng.shuffle(order)
            for idx in order:
                rec = dict(samples[idx])
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    os.replace(tmp_path, plan_path)

    total_units = n * num_train_epochs
    meta = {
        'plan_path': os.path.abspath(plan_path),
        'plan_fingerprint': file_fingerprint(plan_path),
        'raw_dataset_path': os.path.abspath(raw_path),
        'raw_dataset_fingerprint': raw_fp,
        'dataset_size': n,
        'num_train_epochs': num_train_epochs,
        'total_units': total_units,
        'seed': seed,
        'packing': False,
    }
    meta_path = os.path.join(plan_dir, PLAN_META_FILENAME)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f'Built training plan: {plan_path} (n={n}, epochs={num_train_epochs}, '
                f'total_units={total_units}, fp={meta["plan_fingerprint"]})')
    return meta


def load_plan_meta(plan_meta_path: str) -> Dict[str, Any]:
    with open(plan_meta_path, encoding='utf-8') as f:
        return json.load(f)


def count_raw_samples(raw_path: str) -> int:
    """Number of samples across a jsonl file or a dataset directory."""
    return sum(len(_read_file_samples(path)) for path in list_data_files(raw_path))


def iter_plan_records(raw,
                      num_train_epochs: int,
                      seed: int,
                      start_cursor: int,
                      end_cursor: int):
    """Yield ``plan[start_cursor:end_cursor]`` without materializing the whole plan.

    ``raw`` may be a dataset path or an already-loaded sample list (avoids double IO).
    The ordering is identical to :func:`build_train_plan` (same RNG sequence), so a slice
    produced here is byte-for-byte the same sequence a persisted plan would contain.
    """
    samples = raw if isinstance(raw, list) else _read_raw_samples(raw)
    n = len(samples)
    if start_cursor < 0 or end_cursor > n * num_train_epochs:
        raise ValueError(f'cursor out of range: [{start_cursor}, {end_cursor}) vs total {n * num_train_epochs}')
    rng = random.Random(seed)
    pos = 0
    for _ in range(num_train_epochs):
        order = list(range(n))
        rng.shuffle(order)
        lo = max(start_cursor, pos)
        hi = min(end_cursor, pos + n)
        for global_pos in range(lo, hi):
            yield samples[order[global_pos - pos]]
        pos += n
        if pos >= end_cursor:
            break


def verify_plan(plan_meta: Dict[str, Any], plan_meta_path: Optional[str] = None) -> None:
    """Raise if the plan file no longer matches its recorded fingerprint."""
    plan_path = plan_meta['plan_path']
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f'Plan file not found: {plan_path}')
    actual = file_fingerprint(plan_path)
    if actual != plan_meta['plan_fingerprint']:
        raise RuntimeError(f'Training plan has been modified: {plan_path} '
                           f'(expected {plan_meta["plan_fingerprint"]}, got {actual})')
