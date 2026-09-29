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
from typing import Any, Dict, List, Optional, Union

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
# Optional columns that a row may legitimately omit: a multimodal model may be trained on a mix
# of image/video/audio and text-only samples, and only some agent samples carry `tools`. A row
# that omits them is not a schema clash: the missing key is filled with an empty value of the
# same type (see `_unify_resource_columns`). Every other column must match across all rows.
_OPTIONAL_KEYS = ('images', 'videos', 'audios', 'objects', 'tools')


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


def _abs_paths(raw: Union[str, List[str]]) -> Union[str, List[str]]:
    """``os.path.abspath`` for a single path or a list of them (preserves the list order)."""
    if isinstance(raw, (list, tuple)):
        return [os.path.abspath(path) for path in raw]
    return os.path.abspath(raw)


def list_data_files(raw_path: Union[str, List[str]], recursive: bool = False) -> list:
    """List data files for a dataset path (file or directory) or an ordered list of them.

    Files directly under a directory are used first. If none are found there, the scan falls
    back to a recursive search (skipping artifact directories such as ``output/`` and
    ``elastic_relay/``), so a parent directory that only contains a ``jsons/`` subtree still
    works.

    A list of paths (multiple ``--dataset`` values) is expanded **in the given order**; the
    order is defined by us and must never change, because the training plan and therefore
    ``committed_cursor`` are indexed against it.
    """
    if isinstance(raw_path, (list, tuple)):
        files: list = []
        for one in raw_path:
            files.extend(list_data_files(one, recursive=recursive))
        return files
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


def dataset_fingerprint(raw_path: Union[str, List[str]]) -> str:
    """Fingerprint of a dataset path (file or directory) or an ordered list of them."""
    if isinstance(raw_path, (list, tuple)):
        if len(raw_path) == 1:
            # A single path must fingerprint identically as a 1-element list, so a plan/state
            # written before multi-dataset support still validates on resume.
            return dataset_fingerprint(raw_path[0])
        h = hashlib.sha256()
        for i, one in enumerate(raw_path):
            h.update(f'{i}:'.encode('utf-8'))
            h.update(str(one).replace(os.sep, '/').encode('utf-8'))
            h.update(dataset_fingerprint(one).encode('utf-8'))
        return 'sha256:' + h.hexdigest()[:16]
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
            return [{'messages': [{'role': 'assistant', 'content': line.rstrip('\r\n'), 'loss': True}]}
                    for line in f if line.strip()]

    with open(path, encoding='utf-8') as f:
        content = f.read()
    if path.endswith('.json') and content.lstrip()[:1] in '[{':
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None  # not a whole JSON value (e.g. a `.json` file of jsonl): fall back below
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Wrapper-key style, e.g. {"data": [...]} / {"train": [...]} — a common export
            # convention (HuggingFace's json loader would need `field="data"` for this, which
            # swift does not pass). Otherwise treat the object as a single sample.
            for key in ('data', 'train', 'valid', 'validation', 'test'):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            return [data]

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


def _read_raw_samples(raw_path: Union[str, List[str]]) -> list:
    """Load all samples of a dataset path (or an ordered list of them), validating schemas.

    Two levels of validation (see :func:`check_homogeneous`): every individual file must
    have consistent columns, and **all files across all datasets** must share the same
    columns, because the runner merges them into a single slice file that HF ``datasets``
    must be able to load. (swift itself normalizes each dataset independently before
    concatenating; we cannot, since the plan stores raw rows.)
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
        base_keys = _non_resource_keys(base_samples[0])
        for path, samples in per_file:
            if not samples:
                continue
            keys = _non_resource_keys(samples[0])
            if keys != base_keys:
                raise ValueError(
                    f'Datasets have different columns: {base_path} -> {sorted(base_keys)}, '
                    f'{path} -> {sorted(keys)}. All datasets/files must share the same schema because the '
                    'slice merges them into one jsonl; unify the columns upstream. (Only the optional '
                    f'columns {list(_OPTIONAL_KEYS)} may differ.)')

    return _unify_resource_columns([sample for _, samples in per_file for sample in samples])


def load_raw_samples(raw_path: Union[str, List[str]]) -> list:
    """Public alias: load all samples of a dataset path (or list of them) in plan order."""
    return _read_raw_samples(raw_path)


def _non_resource_keys(sample: Dict[str, Any]) -> frozenset:
    """Top-level keys that must match across rows (everything except the optional columns)."""
    return frozenset(key for key in sample.keys() if key not in _OPTIONAL_KEYS)


def _empty_like(value: Any) -> Any:
    """An "empty" value of the same JSON type as ``value`` (for filling a missing column)."""
    if isinstance(value, str):
        return ''
    if isinstance(value, dict):
        return {}
    return []  # list / tuple / None


def _unify_resource_columns(samples: list) -> list:
    """Fill a missing optional column with an empty value of the same type (in place).

    A multimodal model may be trained on a mix of image/video/audio and text-only samples, and
    only some agent rows carry ``tools``; a row that omits these is not a schema clash. We add
    the key with an empty value (``[]`` for lists, ``''`` for strings) so the merged slice stays
    loadable by HF ``datasets`` (the key exists on every row). Only columns already present
    somewhere are added - never new ones.
    """
    defaults = {}
    for sample in samples:
        for key in _OPTIONAL_KEYS:
            if key in sample and key not in defaults:
                defaults[key] = _empty_like(sample[key])
    for sample in samples:
        for key, value in defaults.items():
            if key not in sample:
                sample[key] = value
    return samples


def check_homogeneous(samples: list, source: Optional[str] = None) -> None:
    """Fail loudly when rows differ in columns other than the optional columns.

    A multimodal dataset may mix rows with and without images/videos/audios/objects, and only some
    agent rows carry ``tools``; those optional columns are filled with an empty value where missing
    (see `_unify_resource_columns`) so the merged slice stays loadable. Any other column difference
    is a real schema clash.
    """
    if not samples:
        return
    where = f'File {source}' if source else 'Dataset'
    first = _non_resource_keys(samples[0])
    for i, sample in enumerate(samples[1:], start=1):
        keys = _non_resource_keys(sample)
        if keys != first:
            raise ValueError(
                f'{where} has different columns: row 0 -> {sorted(first)}, '
                f'row {i} -> {sorted(keys)} (missing={sorted(first - keys)}, extra={sorted(keys - first)}). '
                f'Only the optional columns {list(_OPTIONAL_KEYS)} may differ between rows.')


def build_train_plan(raw_path: Union[str, List[str]],
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
        'raw_dataset_path': _abs_paths(raw_path),
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


def count_raw_samples(raw_path: Union[str, List[str]]) -> int:
    """Number of samples across a dataset path (file or directory) or a list of them."""
    return sum(len(_read_file_samples(path)) for path in list_data_files(raw_path))


def iter_plan_records(raw,
                      num_train_epochs: int,
                      seed: int,
                      start_cursor: int,
                      end_cursor: int):
    """Yield ``plan[start_cursor:end_cursor]`` without materializing the whole plan.

    ``raw`` may be a dataset path (str), an ordered list of dataset paths (list[str]), or an
    already-loaded sample list (list[dict], avoids double IO). The ordering is identical to
    :func:`build_train_plan` (same RNG sequence), so a slice produced here is byte-for-byte
    the same sequence a persisted plan would contain.
    """
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        samples = _read_raw_samples(raw)  # list of dataset paths
    elif isinstance(raw, list):
        samples = raw  # already-loaded sample list
    else:
        samples = _read_raw_samples(raw)
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
