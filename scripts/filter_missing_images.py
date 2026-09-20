# Copyright (c) ModelScope Contributors. All rights reserved.
"""Drop jsonl rows whose image references point to missing local files.

Handles the common ms-swift multimodal shapes for the image field(s):

* ``"images": "a.jpg"``                       (single path)
* ``"images": ["a.jpg", "b.jpg"]``            (list of paths)
* ``"images": [{"path": "a.jpg", "bytes": ..}]`` (dicts; path and/or bytes)
* a single ``{"path": ...}`` dict
* base64 / ``data:`` / ``http(s)`` references  (not checked, kept)

A row is dropped if **any** referenced local path cannot be resolved. Relative paths are
tried as-is, then under ``--media-root``, then relative to the input file's directory.

Usage::

    python scripts/filter_missing_images.py \
        --inputs /a.jsonl /b.jsonl ... \
        --out-dir /filtered \
        [--media-root /data] \
        [--image-keys images image] \
        [--report-dir /filtered/reports]

    # overwrite in place (keeps a .bak next to each file)
    python scripts/filter_missing_images.py --inputs /a.jsonl --in-place

By default each input ``foo.jsonl`` is written to ``<out-dir>/foo.jsonl`` (or, without
``--out-dir``, to ``foo.filtered.jsonl`` next to the input).
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

IMG_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff')


def _is_remote_or_embedded(path: str) -> bool:
    p = path.strip().lower()
    return p.startswith(('http://', 'https://', 'data:', 'oss://', 's3://'))


def _resolve_candidates(path: str, media_root: Optional[str], base_dir: str) -> List[str]:
    candidates = [path]
    if not os.path.isabs(path):
        if media_root:
            candidates.append(os.path.join(media_root, path))
        candidates.append(os.path.join(base_dir, path))
    return candidates


def _path_exists(path: str, media_root: Optional[str], base_dir: str) -> bool:
    if _is_remote_or_embedded(path):
        return True
    return any(os.path.exists(c) for c in _resolve_candidates(path, media_root, base_dir))


def _iter_image_paths(value: Any) -> Iterable[str]:
    """Yield every *local path* string contained in an images field value."""
    if value is None:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        path = value.get('path')
        if isinstance(path, str) and path.strip():
            yield path
        # {'bytes': ...} has no path to check
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_image_paths(item)


def _record_missing(record: Dict[str, Any], image_keys: List[str], media_root: Optional[str],
                    base_dir: str) -> Optional[str]:
    """Return the first missing local path in the record, or None if everything resolves."""
    for key in image_keys:
        if key not in record:
            continue
        for path in _iter_image_paths(record[key]):
            if not _path_exists(path, media_root, base_dir):
                return path
    return None


def _output_path(src: str, out_dir: Optional[str], suffix: str, in_place: bool) -> str:
    if in_place:
        return src
    name = os.path.basename(src)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, name)
    root, ext = os.path.splitext(src)
    return f'{root}{suffix}{ext}'


def filter_file(src: str, image_keys: List[str], media_root: Optional[str], out_path: str,
                report_path: Optional[str], in_place: bool) -> Tuple[int, int, List[str]]:
    base_dir = os.path.dirname(os.path.abspath(src))
    total = kept = dropped = 0
    samples: List[str] = []

    tmp_path = out_path + '.tmp'
    report_f = open(report_path, 'w', encoding='utf-8') if report_path else None
    try:
        with open(src, encoding='utf-8') as fin, open(tmp_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                stripped = line.strip()
                if not stripped:
                    continue
                total += 1
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as e:
                    raise ValueError(f'Invalid jsonl in {src}: {e}') from e
                missing = _record_missing(record, image_keys, media_root, base_dir)
                if missing is None:
                    fout.write(stripped + '\n')
                    kept += 1
                else:
                    dropped += 1
                    if report_f is not None:
                        report_f.write(json.dumps({'missing': missing, 'record': record},
                                                  ensure_ascii=False, default=str) + '\n')
                    if len(samples) < 5:
                        samples.append(missing)
        if in_place and os.path.exists(src):
            os.replace(src, src + '.bak')
        os.replace(tmp_path, out_path)
    finally:
        if report_f is not None:
            report_f.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return total, kept, dropped, samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True, help='Input .jsonl files.')
    parser.add_argument('--out-dir', default=None, help='Write filtered files here (same basename).')
    parser.add_argument('--suffix', default='.filtered', help='Suffix when --out-dir is not used.')
    parser.add_argument('--in-place', action='store_true', help='Overwrite inputs (backup as .bak).')
    parser.add_argument('--media-root', default=None, help='Root for resolving relative image paths.')
    parser.add_argument('--image-keys', nargs='+', default=['images', 'image'],
                        help='Record keys holding image references (default: images image).')
    parser.add_argument('--report-dir', default=None, help='Write dropped rows (with reason) here.')
    args = parser.parse_args(argv)

    grand_total = grand_kept = grand_dropped = 0
    for src in args.inputs:
        if not os.path.isfile(src):
            print(f'[SKIP] not a file: {src}', file=sys.stderr)
            continue
        out_path = _output_path(src, args.out_dir, args.suffix, args.in_place)
        report_path = None
        if args.report_dir:
            os.makedirs(args.report_dir, exist_ok=True)
            report_path = os.path.join(args.report_dir, os.path.basename(src) + '.dropped.jsonl')
        total, kept, dropped, samples = filter_file(src, args.image_keys, args.media_root, out_path,
                                                    report_path, args.in_place)
        grand_total += total
        grand_kept += kept
        grand_dropped += dropped
        print(f'{src}\n  total={total} kept={kept} dropped={dropped} -> {out_path}')
        for s in samples:
            print(f'    e.g. missing: {s}')

    print(f'\nDONE total={grand_total} kept={grand_kept} dropped={grand_dropped}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
