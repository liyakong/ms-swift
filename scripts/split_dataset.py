# Copyright (c) ModelScope Contributors. All rights reserved.
"""Split one dataset file into multiple jsonl shards under a directory.

This is useful for testing elastic relay's "dataset is a directory" path: the runner
reads all files under the directory in sorted order, so we write ``part-000.jsonl``,
``part-001.jsonl``, ... with a fixed width to keep the order deterministic.

The split is *sequential*, so concatenating the shards in filename order reproduces the
original sample order exactly.

Usage::

    python scripts/split_dataset.py \
        --input datasets/chinese_law_ft_dataset.json \
        --out-dir datasets/chinese_law_ft_jsonl \
        --num-files 8
"""
import argparse
import json
import os
import sys


def read_records(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        content = f.read()
    if path.endswith('.json') and content.lstrip().startswith('['):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f'Expected a JSON array in {path}')
        return data

    records = []
    for i, line in enumerate(content.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid jsonl at line {i + 1} of {path}: {e}') from e
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Source .json (array) or .jsonl file.')
    parser.add_argument('--out-dir', required=True, help='Directory to write shards into.')
    parser.add_argument('--num-files', type=int, default=8, help='Number of shards (default 8).')
    parser.add_argument('--prefix', default='part', help='Shard filename prefix (default "part").')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing shard files.')
    args = parser.parse_args(argv)

    records = read_records(args.input)
    total = len(records)
    if total == 0:
        raise ValueError(f'No records found in {args.input}')

    os.makedirs(args.out_dir, exist_ok=True)
    width = max(3, len(str(args.num_files - 1)))
    chunk = (total + args.num_files - 1) // args.num_files

    written = 0
    for i in range(args.num_files):
        part = records[i * chunk:(i + 1) * chunk]
        if not part:
            continue
        name = f'{args.prefix}-{i:0{width}d}.jsonl'
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f'{path} exists; pass --overwrite to replace it.')
        with open(path, 'w', encoding='utf-8') as f:
            for record in part:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        written += len(part)
        print(f'{name}: {len(part)} records')

    if written != total:
        raise RuntimeError(f'Wrote {written} records but read {total}; nothing was lost, aborting.')
    print(f'Done: {total} records -> {len(os.listdir(args.out_dir))} files under {args.out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
