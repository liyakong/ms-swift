#!/usr/bin/env python
"""Convert a generic DPO TSV (`prompt`/`chosen`/`rejected` columns) to swift's DPO jsonl.

swift's DPO/ORPO/CPO trainers expect the chosen conversation under `messages` and the
rejected response under `rejected_response` (a plain string; the template expands it into
`rejected_messages`). A raw `prompt`/`chosen`/`rejected` triple is not recognized and would
be dropped as an unused column.

Output row::

    {"messages": [{"role": "user", "content": <prompt>},
                  {"role": "assistant", "content": <chosen>}],
     "rejected_response": <rejected>}

Usage::

    python scripts/tsv_to_swift_dpo.py zhihu_3k_rlfh.tsv datasets/zhihu_dpo/zhihu_dpo.jsonl
"""
import argparse
import csv
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='input TSV with prompt/chosen/rejected columns')
    ap.add_argument('output', help='output swift DPO jsonl')
    ap.add_argument('--prompt-col', default='prompt')
    ap.add_argument('--chosen-col', default='chosen')
    ap.add_argument('--rejected-col', default='rejected')
    ap.add_argument('--keep-columns', nargs='*', default=[], help='extra columns to copy into the output row')
    ap.add_argument('--drop-equal', action='store_true', default=True,
                    help='drop rows where chosen == rejected (default: on)')
    ap.add_argument('--keep-equal', dest='drop_equal', action='store_false')
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f'input not found: {args.input}')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    kept = skipped_empty = skipped_equal = 0
    tmp = args.output + '.tmp'
    with open(args.input, encoding='utf-8-sig', newline='') as fin, \
            open(tmp, 'w', encoding='utf-8', newline='\n') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        cols = (args.prompt_col, args.chosen_col, args.rejected_col)
        fields = reader.fieldnames or []
        missing = [c for c in cols if c not in fields]
        if missing:
            raise SystemExit(f'columns {missing} not found; file has {reader.fieldnames}')
        for row in reader:
            prompt = (row.get(args.prompt_col) or '').strip()
            chosen = (row.get(args.chosen_col) or '').strip()
            rejected = (row.get(args.rejected_col) or '').strip()
            if not prompt or not chosen or not rejected:
                skipped_empty += 1
                continue
            if args.drop_equal and chosen == rejected:
                skipped_equal += 1
                continue
            record = {
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': chosen},
                ],
                'rejected_response': rejected,
            }
            for col in args.keep_columns:
                if row.get(col) not in (None, ''):
                    record[col] = row[col]
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            kept += 1
    os.replace(tmp, args.output)

    print(f'kept={kept} skipped_empty={skipped_empty} skipped_equal={skipped_equal} -> {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
