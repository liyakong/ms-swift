#!/usr/bin/env python
"""Clean a `pip freeze` dump into a valid pip constraints file.

`pip freeze` output is not usable as constraints directly:
  * `-e ...` / VCS lines are rejected by pip ("Constraints cannot have extras"/editables);
  * `pkg @ file://...` / URL lines are rejected;
  * extras (`pkg[extra]==v`) are rejected ("Constraints cannot have extras").

Usage:
    python scripts/clean_constraints.py constraints-tested.txt.raw constraints-tested.txt

Regenerate the raw dump from the verified container:
    docker exec -i <container> pip freeze > constraints-tested.txt.raw
"""
import re
import sys

# Names we intentionally own (installed from local files) or deliberately override
# (librosa 1.0.0 requires numpy>=2.1.0, but the NPU stack pins numpy==1.26.4).
DROP_NAMES = {
    'ms-swift',
    'swanboard',
    'swanlab',
    'librosa',
}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]

    keep = []
    for line in open(src, encoding='utf-8').read().splitlines():
        s = line.strip()
        if not s or s.startswith('-e') or s.startswith('#'):
            continue
        if ' @ ' in s or '://' in s:
            continue
        if not re.match(r'^[A-Za-z0-9][A-Za-z0-9._+-]*(\[[^]]*\])?==\S', s):
            continue
        name, ver = re.split(r'==', s, 1)
        name = re.sub(r'\[[^]]*\]', '', name).strip()
        if name.lower().replace('_', '-') in DROP_NAMES:
            continue
        keep.append(f'{name}=={ver}')

    open(dst, 'w', encoding='utf-8', newline='\n').write('\n'.join(keep) + '\n')
    print(f'kept {len(keep)} lines -> {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
