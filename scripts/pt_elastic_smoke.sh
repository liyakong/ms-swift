#!/usr/bin/env bash
# One-command pretraining (swift pt) elastic-relay smoke test.
#
# Builds an 8-document subset, runs segment 1, touches STOP after STOP_AFTER seconds, then
# resumes as segment 2 and verifies the cursor math. Set NGPU1 != NGPU2 for the elastic case.
#
# Pretraining has no chat template and supervises every token (`loss_scale='all'`), so the
# padding rows carry `"loss": false` (zero loss); the accounting is the SFT one:
#   committed = min(start + global_step * global_batch, total_units)
#
# Required:
#   MODEL=/path/to/base_model
#   DSET=/path/to/corpus      # .txt (one doc per line), or jsonl/json/parquet
# Optional:
#   NGPU1=1 NGPU2=1  OUT=./pt_elastic_smoke  STOP_AFTER=60
#   PDB=2 GRAD_ACCUM=1  TUNER=lora|full  EXTRA_ARGS="..."
#
#   MODEL=... DSET=... NGPU1=2 NGPU2=4 bash scripts/pt_elastic_smoke.sh
set -euo pipefail

: "${MODEL:?set MODEL}"
: "${DSET:?set DSET}"
NGPU1=${NGPU1:-${NGPU:-1}}
NGPU2=${NGPU2:-$NGPU1}
OUT=${OUT:-$PWD/pt_elastic_smoke}
STOP_AFTER=${STOP_AFTER:-60}
PDB=${PDB:-2}
GRAD_ACCUM=${GRAD_ACCUM:-1}
TUNER=${TUNER:-lora}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$OUT"
SMOKE="$OUT/smoke.txt"
python - "$DSET" "$SMOKE" <<'PY'
import json
import os
import sys

src, out = sys.argv[1], sys.argv[2]
files = [src] if os.path.isfile(src) else []
if not files:
    for root, _, names in os.walk(src):
        for n in sorted(names):
            if n.endswith(('.txt', '.jsonl', '.json', '.parquet')):
                files.append(os.path.join(root, n))
        if files:
            break
    files = sorted(files)
assert files, f'no data file under {src}'
f = files[0]


def texts():
    if f.endswith('.txt'):
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    yield line.rstrip('\r\n')
    elif f.endswith('.parquet'):
        import pyarrow.parquet as pq
        for row in pq.read_table(f).to_pylist():
            for k in ('messages', 'text', 'response', 'content', 'output'):
                v = row.get(k)
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    yield v[-1].get('content', '')
                    break
                if isinstance(v, str):
                    yield v
                    break
    else:
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if 'messages' in row:
                    yield row['messages'][-1]['content']
                else:
                    for k in ('text', 'response', 'content', 'output'):
                        if isinstance(row.get(k), str):
                            yield row[k]
                            break


picked = [t for t in texts() if t][:8]
with open(out, 'w', encoding='utf-8', newline='\n') as o:
    for t in picked:
        o.write(t + '\n')
print(f'smoke corpus: {out} ({len(picked)} lines from {f})')
PY

EXTRA=()
if [ "$TUNER" = "full" ]; then
  EXTRA+=(--tuner_type full)
else
  EXTRA+=(--tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear)
fi
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && EXTRA+=($EXTRA_ARGS)

run_seg() {  # $1 = ngpu, $2 = output_dir, $3... = resume args
  local ngpu="$1" out="$2"; shift 2
  NPROC_PER_NODE="$ngpu" python -m swift.elastic_relay.runner pt \
    --model "$MODEL" \
    --dataset "$SMOKE" \
    "$@" \
    --use_chat_template false --loss_scale all \
    --per_device_train_batch_size "$PDB" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_length 1024 \
    --num_train_epochs 1 --learning_rate 1e-5 \
    --save_steps 1 --logging_steps 1 --save_total_limit 3 \
    --output_dir "$out" \
    --dataset_num_proc 1 --dataloader_num_workers 1 \
    --ignore_args_error true \
    "${EXTRA[@]}"
}

OUT1="$OUT/seg1"
OUT2="$OUT/seg2"
echo "=== segment 1 (NGPU1=$NGPU1) -> $OUT1 ==="
run_seg "$NGPU1" "$OUT1" &
PID=$!
echo "training pid=$PID; touching STOP in ${STOP_AFTER}s ..."
sleep "$STOP_AFTER"
touch "$OUT1/STOP" || true
wait "$PID" || true

echo "=== segment 2 (NGPU2=$NGPU2, resume) -> $OUT2 ==="
run_seg "$NGPU2" "$OUT2" --resume_from_checkpoint "$OUT1"

echo "=== verify platform_state.json ==="
python - "$OUT1" "$OUT2" <<'PY'
import glob
import json
import os
import sys

seen = 0
for root in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(root, '**', 'checkpoint-*', 'platform_state.json'), recursive=True)):
        d = json.load(open(p))
        assert d.get('mode') == 'sft', d
        assert d.get('cursor_unit') == 'sample', d
        gb = d['segment_global_batch']
        step = d['segment_global_step']
        exp = min(d['segment_start_cursor'] + step * gb, d['total_units'])
        assert d['committed_cursor'] == exp, (d['committed_cursor'], exp)
        print(f"  {os.path.basename(os.path.dirname(p))}: committed={d['committed_cursor']} "
              f"start={d['segment_start_cursor']} gb={gb} step={step} ok")
        seen += 1
assert seen > 0, 'no platform_state.json found'
print('SMOKE OK')
PY
