#!/usr/bin/env bash
# One-command DPO elastic-relay smoke test.
#
# It builds an 8-pair preference subset, runs segment 1, touches STOP after STOP_AFTER
# seconds, waits for the graceful stop, then resumes as segment 2 and verifies the cursor
# math. Set NGPU1 != NGPU2 to test the elastic card change.
#
# DPO has no generation rounds, so the accounting is the SFT one:
#   committed = min(start + global_step * global_batch, total_units)
# Padding rows (when the tail is not a multiple of the global batch) are excluded from the
# loss inside DPOTrainer via the chosen-side `loss_mask`.
#
# Required:
#   MODEL=/path/to/base_model
#   DSET=/path/to/preference_data      # jsonl/json/parquet, e.g. messages + rejected_response
# Optional:
#   NGPU1=1 NGPU2=1   cards for segment 1 / segment 2 (set different values for the elastic case)
#   OUT=./dpo_smoke   workdir (default ./dpo_elastic_smoke)
#   STOP_AFTER=60     seconds before touching STOP (default 60)
#   PDB=2 GRAD_ACCUM=1 BETA=0.1
#   TUNER=lora|full  (default lora)   REF_MODEL=/path  (recommended for full)
#   EXTRA_ARGS="..."  any additional swift args
#
# Run from the repo root (the one containing swift/elastic_relay):
#   fixed cards:  MODEL=... DSET=... NGPU1=2 bash scripts/dpo_elastic_smoke.sh
#   elastic:      MODEL=... DSET=... NGPU1=2 NGPU2=4 bash scripts/dpo_elastic_smoke.sh
set -euo pipefail

: "${MODEL:?set MODEL}"
: "${DSET:?set DSET}"
NGPU1=${NGPU1:-${NGPU:-1}}
NGPU2=${NGPU2:-$NGPU1}
OUT=${OUT:-$PWD/dpo_elastic_smoke}
STOP_AFTER=${STOP_AFTER:-60}
PDB=${PDB:-2}
GRAD_ACCUM=${GRAD_ACCUM:-1}
BETA=${BETA:-0.1}
TUNER=${TUNER:-lora}
REF_MODEL=${REF_MODEL:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$OUT"
SMOKE="$OUT/smoke.jsonl"
python - "$DSET" "$SMOKE" <<'PY'
import itertools, json, os, sys
src, out = sys.argv[1], sys.argv[2]
if os.path.isfile(src):
    files = [src]
else:
    files = []
    for root, _, names in os.walk(src):
        for n in sorted(names):
            if n.endswith(('.jsonl', '.json', '.parquet')):
                files.append(os.path.join(root, n))
        if files:
            break
    files = sorted(files)
assert files, f'no data file (.jsonl/.json/.parquet) under {src}'
f = files[0]
if f.endswith('.parquet'):
    import pyarrow.parquet as pq
    rows = pq.read_table(f).slice(0, 8).to_pylist()
else:
    with open(f, encoding='utf-8') as fh:
        rows = [json.loads(l) for l in itertools.islice((l for l in fh if l.strip()), 8)]
with open(out, 'w', encoding='utf-8') as o:
    for r in rows:
        o.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
print(f'smoke dataset: {out} ({len(rows)} rows from {f})')
PY

EXTRA=()
if [ "$TUNER" = "full" ]; then
  EXTRA+=(--tuner_type full)
else
  EXTRA+=(--tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear)
fi
[ -n "$REF_MODEL" ] && EXTRA+=(--ref_model "$REF_MODEL")
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && EXTRA+=($EXTRA_ARGS)

run_seg() {  # $1 = ngpu, $2 = output_dir, $3... = resume args
  local ngpu="$1" out="$2"; shift 2
  NPROC_PER_NODE="$ngpu" python -m swift.elastic_relay.runner rlhf \
    --rlhf_type dpo \
    --model "$MODEL" \
    --dataset "$SMOKE" \
    "$@" \
    --beta "$BETA" \
    --per_device_train_batch_size "$PDB" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_length 2048 \
    --num_train_epochs 1 --learning_rate 1e-6 \
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
import glob, json, os, sys
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
