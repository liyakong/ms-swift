# Copyright (c) ModelScope Contributors. All rights reserved.
"""Standalone self-test for ``swift.elastic_relay``.

Runs the pure-logic parts of the feature (plan ordering, deterministic replay,
slicing + padding, calculator, validator, state discovery, runner argv rewriting and
recipe checks) without a GPU, without training, and without modelscope/torch (when they
are missing it installs lightweight stubs for ``swift.utils``).

Usage::

    python tests/elastic_relay_selftest.py
"""
import json
import logging
import os
import shutil
import sys
import tempfile
import types


def _ensure_swift_importable() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import modelscope  # noqa: F401
        return
    except Exception:
        pass

    # modelscope (and therefore `swift.utils`) is unavailable: install a minimal stub.
    for name in [m for m in list(sys.modules) if m == 'swift' or m.startswith('swift.')]:
        sys.modules.pop(name, None)
    fake_swift = types.ModuleType('swift')
    fake_swift.__path__ = [os.path.join(repo_root, 'swift')]
    fake_utils = types.ModuleType('swift.utils')
    fake_utils.get_logger = lambda *args, **kwargs: logging.getLogger('elastic_relay_selftest')
    fake_swift.utils = fake_utils
    sys.modules['swift'] = fake_swift
    sys.modules['swift.utils'] = fake_utils


_ensure_swift_importable()

from swift.elastic_relay import (calculator, constants, grpo, plan_builder, runner, slicer, state,  # noqa: E402
                                 validator)


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _sample(i):
    return {'messages': [{'role': 'user', 'content': f'q{i}'}, {'role': 'assistant', 'content': f'a{i}'}]}


def _test_plan_and_replay(tmp):
    ds = os.path.join(tmp, 'jsons')
    _write_jsonl(os.path.join(ds, 'b.jsonl'), [_sample(100 + i) for i in range(4)])
    _write_jsonl(os.path.join(ds, 'a.jsonl'), [_sample(i) for i in range(6)])

    assert plan_builder.count_raw_samples(ds) == 10
    assert [os.path.basename(p) for p in plan_builder.list_data_files(ds)] == ['a.jsonl', 'b.jsonl']

    meta = plan_builder.build_train_plan(ds, os.path.join(tmp, 'plan'), num_train_epochs=2, seed=3)
    assert meta['dataset_size'] == 10 and meta['total_units'] == 20

    plan_lines = open(meta['plan_path'], encoding='utf-8').read().splitlines()
    rebuilt = [json.dumps(r, ensure_ascii=False) for r in plan_builder.iter_plan_records(ds, 2, 3, 0, 20)]
    assert plan_lines == rebuilt, 'iter_plan_records must reproduce build_train_plan exactly'

    partial = [json.dumps(r, ensure_ascii=False) for r in plan_builder.iter_plan_records(ds, 2, 3, 13, 20)]
    assert partial == plan_lines[13:20]
    return ds, meta, plan_lines


def _test_slicing(tmp, ds, plan_lines):
    slice_path = os.path.join(tmp, 'out', 'seg02.jsonl')
    meta = slicer.write_slice(
        plan_builder.iter_plan_records(ds, 2, 3, 13, 20), 13, 20, 3, slice_path, source=ds)
    assert meta['written'] == 7 and meta['pad_count'] == 3
    rows = open(slice_path, encoding='utf-8').read().splitlines()
    assert len(rows) == 10 and rows[:7] == plan_lines[13:20]
    assert json.loads(rows[7])['messages'][0]['content'] == constants.DUMMY_CONTENT
    assert slicer.count_rows(slice_path) == 10


def _test_file_formats(tmp):
    ds = os.path.join(tmp, 'formats')
    _write_jsonl(os.path.join(ds, 'data.jsonl'), [_sample(1), _sample(2)])
    with open(os.path.join(ds, 'more.json'), 'w', encoding='utf-8') as f:
        json.dump([_sample(3), _sample(4), _sample(5)], f)
    # metadata next to the data must be ignored, not parsed as samples
    with open(os.path.join(ds, 'dataset_infos.json'), 'w', encoding='utf-8') as f:
        json.dump({'splits': ['train']}, f)
    with open(os.path.join(ds, 'dataset_info.jsonl'), 'w', encoding='utf-8') as f:
        f.write('{not json')  # must be skipped by stem, despite the .jsonl suffix
    # a stray output/ subtree (previous run artifacts) must not be scanned as data
    stray = os.path.join(ds, 'output', 'elastic_relay')
    os.makedirs(stray, exist_ok=True)
    with open(os.path.join(ds, 'output', 'elastic_segment.json'), 'w', encoding='utf-8') as f:
        f.write('{not json')
    with open(os.path.join(stray, 'seg01.jsonl'), 'w', encoding='utf-8') as f:
        f.write('{not json')

    files = [os.path.basename(p) for p in plan_builder.list_data_files(ds)]
    assert files == ['data.jsonl', 'more.json'], files
    assert plan_builder.count_raw_samples(ds) == 5
    contents = [r['messages'][0]['content'] for r in plan_builder.load_raw_samples(ds)]
    assert contents == ['q1', 'q2', 'q3', 'q4', 'q5'], contents

    repo_like = os.path.join(tmp, 'repolike')
    os.makedirs(repo_like, exist_ok=True)
    with open(os.path.join(repo_like, 'setup.py'), 'w', encoding='utf-8') as f:
        f.write('')
    try:
        plan_builder.list_data_files(repo_like)
        raise AssertionError('a code-repository directory must be rejected')
    except ValueError as e:
        assert 'code repository' in str(e), e


def _test_dummy_schema():
    alpaca = {'instruction': 'i', 'input': '', 'output': 'o', 'id': 1}
    dummy = slicer.make_dummy_sample(alpaca)
    assert set(dummy.keys()) == set(alpaca.keys()), dummy
    assert dummy['output'] == '' and dummy['id'] == 1

    conv = {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}], 'id': 3}
    dummy2 = slicer.make_dummy_sample(conv)
    assert set(dummy2.keys()) == set(conv.keys()), dummy2
    assert dummy2['messages'] == [{'role': 'user', 'content': constants.DUMMY_CONTENT}]
    assert dummy2['id'] == 3

    assert 'messages' in slicer.make_dummy_sample({})

    # PT (no chat template + loss_scale='all'): assistant + loss:false -> zero loss
    pt = {'messages': [{'role': 'assistant', 'content': 'raw text'}]}
    dpt = slicer.make_dummy_sample(pt, no_chat_template=True)
    assert dpt['messages'] == [{'role': 'assistant', 'content': constants.DUMMY_CONTENT, 'loss': False}], dpt

    # ... and when real messages already carry `loss`, the pad mirrors that exact key set
    # (HF `datasets` infers features then casts every row, so key sets must match).
    ptl = {'messages': [{'role': 'assistant', 'content': 'raw text', 'loss': True}]}
    dptl = slicer.make_dummy_sample(ptl, no_chat_template=True)
    assert set(dptl['messages'][0]) == set(ptl['messages'][0]), (dptl, ptl)
    assert dptl['messages'][0]['loss'] is False

    # swift auto-mapped `text` column must be neutralized too (not in the old fixed list)
    dtext = slicer.make_dummy_sample({'text': 'hello'})
    assert dtext['text'] == '', dtext

    # a custom column mapped to `response` (via --columns) is neutralized when declared
    dcustom = slicer.make_dummy_sample({'my_resp': 'x'}, response_keys=['my_resp'])
    assert dcustom['my_resp'] == '', dcustom

    plan_builder.check_homogeneous([{'a': 1}, {'a': 2}])
    try:
        plan_builder.check_homogeneous([{'a': 1}, {'b': 2}])
        raise AssertionError('heterogeneous rows must raise')
    except ValueError as e:
        assert 'inconsistent columns' in str(e), e


def _test_schema_checks(tmp):
    # within-file mixed columns must be rejected, naming the file
    d1 = os.path.join(tmp, 'mixed_file')
    os.makedirs(d1, exist_ok=True)
    with open(os.path.join(d1, 'bad.jsonl'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(_sample(1)) + '\n')
        f.write(json.dumps(dict(_sample(2), objects=[])) + '\n')
    try:
        plan_builder.load_raw_samples(d1)
        raise AssertionError('a file with mixed columns must be rejected')
    except ValueError as e:
        assert 'bad.jsonl' in str(e) and 'inconsistent columns' in str(e), e

    # cross-file column mismatch must be rejected, naming the files
    d2 = os.path.join(tmp, 'mixed_files')
    os.makedirs(d2, exist_ok=True)
    _write_jsonl(os.path.join(d2, 'a.jsonl'), [_sample(1)])
    _write_jsonl(os.path.join(d2, 'b.jsonl'), [dict(_sample(2), objects=[])])
    try:
        plan_builder.load_raw_samples(d2)
        raise AssertionError('files with different columns must be rejected')
    except ValueError as e:
        assert 'different columns' in str(e), e

    # multiple files, same schema -> fine
    d3 = os.path.join(tmp, 'ok_files')
    os.makedirs(d3, exist_ok=True)
    _write_jsonl(os.path.join(d3, 'a.jsonl'), [_sample(1)])
    _write_jsonl(os.path.join(d3, 'b.jsonl'), [_sample(2)])
    assert len(plan_builder.load_raw_samples(d3)) == 2


def _test_grpo():
    g = grpo.resolve_generation_params(
        per_device_batch=8, world_size=1, num_generations=8, gradient_accumulation_steps=1)
    assert g['generation_batch_size'] == 8 and g['steps_per_generation'] == 1
    assert g['prompts_per_round'] == 1 and g['optimizer_steps_per_round'] == 1

    g2 = grpo.resolve_generation_params(
        per_device_batch=2, world_size=4, num_generations=8, gradient_accumulation_steps=2)
    assert g2['generation_batch_size'] == 16 and g2['prompts_per_round'] == 2
    assert g2['optimizer_steps_per_round'] == 1  # spg(2)*iters(1)/grad_accum(2)

    g3 = grpo.resolve_generation_params(
        per_device_batch=2, world_size=4, num_generations=8,
        gradient_accumulation_steps=2, steps_per_generation=4)
    assert g3['generation_batch_size'] == 32 and g3['prompts_per_round'] == 4
    assert g3['optimizer_steps_per_round'] == 2  # 4*1/2

    # num_iterations increases reuse (more optimizer steps per round)
    g4 = grpo.resolve_generation_params(
        per_device_batch=4, world_size=2, num_generations=8, gradient_accumulation_steps=1, num_iterations=2)
    assert g4['optimizer_steps_per_round'] == 2  # spg(1)*iters(2)/grad_accum(1)

    # invalid: generation_batch_size not divisible by num_generations
    try:
        grpo.resolve_generation_params(per_device_batch=1, world_size=1, num_generations=8,
                                       gradient_accumulation_steps=3)
        raise AssertionError('indivisible generation_batch_size must raise')
    except ValueError:
        pass

    assert grpo.round_aligned_end(3, 20, 4) == 19
    assert grpo.round_aligned_end(0, 10, 3) == 9
    assert grpo.completed_rounds(5, 2) == 2
    assert grpo.committed_prompts(3, 5, 2, 4, 20) == 11
    assert grpo.is_round_boundary(4, 2) is True and grpo.is_round_boundary(5, 2) is False

    # constraint checks
    runner._check_grpo_constraints(['--truncation_strategy', 'left'], {})
    for bad_argv in ([], ['--truncation_strategy', 'delete'],
                     ['--truncation_strategy', 'left', '--dynamic_sample', 'true']):
        try:
            runner._check_grpo_constraints(bad_argv, {})
            raise AssertionError(f'should reject {bad_argv}')
        except NotImplementedError:
            pass


def _test_grpo_runner(tmp):
    ds = os.path.join(tmp, 'grpo_prompts')
    _write_jsonl(os.path.join(ds, 'p0.jsonl'), [_sample(i) for i in range(6)])
    _write_jsonl(os.path.join(ds, 'p1.jsonl'), [_sample(i) for i in range(4)])

    captured = {}
    original_dispatch, original_preflight = runner._dispatch, runner._preflight
    original_env = os.environ.get(constants.CONFIG_ENV)
    original_nproc = os.environ.get('NPROC_PER_NODE')
    runner._preflight = lambda: None

    def _fake_dispatch(dispatch_argv, route=runner.DEFAULT_ROUTE):
        captured['argv'] = list(dispatch_argv)
        captured['route'] = route
        return 0

    runner._dispatch = _fake_dispatch
    os.environ.pop(constants.CONFIG_ENV, None)
    os.environ['NPROC_PER_NODE'] = '1'
    try:
        out = os.path.join(tmp, 'grpo_out')
        argv = ['--model', 'm', '--dataset', ds, '--rlhf_type', 'grpo',
                '--truncation_strategy', 'left', '--per_device_train_batch_size', '8',
                '--num_generations', '8', '--output_dir', out]
        assert runner._prepare(argv, route='rlhf') == 0
        assert captured['route'] == 'rlhf'
        out_argv = captured['argv']
        assert runner._get_flag(out_argv, '--dataset_shuffle') == 'false'
        assert runner._get_flag(out_argv, '--num_train_epochs') == '1'
        assert runner._get_flag(out_argv, '--dataset').endswith('seg01.jsonl')
        cfg = json.loads(os.environ[constants.CONFIG_ENV])
        assert cfg['mode'] == 'grpo' and cfg['prompts_per_round'] == 1
        assert cfg['optimizer_steps_per_round'] == 1
        assert cfg['segment_start_cursor'] == 0 and cfg['segment_end_cursor'] == 10
        rows = open(runner._get_flag(out_argv, '--dataset'), encoding='utf-8').read().splitlines()
        assert len(rows) == 10, len(rows)  # whole rounds, no padding
    finally:
        runner._dispatch = original_dispatch
        runner._preflight = original_preflight
        if original_env is None:
            os.environ.pop(constants.CONFIG_ENV, None)
        else:
            os.environ[constants.CONFIG_ENV] = original_env
        if original_nproc is None:
            os.environ.pop('NPROC_PER_NODE', None)
        else:
            os.environ['NPROC_PER_NODE'] = original_nproc


def _test_list_data_files(tmp):
    # top-level data wins; an output/ subtree is ignored
    d1 = os.path.join(tmp, 'flat_dir')
    os.makedirs(os.path.join(d1, 'output'), exist_ok=True)
    _write_jsonl(os.path.join(d1, 'a.jsonl'), [_sample(1)])
    with open(os.path.join(d1, 'output', 'junk.json'), 'w', encoding='utf-8') as f:
        f.write('{}')
    assert [os.path.basename(p) for p in plan_builder.list_data_files(d1)] == ['a.jsonl']

    # fallback recursion: data only inside jsons/, output/ must be skipped
    d2 = os.path.join(tmp, 'nested_dir')
    os.makedirs(os.path.join(d2, 'jsons'), exist_ok=True)
    os.makedirs(os.path.join(d2, 'output'), exist_ok=True)
    _write_jsonl(os.path.join(d2, 'jsons', 'a.jsonl'), [_sample(1)])
    _write_jsonl(os.path.join(d2, 'jsons', 'b.jsonl'), [_sample(2)])
    with open(os.path.join(d2, 'output', 'junk.json'), 'w', encoding='utf-8') as f:
        f.write('{}')
    rel = [os.path.relpath(p, d2).replace(os.sep, '/') for p in plan_builder.list_data_files(d2)]
    assert rel == ['jsons/a.jsonl', 'jsons/b.jsonl'], rel
    assert len(plan_builder.load_raw_samples(d2)) == 2


def _test_grpo_ref():
    # LoRA GRPO: ref is derived from the base model, no explicit ref_model needed
    r = runner._grpo_ref_recipe(['--rlhf_type', 'grpo'], {})
    assert r['tuner_type'] == 'lora' and r['ref_model'] is None and r['beta'] == 0.04

    # beta == 0 -> no KL, no ref required even for full tuning
    runner._grpo_ref_recipe(['--tuner_type', 'full', '--beta', '0'], {})

    # full + beta != 0 without ref_model -> reject (ref would drift to the checkpoint)
    try:
        runner._grpo_ref_recipe(['--tuner_type', 'full'], {})
        raise AssertionError('full GRPO with beta!=0 must require --ref_model')
    except NotImplementedError:
        pass

    r2 = runner._grpo_ref_recipe(
        ['--tuner_type', 'full', '--beta', '0.04', '--ref_model', '/base',
         '--ref_adapters', 'a', 'b', '--sync_ref_model', 'true'], {})
    assert r2['ref_model'] == '/base' and r2['ref_adapters'] == ['a', 'b']
    assert r2['sync_ref_model'] == 'true'

    # steps_per_generation must NOT be part of the recipe (world-derived)
    generation = grpo.resolve_generation_params(8, 1, num_generations=8, gradient_accumulation_steps=1)
    recipe = runner._grpo_recipe(42, 1, 100, 'sha256:x', generation)
    assert 'steps_per_generation' not in recipe
    assert 'generation_batch_size' not in recipe


def _test_grpo_rollout():
    old = os.environ.get('NPROC_PER_NODE')
    os.environ['NPROC_PER_NODE'] = '2'
    try:
        r = runner._grpo_rollout_recipe([], {})
        assert r['use_vllm'] == 'false' and r['vllm_mode'] is None
        assert r['temperature'] == '' and r['max_completion_length'] == ''

        r2 = runner._grpo_rollout_recipe(
            ['--use_vllm', 'true', '--vllm_mode', 'colocate', '--vllm_tensor_parallel_size', '2',
             '--temperature', '0.9', '--top_p', '0.9', '--max_completion_length', '512'], {})
        assert r2['use_vllm'] == 'true' and r2['vllm_mode'] == 'colocate'
        assert r2['temperature'] == '0.9' and r2['max_completion_length'] == '512'

        try:
            runner._grpo_rollout_recipe(
                ['--use_vllm', 'true', '--vllm_mode', 'colocate', '--vllm_tensor_parallel_size', '3'], {})
            raise AssertionError('colocate TP must divide world_size')
        except ValueError:
            pass
        # non-colocate (server) does not require TP|world
        runner._grpo_rollout_recipe(
            ['--use_vllm', 'true', '--vllm_mode', 'server', '--vllm_tensor_parallel_size', '3'], {})
    finally:
        if old is None:
            os.environ.pop('NPROC_PER_NODE', None)
        else:
            os.environ['NPROC_PER_NODE'] = old


def _test_calculator():
    seg = calculator.plan_segment(13, 20, 16, current_lr=1e-4, checkpoint_path='ckpt')
    assert seg['remaining_units'] == 7 and seg['pad_count'] == 9 and seg['estimated_steps'] == 1
    assert seg['new_lr'] == 1e-4
    assert calculator.plan_segment(20, 20, 16) is None
    scaled = calculator.plan_segment(0, 100, 64, old_global_batch=16, current_lr=1e-4, lr_strategy='sqrt')
    assert scaled['new_lr'] == 2e-4, scaled['new_lr']  # sqrt(4) * 1e-4, attenuation ~1 at cursor 0

    first = calculator.compute_first_segment(20, 2, 2, 4, 1e-4)
    assert first['pad_count'] == 12 and first['estimated_steps'] == 2
    nxt = calculator.compute_next_segment(
        {'committed_cursor': 16, 'total_units': 20, 'segment_global_batch': 16, 'current_lr': 1e-4}, 8, 2, 1,
        lr_strategy='constant')
    assert nxt['start_cursor'] == 16 and nxt['new_lr'] == 1e-4


def _test_validator(tmp, ds, meta):
    slice_path = os.path.join(tmp, 'v', 'seg01.jsonl')
    seg = calculator.compute_first_segment(meta['total_units'], 2, 2, 4, 1e-4)
    slicer.write_slice(plan_builder.iter_plan_records(ds, 2, 3, 0, 20), 0, 20, seg['pad_count'], slice_path, source=ds)
    ok, errors, warns = validator.validate(seg, slice_path, meta, None, num_gpus=2)
    assert ok, errors
    assert any('save_steps' not in w and 'small' in w or 'switch cost' in w for w in warns), warns

    bad = dict(seg, estimated_steps=2)
    ok2, errors2, _ = validator.validate(bad, os.path.join(tmp, 'v', 'missing.jsonl'), meta, None)
    assert not ok2 and errors2


def _test_state(tmp):
    ckpt = os.path.join(tmp, 'run', 'checkpoint-5')
    os.makedirs(ckpt, exist_ok=True)
    payload = {
        'committed_cursor': 16,
        'total_units': 20,
        'segment_global_batch': 16,
        'segment_id': 'seg01',
        'segment_chain': [],
        'committed': True,
        'saved_at': '2026-01-01T00:00:00',
    }
    state.atomic_write_json(os.path.join(ckpt, constants.STATE_FILENAME), payload)
    found = state.find_latest_committed_state([os.path.join(tmp, 'run')])
    assert found is not None and found['committed_cursor'] == 16 and found['checkpoint_path'] == ckpt
    assert state.load_state_file(ckpt)['committed_cursor'] == 16

    # uncommitted states are ignored
    other = os.path.join(tmp, 'run', 'checkpoint-6')
    os.makedirs(other, exist_ok=True)
    state.atomic_write_json(os.path.join(other, constants.STATE_FILENAME), dict(payload, committed=False))
    assert state.find_latest_committed_state([os.path.join(tmp, 'run')])['committed_cursor'] == 16


def _test_runner(tmp, ds, meta):
    raw_fp = plan_builder.dataset_fingerprint(ds)
    ckpt = os.path.join(tmp, 'prev', 'checkpoint-1')
    os.makedirs(ckpt, exist_ok=True)
    with open(os.path.join(ckpt, constants.STATE_FILENAME), 'w', encoding='utf-8') as f:
        json.dump({
            'committed_cursor': 13,
            'committed': True,
            'segment_id': 'seg01',
            'segment_chain': [],
            'seed': 42,
            'num_train_epochs': 3,
            'dataset_size': 10,
            'raw_dataset_fingerprint': raw_fp,
        }, f)

    st = runner._resolve_prev_state(ckpt, {})
    assert st['committed_cursor'] == 13 and st['checkpoint_path'] == ckpt
    assert runner._resolve_prev_state(os.path.join(tmp, 'prev'), {})['committed_cursor'] == 13
    plain = os.path.join(tmp, 'plain')
    os.makedirs(plain, exist_ok=True)
    try:
        runner._resolve_prev_state(plain, {})
        raise AssertionError('missing platform_state.json must raise')
    except RuntimeError:
        pass
    assert runner._resolve_prev_state(plain, {'committed_cursor': 4})['committed_cursor'] == 4

    # argv helpers
    argv = ['--model', 'm', '--dataset', ds, '--resume_from_checkpoint', ckpt, '--lora_rank', '8']
    assert runner._pop_flag(argv, '--dataset') == [ds]
    assert runner._pop_flag(argv, '--resume_from_checkpoint') == [ckpt]
    assert argv == ['--model', 'm', '--lora_rank', '8'], argv

    captured = {}
    original_dispatch, original_env = runner._dispatch, os.environ.get(constants.CONFIG_ENV)
    original_preflight = runner._preflight
    runner._preflight = lambda: None

    def _fake_dispatch(dispatch_argv, route=runner.DEFAULT_ROUTE):
        captured['argv'] = list(dispatch_argv)
        return 0

    runner._dispatch = _fake_dispatch
    old_nproc = os.environ.get('NPROC_PER_NODE')
    os.environ['NPROC_PER_NODE'] = '2'
    try:
        out = os.path.join(tmp, 'cur', 'output')
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, constants.STOP_FILENAME), 'w', encoding='utf-8') as f:
            f.write('stale')
        argv = ['--model', '/mnt/models/m', '--dataset', ds, '--num_train_epochs', '3',
                '--per_device_train_batch_size', '1', '--resume_from_checkpoint', ckpt, '--output_dir', out]
        assert runner._prepare(argv) == 0
        assert not os.path.exists(os.path.join(out, constants.STOP_FILENAME)), 'stale STOP must be removed'
        out_argv = captured['argv']
        assert runner._get_flag(out_argv, '--num_train_epochs') == '1'
        assert runner._get_flag(out_argv, '--model') == '/mnt/models/m'
        assert runner._get_flag(out_argv, '--resume_only_model') == 'true'
        assert runner._get_flag(out_argv, '--ignore_data_skip') == 'true'
        assert runner._get_flag(out_argv, '--dataset_shuffle') == 'false'
        assert runner._get_flag(out_argv, '--dataset').endswith('seg02.jsonl')
        assert runner._get_flag(out_argv, '--loss_type') == 'elastic_cross_entropy'
        assert runner._pop_flag(list(out_argv), '--callbacks') == ['elastic_stop', 'elastic_state']

        cfg = json.loads(os.environ[constants.CONFIG_ENV])
        assert cfg['segment_start_cursor'] == 13 and cfg['total_units'] == 30
        assert cfg['num_gpus'] == 2 and cfg['grad_accum'] == 8 and cfg['segment_id'] == 'seg02'
        rows = open(runner._get_flag(out_argv, '--dataset'), encoding='utf-8').read().splitlines()
        assert len(rows) == 32, len(rows)

        # changing a plan-defining arg must be rejected
        try:
            runner._prepare(['--model', 'm', '--dataset', ds, '--num_train_epochs', '2',
                             '--output_dir', out, '--resume_from_checkpoint', ckpt])
            raise AssertionError('recipe mismatch must raise')
        except RuntimeError as e:
            assert 'recipe changed' in str(e), e

        # missing --output_dir must fail loudly instead of using cwd
        os.environ.pop(constants.CONFIG_ENV, None)
        try:
            runner._prepare(['--model', 'm', '--dataset', ds])
            raise AssertionError('missing --output_dir must raise')
        except ValueError as e:
            assert '--output_dir' in str(e), e
    finally:
        runner._dispatch = original_dispatch
        runner._preflight = original_preflight
        if original_env is None:
            os.environ.pop(constants.CONFIG_ENV, None)
        else:
            os.environ[constants.CONFIG_ENV] = original_env
        if old_nproc is None:
            os.environ.pop('NPROC_PER_NODE', None)
        else:
            os.environ['NPROC_PER_NODE'] = old_nproc


def _test_deepspeed_accum(tmp):
    ds_path = os.path.join(tmp, 'ds_zero2.json')
    with open(ds_path, 'w', encoding='utf-8') as f:
        json.dump({'gradient_accumulation_steps': 4}, f)
    assert runner._deepspeed_grad_accum(['--deepspeed', ds_path], {}) == 4
    assert runner._deepspeed_grad_accum(['--deepspeed', 'zero2'], {}) is None
    auto_path = os.path.join(tmp, 'ds_auto.json')
    with open(auto_path, 'w', encoding='utf-8') as f:
        json.dump({'gradient_accumulation_steps': 'auto'}, f)
    assert runner._deepspeed_grad_accum(['--deepspeed', auto_path], {}) is None
    assert runner._deepspeed_grad_accum([], {'deepspeed': {'gradient_accumulation_steps': 3}}) == 3

    old = os.environ.get('NPROC_PER_NODE')
    os.environ['NPROC_PER_NODE'] = '1'
    try:
        num_gpus, pdb, accum = runner._resolve_parallelism(
            ['--per_device_train_batch_size', '2', '--gradient_accumulation_steps', '1',
             '--deepspeed', ds_path], {})
        assert (num_gpus, pdb, accum) == (1, 2, 4), (num_gpus, pdb, accum)

        # GRPO: DeepSpeed numeric accum must match the args value used for generation accounting
        try:
            runner._resolve_grpo_generation(
                ['--per_device_train_batch_size', '2', '--num_generations', '2', '--deepspeed', ds_path], {})
            raise AssertionError('GRPO + mismatching DeepSpeed accum must raise')
        except NotImplementedError:
            pass
        g = runner._resolve_grpo_generation(
            ['--per_device_train_batch_size', '2', '--num_generations', '2',
             '--gradient_accumulation_steps', '4', '--deepspeed', ds_path], {})
        assert g['gradient_accumulation_steps'] == 4
        # preset ("auto") is always fine
        runner._resolve_grpo_generation(
            ['--per_device_train_batch_size', '8', '--num_generations', '8', '--deepspeed', 'zero2'], {})
    finally:
        if old is None:
            os.environ.pop('NPROC_PER_NODE', None)
        else:
            os.environ['NPROC_PER_NODE'] = old


def _test_unsupported_guards():
    cases = [
        (['a', 'b'], []),
        (['ds#100'], []),
        (['ds'], ['--streaming', 'true']),
        (['ds'], ['--packing', 'true']),
        (['ds'], ['--interleave_prob', '0.5']),
    ]
    for dataset_args, argv in cases:
        try:
            runner._check_unsupported(dataset_args, argv, {})
            raise AssertionError(f'should have rejected dataset_args={dataset_args}, argv={argv}')
        except NotImplementedError:
            pass
    runner._check_unsupported(['ds'], ['--streaming', 'false'], {})
    runner._check_unsupported(['ds'], [], {})


def _test_device_env():
    original = {name: os.environ.get(name) for name in constants.VISIBLE_DEVICE_ENVS}
    try:
        for name in constants.VISIBLE_DEVICE_ENVS:
            os.environ.pop(name, None)
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1,2,3'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        assert constants.visible_device_env()[0] == 'ASCEND_RT_VISIBLE_DEVICES'
        assert constants.count_visible_devices() == 4
        os.environ.pop('ASCEND_RT_VISIBLE_DEVICES')
        assert constants.count_visible_devices() == 8
    finally:
        for name in constants.VISIBLE_DEVICE_ENVS:
            if original[name] is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original[name]


def _test_json_shapes(tmp):
    d = os.path.join(tmp, 'json_shapes')
    os.makedirs(d, exist_ok=True)
    row = {'messages': [{'role': 'assistant', 'content': 'x'}]}
    with open(os.path.join(d, 'arr.json'), 'w', encoding='utf-8') as f:
        json.dump([row, row], f, ensure_ascii=False)
    with open(os.path.join(d, 'one.json'), 'w', encoding='utf-8') as f:
        json.dump({'messages': [{'role': 'assistant', 'content': 'solo'}]}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(d, 'hf.json'), 'w', encoding='utf-8') as f:
        json.dump({'train': [row]}, f, ensure_ascii=False)
    assert len(plan_builder.load_raw_samples(os.path.join(d, 'arr.json'))) == 2
    one = plan_builder.load_raw_samples(os.path.join(d, 'one.json'))
    assert len(one) == 1 and one[0]['messages'][0]['content'] == 'solo', one
    assert len(plan_builder.load_raw_samples(os.path.join(d, 'hf.json'))) == 1


def _test_ensure_message_loss():
    samples = [
        {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]},
        {'messages': [{'role': 'assistant', 'content': 'b', 'loss': False}]},
        {'other': 1},
    ]
    runner._ensure_message_loss(samples)
    assert all(m['loss'] is True for m in samples[0]['messages']), samples[0]
    assert samples[1]['messages'][0]['loss'] is False, samples[1]  # explicit value preserved
    assert 'messages' not in samples[2]


def _test_pt_recipe():
    r = runner._pt_recipe(['--loss_scale', 'all'], {})
    assert r['loss_scale'] == 'all' and r['use_chat_template'] is None
    try:
        runner._verify_recipe(dict(r, mode='sft'),
                              dict(runner._pt_recipe(['--loss_scale', 'last_round'], {}), mode='sft'))
        raise AssertionError('a loss_scale change must invalidate the recipe')
    except RuntimeError:
        pass


def _test_txt(tmp):
    d = os.path.join(tmp, 'txt_data')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'corpus.txt'), 'w', encoding='utf-8') as f:
        f.write('第一行文本\n\nline two\r\nline three\n')
    assert any(p.endswith('corpus.txt') for p in plan_builder.list_data_files(d))
    samples = plan_builder.load_raw_samples(d)
    assert len(samples) == 3, samples  # blank line skipped
    assert all(s['messages'][0]['role'] == 'assistant' for s in samples)
    assert samples[0]['messages'][0]['content'] == '第一行文本'
    assert samples[1]['messages'][0]['content'] == 'line two'
    assert samples[2]['messages'][0]['content'] == 'line three'


def _test_pt_runner(tmp):
    ds = os.path.join(tmp, 'pt_texts')
    rows = [{'messages': [{'role': 'assistant', 'content': f'raw text {i}'}]} for i in range(10)]
    _write_jsonl(os.path.join(ds, 't.jsonl'), rows)

    captured = {}
    original_dispatch, original_preflight = runner._dispatch, runner._preflight
    original_env = os.environ.get(constants.CONFIG_ENV)
    original_nproc = os.environ.get('NPROC_PER_NODE')
    runner._preflight = lambda: None

    def _fake_dispatch(dispatch_argv, route=runner.DEFAULT_ROUTE):
        captured['argv'] = list(dispatch_argv)
        captured['route'] = route
        return 0

    runner._dispatch = _fake_dispatch
    os.environ.pop(constants.CONFIG_ENV, None)
    os.environ['NPROC_PER_NODE'] = '2'
    try:
        out = os.path.join(tmp, 'pt_out')
        argv = ['--model', 'm', '--dataset', ds, '--per_device_train_batch_size', '4', '--output_dir', out]
        assert runner._prepare(argv, route='pt') == 0
        assert captured['route'] == 'pt'
        out_argv = captured['argv']
        assert runner._get_flag(out_argv, '--loss_type') == 'elastic_cross_entropy'
        cfg = json.loads(os.environ[constants.CONFIG_ENV])
        assert cfg['mode'] == 'sft' and cfg.get('rlhf_type') is None
        assert cfg.get('route') == 'pt', cfg
        # 10 texts at gb=16 -> 16 rows; padding rows carry `loss: false` (zero loss under loss_scale='all')
        lines = open(runner._get_flag(out_argv, '--dataset'), encoding='utf-8').read().splitlines()
        assert len(lines) == 16, len(lines)
        for line in lines[:10]:
            msg = json.loads(line)['messages']
            assert msg[0].get('loss') is True, msg  # runner declares `loss` on real rows too
        for line in lines[10:]:
            msg = json.loads(line)['messages']
            assert msg == [{'role': 'assistant', 'content': constants.DUMMY_CONTENT, 'loss': False}], msg
    finally:
        runner._dispatch = original_dispatch
        runner._preflight = original_preflight
        if original_env is None:
            os.environ.pop(constants.CONFIG_ENV, None)
        else:
            os.environ[constants.CONFIG_ENV] = original_env
        if original_nproc is None:
            os.environ.pop('NPROC_PER_NODE', None)
        else:
            os.environ['NPROC_PER_NODE'] = original_nproc


def _test_strip_external_plugins(tmp):
    out = os.path.join(tmp, 'strip_out')
    ckpt = os.path.join(out, 'checkpoint-10')
    os.makedirs(ckpt, exist_ok=True)
    payload = {'external_plugins': ['/swift/elastic_relay/plugin.py'], 'swift_version': '4.5.2'}
    targets = [os.path.join(out, 'args.json'), os.path.join(ckpt, 'args.json')]
    for path in targets:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
    runner._strip_external_plugins(out)
    # unrelated keys must survive; the plugin entries must be gone (portable checkpoints)
    try:
        runner._strip_external_plugins(os.path.join(tmp, 'does_not_exist'))
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f'_strip_external_plugins must tolerate a missing dir: {e}')
    for path in targets:
        data = json.load(open(path, encoding='utf-8'))
        assert data['external_plugins'] == [], data
        assert data['swift_version'] == '4.5.2', data

    # explicit-path form, used by the on_save callback (cleans each checkpoint as it is written)
    explicit = os.path.join(tmp, 'explicit_args.json')
    with open(explicit, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    state.scrub_external_plugins([explicit])
    assert json.load(open(explicit, encoding='utf-8'))['external_plugins'] == []


def main() -> int:
    tmp = tempfile.mkdtemp(prefix='elastic_relay_selftest_')
    try:
        ds, meta, plan_lines = _test_plan_and_replay(tmp)
        _test_slicing(tmp, ds, plan_lines)
        _test_file_formats(tmp)
        _test_dummy_schema()
        _test_schema_checks(tmp)
        _test_grpo()
        _test_grpo_ref()
        _test_grpo_rollout()
        _test_grpo_runner(tmp)
        _test_list_data_files(tmp)
        _test_json_shapes(tmp)
        _test_txt(tmp)
        _test_pt_runner(tmp)
        _test_ensure_message_loss()
        _test_pt_recipe()
        _test_strip_external_plugins(tmp)
        _test_calculator()
        _test_validator(tmp, ds, meta)
        _test_state(tmp)
        _test_runner(tmp, ds, meta)
        _test_deepspeed_accum(tmp)
        _test_unsupported_guards()
        _test_device_env()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('ALL ELASTIC_RELAY SELFTESTS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
