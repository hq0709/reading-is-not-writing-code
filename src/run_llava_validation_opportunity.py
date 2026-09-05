"""Offline prepare, GPU run, and deterministic replay of the LLaVA validation pilot."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import time

import numpy as np

from src import llava_validation_opportunity as core
from src import run_llava_paired_opportunity as accepted
from src.qwen_consolidation_input_closure import OWNERSHIP_COMMIT, OWNERSHIP_RUN_ID

read_json = accepted.read_json
SUMMARY = 'validation-opportunity-summary'
FIT = {'C': 1., 'max_iter': 2000, 'random_state': 0, 'training_split': 'train',
       'projection_dtype': 'float32', 'standardization_dtype': 'float32',
       'clinical_intercept': 'omitted constant; AUROC only'}


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    accepted.write_json(path, json_value(value))


def full_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{40}', value):
        raise argparse.ArgumentTypeError('source-commit must be a full lowercase 40-character SHA')
    return value


def paths(data_root, out, source_commit):
    full_sha(source_commit)
    if os.environ.get('SOURCE_COMMIT', source_commit) != source_commit:
        raise ValueError('source-commit differs from SOURCE_COMMIT')
    root, out = Path(data_root).resolve(strict=True), Path(out).resolve()
    if not out.is_relative_to(root) or out == root:
        raise ValueError('output must be below the accepted data root')
    return root, out


def inside(root, path, require_exists=True):
    path = Path(path).resolve(strict=require_exists)
    if not path.is_relative_to(root):
        raise ValueError('accepted source path escapes data root')
    return path


def terminal(run, run_id, commit):
    meta = dict(line.split('=', 1) for line in (run / 'metadata.env').read_text(encoding='utf-8').splitlines() if '=' in line)
    expected = {'RUN_ID': run_id, 'SOURCE_COMMIT': commit, 'COMMAND_STATUS': '0',
                'DISPATCHER_STATUS': '0', 'CLEANUP_STATUS': '0', 'ABORT_SIGNAL': 'none'}
    if any(meta.get(k) != v for k, v in expected.items()) or any(
            (run / name).read_text(encoding='utf-8').strip() != '0' for name in ('command_exit_status', 'exit_status')):
        raise ValueError('accepted source terminal receipt mismatch')
    return meta


def protocol():
    return {'gate': 'llava-vislast-validation-opportunity', 'exploratory': True,
            'concepts': list(core.CONCEPTS), 'prompts': core.PROMPTS, 'module': core.MODULE,
            'locus': 'vis.last', 'positions': 'all', 'alpha_mode': 'reltoken', 'doses': [-.25, .25],
            'batch_size': 16, 'dtype': 'bfloat16', 'padding_side': 'left',
            'scoring': 'yes_margin_batch', 'calibration_n': 700, 'write_n': 100,
            'calibration_bootstrap': [2000, 20260914], 'write_bootstrap': [5000, 20260915],
            'asset_run_id': accepted.ASSET_RUN_ID, 'asset_commit': accepted.ASSET_COMMIT,
            'ownership_run_id': OWNERSHIP_RUN_ID, 'ownership_commit': OWNERSHIP_COMMIT,
            'model_id': accepted.MODEL_ID, 'model_revision': accepted.MODEL_REVISION,
            'preprocessing': accepted.protocol_metadata()['preprocessing']}


def validate_bundle(bundle):
    expected = {'names', 'vectors', 'projection', 'scale', 'coefficients', 'locus',
                'raw_dim', 'projection_dim', 'projection_seed', 'C', 'probe_seed'}
    if set(bundle) != expected or bundle['names'].tolist() != list(core.CONCEPTS):
        raise ValueError('accepted probe bundle identity mismatch')
    for key, shape in {'vectors': (6, 1024), 'projection': (1024, 512), 'scale': (512,),
                       'coefficients': (6, 512)}.items():
        core.finite_array(bundle[key], shape, key)
    for key, value in {'locus': 'vis.last', 'raw_dim': 1024, 'projection_dim': 512,
                       'projection_seed': 0, 'C': 1., 'probe_seed': 0}.items():
        if bundle[key].item() != value:
            raise ValueError('accepted probe parameter mismatch')
    if np.any(bundle['scale'] <= 0):
        raise ValueError('accepted scale must be positive')
    vectors = (bundle['projection'].astype(np.float64) @ (bundle['coefficients'] / bundle['scale']).T).T
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.allclose(vectors, bundle['vectors'], atol=1e-6, rtol=1e-5):
        raise ValueError('clinical normal/readout binding mismatch')


def load_sources(root):
    """Read accepted metadata and the small direction bundle; reuse existing receipts."""
    runs = inside(root, root / 'runs')
    first = inside(root, runs / accepted.ASSET_RUN_ID)
    own = inside(root, runs / OWNERSHIP_RUN_ID)
    manifest = inside(root, root / 'datasets/nih-chestxray14/manifest.csv')
    sources = {'first_terminal': terminal(first, accepted.ASSET_RUN_ID, accepted.ASSET_COMMIT),
               'ownership_terminal': terminal(own, OWNERSHIP_RUN_ID, OWNERSHIP_COMMIT),
               'model': accepted.load_model_source(inside(root, root / 'models/huggingface'), runs),
               'manifest': str(manifest)}
    inside(root, sources['model']['model_source'])
    artifacts = first / 'artifacts'
    for key, path in {'probe': artifacts / 'probe/probe.json', 'extract': artifacts / 'activations/meta.json',
                      'hook': artifacts / 'hook-verification.json',
                      'ownership': own / 'artifacts/registered-rows.json'}.items():
        path = inside(root, path)
        sources[key + '_path'] = str(path)
        sources[key] = read_json(path)
    probe = sources['probe']
    expected = {'arch': accepted.ARCH, 'source_commit': accepted.ASSET_COMMIT, 'locus': 'vis.last',
                'module': core.MODULE, 'projection_dim': 512, 'projection_seed': 0,
                'C': 1., 'probe_seed': 0, 'type_columns': ['view_AP', 'sex_M', 'age'],
                'control_seeds': list(range(20)), 'direction_names': list(core.CONCEPTS),
                'n_train': 18212, 'n_test': 5343}
    if any(probe.get(k) != v for k, v in expected.items()):
        raise ValueError('accepted probe metadata mismatch')
    extraction = sources['extract']
    if any(extraction.get(k) != v for k, v in {'arch': accepted.ARCH, 'git_sha': accepted.ASSET_COMMIT,
            'model_source': sources['model']['model_source'], 'model_local_only': True,
            'n_rows': 26229, 'loci': ['vis.last'], 'prompt': core.PROMPTS['Effusion']}.items()):
        raise ValueError('accepted extraction metadata mismatch')
    hook = sources['hook']
    if (hook.get('module') != core.MODULE or hook.get('forward_path_proved') is not True
            or hook.get('alpha_zero_bitwise_noop') is not True):
        raise ValueError('accepted consumed-module hook evidence mismatch')
    directions = inside(root, artifacts / 'probe/directions.npz')
    if Path(probe['directions']).resolve() != directions or not probe.get('directions_sha256'):
        raise ValueError('accepted direction receipt identity mismatch')
    sources['directions_path'] = str(directions)
    with np.load(directions, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    validate_bundle(bundle)
    with manifest.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 26229 or len(sources['ownership']['row_ids']) != 400:
        raise ValueError('accepted manifest/ownership population mismatch')
    allocation = core.allocate(rows, accepted.VALIDATION_ROW_IDS, sources['ownership'])
    core.validate_feasibility(allocation)
    for row in allocation['calibration'] + allocation['write'] + allocation['preflight']:
        inside(root, row['image_path'], require_exists=False)
    types = core.build_types(rows)
    if len(set(types)) != 39 or len(probe['controls']) != 20:
        raise ValueError('accepted 39-type control population mismatch')
    for seed, control in enumerate(probe['controls']):
        _, assignment, usable = core.control_labels(types, np.random.default_rng(seed))
        if not usable or control.get('seed') != seed or control.get('assignment') != assignment:
            raise ValueError('accepted type-control assignment mismatch')
    return sources, bundle, rows, allocation


def prepare(data_root, out, source_commit):
    root, out = paths(data_root, out, source_commit)
    if any((out / name).exists() for name in ('prepared.json', 'prepared.npz', 'cohort.json')):
        raise ValueError('prepare artifacts already exist')
    sources, bundle, rows, allocation = load_sources(root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'cohort.json', allocation)
    from src.first_gate import load_single_locus
    ids, raw = load_single_locus(Path(sources['extract_path']).parent)
    by_id = {r['row_id']: r for r in rows}
    if len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise ValueError('pooled activation/manifest coverage mismatch')
    if raw.dtype != np.float16:
        raise ValueError('accepted pooled activations must be float16')
    aligned = [by_id[i] for i in ids]
    arrays = core.prepare_readers(raw, aligned, [r['row_id'] for r in allocation['calibration']],
                                  bundle, sources['probe']['controls'])
    random, sham = core.make_directions(bundle['vectors'])
    arrays.update(bundle)
    arrays.update(random_vectors=random, sham_vectors=sham,
                  calibration_labels=core.labels_for(allocation['calibration']),
                  write_labels=core.labels_for(allocation['write']),
                  calibration_indices=core.bootstrap_indices(700, 2000, core.CAL_SEED),
                  write_indices=core.bootstrap_indices(100, 5000, core.WRITE_SEED))
    for phase in ('calibration', 'write'):
        for name in ('row_id', 'patient_id'):
            arrays[phase + '_' + name] = np.asarray([str(r[name]) for r in allocation[phase]])
    arrays.update(core.reader_statistics(arrays))
    meta = {'source_commit': source_commit, 'protocol': protocol(), 'sources': sources,
            'cohort': allocation, 'control_assignments': sources['probe']['controls'],
            'fit': FIT}
    np.savez_compressed(out / 'prepared.npz', **arrays)
    write_json(out / 'prepared.json', meta)
    return meta


def load_prepared(root, out, source_commit):
    meta = read_json(out / 'prepared.json')
    allocation = read_json(out / 'cohort.json')
    sources, bundle, rows, expected_allocation = load_sources(root)
    if (meta.get('source_commit') != source_commit or meta.get('protocol') != protocol()
            or meta.get('fit') != FIT
            or meta.get('sources') != sources or allocation != expected_allocation or meta.get('cohort') != allocation
            or meta.get('control_assignments') != sources['probe']['controls']):
        raise ValueError('prepared source/cohort/probe identity mismatch')
    with np.load(out / 'prepared.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, value in bundle.items():
        if not np.array_equal(arrays.get(key), value):
            raise ValueError(f'prepared clinical probe mismatch: {key}')
    for phase, n in (('calibration', 700), ('write', 100)):
        if not np.array_equal(arrays[phase + '_labels'], core.labels_for(allocation[phase])):
            raise ValueError('prepared label identity mismatch')
        for name in ('row_id', 'patient_id'):
            if arrays[phase + '_' + name].tolist() != [str(r[name]) for r in allocation[phase]]:
                raise ValueError('prepared patient/row order mismatch')
    core.validate_indices(arrays['calibration_indices'], 700, 2000, core.CAL_SEED)
    core.validate_indices(arrays['write_indices'], 100, 5000, core.WRITE_SEED)
    random, sham = core.make_directions(bundle['vectors'])
    for name, expected in (('random_vectors', random), ('sham_vectors', sham)):
        if not np.array_equal(arrays[name], expected):
            raise ValueError('prepared random/sham identity mismatch')
    features = core.finite_array(arrays['calibration_features'], (700, 512), 'calibration features')
    mean = core.finite_array(arrays['train_mean'], (512,), 'training mean')
    projected = core.finite_array(arrays['calibration_projected'], (700, 512), 'calibration projected activations')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.n_features_in_ = mean, bundle['scale'], 512
    if not np.array_equal(features, scaler.transform(projected)):
        raise ValueError('prepared training mean/scale binding mismatch')
    coef = core.finite_array(arrays['control_coefficients'], (20, 512), 'control coefficients')
    intercept = core.finite_array(arrays['control_intercepts'], (20,), 'control intercepts')
    for key, expected in (('clinical_scores', (features @ bundle['coefficients'].T).T),
                          ('control_scores', (features @ coef.T + intercept).T)):
        if not np.allclose(arrays[key], expected, rtol=1e-12, atol=1e-12):
            raise ValueError('prepared score/parameter binding mismatch')
    by_id = {r['row_id']: r for r in rows}
    train_ids = arrays['train_row_ids'].tolist()
    if len(set(train_ids)) != len(train_ids) or set(train_ids) != {r['row_id'] for r in rows if r['split'] == 'train'}:
        raise ValueError('control fit training population mismatch')
    for phase, records in (('train', [by_id[i] for i in train_ids]), ('calibration', allocation['calibration'])):
        types = core.build_types(records)
        if not np.array_equal(types, arrays[phase + '_types']):
            raise ValueError('control fit type identity mismatch')
        expected = np.asarray([[r['assignment'][t] for t in types] for r in meta['control_assignments']])
        if not np.array_equal(expected, arrays['control_train_labels' if phase == 'train' else 'control_labels']):
            raise ValueError('prepared control assignment/label mismatch')
    for key, expected in core.reader_statistics(arrays).items():
        if not np.array_equal(arrays[key], expected, equal_nan=True):
            raise ValueError('prepared joint reader bootstrap mismatch')
    return meta, allocation, arrays


def budget(pilot_seconds, elapsed_seconds):
    values = [pilot_seconds, elapsed_seconds]
    if not np.isfinite(values).all() or pilot_seconds <= 0 or elapsed_seconds < pilot_seconds:
        raise ValueError('invalid timing pilot measurements')
    projected = elapsed_seconds + 31200 * pilot_seconds / 512 + 180
    return {'pilot_equivalents': 512, 'pilot_seconds': pilot_seconds, 'elapsed_seconds': elapsed_seconds,
            'worst_case_outcomes': 31200, 'projected_seconds': projected, 'limit_seconds': 4500,
            'passed': bool(projected <= 4500)}


def load_scorer(source, gpu):
    from src.gpu_env import bind_gpu
    bind_gpu(gpu)
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from src.registry import REGISTRY
    from src.loci import as_hidden, loci_for
    from src.intervene import Steerer, YES_WORDS, NO_WORDS, yes_margin_batch
    from src.qwen_answer_encoding import singleton_ids

    arch = REGISTRY[accepted.ARCH]
    if (arch.hf_id != accepted.MODEL_ID or arch.family != 'llava' or arch.processor_kwargs != {}
            or torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name(0)):
        raise ValueError('pilot requires the accepted registry and one A100')
    processor = AutoProcessor.from_pretrained(source['model_source'], local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    model = AutoModelForImageTextToText.from_pretrained(source['model_source'], dtype=torch.bfloat16,
        device_map='cuda:0', local_files_only=True).eval()
    accepted.validate_processor(processor, model.config)
    tokens = singleton_ids(processor.tokenizer, YES_WORDS, NO_WORDS)
    locus = next(l for l in loci_for(arch) if l.name == 'vis.last')
    modules = dict(model.named_modules())
    if locus.module != core.MODULE or locus.positions != 'all' or locus.module not in modules or arch.connector not in modules:
        raise ValueError('registered consumed-module binding mismatch')

    def score(prompt, rows=(), vector=None, alpha=0., inspect=False):
        content = ([{'type': 'image'}] if rows else []) + [{'type': 'text', 'text': prompt}]
        text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                             tokenize=False, add_generation_prompt=True)
        images = []
        for row in rows:
            with Image.open(row['image_path']) as image:
                images.append(image.convert('RGB'))
        inputs = processor(text=[text] * max(1, len(rows)), return_tensors='pt', padding=True,
                           **({'images': images} if images else {})).to('cuda:0')
        steerer = Steerer(model, locus.module, positions=locus.positions, mode='reltoken')
        steerer.vec, steerer.alpha = vector, alpha
        captured, handles = {}, []

        def target_hook(_module, _inputs, output):
            hidden = as_hidden(output)
            captured['target'] = hidden.detach().float().cpu()

        def connector_hook(_module, inputs):
            captured['forwarded'] = inputs[0].detach().float().cpu()

        with steerer, torch.inference_mode():
            try:
                if inspect:
                    handles.append(modules[locus.module].register_forward_hook(target_hook))
                    handles.append(modules[arch.connector].register_forward_pre_hook(connector_hook))
                logits = model(**inputs).logits[:, -1:, :]
                margin = np.asarray(yes_margin_batch(logits, processor.tokenizer), dtype=np.float64)
                core.finite_array(margin, (max(1, len(rows)),), 'native margins')
                if inspect:
                    captured['logits'] = logits.detach().float().cpu().numpy()
            finally:
                for handle in handles:
                    handle.remove()
        if inspect:
            target, forwarded = captured['target'], captured['forwarded']
            if target.shape[1:] != (577, 1024) or not torch.equal(target[:, 1:, :], forwarded):
                raise ValueError('no-CLS consumed tensor does not match steered locus')
            captured['forwarded'] = forwarded.numpy()
            del captured['target']
            return margin, captured
        return margin

    return score, tokens, {'gpu_name': torch.cuda.get_device_name(0), 'gpu_count': 1,
                           'chat_template': processor.chat_template, 'module': locus.module,
                           'positions': locus.positions, 'model_source': source['model_source']}


def vector_for(arrays, question, direction):
    index = list(core.CONCEPTS).index(question)
    if direction == 'baseline':
        return None
    if direction == 'clinical':
        return arrays['vectors'][index]
    if direction == 'sham':
        return arrays['sham_vectors'][index]
    return arrays['random_vectors'][int(direction.removeprefix('random'))]


def validate_preflight(flight, source_commit, source):
    if (flight.get('source_commit') != source_commit or flight.get('validation_row_ids') != accepted.VALIDATION_ROW_IDS
            or flight.get('model_source') != source['model_source'] or flight.get('module') != core.MODULE
            or flight.get('passed') is not True):
        raise ValueError('preflight source/identity/status mismatch')
    ids = flight.get('token_ids', [])
    if (len(ids) != 2 or not all(ids) or set(ids[0]) & set(ids[1])
            or any(type(i) is not int or i < 0 for group in ids for i in group)):
        raise ValueError('invalid singleton token groups')
    cases = accepted.qwen.mapping_cases('Consolidation')
    if len(flight.get('mapping_cases', [])) != 4:
        raise ValueError('incomplete generic mapping cases')
    for actual, expected in zip(flight['mapping_cases'], cases, strict=True):
        if any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError('generic mapping identity mismatch')
        margin = float(actual['raw_margin'])
        if not np.isfinite(margin) or margin * (2 * expected['expected_present'] - 1) <= 0:
            raise ValueError('generic mapping sign failed')
    clean = core.finite_array(flight['clean_margin'], (16,), 'preflight clean margin')
    repeat = core.finite_array(flight['repeat_margin'], (16,), 'preflight repeat margin')
    if not np.array_equal(clean, repeat):
        raise ValueError('preflight clean repeat mismatch')
    for flag in ('zero_exact', 'clean_logits_repeat_exact', 'forwarded_no_cls_exact'):
        if flight.get(flag) is not True:
            raise ValueError('preflight exact behavior failed')
    for key in ('logit_max_abs_delta', 'forwarded_max_abs_delta'):
        value = flight[key]
        if not np.isfinite(value) or value <= 0:
            raise ValueError('nonzero finite downstream effect required')
    timing = flight['throughput']
    if timing != budget(timing['pilot_seconds'], timing['elapsed_seconds']) or not timing['passed']:
        raise ValueError('75-minute timing preflight failed')
    runtime = flight['runtime']
    if (runtime.get('gpu_count') != 1 or 'A100' not in runtime.get('gpu_name', '')
            or runtime.get('module') != core.MODULE or runtime.get('positions') != 'all'
            or runtime.get('model_source') != source['model_source'] or not runtime.get('chat_template')):
        raise ValueError('preflight native runtime mismatch')


def preflight(score, tokens, runtime, allocation, arrays, started, source_commit, source, flight):
    rows = allocation['preflight']
    question = core.CONCEPTS[0]
    prompt = core.PROMPTS[question]
    clean, first = score(prompt, rows, inspect=True)
    zero, zero_full = score(prompt, rows, arrays['vectors'][0], 0., inspect=True)
    changed, changed_full = score(prompt, rows, arrays['vectors'][0], .25, inspect=True)
    repeat, repeat_full = score(prompt, rows, inspect=True)
    for data in (first, zero_full, changed_full, repeat_full):
        if not all(np.isfinite(data[k]).all() for k in ('logits', 'forwarded')):
            raise ValueError('preflight full tensors must be finite')
    flight.update({'source_commit': source_commit, 'model_source': source['model_source'], 'module': core.MODULE,
              'validation_row_ids': accepted.VALIDATION_ROW_IDS, 'token_ids': tokens, 'runtime': runtime,
              'clean_margin': clean, 'repeat_margin': repeat, 'changed_margin': changed,
              'zero_exact': bool(np.array_equal(clean, zero) and all(np.array_equal(first[k], zero_full[k]) for k in first)),
              'clean_logits_repeat_exact': bool(all(np.array_equal(first[k], repeat_full[k]) for k in first)),
              'forwarded_no_cls_exact': True,
              'logit_max_abs_delta': float(np.max(np.abs(first['logits'] - changed_full['logits']))),
              'forwarded_max_abs_delta': float(np.max(np.abs(first['forwarded'] - changed_full['forwarded']))),
              'mapping_cases': []})
    for case in accepted.qwen.mapping_cases('Consolidation'):
        flight['mapping_cases'].append({**case, 'raw_margin': float(score(case['prompt'])[0])})
    pilot_start = time.perf_counter()
    pilot_conditions = [('baseline', 0.), ('clinical', -.25), ('clinical', .25),
                        ('random0', -.25), ('random0', .25), ('sham', -.25), ('sham', .25)]
    for i in range(32):
        c = core.CONCEPTS[i % 6]
        d, a = pilot_conditions[i % len(pilot_conditions)]
        score(core.PROMPTS[c], rows, vector_for(arrays, c, d), a)
    flight['throughput'] = budget(time.perf_counter() - pilot_start, time.perf_counter() - started)
    flight['passed'] = flight['throughput']['passed']
    validate_preflight(flight, source_commit, source)
    return flight


def score_grid(score, path, allocation, arrays, source_commit, phase, questions):
    conditions = (core.CONDITIONS[0],) if phase == 'calibration' else core.CONDITIONS
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=core.FIELDS)
        writer.writeheader()
        for c in questions:
            for d, alpha in conditions:
                for start in range(0, len(allocation[phase]), 16):
                    rows = allocation[phase][start:start + 16]
                    margins = core.finite_array(score(core.PROMPTS[c], rows, vector_for(arrays, c, d), alpha),
                                                (len(rows),), 'batch margins')
                    for offset, (row, margin) in enumerate(zip(rows, margins, strict=True)):
                        writer.writerow({'source_commit': source_commit, 'phase': phase, 'question': c,
                                         'prompt': core.PROMPTS[c], 'patient_index': start + offset,
                                         'patient_id': row['patient_id'], 'row_id': row['row_id'],
                                         'image_path': row['image_path'], 'label': row[c],
                                         'direction': d, 'alpha': alpha, 'raw_margin': margin,
                                         'probability': float(core.sigmoid(margin))})
                    stream.flush()
    with temporary.open(newline='', encoding='utf-8') as stream:
        result = core.validate_grid(csv.DictReader(stream), allocation, source_commit, phase, questions)
    temporary.replace(path)
    return result


def run(data_root, out, source_commit, gpu, preflight_only=False):
    root, out = paths(data_root, out, source_commit)
    if gpu < 0:
        raise ValueError('GPU index must be nonnegative')
    if any((out / name).exists() for name in ('preflight.json', 'calibration.csv', 'qualification.json', 'per-image.csv')):
        raise ValueError('GPU run artifacts already exist')
    meta, allocation, arrays = load_prepared(root, out, source_commit)
    started = time.perf_counter()
    flight = {'passed': False, 'source_commit': source_commit, 'validation_row_ids': accepted.VALIDATION_ROW_IDS}
    try:
        score, tokens, runtime = load_scorer(meta['sources']['model'], gpu)
        preflight(score, tokens, runtime, allocation, arrays, started, source_commit, meta['sources']['model'], flight)
    except Exception as error:
        flight.update(passed=False, error=f'{type(error).__name__}: {error}')
        write_json(out / 'preflight.json', flight)
        raise
    write_json(out / 'preflight.json', flight)
    if preflight_only:
        return flight
    calibration = score_grid(score, out / 'calibration.csv', allocation, arrays, source_commit,
                             'calibration', list(core.CONCEPTS))[:, 0]
    qualification, _ = core.calibration_statistics(arrays, calibration)
    qualification['source_commit'] = source_commit
    write_json(out / 'qualification.json', qualification)
    score_grid(score, out / 'per-image.csv', allocation, arrays, source_commit,
               'write', qualification['qualified_questions'])
    return qualification


def summarize(data_root, out, source_commit):
    root, out = paths(data_root, out, source_commit)
    meta, allocation, prepared = load_prepared(root, out, source_commit)
    flight = read_json(out / 'preflight.json')
    validate_preflight(flight, source_commit, meta['sources']['model'])
    with (out / 'calibration.csv').open(newline='', encoding='utf-8') as stream:
        calibration = core.validate_grid(csv.DictReader(stream), allocation, source_commit,
                                         'calibration', list(core.CONCEPTS))[:, 0]
    qualification, arrays = core.calibration_statistics(prepared, calibration)
    qualification['source_commit'] = source_commit
    if read_json(out / 'qualification.json') != json_value(qualification):
        raise ValueError('persisted qualification differs from recomputed calibration decision')
    questions = qualification['qualified_questions']
    with (out / 'per-image.csv').open(newline='', encoding='utf-8') as stream:
        margins = core.validate_grid(csv.DictReader(stream), allocation, source_commit, 'write', questions)
    outcomes, eligible = {}, []
    for i, c in enumerate(questions):
        stats, boot = core.write_statistics(margins[i], prepared['write_labels'][list(core.CONCEPTS).index(c)],
                                            prepared['write_indices'])
        outcomes[c] = stats
        arrays.update({c + '__' + key: value for key, value in boot.items()})
        if stats['opportunity_pass']:
            eligible.append(c)
    route = 'test_cohort_competition_registration' if eligible else 'evidence_synthesis'
    result = {'source_commit': source_commit, 'status': 'OBSERVED', 'exploratory': True,
              'qualification': qualification, 'K': len(questions), 'n_outcomes': 4200 + margins.size,
              'write': outcomes, 'eligible_questions': eligible, 'route': route,
              'protocol': protocol(), 'cohort': allocation, 'sources': meta['sources'],
              'preflight': flight, 'interval_role': 'descriptive validation uncertainty; exploratory screening'}
    arrays.update({key: value for key, value in prepared.items() if key not in arrays})
    arrays.update(write_margin=margins, qualified_questions=np.asarray(questions, dtype=str),
                  eligible_questions=np.asarray(eligible, dtype=str), route=np.asarray(route),
                  source_commit=np.asarray(source_commit))
    np.savez_compressed(out / (SUMMARY + '.npz'), **arrays)
    write_json(out / (SUMMARY + '.json'), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'run', 'summarize'):
        sub = commands.add_parser(command)
        for name in ('data-root', 'out'):
            sub.add_argument('--' + name, type=Path, required=True)
        sub.add_argument('--source-commit', type=full_sha, required=True)
        if command == 'run':
            sub.add_argument('--gpu', type=int, required=True)
            sub.add_argument('--preflight-only', action='store_true')
    args = vars(parser.parse_args(argv))
    return {'prepare': prepare, 'run': run, 'summarize': summarize}[args.pop('command')](**args)


if __name__ == '__main__':
    main()
