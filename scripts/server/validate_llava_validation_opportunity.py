"""Independent terminal replay for the registered LLaVA validation screen."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


CONCEPTS = ['Effusion', 'Atelectasis', 'Pneumothorax', 'Cardiomegaly', 'Mass', 'Nodule']
PREFLIGHT = ['00010613_005', '00007833_003', '00003974_003', '00009259_000',
             '00005977_010', '00009349_012', '00003459_018', '00003933_000',
             '00008451_011', '00010722_006', '00005496_000', '00005759_028',
             '00010294_050', '00000557_000', '00009107_007', '00003393_010']
CONDITIONS = [('baseline', 0.)] + [(d, a) for d in ['clinical'] + ['random' + str(i) for i in range(20)] + ['sham'] for a in [-.25, .25]]
MAX_ERROR = 0.


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text())


def same(actual, expected, name):
    global MAX_ERROR
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(actual.shape == expected.shape, name + ' shape')
    check(np.array_equal(np.isnan(actual), np.isnan(expected)), name + ' NaN locations')
    finite = np.isfinite(expected)
    check(np.isfinite(actual[finite]).all(), name + ' finite')
    error = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.
    MAX_ERROR = max(MAX_ERROR, error)
    check(error < 2e-10, name + ' numerical discrepancy ' + str(error))


def probability(m):
    z = np.exp(-np.abs(m))
    return np.where(m >= 0, 1 / (1 + z), z / (1 + z))


def weights(indices, n):
    return np.array([np.bincount(row, minlength=n) for row in indices])


def auc(scores, labels, count):
    order = np.argsort(scores, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    positive = np.add.reduceat(count[:, order] * labels[order], boundaries, axis=1)
    negative = np.add.reduceat(count[:, order] * (1 - labels[order]), boundaries, axis=1)
    product = positive.sum(1) * negative.sum(1)
    numerator = np.sum(positive * (np.cumsum(negative, axis=1) - .5 * negative), axis=1)
    return np.divide(numerator, product, out=np.full(len(count), np.nan), where=product != 0)


def lower(draws, percentile):
    finite = draws[np.isfinite(draws)]
    return np.percentile(finite, percentile, method='linear') if len(finite) else np.nan


def csv_grid(path, rows, questions, phase, commit, prompts):
    conditions = CONDITIONS[:1] if phase == 'calibration' else CONDITIONS
    with path.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    check(len(records) == len(rows) * len(conditions) * len(questions), phase + ' count')
    margins = np.empty((len(questions), len(conditions), len(rows)))
    cursor = 0
    for qi, q in enumerate(questions):
        for di, (direction, alpha) in enumerate(conditions):
            for i, row in enumerate(rows):
                record = records[cursor]
                cursor += 1
                expected = dict(source_commit=commit, phase=phase, question=q, prompt=prompts[q],
                                patient_index=str(i), patient_id=str(row['patient_id']), row_id=row['row_id'],
                                image_path=row['image_path'], label=str(row[q]), direction=direction)
                check(all(record.get(k) == v for k, v in expected.items()), phase + ' ordered identity')
                margin, p, dose = (float(record[k]) for k in ['raw_margin', 'probability', 'alpha'])
                check(np.isfinite([margin, p, dose]).all() and dose == alpha, phase + ' finite/dose')
                same(p, probability(np.asarray(margin)), phase + ' sigmoid')
                margins[qi, di, i] = margin
    return margins


def verify(run_id, commit):
    root = Path('/home/qingchan/data/concept-flow')
    run = root / 'runs' / run_id
    out = run / 'artifacts'
    check(run.parent == root / 'runs' and len(commit) == 40, 'target identity')
    metadata = dict(line.split('=', 1) for line in (run / 'metadata.env').read_text().splitlines() if '=' in line)
    expected = dict(RUN_ID=run_id, SOURCE_COMMIT=commit, COMMAND_STATUS='0', DISPATCHER_STATUS='0', CLEANUP_STATUS='0', ABORT_SIGNAL='none', GPU_COUNT='1')
    check(all(metadata.get(k) == v for k, v in expected.items()), 'terminal metadata')
    check(all((run / name).read_text().strip() == '0' for name in ['command_exit_status', 'exit_status']), 'terminal status')
    inventory = {}
    for line in (run / 'checksums.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        inventory[name.strip().lstrip('*').removeprefix('./')] = digest
    files = ['metadata.env', 'command_exit_status', 'exit_status'] + ['artifacts/' + name for name in [
        'cohort.json', 'prepared.json', 'prepared.npz', 'preflight.json', 'calibration.csv', 'qualification.json',
        'per-image.csv', 'validation-opportunity-summary.json', 'validation-opportunity-summary.npz']]
    digests = {}
    for name in files:
        with (run / name).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        check(inventory.get(name) == digest, 'new terminal manifest ' + name)
        digests[name] = digest
    cohort, prepared, summary = [read(out / name) for name in ['cohort.json', 'prepared.json', 'validation-opportunity-summary.json']]
    check(prepared['source_commit'] == commit == summary['source_commit'], 'artifact commit')
    check(prepared['cohort'] == cohort == summary['cohort'], 'cohort copies')
    check(prepared['protocol'] == summary['protocol'] and summary['protocol']['concepts'] == CONCEPTS, 'protocol copies')
    check(prepared['sources'] == summary['sources'], 'source copies')
    flight = read(out / 'preflight.json')
    check(flight == summary['preflight'] and flight['source_commit'] == commit, 'preflight copies')
    check(flight['validation_row_ids'] == PREFLIGHT and flight['module'] == 'model.vision_tower.encoder.layers.22', 'preflight identities')
    check(all(flight[k] is True for k in ['passed', 'zero_exact', 'clean_logits_repeat_exact', 'forwarded_no_cls_exact']), 'preflight flags')
    check(np.isfinite(flight['clean_margin']).all() and np.array_equal(flight['clean_margin'], flight['repeat_margin']), 'preflight clean repeat')
    for name in ['logit_max_abs_delta', 'forwarded_max_abs_delta']:
        check(np.isfinite(flight[name]) and flight[name] > 0, 'preflight response ' + name)
    check(len(flight['mapping_cases']) == 4 and all(np.isfinite(r['raw_margin']) and r['raw_margin'] * (2 * r['expected_present'] - 1) > 0 for r in flight['mapping_cases']), 'preflight mapping signs')
    timing = flight['throughput']
    check(np.isfinite([timing['elapsed_seconds'], timing['pilot_seconds']]).all() and 0 < timing['pilot_seconds'] <= timing['elapsed_seconds'], 'preflight timing finite')
    projection = timing['elapsed_seconds'] + 31200 * timing['pilot_seconds'] / 512 + 180
    same(timing['projected_seconds'], projection, 'preflight timing projection')
    check(projection <= 4500 and timing['passed'] is True and timing['worst_case_outcomes'] == 31200, 'preflight timing budget')
    with (root / 'datasets/nih-chestxray14/manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_id = {r['row_id']: r for r in rows}
    excluded = {by_id[r]['patient_id'] for r in PREFLIGHT}
    candidates = {}
    for row in rows:
        if row['split'] == 'val' and row['patient_id'] not in excluded:
            previous = candidates.get(row['patient_id'])
            if previous is None or row['row_id'] < previous['row_id']:
                candidates[row['patient_id']] = row
    order = np.random.default_rng(20260913).permutation(sorted(candidates)).tolist()
    check(len(order) == 839 and len(excluded) == 16, 'allocation population')
    check(cohort['calibration'] == [candidates[p] for p in order[:700]], 'calibration selection')
    check(cohort['write'] == [candidates[p] for p in order[700:800]], 'write selection')
    check(cohort['preflight'] == [by_id[r] for r in PREFLIGHT], 'preflight identity')
    check(cohort['eligible_patient_ids'] == order and cohort['unused_patient_ids'] == order[800:] and cohort['selection_seed'] == 20260913, 'complete allocation order')
    with np.load(out / 'prepared.npz', allow_pickle=False) as archive:
        prep = {k: archive[k] for k in archive.files}
    with np.load(out / 'validation-opportunity-summary.npz', allow_pickle=False) as archive:
        saved = {k: archive[k] for k in archive.files}
    for key, value in prep.items():
        check(key in saved and np.array_equal(value, saved[key], equal_nan=True) if value.dtype.kind in 'biuf' else key in saved and np.array_equal(value, saved[key]), 'prepared copy ' + key)
    labels = np.asarray([[int(row[q]) for row in cohort['calibration']] for q in CONCEPTS])
    check(np.array_equal(prep['calibration_labels'], labels), 'prepared clinical labels')
    for name in ['clinical_scores', 'control_scores', 'control_coefficients', 'control_intercepts', 'calibration_features', 'calibration_projected', 'train_mean']:
        check(np.isfinite(prep[name]).all(), 'finite preparation ' + name)
    same(prep['clinical_scores'], (prep['calibration_features'] @ prep['coefficients'].T).T, 'stored clinical score binding')
    same(prep['control_scores'], (prep['calibration_features'] @ prep['control_coefficients'].T + prep['control_intercepts']).T, 'stored control score binding')
    check(set(prep['train_row_ids'].tolist()) == {r['row_id'] for r in rows if r['split'] == 'train'} and len(prep['train_row_ids']) == 18212, 'train-only fit population')
    ci = np.random.default_rng(20260914).integers(0, 700, size=(2000, 700), dtype=np.int64)
    wi = np.random.default_rng(20260915).integers(0, 100, size=(5000, 100), dtype=np.int64)
    check(np.array_equal(prep['calibration_indices'], ci) and np.array_equal(prep['write_indices'], wi), 'bootstrap identities')
    cw = np.vstack([np.ones(700), weights(ci, 700)])
    real = np.array([auc(s, y, cw) for s, y in zip(prep['clinical_scores'], labels)])
    control = np.array([auc(s, y, cw) for s, y in zip(prep['control_scores'], prep['control_labels'])])
    selectivity = real[:, 1:] - control[:, 1:].mean(0)
    selectivity[:, ~np.isfinite(control[:, 1:]).all(0)] = np.nan
    same(prep['bootstrap_selectivity'], selectivity, 'reader selectivity')
    same(prep['bootstrap_clinical_auroc'], real[:, 1:], 'reader AUROC draws')
    same(prep['bootstrap_control_auroc'], control[:, 1:], 'control AUROC draws')
    same(prep['clinical_auroc'], real[:, 0], 'reader point AUROC')
    same(prep['control_auroc'], control[:, 0], 'control point AUROC')
    margin = csv_grid(out / 'calibration.csv', cohort['calibration'], CONCEPTS, 'calibration', commit, prepared['protocol']['prompts'])[:, 0]
    cap = np.array([auc(s, y, cw) for s, y in zip(margin, labels)])
    same(saved['calibration_margin'], margin, 'calibration margins')
    same(saved['bootstrap_capability_auroc'], cap[:, 1:], 'capability AUROC draws')
    questions = []
    for i, q in enumerate(CONCEPTS):
        enough = min(labels[i].sum(), 700 - labels[i].sum()) >= 10
        rp = enough and np.isfinite(selectivity[i]).sum() >= 1900 and lower(selectivity[i], 2.5) > 0
        cp = enough and np.isfinite(cap[i, 1:]).sum() >= 1900 and lower(cap[i, 1:], 5) > .5
        record = summary['qualification']['questions'][i]
        check(record['question'] == q and record['reader_pass'] == rp and record['capability_pass'] == cp, 'qualification ' + q)
        if rp and cp:
            questions.append(q)
    check(read(out / 'qualification.json') == summary['qualification'], 'qualification copies')
    check(summary['qualification']['qualified_questions'] == questions == saved['qualified_questions'].tolist(), 'qualified order')
    check(summary['qualification']['K'] == len(questions) and summary['qualification']['expected_outcomes'] == 4200 + 4500 * len(questions), 'qualification counts')
    grids = csv_grid(out / 'per-image.csv', cohort['write'], questions, 'write', commit, prepared['protocol']['prompts'])
    same(saved['write_margin'], grids, 'write margins')
    eligible, write_results = [], {}
    for i, q in enumerate(questions):
        m = grids[i]
        result = summary['write'][q]
        y = np.asarray([int(row[q]) for row in cohort['write']])
        same(saved[q + '__diagnostic_auroc'], np.asarray([auc(s, y, np.ones((1, 100)))[0] for s in m]), q + ' diagnostic AUROC')
        same(saved[q + '__diagnostic_brier'], ((probability(m) - y) ** 2).mean(1), q + ' diagnostic Brier')
        for name, values in [('probability', probability(m)), ('margin', m)]:
            responses = (values[2::2] - values[1::2]) / 2
            means = responses.mean(1)
            draws = responses[:, wi].mean(2)
            maxima = np.maximum(0, np.maximum(draws[1:21].max(0), abs(draws[21])))
            excess = means[0] - max(0, means[1:21].max(), abs(means[21]))
            boot = draws[0] - maxima
            same(result[name]['excess'], excess, q + name + ' excess')
            same(result[name]['excess_lower_95'], lower(boot, 5), q + name + ' lower')
            same(result[name]['excess_ci95'], np.percentile(boot, [2.5, 97.5]), q + name + ' interval')
            for key, array in [(name + '_response', responses), ('bootstrap_' + name + '_response', draws),
                               ('bootstrap_' + name + '_excess', boot), ('bootstrap_' + name + '_control_max', maxima),
                               ('bootstrap_' + name + '_positive_change', (values[2::2] - values[0])[:, wi].mean(2)),
                               ('bootstrap_' + name + '_negative_change', (values[0] - values[1::2])[:, wi].mean(2))]:
                same(saved[q + '__' + key], array, q + key)
            same([result[name]['direction_responses'][d] for d, _ in CONDITIONS[1::2]], means, q + name + ' direction means')
            if name == 'probability':
                passed = bool(lower(boot, 5) > 0 and (values[2] - values[0]).mean() > 0 and (values[0] - values[1]).mean() > 0)
                check(result['opportunity_pass'] == passed, q + ' write decision')
                write_results[q] = dict(excess=float(excess), lower95=float(lower(boot, 5)), passed=passed)
                if passed:
                    eligible.append(q)
    check(summary['K'] == len(questions) and summary['n_outcomes'] == 4200 + 4500 * len(questions), 'complete count')
    route = 'test_cohort_competition_registration' if eligible else 'evidence_synthesis'
    check(summary['route'] == route and summary['eligible_questions'] == eligible, 'final route')
    return dict(status='PASS', run_id=run_id, source_commit=commit, verified_new_files=digests,
                n_outcomes=summary['n_outcomes'], qualified_questions=questions, eligible_questions=eligible,
                route=route, write=write_results, max_numerical_discrepancy=MAX_ERROR)


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1], sys.argv[2]), allow_nan=False))
