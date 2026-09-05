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


def manifest_inventory(path):
    inventory = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        check(len(parts) == 2, 'malformed dispatcher manifest')
        digest, name = parts
        name = name.strip().lstrip('*').removeprefix('./')
        check(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
              'malformed dispatcher digest')
        check(name and name not in inventory, 'duplicate dispatcher manifest path')
        inventory[name] = digest
    return inventory


def same(actual, expected, name):
    global MAX_ERROR
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(actual.shape == expected.shape, name + ' shape')
    check(actual.dtype.kind in 'biuf' and expected.dtype.kind in 'biuf', name + ' numeric')
    check(np.array_equal(np.isnan(actual), np.isnan(expected)), name + ' NaN locations')
    finite = np.isfinite(expected)
    check(np.isfinite(actual[finite]).all(), name + ' finite')
    error = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.
    MAX_ERROR = max(MAX_ERROR, error)
    check(error < 2e-10, name + ' numerical discrepancy ' + str(error))


def shaped(arrays, key, shape, *, binary=False, finite=True):
    check(key in arrays, 'missing array ' + key)
    value = np.asarray(arrays[key])
    check(value.shape == shape, key + ' shape')
    check(value.dtype.kind in 'biuf', key + ' numeric')
    if finite:
        check(np.isfinite(value).all(), key + ' finite numeric')
    if binary:
        check(np.isin(value, (0, 1)).all(), key + ' binary')
    return value


def verify_prepared_shapes(prep):
    """Reject truncated control families before zip/mean can hide them."""
    shaped(prep, 'calibration_labels', (6, 700), binary=True)
    shaped(prep, 'clinical_scores', (6, 700))
    shaped(prep, 'control_scores', (20, 700))
    shaped(prep, 'control_labels', (20, 700), binary=True)
    shaped(prep, 'control_coefficients', (20, 512))
    shaped(prep, 'control_intercepts', (20,))
    shaped(prep, 'calibration_features', (700, 512))
    shaped(prep, 'calibration_projected', (700, 512))
    shaped(prep, 'train_mean', (512,))
    shaped(prep, 'coefficients', (6, 512))
    shaped(prep, 'calibration_indices', (2000, 700))
    shaped(prep, 'write_indices', (5000, 100))
    shaped(prep, 'bootstrap_selectivity', (6, 2000), finite=False)
    shaped(prep, 'bootstrap_clinical_auroc', (6, 2000), finite=False)
    shaped(prep, 'bootstrap_control_auroc', (20, 2000), finite=False)
    shaped(prep, 'clinical_auroc', (6,))
    shaped(prep, 'control_auroc', (20,))


def verify_summary_npz_identity(saved, commit, questions, eligible, route):
    check(saved['source_commit'].shape == () and saved['source_commit'].item() == commit,
          'summary NPZ source identity')
    check(saved['qualified_questions'].tolist() == questions, 'summary NPZ qualified questions')
    check(saved['eligible_questions'].tolist() == eligible, 'summary NPZ eligible questions')
    check(saved['route'].shape == () and saved['route'].item() == route, 'summary NPZ route')


def verify_qualification_report(qualification, labels, real, control, selectivity, cap, margin, commit):
    top_keys = {'questions', 'qualified_questions', 'K', 'expected_outcomes', 'source_commit'}
    check(set(qualification) == top_keys and qualification['source_commit'] == commit,
          'qualification schema/source identity')
    check(len(qualification['questions']) == 6, 'qualification question count')
    questions = []
    control_ci = [[lower(row, 2.5), lower(row, 97.5)] for row in control[:, 1:]]
    for i, q in enumerate(CONCEPTS):
        pos = int(labels[i].sum())
        enough = min(pos, 700 - pos) >= 10
        reader_valid = int(np.isfinite(selectivity[i]).sum())
        capability_valid = int(np.isfinite(cap[i, 1:]).sum())
        reader_ci = [lower(selectivity[i], 2.5), lower(selectivity[i], 97.5)]
        capability_ci = [lower(cap[i, 1:], 2.5), lower(cap[i, 1:], 97.5)]
        reader_pass = bool(enough and reader_valid >= 1900 and reader_ci[0] > 0)
        capability_pass = bool(enough and capability_valid >= 1900 and lower(cap[i, 1:], 5) > .5)
        expected = {
            'question': q, 'positive': pos, 'negative': 700 - pos,
            'reader_valid_draws': reader_valid, 'reader_lower_95': reader_ci[0],
            'reader_pass': reader_pass, 'capability_valid_draws': capability_valid,
            'capability_lower_95': lower(cap[i, 1:], 5), 'capability_pass': capability_pass,
            'qualified': reader_pass and capability_pass, 'reader_auroc': real[i, 0],
            'reader_auroc_ci95': [lower(real[i, 1:], 2.5), lower(real[i, 1:], 97.5)],
            'control_auroc': control[:, 0], 'control_auroc_ci95': control_ci,
            'selectivity': real[i, 0] - control[:, 0].mean(), 'selectivity_ci95': reader_ci,
            'capability_auroc': cap[i, 0], 'capability_auroc_ci95': capability_ci,
            'capability_brier': np.mean((probability(margin[i]) - labels[i]) ** 2),
        }
        actual = qualification['questions'][i]
        check(set(actual) == set(expected) and actual['question'] == q,
              'qualification record schema/order ' + q)
        for key in ('question', 'positive', 'negative', 'reader_valid_draws', 'reader_pass',
                    'capability_valid_draws', 'capability_pass', 'qualified'):
            check(type(actual[key]) is type(expected[key]) and actual[key] == expected[key],
                  'qualification field ' + q + ' ' + key)
        for key in set(expected) - {'question', 'positive', 'negative', 'reader_valid_draws',
                                    'reader_pass', 'capability_valid_draws', 'capability_pass', 'qualified'}:
            same(actual[key], expected[key], 'qualification field ' + q + ' ' + key)
        if expected['qualified']:
            questions.append(q)
    check(qualification['qualified_questions'] == questions, 'qualified question order')
    check(qualification['K'] == len(questions)
          and qualification['expected_outcomes'] == 4200 + 4500 * len(questions),
          'qualification counts')
    return questions


def verify_write_report(result, saved, prefix, margins, labels, wi):
    check(set(result) == {'diagnostics', 'probability', 'margin', 'opportunity_pass'},
          prefix + ' write schema')
    probabilities = probability(margins)
    diagnostic_auc = np.asarray([auc(row, labels, np.ones((1, 100)))[0] for row in margins])
    diagnostic_brier = ((probabilities - labels) ** 2).mean(1)
    check(len(result['diagnostics']) == len(CONDITIONS), prefix + ' diagnostic count')
    for i, ((direction, alpha), record) in enumerate(zip(CONDITIONS, result['diagnostics'], strict=True)):
        check(set(record) == {'direction', 'alpha', 'auroc', 'brier'}
              and record['direction'] == direction and record['alpha'] == alpha,
              prefix + ' diagnostic identity')
        same(record['auroc'], diagnostic_auc[i], prefix + ' diagnostic AUROC')
        same(record['brier'], diagnostic_brier[i], prefix + ' diagnostic Brier')
    same(saved[prefix + '__diagnostic_auroc'], diagnostic_auc, prefix + ' diagnostic AUROC array')
    same(saved[prefix + '__diagnostic_brier'], diagnostic_brier, prefix + ' diagnostic Brier array')
    same(saved[prefix + '__raw_margin'], margins, prefix + ' raw-margin array')
    same(saved[prefix + '__probability'], probabilities, prefix + ' probability array')
    passed = None
    for name, values in [('probability', probabilities), ('margin', margins)]:
        report = result[name]
        check(set(report) == {'direction_responses', 'positive_changes', 'negative_changes',
                              'dose_means', 'baseline_mean', 'excess', 'excess_ci95',
                              'excess_lower_95'}, prefix + ' ' + name + ' schema')
        paired = values[1:].reshape(22, 2, 100)
        responses = (paired[:, 1] - paired[:, 0]) / 2
        means = responses.mean(1)
        draws = responses[:, wi].mean(2)
        maxima = np.maximum(0, np.maximum(draws[1:21].max(0), abs(draws[21])))
        excess = means[0] - max(0, means[1:21].max(), abs(means[21]))
        boot = draws[0] - maxima
        plus = paired[:, 1] - values[0]
        minus = values[0] - paired[:, 0]
        check(list(report['direction_responses']) == [d for d, _ in CONDITIONS[1::2]],
              prefix + name + ' direction response order')
        check(list(report['positive_changes']) == [d for d, _ in CONDITIONS[1::2]],
              prefix + name + ' positive change order')
        check(list(report['negative_changes']) == [d for d, _ in CONDITIONS[1::2]],
              prefix + name + ' negative change order')
        same(list(report['direction_responses'].values()), means, prefix + name + ' direction means')
        same(list(report['positive_changes'].values()), plus.mean(1), prefix + name + ' positive means')
        same(list(report['negative_changes'].values()), minus.mean(1), prefix + name + ' negative means')
        same(report['dose_means'], paired.mean(2), prefix + name + ' dose means')
        same(report['baseline_mean'], values[0].mean(), prefix + name + ' baseline mean')
        same(report['excess'], excess, prefix + name + ' excess')
        same(report['excess_lower_95'], lower(boot, 5), prefix + name + ' lower')
        same(report['excess_ci95'], [lower(boot, 2.5), lower(boot, 97.5)], prefix + name + ' interval')
        for key, array in [(name + '_response', responses), ('bootstrap_' + name + '_response', draws),
                           ('bootstrap_' + name + '_excess', boot), ('bootstrap_' + name + '_control_max', maxima),
                           ('bootstrap_' + name + '_positive_change', plus[:, wi].mean(2)),
                           ('bootstrap_' + name + '_negative_change', minus[:, wi].mean(2))]:
            same(saved[prefix + '__' + key], array, prefix + key)
        if name == 'probability':
            passed = bool(lower(boot, 5) > 0 and plus[0].mean() > 0 and minus[0].mean() > 0)
    check(result['opportunity_pass'] == passed, prefix + ' write decision')
    return passed


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
    inventory = manifest_inventory(run / 'SHA256SUMS')
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
    verify_prepared_shapes(prep)
    for key, value in prep.items():
        equal = np.array_equal(value, saved[key], equal_nan=True) if key in saved and value.dtype.kind in 'biuf' else key in saved and np.array_equal(value, saved[key])
        check(equal, 'prepared copy ' + key)
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
    check(read(out / 'qualification.json') == summary['qualification'], 'qualification copies')
    questions = verify_qualification_report(summary['qualification'], labels, real, control,
                                             selectivity, cap, margin, commit)
    check(list(summary['write']) == questions, 'exact summary write-question set/order')
    grids = csv_grid(out / 'per-image.csv', cohort['write'], questions, 'write', commit, prepared['protocol']['prompts'])
    same(saved['write_margin'], grids, 'write margins')
    eligible, write_results = [], {}
    for i, q in enumerate(questions):
        m = grids[i]
        result = summary['write'][q]
        y = np.asarray([int(row[q]) for row in cohort['write']])
        passed = verify_write_report(result, saved, q, m, y, wi)
        write_results[q] = dict(excess=float(result['probability']['excess']),
                                lower95=float(result['probability']['excess_lower_95']), passed=passed)
        if passed:
            eligible.append(q)
    check(summary['K'] == len(questions) and summary['n_outcomes'] == 4200 + 4500 * len(questions), 'complete count')
    route = 'test_cohort_competition_registration' if eligible else 'evidence_synthesis'
    check(summary['route'] == route and summary['eligible_questions'] == eligible, 'final route')
    verify_summary_npz_identity(saved, commit, questions, eligible, route)
    return dict(status='PASS', run_id=run_id, source_commit=commit, verified_new_files=digests,
                n_outcomes=summary['n_outcomes'], qualified_questions=questions, eligible_questions=eligible,
                route=route, write=write_results, max_numerical_discrepancy=MAX_ERROR)


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1], sys.argv[2]), allow_nan=False))
