"""CPU preparation and patient-bootstrap inference for the validation pilot."""

from __future__ import annotations

import numpy as np

from src.bootstrap_probe import boot_auroc
from src.probe import build_types, control_labels
from src.qwen_causal_ownership_gate import CONCEPTS, PROMPTS

CAL_N, WRITE_N = 700, 100
CAL_B, WRITE_B = 2000, 5000
SPLIT_SEED, CAL_SEED, WRITE_SEED = 20260913, 20260914, 20260915
MODULE = 'model.vision_tower.encoder.layers.22'
DIRECTIONS = ('clinical', *(f'random{i}' for i in range(20)), 'sham')
CONDITIONS = (('baseline', 0.), *((d, a) for d in DIRECTIONS for a in (-.25, .25)))
FIELDS = ('source_commit', 'phase', 'question', 'prompt', 'patient_index', 'patient_id',
          'row_id', 'image_path', 'label', 'direction', 'alpha', 'raw_margin', 'probability')


def finite_array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
        raise ValueError(f'{name}: expected finite numeric {shape}')
    return array


def binary_labels(value, shape, name='labels'):
    array = finite_array(value, shape, name)
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f'{name}: labels must be binary')
    return array.astype(np.int8)


def sigmoid(margin):
    margin = np.asarray(margin, dtype=np.float64)
    z = np.exp(-np.abs(margin))
    return np.where(margin >= 0, 1 / (1 + z), z / (1 + z))


def allocate(rows, preflight_ids, ownership):
    """Select identifiers before reading any labels or image outcomes."""
    by_id = {r['row_id']: r for r in rows}
    if len(by_id) != len(rows) or len(preflight_ids) != 16 or len(set(preflight_ids)) != 16:
        raise ValueError('duplicate manifest or preflight identity')
    if any(i not in by_id or by_id[i]['split'] != 'val' for i in preflight_ids):
        raise ValueError('preflight rows must belong to original val')
    patients = {}
    for row in rows:
        patient, split = str(row['patient_id']), row['split']
        if split not in ('train', 'val', 'test') or patients.setdefault(patient, split) != split:
            raise ValueError('manifest patient split leakage')
    excluded = {str(by_id[i]['patient_id']) for i in preflight_ids}
    if len(excluded) != 16:
        raise ValueError('preflight must identify sixteen patients')
    own_ids, own_patients = ownership['row_ids'], ownership['patient_ids']
    if (len(own_ids) != len(own_patients) or len(set(own_ids)) != len(own_ids)
            or len(set(own_patients)) != len(own_patients)):
        raise ValueError('ownership cohort identities are not unique')
    for rid, pid in zip(own_ids, own_patients, strict=True):
        if rid not in by_id or str(by_id[rid]['patient_id']) != pid or by_id[rid]['split'] != 'test':
            raise ValueError('ownership manifest identity or split mismatch')
    candidates = {}
    for row in rows:
        pid = str(row['patient_id'])
        if row['split'] == 'val' and pid not in excluded:
            if pid not in candidates or row['row_id'] < candidates[pid]['row_id']:
                candidates[pid] = row
    if len(candidates) < CAL_N + WRITE_N:
        raise ValueError('UNAVAILABLE: fewer than 800 eligible validation patients')
    order = np.random.default_rng(SPLIT_SEED).permutation(sorted(candidates)).tolist()
    return {'calibration': [candidates[p] for p in order[:CAL_N]],
            'write': [candidates[p] for p in order[CAL_N:CAL_N + WRITE_N]],
            'unused_patient_ids': order[CAL_N + WRITE_N:],
            'preflight': [by_id[i] for i in preflight_ids], 'eligible_patient_ids': order,
            'selection_seed': SPLIT_SEED}


def labels_for(rows):
    return binary_labels([[int(r[c]) for r in rows] for c in CONCEPTS], (6, len(rows)))


def validate_feasibility(allocation):
    for phase, counts, anchors in (
        ('calibration', [34, 43, 12, 25, 39, 48],
         [('9691', '00009691_000'), ('2995', '00002995_000')]),
        ('write', [7, 7, 1, 3, 8, 12], [('6752', '00006752_000'), ('7202', '00007202_002')]),
    ):
        rows = allocation[phase]
        if (labels_for(rows).sum(axis=1).tolist() != counts
                or [(str(rows[i]['patient_id']), rows[i]['row_id']) for i in (0, -1)] != anchors):
            raise ValueError('accepted feasibility counts or allocation anchors mismatch')
    if len(allocation['eligible_patient_ids']) != 839:
        raise ValueError('accepted validation population mismatch')


def bootstrap_indices(n, b, seed):
    return np.random.default_rng(seed).integers(0, n, size=(b, n), dtype=np.int64)


def validate_indices(indices, n, b, seed):
    if (np.asarray(indices).dtype != np.dtype('int64')
            or not np.array_equal(indices, bootstrap_indices(n, b, seed))):
        raise ValueError('registered joint bootstrap identity mismatch')


def multiplicities(indices, n):
    return np.stack([np.bincount(draw, minlength=n) for draw in indices])


def auc(scores, labels):
    return float(boot_auroc(np.asarray(scores), np.asarray(labels),
                           np.ones((1, len(labels)), dtype=np.int64))[0])


def interval(draws, quantiles=(2.5, 97.5)):
    valid = np.asarray(draws)[np.isfinite(draws)]
    return np.percentile(valid, quantiles, method='linear').tolist() if valid.size else [float('nan')] * len(quantiles)


def selectivity_draws(clinical, controls):
    if clinical.shape[0] != 6 or controls.shape != (20, clinical.shape[1]):
        raise ValueError('joint reader/control draw dimensions mismatch')
    result = clinical - controls.mean(axis=0)[None, :]
    result[:, ~np.isfinite(controls).all(axis=0)] = np.nan
    result[~np.isfinite(clinical)] = np.nan
    return result


def make_directions(clinical):
    clinical = np.asarray(clinical)
    if clinical.ndim != 2 or clinical.shape[0] != 6 or not np.isfinite(clinical).all():
        raise ValueError('six finite clinical normals required')
    if not np.allclose(np.linalg.norm(clinical, axis=1), 1, atol=1e-6, rtol=1e-5):
        raise ValueError('clinical normals must be unit length')
    rng = np.random.default_rng(0)
    random = np.stack([rng.standard_normal(clinical.shape[1]) for _ in range(20)])
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    sham = np.stack([rng.permutation(v) for v in clinical])
    sham /= np.linalg.norm(sham, axis=1, keepdims=True)
    return random, sham


def prepare_readers(raw, rows, calibration_ids, bundle, controls):
    """Recover clinical rankings and fit the twenty shared train-only controls."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    by_id = {r['row_id']: i for i, r in enumerate(rows)}
    if len(by_id) != len(rows) or any(i not in by_id for i in calibration_ids):
        raise ValueError('activation/calibration row identity mismatch')
    train = np.asarray([r['split'] == 'train' for r in rows])
    cal = np.asarray([by_id[i] for i in calibration_ids])
    if not train.any() or any(rows[i]['split'] != 'val' for i in cal):
        raise ValueError('probe training/calibration split mismatch')
    raw = finite_array(raw, (len(rows), bundle['projection'].shape[0]), 'pooled activations')
    # The accepted first gate projects float32 inputs and standardizes in-place.
    projected = raw.astype(np.float32) @ bundle['projection'].astype(np.float32)
    mean = projected[train].mean(axis=0, dtype=np.float64)
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = mean, bundle['scale']
    scaler.n_features_in_ = projected.shape[1]
    features = scaler.transform(projected)
    clinical = (features[cal] @ bundle['coefficients'].T).T
    types = build_types(rows, ('view_AP', 'sex_M', 'age'))
    coefficients, intercepts, scores, labels, train_labels = [], [], [], [], []
    if len(controls) != 20:
        raise ValueError('twenty accepted control assignments required')
    for seed, record in enumerate(controls):
        assigned, expected, usable = control_labels(types, np.random.default_rng(seed))
        if record['seed'] != seed or not usable or record['assignment'] != expected:
            raise ValueError('accepted control assignment identity mismatch')
        if len(np.unique(assigned[train])) != 2:
            raise ValueError('train-only control lacks two classes')
        fit = LogisticRegression(C=1., max_iter=2000, random_state=0).fit(features[train], assigned[train])
        coefficients.append(fit.coef_[0])
        intercepts.append(fit.intercept_[0])
        scores.append(fit.decision_function(features[cal]))
        labels.append(assigned[cal])
        train_labels.append(assigned[train])
    return {'train_mean': mean, 'calibration_projected': projected[cal], 'calibration_features': features[cal],
            'clinical_scores': clinical, 'control_coefficients': np.stack(coefficients),
            'control_intercepts': np.asarray(intercepts), 'control_scores': np.stack(scores),
            'control_labels': np.stack(labels), 'control_train_labels': np.stack(train_labels),
            'train_row_ids': np.asarray([r['row_id'] for r in rows if r['split'] == 'train']),
            'train_types': types[train], 'calibration_types': types[cal]}


def reader_statistics(arrays):
    labels = binary_labels(arrays['calibration_labels'], (6, CAL_N))
    clinical = finite_array(arrays['clinical_scores'], (6, CAL_N), 'clinical scores')
    control = finite_array(arrays['control_scores'], (20, CAL_N), 'control scores')
    control_labels_ = binary_labels(arrays['control_labels'], (20, CAL_N), 'control labels')
    indices = arrays['calibration_indices']
    validate_indices(indices, CAL_N, CAL_B, CAL_SEED)
    counts = multiplicities(indices, CAL_N)
    real_boot = np.stack([boot_auroc(s, y, counts) for s, y in zip(clinical, labels)])
    control_boot = np.stack([boot_auroc(s, y, counts) for s, y in zip(control, control_labels_)])
    return {'clinical_auroc': np.asarray([auc(s, y) for s, y in zip(clinical, labels)]),
            'control_auroc': np.asarray([auc(s, y) for s, y in zip(control, control_labels_)]),
            'bootstrap_clinical_auroc': real_boot, 'bootstrap_control_auroc': control_boot,
            'bootstrap_selectivity': selectivity_draws(real_boot, control_boot)}


def qualify(labels, reader_draws, capability_draws):
    labels = binary_labels(labels, (6, CAL_N))
    if reader_draws.shape != (6, CAL_B) or capability_draws.shape != (6, CAL_B):
        raise ValueError('qualification draw dimensions mismatch')
    table, selected = [], []
    for i, c in enumerate(CONCEPTS):
        pos = int(labels[i].sum())
        support = min(pos, CAL_N - pos) >= 10
        reader_valid = int(np.isfinite(reader_draws[i]).sum())
        capability_valid = int(np.isfinite(capability_draws[i]).sum())
        reader_lower = interval(reader_draws[i])[0]
        capability_lower = interval(capability_draws[i], (5,))[0]
        reader_pass = bool(support and reader_valid >= 1900 and reader_lower > 0)
        capability_pass = bool(support and capability_valid >= 1900 and capability_lower > .5)
        admitted = reader_pass and capability_pass
        table.append({'question': c, 'positive': pos, 'negative': CAL_N - pos,
                      'reader_valid_draws': reader_valid, 'reader_lower_95': reader_lower,
                      'reader_pass': reader_pass, 'capability_valid_draws': capability_valid,
                      'capability_lower_95': capability_lower, 'capability_pass': capability_pass,
                      'qualified': admitted})
        if admitted:
            selected.append(c)
    return {'questions': table, 'qualified_questions': selected, 'K': len(selected),
            'expected_outcomes': 4200 + 4500 * len(selected)}


def calibration_statistics(arrays, margins):
    margins = finite_array(margins, (6, CAL_N), 'clean calibration margins')
    result = reader_statistics(arrays)
    counts = multiplicities(arrays['calibration_indices'], CAL_N)
    labels = arrays['calibration_labels']
    capability = np.stack([boot_auroc(s, y, counts) for s, y in zip(margins, labels)])
    result.update(calibration_margin=margins, calibration_probability=sigmoid(margins),
                  bootstrap_capability_auroc=capability)
    qualification = qualify(labels, result['bootstrap_selectivity'], capability)
    for i, row in enumerate(qualification['questions']):
        row.update(reader_auroc=float(result['clinical_auroc'][i]),
                   reader_auroc_ci95=interval(result['bootstrap_clinical_auroc'][i]),
                   control_auroc=result['control_auroc'].tolist(),
                   control_auroc_ci95=[interval(d) for d in result['bootstrap_control_auroc']],
                   selectivity=float(result['clinical_auroc'][i] - result['control_auroc'].mean()),
                   selectivity_ci95=interval(result['bootstrap_selectivity'][i]),
                   capability_auroc=auc(margins[i], labels[i]),
                   capability_auroc_ci95=interval(capability[i]),
                   capability_brier=float(np.mean((sigmoid(margins[i]) - labels[i]) ** 2)))
    return qualification, result


def validate_grid(records, allocation, source_commit, phase, questions):
    if phase not in ('calibration', 'write'):
        raise ValueError('unknown scoring phase')
    rows = allocation[phase]
    conditions = (CONDITIONS[0],) if phase == 'calibration' else CONDITIONS
    if questions != [c for c in CONCEPTS if c in questions]:
        raise ValueError('question identity/order mismatch')
    n = CAL_N if phase == 'calibration' else WRITE_N
    if len(rows) != n or (phase == 'calibration' and questions != list(CONCEPTS)):
        raise ValueError('incomplete cohort/question grid')
    records = iter(records)
    margins = np.empty((len(questions), len(conditions), n))
    for qi, c in enumerate(questions):
        for di, (direction, alpha) in enumerate(conditions):
            for i, row in enumerate(rows):
                actual = next(records, None)
                if actual is None:
                    raise ValueError('missing CSV outcome')
                expected = {'source_commit': source_commit, 'phase': phase, 'question': c,
                            'prompt': PROMPTS[c], 'patient_index': str(i), 'patient_id': str(row['patient_id']),
                            'row_id': row['row_id'], 'image_path': row['image_path'], 'label': str(int(row[c])),
                            'direction': direction}
                try:
                    if set(actual) != set(FIELDS) or any(str(actual.get(k)) != v for k, v in expected.items()):
                        raise ValueError('CSV cohort/source/condition identity mismatch')
                    margin, probability, dose = (float(actual[k]) for k in ('raw_margin', 'probability', 'alpha'))
                    if (not np.isfinite([margin, probability, dose]).all() or dose != alpha
                            or not np.isclose(probability, sigmoid(margin), atol=1e-12, rtol=1e-12)):
                        raise ValueError('CSV nonfinite score, dose or sigmoid mismatch')
                except (TypeError, KeyError) as error:
                    raise ValueError('malformed CSV outcome') from error
                margins[qi, di, i] = margin
    if next(records, None) is not None:
        raise ValueError('extra or duplicate CSV outcomes')
    return margins


def write_statistics(margins, labels, indices):
    margins = finite_array(margins, (45, WRITE_N), 'write margins')
    labels = binary_labels(labels, (WRITE_N,))
    indices = np.asarray(indices)
    if indices.shape != (WRITE_B, WRITE_N) or indices.dtype != np.int64 or np.any(indices < 0) or np.any(indices >= WRITE_N):
        raise ValueError('write bootstrap dimensions/range mismatch')
    probability = sigmoid(margins)
    arrays = {'raw_margin': margins, 'probability': probability,
              'diagnostic_auroc': np.asarray([auc(m, labels) for m in margins]),
              'diagnostic_brier': np.mean((probability - labels) ** 2, axis=1)}
    result = {'diagnostics': [{'direction': d, 'alpha': a, 'auroc': arrays['diagnostic_auroc'][i],
                              'brier': arrays['diagnostic_brier'][i]} for i, (d, a) in enumerate(CONDITIONS)]}
    for name, values in (('probability', probability), ('margin', margins)):
        paired = values[1:].reshape(22, 2, WRITE_N)
        response = (paired[:, 1] - paired[:, 0]) / 2
        means = response.mean(axis=1)
        boot = response[:, indices].mean(axis=2)
        control_max = np.maximum(0, np.maximum(boot[1:21].max(axis=0), np.abs(boot[21])))
        excess = means[0] - max(0, means[1:21].max(), abs(means[21]))
        excess_boot = boot[0] - control_max
        plus = paired[:, 1] - values[0]
        minus = values[0] - paired[:, 0]
        arrays.update({f'{name}_response': response, f'bootstrap_{name}_response': boot,
                       f'bootstrap_{name}_excess': excess_boot,
                       f'bootstrap_{name}_control_max': control_max,
                       f'bootstrap_{name}_positive_change': plus[:, indices].mean(axis=2),
                       f'bootstrap_{name}_negative_change': minus[:, indices].mean(axis=2)})
        result[name] = {'direction_responses': dict(zip(DIRECTIONS, means.tolist())),
                        'positive_changes': dict(zip(DIRECTIONS, plus.mean(axis=1).tolist())),
                        'negative_changes': dict(zip(DIRECTIONS, minus.mean(axis=1).tolist())),
                        'dose_means': paired.mean(axis=2).tolist(),
                        'baseline_mean': float(values[0].mean()), 'excess': float(excess),
                        'excess_ci95': interval(excess_boot), 'excess_lower_95': interval(excess_boot, (5,))[0]}
    p = result['probability']
    result['opportunity_pass'] = bool(p['excess_lower_95'] > 0 and p['positive_changes']['clinical'] > 0
                                      and p['negative_changes']['clinical'] > 0)
    return result, arrays
