from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src import llava_validation_opportunity as core
from src import run_llava_validation_opportunity as runner


def manifest_fixture():
    rows = []
    for i in range(855):
        for suffix in ('001', '000'):
            rows.append(dict(row_id=f'{i:08d}_{suffix}', patient_id=str(i), split='val',
                             image_path=f'/images/{i:08d}_{suffix}.png',
                             **{c: str(i % 2) for c in core.CONCEPTS}))
    rows.append(dict(row_id='train', patient_id='train', split='train'))
    rows.append(dict(row_id='test', patient_id='test', split='test'))
    return rows


class AllocationTests(unittest.TestCase):
    def test_label_blind_sorted_string_split_and_minimum_row(self):
        rows = manifest_fixture()
        excluded = [f'{i:08d}_001' for i in range(16)]
        result = core.allocate(rows, excluded, {'row_ids': ['test'], 'patient_ids': ['test']})
        expected = np.random.default_rng(20260913).permutation(sorted(map(str, range(16, 855))))
        self.assertEqual([r['patient_id'] for r in result['calibration']], expected[:700].tolist())
        self.assertEqual([r['patient_id'] for r in result['write']], expected[700:800].tolist())
        self.assertTrue(all(r['row_id'].endswith('_000') for r in result['write']))
        changed = deepcopy(rows)
        for row in changed:
            for c in core.CONCEPTS:
                row[c] = 'unreadable label'
        again = core.allocate(changed[::-1], excluded, {'row_ids': ['test'], 'patient_ids': ['test']})
        self.assertEqual([r['row_id'] for r in result['write']], [r['row_id'] for r in again['write']])

    def test_split_leakage_overlap_and_unavailable(self):
        rows = manifest_fixture()
        excluded = [f'{i:08d}_001' for i in range(16)]
        ownership = {'row_ids': ['test'], 'patient_ids': ['test']}
        for bad in (rows + [rows[0]], rows + [dict(row_id='leak', patient_id='20', split='train')], rows[:100]):
            with self.assertRaises(ValueError):
                core.allocate(bad, excluded, ownership)
        with self.assertRaises(ValueError):
            core.allocate(rows, excluded, {'row_ids': ['00000020_000'], 'patient_ids': ['20']})

    def test_accepted_feasibility_anchors_and_label_counts(self):
        allocation = allocation_fixture()
        for phase, counts, endpoints in (
            ('calibration', [34, 43, 12, 25, 39, 48], [('9691', '00009691_000'), ('2995', '00002995_000')]),
            ('write', [7, 7, 1, 3, 8, 12], [('6752', '00006752_000'), ('7202', '00007202_002')]),
        ):
            for i, row in enumerate(allocation[phase]):
                for c, count in zip(core.CONCEPTS, counts):
                    row[c] = str(int(i < count))
            for i, (pid, rid) in zip((0, -1), endpoints):
                allocation[phase][i].update(patient_id=pid, row_id=rid)
        core.validate_feasibility(allocation)
        for mutate in (lambda a: a['write'][0].update(patient_id='other'),
                       lambda a: a['calibration'][-1].update(Effusion='1'),
                       lambda a: a['eligible_patient_ids'].pop()):
            bad = deepcopy(allocation)
            mutate(bad)
            with self.assertRaises(ValueError):
                core.validate_feasibility(bad)


class StatisticalTests(unittest.TestCase):
    def test_joint_selectivity_requires_every_control(self):
        clinical = np.full((6, 2000), .9)
        controls = np.full((20, 2000), .5)
        controls[19, :101] = np.nan
        result = core.selectivity_draws(clinical, controls)
        self.assertTrue(np.isnan(result[:, :101]).all())
        np.testing.assert_allclose(result[:, 101:], .4)

    def test_qualification_strict_thresholds_and_valid_draw_support(self):
        labels = np.tile(np.arange(700) < 10, (6, 1)).astype(int)
        reader = np.full((6, 2000), .1)
        capability = np.full((6, 2000), .6)
        reader[1] = 0
        capability[2] = .5
        reader[3, :101] = np.nan
        capability[4, :100] = np.nan
        labels[5, 9] = 0
        result = core.qualify(labels, reader, capability)
        self.assertEqual(result['qualified_questions'], [core.CONCEPTS[0], core.CONCEPTS[4]])
        self.assertEqual(result['expected_outcomes'], 13200)
        self.assertEqual(len(result['questions']), 6)

    def test_bidirectional_control_max_recomputed_inside_each_draw(self):
        # Alternating random winners have zero point mean but positive replica maxima.
        margins = np.zeros((45, 100))
        margins[1] = -.5
        margins[2] = .5
        margins[3, :50], margins[4, :50] = -1, 1
        margins[3, 50:], margins[4, 50:] = 1, -1
        margins[5], margins[6] = -margins[3], -margins[4]
        draws = np.tile(np.arange(50, dtype=np.int64), (5000, 2))
        result, arrays = core.write_statistics(margins, np.zeros(100, int), draws)
        self.assertGreater(result['probability']['excess'], 0)
        self.assertLess(result['probability']['excess_lower_95'], 0)
        self.assertFalse(result['opportunity_pass'])
        self.assertTrue(np.isnan(arrays['diagnostic_auroc']).all())
        self.assertEqual(arrays['bootstrap_probability_excess'].shape, (5000,))

    def test_bootstrap_auroc_ties_and_joint_control_undefined(self):
        from sklearn.metrics import roc_auc_score
        scores = np.array([0., 0., 2., 1., 2., 2.])
        labels = np.array([0, 1, 0, 1, 1, 0])
        indices = np.array([[0, 0, 1, 3, 4, 4], [1, 1, 3, 3, 4, 4]], dtype=np.int64)
        values = core.boot_auroc(scores, labels, core.multiplicities(indices, 6))
        self.assertEqual(values[0], roc_auc_score(labels[indices[0]], scores[indices[0]]))
        self.assertTrue(np.isnan(values[1]))

    def test_draws_use_exact_integer_algorithm(self):
        for n, b, seed in ((700, 2000, 20260914), (100, 5000, 20260915)):
            expected = np.random.default_rng(seed).integers(0, n, size=(b, n), dtype=np.int64)
            core.validate_indices(expected, n, b, seed)
            for wrong in (expected[::-1], expected.astype(np.int32)):
                with self.assertRaises(ValueError):
                    core.validate_indices(wrong, n, b, seed)


def allocation_fixture():
    rows = manifest_fixture()
    for row in rows:
        if row['split'] == 'val':
            row['row_id'] = 'fixture_' + row['row_id']
    for i, rid in enumerate(runner.accepted.VALIDATION_ROW_IDS):
        rows[2 * i]['row_id'] = rid
    return core.allocate(rows, runner.accepted.VALIDATION_ROW_IDS,
                         {'row_ids': ['test'], 'patient_ids': ['test']})


def grid_records(allocation, phase, questions):
    conditions = [('baseline', 0.)] if phase == 'calibration' else core.CONDITIONS
    return [dict(source_commit='a' * 40, phase=phase, question=c, prompt=core.PROMPTS[c],
                 patient_index=i, patient_id=r['patient_id'], row_id=r['row_id'],
                 image_path=r['image_path'], label=r[c], direction=d, alpha=a,
                 raw_margin=0., probability=.5)
            for c in questions for d, a in conditions for i, r in enumerate(allocation[phase])]


class GridTests(unittest.TestCase):
    def test_complete_mixed_and_empty_grid(self):
        allocation = allocation_fixture()
        for questions in ([], [core.CONCEPTS[0], core.CONCEPTS[4]], list(core.CONCEPTS)):
            records = grid_records(allocation, 'write', questions)
            values = core.validate_grid(records, allocation, 'a' * 40, 'write', questions)
            self.assertEqual(values.shape, (len(questions), 45, 100))
        records = grid_records(allocation, 'calibration', list(core.CONCEPTS))
        self.assertEqual(core.validate_grid(records, allocation, 'a' * 40, 'calibration', list(core.CONCEPTS)).size, 4200)

    def test_reject_duplicate_missing_extra_nonfinite_and_identity_drift(self):
        allocation = allocation_fixture()
        records = grid_records(allocation, 'write', [core.CONCEPTS[0]])
        malformed = [records[:-1], records + [records[-1]], [records[0]] + records[:-1], records[::-1]]
        for field, value in [('source_commit', 'b' * 40), ('patient_id', 'wrong'), ('row_id', 'wrong'),
                             ('patient_index', 1), ('label', '2'), ('alpha', .5), ('direction', 'random21'),
                             ('raw_margin', 'nan'), ('raw_margin', 'inf'), ('probability', .51), ('prompt', 'other')]:
            changed = [dict(r) for r in records]
            changed[0][field] = value
            malformed.append(changed)
        for bad in malformed:
            with self.assertRaises(ValueError):
                core.validate_grid(bad, allocation, 'a' * 40, 'write', [core.CONCEPTS[0]])


class ReaderTests(unittest.TestCase):
    def test_twenty_shared_fits_train_only_and_float32_rank_scores(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(31)
        rows = [dict(row_id=str(i), split='train' if i < 80 else 'val',
                     view_AP=str(i % 2), sex_M=str((i // 2) % 2), age=str((i % 10) * 10),
                     **{c: str(i % 2) for c in core.CONCEPTS}) for i in range(100)]
        raw = rng.normal(size=(100, 9)).astype(np.float16)
        projection = rng.normal(size=(9, 5)).astype(np.float32)
        projected = raw.astype(np.float32) @ projection
        scaler = StandardScaler().fit(projected[:80])
        bundle = dict(projection=projection, scale=scaler.scale_, coefficients=rng.normal(size=(6, 5)))
        types = core.build_types(rows)
        controls = [dict(seed=i, assignment=core.control_labels(types, np.random.default_rng(i))[1]) for i in range(20)]
        with patch.object(LogisticRegression, 'fit', autospec=True, side_effect=LogisticRegression.fit) as spy:
            result = core.prepare_readers(raw, rows, [str(i) for i in range(80, 100)], bundle, controls)
        self.assertEqual(spy.call_count, 20)
        expected_x = scaler.transform(projected)
        for call in spy.call_args_list:
            np.testing.assert_array_equal(call.args[1], expected_x[:80])
            self.assertEqual(call.args[2].shape, (80,))
        np.testing.assert_array_equal(result['clinical_scores'], (expected_x[80:] @ bundle['coefficients'].T).T)
        self.assertEqual(result['control_coefficients'].shape, (20, 5))
        changed = deepcopy(rows)
        for row in changed[80:]:
            for c in core.CONCEPTS:
                row[c] = 'changed validation target'
        again = core.prepare_readers(raw, changed, [str(i) for i in range(80, 100)], bundle, controls)
        np.testing.assert_array_equal(result['control_coefficients'], again['control_coefficients'])
        bad = deepcopy(controls)
        bad[0]['assignment'][types[0]] = 1 - bad[0]['assignment'][types[0]]
        with self.assertRaises(ValueError):
            core.prepare_readers(raw, rows, [str(i) for i in range(80, 100)], bundle, bad)

    def test_direction_stream_shares_randoms_and_generates_all_shams(self):
        clinical = np.eye(6, 16)
        random, sham = core.make_directions(clinical)
        rng = np.random.default_rng(0)
        expected = np.stack([rng.standard_normal(16) for _ in range(20)])
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_array_equal(random, expected)
        np.testing.assert_array_equal(sham, np.stack([rng.permutation(v) for v in clinical]))
        arrays = dict(vectors=clinical, random_vectors=random, sham_vectors=sham)
        for c in core.CONCEPTS:
            np.testing.assert_array_equal(runner.vector_for(arrays, c, 'random19'), expected[19])


class BoundaryTests(unittest.TestCase):
    def test_cli_full_sha_environment_and_path_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ['prepare', '--data-root', directory, '--out', str(root / 'out'), '--source-commit', 'a' * 40]
            with patch.object(runner, 'prepare', return_value='prepared') as prepare:
                self.assertEqual(runner.main(args), 'prepared')
            self.assertEqual(prepare.call_args.kwargs['source_commit'], 'a' * 40)
            with patch.dict(os.environ, {'SOURCE_COMMIT': 'b' * 40}):
                with self.assertRaises(ValueError):
                    runner.paths(root, root / 'out', 'a' * 40)
            with self.assertRaises(ValueError):
                runner.paths(root, root.parent / 'out', 'a' * 40)
        for sha in ('a' * 39, 'A' * 40, 'not-a-commit'):
            with self.assertRaises(Exception):
                runner.full_sha(sha)

    def test_cpu_import_and_help_do_not_load_torch(self):
        process = subprocess.run([sys.executable, '-B', '-c',
            "import sys; import src.run_llava_validation_opportunity; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_native_processor_tuple_normalization_uses_accepted_helper(self):
        from transformers.image_utils import SizeDict
        image = SimpleNamespace(do_resize=True, size=SizeDict(shortest_edge=336), do_center_crop=True,
            crop_size=SizeDict(height=336, width=336), do_convert_rgb=True, do_normalize=True,
            do_rescale=True, resample=3, rescale_factor=1/255,
            image_mean=(.48145466, .4578275, .40821073), image_std=(.26862954, .26130258, .27577711))
        processor = SimpleNamespace(image_processor=image, patch_size=14, vision_feature_select_strategy='default',
            num_additional_image_tokens=1, image_token='<image>', chat_template='native')
        config = SimpleNamespace(vision_feature_layer=-2, vision_feature_select_strategy='default',
            vision_config=SimpleNamespace(patch_size=14, image_size=336))
        runner.accepted.validate_processor(processor, config)
        image.image_mean = (.5, .5, .5)
        with self.assertRaises(ValueError):
            runner.accepted.validate_processor(processor, config)

    def test_actual_native_scorer_steerer_recorder_binding_and_variant_max(self):
        import torch
        from src import intervene
        from transformers.image_utils import SizeDict

        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            padding_side = 'right'
            def encode(self, text, **kwargs):
                table = {'yes': 1, ' yes': 3, 'Yes': 5, ' Yes': 7, 'YES': 9, ' YES': 11,
                         'no': 2, ' no': 4, 'No': 6, ' No': 8, 'NO': 10, ' NO': 12}
                return [table[text]]

        class Block(torch.nn.Module):
            def forward(self, hidden):
                return (hidden,)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()
                self.connector = torch.nn.Identity()
                self.config = SimpleNamespace(vision_feature_layer=-2, vision_feature_select_strategy='default',
                    vision_config=SimpleNamespace(patch_size=14, image_size=336))
                self.block.register_forward_hook(self.record)

            def record(self, module, inputs, output):
                self.recorded = output[0]

            def named_modules(self, *args, **kwargs):
                return iter([(core.MODULE, self.block), ('model.multi_modal_projector', self.connector)])

            def forward(self, text, **kwargs):
                self.block(torch.ones((len(text), 577, 1024)))
                forwarded = self.connector(self.recorded[:, 1:, :])
                logits = torch.zeros((len(text), 1, 16))
                logits[:, :, 11] = 4. + forwarded.mean(dim=(1, 2))[:, None]
                logits[:, :, 2] = 2.
                return SimpleNamespace(logits=logits)

        image_processor = SimpleNamespace(do_resize=True, size=SizeDict(shortest_edge=336),
            do_center_crop=True, crop_size=SizeDict(height=336, width=336), do_convert_rgb=True,
            do_normalize=True, do_rescale=True, resample=3, rescale_factor=1/255,
            image_mean=(.48145466, .4578275, .40821073), image_std=(.26862954, .26130258, .27577711))
        processor = Mock(image_processor=image_processor, patch_size=14, num_additional_image_tokens=1,
            vision_feature_select_strategy='default', image_token='<image>', chat_template='native', tokenizer=Tokenizer())
        processor.apply_chat_template.return_value = 'native prompt'
        processor.side_effect = lambda **kwargs: Inputs(kwargs)
        model = Model()
        factories = SimpleNamespace(AutoProcessor=Mock(), AutoModelForImageTextToText=Mock())
        factories.AutoProcessor.from_pretrained.return_value = processor
        factories.AutoModelForImageTextToText.from_pretrained.return_value = model
        image = Mock()
        image.__enter__ = Mock(return_value=image)
        image.__exit__ = Mock(return_value=False)
        image.convert.return_value = 'RGB image'
        with patch.dict(sys.modules, {'transformers': factories}), patch('src.gpu_env.bind_gpu') as bind, \
             patch.object(torch.cuda, 'device_count', return_value=1), \
             patch.object(torch.cuda, 'get_device_name', return_value='NVIDIA A100'), \
             patch('PIL.Image.open', return_value=image), \
             patch.object(intervene, 'yes_margin_batch', wraps=intervene.yes_margin_batch) as margin:
            score, ids, runtime = runner.load_scorer({'model_source': 'local-snapshot'}, 3)
            rows = [{'image_path': 'one.png'}, {'image_path': 'two.png'}]
            clean, full = score(core.PROMPTS['Effusion'], rows, inspect=True)
            zero, zero_full = score(core.PROMPTS['Effusion'], rows, np.ones(1024) / 32, 0., inspect=True)
            changed, changed_full = score(core.PROMPTS['Effusion'], rows, np.ones(1024) / 32, .25, inspect=True)
        np.testing.assert_array_equal(clean, [3., 3.])
        np.testing.assert_array_equal(clean, zero)
        np.testing.assert_array_equal(full['logits'], zero_full['logits'])
        self.assertTrue(np.all(changed > clean))
        self.assertGreater(np.max(changed_full['forwarded'] - full['forwarded']), 0)
        self.assertEqual(full['forwarded'].shape, (2, 576, 1024))
        self.assertEqual(margin.call_count, 3)
        self.assertEqual(ids, [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]])
        self.assertEqual(processor.tokenizer.padding_side, 'left')
        self.assertEqual(processor.call_args.kwargs['images'], ['RGB image', 'RGB image'])
        bind.assert_called_once_with(3)
        factories.AutoProcessor.from_pretrained.assert_called_once_with('local-snapshot', local_files_only=True)
        factories.AutoModelForImageTextToText.from_pretrained.assert_called_once_with(
            'local-snapshot', dtype=torch.bfloat16, device_map='cuda:0', local_files_only=True)

    def test_budget_75_minute_inclusive_boundary(self):
        self.assertTrue(runner.budget(64., 420.).get('passed'))
        self.assertEqual(runner.budget(64., 420.)['projected_seconds'], 4500.)
        self.assertFalse(runner.budget(64., 420.001)['passed'])

    def test_json_undefined_diagnostics_are_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.json'
            runner.write_json(path, {'auc': float('nan'), 'array': np.array([np.nan, .5])})
            self.assertEqual(json.loads(path.read_text()), {'auc': None, 'array': [None, .5]})


def chain_fixture(root):
    allocation = allocation_fixture()
    training = [dict(row_id='train' + str(i), patient_id='train' + str(i), split='train') for i in range(80)]
    rows = training + allocation['calibration'] + allocation['write'] + allocation['preflight']
    for i, row in enumerate(rows):
        row.update(view_AP=str(i % 2), sex_M=str((i // 2) % 2), age=str((i % 10) * 10))
        row.setdefault('image_path', str(root / (row['row_id'] + '.png')))
        for c in core.CONCEPTS:
            row.setdefault(c, str(i % 2))
    projection = np.eye(1024, 512, dtype=np.float32)
    bundle = dict(names=np.asarray(core.CONCEPTS), projection=projection, scale=np.ones(512),
                  coefficients=np.eye(6, 512), vectors=np.eye(6, 1024, dtype=np.float32),
                  locus=np.asarray('vis.last'), raw_dim=np.asarray(1024), projection_dim=np.asarray(512),
                  projection_seed=np.asarray(0), C=np.asarray(1.), probe_seed=np.asarray(0))
    raw = np.zeros((len(rows), 1024), dtype=np.float16)
    for i, row in enumerate(rows[80:780], 80):
        raw[i, :6] = [2 * int(row[c]) - 1 for c in core.CONCEPTS]
    acts = root / 'accepted/activations'
    acts.mkdir(parents=True)
    np.savez_compressed(acts / 'shard0.npz', row_id=np.asarray([r['row_id'] for r in rows]), **{'vis.last': raw})
    types = core.build_types(rows)
    controls = [dict(seed=i, assignment=core.control_labels(types, np.random.default_rng(i))[1]) for i in range(20)]
    sources = {'model': {'model_source': str(root / 'local-snapshot')}, 'extract_path': str(acts / 'meta.json'),
               'probe': {'controls': controls}, 'manifest': str(root / 'manifest.csv')}
    return sources, bundle, rows, allocation


def synthetic_scorer(source, gpu, selected=core.CONCEPTS):
    def score(prompt, rows=(), vector=None, alpha=0., inspect=False):
        if rows:
            c = next(c for c in core.CONCEPTS if core.PROMPTS[c] == prompt)
            margin = np.array([(2 * int(r[c]) - 1) * .1 if c in selected else 0. for r in rows])
            if vector is not None:
                margin += alpha * float(vector[list(core.CONCEPTS).index(c)]) * 4
        else:
            margin = np.asarray([-1. if 'absent' in prompt.split('.')[0] else 1.])
        if inspect:
            # Tensor capture is the model boundary; statistics, allocation and grids remain real.
            value = np.full((len(rows), 1, 2), 1. + (alpha if vector is not None else 0.))
            return margin, {'logits': value, 'forwarded': value}
        return margin
    return score, [[1], [2]], {'gpu_name': 'NVIDIA A100', 'gpu_count': 1, 'chat_template': 'native',
                              'module': core.MODULE, 'positions': 'all', 'model_source': source['model_source']}


class ChainTests(unittest.TestCase):
    def test_prepare_run_replay_mixed_and_empty_k_without_refits_or_model_in_summary(self):
        from sklearn.linear_model import LogisticRegression
        for selected in ([core.CONCEPTS[0], core.CONCEPTS[4]], []):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = chain_fixture(root)
                out = root / 'out'
                with patch.object(runner, 'load_sources', return_value=fixture):
                    runner.prepare(root, out, 'a' * 40)
                    with patch.object(LogisticRegression, 'fit', side_effect=AssertionError('GPU refit')), \
                         patch.object(runner, 'load_scorer', side_effect=lambda s, g: synthetic_scorer(s, g, selected)):
                        runner.run(root, out, 'a' * 40, 0)
                    with patch.object(runner, 'load_scorer', side_effect=AssertionError('summary model load')), \
                         patch.object(LogisticRegression, 'fit', side_effect=AssertionError('summary refit')), \
                         patch('PIL.Image.open', side_effect=AssertionError('CPU replay opened raw image')):
                        result = runner.summarize(root, out, 'a' * 40)
                    self.assertEqual(result['qualification']['qualified_questions'], selected)
                    self.assertEqual(result['eligible_questions'], selected)
                    self.assertEqual(result['n_outcomes'], 4200 + 4500 * len(selected))
                    self.assertEqual(result['route'], 'test_cohort_competition_registration' if selected else 'evidence_synthesis')
                    with np.load(out / (runner.SUMMARY + '.npz')) as archive:
                        before = {k: archive[k] for k in archive.files}
                    replay = runner.summarize(root, out, 'a' * 40)
                    self.assertEqual(runner.json_value(result), runner.json_value(replay))
                    with np.load(out / (runner.SUMMARY + '.npz')) as archive:
                        for key, value in before.items():
                            np.testing.assert_array_equal(archive[key], value)
                    saved = runner.read_json(out / 'qualification.json')
                    saved['K'] += 1
                    runner.write_json(out / 'qualification.json', saved)
                    with self.assertRaisesRegex(ValueError, 'qualification'):
                        runner.summarize(root, out, 'a' * 40)

    def test_prepared_identity_rejects_source_cohort_probe_labels_and_control_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = chain_fixture(root)
            out = root / 'out'
            with patch.object(runner, 'load_sources', return_value=fixture):
                runner.prepare(root, out, 'a' * 40)
                meta = runner.read_json(out / 'prepared.json')
                for mutate in (lambda m: m.update(source_commit='b' * 40),
                               lambda m: m['fit'].update(training_split='val'),
                               lambda m: m['protocol'].update(module='wrong'),
                               lambda m: m['cohort']['write'].reverse(),
                               lambda m: m['sources']['probe'].update(controls=[])):
                    changed = deepcopy(meta)
                    mutate(changed)
                    runner.write_json(out / 'prepared.json', changed)
                    with self.assertRaises(ValueError):
                        runner.load_prepared(root, out, 'a' * 40)
                runner.write_json(out / 'prepared.json', meta)
                with np.load(out / 'prepared.npz') as archive:
                    original = {k: archive[k] for k in archive.files}
                for key in ('vectors', 'clinical_scores', 'control_scores', 'control_labels', 'train_mean',
                            'calibration_projected', 'bootstrap_selectivity', 'bootstrap_control_auroc',
                            'control_train_labels', 'random_vectors', 'sham_vectors', 'calibration_indices',
                            'write_indices', 'calibration_labels', 'write_labels'):
                    changed = {k: v.copy() for k, v in original.items()}
                    changed[key].flat[0] += 1
                    np.savez_compressed(out / 'prepared.npz', **changed)
                    with self.subTest(array=key), self.assertRaises(ValueError):
                        runner.load_prepared(root, out, 'a' * 40)

    def test_preflight_failure_preserves_receipt_before_any_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = chain_fixture(root)
            out = root / 'out'
            with patch.object(runner, 'load_sources', return_value=fixture):
                runner.prepare(root, out, 'a' * 40)
                with patch.object(runner, 'load_scorer', side_effect=ValueError('model unavailable')):
                    with self.assertRaises(ValueError):
                        runner.run(root, out, 'a' * 40, 0)
                self.assertFalse(runner.read_json(out / 'preflight.json')['passed'])
                self.assertFalse((out / 'calibration.csv').exists())

    def test_failed_mapping_preserves_measurements_and_preflight_only_has_no_grid(self):
        for failure in (True, False):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = chain_fixture(root)
                out = root / 'out'
                with patch.object(runner, 'load_sources', return_value=fixture):
                    runner.prepare(root, out, 'a' * 40)
                    score, tokens, runtime = synthetic_scorer(fixture[0]['model'], 0)
                    def adapted(prompt, rows=(), *args, **kwargs):
                        return np.array([0.]) if failure and not rows else score(prompt, rows, *args, **kwargs)
                    with patch.object(runner, 'load_scorer', return_value=(adapted, tokens, runtime)):
                        if failure:
                            with self.assertRaises(ValueError):
                                runner.run(root, out, 'a' * 40, 0, preflight_only=True)
                        else:
                            runner.run(root, out, 'a' * 40, 0, preflight_only=True)
                    flight = runner.read_json(out / 'preflight.json')
                    self.assertEqual(flight['passed'], not failure)
                    self.assertEqual(len(flight['mapping_cases']), 4)
                    self.assertIn('throughput', flight)
                    self.assertFalse((out / 'calibration.csv').exists())
                    self.assertFalse((out / 'qualification.json').exists())


def source_fixture(root):
    """Accepted metadata layout with synthetic identities and absent raw images."""
    fixture = chain_fixture(root)
    bundle = fixture[1]
    rows = []
    for split, n, patients in (('train', 18212, 5783), ('val', 2674, 855), ('test', 5343, 1659)):
        for i in range(n):
            t = len(rows) % 39
            rows.append(dict(row_id=f'{split}_{i:08d}', patient_id=f'{split}_{i % patients}', split=split,
                image_path=str(root / 'missing-images' / f'{split}_{i:08d}.png'),
                view_AP=str(t // 20), sex_M=str((t // 10) % 2), age=str((t % 10) * 10),
                **{c: str(i % 2) for c in core.CONCEPTS}))
    validation = [r for r in rows if r['split'] == 'val']
    for row, rid in zip(validation, runner.accepted.VALIDATION_ROW_IDS):
        row['row_id'] = rid
    manifest = root / 'datasets/nih-chestxray14/manifest.csv'
    manifest.parent.mkdir(parents=True)
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first = root / 'runs' / runner.accepted.ASSET_RUN_ID
    own = root / 'runs' / runner.OWNERSHIP_RUN_ID
    for path, run_id, commit in ((first, runner.accepted.ASSET_RUN_ID, runner.accepted.ASSET_COMMIT),
                                 (own, runner.OWNERSHIP_RUN_ID, runner.OWNERSHIP_COMMIT)):
        path.mkdir(parents=True)
        metadata = dict(RUN_ID=run_id, SOURCE_COMMIT=commit, COMMAND_STATUS='0',
                        DISPATCHER_STATUS='0', CLEANUP_STATUS='0', ABORT_SIGNAL='none')
        (path / 'metadata.env').write_text('\n'.join(f'{k}={v}' for k, v in metadata.items()))
        for name in ('command_exit_status', 'exit_status'):
            (path / name).write_text('0\n')
    model_root = root / 'models/huggingface'
    snapshot = model_root / runner.accepted.SNAPSHOT_RELATIVE_PATH
    snapshot.mkdir(parents=True)
    receipt = model_root / 'asset-receipt.json'
    runner.write_json(receipt, dict(asset=runner.accepted.ARCH, repo_id=runner.accepted.MODEL_ID,
        revision=runner.accepted.MODEL_REVISION, snapshot_relative_path=runner.accepted.SNAPSHOT_RELATIVE_PATH))
    artifacts = first / 'artifacts'
    runner.write_json(artifacts / 'input-verification.json', dict(model_id=runner.accepted.MODEL_ID,
        model_revision=runner.accepted.MODEL_REVISION, source_commit=runner.accepted.ASSET_COMMIT,
        vision_feature_layer=-2, snapshot=str(snapshot), model_receipt=str(receipt), model_receipt_sha256='accepted'))
    runner.write_json(artifacts / 'activations/meta.json', dict(arch=runner.accepted.ARCH,
        git_sha=runner.accepted.ASSET_COMMIT, model_source=str(snapshot), model_local_only=True,
        n_rows=26229, loci=['vis.last'], prompt=core.PROMPTS['Effusion']))
    runner.write_json(artifacts / 'hook-verification.json', dict(module=core.MODULE,
        forward_path_proved=True, alpha_zero_bitwise_noop=True))
    types = core.build_types(rows)
    controls = [dict(seed=i, assignment=core.control_labels(types, np.random.default_rng(i))[1]) for i in range(20)]
    runner.write_json(artifacts / 'probe/probe.json', dict(arch=runner.accepted.ARCH,
        source_commit=runner.accepted.ASSET_COMMIT, locus='vis.last', module=core.MODULE,
        projection_dim=512, projection_seed=0, C=1., probe_seed=0,
        type_columns=['view_AP', 'sex_M', 'age'], control_seeds=list(range(20)),
        direction_names=list(core.CONCEPTS), n_train=18212, n_test=5343,
        directions=str(artifacts / 'probe/directions.npz'), directions_sha256='accepted', controls=controls))
    np.savez_compressed(artifacts / 'probe/directions.npz', **bundle)
    selected = [r for r in rows if r['split'] == 'test'][:400]
    runner.write_json(own / 'artifacts/registered-rows.json', dict(row_ids=[r['row_id'] for r in selected],
        patient_ids=[r['patient_id'] for r in selected]))
    return manifest, artifacts, rows


class SourceTests(unittest.TestCase):
    def test_source_loader_preserves_binding_with_all_raw_images_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, artifacts, rows = source_fixture(root)
            # The synthetic patient names replace the four clinical metadata anchors.
            with patch.object(core, 'validate_feasibility'):
                sources, bundle, loaded, allocation = runner.load_sources(root)
            self.assertEqual(len(loaded), 26229)
            self.assertEqual(len(allocation['calibration']), 700)
            self.assertFalse(Path(allocation['calibration'][0]['image_path']).exists())
            self.assertEqual(sources['model']['model_id'], runner.accepted.MODEL_ID)
            self.assertEqual(bundle['vectors'].shape, (6, 1024))
            selected = allocation['write'][0]['row_id']
            for row in rows:
                if row['row_id'] == selected:
                    row['image_path'] = str(root.parent / 'outside.png')
            with manifest.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with patch.object(core, 'validate_feasibility'), self.assertRaisesRegex(ValueError, 'escapes'):
                runner.load_sources(root)

    def test_source_loader_rejects_probe_model_and_terminal_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, artifacts, _ = source_fixture(root)
            for name, field, value in (('probe/probe.json', 'source_commit', 'b' * 40),
                                       ('activations/meta.json', 'model_local_only', False),
                                       ('hook-verification.json', 'module', 'wrong'),
                                       ('input-verification.json', 'model_revision', 'wrong')):
                path = artifacts / name
                original = runner.read_json(path)
                changed = deepcopy(original)
                changed[field] = value
                runner.write_json(path, changed)
                with patch.object(core, 'validate_feasibility'), self.assertRaises(ValueError):
                    runner.load_sources(root)
                runner.write_json(path, original)
            (artifacts.parent / 'exit_status').write_text('1\n')
            with self.assertRaises(ValueError):
                runner.load_sources(root)


if __name__ == '__main__':
    unittest.main()
