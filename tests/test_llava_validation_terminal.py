from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.server import validate_llava_validation_opportunity as validator
from src import llava_validation_opportunity as core


class TerminalManifestTests(unittest.TestCase):
    def test_reads_dispatcher_manifest_and_rejects_duplicate_or_malformed_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'SHA256SUMS'
            path.write_text('a' * 64 + '  ./metadata.env\n' + 'b' * 64 + '  *artifacts/summary.json\n')
            self.assertEqual(validator.manifest_inventory(path), {
                'metadata.env': 'a' * 64, 'artifacts/summary.json': 'b' * 64})
            for text in ('short  ./metadata.env\n',
                         'a' * 64 + '  ./metadata.env\n' + 'b' * 64 + '  metadata.env\n'):
                path.write_text(text)
                with self.assertRaises(AssertionError):
                    validator.manifest_inventory(path)


class TerminalArrayShapeTests(unittest.TestCase):
    def prepared(self):
        return {
            'calibration_labels': np.zeros((6, 700), dtype=np.int8),
            'clinical_scores': np.zeros((6, 700)),
            'control_scores': np.zeros((20, 700)),
            'control_labels': np.zeros((20, 700), dtype=np.int8),
            'control_coefficients': np.zeros((20, 512)),
            'control_intercepts': np.zeros(20),
            'calibration_features': np.zeros((700, 512)),
            'calibration_projected': np.zeros((700, 512)),
            'train_mean': np.zeros(512),
            'coefficients': np.zeros((6, 512)),
            'calibration_indices': np.zeros((2000, 700), dtype=np.int64),
            'write_indices': np.zeros((5000, 100), dtype=np.int64),
            'bootstrap_selectivity': np.zeros((6, 2000)),
            'bootstrap_clinical_auroc': np.zeros((6, 2000)),
            'bootstrap_control_auroc': np.zeros((20, 2000)),
            'clinical_auroc': np.zeros(6),
            'control_auroc': np.zeros(20),
        }

    def test_exact_twenty_control_family_and_coherent_shapes(self):
        arrays = self.prepared()
        validator.verify_prepared_shapes(arrays)
        for key in ('control_scores', 'control_labels', 'control_coefficients',
                    'control_intercepts', 'bootstrap_control_auroc'):
            changed = dict(arrays)
            changed[key] = changed[key][:-1]
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validator.verify_prepared_shapes(changed)


class TerminalQualificationTests(unittest.TestCase):
    def fixture(self):
        labels = np.tile(np.arange(700) % 2, (6, 1))
        margin = np.where(labels, 1., -1.)
        real = np.full((6, 2001), .8)
        control = np.full((20, 2001), .5)
        selectivity = np.full((6, 2000), .3)
        capability = np.full((6, 2001), .75)
        brier = float(np.mean((validator.probability(margin[0]) - labels[0]) ** 2))
        control_ci = [[.5, .5] for _ in range(20)]
        questions = []
        for question in validator.CONCEPTS:
            questions.append({
                'question': question, 'positive': 350, 'negative': 350,
                'reader_valid_draws': 2000, 'reader_lower_95': .3, 'reader_pass': True,
                'capability_valid_draws': 2000, 'capability_lower_95': .75,
                'capability_pass': True, 'qualified': True, 'reader_auroc': .8,
                'reader_auroc_ci95': [.8, .8], 'control_auroc': [.5] * 20,
                'control_auroc_ci95': control_ci, 'selectivity': .3,
                'selectivity_ci95': [.3, .3], 'capability_auroc': .75,
                'capability_auroc_ci95': [.75, .75], 'capability_brier': brier,
            })
        report = {'questions': questions, 'qualified_questions': list(validator.CONCEPTS),
                  'K': 6, 'expected_outcomes': 31200, 'source_commit': 'a' * 40}
        return report, labels, real, control, selectivity, capability, margin

    def test_replays_every_reported_qualification_statistic(self):
        args = self.fixture()
        self.assertEqual(validator.verify_qualification_report(*args, 'a' * 40), validator.CONCEPTS)
        for field in ('reader_auroc', 'control_auroc_ci95', 'selectivity_ci95',
                      'capability_auroc', 'capability_brier'):
            changed = deepcopy(args[0])
            changed['questions'][0][field] = 0.
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validator.verify_qualification_report(changed, *args[1:], 'a' * 40)


class TerminalWriteTests(unittest.TestCase):
    def fixture(self):
        margins = np.zeros((45, 100))
        margins[1], margins[2] = -1., 1.
        for start in range(3, 43, 2):
            margins[start], margins[start + 1] = -.1, .1
        margins[43], margins[44] = -.2, .2
        labels = np.arange(100) % 2
        indices = np.random.default_rng(20260915).integers(0, 100, size=(5000, 100), dtype=np.int64)
        report, arrays = core.write_statistics(margins, labels, indices)
        saved = {'Effusion__' + key: value for key, value in arrays.items()}
        return report, saved, margins, labels, indices

    def test_replays_diagnostics_changes_and_summary_npz_identity(self):
        args = self.fixture()
        self.assertTrue(validator.verify_write_report(args[0], args[1], 'Effusion', *args[2:]))
        mutations = (
            lambda r: r['diagnostics'][0].update(brier=.9),
            lambda r: r['probability']['positive_changes'].update(clinical=-1.),
            lambda r: r['margin'].update(dose_means=[[0., 0.]] * 22),
        )
        for mutate in mutations:
            changed = deepcopy(args[0])
            mutate(changed)
            with self.assertRaises(AssertionError):
                validator.verify_write_report(changed, args[1], 'Effusion', *args[2:])
        saved = {'source_commit': np.asarray('a' * 40),
                 'qualified_questions': np.asarray(['Effusion']),
                 'eligible_questions': np.asarray(['Effusion']),
                 'route': np.asarray('test_cohort_competition_registration')}
        validator.verify_summary_npz_identity(
            saved, 'a' * 40, ['Effusion'], ['Effusion'], 'test_cohort_competition_registration')
        saved['route'] = np.asarray('evidence_synthesis')
        with self.assertRaises(AssertionError):
            validator.verify_summary_npz_identity(
                saved, 'a' * 40, ['Effusion'], ['Effusion'], 'test_cohort_competition_registration')


if __name__ == '__main__':
    unittest.main()
