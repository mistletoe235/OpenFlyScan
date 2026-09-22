import contextlib
import dataclasses
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from openflyscan.quality_predictor.training import QualityPredictorTrainingConfig
from openflyscan.quality_predictor.training_base import TrainingConfigV2


PROJECT = Path(__file__).resolve().parents[1]


def load_trainer(name):
    specification = importlib.util.spec_from_file_location(name, PROJECT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class PublicTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.plan = self.root / 'plan.json'
        self.config = self.root / 'config.json'
        self.output = self.root / 'run'
        self.plan.write_text(json.dumps({'root': str(self.root), 'input_policy': {'pose': 'input'}}))
        (self.root / 'UAVFF3D/GeoFF3D').mkdir(parents=True)

    def arguments(self, name):
        return [name, '--plan', str(self.plan), '--config', str(self.config),
                '--output', str(self.output), '--expected-world-size', '1', '--chunks-per-rank', '2']

    def test_direct_training_initializes_run_without_cluster_files(self):
        for name, config_type in (('train_quality_predictor', QualityPredictorTrainingConfig),
                                  ('train_quality_predictor_legacy', TrainingConfigV2)):
            with self.subTest(trainer=name):
                self.output = self.root / name
                module = load_trainer(name)
                self.config.write_text(json.dumps(dataclasses.asdict(config_type(expected_world_size=48))))
                source = types.SimpleNamespace(quality_limits=np.array([0.1, 0.9]))
                model_runner = types.ModuleType('geoff3d.slrf.model_runner')
                model_runner.init_model_from_hydra = mock.Mock(side_effect=RuntimeError('backbone startup reached'))
                model_runner.load_checkpoint = mock.Mock()
                model_runner.apply_runtime_prior_policy = mock.Mock()
                model_runner.build_prior_overrides = mock.Mock(return_value=[])
                previous_directory, previous_paths = Path.cwd(), list(sys.path)
                try:
                    with mock.patch.dict(os.environ, {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1',
                                                      'PYTORCH_KERNEL_CACHE_PATH': str(self.output / 'kernel_cache')}), \
                         mock.patch.dict(sys.modules, {'geoff3d.slrf.model_runner': model_runner}), \
                         mock.patch('sys.argv', self.arguments(name) + ['--max-steps', '100']), \
                         mock.patch.object(module, 'read_training_source', return_value=source) as read_source, \
                         mock.patch.object(module, 'training_device', return_value=torch.device('cpu')), \
                         mock.patch.object(module, 'build_data_contract', return_value={'dependency_files': {}}):
                        with self.assertRaisesRegex(RuntimeError, 'backbone startup reached'):
                            module.main()
                finally:
                    os.chdir(previous_directory)
                    sys.path[:] = previous_paths
                read_source.assert_called_once()
                model_runner.init_model_from_hydra.assert_called_once()
                manifest = json.loads((self.output / 'run_manifest.json').read_text())
                self.assertEqual(manifest['mode'], 'train')
                self.assertEqual(manifest['config']['expected_world_size'], 1)
                self.assertEqual(manifest['config']['chunks_per_rank'], 2)
                self.assertEqual(manifest['config']['steps'], 100)
                self.assertTrue((self.output / 'data_contract.json').is_file())
                self.assertTrue((self.output / 'effective_config.json').is_file())
                self.assertFalse((self.output / 'code_snapshot').exists())
                self.assertFalse(list(self.output.rglob('*authorization*')))
                if name == 'train_quality_predictor':
                    self.assertTrue((self.output / 'kernel_cache/rank0').is_dir())

    def test_existing_output_is_unchanged(self):
        module = load_trainer('train_quality_predictor')
        self.config.write_text(json.dumps(dataclasses.asdict(QualityPredictorTrainingConfig())))
        self.output.mkdir()
        sentinel = self.output / 'existing.txt'
        sentinel.write_text('keep')
        with mock.patch('sys.argv', self.arguments('train_quality_predictor')), \
             mock.patch.dict(os.environ, {'WORLD_SIZE': '1'}), \
             mock.patch.object(module, 'read_training_source') as read_source:
            with self.assertRaisesRegex(RuntimeError, 'refusing to reuse nonempty output'):
                module.main()
        read_source.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_world_size_mismatch_still_rejected(self):
        module = load_trainer('train_quality_predictor')
        self.config.write_text(json.dumps(dataclasses.asdict(QualityPredictorTrainingConfig())))
        with mock.patch('sys.argv', self.arguments('train_quality_predictor')), \
             mock.patch.dict(os.environ, {'WORLD_SIZE': '2'}), \
             mock.patch.object(module, 'read_training_source') as read_source:
            with self.assertRaisesRegex(ValueError, 'world size 2 does not match 1'):
                module.main()
        read_source.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_overrides_rejected(self):
        module = load_trainer('train_quality_predictor')
        self.config.write_text(json.dumps(dataclasses.asdict(QualityPredictorTrainingConfig())))
        for option in ('--max-steps', '--chunks-per-rank', '--expected-world-size'):
            with self.subTest(option=option), \
                 mock.patch('sys.argv', self.arguments('train_quality_predictor') + [option, '0']), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
