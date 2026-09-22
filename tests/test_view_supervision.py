import dataclasses
from collections import OrderedDict
import unittest
from unittest import mock

import numpy as np
import torch

from test_region_features import CurrentInputTests
from test_quality_training import fixture, inputs
from openflyscan.quality_predictor.region_features import pack_current_region_inputs
from openflyscan.quality_predictor.directional_quality import view_losses
from openflyscan.quality_predictor.training_source import SourceV3, concatenate_pairs
from openflyscan.quality_predictor.base_supervision import FullLabelPolicy
from openflyscan.quality_predictor.training import QualityPredictorTrainingConfig, QualityPredictor, training_loss
from openflyscan.quality_predictor.training_base import labels_to_device
from openflyscan.quality_predictor.sparse_supervision import ObservationPolicy
from openflyscan.quality_predictor.view_supervision import pack_source_supported_inputs, feature_matched_view_labels


def configuration():
    return QualityPredictorTrainingConfig(view_auxiliary=True, source_patch_recovery=True,
                            full_view_regression_weight=.05, full_view_ranking_weight=.0125)


class ViewAuxiliaryTests(unittest.TestCase):
    def test_recovery_preserves_original_and_teaches_exact_source(self):
        cloud, proposal, features, rgb = CurrentInputTests().make_sample()
        proposal['centers'] = np.array([[0., 0., 10.], [-4.7, 0., 10.], [1000., 0., 10.]])
        original = pack_current_region_inputs(cloud, proposal, *features, rgb)
        recovered = pack_source_supported_inputs(cloud, proposal, *features, rgb)
        gained = recovered.view_mask & ~original.view_mask
        self.assertTrue(gained[1].any())
        self.assertFalse(recovered.view_mask[2].any())
        for field in dataclasses.fields(original):
            before, after = getattr(original, field.name), getattr(recovered, field.name)
            if field.name == 'query_features':
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            else:
                torch.testing.assert_close(before[original.view_mask], after[original.view_mask], rtol=0, atol=0)
        values = np.arange(cloud['maps'][..., 0].size, dtype=np.float32).reshape(cloud['maps'].shape[:3])
        labels, audit = feature_matched_view_labels(cloud, values, recovered.view_mask.shape)
        for region, view, patches, weights in cloud['region_feature_support']:
            self.assertAlmostEqual(float(labels['view_raw'][region, view]),
                                   float(np.sum(values.reshape(-1)[patches]*weights)), places=4)
        self.assertEqual(audit['recovered_view_labels'], int(gained.sum()))
        self.assertLessEqual(cloud['source_recovery_audit']['maximum_reprojection_error_pixels'], 14.)

    def test_bad_own_reprojection_is_not_recovered(self):
        cloud, proposal, features, rgb = CurrentInputTests().make_sample()
        cloud['maps'][..., 0] += 100
        proposal['centers'] = np.array([[100., 0., 10.]])
        result = pack_source_supported_inputs(cloud, proposal, *features, rgb)
        self.assertFalse(result.view_mask.any())
        self.assertEqual(cloud['source_recovery_audit']['recovered_views'], 0)

    def test_auxiliary_does_not_change_regional_forward(self):
        torch.manual_seed(55)
        baseline = QualityPredictor(QualityPredictorTrainingConfig()).eval()
        torch.manual_seed(55)
        auxiliary = QualityPredictor(configuration()).eval()
        sample = inputs()
        sample.view_mask[1] = False
        for field in dataclasses.fields(sample):
            if field.name not in ('query_features', 'view_mask'):
                getattr(sample, field.name)[1] = torch.nan
        torch.testing.assert_close(baseline(sample)['total'], auxiliary(sample)['total'], rtol=0, atol=0)
        result = auxiliary(sample)
        self.assertTrue(torch.isfinite(result['view_risk']).all())
        labels = dict(view_target=torch.tensor([[1.4, -.3, .5, .9]]*2),
                      view_rank_target=torch.tensor([[1.4, -.3, .5, .9]]*2),
                      view_target_mask=torch.ones(2, 4, dtype=torch.bool),
                      state=torch.tensor([0, 1]), groups=torch.tensor([0, 1]))
        regression, ranking, stats = view_losses(result, labels, configuration())
        (regression+ranking).backward()
        self.assertEqual(stats['views'], 4)
        for module in [auxiliary.head.point, auxiliary.head.confidence, auxiliary.head.dino,
                       auxiliary.head.structure_pool.context, auxiliary.head.appearance_pool.context,
                       auxiliary.view_head]:
            self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()
                                   if parameter.grad is not None), 0)
        self.assertIsNone(auxiliary.head.total_fuse[0].weight.grad)
        labels['state'][:] = 1
        regression, ranking, stats = view_losses(result, labels, configuration())
        self.assertEqual(float(regression+ranking), 0.)
        self.assertEqual(stats['views'], 0)

    def test_pair_preserves_all_regional_labels_and_cache(self):
        cloud, proposal, original, missing = fixture()
        values = np.full(cloud['maps'].shape[:3], -2.)
        values[0] = -1.
        cloud['full_teacher_patches'] = (values, None, None)
        cloud['region_feature_support'] = [(region, view, np.array([view*6+region*3]), np.ones(1))
                                            for region in range(2) for view in range(4)]
        cloud['source_recovery_audit'] = dict(recovered_slots=[(0, 0)], recovered_views=1)
        missing['source_recovery_audit'] = dict(recovered_slots=[], recovered_views=0)
        sample, missing_sample = inputs(), inputs(views=3)
        choice = dict(full=cloud['stems'], missing=missing['stems'], deleted=['a'])
        pairs = []
        for config in [QualityPredictorTrainingConfig(), configuration()]:
            source = SourceV3.__new__(SourceV3)
            source.training_config = config
            source.full_cache = OrderedDict()
            source.cache_capacity = 2
            source.v3_counts = dict.fromkeys(['full_hits', 'full_misses', 'full_evictions', 'missing_inferences',
                                             'affected', 'unaffected', 'unknown'], 0)
            source.full_label_policy = FullLabelPolicy()
            source.observation_policy = ObservationPolicy()
            source.records = {'scene': {'teacher_mapping': 'raw_to_rectified'}}
            source.quality_limits = np.array([-3., -1.])
            source.full_pixel_labels = mock.Mock(return_value=values)
            source.extract = mock.Mock(side_effect=lambda scene, stems, seed: (sample, cloud, proposal)
                if len(stems) == 4 else (missing_sample, missing, proposal))
            with mock.patch('openflyscan.quality_predictor.training_source.full_region_supervision', return_value=original):
                pair = source.pair('scene', np.random.default_rng(1), choice=choice)
                repeated = source.pair('scene', np.random.default_rng(1), choice=choice)
            self.assertTrue(repeated[2]['full_cache_hit'])
            self.assertEqual(source.extract.call_count, 3)
            pairs.append(pair)
        for name, value in pairs[0][1].items():
            np.testing.assert_array_equal(value, pairs[1][1][name])
        packed, labels = concatenate_pairs([pairs[1]])
        self.assertFalse(labels['view_target_mask'][labels['state'] == 1].any())
        loss, pieces, statistics = training_loss(QualityPredictor(configuration())(packed),
                                                labels_to_device(labels, 'cpu'), configuration())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(pieces), 8)
        self.assertGreater(statistics['full_view_ranking']['pairs'], 0)

    def test_configuration_rejects_mixed_directional_mode(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(configuration(), directional_quality=True)
        with self.assertRaises(ValueError):
            dataclasses.replace(configuration(), source_patch_recovery=False)


if __name__ == '__main__':
    unittest.main()
