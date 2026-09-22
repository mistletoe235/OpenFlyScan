import dataclasses
from collections import OrderedDict
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as distributed

from openflyscan.quality_predictor.observation_source import sample_window
from openflyscan.quality_predictor.training_source import SourceV3, concatenate_pairs, resample_missing
from openflyscan.quality_predictor.base_supervision import FullLabelPolicy
from openflyscan.quality_predictor.training_base import labels_to_device, modality_inputs, ranking_loss
from openflyscan.quality_predictor.training import QualityPredictorTrainingConfig, QualityPredictor, cross_ranking_loss, gather_total, training_loss
from openflyscan.quality_predictor.model import LocalQueryInputs
from openflyscan.quality_predictor.sparse_supervision import (
    corresponding_missing_labels, dense_source_patch_index, pseudo_targets, source_patch_index,
)
from openflyscan.quality_predictor.modality_controls import grouped_mean_inputs
from scripts.train_quality_predictor import isolate_kernel_cache


def gather_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=2)
    prediction = torch.arange(1, rank+3, dtype=torch.float32, requires_grad=True)
    labels = dict(groups=torch.full((len(prediction),), rank, dtype=torch.long))
    merged, merged_labels = gather_total(prediction, labels)
    merged.square().mean().backward()
    torch.save(dict(merged=merged.detach(), labels=merged_labels, gradient=prediction.grad),
               str(Path(output)/f"rank{rank}.pt"))
    distributed.destroy_process_group()


def fixture():
    maps = np.zeros((4, 2, 3, 3), np.float32)
    maps[:, 1, :, 0] = 10
    maps[0, 1] = np.nan
    maps[3, 0] = np.nan
    cloud = dict(maps=maps, stems=["a", "b", "c", "d"], geometry_evidence_audit=[])
    proposal = dict(centers=np.array([[0., 0, 0], [10., 0, 0]]), origin=np.zeros(3), cell_size=.3)
    labels = dict(target=np.array([-2.0, -1.5], np.float32), mask=np.ones(2, bool),
                  rank_mask=np.ones(2, bool), weight=np.ones(2, np.float32), audit={"rows": []})
    missing = dict(maps=maps[1:], stems=["b", "c", "d"], geometry_evidence_audit=[])
    return cloud, proposal, labels, missing


def inputs(regions=2, views=4):
    config = QualityPredictorTrainingConfig().model_config()
    return LocalQueryInputs(
        torch.randn(regions, views, 1024), torch.randn(regions, views, 1024), torch.randn(regions, views, 1024),
        torch.randn(regions, views, 10), torch.randn(regions, views, config.quality_geometry_dim),
        torch.randn(regions, views, 6), torch.randn(regions, config.query_dim), torch.ones(regions, views, dtype=torch.bool))


class WeakTargetTests(unittest.TestCase):
    def test_dense_reference_covers_unsampled_full_queries(self):
        cloud, proposal, labels, missing = fixture()
        values = np.full(cloud["maps"].shape[:3], -2.)
        values[:, 1] = -1.5
        only_first_query = {**proposal, "centers": proposal["centers"][:1]}
        reference, dense_labels = dense_source_patch_index(cloud, only_first_query, values)
        result = corresponding_missing_labels(reference, dense_labels, missing, proposal, ["a"], (-3., -1.))
        np.testing.assert_array_equal(result["mask"], [True, True])
        np.testing.assert_array_equal(result["affected"], [True, False])
        self.assertNotIn("maps", reference)

    def test_calibrated_units_and_saturation(self):
        result = pseudo_targets(np.full(4, -2.), np.array([2, 3, 10, 12])/30, np.ones(4), (-3., -1.))
        np.testing.assert_allclose(result["penalty_db"], [1.475775, 1.784474, 2.191495, 2.197200], atol=.001)
        self.assertTrue(np.all(result["target"] > .5))

    def test_quality_variation_does_not_reject_matched_neighbourhood(self):
        cloud, proposal, labels, missing = fixture()
        cloud["maps"][:, 1, :, 0] = .4
        missing["maps"] = cloud["maps"][1:]
        values = np.full(cloud["maps"].shape[:3], -2.)
        values[:, 1] = -1.5
        reference, dense_labels = dense_source_patch_index(cloud, proposal, values)
        neighbourhood = dict(centers=np.array([[.2, 0., 0.]]), cell_size=.3)
        result = corresponding_missing_labels(reference, dense_labels, missing, neighbourhood,
                                              ["a"], (-3., -1.))
        self.assertTrue(result["mask"][0])
        self.assertTrue(result["rank_mask"][0])
        self.assertAlmostEqual(result["audit"]["rows"][0]["full_quality_log_mse_range"], .5)
        self.assertAlmostEqual(result["raw"][0]-result["penalty_db"][0]/10, -1.75)

    def test_baseline_before_clipping_and_unclipped_ranking(self):
        result = pseudo_targets([-4., -.9], [.4, .4], [1, 1], (-3., -1.))
        self.assertEqual(result["target"][0], 0)
        self.assertEqual(result["target"][1], 1)
        self.assertLess(result["rank_target"][0], 0)
        self.assertGreater(result["rank_target"][1], 1)

    def test_unaffected_keeps_baseline(self):
        result = pseudo_targets([-2.], [.5], [False], (-3., -1.))
        self.assertEqual(result["target"][0], .5)

    def test_invalid_fraction_rejected(self):
        with self.assertRaises(ValueError):
            pseudo_targets([-2.], [1.1], [True], (-3., -1.))

    def test_correspondence_not_whole_chunk_penalty(self):
        cloud, proposal, labels, missing = fixture()
        result = corresponding_missing_labels(source_patch_index(cloud, proposal), labels, missing,
                                              proposal, ["a"], (-3., -1.))
        np.testing.assert_array_equal(result["mask"], [True, True])
        np.testing.assert_array_equal(result["affected"], [True, False])
        np.testing.assert_allclose(result["fraction"], [1/3, 0])
        self.assertGreater(result["target"][0], .5)
        self.assertEqual(result["target"][1], .75)

    def test_unknown_full_and_single_view_rank_excluded(self):
        cloud, proposal, labels, missing = fixture()
        labels["mask"][0] = False
        labels["rank_mask"][1] = False
        result = corresponding_missing_labels(source_patch_index(cloud, proposal), labels, missing,
                                              proposal, ["a"], (-3., -1.))
        np.testing.assert_array_equal(result["mask"], [False, True])
        self.assertFalse(result["rank_mask"].any())

    def test_bad_deleted_list_rejected(self):
        cloud, proposal, labels, missing = fixture()
        with self.assertRaises(ValueError):
            corresponding_missing_labels(source_patch_index(cloud, proposal), labels, missing,
                                         proposal, ["b"], (-3., -1.))

    def test_ambiguous_or_disappeared_region_is_unknown(self):
        cloud, proposal, labels, missing = fixture()
        unknown = dict(centers=np.array([[100., 0, 0]]), cell_size=.3)
        result = corresponding_missing_labels(source_patch_index(cloud, proposal), labels, missing,
                                              unknown, ["a"], (-3., -1.))
        self.assertFalse(result["mask"].any())


class SamplingCacheTests(unittest.TestCase):
    def test_kernel_cache_isolated_by_global_not_local_rank(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict("os.environ"):
            first = isolate_kernel_cache(root, 0)
            other_node = isolate_kernel_cache(root, 8)
            self.assertNotEqual(first, other_node)
            self.assertTrue(first.is_dir())
            self.assertTrue(other_node.is_dir())
            self.assertEqual(isolate_kernel_cache(root, 0), first)

    def record(self):
        centers = np.stack([np.arange(80), np.sin(np.arange(80))], -1)
        return dict(footprint_data=dict(centers=centers, bbox_mins=centers-3,
                                       bbox_maxs=centers+3, stems=np.array([str(index) for index in range(80)])),
                    weak_indices=np.array([], int))

    def test_resampled_deletions_stay_in_shifted_core(self):
        record = self.record()
        template = sample_window(record, np.random.default_rng(1))
        choices = [resample_missing(record, template, np.random.default_rng(seed)) for seed in range(12)]
        for choice in choices:
            self.assertEqual(len(choice["full"]), 30)
            self.assertIn(len(choice["deleted"]), (10, 11, 12))
            self.assertTrue(set(choice["deleted"]) <= set(choice["full"][:21]))
            self.assertEqual(set(choice["full"])-set(choice["deleted"]), set(choice["missing"]))
        self.assertGreater(len({tuple(choice["deleted"]) for choice in choices}), 1)

    def test_reuse_schedule_and_restart(self):
        source = SourceV3.__new__(SourceV3)
        source.train = ["a", "b", "c"]
        source.records = {"a": self.record()}
        source.training_config = QualityPredictorTrainingConfig()
        first = source.choice_for_step("a", 0, 0, 0, 1)
        reused = source.choice_for_step("a", 3, 0, 0, 1)
        replaced = source.choice_for_step("a", 24, 0, 0, 1)
        self.assertEqual(first["full_sample_seed"], reused["full_sample_seed"])
        self.assertNotEqual(first["full_sample_seed"], replaced["full_sample_seed"])
        self.assertEqual(first["full_seed"], replaced["full_seed"])
        self.assertEqual(reused, source.choice_for_step("a", 3, 0, 0, 1))

    def test_full_cache_hit_bypasses_old_pair_and_fresh_missing(self):
        cloud, proposal, labels, missing = fixture()
        source = SourceV3.__new__(SourceV3)
        source.training_config = QualityPredictorTrainingConfig()
        source.full_cache = OrderedDict()
        source.cache_capacity = 2
        source.v3_counts = dict.fromkeys(["full_hits", "full_misses", "full_evictions", "missing_inferences",
                                         "affected", "unaffected", "unknown"], 0)
        source.full_label_policy = FullLabelPolicy()
        source.records = {"scene": {"teacher_mapping": "raw_to_rectified"}}
        from openflyscan.quality_predictor.sparse_supervision import ObservationPolicy
        source.observation_policy = ObservationPolicy()
        source.quality_limits = np.array([-3., -1.])
        full_inputs, missing_inputs = inputs(), inputs(views=3)
        values = np.full(cloud["maps"].shape[:3], -2.)
        values[:, 1] = -1.5
        cloud["full_teacher_patches"] = (values, None, None)
        source.full_pixel_labels = mock.Mock(return_value=values)
        source.extract = mock.Mock(side_effect=lambda scene, stems, seed:
                                   (full_inputs, cloud, proposal) if len(stems) == 4 else (missing_inputs, missing, proposal))
        choice = dict(full=cloud["stems"], missing=missing["stems"], deleted=["a"], full_seed=22)
        with mock.patch("openflyscan.quality_predictor.training_source.full_region_supervision", return_value=labels), \
             mock.patch("openflyscan.quality_predictor.paired_observation_source.SourceV2.pair", side_effect=AssertionError("old geometry path")):
            first = source.pair("scene", np.random.default_rng(1), choice=choice)
            second = source.pair("scene", np.random.default_rng(2), choice=choice)
            self.assertFalse(first[2]["full_cache_hit"])
            self.assertTrue(second[2]["full_cache_hit"])
            self.assertEqual(source.extract.call_count, 3)
            self.assertEqual(source.v3_counts["missing_inferences"], 2)
            np.testing.assert_array_equal(first[1]["total"], second[1]["total"])
            cached = source.full_cache[("scene", 0)]
            self.assertNotIn("maps", cached)
            self.assertNotIn("cloud", cached)
            self.assertFalse(cached["inputs"].point_features.requires_grad)
            self.assertTrue(source.smoke_cache_requirements()["passed"])
            source.pair("scene", np.random.default_rng(2), choice={**choice, "full_seed": 23})
            self.assertEqual(source.extract.call_count, 5)


class SingleRiskLossTests(unittest.TestCase):
    def test_batched_cross_ranking_matches_loop_and_gradients(self):
        torch.manual_seed(31)
        prediction = torch.rand(13, requires_grad=True)
        target = torch.rand(13)
        groups = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        mask = torch.tensor([True, False, True, True, True, True, False, True, True, True, True, True, True])
        config = QualityPredictorTrainingConfig()
        original, counts = ranking_loss(prediction, target, mask, groups, config, cross=True, scene_ids=groups % 2)
        optimized, optimized_counts = cross_ranking_loss(prediction, target, mask, groups, config, scene_ids=groups % 2)
        torch.testing.assert_close(original, optimized)
        self.assertEqual(counts, optimized_counts)
        torch.testing.assert_close(torch.autograd.grad(original, prediction)[0], torch.autograd.grad(optimized, prediction)[0])

    def test_variable_length_distributed_gather_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            workers = [context.Process(target=gather_worker, args=(rank, str(Path(directory)/"init"), directory))
                       for rank in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(60)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                self.assertEqual(worker.exitcode, 0)
            for rank in range(2):
                result = torch.load(Path(directory)/f"rank{rank}.pt", weights_only=True)
                torch.testing.assert_close(result["merged"], torch.tensor([1., 2., 1., 2., 3.]))
                torch.testing.assert_close(result["gradient"], 4*torch.arange(1, rank+3)/5)

    def test_mean_ablation_preserves_padding_and_other_inputs(self):
        packed = inputs(regions=4)
        packed.view_mask[0, 0] = False
        result = grouped_mean_inputs(packed, torch.tensor([0, 0, 1, 1]), ["confidence_features"])
        torch.testing.assert_close(result.confidence_features[0, 0], packed.confidence_features[0, 0])
        torch.testing.assert_close(result.point_features, packed.point_features)
        torch.testing.assert_close(result.view_mask, packed.view_mask)
        self.assertFalse(torch.equal(result.confidence_features, packed.confidence_features))

    def test_config_forbids_old_supervision(self):
        with self.assertRaises(ValueError):
            QualityPredictorTrainingConfig(missing_structure_regression_weight=.1)

    def labels(self):
        return dict(total=torch.tensor([.1, .8, .2, .75]), rank_target=torch.tensor([.1, .8, .2, .75]),
                    total_mask=torch.ones(4, dtype=torch.bool), total_rank_mask=torch.ones(4, dtype=torch.bool),
                    total_weight=torch.ones(4), state=torch.tensor([0, 0, 1, 1]), groups=torch.tensor([0, 0, 1, 1]))

    def test_uniform_missing_risk_is_penalized_both_directions(self):
        config = QualityPredictorTrainingConfig()
        labels = self.labels()
        correct = {"total": labels["total"].clone().requires_grad_()}
        too_high = {"total": torch.tensor([.1, .8, .99, .99], requires_grad=True)}
        too_low = {"total": torch.tensor([.1, .8, 0., 0.], requires_grad=True)}
        baseline = training_loss(correct, labels, config)[0]
        self.assertGreater(training_loss(too_high, labels, config)[0], baseline)
        self.assertGreater(training_loss(too_low, labels, config)[0], baseline)

    def test_single_output_missing_gradients_reach_modalities(self):
        torch.manual_seed(17)
        config = QualityPredictorTrainingConfig(hidden_dim=32)
        head = QualityPredictor(config)
        predictions = head(inputs(regions=4))
        self.assertEqual(set(predictions), {"total"})
        labels = self.labels()
        labels["total_mask"][:2] = False
        labels["total_rank_mask"][:2] = False
        loss = training_loss(predictions, labels, config)[0]
        loss.backward()
        for branch in ("point", "confidence", "dino", "pose_geometry", "quality_geometry", "query"):
            gradients = [value.grad.abs().sum().item() for name, value in head.named_parameters()
                         if name.startswith("head."+branch+".") and value.grad is not None]
            self.assertGreater(sum(gradients), 0, branch)

    def test_pair_consistent_dropout_and_no_teacher_input(self):
        packed = inputs(regions=4)
        config = QualityPredictorTrainingConfig(dino_dropout=1., confidence_dropout=0.)
        dropped, _ = modality_inputs(packed, torch.zeros(4, dtype=torch.long), config)
        self.assertEqual(float(dropped.dino_features.abs().sum()), 0)
        self.assertEqual({field.name for field in dataclasses.fields(packed)},
                         {"point_features", "confidence_features", "dino_features", "pose_geometry_features",
                          "quality_geometry_features", "rgb_stats", "query_features", "view_mask"})


if __name__ == "__main__":
    unittest.main()
