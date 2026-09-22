import dataclasses
import unittest
from unittest import mock

import numpy as np
import torch

from openflyscan.quality_predictor.regions import propose_current_regions
from openflyscan.quality_predictor.observation_source import Source
from openflyscan.quality_predictor.paired_observation_source import SourceV2
from openflyscan.quality_predictor.base_supervision import FullLabelPolicy, full_region_supervision
from openflyscan.quality_predictor.training_base import TrainingConfigV2, grouped_regression, training_loss
from openflyscan.quality_predictor.model import LocalQueryInputs


def fixture(patches=5):
    cloud = dict(maps=np.zeros((2, 1, 3, 3)), stems=["first", "second"])
    cloud["maps"][1, 0, 2] = [10, 10, 10] if patches == 5 else [0, 0, 0]
    proposal = dict(centers=np.zeros((1, 3)), cell_size=1.)
    values = np.full((2, 1, 3), -2.)
    cells = np.zeros((2, 1, 3, 2), int)
    tracks = {index: {100, 101, 102} for index in range(6)}
    return cloud, proposal, values, cells, tracks


def single_view_fixture():
    cloud, proposal, values, cells, _ = fixture()
    cloud["maps"][0, 0] = [[0., 0., 10.], [10., 0., 10.], [20., 0., 10.]]
    cloud["maps"][1] = 100.
    cloud["camera_poses"] = np.repeat(np.eye(4)[None], 2, axis=0)
    cloud["intrinsics"] = np.repeat(np.array([[[14., 0., 7.], [0., 14., 7.], [0., 0., 1.]]]), 2, axis=0)
    proposal.update(centers=np.array([[10., 0., 10.]]), cell_size=11., source_patch_indices=[np.array([0, 1, 2])])
    values[1] = np.nan
    return cloud, proposal, values, cells, {}


class SingleViewSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.policy = FullLabelPolicy(single_view_supervision=True)

    def test_direct_source_without_sfm_tracks_is_quarter_weight_regression_only(self):
        result = full_region_supervision(*single_view_fixture(), self.policy)
        self.assertEqual(result["target"].tolist(), [-2.])
        self.assertEqual(result["weight"].tolist(), [.25])
        self.assertFalse(result["rank_mask"][0])
        self.assertEqual(result["audit"]["single_view"], 1)
        evidence = result["audit"]["rows"][0]["single_view_evidence"]
        self.assertEqual(evidence["source_anchor_patch"], 1)
        self.assertEqual(len(evidence["teacher_cells"]), 1)

    def test_old_policy_does_not_enable_single_view(self):
        self.assertFalse(full_region_supervision(*single_view_fixture())["mask"][0])
        self.assertEqual(FullLabelPolicy(recovered_weight=.1).recovered_weight, .1)

    def test_unknown_source_or_camera_is_not_accepted(self):
        for missing in ("source_patch_indices", "camera_poses", "intrinsics"):
            cloud, proposal, values, cells, tracks = single_view_fixture()
            (proposal if missing == "source_patch_indices" else cloud).pop(missing)
            self.assertFalse(full_region_supervision(cloud, proposal, values, cells, tracks, self.policy)["mask"][0])

    def test_anchor_must_be_a_region_source_in_the_label_neighborhood(self):
        cloud, proposal, values, cells, tracks = single_view_fixture()
        proposal["source_patch_indices"] = [np.array([3])]
        result = full_region_supervision(cloud, proposal, values, cells, tracks, self.policy)
        self.assertFalse(result["mask"][0])
        self.assertEqual(result["audit"]["rows"][0]["single_view_evidence"]["reason"], "no_direct_source_patch")

    def test_source_projection_mismatch_rejected(self):
        cloud, proposal, values, cells, tracks = single_view_fixture()
        cloud["intrinsics"][0, 0, 2] += 15.4
        result = full_region_supervision(cloud, proposal, values, cells, tracks, self.policy)
        self.assertFalse(result["mask"][0])
        self.assertEqual(result["audit"]["rows"][0]["single_view_evidence"]["reason"], "source_projection_mismatch")

    def test_center_depth_mismatch_rejected(self):
        cloud, proposal, values, cells, tracks = single_view_fixture()
        proposal["centers"][0, 2] = 11.
        result = full_region_supervision(cloud, proposal, values, cells, tracks, self.policy)
        self.assertFalse(result["mask"][0])
        self.assertEqual(result["audit"]["rows"][0]["single_view_evidence"]["reason"], "source_depth_mismatch")

    def test_behind_camera_and_nonfinite_camera_rejected(self):
        for invalid in ("behind", "nonfinite"):
            cloud, proposal, values, cells, tracks = single_view_fixture()
            if invalid == "behind":
                cloud["camera_poses"][0, 2, 3] = 20.
            else:
                cloud["intrinsics"][0, 0, 0] = np.nan
            self.assertFalse(full_region_supervision(cloud, proposal, values, cells, tracks, self.policy)["mask"][0])

    def test_teacher_cells_deduplicated_before_voting(self):
        cloud, proposal, values, cells, tracks = single_view_fixture()
        values[0, 0] = [-2., -2., -1.8]
        cells[0, 0, 2] = [0, 1]
        result = full_region_supervision(cloud, proposal, values, cells, tracks, self.policy)
        self.assertAlmostEqual(float(result["target"][0]), -1.9, places=6)

    def test_disagreeing_cells_cannot_be_cherry_picked(self):
        cloud, proposal, values, cells, tracks = single_view_fixture()
        values[0, 0, 2] = -1.
        cells[0, 0, 2] = [0, 1]
        self.assertFalse(full_region_supervision(cloud, proposal, values, cells, tracks, self.policy)["mask"][0])

    def test_multiview_disagreement_cannot_fall_back_to_single(self):
        cloud, proposal, values, cells, tracks = single_view_fixture()
        values[1] = -1.
        tracks = {index: {1, 2, 3} for index in range(6)}
        result = full_region_supervision(cloud, proposal, values, cells, tracks, self.policy)
        self.assertFalse(result["mask"][0])
        self.assertNotIn("single_view_evidence", result["audit"]["rows"][0])

    def test_multiview_recovery_still_takes_precedence(self):
        result = full_region_supervision(*fixture(), self.policy)
        self.assertEqual(result["weight"].tolist(), [.5])
        self.assertEqual(result["audit"]["single_view"], 0)

    def test_single_view_weight_and_geometry_thresholds_are_bounded(self):
        for change in (dict(single_view_weight=1.), dict(single_view_weight=float("nan")),
                       dict(single_view_supervision=1), dict(maximum_single_view_projection_patches=2.),
                       dict(maximum_single_view_relative_depth_error=.1), dict(recovered_weight=.1)):
            with self.assertRaises(ValueError):
                FullLabelPolicy(**{**change, "single_view_supervision": change.get("single_view_supervision", True)})


class FullSupervisionTests(unittest.TestCase):
    def test_sparse_multiview_track_supported_target_is_downweighted(self):
        result = full_region_supervision(*fixture())
        self.assertEqual(result["target"].tolist(), [-2.])
        self.assertEqual(result["weight"].tolist(), [.5])
        self.assertFalse(result["rank_mask"][0])
        self.assertEqual(len(result["audit"]["rows"][0]["teacher_cells"]), 2)

    def test_existing_targets_and_masks_are_exactly_preserved(self):
        cloud, proposal, values, cells, tracks = fixture(6)
        values[0, 0] = [-4, -3, -2]
        original, mask = Source.region_labels(cloud, proposal, values)
        result = full_region_supervision(cloud, proposal, values, cells, tracks)
        np.testing.assert_array_equal(original, result["target"])
        np.testing.assert_array_equal(mask, result["rank_mask"])
        self.assertEqual(result["weight"].tolist(), [1.])

    def test_source_pixels_recover_correspondence_despite_spread_3d_points(self):
        cloud, proposal, values, cells, tracks = fixture()
        cloud["maps"][1] = [20, 20, 20]
        proposal["source_patch_indices"] = [np.array([0, 1, 2])]
        result = full_region_supervision(cloud, proposal, values, cells, tracks)
        self.assertTrue(result["mask"][0])
        self.assertEqual(result["audit"]["rows"][0]["views"], 1)
        self.assertEqual(result["audit"]["rows"][0]["supervised_views"], 2)
        self.assertEqual(result["audit"]["rows"][0]["recovery_method"], "source_pixel_sfm_tracks_without_3d_radius")
        disabled = full_region_supervision(cloud, proposal, values, cells, tracks, FullLabelPolicy(source_track_recovery=False))
        self.assertFalse(disabled["mask"][0])

    def test_source_track_matching_cannot_import_an_unobserved_view(self):
        cloud, proposal, values, cells, tracks = fixture()
        tracks[6] = {100, 101, 102}
        with self.assertRaisesRegex(ValueError, "current views"):
            full_region_supervision(cloud, proposal, values, cells, tracks)

    def test_source_matching_does_not_choose_anchor_by_teacher_score(self):
        cloud, proposal, values, cells, tracks = fixture()
        cloud["maps"][1] = [20, 20, 20]
        proposal["source_patch_indices"] = [np.array([0, 1, 2])]
        values[0] = -4.
        result = full_region_supervision(cloud, proposal, values, cells, tracks)
        self.assertFalse(result["mask"][0])
        self.assertEqual(result["audit"]["rows"][0]["source_anchor_patch"], 0)

    def test_source_fallback_cannot_discard_a_disagreeing_supported_view(self):
        cloud = dict(maps=np.ones((3, 1, 2, 3))*.1, stems=["first", "second", "third"])
        cloud["maps"][1, 0, 1] = 0.
        cloud["maps"][2, 0, 1] = 20.
        proposal = dict(centers=np.zeros((1, 3)), cell_size=1., source_patch_indices=[np.array([2, 3, 4])])
        values = np.array([[[-1., -1.]], [[-2., -2.]], [[-2., -2.]]])
        cells = np.zeros((3, 1, 2, 2), int)
        tracks = {0: {1, 2, 3}, 2: {1, 2, 3}, 3: {4, 5, 6}, 4: {4, 5, 6}}
        result = full_region_supervision(cloud, proposal, values, cells, tracks)
        self.assertFalse(result["mask"][0])
        self.assertEqual(result["audit"]["rows"][0]["reason"], "teacher_views_disagree_or_unavailable")
        self.assertNotIn("source_anchor_patch", result["audit"]["rows"][0])

    def test_matching_predictions_without_independent_tracks_stay_unknown(self):
        cloud, proposal, values, cells, tracks = fixture()
        for evidence in ({}, {index: {100, 101} for index in tracks}):
            result = full_region_supervision(cloud, proposal, values, cells, evidence)
            self.assertFalse(result["mask"][0])

    def test_single_view_does_not_recover_under_original_policy(self):
        cloud, proposal, values, cells, tracks = fixture()
        values[1] = np.nan
        result = full_region_supervision(cloud, proposal, values, cells, tracks)
        self.assertFalse(result["mask"][0])
        self.assertEqual(result["audit"]["rows"][0]["reason"], "fewer_than_two_labelled_views")

    def test_disagreement_stays_unknown(self):
        cloud, proposal, values, cells, tracks = fixture()
        values[1] = -1.
        result = full_region_supervision(cloud, proposal, values, cells, tracks)
        self.assertFalse(result["mask"][0])

    def test_duplicate_patches_do_not_multiply_independent_tracks(self):
        cloud, proposal, values, cells, tracks = fixture()
        tracks = {index: {100} for index in tracks}
        self.assertFalse(full_region_supervision(cloud, proposal, values, cells, tracks)["mask"][0])

    def test_empty_labels_are_finite_and_unknown(self):
        cloud, proposal, values, cells, tracks = fixture()
        values[:] = np.nan
        result = full_region_supervision(cloud, proposal, values, cells, tracks)
        self.assertFalse(result["mask"].any())
        self.assertTrue(np.isfinite(result["target"]).all())

    def test_thresholds_cannot_silently_relax_reliability(self):
        for change in (dict(recovered_weight=1.), dict(minimum_shared_tracks=1),
                       dict(maximum_view_log_mse_range=float("nan"))):
            with self.assertRaises(ValueError):
                FullLabelPolicy(**change)

    def test_source_patch_provenance_traces_selected_centers(self):
        random = np.random.default_rng(42)
        maps = random.normal(size=(2, 4, 6, 3))
        result = propose_current_regions(maps, np.ones(maps.shape[:-1]), max_regions=2, confidence_quantile=0.)
        for center, indices in zip(result["centers"], result["source_patch_indices"]):
            np.testing.assert_array_equal(center, np.median(maps.reshape(-1, 3)[indices], axis=0))
        self.assertEqual(tuple(result["source_patch_shape"]), maps.shape[:3])

    def test_half_reliability_really_halves_gradient(self):
        prediction = torch.tensor([.2, .3], requires_grad=True)
        target = torch.tensor([.8, .9])
        mask, groups = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long)
        original = grouped_regression(prediction, target, mask, groups)
        first = torch.autograd.grad(original, prediction)[0]
        for weight in (.5, .25):
            weighted = grouped_regression(prediction, target, mask, groups, torch.full((2,), weight))
            self.assertAlmostEqual(float(weighted), weight*float(original))
            second = torch.autograd.grad(weighted, prediction)[0]
            torch.testing.assert_close(second, first*weight)

    def test_recovered_targets_do_not_enter_any_full_ranking(self):
        predictions = dict(total=torch.tensor([.1, .2, .3], requires_grad=True), structure=torch.zeros(3))
        labels = dict(total=torch.tensor([.1, .9, .8]), structure=torch.zeros(3), total_mask=torch.ones(3, dtype=torch.bool),
                      structure_mask=torch.zeros(3, dtype=torch.bool), state=torch.zeros(3, dtype=torch.long),
                      groups=torch.tensor([0, 0, 1]), total_rank_mask=torch.tensor([True, False, False]),
                      total_weight=torch.tensor([1., .5, .25]))
        _, pieces, statistics = training_loss(predictions, labels, TrainingConfigV2())
        self.assertEqual(statistics["full_local_ranking"]["pairs"], 0)
        self.assertEqual(statistics["full_cross_ranking"]["pairs"], 0)
        self.assertGreater(float(pieces["full_regression"]), 0)

    def test_source_pair_wires_recovery_without_changing_missing(self):
        self.check_source_pair(fixture(), FullLabelPolicy(), .5, "recovered")

    def test_source_pair_wires_single_view_without_changing_missing(self):
        self.check_source_pair(single_view_fixture(), FullLabelPolicy(single_view_supervision=True), .25, "single_view")

    def check_source_pair(self, sample, policy, expected_weight, expected_status):
        full, proposal, values, cells, tracks = sample
        full.update(full_teacher_patches=(values, cells, tracks), geometry_evidence_audit=[])
        missing = dict(full)
        errors = np.full_like(values, .01)
        inputs = LocalQueryInputs(**{field.name: torch.zeros((2, 2), dtype=torch.bool if field.name == "view_mask" else torch.float32)
                                     for field in dataclasses.fields(LocalQueryInputs)})
        labels = dict(total=np.zeros(2, np.float32), total_mask=np.array([False, False]),
                      structure=np.array([0, .2], np.float32), structure_mask=np.array([False, False]), groups=np.array([0, 1]))
        source = SourceV2.__new__(SourceV2)
        source.records = {"scene": {"teacher_mapping": "rectified_resize"}}
        source.quality_limits = np.array([-3., -1.])
        source.full_label_policy = policy
        with mock.patch.object(Source, "pair", return_value=(inputs, labels, {}, (full, missing, proposal, proposal, errors))), \
                mock.patch("openflyscan.quality_predictor.paired_observation_source.self_projection_audit", return_value={}):
            _, result, summary, _ = source.pair("scene", np.random.default_rng(0))
        self.assertEqual(result["total_mask"].tolist(), [True, False])
        self.assertEqual(result["total_rank_mask"].tolist(), [False, False])
        self.assertEqual(result["total_weight"].tolist(), [expected_weight, 0])
        np.testing.assert_array_equal(result["structure"], [0, np.float32(.2)])
        self.assertEqual(summary["full_supervision"][expected_status], 1)


if __name__ == "__main__":
    unittest.main()
