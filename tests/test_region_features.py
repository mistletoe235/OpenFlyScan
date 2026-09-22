import dataclasses
import inspect
import unittest
import numpy as np
import torch
from openflyscan.quality_predictor.region_features import pack_current_region_inputs
from openflyscan.quality_predictor.model import LocalQueryQualityConfig,LocalQueryQualityHead,LocalQueryInputs


class CurrentInputTests(unittest.TestCase):
    def make_sample(self):
        y,x=np.indices((4,6));K=np.array([[100.,0,42],[0,100.,28],[0,0,1.]])
        rays=np.stack(((x+.5)*14,(y+.5)*14,np.ones_like(x)),-1)@np.linalg.inv(K).T
        poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[-1,0,1]
        maps=np.stack([rays*10+pose[:3,3] for pose in poses])
        cloud=dict(maps=maps,camera_poses=poses,intrinsics=np.tile(K,(3,1,1)),conf=np.ones((3,4,6)))
        proposal=dict(centers=np.array([[0.,0,10],[1000.,0,10]]),origin=np.array([0.,0,10]),cell_size=2.,valid_points=maps.reshape(-1,3))
        torch.manual_seed(7)
        features=[torch.randn(3,4,6,1024) for _ in range(3)];rgb=torch.rand(3,4,6,6)
        return cloud,proposal,features,rgb

    def make_inputs(self):
        cloud,proposal,features,rgb=self.make_sample()
        return pack_current_region_inputs(cloud,proposal,*features,rgb)

    def test_batched_interpolation_matches_individual_pooling(self):
        cloud, proposal, features, rgb = self.make_sample()
        features[0].requires_grad_(True)
        inputs = pack_current_region_inputs(cloud, proposal, *features, rgb)
        center = proposal['centers'][0]
        for view, pose in enumerate(cloud['camera_poses']):
            projected = cloud['intrinsics'][view] @ (pose[:3, :3].T @ (center-pose[:3, 3]))
            pixel = projected[:2]/projected[2]/14-.5
            column, row = np.floor(pixel).astype(int)
            horizontal, vertical = pixel-[column, row]
            weights = torch.tensor([(1-horizontal)*(1-vertical), horizontal*(1-vertical),
                                    (1-horizontal)*vertical, horizontal*vertical], dtype=torch.float32)
            for source, output in zip(features+[rgb], [inputs.point_features, inputs.confidence_features,
                                                       inputs.dino_features, inputs.rgb_stats]):
                expected = (source[view, row:row+2, column:column+2].reshape(4, -1)*weights[:, None]).sum(0)
                torch.testing.assert_close(output[0, view], expected, rtol=1e-6, atol=1e-6)
                self.assertEqual(int(torch.count_nonzero(output[1, view])), 0)
        inputs.point_features.sum().backward()
        self.assertTrue(torch.isfinite(features[0].grad).all())

    def test_empty_queries_and_all_unobservable_regions(self):
        cloud, proposal, features, rgb = self.make_sample()
        for centers in (proposal['centers'][1:], np.empty((0, 3))):
            sample = {**proposal, 'centers': centers}
            inputs = pack_current_region_inputs(cloud, sample, *features, rgb)
            self.assertEqual(inputs.point_features.shape, (len(centers), 3, 1024))
            self.assertFalse(inputs.view_mask.any())

    def test_no_reference_query_camera_argument(self):
        names=set(inspect.signature(pack_current_region_inputs).parameters)
        self.assertEqual(names,{'cloud','proposal','point_features','confidence_features','dino_features','rgb_stats'})

    def test_empty_region_and_view_permutation(self):
        inputs=self.make_inputs()
        self.assertEqual(inputs.view_mask.sum(1).tolist(),[3,0])
        head=LocalQueryQualityHead(LocalQueryQualityConfig()).eval()
        result=head(inputs)
        self.assertTrue(torch.isfinite(result.total_risk).all())
        shuffled=LocalQueryInputs(**{f.name:(getattr(inputs,f.name)[:,[2,0,1]] if f.name!='query_features' else inputs.query_features)
                                    for f in dataclasses.fields(inputs)})
        torch.testing.assert_close(head(shuffled).total_risk,result.total_risk,atol=1e-6,rtol=1e-5)
        result.total_risk.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
