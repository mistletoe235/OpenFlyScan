"""Build query-conditioned Head inputs from cached DINO and online frozen Pi3X."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from openflyscan.quality_predictor.model import LocalQueryInputs
from openflyscan.quality_predictor.pi3x_cached_dino import CachedDinoEncoder


class FrozenPi3XQueryFeatureBuilder:
    def __init__(self, model, scene_source: Path, dino_by_stem: dict, recenter: np.ndarray):
        from geoff3d.slrf.scene_io import build_views_from_scene

        self.model = model.eval().requires_grad_(False)
        self.device = next(model.parameters()).device
        self.dtype = model.dtype
        self.source = Path(scene_source)
        self.dino_by_stem = dino_by_stem
        self.recenter = np.asarray(recenter, np.float64).reshape(3)
        self.lightweight, self.meta = build_views_from_scene(
            self.source, max_image_size=518, patch_size=14, device=self.device, show_progress=False,
        )
        self.by_stem = {str(row["stem"]): i for i, row in enumerate(self.lightweight)}
        self.target_h, self.target_w = int(self.meta["target_h"]), int(self.meta["target_w"])
        self.patch_h, self.patch_w = self.target_h // 14, self.target_w // 14
        self.placeholder = torch.zeros((1, 3, self.target_h, self.target_w), device=self.device)

    def _camera(self, stem: str) -> tuple[np.ndarray, np.ndarray]:
        from geoff3d.slrf.scene_io import scale_K_to_target

        cam = self.meta["cams"].get(stem)
        if cam is None:
            raise RuntimeError(f"camera prior missing for {stem}")
        K = scale_K_to_target(
            K=np.asarray(cam["K"], np.float64), cam_width=cam.get("width"), cam_height=cam.get("height"),
            source_h=int(cam["height"]), source_w=int(cam["width"]),
            target_h=self.target_h, target_w=self.target_w,
        ).astype(np.float32)
        return np.asarray(cam["T_c2w"], np.float32), K

    def _cached(self, stem: str) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(tensor.to(self.device) for tensor in self._cached_cpu(stem))

    def _cached_cpu(self, stem: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(self.dino_by_stem[stem]["cache"])
        with np.load(path, allow_pickle=False) as data:
            feature = torch.from_numpy(np.array(data["feature"], copy=True))
            rgb = torch.from_numpy(np.array(data["rgb_stats"], copy=True)).float()
        if feature.shape != (self.patch_h, self.patch_w, 1024) or rgb.shape != (self.patch_h, self.patch_w, 6):
            raise RuntimeError(f"bad DINO/RGB cache shape for {stem}: {feature.shape}/{rgb.shape}")
        return feature, rgb

    def _views(self, stems: list[str]) -> tuple[list[dict], list[np.ndarray], list[np.ndarray], torch.Tensor, torch.Tensor]:
        views, global_poses, intrinsics, dino, rgb = [], [], [], [], []
        for stem in stems:
            if stem not in self.by_stem or stem not in self.dino_by_stem:
                raise RuntimeError(f"uncached or absent stem: {stem}")
            global_pose, K = self._camera(stem)
            pose = global_pose.copy()
            pose[:3, 3] -= self.recenter.astype(np.float32)
            feature, rgb_stats = self._cached(stem)
            views.append({
                "img": self.placeholder,
                "is_metric_scale": torch.ones((1,), dtype=torch.bool, device=self.device),
                "is_synthetic": torch.zeros((1,), dtype=torch.bool, device=self.device),
                "true_shape": torch.tensor([[self.target_h, self.target_w]], dtype=torch.int64, device=self.device),
                "data_norm_type": ["identity"], "label": [self.source.name], "instance": [stem],
                "idx": [f"{self.source.name}/{stem}"],
                "camera_intrinsics": torch.from_numpy(K).unsqueeze(0).to(self.device),
                "camera_pose": torch.from_numpy(pose).unsqueeze(0).to(self.device),
            })
            global_poses.append(global_pose)
            intrinsics.append(K)
            dino.append(feature)
            rgb.append(rgb_stats)
        return views, global_poses, intrinsics, torch.stack(dino), torch.stack(rgb)

    def build(
        self, stems: list[str], target_seed_xyz: np.ndarray, target_center: np.ndarray,
        query_camera_T_c2w: np.ndarray, *, seed: int = 20260906,
    ) -> LocalQueryInputs:
        views, global_poses, intrinsics, dino, rgb = self._views(stems)
        captures = {}
        handles = [
            self.model.model.point_decoder.register_forward_hook(
                lambda _m, _i, output: captures.__setitem__("point", output)
            ),
            self.model.model.conf_decoder.register_forward_hook(
                lambda _m, _i, output: captures.__setitem__("confidence", output)
            ),
        ]
        self.model.model.encoder = CachedDinoEncoder(
            dino.reshape(len(stems), self.patch_h * self.patch_w, 1024)
        ).to(self.device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            with torch.inference_mode():
                predictions = self.model(views)
            torch.cuda.synchronize(self.device)
        finally:
            for handle in handles:
                handle.remove()
        patch_start = int(self.model.model.patch_start_idx)
        point = captures["point"][:, patch_start:].reshape(len(stems), self.patch_h, self.patch_w, 1024)
        confidence = captures["confidence"][:, patch_start:].reshape(len(stems), self.patch_h, self.patch_w, 1024)

        target_seed_xyz = np.asarray(target_seed_xyz, np.float32)
        target_center = np.asarray(target_center, np.float32)
        point_out = torch.zeros((len(stems), 1024), device=self.device)
        confidence_out = torch.zeros_like(point_out)
        dino_out = torch.zeros_like(point_out)
        rgb_out = torch.zeros((len(stems), 6), device=self.device)
        pose_out = torch.zeros((len(stems), 10), device=self.device)
        quality_out = torch.zeros((len(stems), 4), device=self.device)
        view_mask = torch.zeros(len(stems), dtype=torch.bool, device=self.device)
        for view_index, (pose, K, pred) in enumerate(zip(global_poses, intrinsics, predictions)):
            world_to_camera = np.linalg.inv(pose)
            points_camera = (world_to_camera[:3, :3] @ target_seed_xyz.T + world_to_camera[:3, 3:4]).T
            in_front = points_camera[:, 2] > 1e-5
            projected = (K @ points_camera[in_front].T).T
            uv = projected[:, :2] / projected[:, 2:3]
            px = np.floor(uv[:, 0] / 14).astype(np.int64)
            py = np.floor(uv[:, 1] / 14).astype(np.int64)
            inside = (px >= 0) & (px < self.patch_w) & (py >= 0) & (py < self.patch_h)
            cells = np.unique(np.stack([py[inside], px[inside]], axis=1), axis=0)
            if not len(cells):
                continue
            yy = torch.from_numpy(cells[:, 0]).long().to(self.device)
            xx = torch.from_numpy(cells[:, 1]).long().to(self.device)
            point_out[view_index] = point[view_index, yy, xx].float().median(dim=0).values
            confidence_out[view_index] = confidence[view_index, yy, xx].float().median(dim=0).values
            dino_out[view_index] = dino[view_index, yy, xx].float().median(dim=0).values
            rgb_out[view_index] = rgb[view_index, yy, xx].median(dim=0).values
            camera = pose[:3, 3]
            relative = camera - target_center
            distance = max(float(np.linalg.norm(relative)), 1e-8)
            camera_to_target = -relative / distance
            optical = pose[:3, 2] / max(float(np.linalg.norm(pose[:3, 2])), 1e-8)
            pose_out[view_index] = torch.tensor([
                *relative, *camera_to_target, distance, float(np.dot(optical, camera_to_target)),
                float(np.median((cells[:, 1] + .5) / self.patch_w)),
                float(np.median((cells[:, 0] + .5) / self.patch_h)),
            ], device=self.device)
            sy = torch.clamp(yy * 14 + 7, max=self.target_h - 1)
            sx = torch.clamp(xx * 14 + 7, max=self.target_w - 1)
            current_xyz = pred["pts3d_cam"][0, sy, sx].float()
            spread = torch.linalg.norm(current_xyz - current_xyz.median(dim=0).values, dim=1)
            conf = pred["conf"][0, sy, sx, 0].float()
            quality_out[view_index] = torch.stack([
                torch.tensor(len(cells) / float(self.patch_h * self.patch_w), device=self.device),
                spread.median(), conf.median(), conf.std() if len(conf) > 1 else conf.new_zeros(()),
            ])
            view_mask[view_index] = True
        query_camera_T_c2w = np.asarray(query_camera_T_c2w, np.float32)
        query_relative = query_camera_T_c2w[:3, 3] - target_center
        query_features = np.concatenate([
            query_relative, query_camera_T_c2w[:3, :3].reshape(-1),
            np.asarray([np.linalg.norm(query_relative)], np.float32),
        ])
        result = LocalQueryInputs(
            point_features=point_out[None], confidence_features=confidence_out[None],
            dino_features=dino_out[None], pose_geometry_features=pose_out[None],
            quality_geometry_features=quality_out[None], rgb_stats=rgb_out[None],
            query_features=torch.from_numpy(query_features).to(self.device)[None], view_mask=view_mask[None],
        )
        return result
