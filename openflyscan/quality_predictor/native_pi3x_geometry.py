"""Pi3X native points/cameras placed by one whole-chunk Sim(3)."""
import numpy as np
import torch
from PIL import Image


from .training_performance import stage, timed


@timed('native_cloud')
def native_cloud(model,builder,stems,seed,*,sample_stride=14):
    from .pi3x_cached_dino import CachedDinoEncoder
    from geoff3d.slrf.geometry_align import estimate_similarity_umeyama
    from geoff3d.models.external.vggt.utils.rotation import quat_to_mat
    with stage('view_prepare'):
        views,poses,intrinsics,dino,rgb=builder._views(stems)
    model.model.encoder=CachedDinoEncoder(dino.reshape(len(stems),builder.patch_h*builder.patch_w,1024)).to(builder.device)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    with stage('pi3x_forward'), torch.inference_mode():preds=model(views)
    centers=np.stack([p['cam_trans'][0].float().cpu().numpy() for p in preds])
    target=np.stack([p[:3,3] for p in poses]);s,R,t,ok,note=estimate_similarity_umeyama(centers,target)
    if not ok:raise RuntimeError(note)
    if sample_stride<1:raise ValueError('sample_stride must be positive')
    yy,xx=np.meshgrid(np.arange(sample_stride//2,builder.target_h,sample_stride),np.arange(sample_stride//2,builder.target_w,sample_stride),indexing='ij')
    maps=np.stack([p['pts3d'][0,yy,xx].float().cpu().numpy() for p in preds]);shape=maps.shape
    maps=(s*maps.reshape(-1,3)@R.T+t).reshape(shape)
    native_poses=np.tile(np.eye(4),(len(stems),1,1))
    for i,p in enumerate(preds):
        native_poses[i,:3,:3]=R@quat_to_mat(p['cam_quats'])[0].float().cpu().numpy()
        native_poses[i,:3,3]=s*centers[i]@R.T+t
    if sample_stride==14:colors=rgb[...,:3].float().cpu().numpy()
    else:
        colors=np.stack([np.asarray(Image.open(builder.meta['image_paths'][stem]).convert('RGB').resize((builder.target_w,builder.target_h),Image.Resampling.BILINEAR))[yy,xx]/255. for stem in stems])
    return dict(stems=stems,maps=maps,conf=np.stack([p['conf'][0,yy,xx,0].float().cpu().numpy() for p in preds]),rgb=colors,camera_poses=native_poses,intrinsics=np.asarray(intrinsics),sample_stride=sample_stride)
