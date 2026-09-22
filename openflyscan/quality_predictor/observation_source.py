"""Current-native paired samples with distinct, honest supervision channels.
Full: measured exposure-corrected GS quality. Missing: predicted geometry-change
proxy relative to Full, NOT missing GS PSNR or true geometry error.
"""
import json,hashlib,dataclasses
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from scipy.spatial import cKDTree
from .footprint_window_sampler import sample_footprint_window
from .native_pi3x_geometry import native_cloud
from .pi3x_features import FrozenPi3XQueryFeatureBuilder
from .regions import propose_current_regions
from .region_features import pack_current_region_inputs
from .model import LocalQueryInputs
from .training_performance import FeaturePrefetcher, PerformanceConfig, SceneTensorCache, StageProfiler, TensorLRU, stage, synchronize_model, timed

def sample_window(record,rng,weak=False,*,anchor_indices=None):
    z=record['footprint_data'];centers=z['centers'];lo=z['bbox_mins'];hi=z['bbox_maxs']
    pool=record['weak_indices'] if weak and len(record['weak_indices']) else np.arange(len(centers))
    if anchor_indices is not None:
        pool=np.intersect1d(pool,anchor_indices) if weak else np.asarray(anchor_indices,dtype=int)
        if not len(pool):raise ValueError('no eligible spatial window anchors')
    base=centers[int(rng.choice(pool))];pre=sample_footprint_window(base,centers,lo,hi)
    extent=np.ptp(centers[list(pre.core_indices)],axis=0);anchor=base+rng.uniform(-.18,.18,2)*extent
    window=sample_footprint_window(anchor,centers,lo,hi);core=np.array(window.core_indices);ids=list(window.stems_indices)
    # New interior location for every shifted window; no old fixed target or ID.
    low,high=np.quantile(centers[core],[.1,.9],axis=0);relative=rng.uniform(.25,.75,2);target=low+relative*(high-low)
    count=int(rng.integers(10,13));order=core[np.argsort(np.linalg.norm(centers[core]-target,axis=1),kind='stable')];deleted=order[:count].tolist()
    stems=list(map(str,z['stems']));full=[stems[i] for i in ids];removed=[stems[i] for i in deleted];missing=[s for s in full if s not in removed]
    assert len(full)==30 and len(missing)==30-count and not set(removed)&set(missing)
    return dict(full=full,missing=missing,deleted=removed,anchor=anchor.tolist(),relative_location=relative.tolist(),weak_anchor=weak)

class Source:
    def __init__(self,plan_path,model,max_regions=64,performance=None):
        self.plan=json.loads(Path(plan_path).read_text());self.root=Path(self.plan['root']);self.model=model;self.max_regions=max_regions;self.records={};self.builders=OrderedDict()
        configured = self.plan.get('performance', {}) if performance is None else performance
        self.performance = configured if isinstance(configured, PerformanceConfig) else PerformanceConfig(**configured)
        self.photo_cache = TensorLRU(self.performance.gpu_feature_cache_mib*1024**2,
                                    self.performance.gpu_feature_cache_entries)
        budget = self.performance.cpu_feature_cache_mib*1024**2
        self.cpu_photo_cache = (SceneTensorCache(budget, [record['scene'] for record in self.plan['records']])
                                if self.performance.cpu_feature_cache_policy == 'scene' else TensorLRU(budget))
        self.prefetcher = FeaturePrefetcher()
        self.cache_counts = dict(builder_hits=0, builder_misses=0, builder_evictions=0,
                                 teacher_hits=0, teacher_misses=0, teacher_evictions=0)
        self.profiler = StageProfiler(self.performance.profile_stages, lambda: synchronize_model(self.model))
        self.synchronous_feature_reads = 0
        entries=json.loads(Path(self.plan['dino_manifest']).read_text())['entries'];self.dino={}
        for e in entries:self.dino.setdefault(e['scene'],{})[e['stem']]=e
        for r in self.plan['records']:
            from .teacher_grid import mapping_models, validate_exposure
            if 'teacher_mapping_contract' in r:mapping_models(r)
            z=dict(np.load(r['footprints']));meta=json.loads((Path(r['evaluation'])/'metrics.json').read_text());validate_exposure(r,meta)
            teacher=dict(np.load(Path(r['evaluation'])/'grid_errors.npz'))
            if meta['grid']!=16:
                teacher['mse']=teacher['mse_grid16'];teacher['valid_pixel_weight']=teacher['valid_pixel_weight_grid16']
            by={Path(str(s)).stem:i for i,s in enumerate(teacher['names'])}
            # Image-level quality only drives offline training mixture, never inputs.
            mse=teacher['mse'];weight=teacher['valid_pixel_weight'];means=(mse*weight).sum((1,2))/weight.sum((1,2));worst=set(np.argsort(means)[-int(np.ceil(.2*len(means))):]);weak=[i for i,s in enumerate(z['stems']) if by.get(str(s),-1) in worst]
            self.records[r['scene']]={**r,'footprint_data':z,'teacher_data':teacher,'teacher_index':by,'weak_indices':weak,'teacher_grid':16}
        self.train=[s for s,r in self.records.items() if r['split']=='train'];self.val=[s for s,r in self.records.items() if r['split']=='validation']
        assert len(self.train)>=2 and set(self.train).isdisjoint(self.val)
        # Freeze global normalization using TRAIN scenes only; balanced scene contribution.
        values=[]
        for s in self.train:
            d=self.records[s]['teacher_data'];v=np.log10(np.maximum(d['mse'][d['valid_pixel_weight']>0],1e-10));values.extend(v[np.linspace(0,len(v)-1,min(len(v),4000)).astype(int)].tolist())
        self.quality_limits=np.quantile(values,[.1,.9]);assert np.diff(self.quality_limits)[0]>.05
    def performance_snapshot(self):
        return dict(counters=dict(self.cache_counts), builders=len(self.builders),
                    cpu_features=self.cpu_photo_cache.snapshot(), gpu_features=self.photo_cache.snapshot(),
                    stages=self.profiler.snapshot(), prefetch=self.prefetcher.snapshot(),
                    synchronous_feature_reads=self.synchronous_feature_reads)

    def iter_prepared_pairs(self, sequence):
        iterator = iter(sequence)

        def prepare(item):
            scene, sample_seed = item
            rng = np.random.default_rng(sample_seed)
            choice = sample_window(self.records[scene], rng, weak=bool(rng.random() < .5))
            token = object()
            if self.performance.prefetch_features:
                builder = self.builder(scene)
                keys = [(scene, stem) for stem in choice['full']]
                keys = [key for key in keys if key not in self.photo_cache and key not in self.cpu_photo_cache]
                self.prefetcher.submit(token, keys, builder._cached_cpu)
            return scene, sample_seed, choice, token

        current = None
        upcoming = None
        try:
            first = next(iterator, None)
            current = prepare(first) if first is not None else None
            while current is not None:
                item = next(iterator, None)
                upcoming = prepare(item) if item is not None else None
                yield current[:3]
                self.prefetcher.retire(current[3])
                current, upcoming = upcoming, None
        finally:
            for prepared in (current, upcoming):
                if prepared is not None:
                    self.prefetcher.retire(prepared[3])

    def close(self):
        self.prefetcher.close()

    @timed('builder')
    def builder(self,s):
        if s not in self.builders:
            self.cache_counts['builder_misses'] += 1
            builder=FrozenPi3XQueryFeatureBuilder(self.model,Path(self.records[s]['source']),self.dino[s],np.zeros(3))
            def cached(stem):
                key=(s,stem)
                tensors = self.photo_cache.get(key)
                if tensors is not None:
                    return tensors
                host = self.cpu_photo_cache.get(key)
                if host is None:
                    with stage('feature_prefetch_wait'):
                        host = self.prefetcher.take(key)
                    if host is None:
                        with stage('feature_disk_read'):
                            host = builder._cached_cpu(stem)
                        self.synchronous_feature_reads += 1
                    self.cpu_photo_cache.put(key, host)
                with stage('feature_to_device'):
                    tensors = tuple(tensor.to(builder.device) for tensor in host)
                self.photo_cache.put(key, tensors)
                return tensors
            builder._cached=cached;self.builders[s]=builder
            limit = self.performance.builder_cache_scenes or len(self.records)
            while len(self.builders)>limit:
                self.builders.popitem(last=False)
                self.cache_counts['builder_evictions'] += 1
        else:
            self.cache_counts['builder_hits'] += 1
        self.builders.move_to_end(s);return self.builders[s]
    @timed('extract')
    def extract(self,s,stems,seed):
        with self.cpu_photo_cache.batch([(s, stem) for stem in stems]):
            return self._extract(s,stems,seed)

    def _extract(self,s,stems,seed):
        builder=self.builder(s);builder.recenter=np.mean([builder._camera(t)[0][:3,3] for t in stems],axis=0);captures={}
        handles=[self.model.model.point_decoder.register_forward_hook(lambda _m,_i,out:captures.__setitem__('point',out)),self.model.model.conf_decoder.register_forward_hook(lambda _m,_i,out:captures.__setitem__('conf',out))]
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):cloud=native_cloud(self.model,builder,stems,seed)
        finally:
            for h in handles:h.remove()
        with stage('decoder_feature_prepare'):
            V,H,W=cloud['maps'].shape[:3];start=int(self.model.model.patch_start_idx)
            point=captures['point'][:,start:].reshape(V,H,W,1024).float().clone().detach();confidence=captures['conf'][:,start:].reshape(V,H,W,1024).float().clone().detach();cached=[builder._cached(t) for t in stems]
            dino=torch.stack([x[0] for x in cached]).float();rgb=torch.stack([x[1] for x in cached]).float()
        with stage('region_proposal'):
            proposal=propose_current_regions(cloud['maps'],cloud['conf'],self.max_regions,confidence_quantile=0.)
        with stage('region_input_pack'):
            if getattr(getattr(self, 'training_config', None), 'source_patch_recovery', False):
                from .view_supervision import pack_source_supported_inputs
                inputs=pack_source_supported_inputs(cloud,proposal,point,confidence,dino,rgb)
            else:
                inputs=pack_current_region_inputs(cloud,proposal,point,confidence,dino,rgb)
        return inputs,cloud,proposal
    @timed('full_pixel_labels')
    def full_pixel_labels(self,s,cloud):
        import sys
        sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'scripts'),str(self.root/'open-lixel-h3dgs-color11-20260808/preprocess')]
        from map_rgb_to_rectified_uv import load_pair,map_uv
        import read_write_model as rw
        record=self.records[s]
        if 'calibration' not in record:
            full=Path(record['full_root']);record['calibration']=load_pair(full/'rgb_sfm',full/'published_scene_source/sparse/0',rw)
        rc,tc,ri,ti=record['calibration'];names={Path(n).stem:n for n in ri};V,H,W=cloud['maps'].shape[:3];yy,xx=np.indices((H,W));pixel=np.stack([xx*14+7,yy*14+7],-1);result=np.full((V,H,W),np.nan)
        g=record['teacher_grid'];d=record['teacher_data']
        for i,stem in enumerate(cloud['stems']):
            if stem not in names or stem not in record['teacher_index']:continue
            name=names[stem];sc=rc[ri[name].camera_id];dc=tc[ti[name].camera_id];raw=(pixel+.5)*[sc.width/(W*14),sc.height/(H*14)]-.5;uv,valid=map_uv(raw,sc,dc)
            xy=np.floor((uv+.5)*[g/dc.width,g/dc.height]).astype(int);valid&=(xy>=0).all(-1)&(xy<g).all(-1)
            x=np.clip(xy[...,0],0,g-1);y=np.clip(xy[...,1],0,g-1);j=record['teacher_index'][stem];valid&=d['valid_pixel_weight'][j,y,x]>0
            values=np.log10(np.maximum(d['mse'][j,y,x],1e-10));result[i][valid]=values[valid]
        return result
    @staticmethod
    @timed('region_labels')
    def region_labels(cloud,proposal,values):
        xyz=cloud['maps'].reshape(-1,3);value=values.reshape(-1);valid=np.isfinite(xyz).all(1)&np.isfinite(value);ids=np.flatnonzero(valid);tree=cKDTree(xyz[valid]);out=np.zeros(len(proposal['centers']),np.float32);mask=np.zeros(len(out),bool);perview=np.prod(cloud['maps'].shape[1:3])
        for i,q in enumerate(proposal['centers']):
            found=ids[tree.query_ball_point(q,proposal['cell_size'])];views=found//perview
            if len(found)<6 or len(np.unique(views))<2:continue
            # Equal vote per observed image, not density-weighted duplicate points.
            out[i]=np.median([np.median(value[found[views==v]]) for v in np.unique(views)]);mask[i]=True
        return out,mask
    @timed('pair_base', source=True)
    def pair(self,s,rng,seed=20260907,choice=None):
        from geoff3d.slrf.geometry_align import estimate_similarity_umeyama
        with stage('sample_window'):
            choice=choice or sample_window(self.records[s],rng,weak=bool(rng.random()<.5))
        with stage('full_extract'):
            fi,f,fp=self.extract(s,choice['full'],seed)
        with stage('missing_extract'):
            mi,m,mp=self.extract(s,choice['missing'],seed)
        with stage('pair_geometry_alignment'):
            index=[f['stems'].index(t) for t in m['stems']];source=m['maps'].reshape(-1,3);target=f['maps'][index].reshape(-1,3);valid=np.isfinite(source).all(1)&np.isfinite(target).all(1);ss=source[valid][::3];tt=target[valid][::3];keep=np.ones(len(ss),bool)
            for _ in range(4):
                scale,R,t,ok,note=estimate_similarity_umeyama(ss[keep],tt[keep]);assert ok,note;e=np.linalg.norm(scale*ss@R.T+t-tt,axis=1);keep=e<=np.quantile(e,.8)
            depth=np.stack([((f['maps'][i]-f['camera_poses'][i,:3,3])@f['camera_poses'][i,:3,:3])[...,2] for i in index]);error=np.linalg.norm((scale*source@R.T+t-target).reshape(m['maps'].shape),axis=-1)/np.maximum(depth,1e-5);error[depth<=0]=np.nan
        raw_full,fullmask=self.region_labels(f,fp,self.full_pixel_labels(s,f));raw_missing,missingmask=self.region_labels(m,mp,error)
        fulltarget=np.clip((raw_full-self.quality_limits[0])/np.diff(self.quality_limits)[0],0,1);missingtarget=np.clip(raw_missing/.05,0,1)
        packed={}
        for field in dataclasses.fields(LocalQueryInputs):
            left,right=getattr(fi,field.name),getattr(mi,field.name)
            if field.name!='query_features':
                pad=30-right.shape[1]
                right=torch.cat([right,torch.zeros((len(right),pad,*right.shape[2:]),device=right.device,dtype=right.dtype)],1)
            packed[field.name]=torch.cat([left,right],0)
        inputs=LocalQueryInputs(**packed);n=len(fulltarget);k=len(missingtarget)
        labels=dict(total=np.r_[fulltarget,np.zeros(k)].astype('float32'),total_mask=np.r_[fullmask,np.zeros(k,bool)],structure=np.r_[np.zeros(n),missingtarget].astype('float32'),structure_mask=np.r_[np.zeros(n,bool),missingmask],groups=np.r_[np.zeros(n,int),np.ones(k,int)])
        summary=dict(scene=s,choice=choice,full_regions=n,missing_regions=k,full_valid=int(fullmask.sum()),missing_valid=int(missingmask.sum()),full_target_range=[float(fulltarget[fullmask].min()),float(fulltarget[fullmask].max())] if fullmask.any() else [],missing_target_range=[float(missingtarget[missingmask].min()),float(missingtarget[missingmask].max())] if missingmask.any() else [],geometry_median_pct=float(np.nanmedian(error)*100),geometry_p90_pct=float(np.nanquantile(error,.9)*100),geometry_over2pct=float(np.nanmean(error>.02)),pair_hash=hashlib.sha256(json.dumps(choice,sort_keys=True).encode()).hexdigest())
        return inputs,labels,summary,(f,m,fp,mp,error)
