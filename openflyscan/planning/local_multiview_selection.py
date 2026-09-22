"""Add local directions by marginal joint pair support, not rendered GS error.

Freeze first strip. No .25 baseline bonus for extra strips: zero new support
must give zero gain. Same capture altitude; no climbing photos. All accepted
and rejected candidate metrics persist, including saturated no-add outcomes.
"""
import numpy as np
from openflyscan.planning.plan_frozen_region_scans import camera_basis
from openflyscan.planning.recapture_overlap_connections import motion_seconds


def combined_support(base,extra,cfg,pair_utility):
    qs=[base[0]['_old_q']]+[c['_new_q'] for c in base]
    ds=[base[0]['_old_d']]+[c['_new_d'] for c in base]
    if extra is not None:qs.append(extra['_new_q']);ds.append(extra['_new_d'])
    return pair_utility(np.concatenate(qs),np.concatenate(ds),cfg)


def direction_gap(a,b):
    da=camera_basis(a['yaw_deg'],a['pitch_deg'])[:,2]
    db=camera_basis(b['yaw_deg'],b['pitch_deg'])[:,2]
    return float(np.degrees(np.arccos(np.clip(da@db,-1,1))))


def evaluate_extra(base,candidate,cfg,pair_utility,before=None):
    height=base[0]['waypoints'][0]['position_enu_m'][2]
    if abs(candidate['waypoints'][0]['position_enu_m'][2]-height)>.01:return dict(reason='different_capture_altitude')
    gap=min(direction_gap(candidate,b) for b in base)
    if gap<cfg.local_min_direction_separation_deg:return dict(reason='redundant_view_direction',direction_gap_deg=gap)
    if candidate['survey_direction'] in {c['survey_direction'] for c in base}:
        return dict(reason='same_direction_bucket')
    if before is None:before=combined_support(base,None,cfg,pair_utility)
    after=combined_support(base,candidate,cfg,pair_utility)
    delta=np.maximum(after-before,0);gain=float(delta.mean())
    fraction=float(np.mean(delta>=cfg.local_point_gain_threshold))
    # Incremental cost includes the shortest end-to-end connecting move, not a
    # new sequence of photographs. Transit remains only a safety proxy.
    oldends=[c['waypoints'][i]['position_enu_m'] for c in base for i in [0,-1]]
    newends=[candidate['waypoints'][i]['position_enu_m'] for i in [0,-1]]
    transfer=min(motion_seconds(p,q,cfg) for p in oldends for q in newends)
    cost=candidate['photo_count']*cfg.interval_s+candidate['length_m']/cfg.speed_mps+transfer
    group_report=[];selection_gain=gain
    reason='eligible' if gain>=cfg.local_min_mean_gain and fraction>=cfg.local_min_improved_fraction else 'insufficient_new_geometry_support'
    if '_surface_groups' in candidate:
        for group in candidate['_surface_groups']:
            ix=group['indices'];g=float(delta[ix].mean());f=float(np.mean(delta[ix]>=cfg.local_point_gain_threshold))
            group_report.append(dict(kind=group['kind'],count=len(ix),mean_gain=g,improved_fraction=f,
                eligible=g>=cfg.local_min_mean_gain and f>=cfg.local_min_improved_fraction))
        # Equal weight across qualified modes prevents the roof point majority
        # suppressing a reliable wall. Tiny/unreliable modes never participate.
        selection_gain=float(np.mean([g['mean_gain'] for g in group_report])) if group_report else 0.
        reason='eligible' if any(g['eligible'] for g in group_report) else 'no_reliable_surface_gain'
    return dict(reason=reason,mean_gain=gain,improved_fraction=fraction,
        before_mean=float(before.mean()),after_mean=float(after.mean()),direction_gap_deg=gap,
        surface_groups=group_report,selection_gain=selection_gain,
        added_photos=candidate['photo_count'],incremental_cost_proxy_s=cost,gain_per_cost=selection_gain/max(cost,1.),
        _after=after,_delta=delta)


def augment_local_directions(areas,candidates,chosen,cfg,pair_utility):
    selected=[dict(c) for c in chosen];initial=[c['id'] for c in chosen];basephotos=sum(c['photo_count'] for c in selected)
    records=[]
    for area in sorted(areas,key=lambda a:-a['risk']):
        if area['task_type']!='local_patch':continue
        base=[c for c in selected if c['area_id']==area['id']]
        if not base:
            records.append(dict(area_id=area['id'],ranks=area['ranks'],status='no_initial_strip'));continue
        base[0]['local_selection_role']='initial_strip';steps=[]
        original=combined_support(base,None,cfg,pair_utility)
        while len(base)<cfg.local_max_directions:
            before=combined_support(base,None,cfg,pair_utility);evaluated=[];options=[]
            for c in candidates:
                if c['area_id']!=area['id'] or c['id'] in {x['id'] for x in base}:continue
                info=evaluate_extra(base,c,cfg,pair_utility,before)
                if info['reason']=='eligible':
                    if cfg.photo_budget and sum(x['photo_count'] for x in selected)+c['photo_count']>cfg.photo_budget:
                        info['reason']='photo_budget'
                    else:options.append((info['gain_per_cost'],-c['photo_count'],c['id'],c,info))
                evaluated.append(dict(candidate=c['id'],pitch=c['pitch_deg'],yaw=c['yaw_deg'],
                    **{k:v for k,v in info.items() if not k.startswith('_')}))
            if not options:
                steps.append(dict(selected=None,evaluated=evaluated,stop='no_eligible_new_direction'));break
            _,_,_,candidate,info=max(options,key=lambda x:x[:3]);best=dict(candidate);best['local_selection_role']='added_direction'
            best['marginal_selection']={k:v for k,v in info.items() if not k.startswith('_')}
            base.append(best);selected.append(best)
            steps.append(dict(selected=best['id'],evaluated=evaluated,
                mean_gain=info['mean_gain'],improved_fraction=info['improved_fraction'],photos=best['photo_count']))
        for c in base:c['survey_id']=base[0]['survey_id'] # Keep a local multiview task contiguous during ordering.
        final=combined_support(base,None,cfg,pair_utility)
        record=dict(area_id=area['id'],ranks=area['ranks'],initial=base[0]['id'],selected=[c['id'] for c in base],
            original_support_mean=float(original.mean()),final_support_mean=float(final.mean()),
            original_per_point_support=original.tolist(),final_per_point_support=final.tolist(),
            additional_photos=sum(c['photo_count'] for c in base[1:]),steps=steps,
            status='added_directions' if len(base)>1 else 'kept_single_strip')
        if '_surface_groups' in base[0] and not base[0]['_surface_groups']:
            record['status']='insufficient_surface_evidence_initial_strip_retained'
        records.append(record)
        print('Local multiview',area['ranks'],'directions',len(base),'extra',record['additional_photos'],
              'pair support',round(original.mean(),3),'->',round(final.mean(),3),flush=True)
    return selected,dict(protocol=('Reliable native surface modes: groupwise marginal best-pair support; unknowns NOT good'
        if cfg.surface_multiview else 'Exact old+selected+candidate joint best-pair support')+'; no GS input, no direction-specific error claim',
        base_action_ids=initial,base_photos=basephotos,total_photos=sum(c['photo_count'] for c in selected),
        all_initial_actions_preserved=set(initial).issubset({c['id'] for c in selected}),
        capture_altitude_policy='extras at same height as initial local strip',regions=records,
        limitations=['Best-pair support may saturate after nadir even if side surfaces remain bad.',
                    'No predicted new image quality, feature matching, or exact visibility.',
                    'At most 3 local directions is a provisional preview cap, not learned optimum.'])
