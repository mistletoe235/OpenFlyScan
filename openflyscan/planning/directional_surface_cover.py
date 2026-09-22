"""Whole-row cover of frozen source-conditioned evidence, without GS/scene rules.

One height per task. Preserve directional evidence relative to the attainable
candidate pool, and expose weak/unreachable evidence rather than discarding it.
"""
import copy
import numpy as np


def whole_rows(actions):
    result=[];seen=set()
    for action in actions:
        for ri,row in enumerate(action['rows']):
            w=row['waypoints']
            # Some historical grids contain two coincident rows. Do not pay twice.
            key=(action['area_id'],tuple(sorted(tuple(np.round(x['position_enu_m']+[x['aircraft_yaw_deg']%360,x['gimbal_pitch_deg']],6)) for x in w)))
            if key in seen:continue
            seen.add(key)
            r=copy.deepcopy(action);r.update(id=action['id']+f'_whole_row{ri}',source_action=action['id'],source_row=ri,
                waypoints=copy.deepcopy(w),rows=[copy.deepcopy(row)],photo_count=len(w),row_count=1,
                length_m=float(np.linalg.norm(np.diff([x['position_enu_m'] for x in w],axis=0),axis=1).sum()),
                coverage_scope='one complete original row; not cropped at arbitrary photo budget')
            for k in ['coverage','old_view_link','reliable_context_link','same_height_old_overlap_preference']:
                r.pop(k,None)
            result.append(r)
    return result


def evidence_sectors(directions):
    d=np.asarray(directions)
    return np.where(d[:,2]>=np.sin(np.radians(67.5)),0,
        1+np.floor(((np.degrees(np.arctan2(d[:,0],d[:,1]))+45)%360)/90).astype(int))


def select_rows(rows,utility,tokens,quality_ratio=.8,required_fraction=.95,coverage_floor=1e-4,condition_on_source=True):
    if not 0<quality_ratio<=1 or not 0<required_fraction<=1:raise ValueError('Invalid coverage policy')
    if not np.isfinite(utility).all() or (utility<0).any():raise ValueError('Invalid utility')
    selected=[];reports=[];ranks=tokens['rank'];weights=tokens['risk_weight']
    sector=evidence_sectors(tokens['source_direction']) if condition_on_source else np.full(len(ranks),-1)
    # Equal control of each source-direction/rank bucket prevents easy regions
    # from compensating for entirely uncovered surfaces elsewhere.
    for area in sorted({r['area_id'] for r in rows}):
        choices=[i for i,r in enumerate(rows) if r['area_id']==area];members=rows[choices[0]]['ranks']
        base_ti=np.flatnonzero(np.isin(ranks,members));base_u=utility[np.ix_(choices,base_ti)]
        dense=rows[choices[0]].get('task_type')=='area_survey'
        # Dense tasks retain the user's oblique-survey semantics. A nadir row
        # cannot stand in for all facade directions just because its scalar
        # utility is high. Each of the five original sweep directions gets its
        # own attainable-evidence coverage constraint, not a whole rectangle.
        directions=sorted({rows[i]['survey_direction'] for i in choices}) if dense else [None]
        ti=np.tile(base_ti,len(directions));u=np.concatenate([
            base_u*np.array([d is None or rows[i]['survey_direction']==d for i in choices])[:,None]
            for d in directions],axis=1)
        capture_sector=np.repeat(np.array([-1 if d is None else d for d in directions]),len(base_ti))
        best=u.max(0)
        reachable=best>coverage_floor;needed=quality_ratio*best
        cover=(u>=needed[None,:])&reachable[None,:]
        buckets=[];bucket_names=[]
        for rank in members:
            for sec in sorted(set(sector[ti][ranks[ti]==rank])):
                for capture_sec in sorted(set(capture_sector)):
                    mask=(ranks[ti]==rank)&(sector[ti]==sec)&(capture_sector==capture_sec)&reachable
                    if mask.any():buckets.append(mask);bucket_names.append((int(rank),int(sec),int(capture_sec)))
        ww=weights[ti]
        def fractions(state):
            return np.array([np.sum(ww[m]*state[m])/np.sum(ww[m]) for m in buckets])
        heights=sorted({round(rows[i]['height_above_target_m'],3) for i in choices});solutions=[]
        for height in heights:
            pool=[j for j,i in enumerate(choices) if round(rows[i]['height_above_target_m'],3)==height]
            state=np.zeros(len(ti),bool);chosen=[];trace=[]
            while pool and len(buckets) and np.any(fractions(state)<required_fraction-1e-9):
                active=fractions(state)<required_fraction-1e-9;balance=np.zeros(len(ti))
                for is_active,m in zip(active,buckets):
                    if is_active:balance[m]+=ww[m]/ww[m].sum()
                options=[]
                for j in pool:
                    r=rows[choices[j]];gain=float(balance@(cover[j]&~state))
                    if gain<=0:continue
                    # Prefer joining neighboring rows and old-image context;
                    # these never count as reconstruction-quality observations.
                    neighbors=[rows[choices[k]] for k in chosen if rows[choices[k]]['source_action']==r['source_action']]
                    gap=min((abs(r['source_row']-x['source_row']) for x in neighbors),default=1)
                    direction_switch=bool(chosen and not neighbors)
                    cost=r['photo_count']+2*direction_switch+.5*max(0,gap-1)
                    ctx=float(r.get('same_height_old_overlap_preference',0))
                    options.append((gain*(1+.1*ctx)/cost,-r['photo_count'],-j,j,gain))
                if not options:break
                *_,j,gain=max(options);chosen.append(j);pool.remove(j);state|=cover[j]
                trace.append(dict(row=rows[choices[j]]['id'],new_balanced_evidence=gain))
            # Remove rows made redundant by later rows; preserve full row geometry.
            for j in list(reversed(chosen)):
                remaining=[k for k in chosen if k!=j]
                state2=np.any(cover[remaining],axis=0) if remaining else np.zeros(len(ti),bool)
                if len(buckets) and np.all(fractions(state2)>=required_fraction-1e-9):chosen=remaining
            # Dense sweep rows stay contiguous between selected extremes; do
            # not skip a lateral strip and then claim preserved side overlap.
            if dense:
                groups={rows[choices[j]]['source_action'] for j in chosen}
                for group in groups:
                    own=[rows[choices[j]]['source_row'] for j in chosen if rows[choices[j]]['source_action']==group]
                    for j in range(len(choices)):
                        r=rows[choices[j]]
                        if r['source_action']==group and min(own)<=r['source_row']<=max(own) and j not in chosen:chosen.append(j)
            state=np.any(cover[chosen],axis=0) if chosen else np.zeros(len(ti),bool)
            frac=fractions(state);count=sum(rows[choices[k]]['photo_count'] for k in chosen)
            feasible=bool(len(buckets) and np.all(frac>=required_fraction-1e-9))
            solutions.append(dict(height=height,chosen=chosen,state=state,fractions=frac,photos=count,complete=feasible,trace=trace))
        # Coverage first, then compactness. No lowering thresholds to get fewer photos.
        sol=max(solutions,key=lambda s:(s['complete'],0. if s['complete'] else float(np.minimum(s['fractions'],required_fraction).sum()),-s['photos']))
        selected.extend(choices[j] for j in sol['chosen'])
        reports.append(dict(area_id=area,ranks=members,height_above_target_m=sol['height'],photos=sol['photos'],
            row_count=len(sol['chosen']),status='directional_evidence_proxy_complete' if sol['complete'] else 'UNRESOLVED_directional_evidence',
            selected_rows=[rows[choices[j]]['id'] for j in sol['chosen']],
            reachable_weight_fraction=float(ww[reachable].sum()/max(ww.sum(),1e-12)),
            weak_best_quality_weight_fraction=float(ww[best<.05].sum()/max(ww.sum(),1e-12)),
            minimum_best_quality=float(best.min()) if len(best) else None,
            dense_five_direction_constraints=dense,
            buckets=[dict(rank=r,source_sector=s,capture_sector=c,covered_fraction=float(f)) for (r,s,c),f in zip(bucket_names,sol['fractions'])],
            unreachable_witness_indices=np.unique(ti[~reachable]).tolist(),trace=sol['trace'],
            scope='Coverage of retained model evidence, not whole-building or GS-error coverage'))
    return selected,reports
