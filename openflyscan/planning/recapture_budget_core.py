"""Budgeted selection primitives. No scene paths, GS, semantic truth, or rank rules."""
from dataclasses import dataclass,fields
import math
import numpy as np


@dataclass(frozen=True)
class BudgetPolicy:
    budgets: tuple = (60,100,160)
    max_strip_photos: int = 9
    strip_shared_photos: int = 2
    source_direction_bandwidth_deg: float = 35.
    desired_linear_resolution_gain: float = 1.5
    content_floor: float = .25
    exploration_fraction: float = .2
    exploration_support_threshold: float = .2
    same_height_context_bonus: float = .1
    minimum_marginal_gain: float = 1e-6
    transfer_weight: float = .2

    @classmethod
    def from_dict(cls,value):
        unknown=set(value)-{f.name for f in fields(cls)}
        if unknown:raise ValueError(f'Unknown policy keys: {sorted(unknown)}')
        obj=cls(**value)
        if not obj.budgets or any(type(b)!=int or b<=0 for b in obj.budgets):raise ValueError('Budget must be positive photo counts')
        if type(obj.max_strip_photos)!=int or type(obj.strip_shared_photos)!=int:raise ValueError('Strip sizes must be integer photo counts')
        numeric=[getattr(obj,f.name) for f in fields(obj) if f.name not in ['budgets','max_strip_photos','strip_shared_photos']]
        if not all(np.isfinite(v) for v in numeric):raise ValueError('Policy numbers must be finite')
        if obj.max_strip_photos<3 or not 0<=obj.strip_shared_photos<obj.max_strip_photos-1:raise ValueError('Invalid strip size')
        if not 0<=obj.exploration_fraction<=1 or not 0<obj.content_floor<=1:raise ValueError('Invalid probability-like setting')
        if obj.source_direction_bandwidth_deg<=0 or obj.desired_linear_resolution_gain<1:raise ValueError('Invalid view quality settings')
        if not 0<=obj.exploration_support_threshold<=1 or min(obj.transfer_weight,obj.same_height_context_bonus,obj.minimum_marginal_gain)<0:raise ValueError('Invalid gain/cost parameter')
        return obj


def split_capture_indices(count,maximum,shared):
    if count<3:return []
    result=[];start=0
    while start<count:
        end=min(count,start+maximum)
        if count-end in (1,2):end=count
        if end-start>maximum:
            end=start+maximum
        if end-start<3:
            start=max(0,count-3);end=count
        result.append(list(range(start,end)))
        if end==count:break
        start=end-shared
    return result


def capped_marginal(utility,current,weights):
    return np.maximum(utility-current,0)@weights


def greedy_budget(candidates,utility,weights,budget,policy,start_position,speed,interval,
                  token_ranks,exploration=None,mode='repair',uncertain_tokens=None):
    """Shared candidate pool, integer photo cap and exact marginal saturation.

    Self objective is a selection diagnostic, NOT a reconstruction metric.
    No arbitrary fill-to-budget; duplicates cannot create objective gain.
    """
    utility=np.asarray(utility);weights=np.asarray(weights);token_ranks=np.asarray(token_ranks)
    if utility.shape!=(len(candidates),len(weights)) or token_ranks.shape!=weights.shape:raise ValueError('Utility/weight shape mismatch')
    if not np.isfinite(utility).all() or not np.isfinite(weights).all() or (utility<0).any() or (weights<0).any():raise ValueError('Invalid utility/weight')
    if speed<=0 or interval<=0:raise ValueError('Positive speed and interval required')
    uncertain_tokens=None if uncertain_tokens is None else np.asarray(uncertain_tokens,bool)
    if uncertain_tokens is not None and uncertain_tokens.shape!=weights.shape:raise ValueError('Uncertainty shape mismatch')
    state=np.zeros(utility.shape[1]);pending=set(range(len(candidates)))
    chosen=[];trace=[];spent=0;explore_spent=0;uncertain_spent=0.;current=np.array(start_position,float)
    exploration=np.zeros(len(candidates),bool) if exploration is None else np.asarray(exploration,bool)
    while pending:
        options=[]
        for i in sorted(pending):
            c=candidates[i];count=c['photo_count']
            if spent+count>budget:continue
            is_explore=bool(exploration[i]) and mode=='repair'
            delta=np.maximum(utility[i]-state,0);gain=float(delta@weights)
            if gain<=policy.minimum_marginal_gain:continue
            # Mixed actions cannot hide weak evidence behind their mean support.
            # Charge a fractional PHOTO-EQUIVALENT, not a measured photo count.
            fraction=float((delta*weights)[uncertain_tokens].sum()/gain) if uncertain_tokens is not None else float(is_explore)
            charge=count*fraction if mode=='repair' else 0.
            if mode=='repair' and uncertain_spent+charge>math.floor(budget*policy.exploration_fraction)+1e-9:continue
            transfer=float(np.linalg.norm(np.array(c['waypoints'][0]['position_enu_m'])-current)/speed)
            cost=count*interval+c['length_m']/speed+policy.transfer_weight*transfer
            context=(1+policy.same_height_context_bonus*c.get('same_height_old_overlap_preference',0)) if mode=='repair' else 1.
            options.append((gain*context/max(cost,1.),-count,c['id'],i,gain,delta,cost,is_explore,fraction,charge))
        if not options:break
        _,_,_,i,gain,delta,cost,is_explore,fraction,charge=max(options,key=lambda r:r[:3]);c=candidates[i]
        spent+=c['photo_count'];explore_spent+=c['photo_count']*is_explore;chosen.append(i);pending.remove(i)
        uncertain_spent+=charge
        gains={int(r):float((delta*weights)[token_ranks==r].sum()) for r in np.unique(token_ranks)}
        trace.append(dict(action=c['id'],photos=c['photo_count'],cumulative_photos=spent,marginal_objective=gain,
            contribution_by_target={str(k):v for k,v in gains.items() if v>1e-8},
            budget_role=('exploration' if fraction>=1-1e-9 else 'mixed' if fraction>1e-9 else 'repair') if mode=='repair' else 'baseline',
            uncertain_gain_fraction=fraction,uncertain_photo_equivalent=charge,cost_proxy_seconds=cost))
        state=np.maximum(state,utility[i]);current=np.array(c['waypoints'][-1]['position_enu_m'])
    return chosen,dict(photo_budget=budget,used_photos=spent,unspent_photos=budget-spent,
        exploration_photos=int(explore_spent),objective=float(state@weights),trace=trace,
        uncertain_photo_equivalent=float(uncertain_spent),
        uncertainty_accounting='Marginal utility share times action photos; not a count of actually uncertain photographs',
        marginal_gain_sum=float(sum(t['marginal_objective'] for t in trace)),
        stop='no positive feasible marginal action within budget/exploration cap',
        objective_is_not_quality_metric=True,per_token_coverage=state.tolist())
