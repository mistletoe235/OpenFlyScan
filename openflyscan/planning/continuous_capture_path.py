"""C2 stop/turn/restart preview trajectory; never certify unknown-space safety.

Each capture row remains a straight fixed-attitude pass. Quintic time scaling
stops smoothly at row endpoints, not at every photograph. Connectors carry no
photos. Camera yaw is unwrapped on the shortest arc.
"""
import copy
import math
import numpy as np


def smoothstep(s):
    return s*s*s*(10+s*(-15+6*s))


def inverse_smoothstep(value):
    lo,hi=0.,1.
    for _ in range(45):
        mid=(lo+hi)/2
        if smoothstep(mid)<value:lo=mid
        else:hi=mid
    return (lo+hi)/2


def angle_delta(start,end):
    return (end-start+180)%360-180


def duration(p0,p1,y0,y1,g0,g1,limits):
    delta=np.asarray(p1)-p0
    return max(.2,1.875*np.linalg.norm(delta)/limits['speed_mps'],
        math.sqrt(5.774*np.linalg.norm(delta)/limits['acceleration_mps2']),
        1.875*max(delta[2],0)/limits['climb_mps'],
        1.875*max(-delta[2],0)/limits['descent_mps'],
        1.875*abs(angle_delta(y0,y1))/limits['yaw_rate_dps'],
        1.875*abs(g1-g0)/limits['pitch_rate_dps'])


def order_rows(rows,start,start_yaw=0.,start_pitch=-45.,limits=None):
    """Finish one area and one direction at a time; traverse rows in snake order."""
    pending={}
    for r in rows:pending.setdefault(r['area_id'],[]).append(r)
    current=np.asarray(start,float);ordered=[];yaw=float(start_yaw);pitch=float(start_pitch)
    def entry_cost(r,j):
        link=r.get('endpoint_context_links',{}).get(str(j))
        # Soft preference only: never add a detour/bridge photograph to force a link.
        w=r['waypoints'][j]
        if limits is not None:
            base=duration(current,w['position_enu_m'],yaw,w['aircraft_yaw_deg'],pitch,w['gimbal_pitch_deg'],limits)
            return base-1.25*(link['score'] if link else 0.)
        return np.linalg.norm(current-np.array(w['position_enu_m']))-5.*(link['score'] if link else 0.)
    while pending:
        area=min(pending,key=lambda a:min(entry_cost(r,j)
                 for r in pending[a] for j in [0,-1]))
        groups={}
        for r in pending.pop(area):groups.setdefault(r['source_action'],[]).append(r)
        while groups:
            key=min(groups,key=lambda a:min(entry_cost(r,j)
                    for r in groups[a] for j in [0,-1]))
            group=sorted(groups.pop(key),key=lambda r:r['source_row'])
            if min(entry_cost(group[-1],j) for j in [0,-1]) < min(entry_cost(group[0],j) for j in [0,-1]):group.reverse()
            for raw in group:
                r=copy.deepcopy(raw)
                reverse=entry_cost(r,-1)<entry_cost(r,0)
                r['entry_context_link']=r.get('endpoint_context_links',{}).get('-1' if reverse else '0')
                if reverse:
                    r['waypoints'].reverse();r['reversed_for_continuity']=True
                ordered.append(r);end=r['waypoints'][-1];current=np.array(end['position_enu_m'])
                yaw=end['aircraft_yaw_deg'];pitch=end['gimbal_pitch_deg']
    return ordered


def build_trajectory(rows,start,start_yaw,start_pitch,limits,tree=None,clearance=12.):
    if not all(np.isfinite(v) and v>0 for v in limits.values()):raise ValueError('Positive finite motion limits required')
    records=[];events=[];segments=[];clock=0.;current=np.array(start,float);yaw=float(start_yaw);pitch=float(start_pitch)
    def segment(end,y1,g1,kind,action,shots=None):
        nonlocal clock,current,yaw,pitch
        p0=current.copy();end=np.array(end,float);dy=angle_delta(yaw,y1)
        T=duration(p0,end,yaw,y1,pitch,g1,limits)
        times=[]
        if shots:
            total=np.linalg.norm(end-p0)
            if total<1e-6:raise ValueError('Degenerate capture row')
            row_speed=min(limits['speed_mps'],min(w.get('speed_mps',limits['speed_mps']) for w in shots))
            if row_speed<=0:raise ValueError('Positive row speed cap required')
            T=max(T,1.875*total/row_speed)
            for w in shots:
                p=np.asarray(w['position_enu_m'])
                if np.linalg.norm(np.cross(p-p0,end-p0))/total>1e-4:raise ValueError('Capture row must be straight')
                if abs(angle_delta(yaw,w['aircraft_yaw_deg']))>1e-5 or abs(pitch-w['gimbal_pitch_deg'])>1e-5:raise ValueError('Capture row must have constant camera attitude')
            for w in shots:
                fraction=np.linalg.norm(np.array(w['position_enu_m'])-p0)/max(total,1e-9)
                times.append(0. if fraction<1e-8 else 1. if fraction>1-1e-8 else inverse_smoothstep(fraction))
            if len(times)>1 and min(np.diff(times))<=0:raise ValueError('Capture points must progress along row')
            if len(times)>1:T=max(T,limits['interval_s']/max(min(np.diff(times)),1e-8))
        samples=np.linspace(0,1,max(2,int(math.ceil(T/.5))+1))
        all_times=sorted(set(samples.tolist()+times));min_clearance=float('inf')
        for s in all_times:
            h=smoothstep(s);point=p0+h*(end-p0)
            records.append(dict(t_s=clock+s*T,position_enu_m=point.tolist(),yaw_unwrapped_deg=yaw+h*dy,
                gimbal_pitch_deg=pitch+h*(g1-pitch),capture=False,segment_type=kind,action=action))
        if tree is not None:
            probes=p0+np.linspace(0,1,max(2,int(np.linalg.norm(end-p0)/1.)+1))[:,None]*(end-p0)
            min_clearance=float(tree.query(probes)[0].min())
        for s,w in zip(times,shots or []):events.append(dict(w,t_s=clock+s*T,action=action,capture=True,
            motion_profile='timed_quintic_row; speed_mps is a limit, not constant execution speed'))
        segments.append(dict(type=kind,action=action,start_s=clock,end_s=clock+T,duration_s=T,
            start=p0.tolist(),end=end.tolist(),yaw_change_deg=dy,pitch_change_deg=g1-pitch,
            theoretical_peak_speed_mps=1.875*np.linalg.norm(end-p0)/T,
            theoretical_peak_acceleration_mps2=5.774*np.linalg.norm(end-p0)/T**2,
            theoretical_peak_yaw_rate_dps=1.875*abs(dy)/T,
            theoretical_peak_pitch_rate_dps=1.875*abs(g1-pitch)/T,
            observed_point_clearance_m=min_clearance if np.isfinite(min_clearance) else None,
            clearance_warning=bool(min_clearance<clearance),velocity_zero_at_endpoints=True))
        clock+=T;current=end;yaw+=dy;pitch=g1
    for r in rows:
        shots=r['waypoints'];first=shots[0];last=shots[-1]
        segment(first['position_enu_m'],first['aircraft_yaw_deg'],first['gimbal_pitch_deg'],'noncapture_connector',r['id'])
        segment(last['position_enu_m'],last['aircraft_yaw_deg'],last['gimbal_pitch_deg'],'capture_row',r['id'],shots)
    return dict(samples=records,capture_events=events,segments=segments,duration_s=clock,
        interpolation='quintic minimum-jerk time scaling; stop-turn-restart at row boundaries; continuous within rows',
        flight_authorized=False,safety='Observed-cloud check only; no unknown obstacle clearance or flight dynamics certification')
