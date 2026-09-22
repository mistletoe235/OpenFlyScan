"""Convert frozen ENU recapture positions to Android V5 schema13 PREVIEW.

Altitude is takeoff-relative. Historical DJI altitude is not verified orthometric
or ellipsoidal survey height. Exact inverse of the source's WGS84 ECEF/ENU
calculation avoids mixing the prior approximate spherical coordinate converter.
No capture point is moved, clamped, or replanned. Standard library only.
"""
import argparse,json,math,time,uuid
from pathlib import Path

A=6378137.;E2=6.6943799901413165e-3
def ecef(lat,lon,h):
    lat,lon=map(math.radians,(lat,lon));s=math.sin(lat);c=math.cos(lat);n=A/math.sqrt(1-E2*s*s)
    return ((n+h)*c*math.cos(lon),(n+h)*c*math.sin(lon),(n*(1-E2)+h)*s)

def enu_to_geo(p,ref):
    lat,lon=map(math.radians,ref[:2]);e,n,u=p;x0,y0,z0=ecef(*ref)
    x=x0-math.sin(lon)*e-math.sin(lat)*math.cos(lon)*n+math.cos(lat)*math.cos(lon)*u
    y=y0+math.cos(lon)*e-math.sin(lat)*math.sin(lon)*n+math.cos(lat)*math.sin(lon)*u
    z=z0+math.cos(lat)*n+math.sin(lat)*u
    q=math.hypot(x,y);ll=math.atan2(z,q*(1-E2))
    for _ in range(12):
        radius=A/math.sqrt(1-E2*math.sin(ll)**2);ll=math.atan2(z+E2*radius*math.sin(ll),q)
    h=q/math.cos(ll)-A/math.sqrt(1-E2*math.sin(ll)**2)
    return [math.degrees(ll),math.degrees(math.atan2(y,x)),h]

def geo_to_enu(geo,ref):
    d=[v-w for v,w in zip(ecef(*geo),ecef(*ref))];lat,lon=map(math.radians,ref[:2]);x,y,z=d
    return [-math.sin(lon)*x+math.cos(lon)*y,-math.sin(lat)*math.cos(lon)*x-math.sin(lat)*math.sin(lon)*y+math.cos(lat)*z,
        math.cos(lat)*math.cos(lon)*x+math.cos(lat)*math.sin(lon)*y+math.sin(lat)*z]

def delta(a,b):return (b-a+180)%360-180
def dist(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def hull(points):
    points=sorted(set(tuple(p[:2]) for p in points))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def half(seq):
        out=[]
        for p in seq:
            while len(out)>=2 and cross(out[-2],out[-1],p)<=0:out.pop()
            out.append(p)
        return out
    return half(points)[:-1]+half(points[::-1])[:-1]

def convert(plan,manifest,camera,takeoff):
    ref=manifest['reference_wgs84'];wps=[];posemap=[];passes=[];passindex=0
    def point(p):
        lat,lon,h=enu_to_geo(p,ref)
        return dict(latitude=lat,longitude=lon,altitude_m=h-takeoff)
    def add(p,yaw,pitch,capture,row,role):
        nonlocal passindex
        kind='CAPTURE_POINT' if capture else 'TRANSIT'
        wps.append(dict(point=point(p),heading_deg=yaw%360,gimbal_pitch_deg=pitch,kind=kind,
            capture_action='CAPTURE_ON_REACH' if capture else 'NONE',capture_interval_m=None,
            pass_index=passindex,capture_view='LOCAL_OBLIQUE'))
        posemap.append(dict(waypoint_index=len(wps)-1,pass_index=passindex,source_row=row['id'],area_id=row['area_id'],
            ranks=row['ranks'],position_enu_m=list(p),sensor_datum_absolute_altitude_m=enu_to_geo(p,ref)[2],capture=capture))
        passes.append(dict(pass_index=passindex,region_id=row['area_id'],role=role,capture_role='SURVEY' if capture else 'NONE',
            source='FROZEN_HEAD_PI3X_CONTINUOUS_COVER',required_for_reconstruction_bridge=False))
        passindex+=1
    current=plan['start']['position_enu_m'];yaw=plan['start']['yaw'];pitch=plan['start']['pitch']
    add(current,yaw,pitch,False,plan['actions'][0],'RECORDED_START_PREVIEW_NOT_TAKEOFF')
    for row in plan['actions']:
        first=row['waypoints'][0];end=first['position_enu_m'];dy=delta(yaw,first['aircraft_yaw_deg']);dp=first['gimbal_pitch_deg']-pitch
        # Interior controls preserve the connector line and shortest angular arc.
        # No 0.5s time samples: schema13 has no timestamps or angular rate fields.
        count=max(1,math.ceil(dist(current,end)/20),math.ceil(abs(dy)/30),math.ceil(abs(dp)/10),math.ceil(abs(end[2]-current[2])/5))
        for i in range(1,count):
            f=i/count;p=[x+f*(y-x) for x,y in zip(current,end)]
            add(p,yaw+f*dy,pitch+f*dp,False,row,'NONCAPTURE_TRANSIT')
        for w in row['waypoints']:add(w['position_enu_m'],w['aircraft_yaw_deg'],w['gimbal_pitch_deg'],True,row,'EXACT_CAPTURE_POINT')
        last=row['waypoints'][-1];current=last['position_enu_m'];yaw=last['aircraft_yaw_deg'];pitch=last['gimbal_pitch_deg']
    captures=[x for x in wps if x['capture_action']=='CAPTURE_ON_REACH'];heights=[x['point']['altitude_m'] for x in wps]
    if any(not 5<=h<=120 for h in heights):raise ValueError('Preview reference yields altitude outside Android 5-120m validator; do not clamp')
    # Conservative common speed: schema13 cannot encode row-specific speed.
    common_speed=min(4.,min(w.get('speed_mps',4.) for r in plan['actions'] for w in r['waypoints']))
    path=sum(dist(a['position_enu_m'],b['position_enu_m']) for a,b in zip(posemap,posemap[1:]))
    # Explicit point-stop surrogate, not the prior continuous-row estimate.
    duration=0.
    for i,(a,b) in enumerate(zip(posemap,posemap[1:])):
        d=dist(a['position_enu_m'],b['position_enu_m']);v=common_speed;acc=plan['limits']['acceleration_mps2']
        translation=2*math.sqrt(d/acc) if d<v*v/acc else d/v+v/acc
        dz=b['position_enu_m'][2]-a['position_enu_m'][2]
        duration+=max(translation,max(dz,0)/2,max(-dz,0)/1.5,abs(delta(wps[i]['heading_deg'],wps[i+1]['heading_deg']))/15,
            abs(wps[i+1]['gimbal_pitch_deg']-wps[i]['gimbal_pitch_deg'])/10)
        if b['capture']:duration+=1. # Current WPMZ converter's hover action.
    regions=[]
    for report in plan['area_reports']:
        own=[r for r in plan['actions'] if r['area_id']==report['area_id']]
        xyz=own[0]['center'];lat,lon,h=enu_to_geo(xyz,ref)
        regions.append(dict(region_id=report['area_id'],priority=min(report['ranks']),kind='FROZEN_RISK_SCAN_TASK',
            risk_score=0.,reasons=['Frozen target ranks '+str(report['ranks']),report['status'],
            'Numeric Head score not carried by this route export; risk_score=0 is an unused display placeholder'],
            target_wgs84=dict(latitude=lat,longitude=lon,absolute_altitude_m=h),
            pass_indices=[p['pass_index'] for p in posemap if p['area_id']==report['area_id']],suggested_survey_photos=report['photos']))
    # Exact numeric risks are filled from the frozen selection by the CLI below.
    hmax=max(heights);constraints=dict(altitude_agl_m=hmax,forward_overlap=.8,side_overlap=.7,speed_mps=common_speed,
        oblique_speed_mps=common_speed,gimbal_pitch_deg=-90.,route_heading_deg=plan['actions'][0]['yaw_deg'],crosshatch=False,
        collection_mode='OBLIQUE_FIVE_DIRECTION',oblique_gimbal_pitch_deg=-45.,boundary_margin_m=0.,altitude_mode='RELATIVE_TO_TAKEOFF',
        target_surface_to_takeoff_m=0.,safe_takeoff_altitude_m=math.ceil(hmax/5)*5,takeoff_speed_mps=2.,descent_speed_mps=1.5,
        takeoff_mode='MANUAL',start_point_mode='FIRST_ROUTE_START',completion_action='HOVER',capture_trigger_mode='DISTANCE',
        timed_capture_interval_s=2.,oblique_forward_overlap=.8,oblique_side_overlap=.7,
        oblique_heading_mode='FIXED_CAPTURE_DIRECTION',enabled_capture_views=['LOCAL_OBLIQUE'])
    mission=dict(schema_version=13,id=str(uuid.uuid4()),name='OpenFlyScan reacquisition preview',created_at_epoch_ms=int(time.time()*1000),
        coordinate_frame='WGS84',camera_profile=dict(id='source-mini2-exif-4000x2250-unverified-live-camera',image_width_px=4000,image_height_px=2250,
            horizontal_fov_deg=camera['horizontal_fov_deg'],vertical_fov_deg=camera['vertical_fov_deg'],minimum_capture_interval_s=2.),
        constraints=constraints,roi=[point([*xy,plan['start']['position_enu_m'][2]]) for xy in hull([x['position_enu_m'] for x in posemap])],
        waypoints=wps,estimated_path_m=path,estimated_photo_count=len(captures),estimated_flight_s=duration,terrain_plan=None,
        active_mapping=dict(schema_version=1,selection_method='frozen_head_pi3x_directional_whole_row_cover',ground_truth_used=False,
            gs_used_for_selection=False,ordinary_gps_used=True,source_capture_count=len(captures),survey_capture_count=len(captures),
            bridge_capture_count=0,source_estimated_route_distance_m=path,regions=regions,passes=passes))
    # Extension ignored by Android decode; NOT an execution interlock.
    mission['export_review']=dict(preview_only=True,safe_to_execute=False,takeoff_absolute_altitude_m=takeoff,
        reference_status='historical_unconfirmed',safety_flag_enforced_by_android=False)
    return mission,posemap

