"""Explicit schema negotiation and unchanged-pose mobile export."""

import math

from .geometry_export import convert


def mission_mode(payload):
    schemas = payload.get('supported_mission_schemas', [13])
    if not isinstance(schemas, list) or any(type(version) is not int or version not in (13, 14) for version in schemas):
        raise ValueError('supported_mission_schemas must contain 13 and/or 14')
    mode = payload.get('recapture_flight_mode', 'STOP_AND_CAPTURE')
    if mode not in ('STOP_AND_CAPTURE', 'CONTINUOUS_EXPERIMENTAL'):
        raise ValueError('unsupported recapture_flight_mode')
    version = 14 if mode == 'CONTINUOUS_EXPERIMENTAL' else 13
    if version not in schemas:
        raise ValueError(f'{mode} requires mission schema {version}')
    return dict(supported_mission_schemas=schemas, recapture_flight_mode=mode, mission_schema_version=version)


def export_mission(plan, manifest, camera, selection, config):
    mode = mission_mode(config)
    altitude = config.get('takeoff_absolute_altitude_m')
    if isinstance(altitude, bool) or altitude is None or not math.isfinite(float(altitude)):
        raise ValueError('a finite takeoff absolute altitude is required for mission export')
    if config.get('test_only') or config.get('altitude_mode') == 'relative_height_test':
        raise ValueError('reconstruction-only sessions cannot export flight missions')
    if not plan.get('actions'):
        raise ValueError('no feasible capture strips')
    mission, mapping = convert(plan, manifest, camera, float(altitude))
    mission['schema_version'] = mode['mission_schema_version']
    if mode['mission_schema_version'] == 14:
        mission['recapture_flight_mode'] = mode['recapture_flight_mode']
    mission['name'] = 'OpenFlyScan reacquisition'
    mission['camera_profile'].update(config.get('camera_profile', {}))
    for region, report in zip(mission['active_mapping']['regions'], plan['area_reports']):
        region['risk_score'] = max(selection['selected'][rank - 1]['score'] for rank in report['ranks'])
        region['reasons'] = region['reasons'][:2]
    mission['export_review'].update(reference_status='provided_takeoff_datum', preview_only=True)
    mission['execution_review'] = dict(safe_to_execute=False, flight_authorized=False,
                                       reason='Requires operator review of takeoff datum, camera and flight conditions')
    ceiling = float(config.get('max_relative_altitude_m', 80))
    if any(point['point']['altitude_m'] > ceiling + 1e-6 for point in mission['waypoints']):
        raise ValueError('mission exceeds the requested takeoff-relative ceiling; no coordinates were clamped')
    return mission, mapping
