import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from aiohttp.test_utils import TestClient, TestServer

from openflyscan.missions.export import mission_mode, export_mission
from openflyscan.missions.geometry_export import geo_to_enu, enu_to_geo
from openflyscan.reconstruction.cameras import camera_rotation, camera_ypr, prepare_scene, upload_metadata
from openflyscan.reconstruction.runner import spatial_groups
from openflyscan.server.http import build_app
from openflyscan.server.pipeline import update_state, atomic_json, write_cloud


class MissionProtocolTests(unittest.TestCase):
    def test_schema_negotiation_never_silently_downgrades(self):
        self.assertEqual(mission_mode({})['mission_schema_version'], 13)
        mode = mission_mode(dict(supported_mission_schemas=[13, 14], recapture_flight_mode='CONTINUOUS_EXPERIMENTAL'))
        self.assertEqual(mode['mission_schema_version'], 14)
        for payload in [dict(recapture_flight_mode='CONTINUOUS_EXPERIMENTAL'),
                        dict(recapture_flight_mode='CONTINUOUS'), dict(supported_mission_schemas=[15]),
                        dict(supported_mission_schemas=[True])]:
            with self.assertRaises(ValueError):
                mission_mode(payload)

    def test_camera_coordinate_roundtrip(self):
        reference = [31.2, 121.5, 20.]
        point = [80., -45., 65.]
        np.testing.assert_allclose(geo_to_enu(enu_to_geo(point, reference), reference), point, atol=1e-7)
        for pose in [[0, -90, 0], [90, -45, 0], [135, -40, 12]]:
            rotation = camera_rotation(pose)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_metadata_validation(self):
        self.assertIsNone(camera_ypr(dict(GimbalYawDegree=0, GimbalPitchDegree=0, GimbalRollDegree=0)))
        result = upload_metadata({'X-Camera-Yaw': '20', 'X-Camera-Pitch': '-45'}, {})
        self.assertEqual(result['GimbalYawDegree'], 20)
        for headers in [{'X-Camera-Yaw': 'nan'}, {'X-OpenFly-Metadata': '{bad'},
                        {'X-OpenFly-Metadata': json.dumps(dict(schema_version=2))},
                        {'X-OpenFly-Metadata': json.dumps(dict(schema_version=1, intrinsics=np.eye(3).tolist()))}]:
            with self.assertRaises((ValueError, TypeError)):
                upload_metadata(headers, {})

    def test_missing_attitude_disables_pose_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            Image.new('RGB', (80, 60)).save(root / 'images/frame.jpg')
            row = dict(stored_name='frame.jpg', sequence=1, latitude=31., longitude=121., altitude_m=50.)
            manifest = prepare_scene(root, [row], dict(horizontal_fov_deg=70), root / 'work')
            self.assertEqual(manifest['pose_prior'], 'none')
            self.assertIsNone(manifest['records'][0]['camera_ypr_deg'])
            self.assertFalse((root / 'work/sensor_scene/images/frame_000001.jpg').is_symlink())

    def test_spatial_groups_cover_every_image(self):
        generator = np.random.default_rng(20260921)
        centers = generator.normal(size=(101, 3))
        groups = spatial_groups(centers)
        self.assertEqual(set(index for core, _ in groups for index in core), set(range(101)))
        self.assertTrue(all(len(indices) == 30 and len(set(indices)) == 30 for _, indices in groups))

    def test_point_cloud_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / 'cloud.npz', xyz=np.array([[1., 2., 3.], [2., 3., 4.]]), rgb=np.ones((2, 3)))
            self.assertEqual(write_cloud(root, root / 'cloud.npz', 180000), 2)
            self.assertEqual((root / 'cloud.bin').read_bytes()[:4], b'V86C')
            self.assertIn(b'element vertex 2', (root / 'point_cloud.ply').read_bytes())


class WorkstationHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / 'sessions').mkdir()
        static = Path(__file__).resolve().parents[1] / 'openflyscan/server/static'
        self.client = TestClient(TestServer(build_app(self.root, static, 'test-token')))
        await self.client.start_server()
        self.headers = {'Authorization': 'Bearer test-token'}

    async def asyncTearDown(self):
        await self.client.close()
        self.directory.cleanup()

    async def create(self, **values):
        response = await self.client.post('/api/sessions', headers=self.headers, json=dict(auto_preview=False, **values))
        self.assertEqual(response.status, 201, await response.text())
        return await response.json()

    async def test_auth_and_health(self):
        response = await self.client.get('/api/sessions')
        self.assertEqual(response.status, 401)
        response = await self.client.get('/health')
        self.assertEqual((await response.json())['service'], 'openflyscan-workstation')

    async def test_explicit_continuous_mode_and_dynamic_result_schema(self):
        session = await self.create(supported_mission_schemas=[13, 14], recapture_flight_mode='CONTINUOUS_EXPERIMENTAL')
        path = self.root / 'sessions' / session['id']
        atomic_json(path / 'artifacts/mission.json', dict(schema_version=14))
        update_state(path, artifacts=dict(mission=f'/api/sessions/{session["id"]}/artifacts/mission.json'))
        response = await self.client.get(f'/api/sessions/{session["id"]}/result', headers=self.headers)
        mission = (await response.json())['openfly_v5_mission']
        self.assertEqual(mission['schema_version'], 14)
        self.assertFalse(mission['safe_to_execute'])

    async def test_unknown_schema_and_nonfinite_config_rejected(self):
        for values in [dict(supported_mission_schemas=[13], recapture_flight_mode='CONTINUOUS_EXPERIMENTAL'),
                       dict(max_relative_altitude_m=float('nan'))]:
            response = await self.client.post('/api/sessions', headers=self.headers, json=values)
            self.assertEqual(response.status, 400)

    async def test_upload_retry_metadata_and_sealed_session(self):
        session = await self.create()
        image = io.BytesIO()
        Image.new('RGB', (80, 60), (100, 120, 140)).save(image, format='JPEG')
        headers = dict(self.headers, **{'X-Latitude': '31', 'X-Longitude': '121', 'X-Altitude': '60',
                                     'X-Camera-Yaw': '10', 'X-Camera-Pitch': '-45', 'X-Filename': 'camera.jpg'})
        url = f'/api/sessions/{session["id"]}/images/1'
        response = await self.client.put(url, headers=headers, data=image.getvalue())
        self.assertEqual(response.status, 201, await response.text())
        self.assertEqual((await response.json())['image']['GimbalYawDegree'], 10)
        response = await self.client.put(url, headers=headers, data=image.getvalue())
        self.assertTrue((await response.json())['duplicate'])
        update_state(self.root / 'sessions' / session['id'], sealed=True)
        response = await self.client.put(url, headers=headers, data=image.getvalue())
        self.assertEqual(response.status, 409)

    async def test_cancel_is_idempotent_and_blocks_work(self):
        session = await self.create()
        url = f'/api/sessions/{session["id"]}'
        for _ in range(2):
            response = await self.client.post(url + '/cancel', headers=self.headers)
            self.assertTrue((await response.json())['cancelled'])
        response = await self.client.post(url + '/finalize', headers=self.headers)
        self.assertEqual(response.status, 409)


if __name__ == '__main__':
    unittest.main()
