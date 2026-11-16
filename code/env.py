"""CARLA 0.9.16 lane-keeping environment with a fixed road-distance horizon.

The only spatial input is the approved Town04 spawn-point CSV. No destination
point-to-point routes or destinations are generated. Each episode starts at one
of the approved spawn points and accumulates forward displacement projected
onto the local CARLA driving-waypoint tangent until the fixed target distance
is reached (or a terminal event / tick limit occurs).
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import queue

import numpy as np
from config import EnvironmentConfig
from weather_physics import WeatherPhysicsController


def read_spawn_ids(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or 'id' not in reader.fieldnames:
            raise ValueError(f'{path}: CSV must contain an id column')
        values = [int(row['id']) for row in reader if row.get('id', '').strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError(f'{path}: missing or duplicated spawn IDs')
    return values


def validate_spawn_ids(ids: list[int], spawn_count: int) -> None:
    invalid = [i for i in ids if i < 0 or i >= spawn_count]
    if invalid:
        raise ValueError(
            f'{len(invalid)} CSV spawn IDs outside 0..{spawn_count - 1}: {invalid[:8]}. '
            'Check CARLA map version and ID indexing.'
        )
    if not ids:
        raise ValueError('The spawn CSV contains no allowed start points.')


def projected_forward_increment(previous_xy, current_xy, previous_forward,
                                current_forward, max_step_m: float) -> float:
    """Positive displacement along the average local lane tangent for one tick.

    Lateral movement is excluded by projection. Reverse movement and implausible
    teleports do not increase route progress.
    """
    delta = np.asarray(current_xy, dtype=np.float64) - np.asarray(previous_xy, dtype=np.float64)
    f0 = np.asarray(previous_forward, dtype=np.float64)
    f1 = np.asarray(current_forward, dtype=np.float64)
    tangent = f0 + f1
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-8:
        tangent = f1
        norm = float(np.linalg.norm(tangent))
    if norm < 1e-8:
        return 0.0
    tangent /= norm
    increment = float(np.dot(delta, tangent))
    if increment <= 0.0 or increment > float(max_step_m):
        return 0.0
    return increment


class CarlaLaneEnv:
    def __init__(self, cfg: EnvironmentConfig):
        try:
            import carla
        except ImportError as e:
            raise RuntimeError('CARLA 0.9.16 PythonAPI must be importable.') from e
        self.carla = carla
        self.cfg = cfg
        self.client = carla.Client(cfg.carla_server_ip, cfg.port)
        self.client.set_timeout(90)
        for label, getter in [('client', 'get_client_version'), ('server', 'get_server_version')]:
            if hasattr(self.client, getter):
                version = str(getattr(self.client, getter)())
                # The installed Windows CARLA 0.9.16 build reports its revision over RPC.
                if not (version.startswith('0.9.16') or version == 'f06817e84'):
                    raise RuntimeError(
                        f'CARLA {label} version {version} does not match the manuscript (0.9.16).'
                    )
        self.world = self.client.get_world()
        if cfg.town not in self.world.get_map().name:
            self.world = self.client.load_world(cfg.town)
        self.original_settings = self.world.get_settings()
        self.original_weather = self.world.get_weather()
        ws = self.world.get_settings()
        ws.synchronous_mode = True
        ws.fixed_delta_seconds = cfg.fixed_delta_seconds
        self.world.apply_settings(ws)
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.route_rng = np.random.default_rng(cfg.random_seed)
        self.allowed_starts = read_spawn_ids(cfg.spawn_csv)
        if self.allowed_starts is None:
            raise ValueError('A spawn-point CSV with an id column is required.')
        validate_spawn_ids(self.allowed_starts, len(self.spawn_points))
        invalid_road_starts = []
        for spawn_id in self.allowed_starts:
            waypoint = self.map.get_waypoint(
                self.spawn_points[spawn_id].location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                invalid_road_starts.append(spawn_id)
        if invalid_road_starts:
            raise ValueError(
                f'{len(invalid_road_starts)} approved spawn IDs do not project to a Driving waypoint: '
                f'{invalid_road_starts[:10]}'
            )

        self.sensor_queue: queue.Queue = queue.Queue()
        self.vehicle = None
        self.sensors = []
        self.collision = False
        self.lane_departure = False
        self.route_length = float(cfg.route_target_distance_m)
        self.progress_distance_m = 0.0
        self.route_id = ''
        self.start_index = -1
        self._last_progress_xy = None
        self._last_lane_forward_xy = None
        self.completion = 0.0
        self.weather = WeatherPhysicsController()
        self._closed = False

    def get_rng_state(self):
        return self.route_rng.bit_generator.state

    def set_rng_state(self, state):
        self.route_rng.bit_generator.state = state

    def _destroy_actors(self):
        for s in reversed(self.sensors):
            try: s.stop()
            except Exception: pass
            try: s.destroy()
            except Exception: pass
        self.sensors.clear()
        if self.vehicle is not None:
            try: self.vehicle.destroy()
            except Exception: pass
            self.vehicle = None

    def _camera_callback(self, image):
        self.sensor_queue.put(image)

    def _get_image(self, frame_id: int):
        for _ in range(30):
            try: image = self.sensor_queue.get(timeout=2.0)
            except queue.Empty: continue
            if image.frame < frame_id:
                continue
            if image.frame != frame_id:
                raise RuntimeError(f'Semantic camera frame mismatch: {image.frame} != {frame_id}')
            image.convert(self.carla.ColorConverter.CityScapesPalette)
            raw = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
            # The retained CarEnv source passes CARLA CityScapesPalette as BGR.
            return np.ascontiguousarray(raw[:, :, :3])
        raise TimeoutError('No matching semantic camera frame; check synchronous CARLA configuration')

    @staticmethod
    def _yaw_delta(a, b):
        return ((a - b + 180.) % 360.) - 180.

    def _road_metrics(self):
        loc = self.vehicle.get_location()
        point = np.array([loc.x, loc.y], dtype=np.float64)
        waypoint = self.map.get_waypoint(
            loc, project_to_road=True, lane_type=self.carla.LaneType.Driving
        )
        if waypoint is None:
            raise RuntimeError('No CARLA Driving waypoint near vehicle position.')

        road_transform = waypoint.transform
        road_loc = road_transform.location
        forward = road_transform.get_forward_vector()
        forward_xy = np.array([float(forward.x), float(forward.y)], dtype=np.float64)
        forward_norm = float(np.linalg.norm(forward_xy))
        if forward_norm < 1e-8:
            raise RuntimeError('CARLA returned a degenerate driving-waypoint tangent.')
        forward_xy /= forward_norm

        # Initialize the odometer baseline after the reset tick. Thereafter,
        # count only positive longitudinal displacement along the local lane.
        if self._last_progress_xy is None:
            self._last_progress_xy = point
            self._last_lane_forward_xy = forward_xy
        else:
            increment = projected_forward_increment(
                self._last_progress_xy,
                point,
                self._last_lane_forward_xy,
                forward_xy,
                self.cfg.max_progress_per_tick_m,
            )
            self.progress_distance_m += increment
            self.completion = min(1.0, self.progress_distance_m / self.route_length)
            self._last_progress_xy = point
            self._last_lane_forward_xy = forward_xy

        dx = float(loc.x - road_loc.x)
        dy = float(loc.y - road_loc.y)
        signed_cross = float(forward.x * dy - forward.y * dx)
        lateral = math.hypot(dx, dy) * (-1.0 if signed_cross < 0.0 else 1.0)
        heading = abs(self._yaw_delta(
            float(self.vehicle.get_transform().rotation.yaw), float(road_transform.rotation.yaw)
        ))
        return lateral, heading

    def _observation_and_info(self, frame_id: int):
        frame = self._get_image(frame_id)
        vel = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        lateral, heading = self._road_metrics()
        info = {
            'speed': float(speed_kmh), 'lateral_dist': lateral, 'angle': heading,
            'route_completion': self.completion,
            'route_progress_m': self.progress_distance_m,
            'collision': bool(self.collision), 'lane_departure': bool(self.lane_departure),
            'route_id': self.route_id, 'route_length_m': self.route_length,
            'start_spawn_index': self.start_index,
        }
        return {'segmentation': frame, 'speed': float(speed_kmh)}, info

    def reset(self, start_speed: float = 10., condition: str = 'Clear'):
        had_vehicle = self.vehicle is not None
        self._destroy_actors()
        if had_vehicle:
            self.world.tick()
        # The paper protocol uses ZERO dynamic traffic and pedestrians.
        actors = self.world.get_actors()
        other_vehicles = [a for a in actors.filter('vehicle.*') if getattr(a, 'is_alive', True)]
        pedestrians = [a for a in actors.filter('walker.pedestrian.*') if getattr(a, 'is_alive', True)]
        if other_vehicles or pedestrians:
            raise RuntimeError(
                f'The CARLA world contains {len(other_vehicles)} other vehicles and '
                f'{len(pedestrians)} pedestrians. The protocol requires zero. '
                'Clear the CARLA world before training.'
            )
        self.collision = self.lane_departure = False
        while not self.sensor_queue.empty():
            try: self.sensor_queue.get_nowait()
            except queue.Empty: break

        self.start_index = int(self.allowed_starts[int(self.route_rng.integers(len(self.allowed_starts)))])
        self.route_length = float(self.cfg.route_target_distance_m)
        self.progress_distance_m = 0.0
        self.completion = 0.0
        self._last_progress_xy = None
        self._last_lane_forward_xy = None
        self.route_id = f'{self.cfg.town}:spawn-{self.start_index}:fixed-{self.route_length:.3f}m'

        bp = self.world.get_blueprint_library()
        candidates = bp.filter('vehicle.tesla.model3')
        if not candidates:
            raise RuntimeError('Tesla Model3 vehicle blueprint missing')
        transform = self.spawn_points[self.start_index]
        self.vehicle = self.world.try_spawn_actor(candidates[0], transform)
        if self.vehicle is None:
            raise RuntimeError(f'Failed to spawn vehicle at {self.start_index}; ensure spawn location is clear')
        camera_bp = bp.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(self.cfg.image_width))
        camera_bp.set_attribute('image_size_y', str(self.cfg.image_height))
        camera_bp.set_attribute('fov', str(self.cfg.camera_fov_deg))
        camera_pos = self.carla.Transform(self.carla.Location(x=self.cfg.camera_x_m, z=self.cfg.camera_z_m))
        cam = self.world.spawn_actor(camera_bp, camera_pos, attach_to=self.vehicle)
        cam.listen(self._camera_callback)
        self.sensors.append(cam)
        collision_sensor = self.world.spawn_actor(bp.find('sensor.other.collision'), self.carla.Transform(), attach_to=self.vehicle)
        departure_sensor = self.world.spawn_actor(bp.find('sensor.other.lane_invasion'), self.carla.Transform(), attach_to=self.vehicle)
        collision_sensor.listen(lambda _: setattr(self, 'collision', True))
        departure_sensor.listen(lambda _: setattr(self, 'lane_departure', True))
        self.sensors.extend([collision_sensor, departure_sensor])
        self.weather.apply(self, condition)
        forward = transform.get_forward_vector()
        value = float(start_speed) / 3.6
        self.vehicle.set_target_velocity(self.carla.Vector3D(forward.x * value, forward.y * value, forward.z * value))
        frame_id = self.world.tick()
        fr = self.weather.last_applied
        for idx, wheel in enumerate(self.vehicle.get_physics_control().wheels):
            observed = float(getattr(wheel, fr['wheel_attribute']))
            expected = fr['applied_wheel_friction'][idx]
            if abs(observed - expected) > max(1e-4, 1e-4 * abs(expected)):
                raise RuntimeError(f'Wheel friction mismatch after world.tick: '
                                   f'wheel={idx}, expected={expected}, observed={observed}')
        observation, info = self._observation_and_info(frame_id)
        return observation, info

    def step(self, action):
        throttle, steer, brake = map(float, action)
        self.vehicle.apply_control(self.carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        frame_id = self.world.tick()
        observation, info = self._observation_and_info(frame_id)
        done = bool(self.collision or self.lane_departure or self.completion >= 0.95)
        return observation, 0.0, done, info

    def close(self):
        if self._closed:
            return
        self._destroy_actors()
        self.world.apply_settings(self.original_settings)
        self.world.set_weather(self.original_weather)
        self._closed = True
