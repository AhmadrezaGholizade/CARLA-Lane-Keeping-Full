from __future__ import annotations

import math
import random
import threading
import time

import carla
import cv2
import numpy as np
from carla import ColorConverter, Location, Rotation, Transform


class SensorTimeoutError(RuntimeError):
    """Recoverable failure to receive the requested synchronous sensor frame."""

    def __init__(self, missing, world_frame, timeout, retries, phase="step", details=None):
        self.missing = list(missing)
        self.world_frame = world_frame
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.phase = str(phase)
        self.details = dict(details or {})
        super().__init__(
            f"Sensor timeout during {self.phase}: {self.missing}; "
            f"timeout={self.timeout:.1f}s retries={self.retries} "
            f"world_frame={self.world_frame}"
        )


class CarEnv:
    def __init__(self, config: dict):
        self.config = config
        self.client = carla.Client(config["carla_server_ip"], config["port"])
        self.client.set_timeout(float(config.get("client_timeout_seconds", 20.0)))
        self.world = self.client.load_world(config["town"])
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.filter(config["vehicle_name"])[0]
        self.actor_list: list[carla.Actor] = []
        self.traffic_actors: list[carla.Actor] = []
        # Keep IDs as well as actor proxies. Batch-spawned actors can briefly
        # lack a local proxy before the next world tick, but their IDs are still
        # required for deterministic cleanup and traffic top-up accounting.
        self.traffic_actor_ids: set[int] = set()
        self.traffic_role_name = str(config.get("traffic_role_name", "rl_training_traffic"))
        self.traffic_profile_config: dict = {}
        self.traffic_profile_by_actor_id: dict[int, str] = {}
        self.traffic_desired_speed_by_actor_id: dict[int, float] = {}
        self.traffic_anchor_by_actor_id: dict[int, str] = {}
        self.traffic_target_total = 0
        self._traffic_rng = random.Random()
        self.vehicle = None
        self._closed = False
        self.use_3_quarter = config["3_quarter"]

        self.rgb_image = None; self.segmented_image = None; self.game_image = None
        self.gps_data = None
        self.collision_occurred = False; self.lane_invasion_occurred = False
        self.collision_event_info = {
            "collision_other_actor_id": -1,
            "collision_other_actor_type": "",
            "collision_impulse": 0.0,
            "collision_frame": -1,
        }

        self._rgb_lock = threading.Lock(); self._rgb_event = threading.Event()
        self._seg_lock = threading.Lock(); self._seg_event = threading.Event()
        self._game_lock = threading.Lock(); self._game_event = threading.Event()
        self._gps_lock = threading.Lock(); self._gps_event = threading.Event()
        self._collision_lock = threading.Lock(); self._lane_lock = threading.Lock()

        # Sensor callbacks from an actor destroyed during the previous episode may
        # still be queued in the Python client.  A generation token prevents those
        # stale callbacks from satisfying the next episode's readiness checks.
        self.sensor_actors: list[carla.Actor] = []
        self._sensor_generation = 0
        self._sensor_condition = threading.Condition()
        self._sensor_frames = {
            "rgb": -1, "segmentation": -1, "game_camera": -1,
            "GPS": -1,
        }
        self._sensor_callback_errors: dict[str, str] = {}
        self._sensor_stale_streak = 0
        self._sensor_lag_accept_count = 0
        self._sensor_preferred_lag = -1
        self._sensor_preferred_lag_confidence = 0
        self._last_sensor_health = {
            "sensor_data_stale": False,
            "sensor_stale_names": "",
            "sensor_max_frame_lag": 0,
            "sensor_stale_streak": 0,
            "sensor_world_frame": -1,
            "sensor_bundle_frame": -1,
            "sensor_bundle_spread": -1,
        }

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config["fixed_delta_seconds"]
        self.world.apply_settings(settings)

    def get_spawn_points(self):
        return self.map.get_spawn_points()

    def get_speed(self):
        velocity = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def get_angle_and_dist(self):
        transform = self.vehicle.get_transform()
        location = transform.location
        waypoint = self.map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        raw_waypoint = self.map.get_waypoint(
            location, project_to_road=False, lane_type=carla.LaneType.Driving
        )
        road_transform = waypoint.transform
        angle = (transform.rotation.yaw - road_transform.rotation.yaw + 180.0) % 360.0 - 180.0
        dx = location.x - road_transform.location.x
        dy = location.y - road_transform.location.y
        forward = road_transform.get_forward_vector()
        cross = forward.x * dy - forward.y * dx
        lateral = math.hypot(dx, dy) * (-1.0 if cross < 0 else 1.0)
        return angle, lateral, waypoint, raw_waypoint is not None

    @staticmethod
    def _actor_speed_kmh(actor) -> float:
        velocity = actor.get_velocity()
        return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    @staticmethod
    def _forward_dot(first, second) -> float:
        return first.x * second.x + first.y * second.y + first.z * second.z

    def _valid_adjacent_lane(self, waypoint, side: str):
        candidate = waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
        if candidate is None or candidate.lane_type != carla.LaneType.Driving:
            return None
        # Reject the adjacent lane when it carries traffic in the opposite
        # direction. CARLA may return such a lane across a centre marking.
        current_forward = waypoint.transform.get_forward_vector()
        candidate_forward = candidate.transform.get_forward_vector()
        if self._forward_dot(current_forward, candidate_forward) < 0.5:
            return None
        return candidate


    @staticmethod
    def _copy_transform(transform, z_offset: float = 0.0):
        location = transform.location
        rotation = transform.rotation
        return carla.Transform(
            carla.Location(
                x=float(location.x),
                y=float(location.y),
                z=float(location.z) + float(z_offset),
            ),
            carla.Rotation(
                pitch=float(rotation.pitch),
                yaw=float(rotation.yaw),
                roll=float(rotation.roll),
            ),
        )

    def _traffic_profile_counts(self) -> dict[str, int]:
        live_ids = {int(actor.id) for actor in self._world_npc_vehicles()}
        return {
            profile: sum(
                1 for actor_id, value in self.traffic_profile_by_actor_id.items()
                if actor_id in live_ids and value == profile
            )
            for profile in ("slow_blocker", "slow", "normal", "fast")
        }

    def traffic_profile_summary(self) -> dict:
        counts = self._traffic_profile_counts()
        live_ids = {int(actor.id) for actor in self._world_npc_vehicles()}
        speeds = [
            float(speed)
            for actor_id, speed in self.traffic_desired_speed_by_actor_id.items()
            if actor_id in live_ids
        ]
        anchors = {"slow_ahead": 0, "fast_rear": 0}
        for actor_id, anchor_name in self.traffic_anchor_by_actor_id.items():
            if actor_id in live_ids and anchor_name in anchors:
                anchors[anchor_name] += 1
        return {
            **{f"{name}_count": int(value) for name, value in counts.items()},
            "slow_ahead_anchors": int(anchors["slow_ahead"]),
            "fast_rear_anchors": int(anchors["fast_rear"]),
            "desired_speed_min_kmh": min(speeds) if speeds else math.nan,
            "desired_speed_max_kmh": max(speeds) if speeds else math.nan,
            "desired_speed_mean_kmh": (sum(speeds) / len(speeds)) if speeds else math.nan,
        }

    def _profile_targets(self, target_total: int) -> dict[str, int]:
        profiles = ("slow_blocker", "slow", "normal", "fast")
        cfg = self.traffic_profile_config
        weights = cfg.get("weights", {})
        raw = {name: max(0.0, float(weights.get(name, 0.0))) for name in profiles}
        weight_sum = sum(raw.values())
        if weight_sum <= 0.0:
            raw = {name: 1.0 for name in profiles}
            weight_sum = float(len(profiles))
        exact = {name: float(target_total) * raw[name] / weight_sum for name in profiles}
        targets = {name: int(math.floor(value)) for name, value in exact.items()}
        remaining = max(0, int(target_total) - sum(targets.values()))
        order = sorted(profiles, key=lambda name: exact[name] - targets[name], reverse=True)
        for name in order[:remaining]:
            targets[name] += 1
        if target_total > 0:
            targets["slow_blocker"] = max(
                targets["slow_blocker"],
                min(int(cfg.get("priority_slow_ahead_count", 0)), int(target_total)),
            )
            targets["fast"] = max(
                targets["fast"],
                min(int(cfg.get("priority_fast_rear_count", 0)), int(target_total)),
            )
            overflow = sum(targets.values()) - int(target_total)
            for name in ("normal", "slow", "fast", "slow_blocker"):
                if overflow <= 0:
                    break
                protected = 0
                if name == "slow_blocker":
                    protected = int(cfg.get("priority_slow_ahead_count", 0))
                elif name == "fast":
                    protected = int(cfg.get("priority_fast_rear_count", 0))
                reducible = max(0, targets[name] - protected)
                take = min(reducible, overflow)
                targets[name] -= take
                overflow -= take
        return targets

    def _select_profile(self, planned_counts: dict[str, int], target_total: int) -> str:
        targets = self._profile_targets(target_total)
        deficits = {
            name: max(0, int(targets.get(name, 0)) - int(planned_counts.get(name, 0)))
            for name in targets
        }
        candidates = [name for name, deficit in deficits.items() if deficit > 0]
        if not candidates:
            candidates = list(targets)
        weights = [max(1, deficits.get(name, 0)) for name in candidates]
        return self._traffic_rng.choices(candidates, weights=weights, k=1)[0]

    def _priority_lane_transforms(self, min_distance_from_ego: float) -> list[dict]:
        cfg = self.traffic_profile_config
        if not cfg.get("enabled", False) or self.vehicle is None:
            return []
        ego_location = self.vehicle.get_location()
        ego_wp = self.map.get_waypoint(
            ego_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if ego_wp is None:
            return []
        z_offset = float(cfg.get("spawn_z_offset_m", 0.35))
        spacing = max(5.0, float(cfg.get("priority_spacing_m", 16.0)))
        occupied = [actor.get_location() for actor in self._world_npc_vehicles()]
        occupied.append(ego_location)
        plans: list[dict] = []

        def usable(transform) -> bool:
            location = transform.location
            if location.distance(ego_location) < float(min_distance_from_ego):
                return False
            for other in occupied:
                if location.distance(other) < 5.0:
                    return False
            for plan in plans:
                if location.distance(plan["transform"].location) < 8.0:
                    return False
            return True

        slow_min, slow_max = map(
            float, cfg.get("slow_ahead_distance_range_m", [38.0, 78.0])
        )
        slow_needed = max(0, int(cfg.get("priority_slow_ahead_count", 0)))
        for index in range(slow_needed):
            distance = min(slow_max, slow_min + index * spacing)
            candidates = ego_wp.next(distance)
            candidate = next((wp for wp in candidates if wp.lane_type == carla.LaneType.Driving), None)
            if candidate is None:
                continue
            transform = self._copy_transform(candidate.transform, z_offset)
            if usable(transform):
                plans.append({
                    "transform": transform,
                    "profile": "slow_blocker",
                    "anchor": "slow_ahead",
                })

        rear_min, rear_max = map(
            float, cfg.get("fast_rear_distance_range_m", [32.0, 72.0])
        )
        adjacent_lanes = [
            lane for lane in (
                self._valid_adjacent_lane(ego_wp, "left"),
                self._valid_adjacent_lane(ego_wp, "right"),
            )
            if lane is not None
        ]
        if not adjacent_lanes:
            adjacent_lanes = [ego_wp]
        fast_needed = max(0, int(cfg.get("priority_fast_rear_count", 0)))
        for index in range(fast_needed):
            lane_wp = adjacent_lanes[index % len(adjacent_lanes)]
            distance = min(rear_max, rear_min + index * spacing)
            candidates = lane_wp.previous(distance)
            candidate = next((wp for wp in candidates if wp.lane_type == carla.LaneType.Driving), None)
            if candidate is None:
                continue
            transform = self._copy_transform(candidate.transform, z_offset)
            if usable(transform):
                plans.append({
                    "transform": transform,
                    "profile": "fast",
                    "anchor": "fast_rear",
                })
        return plans

    def _apply_traffic_profile(self, actor, profile: str, anchor: str = "") -> None:
        cfg = self.traffic_profile_config
        actor_id = int(actor.id)
        desired_ranges = cfg.get("desired_speed_ranges_kmh", {})
        desired_range = desired_ranges.get(profile, [35.0, 55.0])
        if anchor == "slow_ahead":
            desired_range = cfg.get(
                "priority_slow_ahead_speed_range_kmh", desired_range
            )
        elif anchor == "fast_rear":
            desired_range = cfg.get(
                "priority_fast_rear_speed_range_kmh", desired_range
            )
        low, high = sorted(map(float, desired_range))
        desired_speed = self._traffic_rng.uniform(low, high)
        applied = False
        try:
            setter = getattr(self.traffic_manager, "set_desired_speed", None)
            if callable(setter):
                setter(actor, float(desired_speed))
                applied = True
        except Exception as exc:
            print(f"[traffic-profile] set_desired_speed failed for actor={actor_id}: {exc}")

        if not applied:
            fallback = cfg.get("fallback_percentage_speed_difference", {}).get(
                profile, [25.0, 25.0]
            )
            pct_low, pct_high = sorted(map(float, fallback))
            percentage = self._traffic_rng.uniform(pct_low, pct_high)
            try:
                self.traffic_manager.vehicle_percentage_speed_difference(
                    actor, float(percentage)
                )
            except Exception as exc:
                print(f"[traffic-profile] speed fallback failed for actor={actor_id}: {exc}")

        try:
            if profile == "slow_blocker":
                self.traffic_manager.distance_to_leading_vehicle(
                    actor, float(cfg.get("blocker_follow_distance_m", 5.0))
                )
                if bool(cfg.get("disable_auto_lane_change_for_blockers", True)):
                    self.traffic_manager.auto_lane_change(actor, False)
            elif profile == "fast":
                self.traffic_manager.distance_to_leading_vehicle(
                    actor, float(cfg.get("fast_follow_distance_m", 2.0))
                )
                if anchor == "fast_rear" and bool(
                    cfg.get("disable_auto_lane_change_for_priority_fast_rear", True)
                ):
                    self.traffic_manager.auto_lane_change(actor, False)
        except Exception as exc:
            print(f"[traffic-profile] behaviour setup failed for actor={actor_id}: {exc}")

        self.traffic_profile_by_actor_id[actor_id] = str(profile)
        self.traffic_desired_speed_by_actor_id[actor_id] = float(desired_speed)
        if anchor:
            self.traffic_anchor_by_actor_id[actor_id] = str(anchor)

    def _world_npc_vehicles(self) -> list[carla.Actor]:
        """Return every live vehicle in the CARLA world except the ego actor.

        This deliberately does not rely on ``traffic_actor_ids``.  A CARLA
        batch can create visible actors before local proxies are refreshed, and
        a previous interrupted process can also leave NPCs in the world.  Both
        cases must still be visible to traffic-aware rewards and logging.
        """
        ego_id = None
        try:
            if self.vehicle is not None:
                ego_id = int(self.vehicle.id)
        except Exception:
            ego_id = None

        result: list[carla.Actor] = []
        try:
            actors = self.world.get_actors().filter("vehicle.*")
        except Exception:
            return result
        for actor in actors:
            try:
                if ego_id is not None and int(actor.id) == ego_id:
                    continue
                if actor.is_alive:
                    result.append(actor)
            except Exception:
                continue
        return result

    def _refresh_traffic_from_world(self) -> list[carla.Actor]:
        actors = self._world_npc_vehicles()
        self.traffic_actors = actors
        self.traffic_actor_ids = {int(actor.id) for actor in actors}
        live_ids = set(self.traffic_actor_ids)
        self.traffic_profile_by_actor_id = {
            actor_id: value for actor_id, value in self.traffic_profile_by_actor_id.items()
            if actor_id in live_ids
        }
        self.traffic_desired_speed_by_actor_id = {
            actor_id: value
            for actor_id, value in self.traffic_desired_speed_by_actor_id.items()
            if actor_id in live_ids
        }
        self.traffic_anchor_by_actor_id = {
            actor_id: value for actor_id, value in self.traffic_anchor_by_actor_id.items()
            if actor_id in live_ids
        }
        return actors

    def _clear_world_npc_vehicles(self) -> None:
        """Remove stale NPC vehicles left by interrupted training processes."""
        actors = self._world_npc_vehicles()
        if not actors:
            return
        commands = [carla.command.DestroyActor(int(actor.id)) for actor in actors]
        responses = self.client.apply_batch_sync(commands, True)
        failures = [str(r.error) for r in responses if r.error]
        if failures:
            print(f"[cleanup] stale NPC cleanup failed for {len(failures)}/{len(responses)} actors")
        else:
            print(f"[cleanup] removed {len(responses)} stale NPC vehicles")
        self.traffic_actors = []
        self.traffic_actor_ids = set()
        self.traffic_profile_by_actor_id = {}
        self.traffic_desired_speed_by_actor_id = {}
        self.traffic_anchor_by_actor_id = {}

    def _traffic_snapshots(self, ego_location, max_distance: float) -> list[dict]:
        snapshots = []
        for actor in self._refresh_traffic_from_world():
            try:
                if not actor.is_alive:
                    continue
                actor_location = actor.get_location()
                if actor_location.distance(ego_location) > max_distance * 1.5:
                    continue
                actor_wp = self.map.get_waypoint(
                    actor_location, project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if actor_wp is None:
                    continue
                snapshots.append({
                    "id": int(actor.id),
                    "location": actor_location,
                    "waypoint": actor_wp,
                    "speed_kmh": self._actor_speed_kmh(actor),
                })
            except (RuntimeError, AttributeError):
                continue
        return snapshots

    def _lane_vehicle_gaps(
        self, lane_waypoint, max_distance: float, snapshots: list[dict]
    ) -> dict:
        result = {
            "front_gap_m": float(max_distance),
            "front_speed_kmh": math.nan,
            "front_actor_id": -1,
            "rear_gap_m": float(max_distance),
            "rear_speed_kmh": math.nan,
            "rear_actor_id": -1,
        }
        if lane_waypoint is None or not snapshots:
            return result

        lane_forward = lane_waypoint.transform.get_forward_vector()
        lane_origin = lane_waypoint.transform.location
        lane_road_id = int(lane_waypoint.road_id)
        lane_id = int(lane_waypoint.lane_id)

        for snapshot in snapshots:
            actor_wp = snapshot["waypoint"]
            if int(actor_wp.road_id) != lane_road_id or int(actor_wp.lane_id) != lane_id:
                continue
            actor_forward = actor_wp.transform.get_forward_vector()
            if self._forward_dot(lane_forward, actor_forward) < 0.5:
                continue
            delta = snapshot["location"] - lane_origin
            longitudinal = (
                delta.x * lane_forward.x
                + delta.y * lane_forward.y
                + delta.z * lane_forward.z
            )
            speed = float(snapshot["speed_kmh"])
            if 0.0 < longitudinal < result["front_gap_m"]:
                result.update({
                    "front_gap_m": float(longitudinal),
                    "front_speed_kmh": speed,
                    "front_actor_id": int(snapshot["id"]),
                })
            elif longitudinal < 0.0 and abs(longitudinal) < result["rear_gap_m"]:
                result.update({
                    "rear_gap_m": float(abs(longitudinal)),
                    "rear_speed_kmh": speed,
                    "rear_actor_id": int(snapshot["id"]),
                })
        return result

    def get_traffic_context(self, waypoint, max_distance: float = 80.0) -> dict:
        ego_location = self.vehicle.get_location()
        snapshots = self._traffic_snapshots(ego_location, max_distance)
        context = {}
        lanes = {
            "current": waypoint,
            "left": self._valid_adjacent_lane(waypoint, "left"),
            "right": self._valid_adjacent_lane(waypoint, "right"),
        }
        for side, lane_wp in lanes.items():
            gaps = self._lane_vehicle_gaps(lane_wp, max_distance, snapshots)
            for key, value in gaps.items():
                context[f"{side}_{key}"] = value
            if side != "current":
                context[f"{side}_lane_id"] = int(lane_wp.lane_id) if lane_wp else None
                context[f"{side}_lane_road_id"] = int(lane_wp.road_id) if lane_wp else None
                context[f"{side}_lane_center_distance_m"] = (
                    float(ego_location.distance(lane_wp.transform.location))
                    if lane_wp else math.inf
                )
        return context

    def relative_longitudinal_to_actor(self, actor_id: int) -> float:
        """Signed actor distance along the ego heading; negative means passed."""
        ego_transform = self.vehicle.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        for actor in self._refresh_traffic_from_world():
            try:
                if int(actor.id) != int(actor_id) or not actor.is_alive:
                    continue
                delta = actor.get_location() - ego_location
                return float(
                    delta.x * forward.x + delta.y * forward.y + delta.z * forward.z
                )
            except (RuntimeError, AttributeError):
                continue
        return math.inf

    def get_dynamics(self, waypoint=None):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        angular = self.vehicle.get_angular_velocity()
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        longitudinal_velocity = velocity.x * forward.x + velocity.y * forward.y + velocity.z * forward.z
        lateral_velocity = velocity.x * right.x + velocity.y * right.y + velocity.z * right.z
        road_right = waypoint.transform.get_right_vector() if waypoint is not None else right
        road_lateral_velocity = (
            velocity.x * road_right.x + velocity.y * road_right.y + velocity.z * road_right.z
        )
        longitudinal_acceleration = acceleration.x * forward.x + acceleration.y * forward.y + acceleration.z * forward.z
        return {
            "yaw_rate": float(angular.z),
            "longitudinal_velocity": float(longitudinal_velocity),
            "lateral_velocity": float(lateral_velocity),
            "road_lateral_velocity": float(road_lateral_velocity),
            "longitudinal_acceleration": float(longitudinal_acceleration),
        }

    def reset(self, spawn_point_number_ids=None, start_speed=None, traffic_seed=None,
              traffic_speed_diff=0.0, hybrid_mode=True, hybrid_radius=70.0,
              traffic_profile_config=None, traffic_target_total=0):
        self._clear_actors()
        self.traffic_profile_config = dict(traffic_profile_config or {})
        self.traffic_target_total = max(0, int(traffic_target_total or 0))
        self._traffic_rng = random.Random(traffic_seed)
        self.traffic_profile_by_actor_id = {}
        self.traffic_desired_speed_by_actor_id = {}
        self.traffic_anchor_by_actor_id = {}
        if bool(self.config.get("clear_all_npc_vehicles_on_reset", True)):
            self._clear_world_npc_vehicles()
        self._closed = False
        self.rgb_image = self.segmented_image = self.game_image = None
        self.gps_data = None
        self._reset_sensor_tracking(increment_generation=True)
        with self._collision_lock:
            self.collision_occurred = False
            self.collision_event_info = {
                "collision_other_actor_id": -1,
                "collision_other_actor_type": "",
                "collision_impulse": 0.0,
                "collision_frame": -1,
            }
        with self._lane_lock: self.lane_invasion_occurred = False

        spawn_points = self.get_spawn_points()
        if spawn_point_number_ids:
            spawn = spawn_points[random.choice(spawn_point_number_ids)]
        else:
            spawn = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, spawn)
        self.actor_list.append(self.vehicle)

        self.traffic_manager = self.client.get_trafficmanager(8000)
        if traffic_seed is not None:
            self.traffic_manager.set_random_device_seed(int(traffic_seed))
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.global_percentage_speed_difference(float(traffic_speed_diff))
        self.traffic_manager.set_synchronous_mode(True)
        if hybrid_mode:
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(float(hybrid_radius))

        if start_speed is not None:
            self.world.tick()
            forward = self.vehicle.get_transform().get_forward_vector()
            speed_mps = float(start_speed) / 3.6
            self.vehicle.set_target_velocity(carla.Vector3D(
                x=forward.x * speed_mps, y=forward.y * speed_mps, z=forward.z * speed_mps
            ))
            self.world.tick()

        self._spawn_sensors()
        self._wait_initial_sensors()

    def live_traffic_count(self) -> int:
        """Return the actual number of live non-ego vehicles in the world."""
        return len(self._refresh_traffic_from_world())

    def spawn_traffic(
        self,
        count: int,
        min_distance_from_ego=50.0,
        target_total: int | None = None,
    ) -> int:
        """Spawn additional NPCs and assign deterministic mixed-speed profiles.

        Two scenario anchors are attempted before ordinary random placement:
        slow traffic in the ego lane ahead and fast traffic behind in an
        adjacent same-direction lane. Remaining actors are assigned profiles by
        deficit-balanced weights, so top-up calls preserve the requested mix.
        """
        requested = max(0, int(count))
        if requested <= 0:
            return 0
        target_total = max(
            requested,
            int(target_total if target_total is not None else self.traffic_target_total),
        )

        blueprints = [
            bp for bp in self.blueprint_library.filter("vehicle.*")
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 4
        ]
        if not blueprints:
            return 0

        existing_counts = self._traffic_profile_counts()
        planned_counts = dict(existing_counts)
        live_anchor_counts = {"slow_ahead": 0, "fast_rear": 0}
        live_ids = {int(actor.id) for actor in self._world_npc_vehicles()}
        for actor_id, anchor_name in self.traffic_anchor_by_actor_id.items():
            if actor_id in live_ids and anchor_name in live_anchor_counts:
                live_anchor_counts[anchor_name] += 1

        plans: list[dict] = []
        priority_candidates = self._priority_lane_transforms(min_distance_from_ego)
        priority_limits = {
            "slow_ahead": int(self.traffic_profile_config.get("priority_slow_ahead_count", 0)),
            "fast_rear": int(self.traffic_profile_config.get("priority_fast_rear_count", 0)),
        }
        for plan in priority_candidates:
            if len(plans) >= requested:
                break
            anchor_name = str(plan.get("anchor", ""))
            if live_anchor_counts.get(anchor_name, 0) >= priority_limits.get(anchor_name, 0):
                continue
            plans.append(plan)
            profile = str(plan["profile"])
            planned_counts[profile] = planned_counts.get(profile, 0) + 1
            live_anchor_counts[anchor_name] = live_anchor_counts.get(anchor_name, 0) + 1

        ego_location = self.vehicle.get_location()
        occupied_locations = [actor.get_location() for actor in self._world_npc_vehicles()]
        random_spawn_points = [
            self._copy_transform(transform, 0.0)
            for transform in self.get_spawn_points()
            if transform.location.distance(ego_location) >= float(min_distance_from_ego)
            and all(transform.location.distance(location) >= 5.0 for location in occupied_locations)
        ]
        self._traffic_rng.shuffle(random_spawn_points)
        for transform in random_spawn_points:
            if len(plans) >= requested:
                break
            if any(
                transform.location.distance(plan["transform"].location) < 8.0
                for plan in plans
            ):
                continue
            profile = self._select_profile(planned_counts, target_total)
            plans.append({"transform": transform, "profile": profile, "anchor": ""})
            planned_counts[profile] = planned_counts.get(profile, 0) + 1

        if not plans:
            return 0

        def prepare_blueprint():
            blueprint = self._traffic_rng.choice(blueprints)
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", self.traffic_role_name)
            if blueprint.has_attribute("color"):
                colors = blueprint.get_attribute("color").recommended_values
                if colors:
                    blueprint.set_attribute("color", self._traffic_rng.choice(colors))
            return blueprint

        try:
            batch = []
            autopilot = carla.command.SetAutopilot
            future_actor = carla.command.FutureActor
            for plan in plans:
                command = carla.command.SpawnActor(
                    prepare_blueprint(), plan["transform"]
                ).then(
                    autopilot(future_actor, True, self.traffic_manager.get_port())
                )
                batch.append(command)

            responses = self.client.apply_batch_sync(batch, True)
            actor_ids: list[int] = []
            errors: list[str] = []
            successful_plans: list[dict] = []
            for response, plan in zip(responses, plans):
                if response.error:
                    errors.append(str(response.error))
                else:
                    actor_ids.append(int(response.actor_id))
                    successful_plans.append(plan)
            if errors:
                from collections import Counter
                print(
                    f"[traffic] requested={len(batch)} spawned={len(actor_ids)} "
                    f"failed={len(errors)}"
                )
                for message, number in Counter(errors).most_common(3):
                    print(f"[traffic] {number}x {message}")

            self.traffic_actor_ids.update(actor_ids)
            actor_map = {int(actor.id): actor for actor in self.world.get_actors(actor_ids)}
            for actor_id, plan in zip(actor_ids, successful_plans):
                actor = actor_map.get(actor_id)
                if actor is None:
                    continue
                self.traffic_actors.append(actor)
                self.actor_list.append(actor)
                self._apply_traffic_profile(
                    actor, str(plan["profile"]), str(plan.get("anchor", ""))
                )
            self._refresh_traffic_from_world()
            summary = self.traffic_profile_summary()
            print(
                "[traffic-profile] "
                f"blocker={summary['slow_blocker_count']} "
                f"slow={summary['slow_count']} normal={summary['normal_count']} "
                f"fast={summary['fast_count']} "
                f"anchors={summary['slow_ahead_anchors']}/{summary['fast_rear_anchors']} "
                f"desired={summary['desired_speed_min_kmh']:.1f}-"
                f"{summary['desired_speed_max_kmh']:.1f} km/h"
            )
            return len(actor_ids)
        except Exception as batch_error:
            print(f"[traffic] batch spawn failed; using sequential fallback: {batch_error}")
            spawned = 0
            for plan in plans:
                actor = self.world.try_spawn_actor(prepare_blueprint(), plan["transform"])
                if actor is None:
                    continue
                actor.set_autopilot(True, self.traffic_manager.get_port())
                self.actor_list.append(actor)
                self.traffic_actors.append(actor)
                self.traffic_actor_ids.add(int(actor.id))
                self._apply_traffic_profile(
                    actor, str(plan["profile"]), str(plan.get("anchor", ""))
                )
                spawned += 1
            self._refresh_traffic_from_world()
            return spawned

    def _camera_bp(self, sensor_id):
        bp = self.blueprint_library.find(sensor_id)
        bp.set_attribute("image_size_x", str(self.config["image_size_x"]))
        bp.set_attribute("image_size_y", str(self.config["image_size_y"]))
        bp.set_attribute("fov", "90")
        if bp.has_attribute("sensor_tick"):
            bp.set_attribute("sensor_tick", str(self.config["fixed_delta_seconds"]))
        return bp

    def _active_sensor_names(self) -> list[str]:
        names = []
        if self.config.get("rgb", False): names.append("rgb")
        if self.config.get("segmentation", False): names.append("segmentation")
        if self.config.get("game_camera", False): names.append("game_camera")
        if self.config.get("GPS", True): names.append("GPS")
        return names

    def _reset_sensor_tracking(self, increment_generation: bool = True) -> None:
        if increment_generation:
            self._sensor_generation += 1
        with self._sensor_condition:
            for name in self._sensor_frames:
                self._sensor_frames[name] = -1
            self._sensor_callback_errors = {}
            self._sensor_condition.notify_all()
        for event in (
            self._rgb_event, self._seg_event, self._game_event,
            self._gps_event,
        ):
            event.clear()
        self._sensor_stale_streak = 0
        self._last_sensor_health = {
            "sensor_data_stale": False,
            "sensor_stale_names": "",
            "sensor_max_frame_lag": 0,
            "sensor_stale_streak": 0,
            "sensor_world_frame": -1,
            "sensor_bundle_frame": -1,
            "sensor_bundle_spread": -1,
        }

    def _store_processed_sensor_value_locked(self, name: str, value) -> None:
        """Commit a processed sample while ``_sensor_condition`` is held.

        The frame number and its processed payload must become visible as one
        transaction.  Otherwise the training thread can observe frame N-1 while
        the corresponding array has already been replaced by frame N.
        """
        if name == "rgb":
            with self._rgb_lock:
                self.rgb_image = value
            self._rgb_event.set()
        elif name == "segmentation":
            with self._seg_lock:
                self.segmented_image = value
            self._seg_event.set()
        elif name == "game_camera":
            with self._game_lock:
                self.game_image = value
            self._game_event.set()
        elif name == "GPS":
            with self._gps_lock:
                self.gps_data = value
            self._gps_event.set()
        else:
            raise KeyError(f"Unknown sensor name: {name}")

    def _register_sensor(self, sensor, name: str | None, processor) -> None:
        generation = int(self._sensor_generation)

        def callback(data):
            # Ignore callbacks queued by sensors from an earlier episode.
            if generation != self._sensor_generation:
                return
            try:
                if name is None:
                    processor(data)
                    return

                # Expensive image conversion happens outside the global
                # condition lock.  Only the final payload+frame commit is atomic.
                value = processor(data)
                frame = int(getattr(data, "frame", -1))
                with self._sensor_condition:
                    if generation != self._sensor_generation:
                        return
                    previous_frame = int(self._sensor_frames.get(name, -1))
                    if frame < previous_frame:
                        return
                    self._store_processed_sensor_value_locked(name, value)
                    self._sensor_frames[name] = frame
                    self._sensor_condition.notify_all()
            except Exception as exc:
                if name is not None:
                    with self._sensor_condition:
                        self._sensor_callback_errors[name] = repr(exc)
                        self._sensor_condition.notify_all()
                print(f"[sensor-callback-error] {name or sensor.type_id}: {exc}")

        sensor.listen(callback)
        self.sensor_actors.append(sensor)
        self.actor_list.append(sensor)

    def _spawn_sensors(self):
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        sensor = self.world.spawn_actor(
            collision_bp,
            Transform(Location(x=2.5, z=0.7)),
            attach_to=self.vehicle,
        )
        self._register_sensor(sensor, None, self._on_collision)

        if self.config.get("lane_invasion", False):
            bp = self.blueprint_library.find("sensor.other.lane_invasion")
            sensor = self.world.spawn_actor(bp, Transform(), attach_to=self.vehicle)
            self._register_sensor(sensor, None, self._on_lane_invasion)

        if self.config.get("rgb", False):
            sensor = self.world.spawn_actor(
                self._camera_bp("sensor.camera.rgb"),
                Transform(Location(x=2.4, z=1.5), Rotation(pitch=-15)),
                attach_to=self.vehicle,
            )
            self._register_sensor(sensor, "rgb", self._process_rgb)
        if self.config.get("segmentation", False):
            sensor = self.world.spawn_actor(
                self._camera_bp("sensor.camera.semantic_segmentation"),
                Transform(Location(x=2.4, z=1.5), Rotation(pitch=-15)),
                attach_to=self.vehicle,
            )
            self._register_sensor(sensor, "segmentation", self._process_seg)
        if self.config.get("game_camera", False):
            sensor = self.world.spawn_actor(
                self._camera_bp("sensor.camera.rgb"),
                Transform(Location(x=-6, z=2)),
                attach_to=self.vehicle,
            )
            self._register_sensor(sensor, "game_camera", self._process_game)
        if self.config.get("GPS", True):
            bp = self.blueprint_library.find("sensor.other.gnss")
            if bp.has_attribute("sensor_tick"):
                bp.set_attribute(
                    "sensor_tick", str(self.config["fixed_delta_seconds"])
                )
            sensor = self.world.spawn_actor(
                bp, Transform(Location(z=2.0)), attach_to=self.vehicle
            )
            self._register_sensor(sensor, "GPS", self._process_gps)

    def _sensor_bundle_snapshot_locked(
        self,
        names: list[str],
        target_frame: int,
    ) -> dict:
        frames = {
            name: int(self._sensor_frames.get(name, -1))
            for name in names
        }
        unavailable = [name for name, frame in frames.items() if frame < 0]
        available_frames = [frame for frame in frames.values() if frame >= 0]
        bundle_frame = min(available_frames) if available_frames else -1
        newest_frame = max(available_frames) if available_frames else -1
        spread = (
            newest_frame - bundle_frame
            if len(available_frames) == len(names) and available_frames
            else -1
        )
        max_lag = (
            max(int(target_frame) - frame for frame in available_frames)
            if len(available_frames) == len(names) and available_frames
            else -1
        )
        any_ahead = any(frame > int(target_frame) for frame in available_frames)
        allowed_lag = max(
            0, int(self.config.get("sensor_allowed_frame_lag", 1))
        )
        max_spread = max(
            0, int(self.config.get("sensor_max_bundle_frame_spread", 0))
        )

        exact = bool(
            not unavailable
            and spread == 0
            and bundle_frame == int(target_frame)
        )
        acceptable = bool(
            not unavailable
            and not any_ahead
            and spread >= 0
            and spread <= max_spread
            and max_lag >= 0
            and max_lag <= allowed_lag
        )

        if unavailable:
            missing = unavailable
        elif any_ahead:
            missing = [
                name for name, frame in frames.items()
                if frame > int(target_frame)
            ]
        elif spread > max_spread:
            # A mixed-frame bundle is never consumed. Report every member so
            # diagnostics make it clear that the whole bundle is incoherent.
            missing = list(names)
        elif max_lag > allowed_lag:
            missing = [
                name for name, frame in frames.items()
                if int(target_frame) - frame > allowed_lag
            ]
        else:
            missing = []

        return {
            "accepted": acceptable,
            "exact": exact,
            "frames": frames,
            "missing": missing,
            "bundle_frame": int(bundle_frame),
            "newest_frame": int(newest_frame),
            "bundle_spread": int(spread),
            "max_frame_lag": int(max_lag),
            "target_frame": int(target_frame),
            "observation": None,
        }

    def _capture_observation_locked(self) -> dict:
        """Copy all enabled sensor payloads from one atomic bundle."""
        obs = {}
        if self.config.get("rgb", False):
            with self._rgb_lock:
                obs["rgb"] = self.rgb_image.copy()
        if self.config.get("segmentation", False):
            with self._seg_lock:
                obs["segmentation"] = self.segmented_image.copy()
        if self.config.get("game_camera", False):
            with self._game_lock:
                obs["game_camera"] = self.game_image.copy()
        if self.config.get("GPS", True):
            with self._gps_lock:
                obs["gps"] = dict(self.gps_data) if self.gps_data else None
        return obs

    def _log_accepted_sensor_lag(self, phase: str, status: dict) -> None:
        lag = int(status.get("max_frame_lag", 0))
        if lag <= 0:
            return

        if lag == self._sensor_preferred_lag:
            self._sensor_preferred_lag_confidence += 1
        else:
            self._sensor_preferred_lag = lag
            self._sensor_preferred_lag_confidence = 1

        self._sensor_lag_accept_count += 1
        # Initialization messages are always useful. During training, print the
        # first three and then only periodic reminders to avoid flooding stdout.
        should_log = (
            str(phase).startswith("initial")
            or self._sensor_lag_accept_count <= 3
            or self._sensor_lag_accept_count % max(
                1, int(self.config.get("sensor_lag_log_interval", 500))
            ) == 0
        )
        if should_log:
            print(
                "[sensor-lag] accepting coherent sensor bundle: "
                f"phase={phase} world_frame={status['target_frame']} "
                f"sensor_frame={status['bundle_frame']} lag={lag} "
                f"spread={status['bundle_spread']}"
            )

    def _sensor_lag_fast_path_ready(self, status: dict) -> bool:
        """Skip repeated grace waits after deterministic lag is established."""
        if not status.get("accepted", False) or status.get("exact", False):
            return False
        required = max(
            1, int(self.config.get("sensor_lag_fast_path_after", 2))
        )
        return bool(
            int(status.get("max_frame_lag", -1)) == self._sensor_preferred_lag
            and self._sensor_preferred_lag_confidence >= required
        )

    def _wait_for_sensor_bundle(
        self,
        target_frame: int,
        *,
        timeout: float,
        retries: int,
        phase: str,
        capture_observation: bool = False,
    ) -> dict:
        names = self._active_sensor_names()
        if not names:
            return {
                "accepted": True,
                "exact": True,
                "frames": {},
                "missing": [],
                "bundle_frame": int(target_frame),
                "newest_frame": int(target_frame),
                "bundle_spread": 0,
                "max_frame_lag": 0,
                "target_frame": int(target_frame),
                "observation": {} if capture_observation else None,
            }

        exact_grace = max(
            0.0,
            float(self.config.get("sensor_exact_frame_grace_seconds", 0.20)),
        )
        status = None
        for attempt in range(max(0, int(retries)) + 1):
            deadline = time.monotonic() + float(timeout)
            grace_deadline = min(deadline, time.monotonic() + exact_grace)
            with self._sensor_condition:
                while True:
                    callback_errors = {
                        name: self._sensor_callback_errors[name]
                        for name in names
                        if name in self._sensor_callback_errors
                    }
                    if callback_errors:
                        raise SensorTimeoutError(
                            list(callback_errors), target_frame, timeout, retries,
                            phase=f"{phase}_callback_error",
                            details={"callback_errors": callback_errors},
                        )
                    status = self._sensor_bundle_snapshot_locked(names, target_frame)
                    now = time.monotonic()
                    grace_finished = now >= grace_deadline

                    lag_fast_path = self._sensor_lag_fast_path_ready(status)
                    if status["exact"] or (
                        status["accepted"] and (grace_finished or lag_fast_path)
                    ):
                        if capture_observation:
                            status["observation"] = self._capture_observation_locked()
                        self._log_accepted_sensor_lag(phase, status)
                        return status

                    remaining = min(deadline, grace_deadline) - now
                    if grace_finished:
                        remaining = deadline - now
                    if remaining <= 0.0:
                        break
                    self._sensor_condition.wait(timeout=remaining)

            if attempt < retries:
                print(
                    f"[sensor-warning] Missing {status['missing']} for world frame "
                    f"{target_frame} after {timeout:.1f}s; "
                    f"frames={status['frames']} spread={status['bundle_spread']}; "
                    f"waiting longer without another world tick "
                    f"({attempt + 1}/{retries})."
                )
        return status

    def _wait_initial_sensors(self):
        timeout = float(self.config.get("sensor_initial_timeout_seconds", 15.0))
        retries = max(0, int(self.config.get("sensor_initial_timeout_retries", 1)))
        warmup_frames = max(1, int(self.config.get("sensor_warmup_frames", 3)))
        for index in range(warmup_frames):
            target_frame = int(self.world.tick())
            status = self._wait_for_sensor_bundle(
                target_frame,
                timeout=timeout,
                retries=retries,
                phase=f"initial_warmup_{index + 1}",
            )
            if not status["accepted"]:
                raise SensorTimeoutError(
                    status["missing"], target_frame, timeout, retries,
                    phase="initialization",
                    details={
                        "sensor_frames": status["frames"],
                        "sensor_bundle_frame": status["bundle_frame"],
                        "sensor_bundle_spread": status["bundle_spread"],
                        "sensor_max_frame_lag": status["max_frame_lag"],
                    },
                )

    def _destroy_sensor_actors(self) -> None:
        # Invalidate queued callbacks before stopping or destroying any actor.
        self._sensor_generation += 1
        sensors = list(self.sensor_actors)
        self.sensor_actors = []
        for sensor in sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
            except Exception:
                pass
        ids = []
        for sensor in sensors:
            try:
                ids.append(int(sensor.id))
            except Exception:
                pass
        if ids:
            try:
                self.client.apply_batch_sync(
                    [carla.command.DestroyActor(actor_id) for actor_id in ids],
                    True,
                )
            except Exception as exc:
                print(f"[sensor-restart] sensor destroy failed: {exc}")
        id_set = set(ids)
        self.actor_list = [
            actor for actor in self.actor_list
            if int(getattr(actor, "id", -1)) not in id_set
        ]

    def restart_sensors(self) -> None:
        """Rebuild the sensor stack without changing the ego or traffic actors."""
        print("[sensor-restart] rebuilding ego sensor stack...")
        self._destroy_sensor_actors()
        self._reset_sensor_tracking(increment_generation=False)
        self._spawn_sensors()
        self._wait_initial_sensors()
        print("[sensor-restart] sensor stack is healthy again.")

    def _crop(self, image):
        if self.use_3_quarter:
            image = image[image.shape[0] // 4:, :]
        return np.ascontiguousarray(image)

    def _process_rgb(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
        return self._crop(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))

    def _process_seg(self, image):
        image.convert(ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
        return self._crop(array)

    def _process_game(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
        return self._crop(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))

    def _process_gps(self, event):
        return {
            "lat": event.latitude,
            "lon": event.longitude,
            "alt": event.altitude,
        }

    def _on_collision(self, event):
        other_actor = getattr(event, "other_actor", None)
        impulse = getattr(event, "normal_impulse", None)
        impulse_magnitude = 0.0
        if impulse is not None:
            impulse_magnitude = math.sqrt(
                float(impulse.x) ** 2
                + float(impulse.y) ** 2
                + float(impulse.z) ** 2
            )
        with self._collision_lock:
            self.collision_occurred = True
            self.collision_event_info = {
                "collision_other_actor_id": int(getattr(other_actor, "id", -1)),
                "collision_other_actor_type": str(
                    getattr(other_actor, "type_id", "")
                ),
                "collision_impulse": float(impulse_magnitude),
                "collision_frame": int(getattr(event, "frame", -1)),
            }

    def _on_lane_invasion(self, _event):
        with self._lane_lock: self.lane_invasion_occurred = True

    def step(self, action):
        control = carla.VehicleControl(
            throttle=float(action[0]), steer=float(action[1]), brake=float(action[2])
        )
        self.vehicle.apply_control(control)
        target_frame = int(self.world.tick())

        timeout = float(self.config.get("sensor_frame_timeout_seconds", 8.0))
        retries = max(0, int(self.config.get("sensor_timeout_retries", 1)))
        bundle_status = self._wait_for_sensor_bundle(
            target_frame,
            timeout=timeout,
            retries=retries,
            phase="step",
            capture_observation=True,
        )

        if not bundle_status["accepted"]:
            raise SensorTimeoutError(
                bundle_status["missing"], target_frame, timeout, retries,
                phase="step",
                details={
                    "sensor_frames": bundle_status["frames"],
                    "sensor_bundle_frame": bundle_status["bundle_frame"],
                    "sensor_bundle_spread": bundle_status["bundle_spread"],
                    "sensor_max_frame_lag": bundle_status["max_frame_lag"],
                    "stale_streak": self._sensor_stale_streak,
                },
            )

        max_lag = int(bundle_status["max_frame_lag"])
        stale_used = max_lag > 0
        if stale_used:
            self._sensor_stale_streak += 1
        else:
            self._sensor_stale_streak = 0

        self._last_sensor_health = {
            "sensor_data_stale": bool(stale_used),
            "sensor_stale_names": (
                ",".join(self._active_sensor_names()) if stale_used else ""
            ),
            "sensor_max_frame_lag": int(max_lag),
            "sensor_stale_streak": int(self._sensor_stale_streak),
            "sensor_world_frame": int(target_frame),
            "sensor_bundle_frame": int(bundle_status["bundle_frame"]),
            "sensor_bundle_spread": int(bundle_status["bundle_spread"]),
        }

        obs = bundle_status["observation"]
        obs["speed"] = self.get_speed()

        with self._collision_lock:
            collision = self.collision_occurred
            collision_event_info = dict(self.collision_event_info)
        with self._lane_lock: lane = self.lane_invasion_occurred
        angle, lateral, waypoint, on_driving_lane = self.get_angle_and_dist()
        traffic_context = self.get_traffic_context(
            waypoint,
            float(self.config.get("traffic_context_max_distance_m", 80.0)),
        )
        info = {
            "collision": collision, "lane_invasion": lane, "angle": angle,
            "lateral_dist": lateral, "speed": obs["speed"], "road_id": waypoint.road_id,
            "lane_id": waypoint.lane_id, "is_junction": waypoint.is_junction,
            "lane_width": float(waypoint.lane_width),
            "on_driving_lane": bool(on_driving_lane),
            **collision_event_info,
            **traffic_context,
            **self.get_dynamics(waypoint),
            **self._last_sensor_health,
        }
        done = collision or lane
        return obs, None, done, info

    def _clear_actors(self):
        # Reject callbacks already queued by the outgoing episode before actor
        # destruction starts. This is essential when resets happen rapidly.
        self._sensor_generation += 1
        sensors = list(getattr(self, "sensor_actors", []))
        self.sensor_actors = []
        for sensor in sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
            except Exception:
                pass

        actor_by_id = {}
        for actor in self.actor_list:
            try:
                actor_by_id[int(actor.id)] = actor
            except Exception:
                continue
        actor_ids = set(actor_by_id) | set(getattr(self, "traffic_actor_ids", set()))

        if actor_ids:
            try:
                commands = [carla.command.DestroyActor(actor_id) for actor_id in actor_ids]
                self.client.apply_batch_sync(commands, True)
            except Exception as batch_error:
                error_text = str(batch_error).lower()
                if "time-out" in error_text or "timeout" in error_text:
                    print(
                        "[cleanup] CARLA server is unreachable; skipping "
                        "sequential actor destruction. Restart CARLA before resume."
                    )
                else:
                    print(
                        "[cleanup] batch destroy failed; using sequential "
                        f"fallback: {batch_error}"
                    )
                    for actor in reversed(list(actor_by_id.values())):
                        try:
                            if actor.is_alive:
                                actor.destroy()
                        except Exception:
                            pass

        self.actor_list = []
        self.traffic_actors = []
        self.traffic_actor_ids = set()
        self.traffic_profile_by_actor_id = {}
        self.traffic_desired_speed_by_actor_id = {}
        self.traffic_anchor_by_actor_id = {}

    def close(self):
        if self._closed: return
        self._closed = True
        self._clear_actors()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

    def __del__(self):
        if not getattr(self, "_closed", True):
            try: self.close()
            except Exception: pass
