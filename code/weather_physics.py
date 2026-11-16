"""Explicit CARLA Clear/Rain weather and wheel friction for NEW experiments.

Rain parameters and the friction formula were supplied by the study author.
This module applies them to the vehicle physics, not only to scene visuals.
CARLA WeatherParameters alone does not alter vehicle tire friction.
This module alters wheel physics independently via apply_physics_control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WeatherSpec:
    precipitation: float
    wetness: float
    precipitation_deposits: float
    alpha: float = 0.18


CLEAR = WeatherSpec(0.0, 0.0, 0.0)
RAIN = WeatherSpec(50.0, 70.0, 20.0)


def friction_factor(spec: WeatherSpec) -> float:
    x = (0.8 * spec.wetness + 0.2 * spec.precipitation_deposits) / 100.0
    x = max(0.0, min(1.0, x))
    smooth = 3.0 * x ** 2 - 2.0 * x ** 3
    return 1.0 - spec.alpha * smooth


def find_world(env: Any) -> Any:
    for name in ("world", "_world", "carla_world"):
        item = getattr(env, name, None)
        if item is not None:
            return item
    client = getattr(env, "client", None)
    if client is not None and hasattr(client, "get_world"):
        return client.get_world()
    raise RuntimeError("Weather setup requires env.world or env.client.get_world().")


def find_vehicle(env: Any) -> Any:
    for name in ("vehicle", "ego_vehicle", "ego_car", "carla_vehicle", "player", "_vehicle"):
        item = getattr(env, name, None)
        if item is not None and hasattr(item, "get_physics_control"):
            return item
    raise RuntimeError("Cannot find ego vehicle: expose env.vehicle or env.ego_vehicle.")


def set_explicit_weather(world: Any, spec: WeatherSpec) -> None:
    # WeatherParameters modifies visuals, not wheel physics; apply both
    # explicitly. Start each condition from the same CARLA ClearNoon baseline
    # to prevent residual fog/cloud/sun settings from a previous experiment.
    weather = world.get_weather()
    # Do not mutate the CARLA class-level ClearNoon preset (it is shared).
    reference = getattr(type(weather), 'ClearNoon', None)
    fixed = ('cloudiness', 'precipitation', 'precipitation_deposits',
             'wind_intensity', 'sun_azimuth_angle', 'sun_altitude_angle',
             'fog_density', 'fog_distance', 'fog_falloff', 'wetness',
             'scattering_intensity', 'mie_scattering_scale',
             'rayleigh_scattering_scale', 'dust_storm')
    if reference is not None:
        for key in fixed:
            if hasattr(weather, key) and hasattr(reference, key):
                setattr(weather, key, getattr(reference, key))
    else:  # lightweight test double or compatible CARLA API
        for key, val in [('cloudiness', 0.), ('fog_density', 0.),
                         ('fog_distance', 0.), ('fog_falloff', 0.),
                         ('wind_intensity', 0.)]:
            if hasattr(weather, key):
                setattr(weather, key, val)
    weather.precipitation = float(spec.precipitation)
    weather.wetness = float(spec.wetness)
    weather.precipitation_deposits = float(spec.precipitation_deposits)
    world.set_weather(weather)


class WeatherPhysicsController:
    """Apply Clear/Rain once per reset; retain original friction for reused actors.

    A different CARLA actor ID is treated as a new vehicle with its own dry
    friction baseline. `dry_friction` can be explicitly passed when necessary.
    """

    def __init__(self) -> None:
        self._baselines: dict[tuple[int, int | None], tuple[float, ...]] = {}
        self.last_applied: dict[str, Any] | None = None

    @staticmethod
    def _wheel_attr(wheels: list) -> str:
        if not wheels:
            raise RuntimeError("Vehicle has no physics-control wheels.")
        for field in ("tire_friction", "friction_force_multiplier"):
            if hasattr(wheels[0], field):
                return field
        raise RuntimeError("Wheel has neither tire_friction nor friction_force_multiplier.")

    def apply(self, env: Any, condition: str, *, dry_friction: list[float] | None = None) -> dict:
        if condition not in ("Clear", "Rain"):
            raise ValueError("condition must be 'Clear' or 'Rain'.")
        spec = CLEAR if condition == "Clear" else RAIN
        world = find_world(env)
        vehicle = find_vehicle(env)
        physics = vehicle.get_physics_control()
        wheels = list(physics.wheels)
        attr = self._wheel_attr(wheels)
        key = (id(vehicle), getattr(vehicle, "id", None))
        if dry_friction is not None:
            baseline = tuple(map(float, dry_friction))
            self._baselines[key] = baseline
        elif key in self._baselines:
            baseline = self._baselines[key]
        else:
            baseline = tuple(float(getattr(w, attr)) for w in wheels)
            self._baselines[key] = baseline
        if len(baseline) != len(wheels):
            raise RuntimeError("Original friction list does not match wheel count.")
        factor = friction_factor(spec)
        for wheel, original in zip(wheels, baseline):
            setattr(wheel, attr, original * factor)
        physics.wheels = wheels
        # Apply physics before the next simulator tick.
        vehicle.apply_physics_control(physics)
        set_explicit_weather(world, spec)
        record = {
            "condition": condition,
            "precipitation": spec.precipitation,
            "wetness": spec.wetness,
            "deposits": spec.precipitation_deposits,
            "friction_factor": factor,
            "wheel_attribute": attr,
            "dry_wheel_friction": list(baseline),
            "applied_wheel_friction": [float(getattr(w, attr)) for w in wheels],
            "vehicle_actor_id": getattr(vehicle, "id", None),
        }
        self.last_applied = record
        return record


def apply_rain_friction(world: Any, vehicle: Any, precipitation: float = 50.0,
                        wetness: float = 70.0, deposits: float = 20.0,
                        alpha: float = 0.18,
                        dry_friction: list[float] | None = None) -> tuple[list[float], float]:
    """Standalone equivalent of author's function; returns dry values and factor.

    Calling this twice WITHOUT passing the original dry_friction on a reused
    vehicle would compound the reduction. For loops use WeatherPhysicsController.
    """
    spec = WeatherSpec(precipitation, wetness, deposits, alpha)
    physics = vehicle.get_physics_control()
    wheels = list(physics.wheels)
    attr = WeatherPhysicsController._wheel_attr(wheels)
    dry = list(dry_friction) if dry_friction is not None else [float(getattr(w, attr)) for w in wheels]
    factor = friction_factor(spec)
    for wheel, original in zip(wheels, dry):
        setattr(wheel, attr, original * factor)
    physics.wheels = wheels
    vehicle.apply_physics_control(physics)
    set_explicit_weather(world, spec)
    return dry, factor
