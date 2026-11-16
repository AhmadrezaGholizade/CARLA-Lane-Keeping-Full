"""Regression tests for issues that could invalidate a full 6000-episode run."""
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch


def test_spawn_ids_are_validated_against_carla_map():
    from env import validate_spawn_ids
    validate_spawn_ids([1, 3, 5], 7)
    with pytest.raises(ValueError, match='outside'):
        validate_spawn_ids([1, 99], 7)
    with pytest.raises(ValueError, match='no allowed'):
        validate_spawn_ids([], 7)


def test_fixed_distance_is_derived_from_ticks_dt_and_reference_speed():
    from config import nominal_route_distance_m
    distance = nominal_route_distance_m(1500, 0.1, 50.0)
    assert math.isclose(distance, 2083.3333333333335, abs_tol=1e-9)
    assert math.isclose(0.95 * distance, 1979.1666666666665, abs_tol=1e-9)


def test_progress_projection_counts_forward_not_lateral_or_reverse_motion():
    from env import projected_forward_increment
    fwd = (1.0, 0.0)
    assert math.isclose(projected_forward_increment((0, 0), (1.2, 0.8), fwd, fwd, 8.0), 1.2)
    assert projected_forward_increment((0, 0), (0.0, 1.5), fwd, fwd, 8.0) == 0.0
    assert projected_forward_increment((1, 0), (0, 0), fwd, fwd, 8.0) == 0.0
    assert projected_forward_increment((0, 0), (50, 0), fwd, fwd, 8.0) == 0.0


def test_progress_accumulates_continuously_around_a_loop():
    from env import projected_forward_increment
    angles = np.linspace(0.0, 2.0 * math.pi, 101)
    radius = 20.0
    points = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
    tangents = [(-math.sin(a), math.cos(a)) for a in angles]
    progress = 0.0
    for i in range(1, len(points)):
        progress += projected_forward_increment(points[i-1], points[i],
                                                tangents[i-1], tangents[i], 8.0)
    assert abs(progress - 2.0 * math.pi * radius) < 0.1



def test_validation_rejects_a_mismatched_fixed_distance():
    from dataclasses import replace
    from config import EnvironmentConfig, TrainingConfig, validate_config
    with pytest.raises(ValueError, match='Fixed route distance mismatch'):
        validate_config(TrainingConfig(), replace(EnvironmentConfig(), route_target_distance_m=2000.0))


def test_spawn_csv_actual_values():
    from env import read_spawn_ids
    p=Path(__file__).resolve().parent.parent/'data'/'town04_points.csv'
    ids=read_spawn_ids(p)
    assert len(ids)==277
    assert ids[0]==1 and ids[-1]==371
    assert len(set(ids))==277


def test_speed_only_division_not_clipped():
    from preprocessing import normalized_speed
    assert math.isclose(normalized_speed(55), .5)
    assert math.isclose(normalized_speed(220), 2)
    assert normalized_speed(231)>2  # no hidden clipping contrary to manuscript


def test_environment_road_metrics_initializes_and_accumulates_progress():
    from env import CarlaLaneEnv
    class DummyPoint:
        def __init__(self, x, y): self.x=x; self.y=y
    class DummyVehicle:
        def __init__(self): self.loc=DummyPoint(0, 0)
        def get_location(self): return self.loc
        def get_transform(self): return SimpleNamespace(rotation=SimpleNamespace(yaw=0.))
    class FakeMap:
        def get_waypoint(self, location, project_to_road=True, lane_type=None):
            return SimpleNamespace(transform=SimpleNamespace(
                location=DummyPoint(location.x, 0.), rotation=SimpleNamespace(yaw=0.),
                get_forward_vector=lambda:SimpleNamespace(x=1.,y=0.)))
    env=CarlaLaneEnv.__new__(CarlaLaneEnv)
    env.vehicle=DummyVehicle(); env.map=FakeMap()
    env.carla=SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    env.cfg=SimpleNamespace(max_progress_per_tick_m=8.0)
    env.progress_distance_m=0.0; env.route_length=2083.3333333333335; env.completion=0.0
    env._last_progress_xy=None; env._last_lane_forward_xy=None
    lat,hdg=env._road_metrics()
    assert env.progress_distance_m == 0.0 and env.completion == 0.0
    env.vehicle.loc=DummyPoint(5.0, 1.5)
    lat,hdg=env._road_metrics()
    assert math.isclose(env.progress_distance_m, 5.0)
    assert math.isclose(env.completion, 5.0/env.route_length)
    assert math.isclose(lat, 1.5) and hdg == 0.0


def test_replay_store_recovers_identical_semantic_channels(tmp_path):
    from replay_buffer import PrioritizedReplayBuffer
    rb=PrioritizedReplayBuffer(6, refresh_every=2)
    rng=np.random.default_rng(13)
    state=rng.integers(0,256,size=(6,20,26),dtype=np.uint8)
    nxt=rng.integers(0,256,size=(6,20,26),dtype=np.uint8)
    rb.push(state,.5,2,7.,nxt,.5,False,3,21,.99)
    rb.push(nxt,.6,3,3.,state,.6,False,1,21,.99)
    path=tmp_path/'replay.pkl';rb.save(path,episode=33)
    restored=PrioritizedReplayBuffer.load(path)
    from replay_buffer import _decode_stack
    assert np.array_equal(_decode_stack(restored.buffer[0].state_frames),state)
    assert np.array_equal(_decode_stack(restored.buffer[0].next_frames),nxt)
    assert restored.saved_episode==33


def test_alpha_zero_on_same_reward_per_path_is_uniform():
    from replay_buffer import PrioritizedReplayBuffer
    rb=PrioritizedReplayBuffer(6,alpha=0.,refresh_every=3,seed=9)
    rb.buffer=[None]*6
    rb.normalized_rewards[:6]=[-500,-2,-1,1,4,25]
    rb.refresh_reward_priorities()
    assert np.array_equal(rb.priorities[:6],np.ones(6))
    n=6; probs=rb.priorities[:6]/rb.priorities[:6].sum()
    assert np.allclose(probs,np.full(n,1/n))
    beta=.7;weights=(n*probs)**(-beta); weights/=weights.max()
    assert np.allclose(weights,1.)


def test_model_architecture_matches_paper_layers():
    from model import DQN
    m=DQN(45)
    convs=[x for x in m.visual_encoder if isinstance(x,torch.nn.Conv2d)]
    assert [(c.in_channels,c.out_channels,c.kernel_size,c.stride) for c in convs]==[
        (6,96,(5,5),(2,2)),(96,192,(5,5),(2,2)),(192,384,(3,3),(2,2)),
        (384,384,(3,3),(2,2)),(384,512,(3,3),(2,2)),
    ]
    assert m.head[-1].out_features==45


def test_curriculum_epsilon_equation_even_when_cap_lowers_base():
    from train import epsilon_at_episode
    from config import TrainingConfig
    from curriculum import ActionCurriculum
    cfg=TrainingConfig()
    cur=ActionCurriculum(cfg)
    cur.epsilon_boost_until_episode=700
    early=epsilon_at_episode(cfg,600)
    expected=min(early+.1,.35)
    assert math.isclose(cur.effective_epsilon(early,600),expected)
    # This is a mathematical property of the *article's formula*, not a bug
    # caused by the implementation. It can reduce epsilon at early promotion.
    assert expected<early


def test_reward_closed_form_and_terminal_bypass_clip():
    from config import TrainingConfig
    from reward import LaneKeepingReward
    cfg=TrainingConfig()
    fun=LaneKeepingReward(cfg)
    x={'angle':8.,'lateral_dist':.8}
    y={'angle':10.,'lateral_dist':1.}
    improved=fun.compute(x,y,50.)
    assert improved.heading_term == 0.6*8
    assert math.isclose(improved.lateral_term,7.2*.8)
    worse=fun.compute(y,x,50.)
    assert worse.heading_term == -6*10
    assert worse.lateral_term == -72*1
    assert fun.compute({'collision':True},x,50.).total == -500
    assert fun.compute(x,x,0.,stalled=True).total == -100


def test_lateral_error_comes_from_driving_lane_not_route_polyline():
    """Lateral error is measured from the current CARLA driving-lane centre."""
    from env import CarlaLaneEnv
    class Loc:
        def __init__(self,x,y):self.x=x;self.y=y
    class Vehicle:
        def get_location(self):return Loc(10.,1.)
        def get_transform(self):return SimpleNamespace(rotation=SimpleNamespace(yaw=0.))
    class Map:
        def get_waypoint(self,location,project_to_road=True,lane_type=None):
            return SimpleNamespace(transform=SimpleNamespace(location=Loc(10.,0.),
                 rotation=SimpleNamespace(yaw=0.),
                 get_forward_vector=lambda:SimpleNamespace(x=1.,y=0.)))
    obj=CarlaLaneEnv.__new__(CarlaLaneEnv)
    obj.vehicle=Vehicle();obj.map=Map()
    obj.carla=SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    obj.cfg=SimpleNamespace(max_progress_per_tick_m=8.0)
    obj.route_length=50.;obj.progress_distance_m=0.;obj.completion=0.
    obj._last_progress_xy=None;obj._last_lane_forward_xy=None
    lateral,hdg=obj._road_metrics()
    assert abs(lateral-1.0)<1e-9  # NOT distance 9m to route line
    assert hdg==0.
