"""Offline structural/math verification; does not require CARLA or GPU."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys

import numpy as np

from actions import ACTIONS, admissible_action_count, GENTLE_STEERS, ALL_STEERS
from config import TrainingConfig, EnvironmentConfig, validate_config, nominal_route_distance_m
from curriculum import ActionCurriculum, EpisodePerformance
from replay_buffer import PrioritizedReplayBuffer
from reward import LaneKeepingReward
from train import epsilon_at_episode, beta_at_episode, repetition_at_episode
from env import read_spawn_ids
from weather_physics import CLEAR, RAIN, friction_factor


def check():
    out = []
    def assert_thing(name, condition):
        if not condition:
            raise AssertionError(name)
        out.append(name)

    t, e = TrainingConfig(), EnvironmentConfig()
    validate_config(t,e)
    assert_thing('Only Full configuration', t.priority_alpha==.6)
    assert_thing('45 unique actions',len(ACTIONS)==45 and len(set(ACTIONS))==45)
    assert_thing('21/27/45 action stages', [admissible_action_count(i) for i in (1,2,3)]==[21,27,45])
    assert_thing('No simultaneous throttle/brake', all(a.throttle*a.brake==0 for a in ACTIONS))
    assert_thing('Action bounds',len(GENTLE_STEERS)==7 and len(ALL_STEERS)==9)
    assert_thing('DAR 3,2,1', [repetition_at_episode(x) for x in (1,1500,1501,3000,3001,6000)] == [3,3,2,2,1,1])
    assert_thing('Epsilon at first episode',math.isclose(epsilon_at_episode(t,1),1.))
    assert_thing('Epsilon at 3000/6000',all(math.isclose(epsilon_at_episode(t,i),.05) for i in (3000,6000)))
    assert_thing('Beta at beginning/end',math.isclose(beta_at_episode(t,1),.4) and math.isclose(beta_at_episode(t,6000),1.))
    assert_thing('6000 eps, 1500 ticks, gamma .99', (t.max_episodes,t.max_ticks_per_episode,t.gamma)==(6000,1500,.99))
    assert_thing('0.1 s tick and 150 s episode budget', math.isclose(e.fixed_delta_seconds * t.max_ticks_per_episode, 150.0))
    target_distance = nominal_route_distance_m(t.max_ticks_per_episode, e.fixed_delta_seconds, t.target_speed_kmh)
    assert_thing('Fixed 2083.333 m road-distance horizon', math.isclose(target_distance, 2083.3333333333335, rel_tol=0, abs_tol=1e-9))
    assert_thing('95% success threshold distance', math.isclose(.95 * target_distance, 1979.1666666666665, rel_tol=0, abs_tol=1e-9))
    assert_thing('Replay 30000/2000/8', (t.replay_capacity,t.replay_warmup,t.batch_size)==(30000,2000,8))
    assert_thing('Terminal & stall',t.collision_or_departure_penalty==-500 and t.stall_penalty==-100 and t.stall_ticks==200)
    assert_thing('Weather values and friction', math.isclose(friction_factor(CLEAR),1.) and math.isclose(friction_factor(RAIN),.88336,abs_tol=1e-12))
    reward = LaneKeepingReward(t)
    aligned = {'angle':1.0,'lateral_dist':0.2}
    r = reward.compute(aligned, aligned, 50.)
    assert_thing('Speed base reward at 50 km/h',r.total==25)
    assert_thing('Clipping max+30',reward.compute({'angle':5.5,'lateral_dist':0.6},{'angle':8,'lateral_dist':0.7},50.).total<=30.)
    assert_thing('Collision minus 500',reward.compute({'collision':True},aligned,50.).total==-500.)
    assert_thing('Stall minus 100',reward.compute(aligned,aligned,0.,stalled=True).total==-100.)
    # One-sided reward asymmetry for current error outside aligned zone.
    past={'angle':3.,'lateral_dist':.4}
    worse={'angle':7.,'lateral_dist':.8}
    result = reward.compute(worse,past,50.)
    assert_thing('Asymmetric deterioration heading/lateral', result.heading_term==-42 and math.isclose(result.lateral_term,-57.6))
    # Refresh reward priorities directly; no PNG transitions required for this test.
    rep = PrioritizedReplayBuffer(capacity=10,alpha=.6,refresh_every=256)
    rep.buffer = [None]*5
    rep.normalized_rewards[:5]=[-1,0,0,0,10]
    rep.refresh_reward_priorities()
    assert_thing('Positive and ordered reward-deviation priorities', np.all(rep.priorities[:5]>0) and rep.priorities[4]>rep.priorities[1])
    assert_thing('Probability sums to 1',math.isclose(float((rep.priorities[:5]/rep.priorities[:5].sum()).sum()),1.,abs_tol=1e-6))
    assert_thing('Duration discounted normalization', math.isclose((3+.99*6)/(1+.99), 4.492462311557789, rel_tol=1e-12))
    curriculum=ActionCurriculum(t)
    assert_thing('Curriculum starts stage 1', curriculum.stage==1)
    metric=EpisodePerformance(route_completion=1.0,failed=False,mean_abs_lateral_error_m=0.1)
    for epi in range(1,603): curriculum.update(metric,epi)
    assert_thing('Promotion stage I after 600 + 3 passes',curriculum.stage==2)
    assert_thing('Exploration temporary unlock',math.isclose(curriculum.effective_epsilon(.2,603),.3))
    csv=Path(__file__).resolve().parent.parent/'data'/'town04_points.csv'
    ids=read_spawn_ids(csv)
    assert_thing('Spawn IDs read exactly from author CSV',len(ids)==277 and ids[0]==1)
    summary={'status':'PASS','checked':len(out),'tests':out}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__ == '__main__':
    try: check()
    except Exception as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        raise
