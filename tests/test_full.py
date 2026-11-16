from pathlib import Path
import numpy as np
import pytest
import torch


def test_offline_validation():
    from verify import check
    assert check()['checked'] >= 25


def test_model_forward():
    from model import DQN
    model = DQN(45).eval()
    x = torch.zeros((1, 6, 64, 64), dtype=torch.uint8)
    with torch.inference_mode():
        y = model(x, torch.tensor([50/110]))
    assert y.shape == (1,45)
    assert torch.isfinite(y).all()


def test_replay_duration_and_batch():
    from replay_buffer import PrioritizedReplayBuffer
    rb=PrioritizedReplayBuffer(capacity=24,alpha=.6,refresh_every=4)
    state=np.zeros((6,16,16),dtype=np.uint8)
    for i in range(12):
        nxt = (state+i)%255
        rb.push(state,0.45,i%21,3+0.99*6,nxt,0.46,False,2,21,.99)
    assert len(rb)==12
    assert np.isfinite(rb.priorities[:12]).all()
    rb.set_beta(.7)
    batch,indices,weights=rb.sample(8,torch.device('cpu'))
    assert len(set(map(int,indices)))==8
    assert batch['states'].shape==(8,6,16,16)
    assert torch.all(batch['durations']==2)
    assert torch.isfinite(weights).all() and torch.max(weights) <= 1.000001
    assert np.allclose(rb.normalized_rewards[:12],(3+.99*6)/(1+.99))


class FakeLaneEnv:
    def __init__(self, cfg):
        self.cfg=cfg
        self.allowed_starts=[1,2]
        self.map=type('Map',(),{'name':'Town04'})()
        self.rng=np.random.default_rng(1)
        self.n=0
    def reset(self, start_speed=10, condition='Clear'):
        self.n=0
        return self._observation(),self._info()
    def step(self, action):
        self.n+=1
        return self._observation(),0.0,self.n>=5,self._info()
    def _observation(self):
        return {'segmentation':np.zeros((16,20,3),dtype=np.uint8),'speed':50.}
    def _info(self):
        return {'angle':0.,'lateral_dist':0.1,'route_completion':min(self.n/5,1.),
                'collision':False,'lane_departure':False, 'route_id':'fake',
                'route_length_m':2083.3333333333335,'route_progress_m':min(self.n*20.,2083.3333333333335),
                'start_spawn_index':1}
    def get_rng_state(self):
        return self.rng.bit_generator.state
    def close(self):
        pass


def test_one_episode_without_carla(tmp_path,monkeypatch):
    import train
    from config import TrainingConfig, EnvironmentConfig
    monkeypatch.setattr(train,'CarlaLaneEnv',FakeLaneEnv)
    cfg=TrainingConfig(device='cpu',max_episodes=1,output_root=tmp_path)
    trainer=train.FullTrainer(cfg,EnvironmentConfig(spawn_csv=Path(__file__).resolve().parent.parent/'data'/'town04_points.csv'))
    try:
        row=trainer.run_episode(1)
        assert row['success']==1
        assert row['episode_steps']==5
        assert row['policy_decisions']==2  # k=3 then two more ticks
        assert len(trainer.replay)==2
        assert row['route_id']=='fake'
        assert row['route_progress_m'] > 0
    finally:
        trainer.close()


def test_chunked_replay_checkpoint(tmp_path):
    from replay_buffer import PrioritizedReplayBuffer
    path = tmp_path/'replay.pkl'
    rb = PrioritizedReplayBuffer(capacity=12,alpha=.6,refresh_every=4)
    st=np.zeros((6,16,16),dtype=np.uint8)
    for i in range(9):
        rb.push(st,0.2,i%7,float(i),st,0.2,False,1,21,.99)
    rb.save(path,episode=55)
    loaded = PrioritizedReplayBuffer.load(path)
    assert loaded.saved_episode==55
    assert len(loaded)==9
    assert np.allclose(loaded.priorities,rb.priorities)
    assert np.allclose(loaded.normalized_rewards,rb.normalized_rewards)


def test_real_weighted_ddqn_update_on_cpu(tmp_path,monkeypatch):
    import train
    from config import TrainingConfig, EnvironmentConfig
    monkeypatch.setattr(train,'CarlaLaneEnv', FakeLaneEnv)
    cfg=TrainingConfig(device='cpu',output_root=tmp_path)
    tr=train.FullTrainer(cfg,EnvironmentConfig(spawn_csv=Path(__file__).resolve().parent.parent/'data'/'town04_points.csv'))
    try:
        st=np.zeros((6,24,32),dtype=np.uint8)
        for i in range(8):
            tr.replay.push(st,0.45,i,1.5,st,0.45,False,2,21,cfg.gamma)
        loss=tr.learn_step(episode=1)
        assert np.isfinite(loss)
        assert loss>=0
    finally:
        tr.close()


def test_rain_physics_restore_on_same_vehicle():
    from types import SimpleNamespace
    from weather_physics import WeatherPhysicsController
    wheels=[SimpleNamespace(tire_friction=2.0) for _ in range(4)]
    physics=SimpleNamespace(wheels=wheels)
    class Vehicle:
        id=21
        def get_physics_control(self): return physics
        def apply_physics_control(self,updated): pass
    class World:
        def __init__(self): self.weather=SimpleNamespace(precipitation=0.,wetness=0.,precipitation_deposits=0.)
        def get_weather(self): return self.weather
        def set_weather(self,weather): self.weather=weather
    vehicle=Vehicle();world=World()
    controller=WeatherPhysicsController()
    env=SimpleNamespace(world=world,vehicle=vehicle)
    controller.apply(env,'Rain')
    assert all(abs(x.tire_friction-1.76672)<1e-8 for x in wheels)
    assert (world.weather.precipitation,world.weather.wetness,world.weather.precipitation_deposits)==(50.,70.,20.)
    controller.apply(env,'Clear')
    assert all(x.tire_friction==2. for x in wheels)
    assert (world.weather.precipitation,world.weather.wetness,world.weather.precipitation_deposits)==(0.,0.,0.)


