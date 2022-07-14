from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np 

from simbed.common.buffer.reverb import make_reverb_buffer
from simbed.common.buffer.adder_iterator import make_adder, ExhaustiveIteratorFromAdder
from simbed.actors.feedforward_actor import FeedforwardActor
from .observation_ref_source import ObservationReferenceSource
from simbed.common.utils import generate_ts
from acme.environment_loop import EnvironmentLoop


default_kernel = 0.2*RBF(1.5)
def draw_u_from_gaussian_process(ts, kernel=default_kernel, seed=1):
    ts = ts[:,np.newaxis]
    us = GaussianProcessRegressor(kernel=kernel).sample_y(ts, random_state=seed)
    us = us - np.mean(us)

    ramp_up = np.ones_like(ts)
    ramp_up[:100] = np.linspace(0,1,num=100)[:,None]
    # ramp down
    ramp_up[-100:] = np.linspace(1,0,num=100)[:,None]
    us = us*ramp_up
    
    return us.astype(np.float32) 


default_ts = np.asarray(generate_ts(5,0.01))
def make_obs_ref_source(env, ts = default_ts, seeds =[0], device=None):

    assert len(seeds)>0

    buffer = make_reverb_buffer(env, "ref_buffer", ts)
    adder = make_adder(buffer.client, buffer.table.name, ts)
    iterator = ExhaustiveIteratorFromAdder(buffer.client, buffer.table, device)

    # to make the linter happy
    sample = None 
    for seed in seeds:
        us = draw_u_from_gaussian_process(ts, seed=seed)
        actor = FeedforwardActor(us, action_spec=env.action_spec(), adder=adder)
        loop = EnvironmentLoop(env, actor)
        loop.run_episode()
        sample = next(iterator)

    if sample is None:
        raise Exception()
    
    obs_ref_source = ObservationReferenceSource(
        sample.data.observation, sample.data.action
    )

    return obs_ref_source
