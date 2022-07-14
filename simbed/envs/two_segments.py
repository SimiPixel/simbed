from dm_control.rl import control
from dm_control import mujoco
from collections import OrderedDict
import numpy as np 
from acme.wrappers import SinglePrecisionWrapper
from simbed.common.observation_ref_source import ObservationReferenceSource
from typing import Optional
from simbed.common.utils import idx_in_pytree
from acme.jax import utils


class SegmentPhysics(mujoco.Physics):

    def xpos_of_segment_end(self):
        return self.named.data.xpos["segment_end", "x"]

    def set_torque_of_cart(self, u):
        u = np.arctan(u)
        self.set_control(u)


class SegmentTask(control.Task):

    def __init__(self, obs_ref_source: Optional[ObservationReferenceSource] = None):
        super().__init__()
        self._obs_ref_source = obs_ref_source
        
    def initialize_episode(self, physics):
        self.timestep = 0 #TODO This might have to start from 0. Reward at t=0 is `None`

    def before_step(self, action, physics: SegmentPhysics):
        self.timestep += 1
        physics.set_torque_of_cart(action)

    def after_step(self, physics):
        pass 

    def action_spec(self, physics):
        return mujoco.action_spec(physics)

    def get_observation(self, physics):
        obs = OrderedDict()
        obs["xpos_of_segment_end"] = np.atleast_1d(physics.xpos_of_segment_end())
        return obs

    @staticmethod
    def reward_fn(obs, obs_ref):
        obs = utils.batch_concat(obs)
        obs_ref = utils.batch_concat(obs_ref)

        return -np.mean((obs_ref - obs)**2)

    def get_reward(self, physics):
        """Returns absolute deviation/error

        Args:
            physics (_type_): _description_

        Returns:
            float: reward
        """
        obs = self.get_observation(physics)
        if self._obs_ref_source:
            obs_ref = idx_in_pytree(self._obs_ref_source.get_reference_actor(), self.timestep)
        else:
            obs_ref = obs 

        return self.reward_fn(obs, obs_ref).item()

from simbed.envs import common
from simbed.common.utils import generate_ts
def load_segment_env(obs_ref_source=None, time_limit=5, control_timestep=0.01):
    physics = SegmentPhysics.from_xml_string(common.read_model("two_segments.xml"), assets=common.ASSETS)
    ts = generate_ts(time_limit,control_timestep)

    env = control.Environment(
        physics, 
        SegmentTask(obs_ref_source), 
        time_limit=time_limit, 
        control_timestep=control_timestep
    )
    env = SinglePrecisionWrapper(env) 
    return env 
