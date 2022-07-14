import dm_env 
import jax 
from acme import core 
from acme.adders import reverb as adders 
from typing import Optional

from simbed.common.types import nn, actor

def sample_action_from_action_spec(key, action_spec):
    return jax.random.uniform(key, 
        shape=action_spec.shape,
        minval=action_spec.minimum,
        maxval=action_spec.maximum,
        )


class RandomlyActingActor(core.Actor):
    def __init__(self, 
        action_spec,
        ts = None,
        key = None, 
        adder: Optional[adders.ReverbAdder] = None,
        reset_key=False,
    ):

        self.action_spec = action_spec
        self._adder = adder 
        if ts is not None:
            self.max_number_timesteps = len(ts)-1
        else:
            self.max_number_timesteps = None 

        if key is None:
            key = jax.random.PRNGKey(1)
        self._key = key
        self.reset_key=True 
        self.reset()
        self.reset_key=reset_key

    def actor_as_launch_policy_cleanup(self):
        if self.max_number_timesteps is None:
            raise ValueError("Must specifiy `ts` for this method first")

        # resets the actor not just at the last frame but the last 10 frames
        frames_too_early=10 # ideally this should be zero
        if self._state.count >= self.max_number_timesteps-frames_too_early:
            self.reset()

    def observe_first(self, timestep: dm_env.TimeStep):
        self.reset()
        if self._adder:
            self._adder.add_first(timestep)

    def reset(self):
        if self.reset_key:
            key = self._key
        else:
            key = self._state.key 

        self._state = self._initial_actor_state(key)

    def _initial_actor_state(self, key) -> actor.ActorStateless:
        return actor.ActorStateless(key, 0)
    
    def observe(self, action, next_timestep):
        if self._adder:
            self._adder.add(action, next_timestep=next_timestep)

    def update(self, wait: bool = False):
        pass 

    def select_action(self, observation: nn.Observation) -> nn.Action:
        del observation
        key, consume = jax.random.split(self._state.key)
        action = sample_action_from_action_spec(consume, self.action_spec)
        self._state = actor.ActorStateless(key, self._state.count+1)
        return action
        