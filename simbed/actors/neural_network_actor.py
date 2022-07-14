import dm_env 
import jax 
from acme import core 
from acme.adders import reverb as adders 
from acme.jax import utils
from typing import Optional

from simbed.common.observation_ref_source import ObservationReferenceSource
from simbed.common.utils import pytree_like
from simbed.common.types import nn, policy, actor
from .random_actor import RandomlyActingActor

    
class NeuralNetworkActor(RandomlyActingActor):
    def __init__(self, 
        policy: policy.Policy, 
        source: core.VariableSource, 
        ref_obs_source: Optional[ObservationReferenceSource] = None,
        **kwargs
    ):
        self._source = source
        super().__init__(**kwargs)

        self._apply = jax.jit(policy.apply)
        self._preprocess = policy.preprocess
        self.ref_obs_source = ref_obs_source

    @property
    def _params(self) -> nn.Params:
        return self._source.get_variables([])[0]

    def _initial_actor_state(self, key) -> actor.ActorStatefull:
        nn_state: nn.NeuralNetworkState = self._source.get_variables([])[1]
        return actor.ActorStatefull(key, 0, nn_state)

    def select_action(self, observation: nn.Observation) -> nn.Action:
        if self.ref_obs_source:
            observation_ref: nn.Observations = self.ref_obs_source.get_reference_actor()
        else:
            observation_ref = pytree_like(observation)

        inp = self._preprocess(observation, observation_ref, self._state.count)
        nn_state = self._state.state
        nn_state, action = self._apply(self._params, nn_state, inp)

        self._state = actor.ActorStatefull(self._state.key, self._state.count+1, nn_state)
        return action 
