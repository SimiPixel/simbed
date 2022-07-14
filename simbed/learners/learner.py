from acme import core 
from acme.jax import utils 
import reverb 
from abc import abstractstaticmethod, ABC, abstractmethod
import jax 

from flax import struct
from simbed.common.types import nn, policy
from simbed.common.observation_ref_source import ObservationReferenceSource
from typing import List, Iterator, Tuple 
from simbed.common.utils import idx_in_pytree


@struct.dataclass
class LearnerState:
    params_policy: nn.Params
    init_state_policy: nn.NeuralNetworkState
    key: nn.PRNGKey


class LearnerMeta(ABC):
    def __init__(self, 
        policy: policy.Policy,
        iterator: Iterator[reverb.ReplaySample],
        observation_ref_source: ObservationReferenceSource,
        key: nn.PRNGKey = jax.random.PRNGKey(1),
        ):
        self._iterator = iterator 
        self._policy = policy 
        self.observation_ref_source = observation_ref_source
        self._state = self.initial_state(key)
     
    def step(self, n_gradient_steps: int):

        # sample padded data 
        sample = next(self._iterator)

        # get references for optimisation
        obs_ref_opt = self.observation_ref_source.get_references_for_optimisation()
        obs_ref_opt = utils.batch_concat(obs_ref_opt, num_batch_dims=2)

        # get reference of actor
        obs_ref_actor = self.observation_ref_source.get_reference_actor()
        obs_ref_actor = utils.batch_concat(obs_ref_actor, num_batch_dims=1)

        # use data to update params of policy
        for _ in range(n_gradient_steps):
            self._state, loss_value = self.learner_params_update(
                sample, obs_ref_opt, obs_ref_actor, self._state, self._policy
            )

    def get_variables(self, names: List[str]) -> List[nn.Params]:
        return [self._state.params_policy, self._state.init_state_policy]

    def save(self) -> LearnerState:
        return self._state 

    def restore(self, state: LearnerState):
        self._state = state 

    @abstractmethod
    def initial_state(self, key) -> LearnerState:
        pass 

    @abstractstaticmethod
    def learner_params_update(
        sample: reverb.ReplaySample,
        obs_ref_opt: nn.Observationss,
        obs_ref_actor: nn.Observations,
        learner_state: LearnerState,
        policy: policy.Policy
    ) -> Tuple[LearnerState, dict[str, float]]: 
        pass 

class NoLearningLearner(LearnerMeta):

    def initial_state(self, key) -> LearnerState:
        obs_ref_actor = self.observation_ref_source.get_reference_actor()
        dummy_obs = idx_in_pytree(obs_ref_actor, start=0)
        dummy_policy_input = self._policy.preprocess(dummy_obs, obs_ref_actor, 0)
        key, consume = jax.random.split(key)
        params, state = self._policy.init(consume, dummy_policy_input)
        return LearnerState(params, state, key)

    @staticmethod
    def learner_params_update(
        sample: reverb.ReplaySample,
        obs_ref_opt: nn.Observationss,
        obs_ref_actor: nn.Observations,
        learner_state: LearnerState,
        policy: policy.Policy
    ) -> Tuple[LearnerState, dict[str, float]]: 

        new_params = learner_state.params_policy
        learner_state = learner_state.replace(params_policy=new_params)

        return learner_state, {"loss_value": 0.0}
        
