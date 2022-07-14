from typing import NamedTuple, Callable, Tuple
from simbed.common.types import nn 
from simbed.common.utils import idx_in_pytree
from dataclasses import dataclass

@dataclass(frozen=True)
class PolicyInput:
    pass 

#TODO NamedTuple should be a dataclass
class PolicyInputCurrentReferenceOnly(NamedTuple):
    observation: nn.Observation
    observation_ref: nn.Observation

def preprocess_current_reference_only(obs: nn.Observation, 
        obss_ref: nn.Observations, timestep: int):
    return PolicyInputCurrentReferenceOnly(
        obs, idx_in_pytree(obss_ref, start=timestep)
    )

class Policy(NamedTuple):
    init: Callable[[nn.PRNGKey, PolicyInput], 
        Tuple[nn.Params, nn.NeuralNetworkState]]

    apply: Callable[[nn.Params, nn.NeuralNetworkState, PolicyInput],
        Tuple[nn.NeuralNetworkState, nn.Action]]

    preprocess: Callable[[nn.Observation, nn.Observations, int], PolicyInput]
