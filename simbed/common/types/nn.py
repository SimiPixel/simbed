from flax import struct
from typing import NewType, Any
import jax.numpy as jnp 

NestedArrays = Any
PRNGKey = NewType("PRNGKey", jnp.ndarray)

@struct.dataclass
class NeuralNetworkState:
    pass 

@struct.dataclass
class NeuralNetworkStateless(NeuralNetworkState):
    pass 

@struct.dataclass
class NeuralNetworkStatefull(NeuralNetworkState):
    state: NestedArrays

Observation = NewType("Observation", jnp.ndarray)
Observations = NewType("Observations", jnp.ndarray)
Observationss = NewType("Observationss", jnp.ndarray)

Action = NewType("Action", jnp.ndarray)
Actions = NewType("Actions", jnp.ndarray)
Actionss = NewType("Actionss", jnp.ndarray)

Params = NestedArrays
