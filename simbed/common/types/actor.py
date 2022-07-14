from flax import struct 
from simbed.common.types import nn 

@struct.dataclass
class ActorState:
    pass 

@struct.dataclass
class ActorStateless(ActorState):
    key: nn.PRNGKey
    count: int 

@struct.dataclass
class ActorStatefull(ActorStateless):
    state: nn.NestedArrays

@struct.dataclass
class ActorStateNeuralNetworkState(ActorStateless):
    state: nn.NeuralNetworkState
