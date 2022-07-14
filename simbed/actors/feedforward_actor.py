from acme.adders import reverb as adders 
from typing import Optional

from simbed.common.types import nn, actor
from .random_actor import RandomlyActingActor


class FeedforwardActor(RandomlyActingActor):
    def __init__(self, 
        u_ref: nn.Actions, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._u_ref = u_ref

    def select_action(self, observation: nn.Observation) -> nn.Action:
        del observation
        action = self._u_ref[self._state.count]
        self._state = actor.ActorStateless(self._state.key, self._state.count+1)
        return action
