from abc import ABC, abstractmethod
from typing import Optional

import jax 
import jax.numpy as jnp

from simbed.common.types import nn 
from simbed.common.utils import idx_in_pytree


class ObservationReferenceSourceMeta(ABC):
    @abstractmethod
    def get_reference_actor(self) -> nn.Observations:
        pass 

    @abstractmethod
    def get_references_for_optimisation(self) -> nn.Observationss:
        pass 

    def change_reference_of_actor(self, i: int) -> None:
        raise NotImplementedError 

    def change_references_for_optimisation(self) -> None:
        raise NotImplementedError


class ObservationReferenceSource(ObservationReferenceSourceMeta):
    def __init__(self, yss: nn.Observationss, uss: Optional[nn.Actionss] = None, i_actor=0):
        self._i_actor = i_actor
        self._yss = yss 
        self._uss = uss 

    def get_references_for_optimisation(self) -> nn.Observationss:
        return jax.tree_util.tree_map(jnp.atleast_3d, self._yss)

    def get_reference_actor(self) -> nn.Observations:
        return idx_in_pytree(self.get_references_for_optimisation(), start=self._i_actor)

    def change_reference_of_actor(self, i_actor: int) -> None:
        self._i_actor = i_actor
        