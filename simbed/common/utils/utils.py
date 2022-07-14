import jax 
from acme.jax import utils 

def idx_in_pytree(tree, start, stop=None, axis=0, keepdim=False):
    if stop:
        if not keepdim:
            if stop != start + 1:
                raise Exception()  

    if stop is None:
        stop = start + 1

    def slicing_fun(arr):
        shape = arr.shape
        idxs = tuple([slice(shape[i]) for i in range(axis)] + [slice(start, stop)])
        arr = arr[idxs]
        if keepdim:
            return arr
        else:
            return arr[tuple(slice(shape[i]) for i in range(axis))+(0,)]
    return jax.tree_util.tree_map(slicing_fun, tree)

pytree_like = utils.zeros_like

def generate_ts(time_limit, control_timestep):
    """Generate action sampling times

    Args:
        time_limit (float): Upper bound of time. Not included
        control_timestep (float): Sample rate in seconds

    Returns:
        _type_: Array of times
    """
    return jax.numpy.arange(0,time_limit,step=control_timestep)

from simbed.actors.random_actor import RandomlyActingActor
def actor2launch_policy(actor: RandomlyActingActor):
    def _policy(ts):
        action = actor.select_action(ts.observation)
        actor.actor_as_launch_policy_cleanup()
        return action 
    return _policy
