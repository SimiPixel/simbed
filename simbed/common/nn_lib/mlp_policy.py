from simbed.common.types import policy
import haiku as hk 
import jax 
from acme.jax import utils 
from simbed.common.types import nn 

def MLPPolicy(preprocess, hidden_layers=[32,1], act_fn=jax.nn.relu, act_final=False):

    @hk.without_apply_rng
    @hk.transform
    def forward(X: policy.PolicyInput):
        X = utils.batch_concat(X)
        X = hk.nets.MLP(hidden_layers, activation=act_fn, activate_final=act_final)(X)
        X = utils.squeeze_batch_dim(X)
        return X 

    def apply(params: nn.Params, state: nn.NeuralNetworkState, X):
        return state, forward.apply(params, X)

    def init(key, X_dummy):
        return forward.init(key, X_dummy), nn.NeuralNetworkStateless()

    return policy.Policy(
        init, apply, preprocess
    )

