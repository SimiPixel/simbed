from acme.adders import reverb as adders 
import jax
import reverb 
import jax.numpy as jnp 

def to_jax(pytree, device=None):
    pytree = jax.tree_util.tree_map(jnp.asarray, pytree)
    return jax.device_put(pytree, device)

def make_adder(client, replay_buffer_name, ts):
    adder = adders.EpisodeAdder(client, len(ts)+1, priority_fns={replay_buffer_name: lambda *args: 1.0})
    return adder

class ExhaustiveIteratorFromAdder:
    def __init__(self, client: reverb.Client, table: reverb.Table, device=None, padding=32):
        self.table = table 
        self.client = client
        table_name = table.name
        ds = reverb.TrajectoryDataset.from_table_signature(self.client.server_address, table_name, 20)
        self.iterator = ds.batch(1).as_numpy_iterator()
        self.padded_sample = None 
        self.padding=padding
        self.i = -1 
        self.device = device

    def __next__(self):
        number_of_inserts = self.client.server_info()[self.table.name].num_unique_samples
        if self.i+1 < number_of_inserts:
            self.i += 1
            
            sample = to_jax(self.iterator.next(), self.device)
            sample = jax.tree_util.tree_map(lambda a: jnp.atleast_2d(a), sample)

            if self.padded_sample is None:
                self.padded_sample = jax.tree_util.tree_map(
                        lambda a: jnp.zeros((self.padding,)+jnp.atleast_2d(a).shape[1:], dtype=a.dtype), sample
                    )
                self.padded_sample = jax.device_put(self.padded_sample, self.device)

            self.padded_sample = jax.tree_util.tree_map(
                lambda a1, a2: a1.at[self.i:self.i+1].set(a2), self.padded_sample, sample
            )

            return self.padded_sample
        else:
            return self.padded_sample
