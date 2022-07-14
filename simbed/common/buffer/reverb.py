import reverb
from acme import specs
from acme.adders import reverb as adders 
from typing import Tuple, NamedTuple
import socket

class ReverbBuffer(NamedTuple):
    table: reverb.Table 
    server: reverb.Server
    client: reverb.Client 

def make_reverb_buffer(env, name, ts) -> ReverbBuffer:

    N = len(ts)+1

    replay_buffer = reverb.Table(
        name=name, 
        max_size=1_000_000, 
        remover=reverb.selectors.Fifo(),
        sampler=reverb.selectors.Prioritized(1.0),
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        signature = adders.EpisodeAdder.signature(specs.make_environment_spec(env), sequence_length=N),     
        max_times_sampled=1
    )

    replay_server = reverb.Server([replay_buffer], port=None)

    host_name = socket.gethostname()
    #host_name = os.environ.get("REVERB_SERVER_IP_ADDRESS", "localhost")

    replay_server_address = f'{host_name}:%d' % replay_server.port
    client = reverb.Client(replay_server_address)

    return ReverbBuffer(
        replay_buffer, replay_server, client 
    )
