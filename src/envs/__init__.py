from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .sensor import SensorEnv
from .pursuit import PursuitEnv
from .aloha import AlohaEnv
from .hallway import HallwayEnv
from .tag import TagEnv
from .toygame import ToygameEnv
from .disperse import DisperseEnv
from .gather import GatherEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sensor"] = partial(env_fn, env=SensorEnv)
REGISTRY["pursuit"] = partial(env_fn, env=PursuitEnv)
REGISTRY["aloha"] = partial(env_fn, env=AlohaEnv)
REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)
REGISTRY["tag"] = partial(env_fn, env=TagEnv)
REGISTRY["toygame"] = partial(env_fn, env=ToygameEnv)
REGISTRY["disperse"] = partial(env_fn, env=DisperseEnv)
REGISTRY["gather"] = partial(env_fn, env=GatherEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
