from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .sensor import SensorEnv
from .pursuit import PursuitEnv
from .aloha import AlohaEnv
from .hallway import HallwayEnv
from .coordination_games import MatchingGameEnv
from .tag import TaggameEnv
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
REGISTRY["coordination_game_matching"] = partial(env_fn, env=MatchingGameEnv)
REGISTRY["tag"] = partial(env_fn, env=TagEnv)
#REGISTRY["postman"] = partial(env_fn, env=PostmanEnv)
REGISTRY["toygame"] = partial(env_fn, env=ToyEnv)
REGISTRY["disperse"] = partial(env_fn, env=DisperseEnv)
REGISTRY["gather"] = partial(env_fn, env=GatherEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
