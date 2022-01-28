from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .sensors import SensorEnv
from .prey import PreyEnv
from .pursuit import PursuitEnv
from .aloha import AlohaEnv
from .hallway import HallwayEnv
from .hallwaykai import HallwayKaiEnv
from .coordination_games import MatchingGameEnv
from .chasing import ChasingEnv
#from .postman import PostmanEnv
from .toygame2 import Toy2Env
from .disperse import DisperseEnv
from .gather import GatherEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sensor"] = partial(env_fn, env=SensorEnv)
REGISTRY["prey"] = partial(env_fn, env=PreyEnv)
REGISTRY["pursuit"] = partial(env_fn, env=PursuitEnv)
REGISTRY["aloha"] = partial(env_fn, env=AlohaEnv)
REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)
REGISTRY["hallwaykai"] = partial(env_fn, env=HallwayKaiEnv)
REGISTRY["coordination_game_matching"] = partial(env_fn, env=MatchingGameEnv)
REGISTRY["chasing"] = partial(env_fn, env=ChasingEnv)
#REGISTRY["postman"] = partial(env_fn, env=PostmanEnv)
REGISTRY["toygame2"] = partial(env_fn, env=Toy2Env)
REGISTRY["disperse"] = partial(env_fn, env=DisperseEnv)
REGISTRY["gather"] = partial(env_fn, env=GatherEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
