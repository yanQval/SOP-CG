from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .sensors import SensorEnv
from .prey import PreyEnv
from .pursuit import PursuitEnv
from .aloha import AlohaEnv
from .hallway import HallwayEnv
from .hallwaykai import HallwayKaiEnv
from .coordination_games import MatchingGameEnv
from .bearer import BearerEnv
from .bearer2 import Bearer2Env
from .bearer3 import Bearer3Env
from .bearer4 import Bearer4Env
from .bearer5 import Bearer5Env
from .bearer6 import Bearer6Env
from .chasing import ChasingEnv
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
REGISTRY["bearer"] = partial(env_fn, env=BearerEnv)
REGISTRY["bearer2"] = partial(env_fn, env=Bearer2Env)
REGISTRY["bearer3"] = partial(env_fn, env=Bearer3Env)
REGISTRY["bearer4"] = partial(env_fn, env=Bearer4Env)
REGISTRY["bearer5"] = partial(env_fn, env=Bearer5Env)
REGISTRY["bearer6"] = partial(env_fn, env=Bearer6Env)
REGISTRY["chasing"] = partial(env_fn, env=ChasingEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
