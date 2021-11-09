REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .cg_episode_runner import CGEpisodeRunner
REGISTRY["cg_episode"] = CGEpisodeRunner

from .advtrain_cg_episode_runner import AdvCGEpisodeRunner
REGISTRY["advtrain_cg_episode"] = AdvCGEpisodeRunner

from .advtrain_episode_runner import AdvEpisodeRunner
REGISTRY["advtrain_episode"] = AdvEpisodeRunner