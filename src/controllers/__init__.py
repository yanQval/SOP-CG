REGISTRY = {}

from .basic_controller import BasicMAC
from .socg_controller import SocgMAC
from .dcg_controller import DeepCoordinationGraphMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["socg_mac"] = SocgMAC
REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC