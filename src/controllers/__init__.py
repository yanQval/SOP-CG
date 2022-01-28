REGISTRY = {}

from .basic_controller import BasicMAC
from .sopcg_controller import SopcgMAC
from .dcg_controller import DeepCoordinationGraphMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["sopcg_mac"] = SopcgMAC
REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC
