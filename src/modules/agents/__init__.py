REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent
from .rnn_agent import CGRNNAgent
REGISTRY["rnn_cg"] = CGRNNAgent
from .rnn_agent import CommunicationUnit
REGISTRY["comm_unit"] = CommunicationUnit
from .rnn_feature_agent import RNNFeatureAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
