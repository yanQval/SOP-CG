from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .socg_learner import SocgLearner
from .dcg_learner import DCGLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["socg_learner"] = SocgLearner
REGISTRY["dcg_learner"] = DCGLearner