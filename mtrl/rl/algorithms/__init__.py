from mtrl.config.rl import AlgorithmConfig

from .base import Algorithm, OffPolicyAlgorithm
from .mtsac import MTSAC, MTSACConfig
from .mtppo import MTPPOConfig, MTPPO
from .pcgrad import PCGradConfig, PCGrad
from .gradnorm import GradNormConfig, GradNorm

def get_algorithm_for_config(config: AlgorithmConfig) -> type[Algorithm]:
    # Could the class be included in the config so we don't have to keep updating across files?
    if type(config) is MTSACConfig:
        return MTSAC
    elif type(config) is MTPPOConfig:
        return MTPPO
    elif type(config) is PCGradConfig:
        return PCGrad
    elif type(config) is GradNormConfig:
        return GradNorm
    else:
        raise ValueError(f"Invalid config type: {type(config)}")
