from mtrl.config.rl import AlgorithmConfig

from .base import Algorithm, OffPolicyAlgorithm
from .mtsac import MTSAC, MTSACConfig
from .mtppo import MTPPOConfig, MTPPO
from .pcgrad import PCGradConfig, PCGrad

def get_algorithm_for_config(config: AlgorithmConfig) -> type[Algorithm]:
    if type(config) is MTSACConfig:
        return MTSAC
    elif type(config) is MTPPOConfig:
        return MTPPO
    elif type(config) is PCGradConfig:
        return PCGrad
    else:
        raise ValueError(f"Invalid config type: {type(config)}")
