from dataclasses import dataclass


@dataclass
class ReinforceConfig:
    num_episodes: int = 200
    num_test_episodes: int = 200
    gamma: float = 0.99
    hidden_dim: int = 128
    learning_rate: float = 1.5e-2
    env_seed: int = 42
