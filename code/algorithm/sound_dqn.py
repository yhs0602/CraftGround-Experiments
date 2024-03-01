from typing import Optional

import numpy as np
from torch import optim

from algorithm.dqn import DQNAlgorithm
from logger import Logger
from models.dueling_sound_dqn import DuelingSoundDQN


class SoundDQNAlgorithm(DQNAlgorithm):
    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        warmup_episodes: int,
        steps_per_episode: int,
        test_frequency,
        solved_criterion,
        hidden_dim,
        device,
        epsilon_init,
        epsilon_decay,
        epsilon_min,
        update_frequency,
        train_frequency,
        replay_buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
        tau,
        stack_frames: Optional[int] = 1,
        **kwargs,
    ):
        super().__init__(
            env,
            logger,
            num_episodes,
            warmup_episodes,
            steps_per_episode,
            test_frequency,
            solved_criterion,
            hidden_dim,
            device,
            epsilon_init,
            epsilon_decay,
            epsilon_min,
            update_frequency,
            train_frequency,
            replay_buffer_size,
            batch_size,
            gamma,
            learning_rate,
            weight_decay,
            tau,
            stack_frames,
        )
        self.state_dim = (np.prod(env.observation_space.shape),)
        self.policy_net = DuelingSoundDQN(
            self.state_dim, self.action_dim, hidden_dim
        ).to(device)
        self.target_net = DuelingSoundDQN(
            self.state_dim, self.action_dim, hidden_dim
        ).to(device)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
