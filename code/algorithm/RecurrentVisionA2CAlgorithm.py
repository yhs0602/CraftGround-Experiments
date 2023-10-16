import torch
import torch.optim

from algorithm.recurrent_a2c import RecurrentA2CAlgorithm
from logger import Logger
from models.recurrent_a2c import RecurrentVisionA2C


class RecurrentVisionA2CAlgorithm(RecurrentA2CAlgorithm):
    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        steps_per_episode: int,
        test_frequency,
        solved_criterion,
        hidden_dim,
        kernel_size,
        stride,
        device,
        update_frequency,
        train_frequency,
        time_step,
        gamma,
        optimizer,
        beta,
        **kwargs,
    ):
        super().__init__(
            env,
            logger,
            num_episodes,
            steps_per_episode,
            test_frequency,
            solved_criterion,
            hidden_dim,
            device,
            update_frequency,
            train_frequency,
            time_step,
            gamma,
        )
        self.beta = beta
        self.kernel_size = kernel_size
        self.stride = stride
        self.state_dim = env.observation_space.shape
        self.actor_critic = RecurrentVisionA2C(
            self.state_dim,
            self.action_dim,
            kernel_size,
            stride,
            hidden_dim,
            device,
        ).to(device)
        optim_name = optimizer.get("name", "Adam")
        optimizer_class = getattr(torch.optim, optim_name)
        self.actor_critic_optim = optimizer_class(
            self.actor_critic.parameters(), **optimizer["params"]
        )
