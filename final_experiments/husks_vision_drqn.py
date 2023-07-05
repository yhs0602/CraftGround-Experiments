import time

import numpy as np

from env_wrappers.husk_environment import env_makers
from final_experiments.runners.vision_drqn import train_vision_drqn
from final_experiments.wrappers.avoid_damage import AvoidDamageWrapper
from final_experiments.wrappers.simple_navigation import SimpleNavigationWrapper
from final_experiments.wrappers.vision import VisionWrapper
from models.dueling_vision_drqn import DuelingVisionDRQNAgent


def solved_criterion(avg_score, test_score, avg_test_score, episode):
    if episode < 500:
        return False
    if avg_score < 195.0:
        return False
    if test_score < 198.0:
        return False
    if avg_test_score is None:
        return True
    if avg_test_score < 195.0:
        return False
    return True


def run_experiment():
    seed = int(time.time())
    np.random.seed(seed)

    verbose = False
    env_path = None
    port = 8001
    inner_env, sound_list = env_makers["husks-random"](
        verbose, env_path, port, hud_hidden=True
    )
    env = AvoidDamageWrapper(
        VisionWrapper(
            SimpleNavigationWrapper(
                inner_env, num_actions=SimpleNavigationWrapper.TURN_RIGHT + 1
            ),
            x_dim=114,
            y_dim=64,
        )
    )

    train_vision_drqn(
        group="husks_vision_drqn",
        env=env,
        agent_class=DuelingVisionDRQNAgent,
        # env_name="husk-random-terrain",
        batch_size=16,
        gamma=0.99,
        learning_rate=0.00001,
        update_freq=1000,
        hidden_dim=128,
        time_step=8,
        kernel_size=5,
        stride=2,
        weight_decay=0.00001,
        buffer_size=1000000,
        epsilon_init=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        max_steps_per_episode=400,
        num_episodes=2000,
        warmup_episodes=10,
        seed=seed,
        solved_criterion=solved_criterion,
    )


if __name__ == "__main__":
    run_experiment()
