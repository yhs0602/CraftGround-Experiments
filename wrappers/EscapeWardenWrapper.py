from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from mydojo.minecraft import int_to_action
from wrapper_runner import WrapperRunner


class EscapeHuskWrapper(gym.Wrapper):
    def __init__(self):
        self.env = mydojo.make(
            initialInventoryCommands=[],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                # "minecraft:sheep",
                "minecraft:warden ~ ~ ~15",
                # player looks at south (positive Z) when spawn
            ],
            imageSizeX=228,
            imageSizeY=128,
            visibleSizeX=228,
            visibleSizeY=128,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=True,  # superflat world
        )
        super(EscapeHuskWrapper, self).__init__(self.env)
        self.action_space = gym.spaces.Discrete(6)
        initial_env = self.env.initial_env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, initial_env.imageSizeX, initial_env.imageSizeY),
            dtype=np.uint8,
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = int_to_action(action)
        action_arr[2] = 2  # must crawl
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead

        reward = 1  # initial reward
        if is_dead:  #
            if self.initial_env.isHardCore:
                reward = -10000000
                terminated = True
            else:  # send respawn packet
                # pass
                reward = -200
                terminated = True
                # send_respawn(self.json_socket)
                # print("Dead!!!!!")
                # res = self.json_socket.receive_json()  # throw away
        return rgb, reward, terminated, truncated, info  # , done: deprecated

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        return obs


def main():
    env = EscapeHuskWrapper()
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.0005
    update_freq = 25
    runner = WrapperRunner(
        env,
        "EscapeWarden15-6Actions",
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        update_frequency=update_freq,
        solved_criterion=lambda avg_score, episode: avg_score >= 390.0
        and episode >= 100,
    )
    runner.run_wrapper()


if __name__ == "__main__":
    main()
