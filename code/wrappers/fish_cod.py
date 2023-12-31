from collections import deque
from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class FishCodWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, **kwargs):
        self.env = env
        self.fish_deque = deque(maxlen=2)
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        fish_caught = info_obs.misc_statistics["fish_caught"]
        # print(fish_caught)
        self.fish_deque.append(fish_caught)
        if len(self.fish_deque) == 2:
            if self.fish_deque[1] > self.fish_deque[0]:  # fish_caught increased
                # print("Fish Caught")
                reward += 1
                terminated = True

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.fish_deque.clear()
        return obs, info
