from typing import SupportsFloat, Any, Optional, Tuple

from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper
from wrappers.simple_navigation import SimpleNavigationWrapper


# Sound wrapper
class JumpHelperWrapper(CleanUpFastResetWrapper):
    def __init__(self, env):
        self.env = env
        self.prev_pos = None
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        new_pos = (info_obs.x, info_obs.y, info_obs.z)
        if (
            self.prev_pos == new_pos
        ):  # tried to move but couldn't. Because of a wall or something.
            if action in [
                SimpleNavigationWrapper.FORWARD,
                SimpleNavigationWrapper.BACKWARD,
                SimpleNavigationWrapper.MOVE_RIGHT,
                SimpleNavigationWrapper.MOVE_LEFT,
            ]:
                reward -= 0.02

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
    ) -> Tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info_obs = info["obs"]
        self.prev_pos = (info_obs.x, info_obs.y, info_obs.z)
        return obs, info
