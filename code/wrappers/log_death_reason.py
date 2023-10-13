from typing import SupportsFloat, Any

from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class LogDeathReasonWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, **kwargs):
        self.env = env
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        is_dead = info_obs.is_dead

        if is_dead:
            print(f"{info_obs.last_death_message=}")

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated
