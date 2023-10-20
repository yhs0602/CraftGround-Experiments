from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Avoid husk wrapper
class AvoidHuskWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, danger_reward=-0.1, **kwargs):
        self.env = env
        self.target_translation_key = "entity.minecraft.husk"
        self.danger_reward = danger_reward
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        surrounding_entities = info_obs.surrounding_entities

        husks1 = self.count_husks(surrounding_entities[1].entities)
        husks2 = self.count_husks(surrounding_entities[2].entities)
        husks5 = self.count_husks(surrounding_entities[5].entities)

        danger = husks1 + 0.5 * husks2 + 0.25 * husks5

        if danger > 0.1:
            reward = danger * self.danger_reward

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
        return obs, info

    def count_husks(self, entity_list) -> int:
        count = 0
        for animal in entity_list:
            if animal.translation_key == self.target_translation_key:
                count += 1
        return count
