import random
from typing import Optional, Any, Tuple

from gymnasium.core import WrapperObsType

import mydojo
from environments.base_environment import BaseEnvironment
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


class WitherEnvironment(BaseEnvironment):
    def make(
        self,
        verbose: bool,
        env_path: str,
        port: int,
        size_x: int = 114,
        size_y: int = 64,
        hud: bool = False,
        render_action: bool = True,
        render_distance: int = 2,
        simulation_distance: int = 5,
        num_withers: int = 1,
        min_distance: int = 10,
        max_distance: int = 40,
        random_pos: bool = True,  # randomize creeper position
        darkness: bool = False,  # add darkness effect
        terrain: int = 0,  # 0: flat, 1: random, 2: random with water
        can_hunt: bool = False,  # player can hunt creepers
        noisy: bool = False,  # add noisy mobs
        can_destroy: bool = False,  # wither can destroy blocks
        kill: bool = False,  # Difficulty is hard
        *args,
        **kwargs,
    ):
        if darkness:
            darkness_commands = ["effect give @p minecraft:darkness infinite 1 true"]
        else:
            darkness_commands = []
        if noisy:
            mobs_commands = [
                "minecraft:sheep ~ ~ 5",
                "minecraft:cow ~ ~ -5",
                "minecraft:cow ~5 ~ -5",
                "minecraft:sheep ~-5 ~ -5",
            ]
            noisy_sounds = [
                "subtitles.entity.sheep.ambient",  # sheep ambient sound
                "subtitles.block.generic.footsteps",  # player, animal walking
                "subtitles.block.generic.break",  # sheep eating grass
                "subtitles.entity.cow.ambient",  # cow ambient sound
            ]
        else:
            mobs_commands = []
            noisy_sounds = []
        husks_commands = generate_withers(
            num_withers,
            min_distance,
            max_distance,
            dy=20,
            randomize=random_pos,
        )
        killedStatKeys = []
        hunt_sounds = []
        inventory_commands = []
        if can_hunt:
            inventory_commands = [
                "item replace entity @p weapon.offhand with minecraft:shield",
                "give @p minecraft:diamond_sword",
            ]
            killedStatKeys = ["minecraft:wither"]
            hunt_sounds = [
                "subtitles.entity.player.attack.crit",
                "subtitles.entity.player.attack.knockback",
                "subtitles.entity.player.attack.strong",
                "subtitles.entity.player.attack.sweep",
                "subtitles.entity.player.attack.weak",
                "subtitles.entity.wither.shoot",
                "subtitles.entity.wither.death",
                "subtitles.entity.wither.hurt",
                "subtitles.item.shield.block",
            ]

        difficulty_commands = []
        if kill:
            difficulty_commands = ["difficulty hard"]
        else:
            difficulty_commands = ["difficulty easy"]

        explosion_commands = []
        if not can_destroy:
            explosion_commands = ["gamerule mobGriefing false"]
        initial_extra_commands = (
            difficulty_commands
            + explosion_commands
            + darkness_commands
            + mobs_commands
            + husks_commands
            + inventory_commands
        )
        sounds = (
            [
                "entity.wither.ambient",
                "entity.wither.break_block",
                "entity.wither.shoot",
                "entity.generic.explode",
                "subtitles.block.generic.footsteps",
            ]
            + noisy_sounds
            + hunt_sounds
        )

        if terrain == 0:  # flat
            seed = 12345
            initialPosition = None
            isWorldFlat = True
        elif terrain == 1:  # terrain
            seed = 3788863154090864390
            initialPosition = None
            isWorldFlat = False
        elif terrain == 2:  # forest
            seed = 3788863154090864390
            initialPosition = [-117, 75, -15]
            isWorldFlat = False

        class RandomCreeperWrapper(CleanUpFastResetWrapper):
            def __init__(self):
                self.env = mydojo.make(
                    verbose=verbose,
                    env_path=env_path,
                    port=port,
                    initialInventoryCommands=[],
                    initialPosition=initialPosition,  # nullable
                    initialMobsCommands=[
                        # player looks at south (positive Z) when spawn
                    ],
                    imageSizeX=size_x,
                    imageSizeY=size_y,
                    visibleSizeX=size_x,
                    visibleSizeY=size_y,
                    seed=seed,  # nullable
                    allowMobSpawn=False,
                    alwaysDay=True,
                    alwaysNight=False,
                    initialWeather="clear",  # nullable
                    isHardCore=False,
                    isWorldFlat=isWorldFlat,  # superflat world
                    obs_keys=["sound_subtitles"],
                    initialExtraCommands=initial_extra_commands,
                    isHudHidden=not hud,
                    render_action=render_action,
                    render_distance=render_distance,
                    simulation_distance=simulation_distance,
                    killedStatKeys=killedStatKeys,
                )
                super(RandomCreeperWrapper, self).__init__(self.env)

            def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict[str, Any]] = None,
            ) -> Tuple[WrapperObsType, dict[str, Any]]:
                extra_commands = ["tp @e[type=!player] ~ -500 ~"]
                extra_commands.extend(initial_extra_commands)
                options.update(
                    {
                        "extra_commands": extra_commands,
                    }
                )
                obs = self.env.reset(
                    seed=seed,
                    options=options,
                )
                # obs["extra_info"] = {
                #     "husk_dx": dx,
                #     "husk_dz": dz,
                # }
                return obs

        return RandomCreeperWrapper(), sounds


def generate_withers(
    num_husks,
    min_distnace,
    max_distance,
    dy: Optional[int] = None,
    randomize: bool = True,
):
    commands = []
    success_count = 0
    while success_count < num_husks:
        if randomize:
            dx = generate_random(-max_distance, max_distance)
            dz = generate_random(-max_distance, max_distance)
        else:
            dx = 0
            dz = 5
        if dy is None:
            dy = 0
        if dx * dx + dz * dz + dy * dy < min_distnace * min_distnace:
            continue
        commands.append("summon minecraft:wither " + f"~{dx} ~{dy} ~{dz}")

        success_count += 1
        print(f"dx={dx}, dz={dz}")
    return commands


def generate_random(start, end):
    return random.randint(start, end)
