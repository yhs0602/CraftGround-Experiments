import time
from collections import deque
from typing import Optional

import numpy as np
import torch
from torch import optim

from models.bimodal_replay_buffer import BiModalReplayBuffer
from models.dueling_bimodal_dqn import DuelingBiModalDQN
from algorithm.dqn import DQNAlgorithm
from logger import Logger


class BimodalDQNAlgorithm(DQNAlgorithm):
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
        kernel_size,
        stride,
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.state_dim = env.observation_space["vision"].shape
        self.sound_dim = env.observation_space["sound"].shape
        self.policy_net = DuelingBiModalDQN(
            self.state_dim,
            self.sound_dim,
            self.action_dim,
            self.kernel_size,
            self.stride,
            self.hidden_dim,
        ).to(device)
        self.target_net = DuelingBiModalDQN(
            self.state_dim,
            self.sound_dim,
            self.action_dim,
            self.kernel_size,
            self.stride,
            self.hidden_dim,
        ).to(device)
        del self.replay_buffer
        self.replay_buffer = BiModalReplayBuffer(replay_buffer_size)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def test_agent(self, episode):
        self.logger.before_episode(self.env, should_record_video=True, episode=episode)
        state, reset_info = self.env.reset(options={"fast_reset": True})
        episode_reward = 0
        steps_in_episode = 0
        start_time = time.time()
        frames_deque = deque(maxlen=self.stack_frames)
        # Stack the initial state
        for _ in range(self.stack_frames):
            frames_deque.append(state)
        # This assumes the frames are concatenated along the channel dimension
        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=True)
            action = self.exploit_action(frames_deque)
            next_state, reward, done, truncated, info = self.env.step(action)
            episode_reward += reward
            steps_in_episode += 1
            if done:
                break
            frames_deque.append(next_state)
        time_took = time.time() - start_time
        self.logger.after_episode()
        return episode_reward, steps_in_episode, time_took

    def train_agent(self):
        self.logger.before_episode(
            self.env, should_record_video=False, episode=self.episode
        )
        state, info = self.env.reset(options={"fast_reset": True})
        episode_reward = 0
        steps_in_episode = 0
        losses = []
        start_time = time.time()
        frames_deque = deque(maxlen=self.stack_frames)
        next_frames_deque = deque(maxlen=self.stack_frames)
        # Stack the initial state
        for _ in range(self.stack_frames):
            frames_deque.append(state)
            next_frames_deque.append(state)
        # This assumes the frames are concatenated along the channel dimension
        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=False)
            if self.explorer.should_explore() or self.episode < self.warmup_episodes:
                action = np.random.choice(self.action_dim)
            else:  # exploit
                action = self.exploit_action(frames_deque)
            next_state, reward, done, truncated, info = self.env.step(action)
            next_frames_deque.append(next_state)
            episode_reward += reward
            steps_in_episode += 1
            self.total_steps += 1

            # add experience to replay buffer
            self.add_experience(frames_deque, action, next_frames_deque, reward, done)

            # update policy network
            if self.total_steps % self.train_frequency == 0:
                loss = self.update_policy_net()
                losses.append(loss)

            # update target network
            if self.total_steps % self.update_frequency == 0:
                self.update_target_net()

            if done:
                break
            # i.e. state = next_state
            frames_deque.append(state)

        end_time = time.time()
        if self.episode > self.warmup_episodes:
            self.explorer.after_episode()  # update epsilon
        avg_loss = np.mean([loss for loss in losses if loss is not None])
        return (
            episode_reward,
            steps_in_episode,
            end_time - start_time,
            avg_loss,
            info,
        )

    def add_experience(self, state, action, next_state, reward, done):
        # Extract sound and vision arrays from state and next_state
        audios = np.stack([s["sound"] for s in state])
        videos = np.stack([s["vision"] for s in state])
        next_audios = np.stack([s["sound"] for s in next_state])
        next_videos = np.stack([s["vision"] for s in next_state])
        self.replay_buffer.add(
            audios, videos, action, next_audios, next_videos, reward, done
        )

    def exploit_action(self, state) -> int:
        audio_state = np.stack([s["sound"] for s in state])
        video_state = np.stack([s["vision"] for s in state])
        # audio_state = state["sound"]
        # video_state = state["vision"]
        audio_state = torch.FloatTensor(audio_state).to(self.device)
        video_state = torch.FloatTensor(video_state).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(audio_state, video_state).detach()
        self.policy_net.train()
        return q_values.argmax().item()

    def update_policy_net(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return
        # print("Will update model")
        (
            audio,
            video,
            action,
            next_audio,
            next_video,
            reward,
            done,
        ) = self.replay_buffer.sample(self.batch_size)
        audio = audio.to(self.device).squeeze(1)
        video = video.to(self.device).squeeze(1)
        action = action.to(self.device)
        reward = reward.to(self.device).squeeze(1)
        next_audio = next_audio.to(self.device).squeeze(1)
        next_video = next_video.to(self.device).squeeze(1)
        done = done.to(self.device).squeeze(1)

        q_values = (
            self.policy_net(audio, video).gather(1, action.to(torch.int64)).squeeze(1)
        )
        next_q_values = self.target_net(next_audio, next_video).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
