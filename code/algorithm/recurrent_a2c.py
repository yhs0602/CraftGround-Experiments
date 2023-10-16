import time
from collections import deque
from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical

import criterion
from logger import Logger


class RecurrentA2CAlgorithm:
    actor_critic_optim: torch.optim.Optimizer
    actor_critic: torch.nn.Module

    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        steps_per_episode: int,
        test_frequency,
        solved_criterion,
        hidden_dim,
        device,
        update_frequency,
        train_frequency,
        time_step,
        gamma,
        beta,
        **kwargs,
    ):
        self.beta = beta
        self.logger = logger
        self.env = env
        self.num_episodes = num_episodes
        self.test_frequency = test_frequency
        self.steps_per_episode = steps_per_episode
        self.action_dim = env.action_space.n
        self.hidden_dim = hidden_dim

        self.update_frequency = update_frequency
        self.train_frequency = train_frequency

        self.time_step = time_step

        self.gamma = gamma

        self.device = device

        self.total_steps = 0
        self.episode = 0

        solved_criterion_config = solved_criterion
        criterion_cls = getattr(criterion, solved_criterion_config["name"])
        self.solved_criterion = criterion_cls(**solved_criterion_config["params"])

    def run(self):
        self.logger.start_training()
        recent_scores = deque(maxlen=30)
        recent_test_scores = deque(maxlen=10)
        scores = []
        avg_scores = []
        avg_test_scores = []
        avg_score = None
        avg_test_score = None
        test_score = None
        self.total_steps = 0
        for episode in range(0, self.num_episodes):
            self.episode = episode
            if episode % self.test_frequency == 0:  # testing
                test_score, num_steps, time_took = self.test_agent(episode)
                recent_test_scores.append(test_score)
                avg_test_score = np.mean(recent_test_scores)
                avg_test_scores.append(avg_test_score)
                self.logger.log(
                    {
                        "test/step": episode,
                        "test/score": test_score,
                        "test/episode_length": num_steps,
                    }
                )
            else:  # training
                (
                    episode_reward,
                    num_steps,
                    time_took,
                    avg_actor_loss,
                    avg_critic_loss,
                    reset_extra_info,
                    action_entropy,
                ) = self.train_agent()
                scores.append(episode_reward)
                recent_scores.append(episode_reward)
                avg_score = np.mean(recent_scores)
                avg_scores.append(avg_score)
                self.logger.log(
                    {
                        "episode": episode,
                        "score": episode_reward,
                        "avg_score": avg_score,
                        "avg_actor_loss": avg_actor_loss,
                        "avg_critic_loss": avg_critic_loss,
                        "action_entropy": action_entropy,
                        "episode_length": num_steps,
                    }
                )
            if num_steps == 0:
                num_steps = 1
            print(
                f"Seconds per episode{episode}: {time_took}/{num_steps}={time_took / num_steps:.5f} seconds"
            )

            if self.solved_criterion.criterion(
                avg_score, test_score, avg_test_score, episode
            ):
                print(f"Solved in {episode} episodes!")
                break

    def test_agent(self):
        self.logger.before_episode(
            self.env, should_record_video=True, episode=self.episode
        )
        state, reset_info = self.env.reset(options={"fast_reset": True})
        hidden_state, cell_state = self.actor_critic.init_hidden_states(bsize=1)
        episode_reward = 0
        steps_in_episode = 0
        start_time = time.time()
        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=True)
            _, action, (hidden_state, cell_state) = self.get_next_hidden_state(
                state, hidden_state, cell_state
            )
            next_state, reward, done, truncated, info = self.env.step(action)
            episode_reward += reward
            steps_in_episode += 1
            if done:
                break
            state = next_state
        time_took = time.time() - start_time
        self.logger.after_episode()
        return episode_reward, steps_in_episode, time_took

    def train_agent(self):
        self.logger.before_episode(
            self.env, should_record_video=False, episode=self.episode
        )
        state, reset_info = self.env.reset(options={"fast_reset": True})
        hidden_state, cell_state = self.actor_critic.init_hidden_states(bsize=1)
        episode_reward = 0
        steps_in_episode = 0
        actor_losses = []
        critic_losses = []
        start_time = time.time()

        states, actions, rewards, next_states, dones = [], [], [], [], []
        hidden_states, cell_states = [], []
        next_hidden_states, next_cell_states = [], []

        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=False)
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)
            _, action, (hidden_state, cell_state) = self.get_next_hidden_state(
                state, hidden_state, cell_state
            )
            next_state, reward, done, truncated, info = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            next_hidden_states.append(hidden_state)
            next_cell_states.append(cell_state)
            state = next_state
            episode_reward += reward
            steps_in_episode += 1
            if done:
                break

        states_batch_np = np.stack(states)
        actions_batch_np = np.stack(actions)
        rewards_batch_np = np.stack(rewards)
        next_states_batch_np = np.stack(next_states)
        dones_batch_np = np.stack(dones)
        next_hidden_states_batch_np = np.stack(next_hidden_states)
        next_cell_states_batch_np = np.stack(next_cell_states)
        hidden_states_batch_np = np.stack(hidden_states)
        cell_states_batch_np = np.stack(cell_states)

        states_batch = torch.FloatTensor(states_batch_np).to(self.device)
        actions_batch = torch.LongTensor(actions_batch_np).to(self.device)
        rewards_batch = torch.FloatTensor(rewards_batch_np).to(self.device)
        next_states_batch = torch.FloatTensor(next_states_batch_np).to(self.device)
        dones_batch = torch.FloatTensor(dones_batch_np).to(self.device)
        next_hidden_states_batch = torch.FloatTensor(next_hidden_states_batch_np).to(
            self.device
        )
        next_cell_states_batch = torch.FloatTensor(next_cell_states_batch_np).to(
            self.device
        )
        hidden_states_batch = torch.FloatTensor(hidden_states_batch_np).to(self.device)
        cell_states_batch = torch.FloatTensor(cell_states_batch_np).to(self.device)

        probs, value = self.actor_critic(
            states_batch, hidden_states_batch, cell_states_batch
        )
        _, next_value = self.actor_critic(
            next_states_batch, next_hidden_states_batch, next_cell_states_batch
        )

        advantage = rewards_batch + self.gamma * next_value * (1 - dones_batch) - value

        dist = torch.distributions.Categorical(probs=probs)
        entropy = dist.entropy().mean()
        actor_loss = -(
            dist.log_prob(actions_batch) * advantage.detach() + entropy * self.beta
        )
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        self.actor_critic_optim.zero_grad()
        loss.mean().backward()
        self.actor_critic_optim.step()

        time_took = time.time() - start_time

        avg_actor_loss = actor_loss.mean().item()
        avg_critic_loss = critic_loss.mean().item()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

        action_entropy = dist.entropy().mean().item()
        return (
            episode_reward,
            steps_in_episode,
            time_took,
            avg_actor_loss,
            avg_critic_loss,
            reset_info,
            action_entropy,
        )

    def get_next_hidden_state(
        self, state, hidden_state, cell_state
    ) -> Tuple[Categorical, int, Tuple[torch.Tensor, torch.Tensor]]:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor_critic.eval()
        with torch.no_grad():  # TODO: check if this is correct. detach?
            probs, advantages, (new_hidden_state, new_cell_state) = self.actor_critic(
                state,
                batch_size=1,
                time_step=1,
                hidden_state=hidden_state,
                cell_state=cell_state,
            )
        self.actor_critic.train()  # TODO: check if this is correct
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()
        return dist, action, (new_hidden_state, new_cell_state)
