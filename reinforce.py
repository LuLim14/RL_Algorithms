import torch
import torch.nn as nn
import numpy as np
import gym

from collections import deque
from typing import Any
from torch import Tensor
from tqdm import tqdm

from Config import ReinforceConfig


class Policy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int) -> None:
        super(Policy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
            nn.Dropout(p=0.6),
            nn.ReLU()
        )

        self.inner_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.Dropout(p=0.6),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=self.out_dim)

        self.inner_layers = nn.ModuleList([self.inner_layer for _ in range(1)])

    def forward(self, x_input: Tensor) -> Tensor:
        x = self.input_layer(x_input)
        return torch.softmax(self.classifier(x), dim=-1)


def select_action(current_state: np.ndarray, policy_net: Any) -> tuple[int, Any]:
    state = torch.from_numpy(current_state).float().unsqueeze(0)

    probs = policy_net(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample().item()

    return action, m.log_prob(torch.tensor(action, dtype=torch.long))


def discount_reward(rewards: list[float], gamma: float = 0.99) -> deque[float]:
    discounted_reward_at_time = deque()
    current_gamma = gamma
    current_reward = 0
    for r in rewards[::-1]:
        current_reward = r + current_gamma * current_reward
        discounted_reward_at_time.appendleft(current_reward)
    return (discounted_reward_at_time - np.mean(discounted_reward_at_time)) / np.std(discounted_reward_at_time)


def init_weights(layer: Any) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)


def reinforce_train(agent: Any, env: Any, optimizer: Any, num_episodes: int = 1000, gamma: float = 0.99) -> None:
    agent.train()
    for episode in tqdm(range(num_episodes)):
        state = env.reset(seed=reinforce_conf.env_seed)[0]
        log_probs = []
        rewards = []
        episode_reward = 0.0
        done = False

        while not done:
            action, log_prob = select_action(state, agent)
            next_state, reward, done, _, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
            state = np.array(next_state)

        discounted_rewards = discount_reward(rewards, gamma)

        discounted_rewards = torch.tensor(discounted_rewards)

        log_probs = torch.stack(log_probs)
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        print(f'Episode: {episode}, loss: {policy_loss}, episode reward: {episode_reward}')

        if episode_reward > env.spec.reward_threshold:
            print(f'Solving. Running reward: {episode_reward}, need: {env.spec.reward_threshold}')
            break

        if episode % 100 == 0:
            print(f'Episode: {episode}, loss: {policy_loss}, reward: {np.mean(rewards)}')


def test_policy(agent: Any, env: Any, num_episodes: int = 10) -> None:
    agent.eval()
    rewards = []
    for episode in tqdm(range(num_episodes)):
        state = env.reset(seed=reinforce_conf.env_seed)[0]
        done = False
        total_reward = 0

        while not done:
            state = torch.from_numpy(state).float()
            with torch.no_grad():
                probs = agent(state)
            action = torch.argmax(probs).item()
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        print(f'Episode {episode}, reward: {total_reward}')
        if total_reward > env.spec.reward_threshold:
            print(f'Solving. Running reward: {total_reward}, need: {env.spec.reward_threshold}')
            break
    print(f'Mean total reward: {np.mean(rewards)}')


if __name__ == "__main__":
    tmp_env = gym.make("CartPole-v1", render_mode="human")
    reinforce_conf = ReinforceConfig()
    tmp_env.reset(seed=reinforce_conf.env_seed)

    input_dim = tmp_env.observation_space.shape[0]
    output_dim = tmp_env.action_space.n

    agent = Policy(input_dim=input_dim, hidden_dim=reinforce_conf.hidden_dim, out_dim=output_dim)
    agent.apply(init_weights)
    optimizer = torch.optim.Adam(agent.parameters(), lr=reinforce_conf.learning_rate)

    reinforce_train(agent, tmp_env, optimizer, num_episodes=reinforce_conf.num_episodes)
    test_policy(agent, tmp_env, num_episodes=reinforce_conf.num_test_episodes)

    tmp_env.close()
