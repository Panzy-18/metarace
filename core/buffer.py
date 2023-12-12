import pdb
from typing import Tuple, Union
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def update(self, tensor: torch.Tensor) -> None:
        batch_mean = torch.mean(tensor, dim=0)
        batch_var = torch.var(tensor, dim=0)
        batch_count = tensor.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, 
                            batch_mean: torch.Tensor, 
                            batch_var: torch.Tensor, 
                            batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class BaseRolloutStorage:
    
    def __init__(self, 
                 num_steps, 
                 num_processes, 
                 act_dim, 
                 obs_dim, 
                 device, 
                 mini_batch_size,
                 gamma: float = 0.99,
                 epsilon: float = 1e-8,
                 num_agents: int = 1,
    ):
        self.initialized = False
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.num_agents = num_agents
        self.scale_reward = True if num_processes > 1 else False
        self.batch_size = num_steps * num_processes
        self.mini_batch_size = mini_batch_size
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.step = 0
        
        def zeros(*shapes):
            return torch.zeros(*shapes, dtype=torch.float32, device=device)

        # record all agents' obs to get better value.
        # 0 in main agent
        self.observations = zeros(num_steps + 1, num_processes, num_agents, obs_dim) # start from 1
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device) # 1 represents not done
        self.value_preds = zeros(num_steps + 1, num_processes, 1)
        self.returns = None
        
        self.rewards = zeros(num_steps, num_processes, 1)
        self.action_log_probs = zeros(num_steps, num_processes, 1)
        self.actions = zeros(num_steps, num_processes, act_dim)
        self.advantages = None
        
        # To scale rewards
        self.ret = zeros(num_processes, 1)
        self.ret_rms = RunningMeanStd(shape=())
    
    @staticmethod
    def swap_and_flatten(tensor: torch.Tensor) -> torch.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        """
        shape = tensor.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return tensor.transpose(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    @property
    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        return self.step

    def reset(self,
              observation,
              value_pred,
    ):
        self.observations[0].copy_(observation) # for MA-setting
        self.value_preds[0].copy_(value_pred)
        self.step = 0
        self.initialized = True
        
    def after_update(self):
        self.observations[0].copy_(self.observations[-1]) # for MA-setting
        self.value_preds[0].copy_(self.value_preds[-1])
        self.masks[0].copy_(self.masks[-1])

    def insert(self, 
               observation, 
               action, 
               action_log_prob, 
               value_pred, 
               reward, 
               mask,
    ):
        """
        For transition:
        
        obs(src) -> obs(dst)
        
        observation, value_pred, mask is the state of dst \\
        action, action_log_prob, reward is the state of transition
        
        """
        self.observations[self.step + 1].copy_(observation)
        if value_pred is not None:
            self.value_preds[self.step + 1].copy_(value_pred)
        if mask is not None:
            self.masks[self.step + 1].copy_(mask) # start from 1
            
        self.actions[self.step].copy_(action)
        if action_log_prob is not None:
            self.action_log_probs[self.step].copy_(action_log_prob)
        if reward is not None:
            if self.scale_reward:
                self.ret = self.ret * self.gamma + reward
                self.ret_rms.update(self.ret)
            self.rewards[self.step].copy_(reward)

        self.step = (self.step + 1) % self.num_steps

    def _normalize_reward(self, reward: torch.Tensor):
        reward = reward / torch.sqrt(self.ret_rms.var + self.epsilon)
        return reward


class PPORolloutStorage(BaseRolloutStorage):
    
    def __init__(self, 
                 calculate_return="gae",
                 gae_lambda=0.95,
                 **kwargs,
    ):
        super().__init__(**kwargs)
        assert calculate_return in ["gae", "upgo"]
        self.calculate_return = calculate_return
        self.gae_lambda = gae_lambda
        
    def se_feed_forward_generator(self):
        """A generator to provide samples for SE."""
        num_steps, num_processes = self.num_steps, self.num_processes
        batch_size = num_processes * num_steps
        assert batch_size >= self.mini_batch_size, "Number of sampled steps should more than mini batch size."
        
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, drop_last=True)
        for indices in sampler:
            # indices: List[int]
            observations_batch = self.observations[:-1].view(-1, self.num_agents, self.obs_dim)[indices]
            next_observations_batch = self.observations[0:].view(-1, self.num_agents, self.obs_dim)[indices]
            actions_batch = self.actions.view(-1, self.act_dim)[indices]
            rewards_batch = self.rewards.view(-1, 1)[indices]
            yield observations_batch, next_observations_batch, actions_batch, rewards_batch
        
    def ppo_feed_forward_generator(self):
        """A generator to provide samples for PPO."""
        
        num_steps, num_processes = self.num_steps, self.num_processes
        batch_size = num_processes * num_steps
        assert batch_size >= self.mini_batch_size, "Number of sampled steps should more than mini batch size."
        
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, drop_last=True)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(-1, self.num_agents, self.obs_dim)[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            returns_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            
            old_actions_batch = self.actions.view(-1, self.act_dim)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            advantages_batch = self.advantages.view(-1, 1)[indices]
            
            yield observations_batch, old_actions_batch, value_preds_batch, returns_batch, \
                      masks_batch, old_action_log_probs_batch, advantages_batch

    def compute_returns_and_advantages(self):
        returns = torch.zeros(self.num_steps + 1, self.num_processes, 1, dtype=torch.float32, device=self.device)
        advantages = torch.zeros(self.num_steps, self.num_processes, 1, dtype=torch.float32, device=self.device)
        scaled_rewards = self._normalize_reward(self.rewards) if self.scale_reward else self.rewards
        if self.calculate_return == "gae":
            gae = 0
            for step in reversed(range(self.num_steps)):
                delta = scaled_rewards[step] + self.gamma * self.value_preds[step+1] * self.masks[step+1] - self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                advantages[step] = gae
                returns[step] = gae + self.value_preds[step]
        elif self.calculate_return == "upgo":
            # https://zhuanlan.zhihu.com/p/453250696
            returns[-2] = scaled_rewards[-1] + self.gamma * self.value_preds[-1] * self.masks[-1]
            advantages[-1] = returns[-2] - self.value_preds[-2]
            for step in reversed(range(self.num_steps - 1)):
                td_target = scaled_rewards[step] + self.gamma * self.value_preds[step+1] * self.masks[step+1]
                further_target = scaled_rewards[step] + self.gamma * returns[step+1] * self.masks[step+1]
                delta = td_target - self.value_preds[step]
                returns[step] = torch.where(delta >= 0, further_target, td_target)
                advantages[step] = returns[step] - self.value_preds[step]
        else:
            raise NotImplementedError()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.epsilon)
        
        self.returns = returns
        self.advantages = advantages

if __name__ == "__main__":
    rollouts = PPORolloutStorage(
        num_steps=1600, 
        num_processes=20, 
        act_dim=2, 
        obs_dim=161, 
        device="cpu", 
        mini_batch_size=400,
        gamma=0.99,
        epsilon=1e-6,
        calculate_return="upgo",
        gae_lambda=0.95,
        num_agents=2,
    )
    rollouts.se_feed_forward_generator()
    