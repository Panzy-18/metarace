import pdb
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
from typing import Union

class SimpleMLP(nn.Module):
    
    def __init__(self, 
                 input_size,
                 output_size,
                 hidden_size,
                 activation_fn="gelu",
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh() if activation_fn == "tanh" else nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh() if activation_fn == "tanh" else nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
        
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.orthogonal_(param, gain=1 if activation_fn == "tanh" else 2**0.5)
            else:
                nn.init.zeros_(param)
    
    def forward(self, x):
        return self.layers(x)
    
def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

class Actor(nn.Module):
    
    def __init__(self, 
                 input_size,
                 act_dim,
                 hidden_size,
                 discrete: bool = False,
                 actor_logstd: float = None,
                 sample_dist: str = None, # "normal", "beta"
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.discrete = False
        if self.discrete:
            self.sample_dist = "categorical"
        else:
            self.sample_dist = sample_dist
        self.mlp = SimpleMLP(
            input_size=input_size, 
            output_size=act_dim if self.sample_dist == "categorical" else act_dim*2, 
            hidden_size=hidden_size,
            activation_fn="tanh" if self.sample_dist == "normal" else "gelu"
        )

    
    def forward(self, inputs: torch.Tensor, actions: torch.Tensor=None, return_dict=False):
        """
        return actions, action_log_probs, entropy
        
        observations: [..., obs_dim] -> actions: [..., act_dim], logp [..., 1], ent [..., 1]
        """
        if actions is not None:
            assert inputs.shape[:-1] == actions.shape[:-1]
            actions = actions.detach()
        
        batch_shape = inputs.shape[:-1]
        logits = self.mlp(inputs.view(-1, self.input_size))
        logits = logits.reshape(batch_shape + (-1, ))
        if self.sample_dist == "categorical":
            dist = Categorical(logits=logits)
            actions = dist.sample() if actions is None else actions
            action_log_probs = dist.log_prob(actions).unsqueeze(-1)
        elif self.sample_dist == "normal":
            action_mean, action_std = torch.chunk(logits, 2, dim=-1)
            action_std = F.softplus(action_std) # softplus
            action_std.clamp(max=2) # to ensure model would not sample from too wide range.
            dist = Normal(loc=action_mean, scale=action_std)
            actions = dist.sample() if actions is None else actions
            action_log_probs_raw = dist.log_prob(actions)
            action_log_probs = action_log_probs_raw.sum(dim=-1).unsqueeze(-1)
            entropy = dist.entropy().sum(-1).unsqueeze(-1)
        elif self.sample_dist == "beta":
            action_alpha, action_beta = torch.chunk(logits, 2, dim=-1)
            action_alpha = F.softplus(action_alpha) + 1
            action_beta = F.softplus(action_beta) + 1
            dist = Beta(concentration0=action_alpha, concentration1=action_beta)
            actions = dist.sample()*6 - 3 if actions is None else actions
            action_log_probs_raw = dist.log_prob((actions + 3)/6)
            action_log_probs = action_log_probs_raw.sum(dim=-1).unsqueeze(-1)
            entropy = dist.entropy().sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
            
        if return_dict:
            return dict(
                actions=actions,
                action_log_probs=action_log_probs,
                entropy=entropy
            )
        return actions, action_log_probs, entropy
    
class Critic(nn.Module):

    def __init__(self, 
                 input_size,
                 hidden_size,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mlp = SimpleMLP(input_size=input_size, output_size=1, hidden_size=hidden_size)
        
    def forward(self, input: torch.Tensor):
        """
        return values
        
        input: [..., input_dim] -> values: [..., 1]
        """
        batch_shape = input.shape[:-1]
        values = self.mlp(input.view(-1, self.input_size)).reshape(batch_shape + (1,))
        return values

class StateEncoder(nn.Module):
    
    def __init__(self,
                 obs_dim,
                 state_embedding_size,
                 hidden_size,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.state_embedding_size = state_embedding_size
        self.hidden_size = hidden_size
        self.mlp = SimpleMLP(obs_dim, state_embedding_size, hidden_size)
        
    def forward(self, input: torch.Tensor):
        batch_shape = input.shape[:-1]
        state_embedding = self.mlp(input.view(-1, self.obs_dim)).reshape(batch_shape + (-1,))
        return AvgL1Norm(state_embedding)
        

class StatePredictor(nn.Module):
    
    def __init__(self,
                 obs_dim,
                 act_dim,
                 state_embedding_size,
                 hidden_size,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_embedding_size = state_embedding_size
        self.hidden_size = hidden_size
        self.mlp = SimpleMLP(
            input_size=state_embedding_size+act_dim,
            output_size=obs_dim+1, 
            hidden_size=hidden_size
        )
    
    def forward(self, state_embeddings, actions):
        batch_shape = state_embeddings.shape[:-1]
        input = torch.concat([state_embeddings, actions], dim=-1)
        output = self.mlp(input.view(-1, input.shape[-1])).reshape(batch_shape + (-1,))
        next_observations = output[..., :-1]
        rewards = output[..., -1].unsqueeze(-1)
        return next_observations, rewards

        
class BaseModel(nn.Module):
    
    def _to_tensor(self, arr) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        return arr.to(torch.float32).to(self.device)
    
    def compute_actions(self, observations: torch.Tensor) -> torch.Tensor:
        pass
    
    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pass

    def compute_values(self, input: torch.Tensor) -> torch.Tensor:
        pass


class PPOModel(BaseModel):
    
    def __init__(self,
                 obs_dim, 
                 act_dim,
                 hidden_size,
                 discrete: bool = False,
                 actor_logstd: float = None,
                 use_opponent_obs: bool = False,
                 num_agents: bool = 1,
                 sample_dist: str = "normal",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim 
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.discrete = discrete
        self.actor_logstd = actor_logstd
        self.use_opponent_obs = use_opponent_obs
        self.num_agents = num_agents
        self.sample_dist = sample_dist
        self.device = device
        self.actor = Actor(
            input_size=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            discrete=discrete,
            actor_logstd=actor_logstd,
            sample_dist=sample_dist,
        )
        self.critic = Critic(
            input_size=obs_dim*(num_agents if use_opponent_obs else 1),
            hidden_size=hidden_size
        )
        self.to(device)
    
    def compute_actions(self, observations: Union[torch.Tensor, np.ndarray]):
        observations = self._to_tensor(observations)
        return self.actor(observations)
    
    def evaluate_actions(self, 
                         observations: Union[torch.Tensor, np.ndarray],
                         actions: Union[torch.Tensor, np.ndarray],
    ):
        observations = self._to_tensor(observations)
        actions = self._to_tensor(actions)
        return self.actor(observations, actions)

    def compute_values(self, agent_observations: Union[torch.Tensor, np.ndarray]):
        # [..., num_agents, obs_dim]
        # -> [..., 1]
        agent_observations = self._to_tensor(agent_observations)
        batch_shape = agent_observations.shape[:-2]
        if self.use_opponent_obs:
            input = agent_observations.reshape(batch_shape + (-1,))
        else:
            input = agent_observations[..., 0, :]
        return self.critic(input)
        
        
class SEPPOModel(BaseModel):
    
    def __init__(self,
                 obs_dim, 
                 act_dim,
                 hidden_size,
                 state_embedding_size,
                 discrete: bool = False,
                 actor_logstd: float = None,
                 use_opponent_obs: bool = False,
                 num_agents: bool = 1,
                 sample_dist: str = "normal",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim 
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.state_embedding_size = state_embedding_size
        self.discrete = discrete
        self.actor_logstd = actor_logstd
        self.use_opponent_obs = use_opponent_obs
        self.num_agents = num_agents
        self.sample_dist = sample_dist
        self.device = device
        self.actor = Actor(
            input_size=obs_dim+state_embedding_size,
            act_dim=act_dim,
            hidden_size=hidden_size,
            discrete=discrete,
            actor_logstd=actor_logstd,
            sample_dist = sample_dist,
        )
        self.critic = Critic(
            input_size=(obs_dim+state_embedding_size)*(num_agents if use_opponent_obs else 1),
            hidden_size=hidden_size
        )
        self.state_encoder = StateEncoder(
            obs_dim=obs_dim,
            state_embedding_size=state_embedding_size,
            hidden_size=(obs_dim+state_embedding_size)*2,
        )
        self.state_predictor = StatePredictor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            state_embedding_size=state_embedding_size,
            hidden_size=(obs_dim+state_embedding_size)*2,
        )
        self.to(device)
    
    def compute_actions(self, observations: Union[torch.Tensor, np.ndarray]):
        observations = self._to_tensor(observations)
        with torch.no_grad():
            state_embeddings = self.state_encoder(observations)
            state_embeddings = state_embeddings
        inputs = torch.concat([observations, state_embeddings], dim=-1)
        return self.actor(inputs)
    
    def evaluate_actions(self, 
                         observations: Union[torch.Tensor, np.ndarray],
                         actions: Union[torch.Tensor, np.ndarray],
    ):
        observations = self._to_tensor(observations)
        actions = self._to_tensor(actions)
        with torch.no_grad():
            state_embeddings = self.state_encoder(observations)
        inputs = torch.concat([observations, state_embeddings], dim=-1)
        return self.actor(inputs, actions)

    def compute_values(self, agent_observations: Union[torch.Tensor, np.ndarray]):
        # [..., num_agents, obs_dim]
        # -> [..., 1]
        agent_observations = self._to_tensor(agent_observations) #[..., num_agents, obs_dim]
        with torch.no_grad():
            agent_state_embeddings = self.state_encoder(agent_observations) #[..., num_agents, embedding_size]
        input = torch.concat([agent_observations, agent_state_embeddings], dim=-1)
        batch_shape = input.shape[:-2]
        if self.use_opponent_obs:
            input = input.reshape(batch_shape + (-1,))
        else:
            input = input[..., 0, :]
        return self.critic(input)
        
    def predict_states_and_rewards(self, 
                                   observations: Union[torch.Tensor, np.ndarray],
                                   actions: Union[torch.Tensor, np.ndarray],
    ):
        observations = self._to_tensor(observations)
        actions = self._to_tensor(actions)
        state_embeddings = self.state_encoder(observations)
        return self.state_predictor(state_embeddings, actions)
        

class Policy(nn.Module):
    
    def __init__(self, checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()
        self.device = device
        self.initialized = False
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        else:
            print("Policy is created but with no checkpoint.")
        
    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, self.device)
        else:
            raise ValueError("Failed to load weights from {}! File does not exist!".format(checkpoint_path))
        
        config = state_dict.get("config", None)
        if config is None:
            raise ValueError("Failed to load weights from {}! Config does not exist!".format(checkpoint_path))
        for k, v in config.items():
            if not getattr(self, k, None) == v:
                self.initialized = False
            setattr(self, k, v)
        
        # if initialized, do not create new nn.Module
        if not self.initialized:
            if self.use_representation_learning:
                self.actor = Actor(
                    input_size=self.obs_dim+self.state_embedding_size,
                    act_dim=self.act_dim,
                    hidden_size=self.hidden_size,
                    discrete=self.discrete,
                    actor_logstd=self.actor_logstd,
                    sample_dist=self.sample_dist
                )
                self.state_encoder = StateEncoder(
                    obs_dim=self.obs_dim,
                    state_embedding_size=self.state_embedding_size,
                    hidden_size=(self.obs_dim+self.state_embedding_size)*2,
                )
            else:
                self.actor = Actor(
                    input_size=self.obs_dim,
                    act_dim=self.act_dim,
                    hidden_size=self.hidden_size,
                    discrete=self.discrete,
                    actor_logstd=self.actor_logstd,
                    sample_dist=self.sample_dist
                )
            self.initialized = True
        self.to(self.device)
        self.eval()
        self.load_state_dict(state_dict["model"], strict=False)
        print("Use weights from {} as policy!".format(checkpoint_path))
    
    def _to_tensor(self, arr) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        return arr.to(torch.float32).to(self.device)
    
    def __call__(self, observations: np.ndarray) -> np.ndarray:
        observations = self._to_tensor(observations)
        with torch.no_grad():
            if self.use_representation_learning:
                state_embeddings = self.state_encoder(observations)
                inputs = torch.concat([observations, state_embeddings], dim=-1)
            else:
                inputs = observations
            outputs, _, _ = self.actor(inputs)
        return outputs.cpu().numpy()
            
    def reset(self):
        pass