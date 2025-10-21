"""
Rollout Buffer Module for PPO Training

This module implements an experience buffer that stores trajectories (rollouts) collected
during RL training. The buffer is used to batch experiences for policy updates.

How it works:
- Stores transitions: (state, action, reward, log_prob, value, done)
- After collecting full trajectories, computes advantages using GAE
- Returns batched data ready for PPO update

Implementation approach:
- Simple list-based storage (cleared after each update)
- GAE (Generalized Advantage Estimation) for computing advantages
- Handles variable-length sequences by padding to max length
- Normalizes advantages for stable training
- Returns: Advantage estimates for policy gradient + Return targets for value function
"""

import torch
import numpy as np


class RolloutBuffer:
    """
    Experience buffer for storing and processing trajectories

    Collects transitions during rollouts and computes advantages using GAE
    for PPO training.
    """
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Clear all stored data"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, reward, log_prob, value, done):
        """Add single transition"""
        # Store token_ids from state
        self.states.append(state['token_ids'])
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get_batches(self, gamma=0.99, gae_lambda=0.95):
        """
        Compute advantages using GAE and return as tensors
        
        Returns:
            Dictionary with batched data
        """
        # Convert to numpy
        rewards = np.array(self.rewards)
        values = np.array([v.item() if torch.is_tensor(v) else v for v in self.values])
        dones = np.array(self.dones, dtype=np.float32)
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        # Pad states to same length
        max_len = max(len(s) for s in self.states)
        padded_states = []
        for s in self.states:
            if len(s) < max_len:
                # Pad with zeros
                pad_len = max_len - len(s)
                s_padded = torch.cat([s, torch.zeros(pad_len, dtype=s.dtype)])
            else:
                s_padded = s
            padded_states.append(s_padded)
        
        return {
            'states': torch.stack(padded_states),
            'actions': torch.tensor(self.actions, dtype=torch.long),
            'log_probs': torch.tensor(self.log_probs, dtype=torch.float32),
            'returns': torch.tensor(returns, dtype=torch.float32),
            'advantages': torch.tensor(advantages, dtype=torch.float32),
        }
    
    def __len__(self):
        return len(self.rewards)