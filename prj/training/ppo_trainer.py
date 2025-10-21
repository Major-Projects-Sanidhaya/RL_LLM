"""
PPO Trainer Module

This module implements the Proximal Policy Optimization (PPO) algorithm for training
the policy and value networks. PPO is a popular policy gradient method that uses
clipping to ensure stable updates.

How it works:
- Collects trajectories using current policy
- Computes advantages using GAE (done in buffer)
- Updates policy using clipped surrogate objective
- Updates value function using MSE loss
- Multiple epochs of updates on same data for sample efficiency

Implementation approach:
- Policy loss: Clipped surrogate objective prevents large policy changes
- Value loss: MSE between predicted and actual returns
- Entropy bonus: Encourages exploration
- Gradient clipping: Prevents exploding gradients
- Separate optimizers for policy and value networks
- Based on "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SimplePPOTrainer:
    """
    PPO trainer for policy and value network updates

    Implements clipped PPO objective with entropy regularization.
    """
    
    def __init__(self, policy, value_net, lr=3e-4, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        self.policy = policy
        self.value_net = value_net
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Optimizers
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
        
        self.device = next(policy.parameters()).device
    
    def compute_policy_loss(self, states, actions, old_log_probs, advantages):
        """
        Compute clipped PPO policy loss
        """
        # Get current policy distribution
        dist = self.policy.get_action_distribution(states)
        
        # New log probs
        new_log_probs = dist.log_prob(actions)
        
        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        # Policy loss
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Entropy bonus (encourage exploration)
        entropy = dist.entropy().mean()
        
        return policy_loss, entropy
    
    def compute_value_loss(self, states, returns):
        """
        Compute value function loss
        """
        predicted_values = self.value_net(states).squeeze(-1)
        value_loss = nn.functional.mse_loss(predicted_values, returns)
        return value_loss
    
    def update(self, buffer, epochs=4):
        """
        Update policy using PPO
        
        Args:
            buffer: RolloutBuffer with collected trajectories
            epochs: Number of optimization epochs
        """
        # Get batches
        batches = buffer.get_batches()
        
        states = batches['states'].to(self.device)
        actions = batches['actions'].to(self.device)
        old_log_probs = batches['log_probs'].to(self.device)
        returns = batches['returns'].to(self.device)
        advantages = batches['advantages'].to(self.device)
        
        # Multiple epochs of updates
        for epoch in range(epochs):
            # Compute losses
            policy_loss, entropy = self.compute_policy_loss(
                states, actions, old_log_probs, advantages
            )
            value_loss = self.compute_value_loss(states, returns)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update policy
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }