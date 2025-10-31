"""
Hierarchical PPO Trainer (Week 6 Implementation)

This module implements PPO training for the hierarchical two-level policy:
- High-level policy: Learns to generate semantic intentions
- Low-level policy: Learns to generate tokens conditioned on intentions

Training approach:
- Both levels are trained simultaneously using PPO
- High-level receives episode-level rewards
- Low-level receives step-level and episode-level rewards
- Uses separate value functions for each level
- Intention sampling uses reparameterization trick for gradients

Based on hierarchical RL and options framework literature.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple


class HierarchicalPPOTrainer:
    """
    PPO trainer for hierarchical two-level policy

    Trains both high-level (intention) and low-level (token) policies
    simultaneously using modified PPO objectives.
    """

    def __init__(
        self,
        hierarchical_policy,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        intention_coef: float = 0.1
    ):
        """
        Initialize hierarchical PPO trainer

        Args:
            hierarchical_policy: HierarchicalPolicy network
            lr: Learning rate
            clip_ratio: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            intention_coef: Weight for intention diversity
        """
        self.policy = hierarchical_policy
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.intention_coef = intention_coef

        # Single optimizer for entire hierarchical policy
        self.optimizer = optim.Adam(hierarchical_policy.parameters(), lr=lr)

        self.device = next(hierarchical_policy.parameters()).device

    def compute_hierarchical_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_intentions: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute policy loss for hierarchical system

        Computes losses for both:
        1. Low-level policy (token generation given intention)
        2. High-level policy (intention generation)

        Args:
            states: Token sequences [batch, seq_len]
            actions: Selected tokens [batch]
            old_log_probs: Log probs from old policy [batch]
            old_intentions: Intentions from old policy [batch, intention_dim]
            advantages: Computed advantages [batch]

        Returns:
            Tuple of (policy_loss, entropy, intention_loss)
        """
        # Get current policy outputs with intention information
        outputs = self.policy(states, return_intention=True)

        # LOW-LEVEL POLICY LOSS (token generation)
        # Recompute action distribution with OLD intentions to maintain consistency
        action_dist = torch.distributions.Categorical(logits=outputs['logits'])
        new_log_probs = action_dist.log_prob(actions)

        # PPO clipped objective for token policy
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        low_level_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        # Token policy entropy (encourage exploration)
        token_entropy = action_dist.entropy().mean()

        # HIGH-LEVEL POLICY LOSS (intention generation)
        # KL divergence between old and new intention distributions
        # This prevents intentions from changing too rapidly
        if outputs['intention_dist'] is not None:
            # Sample new intentions
            new_intentions = outputs['intention']

            # Compute log prob of old intentions under new distribution
            old_intention_log_prob = outputs['intention_dist'].log_prob(old_intentions).sum(dim=-1)

            # Regularization: encourage intention diversity
            intention_entropy = outputs['intention_dist'].entropy().sum(dim=-1).mean()

            # High-level loss: keep intentions stable but diverse
            high_level_loss = -old_intention_log_prob.mean()
        else:
            high_level_loss = torch.tensor(0.0, device=self.device)
            intention_entropy = torch.tensor(0.0, device=self.device)

        # Combined policy loss
        policy_loss = low_level_loss + self.intention_coef * high_level_loss

        # Total entropy (both levels)
        total_entropy = token_entropy + 0.1 * intention_entropy

        return policy_loss, total_entropy, high_level_loss

    def compute_hierarchical_value_loss(
        self,
        states: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function loss for both levels

        Args:
            states: Token sequences [batch, seq_len]
            returns: Computed returns [batch]

        Returns:
            Combined value loss
        """
        # Get value estimates from both levels
        outputs = self.policy(states, return_intention=True)

        high_value = outputs['high_value'].squeeze(-1)
        low_value = outputs['low_value'].squeeze(-1)

        # Both value functions should predict the same returns
        # (they're at different abstraction levels but same ultimate reward)
        high_value_loss = nn.functional.mse_loss(high_value, returns)
        low_value_loss = nn.functional.mse_loss(low_value, returns)

        # Average the two value losses
        total_value_loss = (high_value_loss + low_value_loss) / 2.0

        return total_value_loss

    def update(self, buffer, epochs: int = 4) -> Dict[str, float]:
        """
        Update hierarchical policy using PPO

        Args:
            buffer: RolloutBuffer with collected trajectories
                    Must include 'intentions' field for hierarchical training
            epochs: Number of optimization epochs

        Returns:
            Dictionary of training statistics
        """
        # Get batches from buffer
        batches = buffer.get_batches()

        states = batches['states'].to(self.device)
        actions = batches['actions'].to(self.device)
        old_log_probs = batches['log_probs'].to(self.device)
        returns = batches['returns'].to(self.device)
        advantages = batches['advantages'].to(self.device)

        # Get old intentions (if available, otherwise sample fresh)
        if 'intentions' in batches:
            old_intentions = batches['intentions'].to(self.device)
        else:
            # Sample intentions from current policy
            with torch.no_grad():
                outputs = self.policy(states, return_intention=True)
                old_intentions = outputs['intention']

        # Training statistics
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'intention_loss': 0.0
        }

        # Multiple epochs of updates
        for epoch in range(epochs):
            # Compute losses
            policy_loss, entropy, intention_loss = self.compute_hierarchical_policy_loss(
                states, actions, old_log_probs, old_intentions, advantages
            )

            value_loss = self.compute_hierarchical_value_loss(states, returns)

            # Total loss
            total_loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
            )

            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

            self.optimizer.step()

            # Accumulate stats
            stats['policy_loss'] += policy_loss.item()
            stats['value_loss'] += value_loss.item()
            stats['entropy'] += entropy.item()
            stats['intention_loss'] += intention_loss.item()

        # Average over epochs
        for key in stats:
            stats[key] /= epochs

        return stats

    def collect_episode_with_intentions(
        self,
        env,
        buffer,
        max_steps: int = 100
    ) -> float:
        """
        Collect a full episode storing both actions and intentions

        This is a helper method for collecting trajectories with the
        hierarchical policy, ensuring intentions are stored in buffer.

        Args:
            env: Environment to interact with
            buffer: Buffer to store transitions
            max_steps: Maximum steps per episode

        Returns:
            Total episode reward
        """
        state = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            # Get token_ids and move to device
            token_ids = state['token_ids'].unsqueeze(0).to(self.device)

            # Get action AND intention from hierarchical policy
            with torch.no_grad():
                outputs = self.policy(token_ids, return_intention=True)

                # Sample action from low-level policy
                action_dist = torch.distributions.Categorical(logits=outputs['logits'])
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                # Get intention from high-level policy
                intention = outputs['intention']

                # Get value estimates (use low-level for step-by-step)
                value = outputs['low_value']

            # Take step in environment
            next_state, reward, done, info = env.step(action.item())

            # Store in buffer WITH intention
            buffer.add(
                state=state,
                action=action.item(),
                reward=reward,
                log_prob=log_prob.item(),
                value=value.item(),
                done=done,
                intention=intention.cpu()  # Store intention for training
            )

            state = next_state
            episode_reward += reward
            steps += 1

        return episode_reward
