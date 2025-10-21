"""
Evaluation Metrics Module

This module provides an evaluator class for assessing trained policy performance.
Runs test episodes and computes metrics like success rate and average reward.

How it works:
- Runs policy on test problems from the dataset
- Collects metrics: average reward, success rate, generation length
- Can generate sample outputs for qualitative inspection
- All evaluation done with gradients disabled for efficiency

Implementation approach:
- Uses torch.no_grad() to disable gradient computation during eval
- Sets policy to eval mode (disables dropout, etc.)
- Samples actions from policy distribution (not argmax)
- Computes aggregate statistics across multiple test episodes
- Returns dict of metrics for logging/monitoring
"""

import torch
from typing import List, Dict


class Evaluator:
    """
    Evaluator for policy performance assessment

    Runs test episodes and computes metrics like success rate and average reward.
    """
    
    def __init__(self, dataset, env, policy, tokenizer):
        self.dataset = dataset
        self.env = env
        self.policy = policy
        self.tokenizer = tokenizer
        self.device = next(policy.parameters()).device
    
    @torch.no_grad()
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate policy on test problems
        
        Returns:
            Dictionary with metrics
        """
        self.policy.eval()
        total_reward = 0
        successes = 0
        episode_lengths = []
        
        for _ in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get action from policy
                token_ids = state['token_ids'].unsqueeze(0).to(self.device)
                dist = self.policy.get_action_distribution(token_ids)
                action = dist.sample().item()
                
                # Take step
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
            episode_lengths.append(info['length'])
            
            # Check if successful (positive reward)
            if episode_reward > 0:
                successes += 1
        
        self.policy.train()
        
        return {
            'avg_reward': total_reward / num_episodes,
            'success_rate': successes / num_episodes,
            'avg_length': np.mean(episode_lengths)
        }
    
    def generate_samples(self, num_samples: int = 5) -> List[str]:
        """Generate sample outputs for inspection"""
        self.policy.eval()
        samples = []
        
        for _ in range(num_samples):
            state = self.env.reset()
            done = False
            
            while not done:
                token_ids = state['token_ids'].unsqueeze(0).to(self.device)
                dist = self.policy.get_action_distribution(token_ids)
                action = dist.sample().item()
                state, _, done, info = self.env.step(action)
            
            samples.append(info['text'])
        
        self.policy.train()
        return samples