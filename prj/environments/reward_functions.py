"""
Reward Functions for Text Generation

This module provides reward functions that score the quality of generated text.
Used in RL training to provide feedback signals to the policy network.

How it works:
- SimpleRewardFunction uses a pre-trained GPT-2 model to compute fluency scores
- Fluency is measured by language model perplexity (lower = better)
- MultiComponentReward combines multiple reward signals (fluency + length penalty)

Implementation approach:
- Loads a frozen GPT-2 model as a reward evaluator
- Computes cross-entropy loss on generated sequences (perplexity proxy)
- Converts loss to reward signal (negative loss scaled appropriately)
- Can be extended with additional reward components (diversity, task-specific metrics, etc.)
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class SimpleRewardFunction:
    """
    Simple reward based on language model fluency

    Uses a small pre-trained GPT-2 model to score text quality based on
    how "natural" or "fluent" the generated text appears.
    """

    def __init__(self, model_name: str = 'gpt2'):
        """
        Initialize reward function with a pre-trained language model

        Args:
            model_name: HuggingFace model name (default: 'gpt2' - smallest variant)
        """
        print(f"Loading reward model: {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained GPT-2 for scoring
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode (no training)

        # Set pad token (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_fluency_reward(self, token_ids: torch.Tensor) -> float:
        """
        Compute fluency reward using language model perplexity

        Lower perplexity (lower cross-entropy loss) indicates more fluent text,
        which translates to higher reward.

        Args:
            token_ids: Tensor of token IDs representing the generated text

        Returns:
            Float reward value (higher = more fluent)
        """
        if len(token_ids) < 2:
            return 0.0  # Can't compute perplexity for very short sequences

        with torch.no_grad():  # No gradients needed for reward model
            # Ensure proper shape [batch_size, seq_len]
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)

            token_ids = token_ids.to(self.device)

            # Get model predictions (compute cross-entropy loss)
            outputs = self.model(token_ids, labels=token_ids)
            loss = outputs.loss.item()

            # Convert loss to reward (lower loss = better)
            # Negative sign because we want to maximize reward, minimize loss
            # Scale by 0.1 to keep rewards in reasonable range
            reward = -loss * 0.1

            return reward

    def compute_reward(self, token_ids: torch.Tensor) -> float:
        """
        Main reward function interface

        Currently just returns fluency reward, but can be extended
        to combine multiple reward components.

        Args:
            token_ids: Tensor of token IDs

        Returns:
            Overall reward score
        """
        return self.compute_fluency_reward(token_ids)


class MultiComponentReward:
    """
    Reward function with multiple components

    Combines multiple reward signals for more sophisticated training.
    Currently includes fluency + length penalty, but can be extended
    with diversity rewards, task-specific metrics, etc.
    """

    def __init__(self):
        """Initialize multi-component reward with fluency scorer"""
        self.fluency_reward = SimpleRewardFunction()

    def compute_length_penalty(self, token_ids: torch.Tensor, target_length: int = 20) -> float:
        """
        Penalize sequences that deviate from target length

        Encourages the model to generate responses of appropriate length
        (not too short, not too long).

        Args:
            token_ids: Tensor of token IDs
            target_length: Desired sequence length

        Returns:
            Penalty value (0 at target length, increasingly negative as length differs)
        """
        length = len(token_ids)
        diff = abs(length - target_length)
        # Linear penalty: -0.01 per token away from target
        return -0.01 * diff

    def compute_reward(self, token_ids: torch.Tensor) -> float:
        """
        Combine multiple reward components into single reward signal

        Current components:
        - Fluency reward (from GPT-2 perplexity)
        - Length penalty (encourages appropriate length)

        Args:
            token_ids: Tensor of token IDs

        Returns:
            Combined reward score
        """
        # Get individual components
        fluency = self.fluency_reward.compute_reward(token_ids)
        length_penalty = self.compute_length_penalty(token_ids)

        # Weighted combination (currently equal weights)
        total_reward = fluency + length_penalty

        return total_reward