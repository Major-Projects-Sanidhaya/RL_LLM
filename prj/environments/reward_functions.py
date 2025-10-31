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
    Reward function with multiple components (Week 5 Implementation)

    Combines multiple reward signals for more sophisticated training.
    Implements the four key components from the research plan:
    - Fluency: How natural does the text sound?
    - Coherence: Does the text make logical sense?
    - Task Completion: Does it accomplish the intended goal?
    - Safety: Is the content appropriate and non-harmful?
    """

    def __init__(self, tokenizer=None, task_evaluator=None):
        """
        Initialize multi-component reward with fluency scorer

        Args:
            tokenizer: Tokenizer for decoding text
            task_evaluator: Optional task-specific evaluator function
        """
        self.fluency_reward = SimpleRewardFunction()
        self.tokenizer = tokenizer
        self.task_evaluator = task_evaluator

        # Unsafe words/patterns for safety check
        self.unsafe_patterns = [
            'kill', 'hate', 'violence', 'attack', 'bomb', 'weapon',
            'racist', 'sexist', 'discriminat'
        ]

    def compute_fluency(self, token_ids: torch.Tensor) -> float:
        """
        Compute fluency reward using language model perplexity

        Returns:
            Float reward for fluency (higher = more fluent)
        """
        return self.fluency_reward.compute_reward(token_ids)

    def compute_coherence(self, token_ids: torch.Tensor, text: str = None) -> float:
        """
        Compute coherence reward based on text structure

        Coherence measures whether the text makes logical sense:
        - Repetition penalty: Penalize repeated tokens
        - Length consistency: Ensure reasonable length
        - Token diversity: Reward diverse vocabulary usage

        Args:
            token_ids: Tensor of token IDs
            text: Decoded text (optional, will decode if not provided)

        Returns:
            Float reward for coherence
        """
        if len(token_ids) < 2:
            return 0.0

        # Check for excessive repetition
        token_list = token_ids.tolist() if torch.is_tensor(token_ids) else token_ids
        unique_tokens = len(set(token_list))
        total_tokens = len(token_list)

        # Diversity ratio (higher is better)
        diversity_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0.0

        # Reward diversity, penalize repetition
        coherence_score = diversity_ratio * 0.2

        # Check for immediate repetition (same token repeated)
        immediate_repeats = sum(1 for i in range(len(token_list) - 1)
                               if token_list[i] == token_list[i+1])
        repetition_penalty = -0.05 * immediate_repeats

        return coherence_score + repetition_penalty

    def compute_task_completion(self, token_ids: torch.Tensor, text: str = None,
                                prompt: str = None, target: str = None) -> float:
        """
        Compute task completion reward

        Evaluates whether the generated text accomplishes the intended goal.
        Can use custom task evaluator or basic heuristics.

        Args:
            token_ids: Tensor of token IDs
            text: Decoded generated text
            prompt: Original prompt/question
            target: Expected answer/output (if available)

        Returns:
            Float reward for task completion
        """
        # If custom task evaluator provided, use it
        if self.task_evaluator is not None:
            return self.task_evaluator(text, prompt, target)

        # Basic heuristic: reasonable length and ends properly
        score = 0.0

        # Reward appropriate length (not too short, not too long)
        length = len(token_ids)
        if 5 <= length <= 100:
            score += 0.1
        elif length < 5:
            score -= 0.2  # Too short
        elif length > 200:
            score -= 0.1  # Too long

        return score

    def compute_safety(self, token_ids: torch.Tensor, text: str = None) -> float:
        """
        Compute safety reward

        Penalizes unsafe, harmful, or inappropriate content.

        Args:
            token_ids: Tensor of token IDs
            text: Decoded text (will decode if not provided)

        Returns:
            Float reward for safety (0 if safe, negative if unsafe)
        """
        # Decode text if not provided
        if text is None and self.tokenizer is not None:
            text = self.tokenizer.decode(token_ids)

        if text is None:
            return 0.0  # Can't evaluate without text

        text_lower = text.lower()

        # Check for unsafe patterns
        for pattern in self.unsafe_patterns:
            if pattern in text_lower:
                return -1.0  # Strong penalty for unsafe content

        return 0.0  # No safety issues detected

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

    def compute_reward(self, token_ids: torch.Tensor, text: str = None,
                      prompt: str = None, target: str = None) -> float:
        """
        Combine multiple reward components into single reward signal (Week 5)

        Implements the four core components:
        1. Fluency: How natural does the text sound?
        2. Coherence: Does the text make logical sense?
        3. Task Completion: Does it accomplish the intended goal?
        4. Safety: Is the content appropriate and non-harmful?

        Args:
            token_ids: Tensor of token IDs
            text: Decoded text (optional)
            prompt: Original prompt (optional, for task completion)
            target: Target answer (optional, for task completion)

        Returns:
            Combined reward score
        """
        # Decode text if needed and tokenizer available
        if text is None and self.tokenizer is not None:
            text = self.tokenizer.decode(token_ids)

        # Get individual components
        fluency = self.compute_fluency(token_ids)
        coherence = self.compute_coherence(token_ids, text)
        task_completion = self.compute_task_completion(token_ids, text, prompt, target)
        safety = self.compute_safety(token_ids, text)

        # Weighted combination
        # Safety is critical (can veto entire output)
        # Fluency is most important for language quality
        # Coherence ensures logical consistency
        # Task completion ensures usefulness
        total_reward = (
            0.4 * fluency +
            0.2 * coherence +
            0.3 * task_completion +
            0.1 * safety
        )

        # If safety is negative, apply strong penalty
        if safety < 0:
            total_reward += safety * 5.0  # Amplify safety penalty

        return total_reward