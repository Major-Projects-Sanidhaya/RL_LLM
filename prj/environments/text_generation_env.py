"""
Text Generation Environment Module

This module implements a gym-style RL environment for autoregressive text generation.
The agent generates text one token at a time, receiving rewards based on fluency.

How it works:
- Environment maintains current token sequence as state
- Agent selects next token as action (from vocabulary)
- Reward is computed using reward_fn (typically based on fluency)
- Episode terminates when max_length is reached or EOS token is generated

Implementation approach:
- State: Dictionary containing current token sequence, step count, and done flag
- Action: Integer token ID to append to sequence
- Reward: Small negative per step (brevity), large positive/negative at end (fluency)
- Follows standard RL interface: reset() and step(action) -> (state, reward, done, info)
"""

import torch
from typing import Dict, Tuple, Any, Optional


class TextGenerationEnvironment:
    """
    Reinforcement learning environment for autoregressive text generation

    Implements gym-style interface for token-by-token text generation.
    """

    def __init__(self, tokenizer, reward_fn, max_length: int = 50):
        """
        Initialize text generation environment

        Args:
            tokenizer: HuggingFace tokenizer for encoding/decoding text
            reward_fn: Reward function to evaluate generated sequences
            max_length: Maximum number of tokens to generate before episode ends
        """
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.max_length = max_length

        # Episode state variables
        self.current_sequence = []  # List of token IDs generated so far
        self.step_count = 0  # Number of tokens generated in current episode
        self.done = False  # Whether episode has terminated

        # Special tokens
        self.eos_token_id = tokenizer.eos_token_id
    
    def reset(self, prompt_text: str) -> Dict:
        """
        Reset environment with new prompt to start a fresh episode

        Args:
            prompt_text: Initial prompt as string (e.g., "Once upon a time")

        Returns:
            Initial state dictionary containing token_ids, step count, and done flag
        """
        # Tokenize the prompt text into token IDs
        token_ids = self.tokenizer.encode(prompt_text, return_tensors='pt')[0]

        # Initialize episode state
        self.current_sequence = token_ids.tolist()
        self.step_count = 0
        self.done = False

        return self.get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment by appending a token

        This is the core RL interaction: agent chooses action (token),
        environment returns next state, reward, and whether episode is done.

        Args:
            action: Token ID to append to current sequence

        Returns:
            Tuple of (next_state, reward, done, info):
            - next_state: Dictionary with updated token sequence
            - reward: Float reward signal for this step
            - done: Boolean indicating if episode has terminated
            - info: Additional information (decoded text, length, etc.)
        """
        # Append the chosen token to sequence
        self.current_sequence.append(action)
        self.step_count += 1

        # Check termination conditions
        self.done = (
            self.step_count >= self.max_length or  # Hit max length
            action == self.eos_token_id  # Generated end-of-sequence token
        )

        # Compute reward
        if self.done:
            # Episode finished - compute final reward based on complete sequence
            token_tensor = torch.tensor(self.current_sequence)
            reward = self.reward_fn.compute_reward(token_tensor)
        else:
            # Step reward: small negative to encourage brevity (shorter generations)
            reward = -0.01

        # Get next state
        next_state = self.get_state()

        # Additional info for debugging/logging
        info = {
            'text': self.tokenizer.decode(self.current_sequence),
            'length': len(self.current_sequence)
        }

        return next_state, reward, self.done, info

    def get_state(self) -> Dict:
        """
        Get current environment state

        Returns:
            Dictionary with:
            - token_ids: Tensor of current token sequence
            - step: Number of generation steps taken
            - done: Whether episode is finished
        """
        return {
            'token_ids': torch.tensor(self.current_sequence),
            'step': self.step_count,
            'done': self.done
        }

    def decode_sequence(self) -> str:
        """
        Decode current token sequence to human-readable text

        Returns:
            String representation of current sequence
        """
        return self.tokenizer.decode(self.current_sequence)