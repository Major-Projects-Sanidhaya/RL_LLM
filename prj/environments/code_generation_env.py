"""
Code Generation Environment Module

This module implements a specialized RL environment for code generation tasks.
Unlike the general text generation environment, this one:
- Samples problems from a dataset
- Evaluates generated code against test cases
- Provides task-specific rewards (correctness-based, not fluency-based)

How it works:
- reset() samples a random problem from dataset and tokenizes its prompt
- Agent generates code token-by-token
- Episode ends when max_length reached, EOS generated, or double newline detected
- Final reward computed by running generated code against test cases

Implementation approach:
- Similar to TextGenerationEnvironment but integrates with dataset
- Termination includes code-specific heuristics (double newline = function complete)
- Reward comes from dataset.compute_reward() which executes code
- Used for math problems and code generation (TinyDataset, SimpleMathDataset, CodeProblemDataset)
"""

import torch
from typing import Dict, Tuple, Any
from .text_generation_env import TextGenerationEnvironment


class CodeGenerationEnvironment:
    """
    Specialized RL environment for code/problem-solving generation tasks

    Integrates with dataset classes to provide problems and evaluate solutions.
    """

    def __init__(self, tokenizer, dataset, max_length: int = 512):
        """
        Initialize code generation environment

        Args:
            tokenizer: HuggingFace tokenizer for encoding/decoding
            dataset: Dataset object (TinyDataset, SimpleMathDataset, or CodeProblemDataset)
            max_length: Maximum tokens to generate (code can be longer than text)
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

        # Episode state
        self.current_problem = None  # Problem dict from dataset
        self.current_sequence = []  # Token IDs being generated
        self.step_count = 0  # Tokens generated so far
        self.done = False  # Episode termination flag

        # Special tokens
        self.eos_token_id = tokenizer.eos_token_id
    
    def reset(self) -> Dict:
        """
        Reset environment with a random problem from the dataset

        Unlike text generation env which takes a prompt string,
        this automatically samples from the dataset.

        Returns:
            Initial state dictionary
        """
        # Sample a random problem from dataset
        self.current_problem = self.dataset.get_random_problem()

        # Tokenize the problem prompt
        prompt = self.current_problem['prompt']
        token_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0]

        # Initialize episode state
        self.current_sequence = token_ids.tolist()
        self.step_count = 0
        self.done = False

        return self.get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step by appending a token to the solution

        Args:
            action: Token ID to append

        Returns:
            (next_state, reward, done, info)
        """
        # Append the chosen token
        self.current_sequence.append(action)
        self.step_count += 1

        # Decode current sequence to check for completion heuristics
        text = self.tokenizer.decode(self.current_sequence)

        # Check termination conditions
        self.done = (
            self.step_count >= self.max_length or  # Hit max length
            action == self.eos_token_id or  # Generated EOS token
            '\n\n' in text[-10:]  # Double newline in last 10 chars (code completion heuristic)
        )

        # Compute reward
        if self.done:
            # Episode complete - extract generated answer and evaluate it
            # Remove the original prompt tokens to get only the generated part
            prompt_length = len(self.tokenizer.encode(self.current_problem['prompt']))
            generated_text = self.tokenizer.decode(
                self.current_sequence[prompt_length:]
            )
            # Use dataset-specific reward function (runs tests for code, checks answer for math)
            reward = self.dataset.compute_reward(generated_text, self.current_problem)
        else:
            # Small negative step reward to encourage brevity
            reward = -0.01

        next_state = self.get_state()

        # Info for logging/debugging
        info = {
            'text': text,
            'length': len(self.current_sequence),
            'problem_id': self.current_problem['task_id']
        }

        return next_state, reward, self.done, info

    def get_state(self) -> Dict:
        """
        Get current environment state

        Returns:
            Dictionary with token_ids, step count, and done flag
        """
        return {
            'token_ids': torch.tensor(self.current_sequence),
            'step': self.step_count,
            'done': self.done
        }