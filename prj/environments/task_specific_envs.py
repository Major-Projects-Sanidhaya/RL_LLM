"""
Task-Specific Environments (Week 5 Implementation)

This module implements specialized RL environments for different language tasks:
- QuestionAnsweringEnvironment: For Q&A tasks
- ConversationEnvironment: For dialogue/conversation tasks
- SummarizationEnvironment: For text summarization (future work)

Each environment provides task-specific:
- State representation
- Reward signals
- Termination conditions
- Evaluation metrics
"""

import torch
from typing import Dict, Tuple, Any, List, Optional


class QuestionAnsweringEnvironment:
    """
    Specialized environment for Question Answering tasks

    Designed for simple Q&A where:
    - Input: A question
    - Output: A concise answer
    - Reward: Based on correctness and quality
    """

    def __init__(self, tokenizer, dataset=None, max_length: int = 100):
        """
        Initialize Q&A environment

        Args:
            tokenizer: HuggingFace tokenizer
            dataset: Optional dataset with Q&A pairs
            max_length: Maximum answer length in tokens
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

        # Episode state
        self.current_question = None
        self.expected_answer = None
        self.current_sequence = []
        self.step_count = 0
        self.done = False

        # Special tokens
        self.eos_token_id = tokenizer.eos_token_id
        self.question_mark_id = tokenizer.encode('?')[0] if '?' in tokenizer.get_vocab() else None

    def reset(self, question: str = None, answer: str = None) -> Dict:
        """
        Reset environment with a new question

        Args:
            question: Question text (if None, sample from dataset)
            answer: Expected answer (if available)

        Returns:
            Initial state dictionary
        """
        # Get question from dataset if not provided
        if question is None and self.dataset is not None:
            problem = self.dataset.get_random_problem()
            question = problem.get('question', problem.get('prompt', ''))
            answer = problem.get('answer', problem.get('expected', None))
        elif question is None:
            question = "What is 2+2?"  # Default question
            answer = "4"

        self.current_question = question
        self.expected_answer = answer

        # Tokenize question as prompt
        prompt_with_qa_format = f"Question: {question}\nAnswer:"
        token_ids = self.tokenizer.encode(prompt_with_qa_format, return_tensors='pt')[0]

        # Initialize state
        self.current_sequence = token_ids.tolist()
        self.step_count = 0
        self.done = False

        return self.get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Generate next token in the answer

        Args:
            action: Token ID to append

        Returns:
            (next_state, reward, done, info)
        """
        # Append token
        self.current_sequence.append(action)
        self.step_count += 1

        # Decode to check for termination
        text = self.tokenizer.decode(self.current_sequence)

        # Q&A specific termination conditions
        self.done = (
            self.step_count >= self.max_length or
            action == self.eos_token_id or
            '\n' in text[-5:]  # Newline indicates end of answer
        )

        # Compute reward
        if self.done:
            # Extract generated answer (remove prompt)
            prompt_length = len(self.tokenizer.encode(f"Question: {self.current_question}\nAnswer:"))
            answer_tokens = self.current_sequence[prompt_length:]
            generated_answer = self.tokenizer.decode(answer_tokens).strip()

            # Compute Q&A specific reward
            reward = self._compute_qa_reward(generated_answer)
        else:
            # Small step penalty
            reward = -0.01

        next_state = self.get_state()

        info = {
            'text': text,
            'question': self.current_question,
            'expected_answer': self.expected_answer,
            'length': len(self.current_sequence)
        }

        return next_state, reward, self.done, info

    def _compute_qa_reward(self, generated_answer: str) -> float:
        """
        Compute reward for Q&A task

        Combines:
        - Correctness (if expected answer available)
        - Conciseness (prefer short, direct answers)
        - Completeness (not too short)
        """
        reward = 0.0

        # If we have expected answer, check correctness
        if self.expected_answer is not None:
            expected_lower = self.expected_answer.lower().strip()
            generated_lower = generated_answer.lower().strip()

            # Exact match
            if expected_lower == generated_lower:
                reward += 10.0
            # Substring match (answer contains expected)
            elif expected_lower in generated_lower:
                reward += 7.0
            # Expected in answer
            elif generated_lower in expected_lower:
                reward += 5.0
            else:
                # No match
                reward -= 1.0

        # Length penalties/rewards
        answer_length = len(generated_answer.split())
        if answer_length < 1:
            reward -= 5.0  # Empty answer
        elif 1 <= answer_length <= 10:
            reward += 1.0  # Good concise answer
        elif answer_length > 50:
            reward -= 2.0  # Too verbose

        return reward

    def get_state(self) -> Dict:
        """Get current environment state"""
        return {
            'token_ids': torch.tensor(self.current_sequence),
            'step': self.step_count,
            'done': self.done,
            'question': self.current_question
        }


class ConversationEnvironment:
    """
    Specialized environment for conversational dialogue

    Designed for multi-turn conversations where:
    - Input: Conversation history + user message
    - Output: Appropriate response
    - Reward: Based on relevance, coherence, and engagement
    """

    def __init__(self, tokenizer, max_length: int = 150, max_turns: int = 5):
        """
        Initialize conversation environment

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum response length
            max_turns: Maximum conversation turns
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_turns = max_turns

        # Conversation state
        self.conversation_history = []
        self.current_user_message = None
        self.current_sequence = []
        self.step_count = 0
        self.done = False
        self.turn_count = 0

        # Special tokens
        self.eos_token_id = tokenizer.eos_token_id

    def reset(self, user_message: str = None, history: List[str] = None) -> Dict:
        """
        Reset environment with new conversation turn

        Args:
            user_message: Current user message (if None, use default)
            history: Previous conversation turns

        Returns:
            Initial state dictionary
        """
        # Set conversation history
        self.conversation_history = history if history is not None else []

        # Use default message if not provided
        if user_message is None:
            user_message = "Hello! How are you?"

        self.current_user_message = user_message

        # Build prompt with conversation context
        prompt = self._build_conversation_prompt()
        token_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0]

        # Initialize state
        self.current_sequence = token_ids.tolist()
        self.step_count = 0
        self.done = False
        self.turn_count = len(self.conversation_history) // 2

        return self.get_state()

    def _build_conversation_prompt(self) -> str:
        """Build prompt from conversation history"""
        prompt_parts = []

        # Add history
        for i, msg in enumerate(self.conversation_history):
            role = "User" if i % 2 == 0 else "Assistant"
            prompt_parts.append(f"{role}: {msg}")

        # Add current user message
        prompt_parts.append(f"User: {self.current_user_message}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Generate next token in response

        Args:
            action: Token ID to append

        Returns:
            (next_state, reward, done, info)
        """
        # Append token
        self.current_sequence.append(action)
        self.step_count += 1

        # Decode current text
        text = self.tokenizer.decode(self.current_sequence)

        # Conversation-specific termination
        self.done = (
            self.step_count >= self.max_length or
            action == self.eos_token_id or
            '\n' in text[-5:]  # Newline ends response
        )

        # Compute reward
        if self.done:
            # Extract response
            prompt_length = len(self.tokenizer.encode(self._build_conversation_prompt()))
            response_tokens = self.current_sequence[prompt_length:]
            generated_response = self.tokenizer.decode(response_tokens).strip()

            # Compute conversation reward
            reward = self._compute_conversation_reward(generated_response)
        else:
            reward = -0.01

        next_state = self.get_state()

        info = {
            'text': text,
            'user_message': self.current_user_message,
            'history': self.conversation_history,
            'length': len(self.current_sequence)
        }

        return next_state, reward, self.done, info

    def _compute_conversation_reward(self, response: str) -> float:
        """
        Compute reward for conversational response

        Evaluates:
        - Length appropriateness
        - Not just repeating user message
        - Contains substantive content
        """
        reward = 0.0

        # Length check
        response_length = len(response.split())
        if response_length < 2:
            reward -= 5.0  # Too short
        elif 3 <= response_length <= 50:
            reward += 2.0  # Good length
        elif response_length > 100:
            reward -= 1.0  # Too long

        # Check for repetition of user message
        user_lower = self.current_user_message.lower()
        response_lower = response.lower()

        if user_lower == response_lower:
            reward -= 3.0  # Just repeating user

        # Check for basic conversation markers (questions, acknowledgments)
        if any(marker in response_lower for marker in ['?', 'thank', 'hello', 'yes', 'no']):
            reward += 0.5

        # Penalize empty or very short responses
        if len(response.strip()) < 5:
            reward -= 5.0

        return reward

    def get_state(self) -> Dict:
        """Get current environment state"""
        return {
            'token_ids': torch.tensor(self.current_sequence),
            'step': self.step_count,
            'done': self.done,
            'conversation_history': self.conversation_history,
            'user_message': self.current_user_message
        }


class SummarizationEnvironment:
    """
    Environment for text summarization tasks (Future Week 7-8)

    Placeholder for future implementation.
    """

    def __init__(self, tokenizer, max_summary_length: int = 100):
        """
        Initialize summarization environment

        Args:
            tokenizer: HuggingFace tokenizer
            max_summary_length: Maximum summary length
        """
        self.tokenizer = tokenizer
        self.max_summary_length = max_summary_length

    def reset(self, source_text: str = None) -> Dict:
        """Reset with new text to summarize"""
        raise NotImplementedError("Summarization environment - Week 7-8")

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Generate next token in summary"""
        raise NotImplementedError("Summarization environment - Week 7-8")
