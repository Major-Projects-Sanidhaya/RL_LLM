"""
RL-LLM: Hierarchical Code Generation Training Script
Adapted from RL_LLM_Colab_Notebook.ipynb for Gilbreth cluster execution
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import random
import math
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Any
from io import StringIO
import json
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP

# Import LLM reward evaluator
try:
    from llm_reward_evaluator import LLMRewardEvaluator, HybridRewardFunction
    LLM_REWARDS_AVAILABLE = True
except ImportError:
    print("Warning: llm_reward_evaluator not found. LLM rewards will not be available.")
    LLM_REWARDS_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train hierarchical RL model for code generation')
    parser.add_argument('--dataset', type=str, default='humaneval',
                       choices=['humaneval', 'stack', 'codechain', 'redpajama'],
                       help='Dataset to use for training')
    parser.add_argument('--num_iterations', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--episodes_per_iter', type=int, default=6,
                       help='Number of episodes per iteration')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=384,
                       help='Model dimension')
    parser.add_argument('--intention_dim', type=int, default=64,
                       help='Intention vector dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=20,
                       help='Logging interval')
    parser.add_argument('--subset_size', type=int, default=20,
                       help='Subset size for HumanEval dataset')

    # LLM reward parameters
    parser.add_argument('--use_llm_rewards', action='store_true',
                       help='Use LLM for reward evaluation')
    parser.add_argument('--llm_backend', type=str, default='ollama',
                       choices=['ollama', 'huggingface', 'gpt', 'claude'],
                       help='LLM backend to use (ollama/huggingface are FREE, gpt/claude are paid)')
    parser.add_argument('--llm_model', type=str, default='llama3.2:3b',
                       help='Model name for LLM backend')
    parser.add_argument('--llm_device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for HuggingFace models')
    parser.add_argument('--ollama_url', type=str, default='http://localhost:11434',
                       help='URL for Ollama API server')
    parser.add_argument('--llm_weight', type=float, default=0.7,
                       help='Weight for LLM score in final reward (0-1)')
    parser.add_argument('--llm_threshold', type=float, default=5.0,
                       help='Only use LLM for code with heuristic reward above this')
    parser.add_argument('--reward_cache_dir', type=str, default='./reward_cache',
                       help='Directory to cache LLM reward evaluations')

    # Legacy support (deprecated)
    parser.add_argument('--use_gpt', action='store_true',
                       help='[DEPRECATED] Use --llm_backend=gpt instead')
    parser.add_argument('--use_claude', action='store_true',
                       help='[DEPRECATED] Use --llm_backend=claude instead')

    return parser.parse_args()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get appropriate device (CUDA if available)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(model, optimizer, iteration, best_reward, history, args, filename):
    """Save training checkpoint"""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_reward': best_reward,
        'training_history': history,
        'args': vars(args)
    }
    torch.save(checkpoint, filename)
    print(f"  ✓ Checkpoint saved: {filename}")

# ============================================================================
# DATASET LOADERS
# ============================================================================

class HumanEvalDataset:
    """HumanEval dataset for code generation - manual loading without datasets library"""

    def __init__(self, split: str = 'test', use_subset: Optional[int] = None,
                 use_llm_rewards: bool = False, reward_function: Optional['HybridRewardFunction'] = None):
        print("Loading HumanEval dataset manually...")

        # Download HumanEval data directly from GitHub
        import urllib.request
        url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
        cache_dir = os.path.expanduser("~/.cache/humaneval")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, "HumanEval.jsonl.gz")
        
        if not os.path.exists(cache_file):
            print(f"Downloading HumanEval from {url}...")
            urllib.request.urlretrieve(url, cache_file)
            print("Download complete!")
        else:
            print(f"Using cached HumanEval from {cache_file}")
        
        # Load the data
        import gzip
        self.problems = []
        
        with gzip.open(cache_file, 'rt') as f:
            for line in f:
                example = json.loads(line)
                problem = {
                    'task_id': example['task_id'],
                    'prompt': example['prompt'],
                    'test': example['test'],
                    'entry_point': example['entry_point']
                }
                self.problems.append(problem)
        
        if use_subset is not None:
            self.problems = self.problems[:use_subset]

        print(f"Loaded {len(self.problems)} code problems")

        # Initialize reward function
        self.use_llm_rewards = use_llm_rewards
        self.reward_function = reward_function

        if use_llm_rewards and reward_function:
            print("✓ Using LLM-based reward evaluation (GPT-4 + Claude)")
        else:
            print("Using heuristic-based reward evaluation")

    def get_random_problem(self) -> Dict:
        return random.choice(self.problems)

    def compute_reward(self, code: str, problem: Dict) -> float:
        """Compute reward using hybrid approach (heuristics + optional LLM)"""
        if self.use_llm_rewards and self.reward_function:
            reward, details = self.reward_function.compute_reward(code, problem)
            # Optionally log details for debugging
            if random.random() < 0.05:  # Log 5% of evaluations
                print(f"\n  Reward: {details.get('final_reward', 0):.2f}", end='')
                if details.get('llm_used'):
                    comp = details.get('llm_evaluation', {}).get('comprehensibility', 0)
                    print(f" (LLM comp: {comp:.1f}/10)", end='')
            return reward
        else:
            # Use improved heuristic-only
            return self._compute_heuristic_reward(code, problem)

    def _compute_heuristic_reward(self, code: str, problem: Dict) -> float:
        """Improved heuristic reward (prevents reward hacking)"""
        code = code.strip()
        if len(code) == 0:
            return -10.0  # Heavy penalty for empty code

        reward = 2.0  # Base reward for non-empty

        # Must have function definition
        if 'def ' in code:
            reward += 4.0
        else:
            return -5.0  # No function is critical failure

        # Should have return
        if 'return' in code:
            reward += 3.0
        else:
            reward -= 1.0  # Penalty for no return

        # Check length (count non-empty lines)
        code_lines = [line for line in code.split('\n') if line.strip()]
        num_lines = len(code_lines)

        if num_lines < 1:
            reward -= 5.0
        elif num_lines < 3:
            reward -= 1.0  # Too short
        elif 3 <= num_lines <= 30:
            reward += 2.0  # Good length
        else:
            reward -= 0.5  # Too long

        # Check for structure
        if ':' in code and any(kw in code for kw in ['if ', 'for ', 'while ', 'elif ']):
            reward += 2.0

        # Check for indentation (suggests actual structure)
        indented = [line for line in code.split('\n') if line.startswith('    ') or line.startswith('\t')]
        if len(indented) >= 2:
            reward += 1.0

        return reward

    def __len__(self):
        return len(self.problems)

# ============================================================================
# ENVIRONMENT
# ============================================================================

class CodeGenerationEnvironment:
    """Specialized RL environment for code generation tasks"""

    def __init__(self, tokenizer, dataset, max_length: int = 512):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

        self.current_problem = None
        self.current_sequence = []
        self.step_count = 0
        self.done = False

        self.eos_token_id = tokenizer.eos_token_id

    def reset(self) -> Dict:
        """Reset with a random code problem"""
        self.current_problem = self.dataset.get_random_problem()

        # Tokenize the prompt
        prompt = self.current_problem['prompt']
        token_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0]

        self.current_sequence = token_ids.tolist()
        self.step_count = 0
        self.done = False

        return self.get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Take a step by generating a token"""
        self.current_sequence.append(action)
        self.step_count += 1

        text = self.tokenizer.decode(self.current_sequence)

        # Termination conditions for code
        self.done = (
            self.step_count >= self.max_length or
            action == self.eos_token_id or
            '\n\n\n' in text[-20:]  # Multiple blank lines = code complete
        )

        if self.done:
            # Extract generated code (remove prompt)
            prompt_length = len(self.tokenizer.encode(self.current_problem['prompt']))
            generated_tokens = self.current_sequence[prompt_length:]
            generated_code = self.tokenizer.decode(generated_tokens)

            # Compute reward using dataset's evaluation
            reward = self.dataset.compute_reward(generated_code, self.current_problem)
        else:
            # Small step penalty
            reward = -0.01

        next_state = self.get_state()

        info = {
            'text': text,
            'length': len(self.current_sequence),
            'problem_id': self.current_problem['task_id']
        }

        return next_state, reward, self.done, info

    def get_state(self) -> Dict:
        return {
            'token_ids': torch.tensor(self.current_sequence),
            'step': self.step_count,
            'done': self.done
        }

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class HierarchicalPolicy(nn.Module):
    """Two-level hierarchical policy for structured generation"""

    def __init__(self, vocab_size: int, d_model: int = 256, intention_dim: int = 64,
                 num_layers: int = 4, nhead: int = 4, max_len: int = 512):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.intention_dim = intention_dim
        self.max_len = max_len

        # Shared state encoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers // 2)

        # HIGH-LEVEL POLICY: State -> Intention distribution
        self.intention_mean = nn.Linear(d_model, intention_dim)
        self.intention_logstd = nn.Linear(d_model, intention_dim)

        # LOW-LEVEL POLICY: State + Intention -> Token distribution
        self.token_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )

        self.intention_proj = nn.Linear(intention_dim, d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

        # Separate value heads
        self.high_value = nn.Linear(d_model, 1)
        self.low_value = nn.Linear(d_model, 1)

    def encode_state(self, token_ids):
        # Truncate input sequence to max_len for positional encoding
        seq_len = token_ids.size(1)
        if seq_len > self.max_len:
            token_ids = token_ids[:, :self.max_len]
            seq_len = self.max_len

        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.state_encoder(x)
        return x

    def sample_intention(self, state_encoding):
        pooled = state_encoding.mean(dim=1)
        mean = self.intention_mean(pooled)
        logstd = self.intention_logstd(pooled)
        
        logstd = torch.clamp(logstd, min=-20, max=2)
        std = torch.exp(logstd)

        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.nan_to_num(std, nan=1e-6, posinf=1.0, neginf=1e-6)
        dist = torch.distributions.Normal(mean, std)
        intention = dist.rsample()
        return intention, dist

    def forward(self, token_ids, intention=None, return_intention=False):
        state_encoding = self.encode_state(token_ids)

        if intention is None:
            intention, intention_dist = self.sample_intention(state_encoding)
        else:
            intention_dist = None

        intention_encoded = self.intention_proj(intention)
        intention_expanded = intention_encoded.unsqueeze(1).expand(
            -1, state_encoding.size(1), -1
        )

        decoded = self.token_decoder(state_encoding, intention_expanded)
        logits = self.output_layer(decoded[:, -1, :])

        if return_intention:
            pooled = state_encoding.mean(dim=1)
            high_value = self.high_value(pooled)
            low_value = self.low_value(pooled)

            if intention_dist is not None:
                 intention_mean = intention_dist.mean
                 intention_std = intention_dist.stddev
            else:
                 intention_mean = None
                 intention_std = None

            return {
                'logits': logits,
                'intention': intention,
                'intention_mean': intention_mean,
                'intention_std': intention_std,
                'high_value': high_value,
                'low_value': low_value
            }
        else:
            return logits

# ============================================================================
# ROLLOUT BUFFER
# ============================================================================

class RolloutBuffer:
    """Experience buffer for storing and processing trajectories"""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.intentions = []

    def add(self, state, action, reward, log_prob, value, done, intention=None):
        self.states.append(state['token_ids'])
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

        if intention is not None:
            self.intentions.append(intention)

    def get_batches(self, gamma=0.99, gae_lambda=0.95):
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

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Pad states
        max_len = max(len(s) for s in self.states)
        padded_states = []
        for s in self.states:
            if len(s) < max_len:
                pad_len = max_len - len(s)
                s_padded = torch.cat([s, torch.zeros(pad_len, dtype=s.dtype)])
            else:
                s_padded = s
            padded_states.append(s_padded)

        batch = {
            'states': torch.stack(padded_states),
            'actions': torch.tensor(self.actions, dtype=torch.long),
            'log_probs': torch.tensor(self.log_probs, dtype=torch.float32),
            'returns': torch.tensor(returns, dtype=torch.float32),
            'advantages': torch.tensor(advantages, dtype=torch.float32),
        }

        if len(self.intentions) > 0:
            batch['intentions'] = torch.stack(self.intentions)

        return batch

    def __len__(self):
        return len(self.rewards)

# ============================================================================
# HIERARCHICAL PPO TRAINER
# ============================================================================

class HierarchicalPPOTrainer:
    """PPO trainer for hierarchical two-level policy"""

    def __init__(self, hierarchical_policy, lr: float = 3e-4, clip_ratio: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01, intention_coef: float = 0.1):
        self.policy = hierarchical_policy
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.intention_coef = intention_coef

        self.optimizer = optim.Adam(hierarchical_policy.parameters(), lr=lr)
        self.device = next(hierarchical_policy.parameters()).device

    def compute_hierarchical_policy_loss(self, states, actions, old_log_probs,
                                         old_intentions, advantages):
        model = self.policy.module if isinstance(self.policy, nn.DataParallel) else self.policy
        outputs = model(states, return_intention=True)

        # Low-level policy loss
        action_dist = torch.distributions.Categorical(logits=outputs['logits'])
        new_log_probs = action_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        low_level_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        token_entropy = action_dist.entropy().mean()

        # High-level policy loss
        if outputs['intention_mean'] is not None:
            intention_dist = torch.distributions.Normal(outputs['intention_mean'], outputs['intention_std'])
            old_intention_log_prob = intention_dist.log_prob(old_intentions).sum(dim=-1)
            intention_entropy = intention_dist.entropy().sum(dim=-1).mean()
            high_level_loss = -old_intention_log_prob.mean()
        else:
            high_level_loss = torch.tensor(0.0, device=self.device)
            intention_entropy = torch.tensor(0.0, device=self.device)

        policy_loss = low_level_loss + self.intention_coef * high_level_loss
        total_entropy = token_entropy + 0.1 * intention_entropy

        return policy_loss, total_entropy, high_level_loss

    def compute_hierarchical_value_loss(self, states, returns):
        model = self.policy.module if isinstance(self.policy, nn.DataParallel) else self.policy
        outputs = model(states, return_intention=True)

        high_value = outputs['high_value'].squeeze(-1)
        low_value = outputs['low_value'].squeeze(-1)

        high_value_loss = nn.functional.mse_loss(high_value, returns)
        low_value_loss = nn.functional.mse_loss(low_value, returns)

        return (high_value_loss + low_value_loss) / 2.0

    def update(self, buffer, epochs: int = 4, mini_batch_size: int = 1):
        """Update policy with mini-batching to reduce memory usage"""
        batches = buffer.get_batches()

        states = batches['states']
        actions = batches['actions']
        old_log_probs = batches['log_probs']
        returns = batches['returns']
        advantages = batches['advantages']

        # Get old intentions once (not in mini-batch loop)
        if 'intentions' in batches:
            old_intentions = batches['intentions']
        else:
            with torch.no_grad():
                # Process in chunks if needed
                all_intentions = []
                chunk_size = mini_batch_size
                for i in range(0, len(states), chunk_size):
                    chunk_states = states[i:i+chunk_size].to(self.device)
                    model = self.policy.module if isinstance(self.policy, nn.DataParallel) else self.policy
                    outputs = model(chunk_states, return_intention=True)
                    all_intentions.append(outputs['intention'].cpu())
                old_intentions = torch.cat(all_intentions, dim=0)

        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'intention_loss': 0.0
        }

        batch_size = len(states)
        num_mini_batches = max(1, (batch_size + mini_batch_size - 1) // mini_batch_size)

        for epoch in range(epochs):
            # Process in mini-batches
            for i in range(0, batch_size, mini_batch_size):
                end_idx = min(i + mini_batch_size, batch_size)

                # Move mini-batch to device
                batch_states = states[i:end_idx].to(self.device)
                batch_actions = actions[i:end_idx].to(self.device)
                batch_old_log_probs = old_log_probs[i:end_idx].to(self.device)
                batch_old_intentions = old_intentions[i:end_idx].to(self.device)
                batch_advantages = advantages[i:end_idx].to(self.device)
                batch_returns = returns[i:end_idx].to(self.device)

                # Compute losses
                policy_loss, entropy, intention_loss = self.compute_hierarchical_policy_loss(
                    batch_states, batch_actions, batch_old_log_probs, batch_old_intentions, batch_advantages
                )
                value_loss = self.compute_hierarchical_value_loss(batch_states, batch_returns)

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Scale loss for gradient accumulation
                total_loss = total_loss / num_mini_batches

                # Backward pass (accumulate gradients)
                total_loss.backward()

                stats['policy_loss'] += policy_loss.item() / num_mini_batches
                stats['value_loss'] += value_loss.item() / num_mini_batches
                stats['entropy'] += entropy.item() / num_mini_batches
                stats['intention_loss'] += intention_loss.item() / num_mini_batches

            # Update weights after all mini-batches
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()

        for key in stats:
            stats[key] /= epochs

        return stats

    def collect_episode_with_intentions(self, env, buffer, max_steps: int = 100):
        state = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            token_ids = state['token_ids'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                model = self.policy.module if isinstance(self.policy, nn.DataParallel) else self.policy
                outputs = model(token_ids,return_intention = True)
                action_dist = torch.distributions.Categorical(logits=outputs['logits'])
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                intention = outputs['intention']
                value = outputs['low_value']

            next_state, reward, done, info = env.step(action.item())

            buffer.add(
                state=state,
                action=action.item(),
                reward=reward,
                log_prob=log_prob.item(),
                value=value.item(),
                done=done,
                intention=intention.cpu()
            )

            state = next_state
            episode_reward += reward
            steps += 1

        return episode_reward

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    args = parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = get_device()

    print("=" * 70)
    print("RL-LLM: Hierarchical Code Generation Training")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Dataset: {args.dataset}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Episodes per iteration: {args.episodes_per_iter}")
    print(f"Max sequence length: {args.max_length}")
    print(f"LLM Rewards: {args.use_llm_rewards}")
    if args.use_llm_rewards:
        # Handle legacy arguments
        if args.use_gpt:
            print("  Warning: --use_gpt is deprecated. Use --llm_backend=gpt instead")
            args.llm_backend = 'gpt'
            if not args.llm_model or args.llm_model == 'llama3.2:3b':
                args.llm_model = 'gpt-4o-mini'
        if args.use_claude:
            print("  Warning: --use_claude is deprecated. Use --llm_backend=claude instead")
            args.llm_backend = 'claude'
            if not args.llm_model or args.llm_model == 'llama3.2:3b':
                args.llm_model = 'claude-3-5-haiku-20241022'

        print(f"  Backend: {args.llm_backend}")
        print(f"  Model: {args.llm_model}")
        print(f"  LLM Weight: {args.llm_weight}")
        if args.llm_backend == 'ollama':
            print(f"  Ollama URL: {args.ollama_url}")
        elif args.llm_backend == 'huggingface':
            print(f"  Device: {args.llm_device}")
    print("=" * 70)
    print()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize LLM reward evaluator if requested
    reward_function = None
    if args.use_llm_rewards:
        if not LLM_REWARDS_AVAILABLE:
            print("ERROR: LLM rewards requested but llm_reward_evaluator not available!")
            print("Make sure llm_reward_evaluator.py is in the same directory.")
            sys.exit(1)

        print("Initializing LLM reward evaluator...")
        try:
            llm_evaluator = LLMRewardEvaluator(
                llm_backend=args.llm_backend,
                model_name=args.llm_model,
                cache_dir=args.reward_cache_dir,
                device=args.llm_device,
                ollama_url=args.ollama_url
            )
            reward_function = HybridRewardFunction(
                llm_evaluator=llm_evaluator,
                use_llm_threshold=args.llm_threshold,
                llm_weight=args.llm_weight
            )
            print("✓ LLM reward evaluator initialized")
        except Exception as e:
            print(f"ERROR initializing LLM evaluator: {e}")
            print("Falling back to heuristic rewards only")
            args.use_llm_rewards = False

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset with LLM rewards
    print(f"Creating {args.dataset} dataset...")
    if args.dataset == 'humaneval':
        dataset = HumanEvalDataset(
            use_subset=args.subset_size,
            use_llm_rewards=args.use_llm_rewards,
            reward_function=reward_function
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet implemented in standalone script")

    print(f"Dataset size: {len(dataset)}")

    # Create environment
    print("Creating Code Generation environment...")
    env = CodeGenerationEnvironment(
        tokenizer=tokenizer,
        dataset=dataset,
        max_length=args.max_length
    )

    # Create hierarchical policy
    print("Creating hierarchical policy...")
    vocab_size = tokenizer.vocab_size

    policy = HierarchicalPolicy(
        vocab_size=vocab_size,
        d_model=args.d_model,
        intention_dim=args.intention_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        max_len=args.max_length
    ).to(device)

    if torch.cuda.device_count() > 1:
       print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
       policy = nn.DataParallel(policy)

    print(f"Policy created with {sum(p.numel() for p in policy.parameters()):,} parameters")

    # Create trainer
    print("Creating hierarchical PPO trainer...")
    trainer = HierarchicalPPOTrainer(policy, lr=args.lr)

    print()
    print("=" * 70)
    print("Starting Training")
    print("=" * 70)
    print()

    # Training loop
    best_reward = float('-inf')
    training_history = []

    for iteration in tqdm(range(args.num_iterations), desc="Training"):
        policy.train()
        buffer = RolloutBuffer()
        episode_rewards = []

        # Collect episodes
        for episode in range(args.episodes_per_iter):
            episode_reward = trainer.collect_episode_with_intentions(
                env=env,
                buffer=buffer,
                max_steps=args.max_length
            )
            episode_rewards.append(episode_reward)

        # Update policy
        if len(buffer) > 0:
            stats = trainer.update(buffer, epochs=4, mini_batch_size=1)
        else:
            stats = {}

        # Logging
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        training_history.append(avg_reward)

        if (iteration + 1) % args.log_interval == 0:
            print(f"\n{'='*70}")
            print(f"Iteration {iteration + 1}/{args.num_iterations}")
            print(f"{'='*70}")
            print(f"  Avg Training Reward: {avg_reward:.2f}")
            print(f"  Policy Loss: {stats.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {stats.get('value_loss', 0):.4f}")
            print(f"  Entropy: {stats.get('entropy', 0):.4f}")
            print(f"  Intention Loss: {stats.get('intention_loss', 0):.4f}")

            # Generate sample code
            policy.eval()
            state = env.reset()
            done = False
            steps = 0

            while not done and steps < 100:
                token_ids = state['token_ids'].unsqueeze(0).to(device)
                with torch.no_grad():
                    model = policy.module if isinstance(policy, nn.DataParallel) else policy
                    outputs = model(token_ids, return_intention=True)
                    action_dist = torch.distributions.Categorical(logits=outputs['logits'])
                    action = action_dist.sample()
                state, _, done, info = env.step(action.item())
                steps += 1

            print(f"\n  Sample Generated Code:")
            print(f"  {'-'*66}")
            code_snippet = info['text'][:200].replace('\n', '\n  ')
            print(f"  {code_snippet}...")
            print(f"  {'-'*66}")

        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            print(f"\n  ✓ New best model! Reward: {best_reward:.2f}")
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(policy, trainer.optimizer, iteration, best_reward,
                          training_history, args, checkpoint_path)

        # Save periodic checkpoints
        if (iteration + 1) % 100 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_iter_{iteration+1}.pt')
            save_checkpoint(policy, trainer.optimizer, iteration, best_reward,
                          training_history, args, checkpoint_path)

    # Save final model
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best reward achieved: {best_reward:.2f}")

    final_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    save_checkpoint(policy, trainer.optimizer, args.num_iterations, best_reward,
                  training_history, args, final_path)

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'training_history': training_history,
            'best_reward': best_reward,
            'args': vars(args),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    main()
