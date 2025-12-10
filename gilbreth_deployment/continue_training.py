"""
RL-LLM: Continuation Training Script
Fine-tune a pre-trained hierarchical policy on a new dataset

Supports:
- MBPP (Mostly Basic Python Problems)
- APPS (Automated Programming Progress Standard)
- CodeSearchNet
- Custom datasets
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
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================================
# CONFIGURATION
# ============================================================================
def resolve_checkpoint_path(args) -> str:
    """
    Resolve which checkpoint to load:

    Priority:
      1. Explicit --checkpoint path, if it exists
      2. best_model.pt in args.checkpoint_dir (e.g. ./checkpoints_continued)
      3. best_model.pt in ./checkpoints (original training dir)

    Raises if nothing is found.
    """
    # 1) Explicit path
    if args.checkpoint and os.path.exists(args.checkpoint):
        return args.checkpoint

    candidates = []

    # 2) best_model.pt in continuation dir (if it exists)
    if args.checkpoint_dir:
        cont_best = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.exists(cont_best):
            candidates.append(cont_best)

    # 3) Original training dir ./checkpoints/best_model.pt
    base_best = os.path.join("./checkpoints", "best_model.pt")
    if os.path.exists(base_best):
        candidates.append(base_best)

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Tried:\n"
            f"  explicit --checkpoint={args.checkpoint}\n"
            f"  {os.path.join(args.checkpoint_dir, 'best_model.pt') if args.checkpoint_dir else ''}\n"
            "  ./checkpoints/best_model.pt"
        )

    # Pick the most recently modified best_model.pt
    best = max(candidates, key=os.path.getmtime)
    print(f"Resolved checkpoint to: {best}")
    return best

def parse_args():
    parser = argparse.ArgumentParser(description='Continue training on new dataset')
    
    # Checkpoint loading
    
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint (vs. fine-tune)')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='mbpp',
                        choices=['mbpp', 'apps', 'codesearchnet', 'custom'],
                        help='Dataset to train on')
    parser.add_argument('--custom_data_path', type=str, default=None,
                        help='Path to custom dataset (JSON format)')
    parser.add_argument('--subset_size', type=int, default=100,
                        help='Subset size for training')
    parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to pre-trained checkpoint (default: latest best_model.pt)')
    # Training hyperparameters
    parser.add_argument('--num_iterations', type=int, default=500,
                        help='Number of training iterations')
    parser.add_argument('--episodes_per_iter', type=int, default=4,
                        help='Episodes per iteration')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='Learning rate decay per iteration')
    
    # Fine-tuning specific
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze the state encoder layers')
    parser.add_argument('--warmup_steps', type=int, default=50,
                        help='Warmup steps for learning rate')
    
    # Reward shaping
    parser.add_argument('--reward_scale', type=float, default=1.0,
                        help='Scale factor for rewards')
    parser.add_argument('--execution_bonus', type=float, default=5.0,
                        help='Bonus reward for code that executes')
    
    # General
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_continued',
                        help='Directory to save new checkpoints')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='Logging interval')
    
    return parser.parse_args()


# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

class PositionalEncoding(nn.Module):
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
    def __init__(self, vocab_size: int, d_model: int = 256, intention_dim: int = 64,
                 num_layers: int = 4, nhead: int = 4, max_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.intention_dim = intention_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers // 2)

        self.intention_mean = nn.Linear(d_model, intention_dim)
        self.intention_logstd = nn.Linear(d_model, intention_dim)

        self.token_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, batch_first=True
            ),
            num_layers=num_layers // 2
        )
        self.intention_proj = nn.Linear(intention_dim, d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.high_value = nn.Linear(d_model, 1)
        self.low_value = nn.Linear(d_model, 1)

    def encode_state(self, token_ids):
        seq_len = token_ids.size(1)
        if seq_len > self.max_len:
            token_ids = token_ids[:, :self.max_len]
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
            return {
                'logits': logits,
                'intention': intention,
                'intention_dist': intention_dist,
                'high_value': self.high_value(pooled),
                'low_value': self.low_value(pooled)
            }
        return logits

    def freeze_encoder_layers(self):
        """Freeze state encoder for fine-tuning"""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.state_encoder.parameters():
            param.requires_grad = False
        print("  ✓ Encoder layers frozen")


# ============================================================================
# DATASET LOADERS
# ============================================================================

class MBPPDataset:
    """MBPP: Mostly Basic Python Problems"""
    
    def __init__(self, split: str = 'train', use_subset: Optional[int] = None):
        print("Loading MBPP dataset...")
        try:
            dataset = load_dataset("mbpp", split=split)
        except Exception as e:
            print(f"Error loading MBPP: {e}")
            print("Trying alternative: google-research-datasets/mbpp")
            try:
                dataset = load_dataset("google-research-datasets/mbpp", split=split)
            except:
                dataset = []
        
        self.problems = []
        for example in dataset:
            # MBPP format: text (description), code (solution), test_list
            problem = {
                'task_id': f"mbpp_{example.get('task_id', len(self.problems))}",
                'prompt': self._create_prompt(example),
                'solution': example.get('code', ''),
                'test_list': example.get('test_list', [])
            }
            self.problems.append(problem)
        
        if use_subset is not None:
            self.problems = self.problems[:use_subset]
        print(f"Loaded {len(self.problems)} MBPP problems")

    def _create_prompt(self, example) -> str:
        """Create a code generation prompt from MBPP example"""
        description = example.get('text', '')
        test_examples = example.get('test_list', [])[:2]  # Use first 2 tests as examples
        
        prompt = f'"""\n{description}\n'
        if test_examples:
            prompt += '\nExamples:\n'
            for test in test_examples:
                prompt += f'    {test}\n'
        prompt += '"""\n\ndef '
        
        return prompt

    def get_random_problem(self) -> Dict:
        return random.choice(self.problems)

    def compute_reward(self, code: str, problem: Dict) -> float:
        """Enhanced reward function for MBPP"""
        reward = 0.0
        
        # Basic structure rewards
        if 'def ' in code:
            reward += 2.0
        if 'return' in code:
            reward += 2.0
        
        # Length-based reward
        code_lines = len(code.split('\n'))
        if 2 <= code_lines <= 50:
            reward += 1.0
        elif code_lines < 2:
            reward -= 2.0
        
        # Control flow bonus
        if ':' in code and ('if' in code or 'for' in code or 'while' in code):
            reward += 1.0
        
        # Syntax validity bonus
        try:
            compile(code, '<string>', 'exec')
            reward += 2.0  # Bonus for valid syntax
        except SyntaxError:
            reward -= 1.0
        
        return reward

    def __len__(self):
        return len(self.problems)


class APPSDataset:
    """APPS: Automated Programming Progress Standard"""
    
    def __init__(self, split: str = 'train', difficulty: str = 'introductory',
                 use_subset: Optional[int] = None):
        print(f"Loading APPS dataset ({difficulty})...")
        try:
            dataset = load_dataset("codeparrot/apps", split=split, 
                                   difficulties=[difficulty])
        except Exception as e:
            print(f"Error loading APPS: {e}")
            dataset = []
        
        self.problems = []
        for example in dataset:
            problem = {
                'task_id': f"apps_{example.get('problem_id', len(self.problems))}",
                'prompt': self._create_prompt(example),
                'solutions': example.get('solutions', ''),
                'input_output': example.get('input_output', '')
            }
            self.problems.append(problem)
        
        if use_subset is not None:
            self.problems = self.problems[:use_subset]
        print(f"Loaded {len(self.problems)} APPS problems")

    def _create_prompt(self, example) -> str:
        question = example.get('question', '')
        # Truncate long questions
        if len(question) > 500:
            question = question[:500] + '...'
        return f'"""\n{question}\n"""\n\ndef solution():\n'

    def get_random_problem(self) -> Dict:
        return random.choice(self.problems)

    def compute_reward(self, code: str, problem: Dict) -> float:
        reward = 0.0
        if 'def ' in code:
            reward += 2.0
        if 'return' in code:
            reward += 2.0
        code_lines = len(code.split('\n'))
        if 2 <= code_lines <= 50:
            reward += 1.0
        if ':' in code and ('if' in code or 'for' in code or 'while' in code):
            reward += 1.0
        try:
            compile(code, '<string>', 'exec')
            reward += 2.0
        except SyntaxError:
            reward -= 1.0
        return reward

    def __len__(self):
        return len(self.problems)


class CodeSearchNetDataset:
    """CodeSearchNet: Natural language to code"""
    
    def __init__(self, language: str = 'python', split: str = 'train',
                 use_subset: Optional[int] = None):
        print(f"Loading CodeSearchNet ({language})...")
        try:
            dataset = load_dataset("code_search_net", language, split=split)
        except Exception as e:
            print(f"Error loading CodeSearchNet: {e}")
            dataset = []
        
        self.problems = []
        for example in dataset:
            if example.get('func_documentation_string'):
                problem = {
                    'task_id': f"csn_{len(self.problems)}",
                    'prompt': self._create_prompt(example),
                    'solution': example.get('func_code_string', '')
                }
                self.problems.append(problem)
        
        if use_subset is not None:
            self.problems = self.problems[:use_subset]
        print(f"Loaded {len(self.problems)} CodeSearchNet problems")

    def _create_prompt(self, example) -> str:
        docstring = example.get('func_documentation_string', '')
        func_name = example.get('func_name', 'solution')
        return f'def {func_name}():\n    """{docstring}"""\n'

    def get_random_problem(self) -> Dict:
        return random.choice(self.problems)

    def compute_reward(self, code: str, problem: Dict) -> float:
        reward = 0.0
        if 'return' in code:
            reward += 2.0
        code_lines = len(code.split('\n'))
        if 1 <= code_lines <= 30:
            reward += 1.0
        try:
            # Try to compile the generated body
            test_code = "def test():\n" + "\n".join("    " + line for line in code.split('\n'))
            compile(test_code, '<string>', 'exec')
            reward += 3.0
        except SyntaxError:
            reward -= 1.0
        return reward

    def __len__(self):
        return len(self.problems)


class CustomDataset:
    """Load custom dataset from JSON file"""
    
    def __init__(self, path: str, use_subset: Optional[int] = None):
        print(f"Loading custom dataset from {path}...")
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.problems = []
        for i, example in enumerate(data):
            problem = {
                'task_id': example.get('task_id', f'custom_{i}'),
                'prompt': example.get('prompt', ''),
                'solution': example.get('solution', ''),
                'test': example.get('test', '')
            }
            self.problems.append(problem)
        
        if use_subset is not None:
            self.problems = self.problems[:use_subset]
        print(f"Loaded {len(self.problems)} custom problems")

    def get_random_problem(self) -> Dict:
        return random.choice(self.problems)

    def compute_reward(self, code: str, problem: Dict) -> float:
        reward = 0.0
        if 'def ' in code:
            reward += 2.0
        if 'return' in code:
            reward += 2.0
        code_lines = len(code.split('\n'))
        if 2 <= code_lines <= 50:
            reward += 1.0
        try:
            compile(code, '<string>', 'exec')
            reward += 2.0
        except SyntaxError:
            pass
        return reward

    def __len__(self):
        return len(self.problems)


# ============================================================================
# ENVIRONMENT
# ============================================================================

class CodeGenerationEnvironment:
    def __init__(self, tokenizer, dataset, max_length: int = 256,
                 reward_scale: float = 1.0, execution_bonus: float = 5.0):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.reward_scale = reward_scale
        self.execution_bonus = execution_bonus
        self.current_problem = None
        self.current_sequence = []
        self.step_count = 0
        self.done = False
        self.eos_token_id = tokenizer.eos_token_id

    def reset(self) -> Dict:
        self.current_problem = self.dataset.get_random_problem()
        prompt = self.current_problem['prompt']
        token_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0]
        self.current_sequence = token_ids.tolist()
        self.step_count = 0
        self.done = False
        return self.get_state()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        self.current_sequence.append(action)
        self.step_count += 1
        text = self.tokenizer.decode(self.current_sequence)
        
        self.done = (
            self.step_count >= self.max_length or
            action == self.eos_token_id or
            '\n\n\n' in text[-20:]
        )
        
        if self.done:
            prompt_length = len(self.tokenizer.encode(self.current_problem['prompt']))
            generated_tokens = self.current_sequence[prompt_length:]
            generated_code = self.tokenizer.decode(generated_tokens)
            
            reward = self.dataset.compute_reward(generated_code, self.current_problem)
            reward *= self.reward_scale
            
            # Execution bonus
            try:
                full_code = self.current_problem['prompt'] + generated_code
                compile(full_code, '<string>', 'exec')
                reward += self.execution_bonus
            except:
                pass
        else:
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
# ROLLOUT BUFFER & TRAINER (same as original)
# ============================================================================

class RolloutBuffer:
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
        
        max_len = max(len(s) for s in self.states)
        padded_states = []
        for s in self.states:
            if len(s) < max_len:
                s_padded = torch.cat([s, torch.zeros(max_len - len(s), dtype=s.dtype)])
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


class HierarchicalPPOTrainer:
    def __init__(self, policy, lr: float = 1e-4, clip_ratio: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 intention_coef: float = 0.1, mini_batch_size: int = 32):
        self.policy = policy
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.intention_coef = intention_coef
        self.mini_batch_size = mini_batch_size
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, policy.parameters()), 
            lr=lr
        )
        self.device = next(policy.parameters()).device
        self.scheduler = None

    def set_lr_scheduler(self, warmup_steps: int, decay_rate: float):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return decay_rate ** (step - warmup_steps)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def compute_hierarchical_policy_loss(self, states, actions, old_log_probs, 
                                          old_intentions, advantages):
        outputs = self.policy(states, return_intention=True)
        action_dist = torch.distributions.Categorical(logits=outputs['logits'])
        new_log_probs = action_dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        low_level_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        token_entropy = action_dist.entropy().mean()
        
        if outputs['intention_dist'] is not None:
            old_intention_log_prob = outputs['intention_dist'].log_prob(old_intentions).sum(dim=-1)
            intention_entropy = outputs['intention_dist'].entropy().sum(dim=-1).mean()
            high_level_loss = -old_intention_log_prob.mean()
        else:
            high_level_loss = torch.tensor(0.0, device=self.device)
            intention_entropy = torch.tensor(0.0, device=self.device)
        
        policy_loss = low_level_loss + self.intention_coef * high_level_loss
        total_entropy = token_entropy + 0.1 * intention_entropy
        return policy_loss, total_entropy, high_level_loss

    def compute_hierarchical_value_loss(self, states, returns):
        outputs = self.policy(states, return_intention=True)
        high_value = outputs['high_value'].squeeze(-1)
        low_value = outputs['low_value'].squeeze(-1)
        return (nn.functional.mse_loss(high_value, returns) + 
                nn.functional.mse_loss(low_value, returns)) / 2.0

    def update(self, buffer, epochs: int = 4):
        batches = buffer.get_batches()
        states = batches['states'].to(self.device)
        actions = batches['actions'].to(self.device)
        old_log_probs = batches['log_probs'].to(self.device)
        returns = batches['returns'].to(self.device)
        advantages = batches['advantages'].to(self.device)

        if 'intentions' in batches:
            old_intentions = batches['intentions'].to(self.device)
        else:
            with torch.no_grad():
                outputs = self.policy(states, return_intention=True)
                old_intentions = outputs['intention']

        num_samples = states.size(0)
        stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'intention_loss': 0.0}

        for epoch in range(epochs):
            perm = torch.randperm(num_samples, device=self.device)
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            epoch_intention_loss = 0.0
            num_minibatches = 0

            for start in range(0, num_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = perm[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                mb_old_intentions = old_intentions[idx]

                policy_loss, entropy, intention_loss = self.compute_hierarchical_policy_loss(
                    mb_states, mb_actions, mb_old_log_probs, mb_old_intentions, mb_advantages
                )
                value_loss = self.compute_hierarchical_value_loss(mb_states, mb_returns)
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                epoch_intention_loss += intention_loss.item()
                num_minibatches += 1

                # Optional: free memory between minibatches
                del mb_states, mb_actions, mb_old_log_probs, mb_returns, mb_advantages, mb_old_intentions
                torch.cuda.empty_cache()

            # Average over minibatches
            if num_minibatches > 0:
                stats['policy_loss'] = epoch_policy_loss / num_minibatches
                stats['value_loss'] = epoch_value_loss / num_minibatches
                stats['entropy'] = epoch_entropy / num_minibatches
                stats['intention_loss'] = epoch_intention_loss / num_minibatches

        if self.scheduler:
            self.scheduler.step()

        return stats

    def collect_episode_with_intentions(self, env, buffer, max_steps: int = 100):
        state = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done and steps < max_steps:
            token_ids = state['token_ids'].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.policy(token_ids, return_intention=True)
                action_dist = torch.distributions.Categorical(logits=outputs['logits'])
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                intention = outputs['intention']
                value = outputs['low_value']
            
            next_state, reward, done, info = env.step(action.item())
            buffer.add(state, action.item(), reward, log_prob.item(), 
                      value.item(), done, intention.cpu())
            state = next_state
            episode_reward += reward
            steps += 1
        
        return episode_reward


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("RL-LLM: Continuation Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Base checkpoint: {args.checkpoint}")
    print(f"New dataset: {args.dataset}")
    print(f"Learning rate: {args.lr}")
    print("=" * 70)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load checkpoint
    ckpt_path = resolve_checkpoint_path(args)

    # Load checkpoint
    print(f"\nLoading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint.get('args', {})

    saved_max_len = model_args.get('max_length', 256)

    # Create model with the same hyperparameters used in training
    policy = HierarchicalPolicy(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args.get('d_model', 256),
        intention_dim=model_args.get('intention_dim', 64),
        num_layers=model_args.get('num_layers', 4),
        nhead=model_args.get('nhead', 4),
        max_len=saved_max_len
    ).to(device)

    # Handle DataParallel checkpoints (module.* keys)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith("module.") for k in state_dict.keys()):
        print("  Detected DataParallel checkpoint, stripping 'module.' prefix...")
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    policy.load_state_dict(state_dict)
    print(f"✓ Loaded model from iteration {checkpoint.get('iteration', 'unknown')}")
    print(f"  Previous best reward: {checkpoint.get('best_reward', 'unknown')}")
    
    # Optionally freeze encoder
    if args.freeze_encoder:
        policy.freeze_encoder_layers()
    
    # Create dataset
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == 'mbpp':
        dataset = MBPPDataset(split='train', use_subset=args.subset_size)
    elif args.dataset == 'apps':
        dataset = APPSDataset(split='train', use_subset=args.subset_size)
    elif args.dataset == 'codesearchnet':
        dataset = CodeSearchNetDataset(split='train', use_subset=args.subset_size)
    elif args.dataset == 'custom':
        if not args.custom_data_path:
            raise ValueError("Must provide --custom_data_path for custom dataset")
        dataset = CustomDataset(args.custom_data_path, use_subset=args.subset_size)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create environment
    env = CodeGenerationEnvironment(
        tokenizer=tokenizer,
        dataset=dataset,
        max_length=args.max_length,
        reward_scale=args.reward_scale,
        execution_bonus=args.execution_bonus
    )
    
    # Create trainer
    trainer = HierarchicalPPOTrainer(policy, lr=args.lr, mini_batch_size=16)
    trainer.set_lr_scheduler(args.warmup_steps, args.lr_decay)
    
    # Optionally load optimizer state
    if args.resume and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Resumed optimizer state")
    
    print("\n" + "=" * 70)
    print("Starting Continuation Training")
    print("=" * 70 + "\n")
    
    # Training loop
    best_reward = checkpoint.get('best_reward', float('-inf'))
    training_history = checkpoint.get('training_history', [])
    start_iteration = checkpoint.get('iteration', 0) if args.resume else 0
    
    for iteration in tqdm(range(args.num_iterations), desc="Training"):
        policy.train()
        buffer = RolloutBuffer()
        episode_rewards = []
        
        for _ in range(args.episodes_per_iter):
            reward = trainer.collect_episode_with_intentions(env, buffer, args.max_length)
            episode_rewards.append(reward)
        
        if len(buffer) > 0:
            stats = trainer.update(buffer, epochs=2)
        else:
            stats = {}

        torch.cuda.empty_cache()
        
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        training_history.append(avg_reward)
        
        if (iteration + 1) % args.log_interval == 0:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"\n{'='*70}")
            print(f"Iteration {iteration + 1}/{args.num_iterations}")
            print(f"{'='*70}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Policy Loss: {stats.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {stats.get('value_loss', 0):.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Sample generation
            policy.eval()
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < 100:
                token_ids = state['token_ids'].unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = policy(token_ids, return_intention=True)
                    action = torch.distributions.Categorical(logits=outputs['logits']).sample()
                state, _, done, info = env.step(action.item())
                steps += 1
            print(f"\n  Sample ({info['problem_id']}):")
            print(f"  {'-'*66}")
            print(f"  {info['text'][:300].replace(chr(10), chr(10) + '  ')}...")
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save({
                'iteration': start_iteration + iteration + 1,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_reward': best_reward,
                'training_history': training_history,
                'args': vars(args),
                'base_checkpoint': args.checkpoint
            }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f"\n  ✓ New best! Reward: {best_reward:.2f}")
    
    # Save final model
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best reward: {best_reward:.2f}")
    
    torch.save({
        'iteration': start_iteration + args.num_iterations,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'best_reward': best_reward,
        'training_history': training_history,
        'args': vars(args)
    }, os.path.join(args.checkpoint_dir, 'final_model.pt'))
    
    with open(os.path.join(args.checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump({
            'training_history': training_history,
            'best_reward': best_reward,
            'args': vars(args),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)


if __name__ == "__main__":
    main()