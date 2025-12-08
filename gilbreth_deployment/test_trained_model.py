"""
RL-LLM: Model Testing and Evaluation Script
Tests the trained hierarchical policy on code generation tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import random
import math
import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Test trained hierarchical RL model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of code samples to generate')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--eval_all', action='store_true',
                        help='Evaluate on all HumanEval problems')
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


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def compute_reward(code: str, problem: Dict) -> Dict[str, float]:
    """Compute detailed reward breakdown"""
    rewards = {
        'has_def': 0.0,
        'has_return': 0.0,
        'good_length': 0.0,
        'has_control_flow': 0.0,
        'total': 0.0
    }
    
    if 'def ' in code:
        rewards['has_def'] = 2.0
    if 'return' in code:
        rewards['has_return'] = 2.0
    
    code_lines = len(code.split('\n'))
    if 2 <= code_lines <= 50:
        rewards['good_length'] = 1.0
    elif code_lines < 2:
        rewards['good_length'] = -2.0
    
    if ':' in code and ('if' in code or 'for' in code or 'while' in code):
        rewards['has_control_flow'] = 1.0
    
    rewards['total'] = sum(v for k, v in rewards.items() if k != 'total')
    return rewards


def try_execute_code(code: str, problem: Dict) -> Dict:
    """Attempt to execute generated code with test cases"""
    result = {
        'syntax_valid': False,
        'executes': False,
        'passes_tests': False,
        'error': None
    }
    
    try:
        compile(code, '<string>', 'exec')
        result['syntax_valid'] = True
    except SyntaxError as e:
        result['error'] = f"SyntaxError: {e}"
        return result
    
    # Try to execute
    try:
        exec_globals = {}
        exec(code, exec_globals)
        result['executes'] = True
        
        # Try running test cases if available
        if 'test' in problem and 'entry_point' in problem:
            try:
                test_code = problem['test']
                exec(test_code, exec_globals)
                # Check if the check function exists and call it
                if 'check' in exec_globals:
                    exec_globals['check'](exec_globals.get(problem['entry_point']))
                    result['passes_tests'] = True
            except Exception as e:
                result['error'] = f"Test failed: {e}"
    except Exception as e:
        result['error'] = f"Runtime error: {e}"
    
    return result


def top_k_top_p_filtering(logits, top_k=50, top_p=0.95, temperature=1.0):
    """Apply top-k and top-p (nucleus) filtering to logits"""
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
    
    return logits


# ============================================================================
# MAIN TESTING CLASS
# ============================================================================

class ModelTester:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get model args
        args = checkpoint.get('args', {})
        self.model_args = args

        # Create model
        self.policy = HierarchicalPolicy(
            vocab_size=self.tokenizer.vocab_size,
            d_model=args.get('d_model', 384),  # Match your training default
            intention_dim=args.get('intention_dim', 64),
            num_layers=args.get('num_layers', 4),
            nhead=args.get('nhead', 4),
            max_len=args.get('max_length', 512)  # Match your training default
        ).to(device)

        # Load state dict with DataParallel handling
        state_dict = checkpoint['model_state_dict']
        
        # Remove 'module.' prefix if present (from DataParallel)
        if list(state_dict.keys())[0].startswith('module.'):
            print("  Detected DataParallel checkpoint, removing 'module.' prefix...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        print(f"✓ Model loaded (iteration {checkpoint.get('iteration', 'unknown')})")
        print(f"  Best training reward: {checkpoint.get('best_reward', 'unknown')}")
        print(f"  Parameters: {sum(p.numel() for p in self.policy.parameters()):,}")

    def generate_code(self, prompt: str, max_length: int = 256,
                      temperature: float = 1.0, top_k: int = 50, 
                      top_p: float = 0.95) -> str:
        """Generate code completion for a prompt"""
        token_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_length = token_ids.size(1)
        
        # Sample a fixed intention for consistent generation
        with torch.no_grad():
            state_encoding = self.policy.encode_state(token_ids)
            intention, _ = self.policy.sample_intention(state_encoding)
        
        generated = token_ids[0].tolist()
        
        for step in range(max_length):
            input_ids = torch.tensor([generated[-self.policy.max_len:]]).to(self.device)
            
            with torch.no_grad():
                logits = self.policy(input_ids, intention=intention)
            
            # Apply sampling
            filtered_logits = top_k_top_p_filtering(
                logits[0], top_k=top_k, top_p=top_p, temperature=temperature
            )
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            # Check termination
            text = self.tokenizer.decode(generated)
            if (next_token == self.tokenizer.eos_token_id or 
                '\n\n\n' in text[-20:]):
                break
        
        # Return only the generated part
        generated_tokens = generated[prompt_length:]
        return self.tokenizer.decode(generated_tokens)

    def evaluate_on_humaneval(self, num_problems: Optional[int] = None, 
                               verbose: bool = False) -> Dict:
        """Evaluate model on HumanEval benchmark"""
        print("\nLoading HumanEval dataset...")
        try:
            dataset = load_dataset("openai_humaneval", split="test")
        except Exception as e:
            print(f"Error loading HumanEval: {e}")
            return {}
        
        problems = list(dataset)
        if num_problems:
            problems = problems[:num_problems]
        
        print(f"Evaluating on {len(problems)} problems...\n")
        
        results = []
        reward_totals = defaultdict(float)
        execution_stats = defaultdict(int)
        
        for i, problem in enumerate(problems):
            task_id = problem['task_id']
            prompt = problem['prompt']
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"Problem {i+1}/{len(problems)}: {task_id}")
                print(f"{'='*70}")
                print(f"Prompt:\n{prompt[:200]}...")
            
            # Generate code
            generated = self.generate_code(prompt)
            full_code = prompt + generated
            
            # Compute rewards
            rewards = compute_reward(generated, problem)
            for k, v in rewards.items():
                reward_totals[k] += v
            
            # Try execution
            exec_result = try_execute_code(full_code, problem)
            for k, v in exec_result.items():
                if isinstance(v, bool) and v:
                    execution_stats[k] += 1
            
            result = {
                'task_id': task_id,
                'prompt': prompt,
                'generated': generated,
                'rewards': rewards,
                'execution': exec_result
            }
            results.append(result)
            
            if verbose:
                print(f"\nGenerated code:\n{'-'*40}")
                print(generated[:500])
                print(f"{'-'*40}")
                print(f"Rewards: {rewards}")
                print(f"Execution: {exec_result}")
            else:
                status = "✓" if exec_result['syntax_valid'] else "✗"
                print(f"  [{status}] {task_id}: reward={rewards['total']:.1f}, "
                      f"syntax={'valid' if exec_result['syntax_valid'] else 'invalid'}")
        
        # Compute summary statistics
        n = len(problems)
        summary = {
            'num_problems': n,
            'avg_rewards': {k: v/n for k, v in reward_totals.items()},
            'execution_rates': {k: v/n for k, v in execution_stats.items()},
            'results': results
        }
        
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Problems evaluated: {n}")
        print(f"\nAverage Rewards:")
        for k, v in summary['avg_rewards'].items():
            print(f"  {k}: {v:.3f}")
        print(f"\nExecution Rates:")
        for k, v in summary['execution_rates'].items():
            print(f"  {k}: {v*100:.1f}%")
        
        return summary

    def interactive_generation(self):
        """Interactive mode for testing generation"""
        print("\n" + "="*70)
        print("INTERACTIVE CODE GENERATION")
        print("="*70)
        print("Enter a function signature/docstring to complete.")
        print("Type 'quit' to exit.\n")
        
        while True:
            print("\nEnter prompt (or 'quit'):")
            lines = []
            while True:
                line = input()
                if line.lower() == 'quit':
                    return
                if line == '---':  # End of prompt marker
                    break
                lines.append(line)
            
            prompt = '\n'.join(lines)
            if not prompt.strip():
                continue
            
            print("\nGenerating...")
            generated = self.generate_code(prompt)
            
            print(f"\n{'='*40}")
            print("Generated code:")
            print(f"{'='*40}")
            print(generated)
            print(f"{'='*40}")
            
            rewards = compute_reward(generated, {})
            print(f"Rewards: {rewards}")


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
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    tester = ModelTester(args.checkpoint, device)
    
    # Run evaluation
    if args.eval_all:
        results = tester.evaluate_on_humaneval(verbose=args.verbose)
    else:
        results = tester.evaluate_on_humaneval(
            num_problems=args.num_samples, 
            verbose=args.verbose
        )
    
    # Save results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f'eval_results_{timestamp}.json')
        
        # Remove non-serializable items
        save_results = {
            'num_problems': results['num_problems'],
            'avg_rewards': results['avg_rewards'],
            'execution_rates': results['execution_rates'],
            'args': vars(args),
            'timestamp': timestamp
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()