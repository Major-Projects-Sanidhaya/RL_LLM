"""
Main Training Script

This is the entry point for training RL-based language models on various tasks.
It orchestrates the entire training pipeline from data loading to model checkpointing.

How it works:
- Loads dataset (tiny/math/code) and tokenizer
- Creates RL environment and policy/value networks
- Training loop: Collect trajectories -> Update with PPO -> Evaluate -> Save best model
- Progressive curriculum: Start with tiny, move to math, then code

Implementation approach:
- Uses PPO algorithm for stable policy updates
- Collects multiple episodes per iteration for sample efficiency
- Evaluates periodically and saves best checkpoint
- Prints sample generations for qualitative monitoring
- Modular design: Easy to swap datasets, environments, or algorithms

Training progression:
1. Start with TinyDataset (10 simple problems) to verify pipeline works
2. Move to SimpleMathDataset (arithmetic) for intermediate difficulty
3. Finally tackle CodeProblemDataset (HumanEval) for full challenge
"""

import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import sys
import argparse

# Add argument parser for cluster usage
parser = argparse.ArgumentParser(description='Train RL-LLM')
parser.add_argument('--dataset', type=str, default='tiny', choices=['tiny', 'math', 'code'])
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--max-length', type=int, default=50)
parser.add_argument('--save-dir', type=str, default='checkpoints')
parser.add_argument('--log-interval', type=int, default=10)

args = parser.parse_args()

from data.dataset_loader import create_dataset
from environments.text_generation_env import TextGenerationEnvironment
from environments.code_generation_env import CodeGenerationEnvironment
from environments.reward_functions import SimpleRewardFunction
from models.networks import SimpleTokenPolicy, SimpleValueNetwork
from training.buffer import RolloutBuffer
from training.ppo_trainer import SimplePPOTrainer
from training.utils import set_seed, get_device
from evaluation.metrics import Evaluator


def train_simple_policy(
    dataset_type='tiny',
    num_iterations=100,
    episodes_per_iteration=10,
    max_length=50,
    save_dir='checkpoints'
):
    """
    Train simple policy with PPO algorithm

    This is the main training function that sets up all components and runs
    the training loop.

    Args:
        dataset_type: 'tiny', 'math', or 'code' - which dataset to train on
        num_iterations: Number of training iterations (collect + update cycles)
        episodes_per_iteration: Episodes to collect before each PPO update
        max_length: Maximum tokens to generate per episode
        save_dir: Directory to save model checkpoints

    Returns:
        Tuple of (policy, value_net) - the trained networks
    """
    # Setup
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print(f"Creating {dataset_type} dataset...")
    if dataset_type == 'tiny':
        dataset = create_dataset('tiny')
    elif dataset_type == 'math':
        dataset = create_dataset('math', num_problems=100)
    elif dataset_type == 'code':
        dataset = create_dataset('code', use_subset=5)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create environment
    print("Creating environment...")
    if dataset_type in ['tiny', 'math']:
        reward_fn = SimpleRewardFunction()
        env = CodeGenerationEnvironment(tokenizer, dataset, max_length=max_length)
    else:
        env = CodeGenerationEnvironment(tokenizer, dataset, max_length=max_length)
    
    # Create models
    print("Creating models...")
    vocab_size = tokenizer.vocab_size
    policy = SimpleTokenPolicy(vocab_size, d_model=256, num_layers=2).to(device)
    value_net = SimpleValueNetwork(vocab_size, d_model=256, num_layers=2).to(device)
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Value parameters: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # Create trainer
    trainer = SimplePPOTrainer(policy, value_net, lr=3e-4)
    
    # Create buffer
    buffer = RolloutBuffer()
    
    # Create evaluator
    evaluator = Evaluator(dataset, env, policy, tokenizer)
    
    # Training loop
    print("\nStarting training...")
    best_reward = float('-inf')
    
    for iteration in range(num_iterations):
        policy.train()
        value_net.train()
        
        # Collect episodes
        episode_rewards = []
        
        for episode in range(episodes_per_iteration):
            # Reset environment
            state = env.reset()
            done = False
            episode_reward = 0
            
            # Run episode
            while not done:
                # Get token_ids and move to device
                token_ids = state['token_ids'].unsqueeze(0).to(device)
                
                # Get action from policy
                with torch.no_grad():
                    dist = policy.get_action_distribution(token_ids)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    # Get value estimate
                    value = value_net(token_ids)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action.item())
                
                # Store in buffer
                buffer.add(
                    state=state,
                    action=action.item(),
                    reward=reward,
                    log_prob=log_prob.item(),
                    value=value.item(),
                    done=done
                )
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        # Update policy
        if len(buffer) > 0:
            stats = trainer.update(buffer, epochs=4)
            buffer.clear()
        else:
            stats = {}
        
        # Logging
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        
        if (iteration + 1) % 10 == 0:
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Policy Loss: {stats.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {stats.get('value_loss', 0):.4f}")
            print(f"  Entropy: {stats.get('entropy', 0):.4f}")
            
            # Evaluate
            eval_stats = evaluator.evaluate(num_episodes=5)
            print(f"  Eval Avg Reward: {eval_stats['avg_reward']:.2f}")
            print(f"  Eval Success Rate: {eval_stats['success_rate']:.2%}")
            
            # Generate samples
            print("\n  Sample generations:")
            samples = evaluator.generate_samples(num_samples=3)
            for i, sample in enumerate(samples):
                print(f"    {i+1}. {sample[:100]}...")
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save({
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_net.state_dict(),
                'best_reward': best_reward
            }, os.path.join(save_dir, 'best_model.pt'))
    
    print("\nTraining complete!")
    print(f"Best reward: {best_reward:.2f}")
    
    return policy, value_net


def main():
    """Main entry point"""
    print("="*60)
    print(f"Training on {args.dataset.upper()} Dataset")
    print(f"Device: {get_device()}")
    print(f"Iterations: {args.iterations}")
    print("="*60)
    
    train_simple_policy(
        dataset_type=args.dataset,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        max_length=args.max_length,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
    
    # Uncomment to train on math after tiny works
    # print("\n" + "="*60)
    # print("Training on Math Dataset")
    # print("="*60)
    # train_simple_policy(
    #     dataset_type='math',
    #     num_iterations=200,
    #     episodes_per_iteration=10,
    #     max_length=50
    # )

