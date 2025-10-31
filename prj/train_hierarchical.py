"""
Hierarchical Policy Training Script (Week 6 Implementation)

This script trains the two-level hierarchical policy for text generation:
- High-level policy: Generates semantic intentions/plans
- Low-level policy: Generates tokens conditioned on intentions

Training approach:
- Uses task-specific environments (Q&A, conversation)
- Multi-component reward function (fluency, coherence, task completion, safety)
- Hierarchical PPO for stable training
- Progressive evaluation on simple tasks

Week 6 Milestone Goal:
"Working hierarchical system that can generate coherent text for simple tasks
like question-answering and basic conversation"
"""

import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Train Hierarchical RL-LLM')
parser.add_argument('--task', type=str, default='qa', choices=['qa', 'conversation'])
parser.add_argument('--iterations', type=int, default=200)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--max-length', type=int, default=100)
parser.add_argument('--save-dir', type=str, default='checkpoints/hierarchical')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--intention-dim', type=int, default=64)

args = parser.parse_args()

# Imports
from environments.task_specific_envs import QuestionAnsweringEnvironment, ConversationEnvironment
from environments.reward_functions import MultiComponentReward
from models.hierarchical_policy import HierarchicalPolicy
from training.hierarchical_ppo_trainer import HierarchicalPPOTrainer
from training.buffer import RolloutBuffer
from training.utils import set_seed, get_device
from data.dataset_loader import create_dataset


def evaluate_hierarchical_policy(policy, env, tokenizer, num_episodes=5):
    """
    Evaluate hierarchical policy on task

    Args:
        policy: Trained hierarchical policy
        env: Environment to test on
        tokenizer: Tokenizer for decoding
        num_episodes: Number of test episodes

    Returns:
        Dictionary of evaluation metrics
    """
    policy.eval()
    device = next(policy.parameters()).device

    total_reward = 0.0
    successful_episodes = 0
    generations = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            token_ids = state['token_ids'].unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = policy(token_ids, return_intention=True)
                action_dist = torch.distributions.Categorical(logits=outputs['logits'])
                action = action_dist.sample()

            next_state, reward, done, info = env.step(action.item())
            state = next_state
            episode_reward += reward

        total_reward += episode_reward

        # Count as successful if positive reward
        if episode_reward > 0:
            successful_episodes += 1

        # Save generation
        generations.append(info['text'])

    return {
        'avg_reward': total_reward / num_episodes,
        'success_rate': successful_episodes / num_episodes,
        'generations': generations
    }


def train_hierarchical_policy(
    task_type='qa',
    num_iterations=200,
    episodes_per_iteration=10,
    max_length=100,
    save_dir='checkpoints/hierarchical',
    intention_dim=64
):
    """
    Train hierarchical policy on task-specific environment

    This implements the Week 6 milestone:
    "Working hierarchical system that can generate coherent text for simple tasks
    like question-answering and basic conversation"

    Args:
        task_type: 'qa' or 'conversation'
        num_iterations: Number of training iterations
        episodes_per_iteration: Episodes per iteration
        max_length: Maximum generation length
        save_dir: Checkpoint directory
        intention_dim: Dimension of intention vectors

    Returns:
        Trained hierarchical policy
    """
    # Setup
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"Week 6: Hierarchical Policy Training")
    print(f"Task: {task_type.upper()}")
    print(f"{'='*60}\n")

    os.makedirs(save_dir, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset for task
    print(f"Creating dataset for {task_type}...")
    if task_type == 'qa':
        # Use TinyDataset for simple Q&A
        dataset = create_dataset('tiny')
    else:
        dataset = None  # Conversation doesn't need dataset

    # Create task-specific environment
    print(f"Creating {task_type} environment...")
    if task_type == 'qa':
        env = QuestionAnsweringEnvironment(
            tokenizer=tokenizer,
            dataset=dataset,
            max_length=max_length
        )
    else:  # conversation
        env = ConversationEnvironment(
            tokenizer=tokenizer,
            max_length=max_length
        )

    # Create multi-component reward function
    print("Creating multi-component reward function...")
    reward_fn = MultiComponentReward(tokenizer=tokenizer)

    # Create hierarchical policy
    print(f"Creating hierarchical policy (intention_dim={intention_dim})...")
    vocab_size = tokenizer.vocab_size
    policy = HierarchicalPolicy(
        vocab_size=vocab_size,
        d_model=256,
        intention_dim=intention_dim,
        num_layers=4,
        nhead=4
    ).to(device)

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create hierarchical PPO trainer
    print("Creating hierarchical PPO trainer...")
    trainer = HierarchicalPPOTrainer(policy, lr=3e-4)

    # Create buffer
    buffer = RolloutBuffer()

    # Training loop
    print(f"\nStarting training for {num_iterations} iterations...")
    print(f"Episodes per iteration: {episodes_per_iteration}\n")

    best_reward = float('-inf')
    training_history = []

    for iteration in range(num_iterations):
        policy.train()

        # Collect episodes using hierarchical policy
        episode_rewards = []

        for episode in range(episodes_per_iteration):
            # Collect episode with intentions stored
            episode_reward = trainer.collect_episode_with_intentions(
                env=env,
                buffer=buffer,
                max_steps=max_length
            )
            episode_rewards.append(episode_reward)

        # Update policy
        if len(buffer) > 0:
            stats = trainer.update(buffer, epochs=4)
            buffer.clear()
        else:
            stats = {}

        # Logging
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        training_history.append(avg_reward)

        if (iteration + 1) % args.log_interval == 0:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            print(f"  Avg Training Reward: {avg_reward:.2f}")
            print(f"  Policy Loss: {stats.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {stats.get('value_loss', 0):.4f}")
            print(f"  Entropy: {stats.get('entropy', 0):.4f}")
            print(f"  Intention Loss: {stats.get('intention_loss', 0):.4f}")

            # Evaluate
            print("\n  Evaluating...")
            eval_stats = evaluate_hierarchical_policy(
                policy, env, tokenizer, num_episodes=5
            )
            print(f"  Eval Avg Reward: {eval_stats['avg_reward']:.2f}")
            print(f"  Eval Success Rate: {eval_stats['success_rate']:.1%}")

            # Show sample generations
            print("\n  Sample Generations:")
            for i, gen in enumerate(eval_stats['generations'][:3]):
                # Truncate for display
                display_text = gen[:150] + "..." if len(gen) > 150 else gen
                print(f"    {i+1}. {display_text}")

        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint = {
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'best_reward': best_reward,
                'task_type': task_type,
                'intention_dim': intention_dim
            }
            torch.save(checkpoint, os.path.join(save_dir, f'best_{task_type}_model.pt'))
            print(f"\n  ✓ New best model saved! Reward: {best_reward:.2f}")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Training Complete - Final Evaluation")
    print(f"{'='*60}\n")

    final_eval = evaluate_hierarchical_policy(
        policy, env, tokenizer, num_episodes=10
    )

    print(f"Final Average Reward: {final_eval['avg_reward']:.2f}")
    print(f"Final Success Rate: {final_eval['success_rate']:.1%}")

    print("\nFinal Sample Generations:")
    for i, gen in enumerate(final_eval['generations'][:5]):
        print(f"\n{i+1}. {gen}")

    # Save final model
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'final_reward': final_eval['avg_reward'],
        'success_rate': final_eval['success_rate'],
        'task_type': task_type,
        'training_history': training_history
    }, os.path.join(save_dir, f'final_{task_type}_model.pt'))

    print(f"\n✓ Training complete! Models saved to {save_dir}")

    return policy


def main():
    """Main entry point"""
    print(f"\n{'='*70}")
    print("WEEK 6 MILESTONE: Hierarchical Policy Training")
    print("Goal: Generate coherent text for Q&A and conversation tasks")
    print(f"{'='*70}\n")

    train_hierarchical_policy(
        task_type=args.task,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        max_length=args.max_length,
        save_dir=args.save_dir,
        intention_dim=args.intention_dim
    )


if __name__ == "__main__":
    main()
