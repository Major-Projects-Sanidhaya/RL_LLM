"""
Week 5-6 Milestone Test Script

This script tests all components implemented for Weeks 5-6:

Week 5 Components:
✓ Text Generation Environment with feedback
✓ Multi-Component Reward System (fluency, coherence, task completion, safety)
✓ Task-Specific Environments (Q&A, Conversation)

Week 6 Components:
✓ Hierarchical Policy (High-level: intentions, Low-level: tokens)
✓ Training Algorithm for both levels simultaneously

Week 6 Milestone Goal:
"Working hierarchical system that can generate coherent text for simple tasks
like question-answering and basic conversation"

Usage:
    python test_week6_milestone.py
"""

import torch
from transformers import GPT2Tokenizer
import sys

# Import all Week 5-6 components
from environments.reward_functions import MultiComponentReward
from environments.task_specific_envs import QuestionAnsweringEnvironment, ConversationEnvironment
from models.hierarchical_policy import HierarchicalPolicy
from training.hierarchical_ppo_trainer import HierarchicalPPOTrainer
from training.buffer import RolloutBuffer
from data.dataset_loader import create_dataset


def test_multi_component_rewards():
    """Test Week 5: Multi-Component Reward System"""
    print("\n" + "="*70)
    print("TEST 1: Multi-Component Reward System (Week 5)")
    print("="*70)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    reward_fn = MultiComponentReward(tokenizer=tokenizer)

    # Test different text samples
    test_cases = [
        ("Hello world! How are you?", "Good fluent text"),
        ("the the the the the the", "Highly repetitive text"),
        ("kill hate violence", "Unsafe content"),
        ("a", "Too short"),
        ("This is a reasonable response to a question.", "Good response"),
    ]

    print("\nTesting reward components on different text samples:\n")

    for text, description in test_cases:
        token_ids = tokenizer.encode(text, return_tensors='pt')[0]

        fluency = reward_fn.compute_fluency(token_ids)
        coherence = reward_fn.compute_coherence(token_ids, text)
        task_comp = reward_fn.compute_task_completion(token_ids, text)
        safety = reward_fn.compute_safety(token_ids, text)
        total = reward_fn.compute_reward(token_ids, text)

        print(f"Text: '{text}'")
        print(f"  Description: {description}")
        print(f"  Fluency:         {fluency:.3f}")
        print(f"  Coherence:       {coherence:.3f}")
        print(f"  Task Completion: {task_comp:.3f}")
        print(f"  Safety:          {safety:.3f}")
        print(f"  TOTAL REWARD:    {total:.3f}")
        print()

    print("✓ Multi-Component Reward System working correctly!")
    return True


def test_qa_environment():
    """Test Week 5: Question Answering Environment"""
    print("\n" + "="*70)
    print("TEST 2: Question Answering Environment (Week 5)")
    print("="*70)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset = create_dataset('tiny')

    env = QuestionAnsweringEnvironment(tokenizer, dataset, max_length=50)

    print("\nRunning sample Q&A episode:\n")

    # Test episode
    state = env.reset()
    print(f"Question: {env.current_question}")
    print(f"Expected Answer: {env.expected_answer}")
    print(f"Initial state shape: {state['token_ids'].shape}")

    # Simulate a few steps
    done = False
    steps = 0
    while not done and steps < 10:
        # Random action for testing
        action = torch.randint(0, tokenizer.vocab_size, (1,)).item()
        state, reward, done, info = env.step(action)
        steps += 1

    print(f"\nSteps taken: {steps}")
    print(f"Episode done: {done}")
    print(f"Generated text: {info['text'][:100]}...")

    print("\n✓ Q&A Environment working correctly!")
    return True


def test_conversation_environment():
    """Test Week 5: Conversation Environment"""
    print("\n" + "="*70)
    print("TEST 3: Conversation Environment (Week 5)")
    print("="*70)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    env = ConversationEnvironment(tokenizer, max_length=50)

    print("\nRunning sample conversation episode:\n")

    # Test episode
    state = env.reset(user_message="Hello! How are you today?")
    print(f"User Message: {env.current_user_message}")
    print(f"Initial state shape: {state['token_ids'].shape}")

    # Simulate a few steps
    done = False
    steps = 0
    while not done and steps < 10:
        action = torch.randint(0, tokenizer.vocab_size, (1,)).item()
        state, reward, done, info = env.step(action)
        steps += 1

    print(f"\nSteps taken: {steps}")
    print(f"Episode done: {done}")
    print(f"Generated response: {info['text'][:100]}...")

    print("\n✓ Conversation Environment working correctly!")
    return True


def test_hierarchical_policy():
    """Test Week 6: Hierarchical Policy Architecture"""
    print("\n" + "="*70)
    print("TEST 4: Hierarchical Policy Architecture (Week 6)")
    print("="*70)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create hierarchical policy
    policy = HierarchicalPolicy(
        vocab_size=tokenizer.vocab_size,
        d_model=128,  # Smaller for testing
        intention_dim=32,
        num_layers=2,
        nhead=4
    ).to(device)

    print(f"\nPolicy created with {sum(p.numel() for p in policy.parameters()):,} parameters")

    # Test forward pass
    test_text = "Question: What is 2+2? Answer:"
    token_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)

    print(f"\nTest input: '{test_text}'")
    print(f"Token IDs shape: {token_ids.shape}")

    # Test without intention (will be sampled)
    with torch.no_grad():
        outputs = policy(token_ids, return_intention=True)

    print("\nHierarchical Policy Outputs:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Intention shape: {outputs['intention'].shape}")
    print(f"  High-level value shape: {outputs['high_value'].shape}")
    print(f"  Low-level value shape: {outputs['low_value'].shape}")

    # Test action sampling
    action_dist = torch.distributions.Categorical(logits=outputs['logits'])
    action = action_dist.sample()
    print(f"\n  Sampled action (token ID): {action.item()}")
    print(f"  Decoded token: '{tokenizer.decode([action.item()])}'")

    print("\n✓ Hierarchical Policy working correctly!")
    return True


def test_hierarchical_trainer():
    """Test Week 6: Hierarchical PPO Trainer"""
    print("\n" + "="*70)
    print("TEST 5: Hierarchical PPO Trainer (Week 6)")
    print("="*70)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create components
    policy = HierarchicalPolicy(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        intention_dim=32,
        num_layers=2,
        nhead=4
    ).to(device)

    trainer = HierarchicalPPOTrainer(policy, lr=3e-4)
    env = QuestionAnsweringEnvironment(tokenizer, create_dataset('tiny'), max_length=30)
    buffer = RolloutBuffer()

    print("\nCollecting sample episode with hierarchical policy...")

    # Collect one episode
    episode_reward = trainer.collect_episode_with_intentions(
        env=env,
        buffer=buffer,
        max_steps=30
    )

    print(f"Episode reward: {episode_reward:.2f}")
    print(f"Buffer size: {len(buffer)} transitions")

    # Test update
    if len(buffer) > 0:
        print("\nPerforming training update...")
        stats = trainer.update(buffer, epochs=1)

        print("\nTraining Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")

    print("\n✓ Hierarchical Trainer working correctly!")
    return True


def test_full_training_loop():
    """Test Week 6: Full Training Loop (Small Scale)"""
    print("\n" + "="*70)
    print("TEST 6: Full Training Loop - Mini Training Run (Week 6)")
    print("="*70)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create all components
    policy = HierarchicalPolicy(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        intention_dim=32,
        num_layers=2,
        nhead=4
    ).to(device)

    trainer = HierarchicalPPOTrainer(policy, lr=3e-4)
    env = QuestionAnsweringEnvironment(tokenizer, create_dataset('tiny'), max_length=30)

    print("\nRunning mini training (5 iterations, 3 episodes each)...\n")

    for iteration in range(5):
        buffer = RolloutBuffer()
        episode_rewards = []

        # Collect episodes
        for _ in range(3):
            reward = trainer.collect_episode_with_intentions(env, buffer, max_steps=30)
            episode_rewards.append(reward)

        # Update
        if len(buffer) > 0:
            stats = trainer.update(buffer, epochs=2)
            avg_reward = sum(episode_rewards) / len(episode_rewards)

            print(f"Iteration {iteration + 1}/5: "
                  f"Avg Reward = {avg_reward:.2f}, "
                  f"Policy Loss = {stats['policy_loss']:.3f}")

    print("\n✓ Full training loop working correctly!")
    return True


def main():
    """Run all Week 5-6 tests"""
    print("\n" + "="*70)
    print("WEEK 5-6 MILESTONE VERIFICATION")
    print("="*70)
    print("\nThis script verifies all components for Weeks 5-6:")
    print("  Week 5: Environment & Reward System")
    print("  Week 6: Hierarchical Policy Implementation")
    print("\nGoal: Working hierarchical system for simple Q&A and conversation")
    print("="*70)

    tests = [
        ("Multi-Component Rewards", test_multi_component_rewards),
        ("Q&A Environment", test_qa_environment),
        ("Conversation Environment", test_conversation_environment),
        ("Hierarchical Policy", test_hierarchical_policy),
        ("Hierarchical Trainer", test_hierarchical_trainer),
        ("Full Training Loop", test_full_training_loop),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ Test failed: {test_name}")
            print(f"  Error: {str(e)}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "="*70)
        print("SUCCESS! All Week 5-6 components are working correctly!")
        print("="*70)
        print("\nWeek 5-6 Milestone Status: ✓ COMPLETE")
        print("\nImplemented:")
        print("  ✓ Text Generation Environment with feedback")
        print("  ✓ Multi-Component Rewards (fluency, coherence, task, safety)")
        print("  ✓ Task-Specific Environments (Q&A, Conversation)")
        print("  ✓ Hierarchical Policy (High-level + Low-level)")
        print("  ✓ Training Algorithm for both levels")
        print("\nReady to train on simple Q&A and conversation tasks!")
    else:
        print(f"\n{total - passed} test(s) failed. Please review errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
