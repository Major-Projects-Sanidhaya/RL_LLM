"""
Minimal quickstart to test everything works
"""

import torch
from transformers import GPT2Tokenizer
from data.dataset_loader import TinyDataset
from environments.code_generation_env import CodeGenerationEnvironment
from models.networks import SimpleTokenPolicy

def quickstart():
    """Minimal test of the full pipeline"""
    print("QuickStart Test")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("   ✓ Tokenizer loaded")
    
    # Create dataset
    print("\n2. Creating dataset...")
    dataset = TinyDataset()
    print(f"   ✓ Dataset created with {len(dataset)} problems")
    
    # Create environment
    print("\n3. Creating environment...")
    env = CodeGenerationEnvironment(tokenizer, dataset, max_length=20)
    print("   ✓ Environment created")
    
    # Create policy
    print("\n4. Creating policy...")
    policy = SimpleTokenPolicy(tokenizer.vocab_size, d_model=128, num_layers=2).to(device)
    print(f"   ✓ Policy created with {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    # Run one episode
    print("\n5. Running one episode...")
    state = env.reset()
    print(f"   Initial prompt: {tokenizer.decode(state['token_ids'])}")
    
    done = False
    steps = 0
    while not done and steps < 20:
        token_ids = state['token_ids'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            dist = policy.get_action_distribution(token_ids)
            action = dist.sample()
        
        state, reward, done, info = env.step(action.item())
        steps += 1
    
    print(f"   Generated: {info['text']}")
    print(f"   Reward: {reward}")
    print(f"   Steps: {steps}")
    print("\n✓ QuickStart complete! Everything is working.")
    print("\nNext steps:")
    print("  1. Run: python test_dataset.py")
    print("  2. Run: python main.py")


if __name__ == "__main__":
    quickstart()