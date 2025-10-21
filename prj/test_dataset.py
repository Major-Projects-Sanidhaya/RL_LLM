"""
Test dataset loading
Run this first to verify everything is set up correctly
"""

from data.dataset_loader import TinyDataset, SimpleMathDataset, CodeProblemDataset

def test_tiny_dataset():
    print("="*60)
    print("Testing Tiny Dataset")
    print("="*60)
    
    dataset = TinyDataset()
    print(f"Dataset size: {len(dataset)}")
    
    # Get random problem
    problem = dataset.get_random_problem()
    print(f"\nSample problem:")
    print(f"  Prompt: {problem['prompt']}")
    print(f"  Answer: {problem['answer']}")
    
    # Test evaluation
    correct_answer = problem['answer']
    reward = dataset.compute_reward(correct_answer, problem)
    print(f"\nReward for correct answer '{correct_answer}': {reward}")
    
    wrong_answer = "wrong"
    reward = dataset.compute_reward(wrong_answer, problem)
    print(f"Reward for wrong answer '{wrong_answer}': {reward}")


def test_math_dataset():
    print("\n" + "="*60)
    print("Testing Math Dataset")
    print("="*60)
    
    dataset = SimpleMathDataset(num_problems=10)
    print(f"Dataset size: {len(dataset)}")
    
    # Show a few problems
    for i in range(3):
        problem = dataset[i]
        print(f"\nProblem {i+1}:")
        print(f"  Prompt: {problem['prompt']}")
        print(f"  Answer: {problem['answer']}")
        
        # Test evaluation
        reward = dataset.compute_reward(problem['answer'], problem)
        print(f"  Reward for correct: {reward}")


def test_code_dataset():
    print("\n" + "="*60)
    print("Testing Code Dataset")
    print("="*60)
    
    try:
        dataset = CodeProblemDataset(use_subset=2)
        print(f"Dataset size: {len(dataset)}")
        
        # Show first problem
        problem = dataset[0]
        print(f"\nFirst problem:")
        print(f"  Task ID: {problem['task_id']}")
        print(f"  Prompt:\n{problem['prompt'][:200]}...")
        
    except Exception as e:
        print(f"Error loading code dataset: {e}")
        print("This is normal if you don't have internet connection")


if __name__ == "__main__":
    test_tiny_dataset()
    test_math_dataset()
    test_code_dataset()