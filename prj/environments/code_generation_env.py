# File: data/dataset_loader.py

"""
WHAT TO CODE:
- Load HumanEval dataset
- Parse problems and test cases
- Provide prompts for RL environment
- Evaluate code correctness (for rewards)

HINTS:
- Use datasets library from HuggingFace
- Store problems in memory (only 164 problems)
- Don't load any solutions - we're doing pure RL!
"""

from datasets import load_dataset
import random

class CodeProblemDataset:
    """
    WHAT THIS DOES:
    - Loads programming problems
    - Provides random prompts for training
    - Executes code to check correctness
    """
    
    def __init__(self, split='test', use_subset=None):
        """
        WHAT TO CODE:
        - Load HumanEval from HuggingFace
        - Extract only: task_id, prompt, test cases, entry_point
        - Store in list for easy sampling
        
        HINTS:
        - pip install datasets
        - Use: load_dataset("openai_humaneval")
        - use_subset: if not None, only use first N problems for faster iteration
        
        EXAMPLE PROBLEM STRUCTURE:
        {
            'task_id': 'HumanEval/0',
            'prompt': 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n   
              """ #Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    ...',
        """
            'canonical_solution': '...',  # DON'T USE THIS - we're doing RL!
            'test': 'def check(candidate):\n    assert candidate(...) == ...',
            'entry_point': 'has_close_elements'
        }
        """
        
        # TODO: Load dataset
        # dataset = load_dataset("openai_humaneval", split="test")
        
        # TODO: Extract only what we need
        # self.problems = []
        # for example in dataset:
        #     problem = {
        #         'task_id': example['task_id'],
        #         'prompt': example['prompt'],
        #         'test': example['test'],
        #         'entry_point': example['entry_point']
        #     }
        #     self.problems.append(problem)
        
        # TODO: If use_subset specified, only keep first N
        # if use_subset:
        #     self.problems = self.problems[:use_subset]
        
        pass
    
    def get_random_problem(self):
        """
        WHAT TO CODE:
        - Return a random problem for training
        
        HINTS:
        - Use random.choice(self.problems)
        - Return the full problem dict
        """
        pass
    
    def get_problem_by_id(self, task_id):
        """
        WHAT TO CODE:
        - Return specific problem by task_id
        - Useful for evaluation
        
        HINTS:
        - Search through self.problems
        - Return problem where problem['task_id'] == task_id
        """
        pass
    
    def evaluate_code(self, code, problem):
        """
        WHAT TO CODE:
        - Execute the generated code
        - Run test cases
        - Return (passed, total_tests, error_message)
        
        HINTS:
        - This is tricky - need to execute code safely
        - Use exec() with timeout
        - Catch exceptions
        - Return tuple: (num_passed, num_total)
        
        IMPORTANT SAFETY:
        - DON'T run untrusted code directly in production
        - For research: ok with subprocess timeout
        - For deployment: use Docker containers
        """
        
        # TODO: Combine generated code with test code
        # full_code = code + '\n' + problem['test'] + '\n' + f"check({problem['entry_point']})"
        
        # TODO: Try to execute
        # try:
        #     exec(full_code, {})
        #     return (True, 1, None)  # Success
        # except Exception as e:
        #     return (False, 0, str(e))  # Failed
        
        pass
    
    def compute_reward(self, code, problem):
        """
        WHAT TO CODE:
        - Wrapper around evaluate_code
        - Convert test results to reward signal
        
        HINTS:
        - If code passes: reward = +10
        - If code fails: reward = -1
        - If syntax error: reward = -5
        - Can add partial credit later
        """
        
        # TODO: Evaluate code
        # passed, total, error = self.evaluate_code(code, problem)
        
        # TODO: Assign rewards
        # if passed:
        #     return 10.0
        # elif error and "SyntaxError" in error:
        #     return -5.0
        # else:
        #     return -1.0
        
        pass
    
    def __len__(self):
        """Return number of problems"""
        return len(self.problems)
    
    def __getitem__(self, idx):
        """Get problem by index"""
        return self.problems[idx]


# ============================================
# SIMPLER ALTERNATIVE: Use Simple Math Problems First
# ============================================

class SimpleMathDataset:
    """
    WHAT THIS DOES:
    - Generate simple arithmetic problems
    - Much easier to start with than code
    - Good for debugging your RL setup
    
    USE THIS FIRST to test your RL implementation!
    """
    
    def __init__(self, num_problems=1000):
        """
        WHAT TO CODE:
        - Generate random math problems
        - Store them in memory
        
        HINTS:
        - Problems like: "What is 5 + 3?"
        - Answer: "8"
        - Easy to evaluate!
        """
        
        # TODO: Generate problems
        # self.problems = []
        # for i in range(num_problems):
        #     a = random.randint(1, 20)
        #     b = random.randint(1, 20)
        #     problem = {
        #         'prompt': f"What is {a} + {b}? Answer:",
        #         'answer': str(a + b),
        #         'task_id': f'math_{i}'
        #     }
        #     self.problems.append(problem)
        
        pass
    
    def get_random_problem(self):
        """Return random math problem"""
        return random.choice(self.problems)
    
    def evaluate_answer(self, generated_text, problem):
        """
        WHAT TO CODE:
        - Extract answer from generated text
        - Compare with correct answer
        - Return True/False
        
        HINTS:
        - Generated might be: "The answer is 8" or just "8"
        - Use string matching or regex
        - Be lenient in parsing
        """
        
        # TODO: Extract number from generated_text
        # import re
        # numbers = re.findall(r'\d+', generated_text)
        # if numbers and numbers[0] == problem['answer']:
        #     return True
        # return False
        
        pass
    
    def compute_reward(self, generated_text, problem):
        """
        WHAT TO CODE:
        - Convert evaluation to reward
        
        HINTS:
        - Correct: +10
        - Wrong: -1
        """
        
        # TODO: Evaluate and return reward
        # if self.evaluate_answer(generated_text, problem):
        #     return 10.0
        # return -1.0
        
        pass