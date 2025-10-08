# File: environments/reward_functions.py

class SimpleRewardFunction:
    """
    WHAT TO CODE:
    - Start with JUST fluency reward (use a small pretrained LM)
    - Later add more components
    
    HINTS:
    - Load GPT2-small from transformers as reference model
    - Compute perplexity of current sequence
    - Lower perplexity = higher reward
    """
    
    def __init__(self):
        # TODO: Load small language model (GPT2-small)
        # TODO: Set device (cuda if available)
        # TODO: Initialize any constants
        pass
    
    def compute_fluency_reward(self, token_ids):
        """
        WHAT TO CODE:
        - Convert token_ids to tensor
        - Get log probabilities from reference LM
        - Compute average log prob (negative perplexity)
        - Return as reward (higher = better)
        
        HINTS:
        - Use model(input_ids).logits
        - Use torch.nn.functional.cross_entropy with reduction='none'
        - Take negative of loss (we want to maximize probability)
        - Scale to reasonable range (multiply by 0.1 or similar)
        """
        pass
    
    def compute_reward(self, token_ids):
        """
        WHAT TO CODE:
        - For now, just return fluency_reward
        - Later: add task_reward, safety_reward, etc.
        
        HINTS:
        - Keep this simple at first
        - Add components one at a time
        """
        reward = self.compute_fluency_reward(token_ids)
        return reward