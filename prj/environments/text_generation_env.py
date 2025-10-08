# File: environments/text_generation_env.py

class TextGenerationEnvironment:
    """
    WHAT TO CODE:
    - Initialize with tokenizer (use GPT2Tokenizer from transformers)
    - Track current sequence of tokens
    - Define max_length (start with 50 tokens)
    - Store the prompt/context
    
    HINTS:
    - Use a list to store current_sequence
    - Keep track of step_count
    - Define done flag for episode termination
    """
    
    def __init__(self, tokenizer, max_length, reward_fn):
        # TODO: Initialize tokenizer
        # TODO: Set max_length
        # TODO: Store reward_fn for later use
        # TODO: Initialize empty current_sequence list
        # TODO: Set step_count = 0
        pass
    
    def reset(self, prompt_text):
        """
        WHAT TO CODE:
        - Convert prompt_text to token IDs using tokenizer
        - Set current_sequence to these initial tokens
        - Reset step_count to 0
        - Return initial state
        
        HINTS:
        - State should be a dictionary with:
          {'token_ids': current_sequence, 'step': step_count}
        - Use tokenizer.encode(prompt_text, return_tensors='pt')
        """
        pass
    
    def step(self, action):
        """
        WHAT TO CODE:
        - Append action (token_id) to current_sequence
        - Increment step_count
        - Compute reward using reward_fn
        - Check if episode is done (max_length or EOS token)
        - Return (next_state, reward, done, info)
        
        HINTS:
        - action is an integer (token ID from 0 to vocab_size-1)
        - done = (step_count >= max_length) or (action == eos_token_id)
        - info dict can contain decoded text for debugging
        """
        pass
    
    def get_state(self):
        """
        WHAT TO CODE:
        - Return current state as dictionary
        
        HINTS:
        - Include token_ids, step_count, and maybe text for debugging
        """
        pass