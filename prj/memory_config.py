"""
Memory optimization settings for GPU training
Add this at the beginning of your notebook to reduce memory usage
"""

import torch
import gc

def optimize_memory():
    """Configure PyTorch for memory-efficient training"""

    # Enable memory efficient attention if available
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set memory allocator configuration
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Enable gradient checkpointing (saves memory during backprop)
    print("✓ Memory optimization enabled")
    print(f"  - TF32 enabled for faster computation")
    print(f"  - Expandable segments for memory fragmentation")

    return True

def clear_memory():
    """Clear GPU cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")

def get_recommended_batch_config(available_memory_gb=40):
    """Get recommended batch sizes based on available GPU memory"""

    if available_memory_gb >= 80:
        return {
            'episodes_per_iteration': 5,
            'max_length': 512,
            'message': 'A100-80GB: Full capacity'
        }
    elif available_memory_gb >= 40:
        return {
            'episodes_per_iteration': 3,  # Reduce episodes
            'max_length': 256,  # Reduce sequence length
            'message': 'A100-40GB: Reduced batch size'
        }
    else:
        return {
            'episodes_per_iteration': 2,
            'max_length': 128,
            'message': 'V100-32GB: Minimal batch size'
        }

# Instructions for notebook:
"""
ADD THIS TO THE TOP OF YOUR NOTEBOOK (after imports):

from memory_config import optimize_memory, clear_memory, get_recommended_batch_config

# Optimize memory
optimize_memory()

# Get recommended config
config = get_recommended_batch_config(80)  # Use 80 for A100-80GB
print(f"Using config: {config['message']}")

# Update training parameters
EPISODES_PER_ITERATION = config['episodes_per_iteration']
MAX_LENGTH = config['max_length']

# Clear memory periodically during training
# Add this inside your training loop every 50 iterations:
if iteration % 50 == 0:
    clear_memory()
"""
