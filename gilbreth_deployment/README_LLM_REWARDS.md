# LLM-Based Reward Evaluation for RL Code Generation

This enhancement adds GPT-4 and Claude-based evaluation of code comprehensibility to the training process.

## Features

- **Hybrid Rewards**: Combines fast heuristics with LLM evaluation
- **Dual LLM Evaluation**: Uses both GPT-4 and Claude for robust scoring
- **Comprehensibility Focus**: Evaluates code readability, structure, naming, documentation
- **Smart Caching**: Caches LLM responses to reduce API costs
- **Configurable**: Can use heuristics-only, LLMs-only, or hybrid approach

## Setup

### 1. Install Dependencies

```bash
pip install openai anthropic
```

### 2. Set API Keys

On Gilbreth, add to your submit script or `.bashrc`:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Upload Files

Upload these files to Gilbreth:
- `llm_reward_evaluator.py` - LLM evaluation module
- `train_code_generation.py` - Updated training script

```bash
scp llm_reward_evaluator.py train_code_generation.py \
    shars11@gilbreth.rcac.purdue.edu:/scratch/gilbreth/shars11/RL-LLM/
```

## Usage

### Basic Training (Heuristics Only - FREE)

```bash
python train_code_generation.py \
    --num_iterations 1000 \
    --episodes_per_iter 6 \
    --subset_size 20
```

This uses the **improved heuristic rewards** that prevent the model from reward hacking.

### Training with LLM Rewards

```bash
python train_code_generation.py \
    --num_iterations 1000 \
    --episodes_per_iter 6 \
    --subset_size 20 \
    --use_llm_rewards \
    --use_gpt \
    --use_claude \
    --llm_weight 0.7 \
    --llm_threshold 5.0
```

**Parameters:**
- `--use_llm_rewards`: Enable LLM evaluation
- `--use_gpt`: Use GPT-4 mini for evaluation
- `--use_claude`: Use Claude Haiku for evaluation
- `--llm_weight`: Weight for LLM score (0-1), default 0.7
- `--llm_threshold`: Only use LLM for code scoring above this heuristic threshold

### Cost-Effective Training

To minimize API costs while still using LLM feedback:

```bash
python train_code_generation.py \
    --use_llm_rewards \
    --use_gpt \
    --llm_weight 0.5 \
    --llm_threshold 8.0  # Only evaluate promising code
```

This only calls the LLM for code that passes basic heuristics (reward > 8.0).

## Reward Components

### Heuristic Rewards (Fast, Free)

The improved heuristics check for:
1. **Non-empty code** (+2.0) - Prevents reward hacking
2. **Function definition** (+4.0 or -5.0) - Must have `def`
3. **Return statement** (+3.0 or -1.0)
4. **Good length** (+2.0) - 3-30 lines of actual code
5. **Control flow** (+2.0) - Has if/for/while
6. **Indentation** (+1.0) - Proper code structure

**Max heuristic reward:** ~14 points

### LLM Evaluation (Comprehensive, Costs API calls)

LLMs evaluate code on:
1. **Correctness** (0-10) - Does it solve the problem?
2. **Comprehensibility** (0-10) - Easy to read and understand?
3. **Code Structure** (0-10) - Well-organized logic?
4. **Variable Naming** (0-10) - Descriptive names?
5. **Documentation** (0-10) - Complex parts explained?
6. **Efficiency** (0-10) - Reasonably efficient?
7. **Completeness** (0-10) - Fully addresses requirements?

**LLM scores are averaged** between GPT-4 and Claude, then scaled to reward points.

### Final Reward Calculation

```
final_reward = (
    heuristic_weight * heuristic_reward +
    0.3 * execution_reward +  # Syntax/execution bonus
    llm_weight * llm_reward
)
```

## Example Comparison

### Bad Code (Reward Hacking - OLD SYSTEM)
```python


```
**Old reward:** 1.0 (got points for "good length")
**New heuristic:** -10.0 (heavy penalty for empty code)

### Minimal Code
```python
def add(a, b):
    return a + b
```
**Heuristic:** ~9.0
**LLM (GPT-4):** ~6.5/10 (correct but minimal)
**LLM (Claude):** ~7.0/10
**Final:** ~13 points

### Good Code
```python
def add_two_numbers(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    result = a + b
    return result
```
**Heuristic:** ~12.0
**LLM (GPT-4):** ~8.5/10 (clear, documented)
**LLM (Claude):** ~9.0/10
**Final:** ~25 points

## Cost Estimates

With caching enabled:
- **First epoch**: ~$0.10-0.20 per 100 code evaluations
- **Subsequent epochs**: Near zero (cached)

Tips to reduce costs:
1. Use `--llm_threshold 8.0` to only evaluate decent code
2. Use smaller `--subset_size` for initial experiments
3. Cache persists across runs - reusing same problems is cheap

## Monitoring

During training, you'll see:
```
Iteration 20/1000
  Avg Training Reward: 15.32
  Reward: 18.50 (LLM comp: 8.2/10)
```

This shows:
- Average reward across episodes
- Individual evaluation with LLM comprehensibility score

## Troubleshooting

### "LLM rewards requested but llm_reward_evaluator not available"
- Make sure `llm_reward_evaluator.py` is in the same directory as `train_code_generation.py`

### "Error loading HumanEval"
- Check that `datasets` package is installed
- Or the script will fall back to manual download

### API Rate Limits
- Caching helps immensely
- Consider using only one LLM (`--use_gpt` or `--use_claude`)
- Increase `--llm_threshold` to reduce API calls

### High Costs
- Use `--llm_threshold 10.0` for very selective evaluation
- Reduce `--episodes_per_iter`
- Use smaller `--subset_size` during development

## Files

- `llm_reward_evaluator.py` - LLM evaluation module (standalone)
- `train_code_generation.py` - Main training script (modified)
- `reward_cache/` - Cache directory for LLM responses (auto-created)

## Next Steps

1. **Start with heuristics-only** to verify training works
2. **Add LLM rewards** once basic training is stable
3. **Tune llm_weight** based on your priorities (0.5-0.9)
4. **Monitor costs** in your OpenAI/Anthropic dashboards

## Example Submit Script

```bash
#!/bin/bash
#SBATCH --job-name=rl_llm_train
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Set API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Load modules
module load cuda/12.1.1 python/3.11.5

# Activate environment
source venv/bin/activate

# Run training with LLM rewards
python train_code_generation.py \
    --num_iterations 2000 \
    --episodes_per_iter 8 \
    --subset_size 50 \
    --use_llm_rewards \
    --use_gpt \
    --use_claude \
    --llm_weight 0.7 \
    --llm_threshold 7.0
```
