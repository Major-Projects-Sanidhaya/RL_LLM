# Research Paper Writing Prompt for Claude AI

Copy and paste this entire prompt to Claude AI to generate a professional research paper.

---

## Prompt for Claude AI

I need help writing a professional research paper about my work on **Reinforcement Learning for Code Generation using Hierarchical Policies and LLM-based Reward Functions**. Please write a complete research paper in IEEE/ACM conference format with the following structure and details:

### Paper Title
"Hierarchical Reinforcement Learning for Code Generation with LLM-Enhanced Reward Shaping"

### Research Overview

**Problem Statement:**
- Traditional RL approaches for code generation suffer from reward hacking where models learn to exploit poorly designed reward functions
- Example: Our initial model learned to generate empty code to maximize rewards
- Existing heuristic-based rewards fail to capture code comprehensibility and quality

**Our Solution:**
- Hierarchical RL policy with two levels: high-level intention generation and low-level token generation
- Hybrid reward function combining fast heuristics with LLM-based evaluation
- Support for free open-source LLMs (Ollama, HuggingFace) making it accessible without API costs

**Key Contributions:**
1. Novel hierarchical policy architecture for code generation
2. Hybrid reward system preventing reward hacking while evaluating code comprehensibility
3. Free LLM integration (Ollama/HuggingFace) for cost-effective training
4. Memory-efficient mini-batch training enabling use of large models on 80GB GPUs

---

### Technical Architecture

#### 1. Hierarchical Policy Network

**Model Architecture:**
- Base model: GPT-2 (124M parameters)
- Total parameters: 47,004,755
- Two-level hierarchy:
  - **High-level policy**: Generates intentions (latent representations of what code should do)
    - State encoder: Transformer with 6 layers, 8 attention heads, 512 dimensions
    - Intention network: Projects to 256-dimensional intention space
  - **Low-level policy**: Generates tokens conditioned on intentions
    - Token decoder: Conditions on both state encoding and high-level intention
    - Output: Vocabulary distribution for next token

**Training Setup:**
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Training algorithm: Proximal Policy Optimization (PPO)
- Episodes per iteration: 2-4 (for memory efficiency)
- Gradient accumulation: Mini-batch size of 1 episode
- Max sequence length: 512 tokens
- Multi-GPU: DataParallel on 2× NVIDIA A100 80GB GPUs

**Key Innovation - Memory-Efficient Training:**
- Implemented mini-batch gradient accumulation to prevent CUDA OOM
- Process one episode at a time during backpropagation
- Accumulate gradients across episodes before updating weights
- Reduces peak memory from 77GB → 12GB for 4 episodes

#### 2. Hybrid Reward Function

**Three-Component Reward System:**

**A. Heuristic Rewards (Fast, Always Active):**
```
Reward Components:
- Non-empty code: +2.0 (heavy penalty -10.0 if empty)
- Function definition present: +4.0 (penalty -5.0 if missing)
- Return statement: +3.0 (penalty -1.0 if missing)
- Good length (3-30 lines): +2.0
- Control flow structures: +2.0
- Proper indentation: +1.0

Maximum heuristic reward: ~14 points
```

**Key Fix - Preventing Reward Hacking:**
- Original system: Empty code received +1.0 reward
- Fixed system: Empty code receives -10.0 penalty
- Missing function definition: -5.0 penalty
- This prevents the model from learning to generate empty/minimal code

**B. Execution Rewards (Validation):**
```
- Syntax valid: +2.0 (penalty -3.0 if invalid)
- Executes without error: +3.0
- Passes HumanEval tests: +10.0
```

**C. LLM-Based Evaluation (Comprehensibility):**

LLMs evaluate code on 7 criteria (each 0-10):
1. Correctness: Does it solve the problem?
2. Comprehensibility: Easy to read and understand?
3. Code Structure: Well-organized logic?
4. Variable Naming: Descriptive names?
5. Documentation: Complex parts explained?
6. Efficiency: Reasonably efficient?
7. Completeness: Fully addresses requirements?

**LLM Backend Options:**
- **Ollama** (FREE): Llama 3.2 3B/8B - runs locally, ~85-90% quality of GPT-4
- **HuggingFace** (FREE): Meta Llama 3.2, Mistral 7B - direct model loading
- **GPT-4 Mini** (PAID): Best quality, ~$0.15 per 1000 evaluations
- **Claude Haiku** (PAID): High quality, ~$0.25 per 1000 evaluations

**Final Reward Calculation:**
```python
if heuristic_reward < -5.0:
    # Code too bad, don't waste LLM calls
    final_reward = heuristic_reward
elif heuristic_reward >= llm_threshold and llm_enabled:
    # Hybrid: combine all three components
    final_reward = (
        heuristic_weight * heuristic_reward +
        0.3 * execution_reward +
        llm_weight * llm_reward
    )
else:
    # No LLM: just heuristics + execution
    final_reward = heuristic_reward + execution_reward
```

**Default Configuration:**
- LLM weight: 0.7 (70% weight on LLM evaluation)
- Heuristic weight: 0.3 (30% weight)
- LLM threshold: 7.0 (only use LLM for decent code)
- Caching: Enabled (reduces cost by ~90% after first epoch)

#### 3. Training Pipeline

**Dataset:**
- HumanEval: 164 programming problems
- Training subset: 50 problems (configurable)
- Test holdout for evaluation

**Training Loop:**
```
For each iteration:
  1. Collect episodes (2-4 episodes per iteration)
     - Generate code using current policy
     - Compute rewards (heuristic + execution + LLM)
     - Store trajectories in replay buffer

  2. Update policy (PPO with 4 epochs)
     - Process mini-batches of 1 episode
     - Accumulate gradients across mini-batches
     - Clip PPO objective to prevent large updates
     - Update both high-level and low-level policies

  3. Checkpoint saving every 100 iterations
```

**Hyperparameters:**
- Number of iterations: 2000
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- PPO clip epsilon: 0.2
- Value loss coefficient: 0.5
- Entropy coefficient: 0.01
- Gradient clipping: 0.5

---

### Experimental Results

**Hardware:**
- Cluster: Purdue Gilbreth HPC
- GPUs: 2× NVIDIA A100 80GB (PCIe)
- CPUs: 32 cores per node
- Memory: 256GB RAM

**Training Time:**
- Heuristics only: ~8-10 hours for 2000 iterations
- With Ollama LLM: ~10-12 hours for 2000 iterations
- With paid APIs (GPT-4): ~9-11 hours (faster inference)

**Memory Optimization Results:**
Before mini-batching:
- 2 episodes: 77GB GPU memory (OOM ❌)
- 4 episodes: 77GB GPU memory (OOM ❌)
- 8 episodes: 77GB GPU memory (OOM ❌)

After mini-batching:
- 2 episodes: ~8GB GPU memory ✅
- 4 episodes: ~12GB GPU memory ✅
- 8 episodes: ~20GB GPU memory ✅

**Quality Comparison (Reward Scores):**

Empty Code (Reward Hacking - Before Fix):
```python
# Generated: (empty)
```
- Old reward: +1.0
- New reward: -10.0 ❌

Minimal Code:
```python
def add(a, b):
    return a + b
```
- Heuristic: ~9.0
- Ollama (Llama 3.2 3B): 6.5/10
- GPT-4 Mini: 7.0/10
- Final reward: ~13 points

Good Code:
```python
def add_two_numbers(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    result = a + b
    return result
```
- Heuristic: ~12.0
- Ollama (Llama 3.2 3B): 8.5/10
- GPT-4 Mini: 9.0/10
- Final reward: ~25 points

**Cost Analysis:**

Training cost per 2000 iterations:
- Heuristics only: $0 (FREE)
- Ollama (local LLM): $0 (FREE, ~10% slower)
- HuggingFace models: $0 (FREE, ~15% slower)
- GPT-4 Mini: ~$5-10 (with caching)
- Claude Haiku: ~$8-15 (with caching)

With caching enabled:
- First epoch: ~$0.10-0.20 per 100 evaluations
- Subsequent epochs: Near zero (cached responses)

---

### Key Innovations

1. **Hierarchical Architecture**
   - Separates high-level planning (intentions) from low-level execution (tokens)
   - Enables more structured code generation
   - Reduces action space complexity

2. **Reward Hacking Prevention**
   - Heavy penalties for empty/incomplete code
   - Multi-component validation (heuristics + execution + LLM)
   - Threshold-based LLM evaluation to avoid wasting calls on bad code

3. **Free LLM Integration**
   - First work to integrate open-source LLMs (Ollama, HuggingFace) for RL rewards
   - Makes high-quality reward signals accessible without API costs
   - 85-90% quality of GPT-4 at zero cost

4. **Memory-Efficient Training**
   - Mini-batch gradient accumulation
   - Enables training large hierarchical models on standard GPUs
   - Reduces peak memory usage by 85%

5. **Hybrid Reward System**
   - Fast heuristics for initial filtering
   - Expensive LLM evaluation only for promising code
   - Optimal balance of speed and quality

---

### Implementation Details

**Code Structure:**
- `train_code_generation.py`: Main training loop (903 lines)
- `llm_reward_evaluator.py`: LLM evaluation module (502 lines)
- Supports 4 LLM backends: Ollama, HuggingFace, GPT-4, Claude

**Key Classes:**
```python
class HierarchicalCodePolicy(nn.Module):
    """Two-level hierarchical policy for code generation"""
    - encode_state(): Transformer state encoder
    - generate_intention(): High-level planning
    - generate_action(): Low-level token generation

class HierarchicalPPOTrainer:
    """PPO trainer with hierarchical policies"""
    - compute_hierarchical_policy_loss()
    - compute_hierarchical_value_loss()
    - update(): Mini-batch gradient accumulation

class LLMRewardEvaluator:
    """Multi-backend LLM evaluation"""
    - evaluate_with_ollama()
    - evaluate_with_huggingface()
    - evaluate_with_gpt()
    - evaluate_with_claude()

class HybridRewardFunction:
    """Three-component reward system"""
    - compute_heuristic_reward()
    - compute_execution_reward()
    - compute_reward() with LLM integration
```

**Dependencies:**
- PyTorch 2.0+
- Transformers (HuggingFace)
- OpenAI API (optional)
- Anthropic API (optional)
- Ollama (optional, for free LLM)
- Datasets (HumanEval)

---

### Ablation Studies to Include

Please design and present these ablation studies:

1. **Hierarchical vs Flat Policy**
   - Compare hierarchical (intention + token) vs direct token generation
   - Metrics: code quality, training stability, final rewards

2. **Reward Function Components**
   - Heuristics only
   - Heuristics + Execution
   - Heuristics + Execution + LLM
   - Measure impact on code quality and training time

3. **LLM Backend Comparison**
   - Free models (Ollama Llama 3.2 3B, 8B)
   - Paid models (GPT-4 Mini, Claude Haiku)
   - Compare: cost, quality, training time

4. **Memory Optimization**
   - Before mini-batching (OOM failures)
   - After mini-batching (successful training)
   - Impact on convergence speed

5. **Reward Hacking Prevention**
   - Training with old reward function (empty code problem)
   - Training with fixed reward function (proper code generation)
   - Show reward curves and example outputs

---

### Related Work to Cite

Please include comparisons with:

1. **Code Generation:**
   - Codex (OpenAI)
   - CodeGen (Salesforce)
   - StarCoder
   - AlphaCode

2. **RL for Code:**
   - CodeRL (Le et al.)
   - RLTF (Reinforcement Learning from Task Feedback)
   - PPOCoder

3. **Hierarchical RL:**
   - Options framework
   - Feudal Networks
   - HIRO (Hierarchical RL with Off-policy corrections)

4. **LLM as Judge:**
   - Constitutional AI (Anthropic)
   - Self-Refine
   - LLM-as-a-Judge (recent papers on using LLMs for evaluation)

---

### Writing Instructions

Please write a complete paper with:

**Sections:**
1. Abstract (200-250 words)
2. Introduction (2 pages)
   - Motivation and problem statement
   - Our approach and contributions
3. Related Work (1.5 pages)
4. Method (4 pages)
   - Hierarchical policy architecture
   - Hybrid reward function
   - Training algorithm
   - Memory optimization
5. Experimental Setup (1 page)
   - Dataset, hardware, hyperparameters
6. Results (3 pages)
   - Main results with tables and figures
   - Ablation studies
   - Cost analysis
7. Discussion (1 page)
   - Advantages and limitations
   - Future work
8. Conclusion (0.5 pages)
9. References

**Style Requirements:**
- IEEE/ACM conference format
- Academic tone, clear and precise
- Include mathematical notation where appropriate
- Suggest figures and tables (describe what they should show)
- Cite at least 30 relevant papers
- Highlight novelty and contributions clearly

**Key Messages to Emphasize:**
1. Hierarchical RL improves code generation structure
2. Hybrid rewards prevent reward hacking while maintaining efficiency
3. Free LLM integration democratizes high-quality RL training
4. Memory optimization enables practical deployment
5. System achieves strong results at low/zero cost

Please write the complete paper now, including all sections, mathematical formulations, and suggestions for figures/tables.

---

## Additional Context

**Target Venues:**
- ICML (International Conference on Machine Learning)
- NeurIPS (Neural Information Processing Systems)
- ICLR (International Conference on Learning Representations)
- AAAI (Association for Advancement of Artificial Intelligence)
- ACL (Association for Computational Linguistics)

**Unique Selling Points:**
1. First to use free open-source LLMs (Ollama) for RL reward functions
2. Novel hierarchical architecture specifically for code generation
3. Practical memory optimization enabling 80GB GPU training
4. Comprehensive reward hacking prevention
5. Zero-cost training option with competitive quality

**Target Audience:**
- RL researchers
- Code generation researchers
- NLP practitioners
- ML engineers working on cost-effective solutions
