# Using FREE LLMs for Code Evaluation

This guide shows you how to use **completely FREE** open-source LLMs instead of paid GPT-4 or Claude APIs.

## 🎯 Quick Start (Recommended: Ollama)

### Option 1: Ollama (Easiest - Runs Locally)

Ollama is the **easiest and fastest** way to use free LLMs. It runs locally on your machine.

#### 1. Install Ollama on Gilbreth

```bash
# On Gilbreth login node or compute node
curl -fsSL https://ollama.com/install.sh | sh

# Or download manually from https://github.com/ollama/ollama/releases
```

#### 2. Start Ollama Server

```bash
# Start Ollama in the background
ollama serve &

# Download a small, fast model (3B parameters, ~2GB)
ollama pull llama3.2:3b

# Or use a larger model for better quality (8B parameters, ~4.7GB)
ollama pull llama3.2:8b
```

#### 3. Run Training with Ollama

```bash
python train_code_generation.py \
    --num_iterations 1000 \
    --episodes_per_iter 6 \
    --subset_size 20 \
    --use_llm_rewards \
    --llm_backend ollama \
    --llm_model llama3.2:3b \
    --llm_weight 0.7 \
    --llm_threshold 5.0
```

**That's it! Completely free, no API keys needed!**

---

## Option 2: HuggingFace Models (More Control)

HuggingFace gives you direct access to thousands of models, but requires more setup.

### 1. Install Dependencies

```bash
pip install transformers torch accelerate
```

### 2. Choose a Model

Popular free code evaluation models:

| Model | Size | Memory | Quality | Speed |
|-------|------|--------|---------|-------|
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~6GB | Good | Fast |
| `meta-llama/Llama-3.2-8B-Instruct` | 8B | ~16GB | Better | Medium |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~14GB | Good | Medium |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14GB | Great | Medium |

### 3. Run Training with HuggingFace

```bash
python train_code_generation.py \
    --num_iterations 1000 \
    --episodes_per_iter 6 \
    --subset_size 20 \
    --use_llm_rewards \
    --llm_backend huggingface \
    --llm_model meta-llama/Llama-3.2-3B-Instruct \
    --llm_device cuda \
    --llm_weight 0.7 \
    --llm_threshold 5.0
```

**Note:** First run will download the model (~2-6GB). Subsequent runs use cached model.

---

## 📊 Comparison: Free vs Paid

| Feature | Ollama | HuggingFace | GPT-4 | Claude |
|---------|--------|-------------|-------|--------|
| **Cost** | 🟢 FREE | 🟢 FREE | 🔴 ~$0.15/1K evals | 🔴 ~$0.25/1K evals |
| **Speed** | 🟢 Fast | 🟡 Medium | 🟢 Fast | 🟢 Fast |
| **Setup** | 🟢 Easy | 🟡 Medium | 🟢 Easy | 🟢 Easy |
| **Quality** | 🟡 Good | 🟡 Good | 🟢 Excellent | 🟢 Excellent |
| **Offline** | 🟢 Yes | 🟢 Yes | 🔴 No | 🔴 No |
| **GPU Needed** | 🟡 Recommended | 🟢 Yes | 🟢 No | 🟢 No |

**Recommendation:** Use **Ollama with llama3.2:3b** for best balance of speed, quality, and ease of use.

---

## 🔧 Configuration Options

### All LLM Parameters

```bash
python train_code_generation.py \
    --use_llm_rewards \              # Enable LLM evaluation
    --llm_backend ollama \            # Backend: ollama, huggingface, gpt, claude
    --llm_model llama3.2:3b \         # Model name (depends on backend)
    --llm_device cuda \               # Device for HF: cuda or cpu
    --ollama_url http://localhost:11434 \  # Ollama server URL
    --llm_weight 0.7 \                # Weight for LLM vs heuristics (0-1)
    --llm_threshold 5.0 \             # Only eval code with heuristic > this
    --reward_cache_dir ./reward_cache  # Cache directory
```

### Backend-Specific Model Names

**Ollama:**
```bash
--llm_backend ollama --llm_model llama3.2:3b
--llm_backend ollama --llm_model llama3.2:8b
--llm_backend ollama --llm_model codellama:7b
--llm_backend ollama --llm_model mistral:7b
```

**HuggingFace:**
```bash
--llm_backend huggingface --llm_model meta-llama/Llama-3.2-3B-Instruct
--llm_backend huggingface --llm_model meta-llama/Llama-3.2-8B-Instruct
--llm_backend huggingface --llm_model mistralai/Mistral-7B-Instruct-v0.3
--llm_backend huggingface --llm_model Qwen/Qwen2.5-7B-Instruct
```

**Paid APIs (for reference):**
```bash
--llm_backend gpt --llm_model gpt-4o-mini
--llm_backend claude --llm_model claude-3-5-haiku-20241022
```

---

## 💡 Tips for Best Performance

### 1. Start with Small Model
Use `llama3.2:3b` first to verify everything works, then upgrade to larger models if needed.

### 2. Use Caching
The system automatically caches LLM responses. Reusing the same problems is essentially free!

### 3. Adjust Threshold
Higher `--llm_threshold` means LLM only evaluates promising code, saving compute:
```bash
--llm_threshold 8.0  # Very selective (faster)
--llm_threshold 5.0  # Balanced (default)
--llm_threshold 0.0  # Evaluate everything (slower)
```

### 4. GPU Memory Management
If running out of GPU memory:
- Use smaller model (3B instead of 8B)
- Use `--llm_device cpu` for HuggingFace
- Reduce `--episodes_per_iter`

---

## 🚀 Example Submit Scripts

### SBATCH Script with Ollama

```bash
#!/bin/bash
#SBATCH --job-name=rl_llm_free
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Load modules
module load cuda/12.1.1 python/3.11.5

# Activate environment
source venv/bin/activate

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!
sleep 5  # Wait for server to start

# Make sure model is downloaded
ollama pull llama3.2:3b

# Run training with FREE Ollama
python train_code_generation.py \
    --num_iterations 2000 \
    --episodes_per_iter 8 \
    --subset_size 50 \
    --use_llm_rewards \
    --llm_backend ollama \
    --llm_model llama3.2:3b \
    --llm_weight 0.7 \
    --llm_threshold 7.0

# Stop Ollama when done
kill $OLLAMA_PID
```

### SBATCH Script with HuggingFace

```bash
#!/bin/bash
#SBATCH --job-name=rl_llm_hf
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Load modules
module load cuda/12.1.1 python/3.11.5

# Activate environment
source venv/bin/activate

# Install HF dependencies (first time only)
# pip install transformers torch accelerate

# Run training with FREE HuggingFace model
python train_code_generation.py \
    --num_iterations 2000 \
    --episodes_per_iter 8 \
    --subset_size 50 \
    --use_llm_rewards \
    --llm_backend huggingface \
    --llm_model meta-llama/Llama-3.2-3B-Instruct \
    --llm_device cuda \
    --llm_weight 0.7 \
    --llm_threshold 7.0
```

---

## 🐛 Troubleshooting

### "Connection refused" (Ollama)
```bash
# Check if Ollama is running
ps aux | grep ollama

# Restart Ollama
pkill ollama
ollama serve &
```

### "Model not found" (Ollama)
```bash
# List downloaded models
ollama list

# Download the model
ollama pull llama3.2:3b
```

### "Out of memory" (HuggingFace)
```bash
# Use smaller model
--llm_model meta-llama/Llama-3.2-3B-Instruct  # Instead of 8B

# Or use CPU (slower but works)
--llm_device cpu
```

### "Model download slow" (HuggingFace)
```bash
# Set HuggingFace cache to scratch (faster storage)
export HF_HOME=/scratch/gilbreth/$USER/huggingface_cache
```

### "Import error: transformers"
```bash
# Install missing dependencies
pip install transformers torch accelerate
```

---

## 📈 Expected Results

With free LLMs, you should see:

```
Iteration 20/1000
  Avg Training Reward: 15.32
  Reward: 18.50 (LLM comp: 7.8/10)  # Ollama evaluation
```

**Quality Comparison:**
- **Ollama llama3.2:3b**: ~85% as good as GPT-4 (FREE!)
- **Ollama llama3.2:8b**: ~90% as good as GPT-4 (FREE!)
- **HuggingFace Llama-3.2-8B**: ~90% as good as GPT-4 (FREE!)
- **GPT-4 / Claude**: 100% (costs money)

For training RL models, free LLMs are **more than sufficient**!

---

## 🎓 Next Steps

1. **Test locally first:**
   ```bash
   python llm_reward_evaluator.py  # Test the evaluator
   ```

2. **Run small training job:**
   ```bash
   # Just 100 iterations to verify it works
   python train_code_generation.py \
       --num_iterations 100 \
       --use_llm_rewards \
       --llm_backend ollama \
       --llm_model llama3.2:3b
   ```

3. **Scale up:**
   ```bash
   # Full training run
   sbatch submit_train.sh
   ```

---

## 📚 Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Models Library](https://ollama.com/library)
- [HuggingFace Model Hub](https://huggingface.co/models?pipeline_tag=text-generation)
- [Llama 3.2 Models](https://huggingface.co/meta-llama)

## 🆚 When to Use Each Option

| Use Case | Recommended Backend |
|----------|-------------------|
| **Quick experiments on Gilbreth** | Ollama (llama3.2:3b) |
| **Production training** | Ollama (llama3.2:8b) |
| **Need specific model** | HuggingFace |
| **Have budget, want best quality** | GPT-4 or Claude |
| **Offline/air-gapped** | HuggingFace (download once) |
| **Multiple servers** | Ollama (easy to replicate) |

**Bottom line:** Start with **Ollama + llama3.2:3b**. It's the perfect balance of ease, speed, and quality!
