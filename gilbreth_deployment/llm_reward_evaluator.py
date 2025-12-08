"""
LLM-based Reward Evaluator for Code Comprehensibility
Supports FREE open-source LLMs (Ollama, HuggingFace) and paid APIs (GPT-4, Claude)
"""

import os
import json
import hashlib
from typing import Dict, Optional, Tuple
from functools import lru_cache

# Optional imports for different LLM providers
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class LLMRewardEvaluator:
    """Evaluates code comprehensibility using FREE or paid LLMs"""

    def __init__(self,
                 llm_backend: str = "ollama",  # "ollama", "huggingface", "gpt", "claude"
                 model_name: str = "llama3.2:3b",  # Model to use
                 cache_dir: str = "./reward_cache",
                 device: str = "cuda",  # For HuggingFace models
                 ollama_url: str = "http://localhost:11434"):  # Ollama API URL
        """
        Initialize LLM evaluators

        Args:
            llm_backend: Which LLM to use ("ollama", "huggingface", "gpt", "claude")
            model_name: Model identifier (e.g., "llama3.2:3b" for Ollama, "meta-llama/Llama-3.2-3B-Instruct" for HF)
            cache_dir: Directory to cache LLM responses
            device: Device for HuggingFace models ("cuda" or "cpu")
            ollama_url: URL for Ollama API (if using Ollama)
        """
        self.llm_backend = llm_backend.lower()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.ollama_url = ollama_url
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize the appropriate backend
        if self.llm_backend == "ollama":
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests library required for Ollama. Install with: pip install requests")
            print(f"Using Ollama with model: {model_name}")

        elif self.llm_backend == "huggingface":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers library required. Install with: pip install transformers torch")
            print(f"Loading HuggingFace model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            if device == "cpu":
                self.model = self.model.to(device)
            print(f"Model loaded on {device}")

        elif self.llm_backend == "gpt":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai library required. Install with: pip install openai")
            self.gpt_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            print(f"Using OpenAI GPT with model: {model_name}")

        elif self.llm_backend == "claude":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic library required. Install with: pip install anthropic")
            self.claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            print(f"Using Anthropic Claude with model: {model_name}")

        else:
            raise ValueError(f"Unknown backend: {llm_backend}. Choose from: ollama, huggingface, gpt, claude")

        # Evaluation prompt template
        self.eval_prompt = """You are evaluating Python code for comprehensibility and quality.

**Task Prompt:**
{prompt}

**Generated Code:**
```python
{code}
```

Please evaluate this code on the following criteria (score each 0-10):

1. **Correctness**: Does the code correctly implement the requested functionality?
2. **Comprehensibility**: Is the code easy to read and understand?
3. **Code Structure**: Is the code well-organized with clear logic flow?
4. **Variable Naming**: Are variables and functions named descriptively?
5. **Comments/Documentation**: Are complex parts explained (if needed)?
6. **Efficiency**: Is the solution reasonably efficient?
7. **Completeness**: Does it fully address the prompt requirements?

Provide your response in this exact JSON format:
{{
  "correctness": <score 0-10>,
  "comprehensibility": <score 0-10>,
  "structure": <score 0-10>,
  "naming": <score 0-10>,
  "documentation": <score 0-10>,
  "efficiency": <score 0-10>,
  "completeness": <score 0-10>,
  "overall_score": <score 0-10>,
  "reasoning": "<brief explanation>"
}}
"""

    def _get_cache_key(self, prompt: str, code: str) -> str:
        """Generate cache key for prompt+code+model combination"""
        content = f"{self.llm_backend}:{self.model_name}:{prompt}:{code}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_eval(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached evaluation if exists"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def _save_cached_eval(self, cache_key: str, evaluation: Dict):
        """Save evaluation to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(evaluation, f, indent=2)
        except:
            pass

    def evaluate_with_ollama(self, prompt: str, code: str) -> Optional[Dict]:
        """Evaluate code using Ollama (FREE local LLM)"""
        cache_key = self._get_cache_key(prompt, code)
        cached = self._get_cached_eval(cache_key)
        if cached:
            return cached

        try:
            eval_prompt = self.eval_prompt.format(prompt=prompt, code=code)

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": eval_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result_text = response.json()["response"]
                result = self._extract_json(result_text)

                if result:
                    self._save_cached_eval(cache_key, result)
                    return result

        except Exception as e:
            print(f"Ollama evaluation error: {e}")
            return None

    def evaluate_with_huggingface(self, prompt: str, code: str) -> Optional[Dict]:
        """Evaluate code using HuggingFace model (FREE local LLM)"""
        cache_key = self._get_cache_key(prompt, code)
        cached = self._get_cached_eval(cache_key)
        if cached:
            return cached

        try:
            eval_prompt = self.eval_prompt.format(prompt=prompt, code=code)

            # Format for instruction-following models
            messages = [
                {"role": "system", "content": "You are a code quality evaluator. Always respond with valid JSON."},
                {"role": "user", "content": eval_prompt}
            ]

            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = eval_prompt

            # Tokenize and generate
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            result_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Parse JSON from response
            result = self._extract_json(result_text)

            if result:
                self._save_cached_eval(cache_key, result)
                return result

        except Exception as e:
            print(f"HuggingFace evaluation error: {e}")
            return None

    def evaluate_with_gpt(self, prompt: str, code: str) -> Optional[Dict]:
        """Evaluate code using GPT-4 (PAID API)"""
        cache_key = self._get_cache_key(prompt, code)
        cached = self._get_cached_eval(cache_key)
        if cached:
            return cached

        try:
            eval_prompt = self.eval_prompt.format(prompt=prompt, code=code)

            response = self.gpt_client.chat.completions.create(
                model=self.model_name or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a code quality evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            result_text = response.choices[0].message.content
            result = self._extract_json(result_text)

            if result:
                self._save_cached_eval(cache_key, result)
                return result

        except Exception as e:
            print(f"GPT evaluation error: {e}")
            return None

    def evaluate_with_claude(self, prompt: str, code: str) -> Optional[Dict]:
        """Evaluate code using Claude (PAID API)"""
        cache_key = self._get_cache_key(prompt, code)
        cached = self._get_cached_eval(cache_key)
        if cached:
            return cached

        try:
            eval_prompt = self.eval_prompt.format(prompt=prompt, code=code)

            response = self.claude_client.messages.create(
                model=self.model_name or "claude-3-5-haiku-20241022",
                max_tokens=500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": eval_prompt}
                ]
            )

            result_text = response.content[0].text
            result = self._extract_json(result_text)

            if result:
                self._save_cached_eval(cache_key, result)
                return result

        except Exception as e:
            print(f"Claude evaluation error: {e}")
            return None

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response"""
        try:
            # Try direct parse
            return json.loads(text)
        except:
            pass

        # Try to find JSON in markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # Try to find any JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass

        return None

    def evaluate_code(self, prompt: str, code: str) -> Dict:
        """
        Evaluate code using the configured LLM backend

        Returns:
            Dictionary with scores and evaluation details
        """
        # Route to appropriate evaluator based on backend
        if self.llm_backend == "ollama":
            eval_result = self.evaluate_with_ollama(prompt, code)
        elif self.llm_backend == "huggingface":
            eval_result = self.evaluate_with_huggingface(prompt, code)
        elif self.llm_backend == "gpt":
            eval_result = self.evaluate_with_gpt(prompt, code)
        elif self.llm_backend == "claude":
            eval_result = self.evaluate_with_claude(prompt, code)
        else:
            eval_result = None

        # Return standardized format
        if eval_result is None:
            return {
                'combined': {
                    'overall_score': 0.0,
                    'correctness': 0.0,
                    'comprehensibility': 0.0,
                    'has_evaluation': False
                }
            }

        return {
            'combined': {
                **eval_result,
                'has_evaluation': True
            }
        }


class HybridRewardFunction:
    """Combines fast heuristics with LLM evaluation for comprehensive rewards"""

    def __init__(self, llm_evaluator: Optional[LLMRewardEvaluator] = None,
                 use_llm_threshold: float = 0.0, llm_weight: float = 0.7):
        """
        Initialize hybrid reward function

        Args:
            llm_evaluator: LLM evaluator instance (None = heuristics only)
            use_llm_threshold: Only use LLM for code with heuristic reward > this threshold
            llm_weight: Weight for LLM score vs heuristic score (0-1)
        """
        self.llm_evaluator = llm_evaluator
        self.use_llm_threshold = use_llm_threshold
        self.llm_weight = llm_weight
        self.heuristic_weight = 1.0 - llm_weight

    def compute_heuristic_reward(self, code: str, problem: Dict) -> Tuple[float, Dict]:
        """
        Fast heuristic-based reward computation

        Returns:
            (reward_score, details_dict)
        """
        details = {
            'has_def': 0.0,
            'has_return': 0.0,
            'good_length': 0.0,
            'has_structure': 0.0,
            'has_indentation': 0.0,
            'non_empty': 0.0
        }

        # Strip and check if empty
        code = code.strip()
        if len(code) == 0:
            return -10.0, details

        details['non_empty'] = 2.0
        reward = 2.0

        # Check if code contains function definition
        if 'def ' in code:
            details['has_def'] = 4.0
            reward += 4.0
        else:
            return -5.0, details  # No function is critical failure

        # Check for return statement
        if 'return' in code:
            details['has_return'] = 3.0
            reward += 3.0

        # Check length (count non-empty lines)
        code_lines = [line for line in code.split('\n') if line.strip()]
        num_lines = len(code_lines)

        if num_lines < 1:
            details['good_length'] = -5.0
            reward -= 5.0
        elif num_lines < 3:
            details['good_length'] = -1.0
            reward -= 1.0
        elif 3 <= num_lines <= 30:
            details['good_length'] = 2.0
            reward += 2.0
        else:
            details['good_length'] = -0.5
            reward -= 0.5

        # Check for control flow structures
        has_control = any(keyword in code for keyword in ['if ', 'for ', 'while ', 'elif '])
        if has_control and ':' in code:
            details['has_structure'] = 2.0
            reward += 2.0

        # Check for proper indentation (suggests structure)
        indented_lines = [line for line in code.split('\n') if line.startswith('    ') or line.startswith('\t')]
        if len(indented_lines) >= 2:
            details['has_indentation'] = 1.0
            reward += 1.0

        # Total from details
        total_reward = sum(details.values())

        return total_reward, details

    def compute_execution_reward(self, code: str, problem: Dict) -> Tuple[float, Dict]:
        """
        Try to execute code and give rewards for passing tests

        Returns:
            (reward_score, execution_details)
        """
        exec_details = {
            'syntax_valid': False,
            'executes': False,
            'passes_tests': False,
            'error': None
        }

        reward = 0.0

        # Check syntax
        try:
            compile(code, '<string>', 'exec')
            exec_details['syntax_valid'] = True
            reward += 2.0
        except SyntaxError as e:
            exec_details['error'] = f"SyntaxError: {e}"
            return -3.0, exec_details

        # Try execution
        try:
            exec_globals = {}
            exec(code, exec_globals)
            exec_details['executes'] = True
            reward += 3.0

            # Try running tests if available
            if 'test' in problem and 'entry_point' in problem:
                try:
                    test_code = problem['test']
                    exec(test_code, exec_globals)
                    if 'check' in exec_globals:
                        exec_globals['check'](exec_globals.get(problem['entry_point']))
                        exec_details['passes_tests'] = True
                        reward += 10.0  # Big reward for passing tests!
                except Exception as e:
                    exec_details['error'] = f"Test failed: {str(e)[:100]}"
        except Exception as e:
            exec_details['error'] = f"Runtime: {str(e)[:100]}"

        return reward, exec_details

    def compute_reward(self, code: str, problem: Dict) -> Tuple[float, Dict]:
        """
        Compute comprehensive reward using heuristics, execution, and optional LLM evaluation

        Returns:
            (final_reward, details_dict)
        """
        all_details = {}

        # 1. Fast heuristic check
        heuristic_reward, heuristic_details = self.compute_heuristic_reward(code, problem)
        all_details['heuristic'] = heuristic_details
        all_details['heuristic_reward'] = heuristic_reward

        # If code is terrible, don't waste LLM calls
        if heuristic_reward < -5.0:
            all_details['final_reward'] = heuristic_reward
            all_details['llm_used'] = False
            return heuristic_reward, all_details

        # 2. Execution check (if heuristics passed)
        exec_reward, exec_details = self.compute_execution_reward(code, problem)
        all_details['execution'] = exec_details
        all_details['execution_reward'] = exec_reward

        # 3. LLM evaluation (if enabled and code is decent)
        llm_reward = 0.0
        if self.llm_evaluator and heuristic_reward >= self.use_llm_threshold:
            try:
                prompt = problem.get('prompt', '')
                llm_eval = self.llm_evaluator.evaluate_code(prompt, code)

                if llm_eval['combined']['has_evaluation']:
                    # Convert LLM score (0-10) to reward scale
                    llm_overall = llm_eval['combined']['overall_score']
                    llm_comprehensibility = llm_eval['combined']['comprehensibility']

                    # Weight comprehensibility highly
                    llm_reward = (llm_overall * 0.6 + llm_comprehensibility * 0.4) * 2.0  # Scale to ~20 max

                    all_details['llm_evaluation'] = llm_eval['combined']
                    all_details['llm_reward'] = llm_reward
                    all_details['llm_used'] = True
                else:
                    all_details['llm_used'] = False
            except Exception as e:
                print(f"LLM evaluation failed: {e}")
                all_details['llm_used'] = False
        else:
            all_details['llm_used'] = False

        # 4. Combine all rewards
        if all_details['llm_used']:
            # Weighted combination of heuristic, execution, and LLM
            final_reward = (
                self.heuristic_weight * heuristic_reward +
                0.3 * exec_reward +
                self.llm_weight * llm_reward
            )
        else:
            # Just heuristic + execution
            final_reward = heuristic_reward + exec_reward

        all_details['final_reward'] = final_reward

        return final_reward, all_details


# Example usage and testing
if __name__ == "__main__":
    # Test with example code
    test_prompt = """
def add_two_numbers(a: int, b: int) -> int:
    \"\"\" Add two numbers and return the result \"\"\"
"""

    test_code_good = """
def add_two_numbers(a: int, b: int) -> int:
    \"\"\" Add two numbers and return the result \"\"\"
    # Simple addition of two integers
    result = a + b
    return result
"""

    test_code_bad = """
def add_two_numbers(a, b):
    return a+b
"""

    # Initialize with FREE Ollama (recommended)
    print("Initializing with Ollama (free local LLM)...")
    llm_eval = LLMRewardEvaluator(
        llm_backend="ollama",
        model_name="llama3.2:3b"  # Small, fast model
    )

    # Alternative: Use HuggingFace (slower but also free)
    # llm_eval = LLMRewardEvaluator(
    #     llm_backend="huggingface",
    #     model_name="meta-llama/Llama-3.2-3B-Instruct",
    #     device="cuda"  # or "cpu"
    # )

    # Alternative: Use paid APIs (requires API keys)
    # llm_eval = LLMRewardEvaluator(llm_backend="gpt", model_name="gpt-4o-mini")
    # llm_eval = LLMRewardEvaluator(llm_backend="claude", model_name="claude-3-5-haiku-20241022")

    hybrid_reward = HybridRewardFunction(
        llm_evaluator=llm_eval,
        use_llm_threshold=0.0,
        llm_weight=0.7
    )

    # Test evaluation
    print("=" * 70)
    print("Testing GOOD code:")
    print("=" * 70)
    problem = {'prompt': test_prompt}
    reward, details = hybrid_reward.compute_reward(test_code_good, problem)
    print(f"Final Reward: {reward:.2f}")
    print(json.dumps(details, indent=2))

    print("\n" + "=" * 70)
    print("Testing BAD code:")
    print("=" * 70)
    reward, details = hybrid_reward.compute_reward(test_code_bad, problem)
    print(f"Final Reward: {reward:.2f}")
    print(json.dumps(details, indent=2))
