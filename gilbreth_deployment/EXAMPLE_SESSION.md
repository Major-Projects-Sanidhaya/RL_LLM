# Example Training Session on Gilbreth

This document shows a complete example session from start to finish.

## Session Timeline

**Total time**: ~6 hours (including queue wait and training)
**Active time**: ~15 minutes (rest is automated)

---

## Part 1: Local Machine (5 minutes)

### Step 1: Navigate to deployment folder

```bash
$ cd C:\Users\shars11\Documents\RL_LLM\gilbreth_deployment
$ ls
check_gpu.sh  download_results.sh  monitor_training.sh  README.md
setup_gilbreth.sh  submit_training.sh  train_code_generation.py
upload_to_gilbreth.sh  ...
```

### Step 2: Upload to Gilbreth

```bash
$ bash upload_to_gilbreth.sh shars11

========================================
Upload to Gilbreth
========================================
Local: .
Remote: shars11@gilbreth.rcac.purdue.edu:~/RL-LLM

Uploading files to Gilbreth...
sending incremental file list
./
check_gpu.sh
          1,087 100%    0.00kB/s    0:00:00 (xfr#1, to-chk=11/13)
setup_gilbreth.sh
          2,234 100%    2.13MB/s    0:00:00 (xfr#2, to-chk=10/13)
...
train_code_generation.py
         27,456 100%   26.18MB/s    0:00:00 (xfr#3, to-chk=8/13)

sent 65,432 bytes  received 234 bytes  43,777.33 bytes/sec
total size is 108K  speedup is 1.65

========================================
Upload Complete!
========================================

Next steps:
1. SSH to Gilbreth:
   ssh shars11@gilbreth.rcac.purdue.edu
...
```

---

## Part 2: Gilbreth - First Time Setup (10 minutes)

### Step 3: SSH to Gilbreth

```bash
$ ssh shars11@gilbreth.rcac.purdue.edu
Password: ********,push
Duo two-factor login

Pushed a login request to your device...
Success. Logging you in...

Last login: Mon Jan 18 09:23:45 2025 from ...

   ______ ______ __    ____   ____   ______ ______ __  __
  / ____//  _/ /  |  / __ ) / __ \ / ____//_  __// / / /
 / / __  / // /| | / __  |/ /_/ // __/    / /  / /_/ /
/ /_/ /_/ // ___ |/ /_/ // _, _// /___   / /  / __  /
\____//___//_/  |_/_____//_/ |_|/_____/  /_/  /_/ /_/

Community Cluster for GPU Computing

[shars11@gilbreth-fe00 ~]$
```

### Step 4: Navigate and setup

```bash
[shars11@gilbreth-fe00 ~]$ cd ~/RL-LLM
[shars11@gilbreth-fe00 RL-LLM]$ ls
check_gpu.sh  setup_gilbreth.sh  train_code_generation.py  ...

[shars11@gilbreth-fe00 RL-LLM]$ bash setup_gilbreth.sh

========================================
Setting up RL-LLM on Gilbreth Cluster
Hierarchical RL for Code Generation
========================================

Loading modules...
Creating conda environment 'rl-llm'...
Collecting package metadata (current_repodata.json): done
Solving environment: done
...
Installing PyTorch with CUDA 12.1...
Looking in indexes: https://download.pytorch.org/whl/cu121
Collecting torch>=2.0.0
  Downloading torch-2.8.0-cp310-cp310-linux_x86_64.whl (900 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 900.0/900.0 MB 12.3 MB/s eta 0:00:00
...
Installing dependencies...
Successfully installed transformers-4.36.0 datasets-2.16.0 ...

Creating directories...

Testing GPU availability...
PyTorch version: 2.8.0+cu121
CUDA available: True
CUDA version: 12.1
GPU count: 1
GPU name: NVIDIA A100-SXM4-80GB

========================================
Setup complete!
========================================

To activate the environment in future sessions:
  module load anaconda cuda/12.1
  source activate rl-llm

To submit jobs:
  sbatch submit_training.sh
```

---

## Part 3: Submit Training (1 minute)

### Step 5: Check GPU availability

```bash
[shars11@gilbreth-fe00 RL-LLM]$ bash check_gpu.sh

========================================
Gilbreth GPU Status
========================================

Available GPU Partitions:
----------------------------------------
PARTITION    AVAIL  TIMELIMIT  NODES  STATE  NODELIST   CPUS       MEMORY     GRES
gpu          up     7-00:00:00 8      idle   gilb-c[01-08] 80/160  500000/1000000 gpu:2
a100         up     7-00:00:00 4      idle   gilb-i[01-04] 128/256 1000000/2000000 gpu:a100:4
a100-80gb    up     7-00:00:00 2      idle   gilb-j[01-02] 128/256 2000000/4000000 gpu:a100:2

GPU Queue Status:
----------------------------------------
JOBID    PARTITION  NAME                USER     ST TIME     NODES  REASON
145673   a100       ml-training         user1    R  2:34:56  1      None
145678   gpu        test-job            user2    PD 0:00     1      Resources

Your Jobs:
----------------------------------------
(No jobs currently)
```

### Step 6: Submit training job

```bash
[shars11@gilbreth-fe00 RL-LLM]$ sbatch submit_training.sh
Submitted batch job 145823

[shars11@gilbreth-fe00 RL-LLM]$ squeue -u $USER
JOBID      PARTITION  NAME            USER     ST TIME     NODES  REASON
145823     gpu        rl-llm-code-gen shars11  PD 0:00     1      Priority
```

Job is pending (PD = Pending, waiting for resources)

---

## Part 4: Monitor Training (ongoing)

### Step 7: Wait for job to start

Wait ~5-30 minutes depending on queue

```bash
[shars11@gilbreth-fe00 RL-LLM]$ squeue -u $USER
JOBID      PARTITION  NAME            USER     ST TIME     NODES  REASON
145823     gpu        rl-llm-code-gen shars11  R  0:02:15  1      None
```

Job is now running (R = Running)!

### Step 8: Monitor progress

```bash
[shars11@gilbreth-fe00 RL-LLM]$ bash monitor_training.sh 145823

Monitoring Job ID: 145823

Job Status:
JobId=145823 JobName=rl-llm-code-gen
   UserId=shars11(12345) GroupId=users(100) MCS_label=N/A
   Priority=4294901719 Nice=0 Account=pfw-cs QOS=normal
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=00:05:42 TimeLimit=12:00:00 TimeMin=N/A
   SubmitTime=2025-01-18T09:45:23 EligibleTime=2025-01-18T09:45:23
   AccrueTime=2025-01-18T09:45:23
   StartTime=2025-01-18T09:47:08 EndTime=2025-01-18T21:47:08 Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2025-01-18T09:47:08
   Partition=gpu AllocNode:Sid=gilbreth-fe00:123456
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=gilb-c03
   BatchHost=gilb-c03
   NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 ReqB:S:C:T=0:0:*:*
   TRES=cpu=8,mem=64G,node=1,billing=8,gres/gpu=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=8 MinMemoryNode=64G MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/home/shars11/RL-LLM/submit_training.sh
   WorkDir=/home/shars11/RL-LLM
   StdErr=/home/shars11/RL-LLM/logs/training_error_145823.txt
   StdOut=/home/shars11/RL-LLM/logs/training_output_145823.txt
   Power=

----------------------------------------
Latest output from logs/training_output_145823.txt:
----------------------------------------
========================================
RL-LLM Code Generation Training
========================================
Job ID: 145823
Node: gilb-c03
Starting time: Mon Jan 18 09:47:10 EST 2025
Account: pfw-cs
Partition: gpu

Loading modules...
Activating conda environment...

Environment Information:
Python: /home/shars11/.conda/envs/rl-llm/bin/python
Python version: Python 3.10.13

GPU Information:
Mon Jan 18 09:47:15 2025
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.1    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   32C    P0    56W / 400W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

Current directory: /home/shars11/RL-LLM
Contents:
total 104K
-rwxr-xr-x 1 shars11 users 1.1K Jan 18 09:30 check_gpu.sh
...

Starting training...
========================================

======================================================================
RL-LLM: Hierarchical Code Generation Training
======================================================================
Device: cuda
GPU: NVIDIA A100-SXM4-80GB
GPU Memory: 85.17 GB
Dataset: humaneval
Iterations: 1000
Episodes per iteration: 5
Max sequence length: 512
======================================================================

Loading tokenizer...
Creating humaneval dataset...
Loading HumanEval dataset...
Loaded 20 code problems
Dataset size: 20
Creating Code Generation environment...
Creating hierarchical policy...
Policy created with 29,518,291 parameters
Creating hierarchical PPO trainer...

======================================================================
Starting Training
======================================================================

Training:   0%|          | 0/1000 [00:00<?, ?it/s]
Training:   1%|          | 10/1000 [01:23<2:18:45,  8.41s/it]
Training:   2%|▏         | 20/1000 [02:47<2:17:34,  8.43s/it]

======================================================================
Iteration 20/1000
======================================================================
  Avg Training Reward: 2.34
  Policy Loss: 0.1234
  Value Loss: 0.0567
  Entropy: 3.4567
  Intention Loss: 0.0123

  Sample Generated Code:
  ------------------------------------------------------------------
  from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
      for i in range(len(numbers)):
          for j in range(i + 1, len(numbers)):
              if abs(numbers[i] - numbers[j]) < threshold:
                  return True
      return False
  ------------------------------------------------------------------

  ✓ New best model! Reward: 2.34
  ✓ Checkpoint saved: ./checkpoints/best_model.pt

Training:   2%|▏         | 20/1000 [02:47<2:17:34,  8.43s/it]
...
----------------------------------------
To watch real-time updates:
  tail -f logs/training_output_145823.txt
```

### Step 9: Watch real-time (optional)

```bash
[shars11@gilbreth-fe00 RL-LLM]$ tail -f logs/training_output_145823.txt
Training:   4%|▍         | 40/1000 [05:35<2:15:12,  8.45s/it]

======================================================================
Iteration 40/1000
======================================================================
  Avg Training Reward: 3.12
  Policy Loss: 0.0987
  Value Loss: 0.0445
  Entropy: 3.3456
  Intention Loss: 0.0098

  Sample Generated Code:
  ------------------------------------------------------------------
  from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
      """Check if any two numbers are closer than threshold"""
      for i in range(len(numbers)):
          for j in range(i + 1, len(numbers)):
              if abs(numbers[i] - numbers[j]) < threshold:
                  return True
      return False
  ------------------------------------------------------------------

Training:   5%|▌         | 50/1000 [06:58<2:13:45,  8.44s/it]
...

^C  (Ctrl+C to exit tail)
```

---

## Part 5: While Training Runs (Local Machine)

### Step 10: Disconnect and check later

You can disconnect from SSH, training continues!

```bash
[shars11@gilbreth-fe00 RL-LLM]$ exit
logout
Connection to gilbreth.rcac.purdue.edu closed.

$  # Back on local machine
```

### Step 11: Periodically download results

From your **local machine**:

```bash
$ cd C:\Users\shars11\Documents\RL_LLM\gilbreth_deployment
$ bash download_results.sh shars11

========================================
Download Results from Gilbreth
========================================
Remote: shars11@gilbreth.rcac.purdue.edu:~/RL-LLM
Local: ./gilbreth_results

Downloading checkpoints...
receiving incremental file list
checkpoints/
checkpoints/best_model.pt
    124,567,890 100%   15.23MB/s    0:00:08 (xfr#1, to-chk=2/4)
checkpoints/training_history.json
          1,234 100%  150.48kB/s    0:00:00 (xfr#2, to-chk=1/4)

sent 125 bytes  received 124,569,789 bytes  14,234,123.45 bytes/sec
total size is 124.6M  speedup is 1.00

Downloading logs...
...

========================================
Download Complete!
========================================
Files saved to: ./gilbreth_results

Checkpoints: ./gilbreth_results/checkpoints/
Logs: ./gilbreth_results/logs/
Results: ./gilbreth_results/results/
```

---

## Part 6: After Training Completes (6-12 hours later)

### Step 12: Check job completion

SSH back in:

```bash
$ ssh shars11@gilbreth.rcac.purdue.edu
[shars11@gilbreth-fe00 ~]$ cd RL-LLM

[shars11@gilbreth-fe00 RL-LLM]$ squeue -u $USER
JOBID      PARTITION  NAME            USER     ST TIME     NODES  REASON
(No jobs - training completed!)

[shars11@gilbreth-fe00 RL-LLM]$ sacct -j 145823
JobID           JobName      State      Elapsed    MaxRSS
------------ ------------ ---------- ---------- ---------
145823       rl-llm-cod+  COMPLETED   05:42:33
145823.batch batch        COMPLETED   05:42:33   8192000K
```

Job completed successfully!

### Step 13: Check output

```bash
[shars11@gilbreth-fe00 RL-LLM]$ tail -n 50 logs/training_output_145823.txt

Training: 100%|██████████| 1000/1000 [2:20:15<00:00,  8.42s/it]

======================================================================
Iteration 1000/1000
======================================================================
  Avg Training Reward: 7.89
  Policy Loss: 0.0234
  Value Loss: 0.0123
  Entropy: 2.8765
  Intention Loss: 0.0045

  Sample Generated Code:
  ------------------------------------------------------------------
  from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
      """
      Check if any two numbers in the list are closer to each other
      than the given threshold.

      >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
      False
      >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
      True
      """
      for i in range(len(numbers)):
          for j in range(i + 1, len(numbers)):
              if abs(numbers[i] - numbers[j]) < threshold:
                  return True
      return False
  ------------------------------------------------------------------

======================================================================
Training Complete!
======================================================================
Best reward achieved: 8.12

  ✓ Checkpoint saved: ./checkpoints/final_model.pt
Training history saved to: ./checkpoints/training_history.json

======================================
Job completed at: Mon Jan 18 15:29:43 EST 2025
Exit status: 0
======================================

Training completed successfully!
Checkpoints saved in: ~/RL-LLM/checkpoints/

To download results from your local machine:
  rsync -av shars11@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./checkpoints/
  rsync -av shars11@gilbreth.rcac.purdue.edu:~/RL-LLM/logs/ ./logs/
```

### Step 14: Check saved files

```bash
[shars11@gilbreth-fe00 RL-LLM]$ ls -lh checkpoints/
total 238M
-rw-r--r-- 1 shars11 users 119M Jan 18 15:29 best_model.pt
-rw-r--r-- 1 shars11 users 119M Jan 18 15:29 final_model.pt
-rw-r--r-- 1 shars11 users  45K Jan 18 15:29 training_history.json

[shars11@gilbreth-fe00 RL-LLM]$ cat checkpoints/training_history.json | head -n 20
{
  "training_history": [
    0.23,
    0.45,
    0.89,
    1.23,
    1.67,
    2.01,
    2.34,
    ...
    7.89,
    8.12
  ],
  "best_reward": 8.12,
  "args": {
    "dataset": "humaneval",
    "num_iterations": 1000,
    "episodes_per_iter": 5,
    ...
```

---

## Part 7: Final Download (Local Machine)

### Step 15: Download all results

```bash
$ cd C:\Users\shars11\Documents\RL_LLM\gilbreth_deployment
$ bash download_results.sh shars11

========================================
Download Results from Gilbreth
========================================
...
Download Complete!

$ cd gilbreth_results/checkpoints
$ ls -lh
total 238M
-rw-r--r-- 1 user group 119M Jan 18 15:29 best_model.pt
-rw-r--r-- 1 user group 119M Jan 18 15:29 final_model.pt
-rw-r--r-- 1 user group  45K Jan 18 15:29 training_history.json
```

---

## Summary

✅ **Total time**: ~6 hours (mostly automated)
✅ **Your time**: ~15 minutes (setup, submit, monitor, download)
✅ **Result**: Trained hierarchical RL model for code generation
✅ **Files**: 2 model checkpoints + training history

### What You Got

- **best_model.pt**: Best performing model (reward: 8.12)
- **final_model.pt**: Final model after 1000 iterations
- **training_history.json**: Full training metrics

### Next Steps

1. Load the model for inference
2. Evaluate on full HumanEval (164 problems)
3. Fine-tune on more data
4. Deploy for code generation tasks

---

**Training completed successfully! 🎉**
