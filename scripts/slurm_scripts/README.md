# SLURM Scripts for Emotion Classification Experiments

This directory contains SLURM batch scripts for running emotion classification experiments on a SLURM cluster with maximum parallelization.

## Overview

The experiments are organized into **three independent pipelines** that run in parallel after setup:

### Experiment Pipeline (Optimized)

```
Setup (30m)
    ↓
    ├─→ Zero-shot Eval (2h) ──────────┐
    │                                  │
    ├─→ CLIP: Train (1d) → Eval (2h) ─┼─→ Comparison (1h)
    │                                  │
    └─→ MERU: Train (1.5d) → Eval (2h)┘
         ALL RUN IN PARALLEL
```

**Key Insight**: Zero-shot, CLIP, and MERU experiments are completely independent and can run simultaneously!

**Total time: ~1.5 days** (limited by MERU training, the longest job)

## Quick Start

### Recommended: Use Master Script

**For sanity check (30 minutes):**
```bash
bash run_experiments.sh sanity
```

**For full experiment (~1.5 days):**
```bash
bash run_experiments.sh full
```

That's it! The master script handles everything:
- ✅ Sets all environment variables
- ✅ Validates paths
- ✅ Submits all jobs in parallel
- ✅ Supports sanity checks (limited samples)
- ✅ Configures wandb automatically

**See [USAGE.md](./USAGE.md) for complete guide.**

### Manual Submission (Advanced)

If you prefer manual control:

1. **Setup Weights & Biases (optional but recommended)**:
   ```bash
   pip install wandb
   wandb login
   ```
   See [WANDB_SETUP.md](./WANDB_SETUP.md) for detailed wandb configuration.

2. **Submit all experiments in parallel**:
   ```bash
   bash submit_all_parallel.sh
   ```

3. **Monitor jobs**:
   ```bash
   squeue -u $USER
   watch -n 5 squeue -u $USER  # Auto-refresh every 5 seconds
   ```

4. **Check progress**:
   ```bash
   tail -f /home/gmago/AA/outputs/slurm_out/emoset_*.txt
   ```

5. **View results in wandb** (if enabled):
   - https://wandb.ai/YOUR_USERNAME/emotion-classification

## Scripts

### 00_setup.slurm
- **Purpose**: Environment setup and verification
- **Resources**: 1 CPU, 8GB RAM, 30 min
- **No GPU required**

### 01_zero_shot_full.slurm
- **Purpose**: Zero-shot CLIP baseline evaluation
- **Resources**: 4 CPUs, 1 GPU, 32GB RAM, 2 hours
- **Dependencies**: Setup only
- **Runs in parallel with CLIP and MERU**

### 02_clip_full_pipeline.slurm
- **Purpose**: Complete CLIP pipeline (train + evaluation on val/test)
- **Resources**: 8 CPUs, 1 GPU, 64GB RAM, 1 day 4 hours
- **Dependencies**: Setup only
- **Runs in parallel with Zero-shot and MERU**
- **Stages**:
  1. Train CLIP+CoOp (50 epochs)
  2. Evaluate on validation set
  3. Evaluate on test set

### 03_meru_full_pipeline.slurm
- **Purpose**: Complete MERU pipeline (train + evaluation on val/test)
- **Resources**: 8 CPUs, 1 GPU, 64GB RAM, 1 day 16 hours
- **Dependencies**: Setup only
- **Runs in parallel with Zero-shot and CLIP**
- **Stages**:
  1. Train MERU+CoOp (100 epochs)
  2. Evaluate on validation set
  3. Evaluate on test set

### 04_compare_all.slurm
- **Purpose**: Generate comparison report across all models
- **Resources**: 4 CPUs, 1 GPU, 32GB RAM, 1 hour
- **Dependencies**: Zero-shot, CLIP, and MERU all complete
- **Runs after all three pipelines finish**

## Parallelization Benefits

### Old Sequential Approach
```
Setup → Zero-shot → CLIP (train+eval) → MERU (train+eval) → Compare
Total: ~3+ days
```

### New Parallel Approach
```
Setup → [Zero-shot || CLIP || MERU] → Compare
Total: ~1.5 days (50% time savings!)
```

### Resource Utilization
- **3 GPUs**: All three experiments run simultaneously (optimal)
- **2 GPUs**: Two experiments run at once, third queues
- **1 GPU**: Experiments run sequentially but auto-scheduled by SLURM

## Configuration

### Default Paths
- Dataset: `/ivi/zfs/s0/original_homes/gmago/emoset`
- Models: `/ivi/zfs/s0/original_homes/gmago/models/`
  - CLIP: `clip_vit_b.pth`
  - MERU: `meru_vit_b.pth`
- Output: `/home/gmago/Emotions/outputs/experiments/`
- Environment: `/ivi/zfs/s0/original_homes/gmago/envs/aa/bin/activate`
- Wandb: `/home/gmago/AA/outputs/wandb` (project: `emotion-classification`)

### Training Parameters

**CLIP+CoOp**:
- Epochs: 50
- Batch size: 32
- Learning rate: 0.002
- Context tokens: 16
- Optimizer: SGD

**MERU+CoOp**:
- Epochs: 100
- Batch size: 32
- Learning rate: 0.002
- Context tokens: 16
- Entailment weight: 0.2
- Optimizer: AdamW

**Evaluation**:
- Batch size: 128

### Modifying Parameters

Edit the relevant `.slurm` file and change the export variables:
```bash
export NUM_EPOCHS="50"
export BATCH_SIZE="32"
export LEARNING_RATE="0.002"
export NUM_CTX="16"
export ENTAIL_WEIGHT="0.2"  # MERU only
```

## Monitoring and Debugging

### Check Job Status
```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Check completed jobs
sacct -j <job_id>

# Check all jobs from submission
sacct -j <setup_id>,<zero_id>,<clip_id>,<meru_id>,<compare_id>
```

### View Outputs
```bash
# SLURM logs (live tail)
tail -f /home/gmago/AA/outputs/slurm_out/emoset_zero_shot_full_*.txt
tail -f /home/gmago/AA/outputs/slurm_out/emoset_clip_full_*.txt
tail -f /home/gmago/AA/outputs/slurm_out/emoset_meru_full_*.txt

# Experiment outputs
ls /home/gmago/Emotions/outputs/experiments/exp_*/

# View checkpoints
ls /home/gmago/Emotions/outputs/experiments/exp_*/*/checkpoints/
```

### Cancel Jobs
```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel job chain (use IDs from submission output)
scancel <setup_id> <zero_id> <clip_id> <meru_id> <compare_id>
```

## Resource Requirements

| Job | CPUs | GPU | RAM | Time | Depends On | Parallel With |
|-----|------|-----|-----|------|------------|---------------|
| Setup | 1 | 0 | 8GB | 30m | - | - |
| Zero-shot | 4 | 1 | 32GB | 2h | Setup | CLIP, MERU |
| CLIP Full | 8 | 1 | 64GB | 1d 4h | Setup | Zero-shot, MERU |
| MERU Full | 8 | 1 | 64GB | 1d 16h | Setup | Zero-shot, CLIP |
| Compare | 4 | 1 | 32GB | 1h | All three | - |

**Total GPU hours**: ~80 GPU-hours
**Total wallclock time**: ~1.5 days (with 3 GPUs) vs ~3 days (sequential)

## Running Individual Experiments

You can also run experiments individually:

```bash
# Run only zero-shot
sbatch scripts/slurm_scripts/01_zero_shot_full.slurm

# Run only CLIP pipeline
EXPERIMENT_DIR="/path/to/output" sbatch --export=ALL,EXPERIMENT_DIR scripts/slurm_scripts/02_clip_full_pipeline.slurm

# Run only MERU pipeline
EXPERIMENT_DIR="/path/to/output" sbatch --export=ALL,EXPERIMENT_DIR scripts/slurm_scripts/03_meru_full_pipeline.slurm
```

## Troubleshooting

### Job Fails Immediately
- Check SLURM error logs in `/home/gmago/AA/outputs/slurm_error/`
- Verify paths in the script match your system
- Ensure environment activation works: `source /ivi/zfs/s0/original_homes/gmago/envs/aa/bin/activate`

### Out of Memory
- Reduce `BATCH_SIZE` in the script
- Request more memory: Edit `#SBATCH --mem=` in the script

### Checkpoint Not Found (in comparison)
- Ensure CLIP/MERU pipelines completed successfully
- Check `$EXPERIMENT_DIR/clip_coop/checkpoints/best_model.pth`
- Check `$EXPERIMENT_DIR/meru_coop/checkpoints/best_model.pth`

### Training Takes Too Long
- Reduce `NUM_EPOCHS` for testing
- Increase `BATCH_SIZE` if memory allows
- Use a smaller subset of data for debugging

### Dependencies Not Working
- SLURM job IDs must be valid
- Check dependency status: `scontrol show job <job_id> | grep Dependency`
- Jobs with `afterok` only run if previous jobs succeed

## Advanced Usage

### Manual Dependency Chain

For more control over submission:

```bash
cd /home/gmago/AA/meru/scripts/slurm_scripts

# Stage 1: Setup
SETUP=$(sbatch --parsable 00_setup.slurm)

# Stage 2: All experiments in parallel
ZERO=$(sbatch --parsable --dependency=afterok:$SETUP 01_zero_shot_full.slurm)
CLIP=$(sbatch --parsable --dependency=afterok:$SETUP 02_clip_full_pipeline.slurm)
MERU=$(sbatch --parsable --dependency=afterok:$SETUP 03_meru_full_pipeline.slurm)

# Stage 3: Comparison (wait for all)
COMPARE=$(sbatch --parsable --dependency=afterok:$ZERO:$CLIP:$MERU 04_compare_all.slurm)

echo "Submitted: Setup=$SETUP, Zero=$ZERO, CLIP=$CLIP, MERU=$MERU, Compare=$COMPARE"
```

### Custom Experiment Directory

```bash
# Create custom experiment directory
EXP_DIR="/custom/path/my_experiment"
mkdir -p "$EXP_DIR"

# Submit with custom directory
sbatch --export=ALL,EXPERIMENT_DIR="$EXP_DIR" 02_clip_full_pipeline.slurm
sbatch --export=ALL,EXPERIMENT_DIR="$EXP_DIR" 03_meru_full_pipeline.slurm
```

### Rerun Only Evaluation

If training completed but evaluation failed:

```bash
# Extract checkpoint path from training output
CLIP_CKPT="/path/to/experiment/clip_coop/checkpoints/best_model.pth"
MERU_CKPT="/path/to/experiment/meru_coop/checkpoints/best_model.pth"

# Rerun comparison
sbatch --export=ALL,CLIP_COOP_CHECKPOINT="$CLIP_CKPT",MERU_COOP_CHECKPOINT="$MERU_CKPT" 04_compare_all.slurm
```

## Tips

1. **Test with Setup First**: Run `sbatch 00_setup.slurm` to verify environment
2. **Monitor Resource Usage**: Use `seff <job_id>` after completion to see efficiency
3. **Use Multiple GPUs**: If available, all three experiments will run simultaneously
4. **Experiment Naming**: Each run creates `exp_YYYYMMDD_HHMMSS` directory
5. **Log Rotation**: SLURM logs include job ID in filename for easy tracking
6. **Wandb Tracking**: See [WANDB_SETUP.md](./WANDB_SETUP.md) for experiment tracking with wandb

## Example Workflow

```bash
# 1. Submit all experiments
bash submit_all_parallel.sh

# Output shows job IDs:
# Setup job ID: 123456
# Zero-shot job ID: 123457
# CLIP job ID: 123458
# MERU job ID: 123459
# Comparison job ID: 123460

# 2. Monitor progress
watch -n 10 'squeue -u $USER'

# 3. Check live outputs
tail -f /home/gmago/AA/outputs/slurm_out/emoset_clip_full_123458.txt

# 4. After completion, check results
ls /home/gmago/Emotions/outputs/experiments/exp_20260331_143000/

# 5. View final comparison
cat /home/gmago/Emotions/outputs/experiments/exp_20260331_143000/comparison_*/comparison_report.txt
```

## FAQ

**Q: Can I run just CLIP without MERU?**
A: Yes! Just submit `02_clip_full_pipeline.slurm` individually.

**Q: How do I adjust training epochs for testing?**
A: Edit `NUM_EPOCHS` in the respective `.slurm` file (e.g., set to 5 for quick testing).

**Q: What if I only have 1 GPU available?**
A: SLURM will queue jobs automatically. They'll run one after another.

**Q: Can I use different model checkpoints?**
A: Yes, edit `PRETRAINED_CLIP` and `PRETRAINED_MERU` paths in the scripts.

**Q: How do I save costs with fewer experiments?**
A: Run only what you need. Zero-shot is quick (2h), CLIP is medium (1d), MERU is long (1.5d).

**Q: How do I track experiments with wandb?**
A: Install wandb (`pip install wandb`), login (`wandb login`), and run normally. See [WANDB_SETUP.md](./WANDB_SETUP.md) for details.

**Q: Can I disable wandb if I don't want it?**
A: Yes! Set `export WANDB_MODE=disabled` before submitting, or don't install wandb at all.
